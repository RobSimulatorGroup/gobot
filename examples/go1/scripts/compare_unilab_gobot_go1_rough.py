#!/usr/bin/env python3
"""Compare short Go1 rough-terrain rollouts in Gobot and UniLab.

The script is intentionally diagnostic: it runs fixed zero/random actions and
prints JSON lines with environment-level metrics.  It does not train or save a
policy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UNILAB_ROOT = REPO_ROOT / "UniLab"
DEFAULT_UNILAB_CFG = DEFAULT_UNILAB_ROOT / "conf/ppo/task/go1_joystick_rough/mujoco.yaml"
DEFAULT_POLICY = REPO_ROOT / "examples/go1/policies/go1.pt"


def _resolve_policy_path(value: str | Path, project_path: Path) -> Path:
    path = str(value)
    if path.startswith("res://"):
        return project_path / path.removeprefix("res://")
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def _load_policy(policy_path: Path):
    suffix = policy_path.suffix.lower()
    if suffix == ".pt":
        return _TorchPolicy(policy_path)
    if suffix == ".onnx":
        return _OnnxPolicy(policy_path)
    raise ValueError(f"unsupported policy file {policy_path}; expected .pt or .onnx")


def _policy_action_batch(policy: Any, obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    return np.asarray([policy.action(row) for row in obs], dtype=np.float32)


class _OnnxPolicy:
    def __init__(self, path: Path) -> None:
        import onnxruntime as ort

        self.np = np
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.obs_dim = int(input_shape[-1]) if input_shape and isinstance(input_shape[-1], int) else None

    def action(self, observation: np.ndarray) -> np.ndarray:
        obs = self.np.asarray(observation, dtype=self.np.float32).reshape(1, -1)
        if self.obs_dim is not None and obs.shape[1] != self.obs_dim:
            raise RuntimeError(f"policy expected {self.obs_dim} observations, got {obs.shape[1]}")
        return self.session.run([self.output_name], {self.input_name: obs})[0].reshape(-1)


class _TorchPolicy:
    def __init__(self, path: Path) -> None:
        import torch

        self.torch = torch
        self.device = torch.device("cpu")
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        actor_state = checkpoint.get("actor_state_dict", {})
        self.obs_dim = _checkpoint_obs_dim(actor_state)
        self.action_dim = _checkpoint_action_dim(actor_state)
        self.module = torch.nn.Sequential(*_build_mlp_layers(torch, actor_state, self.obs_dim, self.action_dim)).to(self.device)
        self.module.load_state_dict(
            {
                key.removeprefix("mlp."): value
                for key, value in actor_state.items()
                if key.startswith("mlp.")
            },
            strict=True,
        )
        self.module.eval()
        self.mean = _checkpoint_normalizer_tensor(torch, actor_state, "obs_normalizer._mean")
        self.std = _checkpoint_normalizer_tensor(torch, actor_state, "obs_normalizer._std")
        if self.std is None:
            var = _checkpoint_normalizer_tensor(torch, actor_state, "obs_normalizer._var")
            if var is not None:
                self.std = torch.sqrt(torch.clamp(var, min=1.0e-12))
        if self.mean is not None and self.std is None:
            self.std = torch.ones_like(self.mean)

    def action(self, observation: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            obs = self.torch.as_tensor(observation, dtype=self.torch.float32, device=self.device).reshape(1, -1)
            if obs.shape[1] != self.obs_dim:
                raise RuntimeError(f"policy expected {self.obs_dim} observations, got {obs.shape[1]}")
            if self.mean is not None:
                obs = (obs - self.mean.reshape(1, -1)) / self.std.reshape(1, -1).clamp_min(1.0e-6)
            output = self.module(obs)
        return output.reshape(-1).cpu().numpy()


def _checkpoint_mlp_layer_index(key: str) -> int:
    parts = key.split(".")
    if len(parts) < 3:
        return -1
    try:
        return int(parts[1])
    except ValueError:
        return -1


def _checkpoint_obs_dim(actor_state: dict[str, Any]) -> int:
    mean = actor_state.get("obs_normalizer._mean")
    if mean is not None and getattr(mean, "ndim", 0) > 0:
        return int(mean.reshape(-1).shape[0])
    first_linear = actor_state.get("mlp.0.weight")
    if first_linear is not None and getattr(first_linear, "ndim", 0) == 2:
        return int(first_linear.shape[1])
    raise RuntimeError("checkpoint actor_state_dict does not contain an observation normalizer or mlp.0.weight")


def _checkpoint_action_dim(actor_state: dict[str, Any]) -> int:
    last_weight = None
    last_layer_index = -1
    for key, value in actor_state.items():
        if not key.startswith("mlp.") or not key.endswith(".weight") or getattr(value, "ndim", 0) != 2:
            continue
        layer_index = _checkpoint_mlp_layer_index(key)
        if layer_index > last_layer_index:
            last_layer_index = layer_index
            last_weight = value
    if last_weight is None:
        raise RuntimeError("checkpoint actor_state_dict does not contain mlp.*.weight tensors")
    return int(last_weight.shape[0])


def _build_mlp_layers(torch: Any, actor_state: dict[str, Any], obs_dim: int, action_dim: int) -> list[Any]:
    weights = []
    for key, value in actor_state.items():
        if key.startswith("mlp.") and key.endswith(".weight") and getattr(value, "ndim", 0) == 2:
            weights.append((_checkpoint_mlp_layer_index(key), value))
    weights.sort(key=lambda item: item[0])
    dims = [int(weights[0][1].shape[1])] if weights else [int(obs_dim)]
    dims.extend(int(weight.shape[0]) for _, weight in weights)
    if dims[0] != int(obs_dim) or dims[-1] != int(action_dim):
        raise RuntimeError(f"checkpoint MLP dims {dims} do not match obs={obs_dim} action={action_dim}")
    layers: list[Any] = []
    for index in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[index], dims[index + 1]))
        if index < len(dims) - 2:
            layers.append(torch.nn.ELU())
    return layers


def _checkpoint_normalizer_tensor(torch: Any, actor_state: dict[str, Any], name: str):
    value = actor_state.get(name)
    if value is None:
        return None
    return torch.as_tensor(value, dtype=torch.float32).reshape(-1)


def _dump_path(args: argparse.Namespace, backend: str) -> Path | None:
    if args.dump_npz is None:
        return None
    path = Path(args.dump_npz)
    if args.backend == "both":
        return path.with_name(f"{path.stem}_{backend}{path.suffix or '.npz'}")
    return path


def _append_history(history: dict[str, list[np.ndarray]], **values: Any) -> None:
    for key, value in values.items():
        history.setdefault(key, []).append(np.asarray(value).copy())


def _write_history(path: Path | None, history: dict[str, list[np.ndarray]], metadata: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(values) for key, values in history.items()}
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_)
    np.savez_compressed(path, **arrays)
    print(json.dumps({"backend": metadata.get("backend"), "dump_npz": str(path)}, sort_keys=True), flush=True)


def _round_float(value: Any) -> float:
    return round(float(value), 6)


def _mean(value: Any, default: float = 0.0) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return default
    return _round_float(np.mean(array))


def _field(state: Any, name: str, default: float = 0.0) -> Any:
    return getattr(state, name, default)


def _reward_log(prefix: str, logs: dict[str, Any], names: tuple[str, ...]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in names:
        key = f"{prefix}{name}"
        if key in logs:
            result[name] = _round_float(logs[key])
    return result


def _action_batch(
    *,
    mode: str,
    rng: np.random.Generator,
    num_envs: int,
    num_actions: int,
) -> np.ndarray:
    if mode == "zero":
        return np.zeros((num_envs, num_actions), dtype=np.float32)
    if mode == "random":
        return rng.uniform(-1.0, 1.0, (num_envs, num_actions)).astype(np.float32)
    if mode == "policy":
        raise ValueError("policy actions require a loaded policy and current observations")
    raise ValueError(f"unsupported action mode {mode!r}")


def _print_record(record: dict[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def _heading_error(target: np.ndarray, quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    if quat.size == 0:
        return np.zeros_like(np.asarray(target, dtype=np.float32))
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return ((np.asarray(target, dtype=np.float32) - yaw + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _model_debug_summary(debug: dict[str, Any]) -> dict[str, Any]:
    actuators = list(debug.get("actuators", ()))
    joints = list(debug.get("joints", ()))
    bodies = list(debug.get("bodies", ()))
    geoms = list(debug.get("geoms", ()))
    hfields = list(debug.get("hfields", ()))
    sensors = list(debug.get("sensors", ()))
    return {
        "option": debug.get("option", {}),
        "nq": debug.get("nq"),
        "nv": debug.get("nv"),
        "nu": debug.get("nu"),
        "ngeom": debug.get("ngeom"),
        "nhfield": debug.get("nhfield"),
        "nsensor": debug.get("nsensor"),
        "actuators": actuators[:12],
        "joints": joints[:13],
        "bodies": bodies[:14],
        "geoms": geoms[:40],
        "hfields": hfields,
        "sensors": sensors[:40],
    }


def _array_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float64)
    result: dict[str, Any] = {"shape": list(array.shape)}
    if array.size == 0:
        return result
    result.update(
        {
            "min": _round_float(np.min(array)),
            "max": _round_float(np.max(array)),
            "mean": _round_float(np.mean(array)),
            "std": _round_float(np.std(array)),
        }
    )
    return result


def _parse_vec3(value: str | Sequence[float], *, name: str) -> np.ndarray:
    return _parse_vec(value, 3, name=name)


def _parse_vec2(value: str | Sequence[float], *, name: str) -> np.ndarray:
    return _parse_vec(value, 2, name=name)


def _parse_vec(value: str | Sequence[float], count: int, *, name: str) -> np.ndarray:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        parts = list(value)
    if len(parts) != count:
        raise ValueError(f"{name} must contain exactly {count} values")
    return np.asarray([float(part) for part in parts], dtype=np.float32)


def _fixed_base_position(args: argparse.Namespace, num_envs: int) -> np.ndarray:
    xy = _parse_vec2(args.fixed_base_xy, name="fixed base xy")
    position = np.asarray([xy[0], xy[1], float(args.fixed_base_z)], dtype=np.float32)
    return np.broadcast_to(position, (num_envs, 3)).astype(np.float32, copy=True)


def _fixed_command(args: argparse.Namespace, num_envs: int) -> np.ndarray:
    return np.broadcast_to(_parse_vec3(args.fixed_command, name="fixed command"), (num_envs, 3)).astype(np.float32, copy=True)


def _fixed_heading(args: argparse.Namespace, num_envs: int) -> np.ndarray:
    return np.full((num_envs,), float(args.fixed_heading), dtype=np.float32)


def _summarize_gobot(env: Any, step: int, *, phase: str) -> dict[str, Any]:
    state = env.backend.state
    reward_terms = np.asarray(_field(state, "reward_terms", np.empty((env.num_envs, 0))), dtype=np.float32)
    term_names = tuple(getattr(env, "_reward_term_names", ()))
    terms: dict[str, float] = {}
    for index, name in enumerate(term_names):
        if index < reward_terms.shape[1]:
            terms[str(name)] = _mean(reward_terms[:, index])

    foot_force = np.asarray(_field(state, "foot_contact_force", np.empty((env.num_envs, 0, 3))), dtype=np.float32)
    if foot_force.size:
        contact_threshold = float(env.cfg_obj.unilab_rewards.contact_threshold)
        foot_force_norm = np.linalg.norm(foot_force, axis=2)
        foot_contact_ratio = _mean(foot_force_norm > contact_threshold)
        foot_force_sum = _mean(np.sum(np.minimum(foot_force_norm, 1500.0), axis=1))
        foot_force_max = _round_float(np.max(foot_force_norm))
    else:
        foot_contact_ratio = 0.0
        foot_force_sum = 0.0
        foot_force_max = 0.0

    upvector = np.asarray(_field(state, "upvector", np.zeros((env.num_envs, 3))), dtype=np.float32)
    base_position = np.asarray(_field(state, "base_position", np.zeros((env.num_envs, 3))), dtype=np.float32)
    base_quat = np.asarray(_field(state, "base_quaternion", np.zeros((env.num_envs, 4))), dtype=np.float32)
    base_linvel_b = np.asarray(_field(state, "base_linear_velocity_body", np.zeros((env.num_envs, 3))), dtype=np.float32)
    command = np.asarray(_field(state, "command", np.zeros((env.num_envs, 3))), dtype=np.float32)
    reward = np.asarray(_field(state, "reward", np.zeros((env.num_envs,))), dtype=np.float32)
    heading_target = np.asarray(_field(state, "command_heading_target", np.zeros((env.num_envs,))), dtype=np.float32)
    heading_error = np.asarray(_field(state, "command_heading_error", np.zeros((env.num_envs,))), dtype=np.float32)
    if base_quat.size:
        w = base_quat[:, 0]
        x = base_quat[:, 1]
        y = base_quat[:, 2]
        z = base_quat[:, 3]
        base_yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    else:
        base_yaw = np.zeros((env.num_envs,), dtype=np.float32)

    return {
        "backend": "gobot",
        "phase": phase,
        "step": int(step),
        "reward_mean": _mean(reward),
        "base_z_mean": _mean(base_position[:, 2]),
        "base_z_min": _round_float(np.min(base_position[:, 2])) if base_position.size else 0.0,
        "upright_mean": _mean(upvector[:, 2]) if upvector.size else 0.0,
        "linvel_xy_mean": _mean(np.linalg.norm(base_linvel_b[:, :2], axis=1)) if base_linvel_b.size else 0.0,
        "linvel_z_abs_mean": _mean(np.abs(base_linvel_b[:, 2])) if base_linvel_b.size else 0.0,
        "command_xy_mean": _mean(np.linalg.norm(command[:, :2], axis=1)) if command.size else 0.0,
        "command_yaw_abs": _mean(np.abs(command[:, 2])) if command.size else 0.0,
        "base_yaw_mean": _mean(base_yaw),
        "heading_target_mean": _mean(heading_target),
        "heading_error_abs": _mean(np.abs(heading_error)),
        "foot_contact_ratio": foot_contact_ratio,
        "foot_force_sum": foot_force_sum,
        "foot_force_max": foot_force_max,
        "undesired_contact_count": _mean(_field(state, "undesired_contact_count")),
        "undesired_base_contact_count": _mean(_field(state, "undesired_base_contact_count")),
        "undesired_hip_contact_count": _mean(_field(state, "undesired_hip_contact_count")),
        "undesired_thigh_contact_count": _mean(_field(state, "undesired_thigh_contact_count")),
        "undesired_calf_contact_count": _mean(_field(state, "undesired_calf_contact_count")),
        "trunk_head_collision_count": _mean(_field(state, "trunk_head_collision_count")),
        "shank_collision_count": _mean(_field(state, "shank_collision_count")),
        "landing_force": _mean(np.sum(np.asarray(_field(state, "landing_force", 0.0), dtype=np.float32), axis=1))
        if np.asarray(_field(state, "landing_force", 0.0)).ndim == 2
        else _mean(_field(state, "landing_force", 0.0)),
        "reset_reason_mean": _mean(getattr(env, "_reset_reasons", np.zeros((env.num_envs,), dtype=np.int64))),
        "terrain_level_mean": _mean(getattr(env, "_terrain_levels", np.zeros((env.num_envs,), dtype=np.float32))),
        "reward_terms": terms,
    }


def _gobot_snapshot(env: Any, obs: np.ndarray | None, action: np.ndarray | None) -> dict[str, np.ndarray]:
    state = env.backend.state
    command = np.asarray(_field(state, "command", np.zeros((env.num_envs, 3))), dtype=np.float32)
    heading_target = np.asarray(_field(state, "command_heading_target", np.zeros((env.num_envs,))), dtype=np.float32)
    base_quat = np.asarray(_field(state, "base_quaternion", np.zeros((env.num_envs, 4))), dtype=np.float32)
    return {
        "obs": np.asarray(obs if obs is not None else state.actor_obs, dtype=np.float32),
        "action": np.asarray(
            action if action is not None else _field(state, "action", np.zeros((env.num_envs, env.num_actions))),
            dtype=np.float32,
        ),
        "submitted_action": np.asarray(
            _field(state, "submitted_action", np.zeros((env.num_envs, env.num_actions))),
            dtype=np.float32,
        ),
        "last_action": np.asarray(_field(state, "last_action", np.zeros((env.num_envs, env.num_actions))), dtype=np.float32),
        "target_position": np.asarray(
            _field(state, "target_position", np.zeros((env.num_envs, env.num_actions))),
            dtype=np.float32,
        ),
        "joint_position": np.asarray(_field(state, "joint_position", np.zeros((env.num_envs, env.num_actions))), dtype=np.float32),
        "joint_velocity": np.asarray(_field(state, "joint_velocity", np.zeros((env.num_envs, env.num_actions))), dtype=np.float32),
        "base_position": np.asarray(_field(state, "base_position", np.zeros((env.num_envs, 3))), dtype=np.float32),
        "base_quaternion": base_quat,
        "base_linear_velocity_body": np.asarray(
            _field(state, "base_linear_velocity_body", np.zeros((env.num_envs, 3))),
            dtype=np.float32,
        ),
        "base_angular_velocity_body": np.asarray(
            _field(state, "base_angular_velocity_body", np.zeros((env.num_envs, 3))),
            dtype=np.float32,
        ),
        "upvector": np.asarray(_field(state, "upvector", np.zeros((env.num_envs, 3))), dtype=np.float32),
        "command": command,
        "heading_target": heading_target,
        "heading_error": _heading_error(heading_target, base_quat),
        "foot_contact_force": np.asarray(
            _field(state, "foot_contact_force", np.zeros((env.num_envs, 0, 3))),
            dtype=np.float32,
        ),
    }


def _gobot_model_debug(env: Any) -> dict[str, Any]:
    model_debug = getattr(env.backend, "model_debug", None)
    if model_debug is None:
        return {}
    try:
        return _model_debug_summary(dict(model_debug(0)))
    except Exception as error:
        return {"error": str(error)}


def _gobot_terrain_debug(project_path: Path) -> dict[str, Any]:
    from gobot.rl.locomotion.math import _json_vec
    from gobot.rl.locomotion.terrain import TerrainSampler

    scene_path = Path(project_path) / "terrain_scene.jscn"
    if not scene_path.exists():
        return {"error": f"terrain scene not found: {scene_path}"}
    try:
        data = json.loads(scene_path.read_text(encoding="utf-8"))
        terrain_node = next((node for node in data.get("__NODES__", []) if node.get("type") == "Terrain3D"), None)
        if terrain_node is None:
            return {"error": f"Terrain3D node not found in {scene_path}"}
        properties = terrain_node.get("properties", {})
        spawn_origins = np.asarray([_json_vec(value, 3) for value in properties.get("spawn_origins", [])], dtype=np.float64)
        heightfields = []
        for heightfield in properties.get("heightfields", []):
            rows = int(heightfield.get("rows", 0))
            cols = int(heightfield.get("cols", 0))
            heights = np.asarray(heightfield.get("heights", []), dtype=np.float64)
            normalized = np.asarray(heightfield.get("normalized_elevation", []), dtype=np.float64)
            entry = {
                "rows": rows,
                "cols": cols,
                "center": _json_vec(heightfield.get("center"), 3).tolist(),
                "size": _json_vec(heightfield.get("size"), 2).tolist(),
                "base_thickness": _round_float(heightfield.get("base_thickness", 0.0)),
                "z_offset": _round_float(heightfield.get("z_offset", 0.0)),
                "heights": _array_summary(heights.reshape(rows, cols) if rows * cols == heights.size else heights),
                "normalized_elevation": _array_summary(
                    normalized.reshape(rows, cols) if rows * cols == normalized.size else normalized
                ),
            }
            heightfields.append(entry)
        sampler = TerrainSampler(scene_path)
        sample_xy = np.asarray(
            [
                [0.0, 0.0],
                [-4.0, -4.0],
                [-20.0, -20.0],
                [20.0, 20.0],
                [0.5, 0.0],
                [0.0, 0.5],
                [-0.5, 0.0],
                [0.0, -0.5],
            ],
            dtype=np.float32,
        )
        return {
            "scene_path": str(scene_path),
            "spawn_origins": _array_summary(spawn_origins),
            "spawn_origin_head": np.asarray(spawn_origins[:10], dtype=np.float64).tolist(),
            "heightfields": heightfields,
            "sample_xy": sample_xy.tolist(),
            "sample_height": np.asarray(sampler.heights_at(sample_xy), dtype=np.float32).tolist(),
            "bounds": list(sampler.bounds() or ()),
        }
    except Exception as error:
        return {"error": str(error)}


def _apply_gobot_fixed_reset(env: Any, args: argparse.Namespace) -> None:
    env_ids = np.arange(env.num_envs, dtype=np.int64)
    base_positions = _fixed_base_position(args, env.num_envs)
    base_orientations = np.zeros((env.num_envs, 4), dtype=np.float32)
    base_orientations[:, 0] = 1.0
    zero_base = np.zeros((env.num_envs, 3), dtype=np.float32)
    joint_positions = np.broadcast_to(env.default_joint_pos.reshape(1, -1), (env.num_envs, env.num_actions)).astype(np.float32, copy=True)
    joint_velocities = np.zeros_like(joint_positions)
    env.backend.reset_robot_states(
        env_ids.tolist(),
        base_positions=base_positions,
        base_orientations=base_orientations,
        base_linear_velocities=zero_base,
        base_angular_velocities=zero_base,
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        joint_position_targets=joint_positions,
    )
    state = env.backend.state
    rows = env_ids.astype(np.int64, copy=False)
    state.action[rows] = 0.0
    state.previous_action[rows] = 0.0
    state.last_action[rows] = 0.0
    env._previous_actions[rows] = 0.0
    env._last_actions[rows] = 0.0
    env._episode_length_np[rows] = 0
    env._unilab_info_steps[rows] = 0
    env.backend.set_commands(
        rows.tolist(),
        commands=_fixed_command(args, env.num_envs),
        heading_targets=_fixed_heading(args, env.num_envs),
        time_left=np.zeros((env.num_envs,), dtype=np.float32),
    )
    env.backend.set_command_steps(rows.tolist(), np.zeros((env.num_envs,), dtype=np.uint32))
    env._run_task_runtime(advance_time=False)
    env._obs = np.asarray(env.backend.state.actor_obs, dtype=np.float32).copy()
    env._critic_obs = np.asarray(env.backend.state.critic_obs, dtype=np.float32).copy()
    if env._state is not None:
        env._sync_batch_env_state(
            reward=np.asarray(env.backend.state.reward, dtype=np.float32),
            terminated=np.asarray(env.backend.state.terminated, dtype=bool),
            truncated=np.zeros((env.num_envs,), dtype=bool),
            obs_actor=env._obs,
            obs_critic=env._critic_obs,
        )


def run_gobot(args: argparse.Namespace) -> int:
    train_dir = REPO_ROOT / "examples/go1/train"
    if str(train_dir) not in sys.path:
        sys.path.insert(0, str(train_dir))

    from _repo_imports import prefer_repo_gobot

    prefer_repo_gobot()

    from go1_velocity_cfg import go1_rough_velocity_cfg
    from go1_velocity_env import Go1VelocityEnv

    cfg = go1_rough_velocity_cfg(project_path=args.project_path)
    if args.eval_reset:
        cfg.episode_length_s = float(1_000_000_000)
        cfg.terrain_curriculum = False
        cfg.randomize_rough_reset_pose = False
        cfg.observations.actor_noise = False
        cfg.domain_randomization.enabled = False
        cfg.push_enabled = False
    else:
        cfg.observations.actor_noise = not args.no_obs_noise
        if args.no_domain_rand:
            cfg.domain_randomization.enabled = False
        if args.no_push:
            cfg.push_enabled = False

    rng = np.random.default_rng(int(args.seed) + 31)
    policy = _load_policy(_resolve_policy_path(args.policy, args.project_path)) if args.actions == "policy" else None
    history: dict[str, list[np.ndarray]] = {}
    env = Go1VelocityEnv(
        cfg,
        num_envs=int(args.num_envs),
        device="cpu",
        seed=int(args.seed),
        sim_workers=int(args.sim_workers),
        collect_step_extras=True,
    )
    try:
        if args.fixed_reset:
            _apply_gobot_fixed_reset(env, args)
        obs = np.asarray(env.get_observations()["actor"], dtype=np.float32)
        _append_history(history, **_gobot_snapshot(env, obs, None))
        reset_record = _summarize_gobot(env, 0, phase="reset")
        reset_record["model_debug"] = _gobot_model_debug(env) if args.dump_model_debug else {}
        reset_record["terrain_debug"] = _gobot_terrain_debug(args.project_path) if args.dump_terrain_debug else {}
        _print_record(reset_record)
        for step in range(1, int(args.steps) + 1):
            if policy is not None:
                action = _policy_action_batch(policy, obs)
            else:
                action = _action_batch(
                    mode=args.actions,
                    rng=rng,
                    num_envs=env.num_envs,
                    num_actions=env.num_actions,
                )
            env.step(action)
            obs = np.asarray(env.get_observations()["actor"], dtype=np.float32)
            _append_history(history, **_gobot_snapshot(env, obs, action))
            if step == int(args.steps) or step % int(args.report_every) == 0:
                _print_record(_summarize_gobot(env, step, phase="step"))
    finally:
        metadata = {
            "backend": "gobot",
            "actions": args.actions,
            "policy": str(_resolve_policy_path(args.policy, args.project_path)) if args.actions == "policy" else "",
            "num_envs": int(args.num_envs),
            "steps": int(args.steps),
            "seed": int(args.seed),
            "model_debug": _gobot_model_debug(env) if args.dump_model_debug else {},
            "terrain_debug": _gobot_terrain_debug(args.project_path) if args.dump_terrain_debug else {},
        }
        _write_history(_dump_path(args, "gobot"), history, metadata)
        env.close()
    return 0


def _force_norm_columns(force: np.ndarray, num_envs: int) -> np.ndarray:
    force = np.asarray(force, dtype=np.float32).reshape(num_envs, -1)
    if force.shape[1] == 0:
        return force
    if force.shape[1] % 3 == 0:
        return np.linalg.norm(force.reshape(num_envs, -1, 3), axis=2)
    return np.abs(force)


def _unilab_reward_config(yaml_data: dict[str, Any]) -> dict[str, Any]:
    env_cfg = dict(yaml_data.get("env", {}))
    env_cfg["reward_config"] = dict(yaml_data.get("reward", {}))
    return env_cfg


def _apply_unilab_diagnostic_overrides(env_cfg: dict[str, Any], args: argparse.Namespace) -> None:
    env_cfg.setdefault("terrain_curriculum", {})["seed"] = int(args.seed)
    if args.eval_reset:
        env_cfg.setdefault("terrain_curriculum", {})["enabled"] = False
        env_cfg.setdefault("domain_rand", {})["push_robots"] = False
    if args.no_domain_rand or args.eval_reset:
        domain_rand = env_cfg.setdefault("domain_rand", {})
        for key in (
            "randomize_base_mass",
            "random_com",
            "randomize_kp",
            "randomize_kd",
            "randomize_foot_friction",
        ):
            if key in domain_rand:
                domain_rand[key] = False
    if args.no_push or args.eval_reset:
        env_cfg.setdefault("domain_rand", {})["push_robots"] = False


def _summarize_unilab(env: Any, state: Any, step: int, *, phase: str) -> dict[str, Any]:
    base_pos = np.asarray(env._backend.get_base_pos(), dtype=np.float32)
    base_quat = np.asarray(env._backend.get_base_quat(), dtype=np.float32)
    upvector = np.asarray(env._backend.get_sensor_data("upvector"), dtype=np.float32)
    linvel = np.asarray(env._backend.get_sensor_data("local_linvel"), dtype=np.float32)
    commands = np.asarray(state.info.get("commands", np.zeros((env.num_envs, 3))), dtype=np.float32)
    reward = np.asarray(state.reward, dtype=np.float32)
    heading_target = np.asarray(state.info.get("heading_commands", np.zeros((env.num_envs,))), dtype=np.float32)
    if base_quat.size:
        w = base_quat[:, 0]
        x = base_quat[:, 1]
        y = base_quat[:, 2]
        z = base_quat[:, 3]
        base_yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        heading_error = (heading_target - base_yaw + np.pi) % (2.0 * np.pi) - np.pi
    else:
        base_yaw = np.zeros((env.num_envs,), dtype=np.float32)
        heading_error = np.zeros((env.num_envs,), dtype=np.float32)

    contact_threshold = float(env._reward_cfg.contact_threshold)
    foot_forces = []
    for name in env._cfg.sensor.feet_force:
        foot_forces.append(_force_norm_columns(env._backend.get_sensor_data(name), env.num_envs))
    if foot_forces:
        foot_force_norm = np.concatenate(foot_forces, axis=1)
        foot_contact_ratio = _mean(foot_force_norm > contact_threshold)
        foot_force_sum = _mean(np.sum(np.minimum(foot_force_norm, 1500.0), axis=1))
        foot_force_max = _round_float(np.max(foot_force_norm))
    else:
        foot_contact_ratio = 0.0
        foot_force_sum = 0.0
        foot_force_max = 0.0

    undesired = []
    for name in env._cfg.sensor.undesired_contact:
        undesired.append(_force_norm_columns(env._backend.get_sensor_data(name), env.num_envs))
    if undesired:
        undesired_force = np.concatenate(undesired, axis=1)
        undesired_count = np.sum(undesired_force > contact_threshold, axis=1)
    else:
        undesired_count = np.zeros((env.num_envs,), dtype=np.float32)

    reward_terms = _reward_log("reward/", dict(state.info.get("log", {})), tuple(env._reward_cfg.scales.keys()))
    return {
        "backend": "unilab",
        "phase": phase,
        "step": int(step),
        "reward_mean": _mean(reward),
        "base_z_mean": _mean(base_pos[:, 2]),
        "base_z_min": _round_float(np.min(base_pos[:, 2])) if base_pos.size else 0.0,
        "upright_mean": _mean(upvector[:, 2]) if upvector.size else 0.0,
        "linvel_xy_mean": _mean(np.linalg.norm(linvel[:, :2], axis=1)) if linvel.size else 0.0,
        "linvel_z_abs_mean": _mean(np.abs(linvel[:, 2])) if linvel.size else 0.0,
        "command_xy_mean": _mean(np.linalg.norm(commands[:, :2], axis=1)) if commands.size else 0.0,
        "command_yaw_abs": _mean(np.abs(commands[:, 2])) if commands.size else 0.0,
        "base_yaw_mean": _mean(base_yaw),
        "heading_target_mean": _mean(heading_target),
        "heading_error_abs": _mean(np.abs(heading_error)),
        "foot_contact_ratio": foot_contact_ratio,
        "foot_force_sum": foot_force_sum,
        "foot_force_max": foot_force_max,
        "undesired_contact_count": _mean(undesired_count),
        "landing_force": foot_force_sum,
        "reset_reason_mean": 0.0,
        "reward_terms": reward_terms,
    }


def _unilab_snapshot(env: Any, state: Any, action: np.ndarray | None) -> dict[str, np.ndarray]:
    info = state.info
    obs = np.asarray(state.obs["obs"], dtype=np.float32)
    dof_pos = np.asarray(env.get_dof_pos(), dtype=np.float32)
    dof_vel = np.asarray(env.get_dof_vel(), dtype=np.float32)
    current_action = np.asarray(info.get("current_actions", np.zeros((env.num_envs, env._num_action))), dtype=np.float32)
    target = current_action * np.asarray(env._action_scale, dtype=np.float32).reshape(1, -1) + np.asarray(env.default_angles, dtype=np.float32).reshape(1, -1)
    base_quat = np.asarray(env._backend.get_base_quat(), dtype=np.float32)
    heading_target = np.asarray(info.get("heading_commands", np.zeros((env.num_envs,))), dtype=np.float32)
    foot_contact_force = np.zeros((env.num_envs, len(env._cfg.sensor.feet_force), 3), dtype=np.float32)
    for index, name in enumerate(env._cfg.sensor.feet_force):
        foot_contact_force[:, index, :] = np.asarray(env._backend.get_sensor_data(name), dtype=np.float32)
    return {
        "obs": obs,
        "critic_obs": np.asarray(state.obs.get("critic", np.zeros((env.num_envs, 0))), dtype=np.float32),
        "action": np.asarray(action if action is not None else current_action, dtype=np.float32),
        "submitted_action": current_action,
        "last_action": np.asarray(info.get("last_actions", np.zeros_like(current_action)), dtype=np.float32),
        "target_position": target.astype(np.float32, copy=False),
        "joint_position": dof_pos,
        "joint_velocity": dof_vel,
        "base_position": np.asarray(env._backend.get_base_pos(), dtype=np.float32),
        "base_quaternion": base_quat,
        "base_linear_velocity_body": np.asarray(env._backend.get_sensor_data("local_linvel"), dtype=np.float32),
        "base_angular_velocity_body": np.asarray(env._backend.get_sensor_data("gyro"), dtype=np.float32),
        "upvector": np.asarray(env._backend.get_sensor_data("upvector"), dtype=np.float32),
        "command": np.asarray(info.get("commands", np.zeros((env.num_envs, 3))), dtype=np.float32),
        "heading_target": heading_target,
        "heading_error": _heading_error(heading_target, base_quat),
        "foot_contact_force": foot_contact_force,
    }


def _unilab_model_debug(env: Any) -> dict[str, Any]:
    try:
        import mujoco

        model = env._backend.model
        option = {
            "timestep": float(model.opt.timestep),
            "solver": int(model.opt.solver),
            "integrator": int(model.opt.integrator),
            "cone": int(model.opt.cone),
            "jacobian": int(model.opt.jacobian),
            "iterations": int(model.opt.iterations),
            "ls_iterations": int(model.opt.ls_iterations),
            "noslip_iterations": int(model.opt.noslip_iterations),
            "ccd_iterations": int(model.opt.ccd_iterations),
            "tolerance": float(model.opt.tolerance),
            "ls_tolerance": float(model.opt.ls_tolerance),
            "noslip_tolerance": float(model.opt.noslip_tolerance),
            "ccd_tolerance": float(model.opt.ccd_tolerance),
            "impratio": float(model.opt.impratio),
        }
        actuators = []
        for actuator_id in range(min(int(model.nu), 12)):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or ""
            joint_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
                if 0 <= joint_id < model.njnt
                else ""
            )
            actuators.append(
                {
                    "id": actuator_id,
                    "name": actuator_name,
                    "joint_name": joint_name,
                    "gaintype": int(model.actuator_gaintype[actuator_id]),
                    "biastype": int(model.actuator_biastype[actuator_id]),
                    "trnid": np.asarray(model.actuator_trnid[actuator_id], dtype=np.int32).tolist(),
                    "gainprm": np.asarray(model.actuator_gainprm[actuator_id], dtype=np.float64).tolist(),
                    "biasprm": np.asarray(model.actuator_biasprm[actuator_id], dtype=np.float64).tolist(),
                    "ctrlrange": np.asarray(model.actuator_ctrlrange[actuator_id], dtype=np.float64).tolist(),
                    "forcerange": np.asarray(model.actuator_forcerange[actuator_id], dtype=np.float64).tolist(),
                    "joint_id": joint_id,
                }
            )
        joints = []
        for joint_id in range(min(int(model.njnt), 13)):
            dof = int(model.jnt_dofadr[joint_id])
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
            joints.append(
                {
                    "id": joint_id,
                    "name": joint_name,
                    "type": int(model.jnt_type[joint_id]),
                    "qposadr": int(model.jnt_qposadr[joint_id]),
                    "dofadr": dof,
                    "limited": int(model.jnt_limited[joint_id]),
                    "range": np.asarray(model.jnt_range[joint_id], dtype=np.float64).tolist(),
                    "damping": float(model.dof_damping[dof]) if 0 <= dof < model.nv else 0.0,
                    "armature": float(model.dof_armature[dof]) if 0 <= dof < model.nv else 0.0,
                    "frictionloss": float(model.dof_frictionloss[dof]) if 0 <= dof < model.nv else 0.0,
                }
            )
        bodies = []
        for body_id in range(int(model.nbody)):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            name_key = body_name.lower()
            include = body_id == 0 or any(
                token in name_key for token in ("foot", "hip", "thigh", "calf", "base", "trunk", "fl", "fr", "rl", "rr")
            )
            if not include:
                continue
            bodies.append(
                {
                    "id": body_id,
                    "name": body_name,
                    "parent_id": int(model.body_parentid[body_id]),
                    "mass": float(model.body_mass[body_id]),
                    "ipos": np.asarray(model.body_ipos[body_id], dtype=np.float64).tolist(),
                    "iquat": np.asarray(model.body_iquat[body_id], dtype=np.float64).tolist(),
                    "inertia": np.asarray(model.body_inertia[body_id], dtype=np.float64).tolist(),
                }
            )
        geoms = []
        for geom_id in range(int(model.ngeom)):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            body_id = int(model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            geom_type = int(model.geom_type[geom_id])
            terrain_geom = (
                geom_type == int(mujoco.mjtGeom.mjGEOM_HFIELD)
                or geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE)
                or int(model.geom_group[geom_id]) == 5
            )
            name_key = f"{geom_name} {body_name}".lower()
            include = terrain_geom or any(
                token in name_key for token in ("foot", "hip", "thigh", "calf", "base", "trunk", "fl", "fr", "rl", "rr")
            )
            if not include:
                continue
            geoms.append(
                {
                    "id": geom_id,
                    "name": geom_name,
                    "body_id": body_id,
                    "body_name": body_name,
                    "type": geom_type,
                    "dataid": int(model.geom_dataid[geom_id]),
                    "group": int(model.geom_group[geom_id]),
                    "size": np.asarray(model.geom_size[geom_id], dtype=np.float64).tolist(),
                    "pos": np.asarray(model.geom_pos[geom_id], dtype=np.float64).tolist(),
                    "quat": np.asarray(model.geom_quat[geom_id], dtype=np.float64).tolist(),
                    "friction": np.asarray(model.geom_friction[geom_id], dtype=np.float64).tolist(),
                    "condim": int(model.geom_condim[geom_id]),
                    "margin": float(model.geom_margin[geom_id]),
                    "gap": float(model.geom_gap[geom_id]),
                    "priority": int(model.geom_priority[geom_id]),
                    "solref": np.asarray(model.geom_solref[geom_id], dtype=np.float64).tolist(),
                    "solimp": np.asarray(model.geom_solimp[geom_id], dtype=np.float64).tolist(),
                    "contype": int(model.geom_contype[geom_id]),
                    "conaffinity": int(model.geom_conaffinity[geom_id]),
                }
            )
        hfields = []
        for hfield_id in range(int(model.nhfield)):
            hfields.append(
                {
                    "id": hfield_id,
                    "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_HFIELD, hfield_id) or "",
                    "nrow": int(model.hfield_nrow[hfield_id]),
                    "ncol": int(model.hfield_ncol[hfield_id]),
                    "size": np.asarray(model.hfield_size[hfield_id], dtype=np.float64).tolist(),
                    "adr": int(model.hfield_adr[hfield_id]),
                }
            )
        sensors = []
        sensor_data = np.asarray(getattr(env._backend, "_sensor_data", np.zeros((env.num_envs, 0))), dtype=np.float64)
        for sensor_id in range(min(int(model.nsensor), 40)):
            obj_id = int(model.sensor_objid[sensor_id])
            obj_type = int(model.sensor_objtype[sensor_id])
            try:
                obj_name = mujoco.mj_id2name(model, mujoco.mjtObj(obj_type), obj_id) or ""
            except Exception:
                obj_name = ""
            adr = int(model.sensor_adr[sensor_id])
            dim = int(model.sensor_dim[sensor_id])
            data = sensor_data[0, adr : adr + dim].tolist() if sensor_data.ndim == 2 and adr >= 0 and sensor_data.shape[1] >= adr + dim else []
            sensors.append(
                {
                    "id": sensor_id,
                    "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id) or "",
                    "type": int(model.sensor_type[sensor_id]),
                    "datatype": int(model.sensor_datatype[sensor_id]),
                    "objtype": obj_type,
                    "objid": obj_id,
                    "objname": obj_name,
                    "dim": dim,
                    "adr": adr,
                    "data": data,
                }
            )
        return {
            "option": option,
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "ngeom": int(model.ngeom),
            "nhfield": int(model.nhfield),
            "nsensor": int(model.nsensor),
            "actuators": actuators,
            "joints": joints,
            "bodies": bodies[:14],
            "geoms": geoms[:40],
            "hfields": hfields,
            "sensors": sensors,
        }
    except Exception as error:
        return {"error": str(error)}


def _unilab_terrain_debug(env: Any) -> dict[str, Any]:
    try:
        terrain_origins = np.asarray(getattr(env._backend, "terrain_origins", []), dtype=np.float64)
        sampler = getattr(env._backend, "terrain_surface_sampler", None)
        sample_xy = np.asarray(
            [
                [0.0, 0.0],
                [-4.0, -4.0],
                [-20.0, -20.0],
                [20.0, 20.0],
                [0.5, 0.0],
                [0.0, 0.5],
                [-0.5, 0.0],
                [0.0, -0.5],
            ],
            dtype=np.float32,
        )
        sample_height: list[float] = []
        sampler_summary: dict[str, Any] = {}
        if sampler is not None:
            sample_height = np.asarray(sampler.sample_height(sample_xy), dtype=np.float32).tolist()
            sampler_summary = {
                "size": list(getattr(sampler, "size", ())),
                "horizontal_scale": _round_float(getattr(sampler, "horizontal_scale", 0.0)),
                "z_min": _round_float(getattr(sampler, "z_min", 0.0)),
                "height_extent": _round_float(getattr(sampler, "height_extent", 0.0)),
                "heights_uint16": _array_summary(getattr(sampler, "heights_uint16", [])),
            }
        terrain_cfg = env._cfg.scene.terrain.generator if env._cfg.scene is not None and env._cfg.scene.terrain is not None else None
        cfg_summary: dict[str, Any] = {}
        if terrain_cfg is not None:
            cfg_summary = {
                "size": list(terrain_cfg.size),
                "border_width": _round_float(terrain_cfg.border_width),
                "num_rows": int(terrain_cfg.num_rows),
                "num_cols": int(terrain_cfg.num_cols),
                "horizontal_scale": _round_float(terrain_cfg.horizontal_scale),
                "vertical_scale": _round_float(terrain_cfg.vertical_scale),
                "curriculum": bool(terrain_cfg.curriculum),
                "seed": terrain_cfg.seed,
            }
        return {
            "cfg": cfg_summary,
            "spawn_origins": _array_summary(terrain_origins),
            "spawn_origin_head": np.asarray(terrain_origins.reshape(-1, 3)[:10], dtype=np.float64).tolist()
            if terrain_origins.size
            else [],
            "sampler": sampler_summary,
            "sample_xy": sample_xy.tolist(),
            "sample_height": sample_height,
        }
    except Exception as error:
        return {"error": str(error)}


def _apply_unilab_fixed_reset(env: Any, state: Any, args: argparse.Namespace) -> Any:
    env_ids = np.arange(env.num_envs, dtype=np.int32)
    qpos = np.tile(np.asarray(env._init_qpos, dtype=np.float32), (env.num_envs, 1))
    qvel = np.zeros((env.num_envs, env._backend.nv), dtype=np.float32)
    qpos[:, 0:3] = _fixed_base_position(args, env.num_envs)
    qpos[:, 3:7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    env._backend.set_state(env_ids, qpos, qvel, randomization=None)
    zeros = np.zeros((env.num_envs, env._num_action), dtype=np.float32)
    state.info["current_actions"] = zeros.copy()
    state.info["last_actions"] = zeros.copy()
    state.info["commands"] = _fixed_command(args, env.num_envs)
    state.info["heading_commands"] = _fixed_heading(args, env.num_envs)
    state.info["steps"] = np.zeros((env.num_envs,), dtype=np.uint32)
    state.info["qacc"] = zeros.copy()
    state.info["torques"] = zeros.copy()
    return env.update_state(state)


def run_unilab_child(args: argparse.Namespace) -> int:
    import yaml
    from unilab.base.registry import ensure_registries, make

    np.random.seed(int(args.seed))
    os.chdir(args.unilab_root)
    with open(args.unilab_config, "r", encoding="utf-8") as handle:
        yaml_data = yaml.safe_load(handle)
    ensure_registries()
    env_cfg = _unilab_reward_config(yaml_data)
    _apply_unilab_diagnostic_overrides(env_cfg, args)
    env = make(
        "Go1JoystickRough",
        sim_backend="mujoco",
        num_envs=int(args.num_envs),
        env_cfg_override=env_cfg,
    )
    if args.eval_reset:
        env.cfg.domain_rand.push_robots = False
        env.cfg.terrain_curriculum.enabled = False
    rng = np.random.default_rng(int(args.seed) + 31)
    policy = _load_policy(_resolve_policy_path(args.policy, args.project_path)) if args.actions == "policy" else None
    history: dict[str, list[np.ndarray]] = {}
    state = env.init_state()
    if args.eval_reset:
        qpos = np.tile(env._init_qpos, (env.num_envs, 1))
        qvel = np.zeros((env.num_envs, env._backend.nv), dtype=np.float32)
        spawn_origins = env._spawn.origins_for(np.arange(env.num_envs, dtype=np.int64))
        qpos[:, :3] += spawn_origins
        env._backend.set_state(np.arange(env.num_envs, dtype=np.int32), qpos, qvel, randomization=None)
        state = env.update_state(state)
    if args.fixed_reset:
        state = _apply_unilab_fixed_reset(env, state, args)
    try:
        _append_history(history, **_unilab_snapshot(env, state, None))
        reset_record = _summarize_unilab(env, state, 0, phase="reset")
        reset_record["model_debug"] = _unilab_model_debug(env) if args.dump_model_debug else {}
        reset_record["terrain_debug"] = _unilab_terrain_debug(env) if args.dump_terrain_debug else {}
        _print_record(reset_record)
        num_actions = int(getattr(env, "_num_action", env.action_space.shape[0]))
        for step in range(1, int(args.steps) + 1):
            if policy is not None:
                action = _policy_action_batch(policy, np.asarray(state.obs["obs"], dtype=np.float32))
            else:
                action = _action_batch(
                    mode=args.actions,
                    rng=rng,
                    num_envs=env.num_envs,
                    num_actions=num_actions,
                )
            state = env.step(action)
            _append_history(history, **_unilab_snapshot(env, state, action))
            if step == int(args.steps) or step % int(args.report_every) == 0:
                _print_record(_summarize_unilab(env, state, step, phase="step"))
    finally:
        metadata = {
            "backend": "unilab",
            "actions": args.actions,
            "policy": str(_resolve_policy_path(args.policy, args.project_path)) if args.actions == "policy" else "",
            "num_envs": int(args.num_envs),
            "steps": int(args.steps),
            "seed": int(args.seed),
            "model_debug": _unilab_model_debug(env) if args.dump_model_debug else {},
            "terrain_debug": _unilab_terrain_debug(env) if args.dump_terrain_debug else {},
        }
        _write_history(_dump_path(args, "unilab"), history, metadata)
        close = getattr(env, "close", None)
        if close is not None:
            close()
        elif hasattr(env, "_backend") and hasattr(env._backend, "close"):
            env._backend.close()
    return 0


def run_unilab_parent(args: argparse.Namespace) -> int:
    unilab_root = Path(args.unilab_root).resolve()
    python = unilab_root / ".venv/bin/python"
    if not python.exists():
        print(f"UniLab venv python not found: {python}", file=sys.stderr)
        return 2
    env = os.environ.copy()
    src_paths = [
        str(unilab_root / "src"),
        str(REPO_ROOT),
        str(REPO_ROOT / "python"),
        str(REPO_ROOT / "build/python"),
    ]
    if env.get("PYTHONPATH"):
        src_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(src_paths)
    env.setdefault("XDG_CACHE_HOME", "/tmp/unilab-cache")
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python),
        str(Path(__file__).resolve()),
        "--_unilab-child",
        "--unilab-root",
        str(unilab_root),
        "--unilab-config",
        str(Path(args.unilab_config).resolve()),
        "--backend",
        str(args.backend),
        "--project-path",
        str(Path(args.project_path).resolve()),
        "--policy",
        str(args.policy),
        "--num-envs",
        str(args.num_envs),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--actions",
        str(args.actions),
        "--report-every",
        str(args.report_every),
    ]
    if args.eval_reset:
        cmd.append("--eval-reset")
    if args.fixed_reset:
        cmd.append("--fixed-reset")
        cmd.append(f"--fixed-base-xy={args.fixed_base_xy}")
        cmd.append(f"--fixed-command={args.fixed_command}")
        cmd.append(f"--fixed-heading={args.fixed_heading}")
        cmd.append(f"--fixed-base-z={args.fixed_base_z}")
    if args.no_domain_rand:
        cmd.append("--no-domain-rand")
    if args.no_push:
        cmd.append("--no-push")
    if args.no_obs_noise:
        cmd.append("--no-obs-noise")
    if args.dump_npz is not None:
        cmd.extend(["--dump-npz", str(Path(args.dump_npz).resolve())])
    if args.dump_model_debug:
        cmd.append("--dump-model-debug")
    if args.dump_terrain_debug:
        cmd.append("--dump-terrain-debug")
    result = subprocess.run(cmd, cwd=str(unilab_root), env=env, check=False)
    return int(result.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gobot", "unilab", "both"), default="both")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actions", choices=("zero", "random", "policy"), default="zero")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--report-every", type=int, default=16)
    parser.add_argument("--dump-npz", type=Path, default=None)
    parser.add_argument("--dump-model-debug", action="store_true")
    parser.add_argument("--dump-terrain-debug", action="store_true")
    parser.add_argument("--sim-workers", type=int, default=0)
    parser.add_argument("--project-path", type=Path, default=REPO_ROOT / "examples/go1")
    parser.add_argument("--unilab-root", type=Path, default=DEFAULT_UNILAB_ROOT)
    parser.add_argument("--unilab-config", type=Path, default=DEFAULT_UNILAB_CFG)
    parser.add_argument("--eval-reset", action="store_true", help="Use Gobot video/eval-style reset flags.")
    parser.add_argument("--fixed-reset", action="store_true", help="Force both backends to the same deterministic reset state.")
    parser.add_argument("--fixed-base-xy", default="0.0,0.0", help="Base x,y for --fixed-reset.")
    parser.add_argument("--fixed-command", default="0.0,0.0,0.0", help="Command vx,vy,yaw for --fixed-reset.")
    parser.add_argument("--fixed-heading", type=float, default=0.0, help="Heading target for --fixed-reset.")
    parser.add_argument("--fixed-base-z", type=float, default=0.32, help="Base z for --fixed-reset.")
    parser.add_argument("--no-domain-rand", action="store_true", help="Disable Gobot domain randomization.")
    parser.add_argument("--no-push", action="store_true", help="Disable Gobot push forces.")
    parser.add_argument("--no-obs-noise", action="store_true", help="Disable Gobot actor observation noise.")
    parser.add_argument("--_unilab-child", action="store_true", help=argparse.SUPPRESS)
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.fixed_reset:
        args.no_domain_rand = True
        args.no_push = True
        args.no_obs_noise = True
    return args


def main(argv: list[str] | None = None) -> int:
    args = normalize_args(build_parser().parse_args(argv))
    args.report_every = max(1, int(args.report_every))
    if args._unilab_child:
        return run_unilab_child(args)

    status = 0
    if args.backend in {"gobot", "both"}:
        status = max(status, run_gobot(args))
    if args.backend in {"unilab", "both"}:
        status = max(status, run_unilab_parent(args))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
