"""Export Gobot's deterministic Go1 trace for parity comparison."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess

import torch

import gobot

from examples.go1.go1_velocity_contract import GO1_TASK_VERSION
from examples.go1.train.go1_velocity_cfg import (
    GO1_ROUGH_REWARD_TERM_NAMES,
    go1_velocity_cfg,
)
from examples.go1.train.go1_warp_velocity_env import Go1WarpVelocityEnv

from .go1_parity import (
    PARITY_ACTIONS,
    PARITY_COMMAND,
    PARITY_NUM_ENVS,
    PARITY_SEED,
    PARITY_TERRAIN_ROWS,
    PARITY_TERRAIN_SEED,
    PARITY_TERRAIN_TYPES,
    REFERENCE_TASK_ID,
    TRACE_SCHEMA_VERSION,
    mujoco_model_diagnostics,
    to_json_value,
    write_trace,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _repository_revision() -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def deterministic_cfg():
    cfg = go1_velocity_cfg(project_path=REPO_ROOT / "examples/go1")
    cfg.observations.actor_noise = False
    cfg.domain_randomization.enabled = False
    cfg.push_enabled = False
    cfg.spawn_jitter = 0.0
    cfg.reset_z_range = (0.03, 0.03)
    cfg.randomize_reset_yaw = False
    cfg.max_init_terrain_level = 0
    cfg.command.rel_standing_envs = 0.0
    cfg.command.rel_heading_envs = 0.0
    cfg.command.rel_world_envs = 0.0
    cfg.command.rel_forward_envs = 0.0
    cfg.command.ranges.lin_vel_x = (PARITY_COMMAND[0], PARITY_COMMAND[0])
    cfg.command.ranges.lin_vel_y = (PARITY_COMMAND[1], PARITY_COMMAND[1])
    cfg.command.ranges.ang_vel_z = (PARITY_COMMAND[2], PARITY_COMMAND[2])
    return cfg


def _snapshot(env: Go1WarpVelocityEnv, obs, *, step: int, action, state=None) -> dict:
    terrain = env.provider.raycast_sensor("terrain_scan")
    snapshot = {
        "step": step,
        "action": action,
        "qpos": env.provider.arrays["qpos"][0],
        "qvel": env.provider.arrays["qvel"][0],
        "actor": obs["actor"][0],
        "critic": obs["critic"][0],
        "command": env.command_b[0],
        "terrain": {
            "level": env._terrain_levels[0],
            "type": env._terrain_types[0],
            "origin": env._env_origins[0],
        },
        "sensors": {
            "terrain_distances": terrain["distances"][0],
            "terrain_hit_z": terrain["hit_pos_w"][0, :, 2],
            "foot_height": env._foot_height[0],
            "current_air_time": env._current_air_time[0],
            "contact_found": env._feet_contact["found"][0] > 0,
            "contact_force": env._feet_contact["force"][0],
        },
        "reward": None if state is None else state.reward[0],
        "reward_terms": None if state is None else env._reward_terms[0],
        "terminated": None if state is None else state.terminated[0],
        "truncated": None if state is None else state.truncated[0],
    }
    return to_json_value(snapshot)


def _assign_terrain_matrix(env: Go1WarpVelocityEnv) -> None:
    levels = torch.arange(
        PARITY_TERRAIN_ROWS, dtype=torch.long, device=env.device
    ).repeat_interleave(len(PARITY_TERRAIN_TYPES))
    types = torch.arange(
        len(PARITY_TERRAIN_TYPES), dtype=torch.long, device=env.device
    ).repeat(PARITY_TERRAIN_ROWS)
    env._terrain_levels.copy_(levels)
    env._terrain_types.copy_(types)
    env._env_origins.copy_(env._terrain_origin_for(env._all_env_ids))


def _terrain_matrix_snapshot(env: Go1WarpVelocityEnv, obs, *, state=None) -> dict:
    result = {
        "qpos": env.provider.arrays["qpos"],
        "qvel": env.provider.arrays["qvel"],
        "height_scan": obs["actor"][:, -187:],
        "foot_height": env._foot_height,
        "contact_found": env._feet_contact["found"] > 0,
        "reward": None if state is None else state.reward,
        "reward_terms": None if state is None else env._reward_terms,
        "terminated": None if state is None else state.terminated,
        "truncated": None if state is None else state.truncated,
    }
    return to_json_value(result)


def export_trace(*, output: str | Path, device: str) -> Path:
    env = Go1WarpVelocityEnv(
        deterministic_cfg(),
        num_envs=PARITY_NUM_ENVS,
        device=device,
        seed=PARITY_SEED,
        collect_step_extras=False,
        capture_graphs=True,
        context=gobot.app.create_context(),
    )
    try:
        _assign_terrain_matrix(env)
        obs, _ = env.reset(seed=PARITY_SEED)
        terrain_origins = to_json_value(env._env_origins)
        reset_matrix = _terrain_matrix_snapshot(env, obs)
        snapshots = [_snapshot(env, obs, step=0, action=None)]
        first_step_matrix = None
        for step, values in enumerate(PARITY_ACTIONS, start=1):
            action = torch.tensor((values,), dtype=torch.float32, device=device).expand(
                env.num_envs, -1
            )
            state = env.step(action)
            if step == 1:
                first_step_matrix = _terrain_matrix_snapshot(
                    env, state.obs, state=state
                )
            snapshots.append(
                _snapshot(
                    env,
                    state.obs,
                    step=step,
                    action=list(values),
                    state=state,
                )
            )
        trace = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "metadata": {
                "source": "gobot",
                "task_id": REFERENCE_TASK_ID,
                "task_version": GO1_TASK_VERSION,
                "source_revision": _repository_revision(),
                "seed": PARITY_SEED,
                "terrain_seed": PARITY_TERRAIN_SEED,
                "device": device,
                "packages": {
                    "gobot": _package_version("gobot"),
                    "mujoco": _package_version("mujoco"),
                    "mujoco-warp": _package_version("mujoco-warp"),
                    "torch": _package_version("torch"),
                    "warp-lang": _package_version("warp-lang"),
                },
                "physics_dt": float(env.physics_dt),
                "decimation": int(env.decimation),
                "step_dt": float(env.step_dt),
                "joint_names": list(env.joint_names),
                "reward_names": list(GO1_ROUGH_REWARD_TERM_NAMES),
                "actor_dim": int(env.num_obs),
                "critic_dim": int(env.num_privileged_obs),
            },
            "terrain_matrix": {
                "rows": PARITY_TERRAIN_ROWS,
                "cols": len(PARITY_TERRAIN_TYPES),
                "type_names": list(PARITY_TERRAIN_TYPES),
                "levels": [
                    level
                    for level in range(PARITY_TERRAIN_ROWS)
                    for _ in PARITY_TERRAIN_TYPES
                ],
                "types": list(range(len(PARITY_TERRAIN_TYPES))) * PARITY_TERRAIN_ROWS,
                "origins": terrain_origins,
                "reset": reset_matrix,
                "first_step": first_step_matrix,
            },
            "mujoco_model": mujoco_model_diagnostics(env.provider._mj_model),
            "snapshots": snapshots,
        }
        return write_trace(output, trace)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    path = export_trace(output=args.output, device=args.device)
    print(f"Wrote Gobot trace: {path}")


if __name__ == "__main__":
    main()
