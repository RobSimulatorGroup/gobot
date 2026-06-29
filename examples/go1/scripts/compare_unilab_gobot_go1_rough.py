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
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UNILAB_ROOT = REPO_ROOT / "UniLab"
DEFAULT_UNILAB_CFG = DEFAULT_UNILAB_ROOT / "conf/ppo/task/go1_joystick_rough/mujoco.yaml"


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
    raise ValueError(f"unsupported action mode {mode!r}")


def _print_record(record: dict[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


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
    env = Go1VelocityEnv(
        cfg,
        num_envs=int(args.num_envs),
        device="cpu",
        seed=int(args.seed),
        sim_workers=int(args.sim_workers),
        collect_step_extras=True,
    )
    try:
        _print_record(_summarize_gobot(env, 0, phase="reset"))
        for step in range(1, int(args.steps) + 1):
            action = _action_batch(
                mode=args.actions,
                rng=rng,
                num_envs=env.num_envs,
                num_actions=env.num_actions,
            )
            env.step(action)
            if step == int(args.steps) or step % int(args.report_every) == 0:
                _print_record(_summarize_gobot(env, step, phase="step"))
    finally:
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
    state = env.init_state()
    if args.eval_reset:
        qpos = np.tile(env._init_qpos, (env.num_envs, 1))
        qvel = np.zeros((env.num_envs, env._backend.nv), dtype=np.float32)
        spawn_origins = env._spawn.origins_for(np.arange(env.num_envs, dtype=np.int64))
        qpos[:, :3] += spawn_origins
        env._backend.set_state(np.arange(env.num_envs, dtype=np.int32), qpos, qvel, randomization=None)
        state = env.update_state(state)
    try:
        _print_record(_summarize_unilab(env, state, 0, phase="reset"))
        num_actions = int(getattr(env, "_num_action", env.action_space.shape[0]))
        for step in range(1, int(args.steps) + 1):
            action = _action_batch(
                mode=args.actions,
                rng=rng,
                num_envs=env.num_envs,
                num_actions=num_actions,
            )
            state = env.step(action)
            if step == int(args.steps) or step % int(args.report_every) == 0:
                _print_record(_summarize_unilab(env, state, step, phase="step"))
    finally:
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
    src = str(unilab_root / "src")
    env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else f"{src}:{env['PYTHONPATH']}"
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
    if args.no_domain_rand:
        cmd.append("--no-domain-rand")
    if args.no_push:
        cmd.append("--no-push")
    if args.no_obs_noise:
        cmd.append("--no-obs-noise")
    result = subprocess.run(cmd, cwd=str(unilab_root), env=env, check=False)
    return int(result.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gobot", "unilab", "both"), default="both")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actions", choices=("zero", "random"), default="zero")
    parser.add_argument("--report-every", type=int, default=16)
    parser.add_argument("--sim-workers", type=int, default=0)
    parser.add_argument("--project-path", type=Path, default=REPO_ROOT / "examples/go1")
    parser.add_argument("--unilab-root", type=Path, default=DEFAULT_UNILAB_ROOT)
    parser.add_argument("--unilab-config", type=Path, default=DEFAULT_UNILAB_CFG)
    parser.add_argument("--eval-reset", action="store_true", help="Use Gobot video/eval-style reset flags.")
    parser.add_argument("--no-domain-rand", action="store_true", help="Disable Gobot domain randomization.")
    parser.add_argument("--no-push", action="store_true", help="Disable Gobot push forces.")
    parser.add_argument("--no-obs-noise", action="store_true", help="Disable Gobot actor observation noise.")
    parser.add_argument("--_unilab-child", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
