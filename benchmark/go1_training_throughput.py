"""Measure real Go1 training-step throughput through the RSL-RL wrapper."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GO1_PROJECT = REPO_ROOT / "examples/go1"

try:
    from examples.go1.train._repo_imports import prefer_repo_gobot
except ImportError:
    import sys as _sys

    _repo_root_string = str(REPO_ROOT)
    if _repo_root_string not in _sys.path:
        _sys.path.insert(0, _repo_root_string)
    from examples.go1.train._repo_imports import prefer_repo_gobot

prefer_repo_gobot()

import torch

from examples.go1.train.go1_velocity_cfg import go1_velocity_cfg
from examples.go1.train.go1_velocity_env import Go1VelocityEnv
from gobot.rl.rsl_rl import RslRlVecEnvWrapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="go1_flat", help="Go1 velocity task: go1_flat or go1_rough.")
    parser.add_argument("--num-envs", "--num_envs", type=int, default=2048)
    parser.add_argument("--steps", "--num-steps", "--num_steps", type=int, default=20)
    parser.add_argument("--warmup-steps", "--warmup_steps", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-workers", type=int, default=0)
    parser.add_argument("--actions", choices=("zero", "random", "policy"), default="policy")
    parser.add_argument("--no-obs-noise", dest="obs_noise", action="store_false", default=True)
    parser.add_argument("--terrain-curriculum", action="store_true", default=False)
    parser.add_argument("--profile-step", action="store_true")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    cfg = go1_velocity_cfg(args.task, project_path=GO1_PROJECT)
    cfg.observations.actor_noise = bool(args.obs_noise)
    cfg.terrain_curriculum = bool(args.terrain_curriculum)

    core_env = Go1VelocityEnv(
        cfg,
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        sim_workers=args.sim_workers,
        profile_step=args.profile_step,
    )
    env = RslRlVecEnvWrapper(core_env, device=args.device)
    actor = _Actor(env.num_obs, env.num_actions, device=args.device, seed=args.seed)
    generator = torch.Generator(device=args.device)
    generator.manual_seed(int(args.seed) + 10_000)

    try:
        obs = env.get_observations()
        for _ in range(args.warmup_steps):
            actions = _make_actions(args.actions, env, actor, obs, generator)
            obs, _, _, _ = env.step(actions)

        timing_records: dict[str, list[float]] = {}
        total_t0 = time.perf_counter()
        for _ in range(args.steps):
            actions = _make_actions(args.actions, env, actor, obs, generator)
            obs, _, _, _ = env.step(actions)
            _append_record(timing_records, env.last_step_profile_ms())
            state_timing = core_env.state.info.get("timing", {}) if core_env.state is not None else {}
            _append_record(timing_records, {f"core_{key}": float(value) for key, value in state_timing.items()})
        elapsed = time.perf_counter() - total_t0

        metrics = _metrics(args=args, env=env, core_env=core_env, elapsed=elapsed, timing_records=timing_records)
        print(_format_table(metrics))
        if args.json_out:
            path = Path(args.json_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"\nWrote JSON: {path}")
    finally:
        env.close()


class _Actor:
    def __init__(self, obs_dim: int, action_dim: int, *, device: str, seed: int) -> None:
        torch.manual_seed(int(seed))
        self.model = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, action_dim),
            torch.nn.Tanh(),
        ).to(device)

    def __call__(self, obs) -> torch.Tensor:
        with torch.no_grad():
            return self.model(obs["policy"])


def _make_actions(mode: str, env: RslRlVecEnvWrapper, actor: _Actor, obs, generator: torch.Generator) -> torch.Tensor:
    if mode == "policy":
        return actor(obs)
    shape = (env.num_envs, env.num_actions)
    if mode == "zero":
        return torch.zeros(shape, dtype=torch.float32, device=env.device)
    return torch.rand(shape, dtype=torch.float32, device=env.device, generator=generator) * 2.0 - 1.0


def _append_record(records: dict[str, list[float]], values: dict[str, float]) -> None:
    for key, value in values.items():
        records.setdefault(str(key), []).append(float(value))


def _metrics(*, args, env: RslRlVecEnvWrapper, core_env: Go1VelocityEnv, elapsed: float, timing_records: dict[str, list[float]]) -> dict:
    env_steps = int(args.steps * env.num_envs)
    wrapper_total = np.asarray(timing_records.get("wrapper_total_ms", []), dtype=np.float64)
    total_s = float(wrapper_total.sum() / 1000.0)
    return {
        "task": core_env.cfg_obj.name,
        "device": args.device,
        "actions": args.actions,
        "num_envs": env.num_envs,
        "steps": int(args.steps),
        "warmup_steps": int(args.warmup_steps),
        "sim_workers_requested": int(args.sim_workers),
        "sim_workers_resolved": int(core_env.resolved_sim_workers),
        "env_steps": env_steps,
        "elapsed_seconds": float(elapsed),
        "train_env_steps_per_s": float(env_steps / total_s) if total_s > 0.0 else 0.0,
        "mean_wrapper_step_ms": float(wrapper_total.mean()) if wrapper_total.size else 0.0,
        "p50_wrapper_step_ms": float(np.median(wrapper_total)) if wrapper_total.size else 0.0,
        "p95_wrapper_step_ms": _percentile(wrapper_total, 0.95),
        "timing_mean_ms": _mean(timing_records),
        "timing_median_ms": _median(timing_records),
    }


def _percentile(values: np.ndarray, fraction: float) -> float:
    if values.size == 0:
        return 0.0
    sorted_values = np.sort(values)
    index = min(sorted_values.size - 1, max(0, round((sorted_values.size - 1) * fraction)))
    return float(sorted_values[index])


def _mean(records: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(np.mean(np.asarray(values, dtype=np.float64))) for key, values in records.items() if values}


def _median(records: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(np.median(np.asarray(values, dtype=np.float64))) for key, values in records.items() if values}


def _format_table(metrics: dict) -> str:
    rows = [
        ("train_env_steps_per_s", metrics["train_env_steps_per_s"]),
        ("mean_wrapper_step_ms", metrics["mean_wrapper_step_ms"]),
        ("p50_wrapper_step_ms", metrics["p50_wrapper_step_ms"]),
        ("p95_wrapper_step_ms", metrics["p95_wrapper_step_ms"]),
    ]
    for key in (
        "action_to_numpy_ms",
        "env_step_ms",
        "obs_to_tensor_ms",
        "extras_to_tensor_ms",
        "core_backend_physics_ms",
        "core_native_step_total_ms",
        "core_native_total_ms",
        "core_native_prepare_action_ms",
        "core_native_apply_ctrl_ms",
        "core_native_mj_step_ms",
        "core_native_extract_state_ms",
        "core_native_command_ms",
        "core_native_reward_ms",
        "core_native_obs_ms",
        "core_update_state_ms",
        "core_reset_done_ms",
        "core_env_step_total_ms",
    ):
        if key in metrics["timing_median_ms"]:
            rows.append((f"median_{key}", metrics["timing_median_ms"][key]))
    width = max(len(name) for name, _ in rows)
    lines = [
        f"Task: {metrics['task']}",
        f"Device: {metrics['device']}",
        f"Envs: {metrics['num_envs']}",
        f"Actions: {metrics['actions']}",
        "",
        "Metric".ljust(width) + "  Value",
        "-" * width + "  " + "-" * 12,
    ]
    lines.extend(f"{name.ljust(width)}  {value:.3f}" for name, value in rows)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
