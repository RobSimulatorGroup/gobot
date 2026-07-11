"""Benchmark Go1 velocity environment stepping throughput."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GO1_PROJECT = REPO_ROOT / "examples/go1"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.go1.train.go1_velocity_cfg import go1_velocity_cfg
from examples.go1.train.go1_velocity_env import Go1VelocityEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", "--num_envs", type=int, default=2048)
    parser.add_argument("--steps", "--num-steps", "--num_steps", type=int, default=20, help="Measured env.step calls.")
    parser.add_argument("--warmup-steps", "--warmup_steps", type=int, default=5, help="Unmeasured env.step calls before timing.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-workers", type=int, default=0)
    parser.add_argument("--max-episode-length", type=int, default=None)
    parser.add_argument("--actions", choices=("zero", "random"), default="random")
    parser.add_argument("--no-obs-noise", dest="obs_noise", action="store_false", default=True)
    parser.add_argument("--terrain-curriculum", action="store_true", default=False)
    parser.add_argument("--profile-step", action="store_true", help="Enable Go1 env phase timing logs.")
    parser.add_argument("--no-step-extras", action="store_true", default=False, help="Disable per-step reward-term/log extras during timing.")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path for benchmark metrics JSON.")
    args = parser.parse_args()

    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps cannot be negative")

    cfg = go1_velocity_cfg(project_path=GO1_PROJECT)
    cfg.observations.actor_noise = bool(args.obs_noise)
    cfg.terrain_curriculum = bool(args.terrain_curriculum)

    env = Go1VelocityEnv(
        cfg,
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        max_episode_length=args.max_episode_length,
        sim_workers=args.sim_workers,
        profile_step=args.profile_step,
        collect_step_extras=not args.no_step_extras,
    )

    try:
        print(f"Task: {cfg.name}")
        print(f"Device: {args.device}")
        print(f"Envs: {env.num_envs}")
        print(f"Actions: {args.actions}")
        print(f"Warmup steps: {args.warmup_steps}")
        print(f"Measured steps: {args.steps}")
        print(f"Sim workers: requested={args.sim_workers} resolved={env.resolved_sim_workers}")
        print(f"Obs actor/critic/actions: {env.num_obs}/{env.num_privileged_obs}/{env.num_actions}")
        print(f"Physics dt: {env.physics_dt:.6f}s")
        print(f"Policy dt: {env.step_dt:.6f}s")
        print(f"Decimation: {env.decimation}")

        generator = np.random.default_rng(int(args.seed) + 10_000)

        for _ in range(args.warmup_steps):
            env.step(_make_actions(env, args.actions, generator))

        timing_records: dict[str, list[float]] = {}
        total_begin = time.perf_counter()
        for _ in range(args.steps):
            state = env.step(_make_actions(env, args.actions, generator))
            for key, value in state.info.get("timing", {}).items():
                timing_records.setdefault(str(key), []).append(float(value))
        elapsed = time.perf_counter() - total_begin

        metrics = build_benchmark_metrics(
            cfg_name=cfg.name,
            env=env,
            args=args,
            elapsed=elapsed,
            timing_records=timing_records,
        )
        metrics.update({
            "task": cfg.name,
            "device": args.device,
        })
        if args.profile_step:
            metrics["profile_ms"] = env.profile_summary()

        print("")
        print("Benchmark:")
        for key in (
            "elapsed_seconds",
            "throughput_env_steps_per_s",
            "physics_ticks_per_second",
            "mean_step_ms",
            "p50_step_ms",
            "p95_step_ms",
        ):
            print(f"  {key}: {metrics[key]:.3f}")
        print("  breakdown_median_ms:")
        for key in (
            "env_step_total_ms",
            "apply_action_ms",
            "step_core_ms",
            "backend_apply_action_ms",
            "backend_physics_ms",
            "native_step_total_ms",
            "native_total_ms",
            "native_prepare_action_ms",
            "native_apply_ctrl_ms",
            "native_mj_step_ms",
            "native_extract_state_ms",
            "native_command_ms",
            "native_reward_ms",
            "native_obs_ms",
            "backend_refresh_cache_ms",
            "update_state_ms",
            "reset_done_ms",
        ):
            if key in metrics["timing_median_ms"]:
                print(f"    {key}: {metrics['timing_median_ms'][key]:.3f}")
        if args.profile_step:
            print("  profile_ms:")
            for name, value in sorted(metrics["profile_ms"].items()):
                print(f"    {name}: {value:.3f}")

        if args.json_out:
            output_path = Path(args.json_out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"\nWrote JSON: {output_path}")
    finally:
        env.close()


def _make_actions(env: Go1VelocityEnv, mode: str, generator: np.random.Generator) -> np.ndarray:
    shape = (env.num_envs, env.num_actions)
    if mode == "zero":
        return np.zeros(shape, dtype=np.float32)
    return generator.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def build_benchmark_metrics(*, cfg_name: str, env: Go1VelocityEnv, args, elapsed: float, timing_records: dict[str, list[float]]) -> dict:
    env_steps = int(args.steps * env.num_envs)
    physics_ticks = int(env_steps * env.decimation)
    total_arr = np.asarray(timing_records.get("env_step_total_ms", []), dtype=np.float64)
    total_s = float(total_arr.sum() / 1000.0)
    throughput = float(env_steps / total_s) if total_s > 0.0 else 0.0
    return {
        "task_name": cfg_name,
        "sim_backend": "gobot_mujoco_cpu",
        "num_envs": env.num_envs,
        "steps": int(args.steps),
        "num_steps": int(args.steps),
        "warmup_steps": int(args.warmup_steps),
        "actions": args.actions,
        "collect_step_extras": bool(not getattr(args, "no_step_extras", False)),
        "sim_workers_requested": int(args.sim_workers),
        "sim_workers_resolved": int(env.resolved_sim_workers),
        "physics_dt": float(env.physics_dt),
        "policy_dt": float(env.step_dt),
        "decimation": int(env.decimation),
        "elapsed_seconds": float(elapsed),
        "env_steps": env_steps,
        "throughput_env_steps_per_s": throughput,
        "env_steps_per_second": throughput,
        "physics_ticks": physics_ticks,
        "physics_ticks_per_second": float(physics_ticks / total_s) if total_s > 0.0 else 0.0,
        "timing_records": timing_records,
        "timing_mean_ms": _timing_mean(timing_records),
        "timing_median_ms": _timing_median(timing_records),
        "mean_step_ms": float(total_arr.mean()) if total_arr.size else 0.0,
        "min_step_ms": float(total_arr.min()) if total_arr.size else 0.0,
        "max_step_ms": float(total_arr.max()) if total_arr.size else 0.0,
        "p50_step_ms": float(np.median(total_arr)) if total_arr.size else 0.0,
        "p95_step_ms": _percentile_ms(total_arr, 0.95),
    }


def _percentile_ms(values: np.ndarray, fraction: float) -> float:
    if values.size == 0:
        return 0.0
    sorted_values = np.sort(values)
    index = min(sorted_values.size - 1, max(0, round((sorted_values.size - 1) * fraction)))
    return float(sorted_values[index])


def _timing_mean(records: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(np.mean(np.asarray(values, dtype=np.float64))) for key, values in records.items() if values}


def _timing_median(records: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(np.median(np.asarray(values, dtype=np.float64))) for key, values in records.items() if values}


if __name__ == "__main__":
    main()
