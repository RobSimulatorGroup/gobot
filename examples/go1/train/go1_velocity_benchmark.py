"""Benchmark Go1 velocity environment stepping throughput."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

try:
    from .go1_velocity_cfg import go1_velocity_cfg
    from .go1_velocity_env import Go1VelocityEnv
except ImportError:
    from go1_velocity_cfg import go1_velocity_cfg
    from go1_velocity_env import Go1VelocityEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="go1_flat", help="Go1 velocity task: go1_flat or go1_rough.")
    parser.add_argument("--num-envs", "--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100, help="Measured env.step calls.")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Unmeasured env.step calls before timing.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-workers", type=int, default=0)
    parser.add_argument("--max-episode-length", type=int, default=None)
    parser.add_argument("--actions", choices=("zero", "random"), default="random")
    parser.add_argument("--no-obs-noise", dest="obs_noise", action="store_false", default=True)
    parser.add_argument("--terrain-curriculum", action="store_true", default=False)
    parser.add_argument("--profile-step", action="store_true", help="Enable Go1 env phase timing logs.")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path for benchmark metrics JSON.")
    args = parser.parse_args()

    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps cannot be negative")

    project_path = Path(__file__).resolve().parents[1]
    cfg = go1_velocity_cfg(args.task, project_path=project_path)
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

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(args.seed) + 10_000)

        for _ in range(args.warmup_steps):
            env.step(_make_actions(env, args.actions, generator))

        step_times: list[float] = []
        total_begin = time.perf_counter()
        for _ in range(args.steps):
            begin = time.perf_counter()
            env.step(_make_actions(env, args.actions, generator))
            step_times.append(time.perf_counter() - begin)
        elapsed = time.perf_counter() - total_begin

        env_steps = int(args.steps * env.num_envs)
        physics_ticks = int(env_steps * env.decimation)
        metrics = {
            "task": cfg.name,
            "device": args.device,
            "num_envs": env.num_envs,
            "steps": int(args.steps),
            "warmup_steps": int(args.warmup_steps),
            "actions": args.actions,
            "sim_workers_requested": int(args.sim_workers),
            "sim_workers_resolved": int(env.resolved_sim_workers),
            "physics_dt": float(env.physics_dt),
            "policy_dt": float(env.step_dt),
            "decimation": int(env.decimation),
            "elapsed_seconds": float(elapsed),
            "step_calls_per_second": float(args.steps / elapsed),
            "env_steps": env_steps,
            "env_steps_per_second": float(env_steps / elapsed),
            "physics_ticks": physics_ticks,
            "physics_ticks_per_second": float(physics_ticks / elapsed),
            "mean_step_ms": float(statistics.fmean(step_times) * 1000.0),
            "min_step_ms": float(min(step_times) * 1000.0),
            "max_step_ms": float(max(step_times) * 1000.0),
            "p50_step_ms": _percentile_ms(step_times, 0.50),
            "p95_step_ms": _percentile_ms(step_times, 0.95),
        }
        if args.profile_step:
            metrics["profile_ms"] = env.profile_summary()

        print("")
        print("Benchmark:")
        for key in (
            "elapsed_seconds",
            "step_calls_per_second",
            "env_steps_per_second",
            "physics_ticks_per_second",
            "mean_step_ms",
            "p50_step_ms",
            "p95_step_ms",
        ):
            print(f"  {key}: {metrics[key]:.3f}")
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


def _make_actions(env: Go1VelocityEnv, mode: str, generator: torch.Generator) -> torch.Tensor:
    shape = (env.num_envs, env.num_actions)
    if mode == "zero":
        return torch.zeros(shape, dtype=torch.float32, device=env.device)
    actions = torch.rand(shape, dtype=torch.float32, generator=generator) * 2.0 - 1.0
    return actions.to(device=env.device)


def _percentile_ms(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * fraction)))
    return float(sorted_values[index] * 1000.0)


if __name__ == "__main__":
    main()
