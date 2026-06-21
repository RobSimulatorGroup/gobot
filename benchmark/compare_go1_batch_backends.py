"""Compare Go1 training, Gobot raw MuJoCo, and UniLab-style batch stepping."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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

from benchmark.go1_training_throughput import _Actor, _append_record
from benchmark.go1_training_throughput import _make_actions as _make_training_actions
from benchmark.go1_training_throughput import _metrics as _training_metrics
from benchmark.go1_velocity_benchmark import build_benchmark_metrics
from benchmark.mujoco_uni_batch_benchmark import run_batch_benchmark
from examples.go1.train.go1_velocity_cfg import go1_velocity_cfg
from examples.go1.train.go1_velocity_env import Go1VelocityEnv
from gobot.rl.rsl_rl import RslRlVecEnvWrapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="go1_flat")
    parser.add_argument("--num-envs", "--num_envs", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", "--warmup_steps", type=int, default=5)
    parser.add_argument("--sim-workers", type=int, default=0)
    parser.add_argument("--raw-threads", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actions", choices=("zero", "random", "policy"), default="random")
    parser.add_argument("--raw-actions", choices=("zero", "random"), default=None)
    parser.add_argument("--include-training", action="store_true", default=False)
    parser.add_argument("--include-batch-env", action="store_true", default=False)
    parser.add_argument("--include-rollout", action="store_true", default=True)
    parser.add_argument("--no-obs-noise", dest="obs_noise", action="store_false", default=True)
    parser.add_argument("--no-step-extras", action="store_true", default=False)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    raw_actions = args.raw_actions or ("zero" if args.actions == "zero" else "random")
    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "task": args.task,
        "num_envs": int(args.num_envs),
        "steps": int(args.steps),
        "warmup_steps": int(args.warmup_steps),
        "device": args.device,
        "actions": args.actions,
        "raw_actions": raw_actions,
        "entries": {},
    }

    env_metrics = _run_go1_env(args)
    metrics["entries"]["gobot_env"] = env_metrics
    rows.append(_env_row("Gobot Go1 env", env_metrics))

    if args.include_training:
        train_metrics = _run_go1_training(args)
        metrics["entries"]["gobot_training_wrapper"] = train_metrics
        rows.append(_training_row("Gobot RSL-RL wrapper", train_metrics))

    raw_common = {
        "xml": GO1_PROJECT / "assets/xml/go1_scene.xml",
        "num_envs": int(args.num_envs),
        "steps": int(args.steps),
        "warmup_steps": int(args.warmup_steps),
        "nstep": int(env_metrics["decimation"]),
        "threads": int(args.raw_threads),
        "actions": raw_actions,
    }
    raw_gobot = run_batch_benchmark(backend="gobot", **raw_common)
    metrics["entries"]["gobot_raw_mujoco_pool"] = raw_gobot
    rows.append(_raw_row("Gobot raw _MujocoBatchPool", raw_gobot))

    if args.include_batch_env:
        try:
            batch_env = run_batch_benchmark(backend="batch_env", **raw_common)
            metrics["entries"]["unilab_batch_env"] = batch_env
            rows.append(_raw_row("UniLab/mujoco batch_env", batch_env))
        except Exception as exc:  # noqa: BLE001 - benchmark should report unavailable optional backends.
            metrics["entries"]["unilab_batch_env"] = {"error": str(exc)}
            rows.append(_error_row("UniLab/mujoco batch_env", exc))

    if args.include_rollout:
        try:
            rollout = run_batch_benchmark(backend="rollout", **raw_common)
            metrics["entries"]["mujoco_rollout"] = rollout
            rows.append(_raw_row("Official mujoco.rollout", rollout))
        except Exception as exc:  # noqa: BLE001
            metrics["entries"]["mujoco_rollout"] = {"error": str(exc)}
            rows.append(_error_row("Official mujoco.rollout", exc))

    print(_format_markdown_table(rows))
    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote JSON: {output_path}")


def _run_go1_env(args: argparse.Namespace) -> dict[str, Any]:
    cfg = go1_velocity_cfg(args.task, project_path=GO1_PROJECT)
    cfg.observations.actor_noise = bool(args.obs_noise)
    env = Go1VelocityEnv(
        cfg,
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        sim_workers=args.sim_workers,
        profile_step=False,
        collect_step_extras=not args.no_step_extras,
    )
    try:
        generator = np.random.default_rng(int(args.seed) + 10_000)
        for _ in range(args.warmup_steps):
            env.step(_make_env_actions(env, args.actions, generator))
        timing_records: dict[str, list[float]] = {}
        total_begin = time.perf_counter()
        for _ in range(args.steps):
            state = env.step(_make_env_actions(env, args.actions, generator))
            for key, value in state.info.get("timing", {}).items():
                timing_records.setdefault(str(key), []).append(float(value))
        elapsed = time.perf_counter() - total_begin
        bench_args = SimpleNamespace(
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            actions=args.actions,
            sim_workers=args.sim_workers,
        )
        metrics = build_benchmark_metrics(
            cfg_name=cfg.name,
            env=env,
            args=bench_args,
            elapsed=elapsed,
            timing_records=timing_records,
        )
        metrics["collect_step_extras"] = bool(not args.no_step_extras)
        return metrics
    finally:
        env.close()


def _run_go1_training(args: argparse.Namespace) -> dict[str, Any]:
    cfg = go1_velocity_cfg(args.task, project_path=GO1_PROJECT)
    cfg.observations.actor_noise = bool(args.obs_noise)
    core_env = Go1VelocityEnv(
        cfg,
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        sim_workers=args.sim_workers,
        profile_step=False,
        collect_step_extras=not args.no_step_extras,
    )
    env = RslRlVecEnvWrapper(core_env, device=args.device)
    actor = _Actor(env.num_obs, env.num_actions, device=args.device, seed=args.seed)
    generator = env.torch.Generator(device=args.device)
    generator.manual_seed(int(args.seed) + 10_000)
    try:
        obs = env.get_observations()
        for _ in range(args.warmup_steps):
            actions = _make_training_actions(args.actions, env, actor, obs, generator)
            obs, _, _, _ = env.step(actions)
        timing_records: dict[str, list[float]] = {}
        total_begin = time.perf_counter()
        for _ in range(args.steps):
            actions = _make_training_actions(args.actions, env, actor, obs, generator)
            obs, _, _, _ = env.step(actions)
            _append_record(timing_records, env.last_step_profile_ms())
            state_timing = core_env.state.info.get("timing", {}) if core_env.state is not None else {}
            _append_record(timing_records, {f"core_{key}": float(value) for key, value in state_timing.items()})
        elapsed = time.perf_counter() - total_begin
        metrics = _training_metrics(args=args, env=env, core_env=core_env, elapsed=elapsed, timing_records=timing_records)
        metrics["collect_step_extras"] = bool(not args.no_step_extras)
        return metrics
    finally:
        env.close()


def _make_env_actions(env: Go1VelocityEnv, mode: str, generator: np.random.Generator) -> np.ndarray:
    shape = (env.num_envs, env.num_actions)
    if mode == "zero":
        return np.zeros(shape, dtype=np.float32)
    return generator.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def _env_row(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    median = metrics.get("timing_median_ms", {})
    return {
        "name": name,
        "env_steps_s": metrics.get("throughput_env_steps_per_s"),
        "total_ms": median.get("env_step_total_ms", metrics.get("mean_step_ms")),
        "mj_step_ms": median.get("native_mj_step_ms"),
        "state_extract_ms": median.get("native_extract_state_ms"),
        "reward_ms": median.get("native_reward_ms"),
        "obs_ms": median.get("native_obs_ms"),
        "reset_ms": median.get("reset_done_ms"),
        "python_ms": _sum_present(median, ("apply_action_ms", "backend_refresh_cache_ms", "update_state_ms", "reset_done_ms")),
        "notes": f"workers={metrics.get('sim_workers_resolved')}",
    }


def _training_row(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    median = metrics.get("timing_median_ms", {})
    return {
        "name": name,
        "env_steps_s": metrics.get("train_env_steps_per_s"),
        "total_ms": median.get("wrapper_total_ms", metrics.get("mean_wrapper_step_ms")),
        "mj_step_ms": median.get("core_native_mj_step_ms"),
        "state_extract_ms": median.get("core_native_extract_state_ms"),
        "reward_ms": median.get("core_native_reward_ms"),
        "obs_ms": median.get("core_native_obs_ms"),
        "reset_ms": median.get("core_reset_done_ms"),
        "python_ms": _sum_present(median, ("action_to_numpy_ms", "obs_to_tensor_ms", "extras_to_tensor_ms")),
        "notes": f"workers={metrics.get('sim_workers_resolved')}",
    }


def _raw_row(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    median = metrics.get("timing_median_ms", {})
    return {
        "name": name,
        "env_steps_s": metrics.get("env_steps_per_second"),
        "total_ms": median.get("total_ms", metrics.get("p50_call_ms")),
        "mj_step_ms": median.get("mj_step_ms"),
        "state_extract_ms": median.get("state_store_ms"),
        "reward_ms": None,
        "obs_ms": None,
        "reset_ms": None,
        "python_ms": None,
        "notes": f"backend={metrics.get('backend')}, threads={metrics.get('threads')}",
    }


def _error_row(name: str, error: Exception) -> dict[str, Any]:
    return {
        "name": name,
        "env_steps_s": None,
        "total_ms": None,
        "mj_step_ms": None,
        "state_extract_ms": None,
        "reward_ms": None,
        "obs_ms": None,
        "reset_ms": None,
        "python_ms": None,
        "notes": f"unavailable: {error}",
    }


def _sum_present(values: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    present = [float(values[key]) for key in keys if key in values]
    return float(sum(present)) if present else None


def _format_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Backend",
        "env-steps/s",
        "total ms",
        "mj_step ms",
        "extract ms",
        "reward ms",
        "obs ms",
        "reset ms",
        "python/wrapper ms",
        "Notes",
    ]
    body = []
    for row in rows:
        body.append(
            [
                str(row["name"]),
                _fmt(row["env_steps_s"], precision=0),
                _fmt(row["total_ms"]),
                _fmt(row["mj_step_ms"]),
                _fmt(row["state_extract_ms"]),
                _fmt(row["reward_ms"]),
                _fmt(row["obs_ms"]),
                _fmt(row["reset_ms"]),
                _fmt(row["python_ms"]),
                str(row["notes"]),
            ]
        )
    widths = [len(header) for header in headers]
    for row in body:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    lines = [
        "| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |",
        "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |" for row in body)
    return "\n".join(lines)


def _fmt(value: Any, *, precision: int = 3) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "-"
    if precision == 0:
        return f"{numeric:.0f}"
    return f"{numeric:.{precision}f}"


if __name__ == "__main__":
    main()
