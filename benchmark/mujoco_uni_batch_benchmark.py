"""Benchmark Gobot and UniLab-style MuJoCo batch state-array stepping.

The script prefers Gobot's internal ``_MujocoBatchPool`` when available. It can
also benchmark the mujoco_uni ``mujoco.batch_env.BatchEnvPool`` API when
``mujoco._batch_env`` is installed, or official ``mujoco.rollout.Rollout``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
import importlib.util
import json
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GO1_PROJECT = REPO_ROOT / "examples/go1"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=None, help="MJCF path. Defaults to examples/go1/assets/xml/go1_scene.xml.")
    parser.add_argument("--num-envs", "--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100, help="Measured calls.")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--nstep", type=int, default=10, help="MuJoCo physics steps per measured call.")
    parser.add_argument("--threads", type=int, default=0, help="Worker threads. 0 lets the backend choose serial/pool defaults.")
    parser.add_argument("--actions", choices=("zero", "random"), default="random")
    parser.add_argument("--backend", choices=("auto", "gobot", "batch_env", "rollout"), default="auto")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps cannot be negative")
    if args.nstep <= 0:
        raise ValueError("--nstep must be positive")

    metrics = run_batch_benchmark(
        xml=args.xml,
        num_envs=args.num_envs,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        nstep=args.nstep,
        threads=args.threads,
        actions=args.actions,
        backend=args.backend,
    )

    print("")
    print(f"Backend: {metrics['backend']}")
    print(f"XML: {metrics['xml']}")
    print(f"Envs: {metrics['num_envs']}")
    print(f"Threads: {metrics['threads']}")
    print(f"Actions: {metrics['actions']}")
    print(f"Warmup calls: {metrics['warmup_steps']}")
    print(f"Measured calls: {metrics['steps']}")
    print(f"nstep: {metrics['nstep']}")
    print(
        "nq/nv/nu/nstate/ncontrol/nsensordata: "
        f"{metrics['nq']}/{metrics['nv']}/{metrics['nu']}/"
        f"{metrics['nstate']}/{metrics['ncontrol']}/{metrics['nsensordata']}"
    )

    print("")
    print("Benchmark:")
    for key in (
        "elapsed_seconds",
        "step_calls_per_second",
        "env_steps_per_second",
        "physics_ticks_per_second",
        "mean_call_ms",
        "p50_call_ms",
        "p95_call_ms",
    ):
        print(f"  {key}: {metrics[key]:.3f}")
    if metrics.get("timing_median_ms"):
        print("  breakdown_median_ms:")
        for key, value in metrics["timing_median_ms"].items():
            print(f"    {key}: {value:.3f}")

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote JSON: {output_path}")


def run_batch_benchmark(
    *,
    xml: str | Path | None = None,
    num_envs: int = 64,
    steps: int = 100,
    warmup_steps: int = 10,
    nstep: int = 10,
    threads: int = 0,
    actions: str = "random",
    backend: str = "auto",
) -> dict[str, Any]:
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")
    if nstep <= 0:
        raise ValueError("nstep must be positive")

    xml_path = Path(xml) if xml is not None else GO1_PROJECT / "assets/xml/go1_scene.xml"
    if not xml_path.is_absolute():
        xml_path = REPO_ROOT / xml_path

    model = _load_model(xml_path)
    model.opt.timestep = 0.002
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    ncontrol = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_CTRL)
    state = _initial_state(model, int(num_envs), nstate)
    rng = np.random.default_rng(10_123)
    resolved_backend = _resolve_backend(backend)
    args = SimpleNamespace(
        num_envs=int(num_envs),
        steps=int(steps),
        warmup_steps=int(warmup_steps),
        nstep=int(nstep),
        threads=int(threads),
        actions=str(actions),
    )

    if resolved_backend == "gobot":
        metrics = _run_gobot_batch(xml_path, ncontrol, rng, args)
    elif resolved_backend == "batch_env":
        metrics = _run_batch_env(model, state, ncontrol, rng, args)
    else:
        metrics = _run_rollout(model, state, ncontrol, rng, args)

    metrics.update(
        {
            "backend": resolved_backend,
            "xml": str(xml_path),
            "num_envs": int(num_envs),
            "steps": int(steps),
            "warmup_steps": int(warmup_steps),
            "nstep": int(nstep),
            "threads": int(threads),
            "actions": str(actions),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "nstate": int(nstate),
            "ncontrol": int(ncontrol),
            "nsensordata": int(model.nsensordata),
        }
    )
    env_steps = int(steps * num_envs)
    physics_ticks = int(env_steps * nstep)
    metrics["env_steps"] = env_steps
    metrics["physics_ticks"] = physics_ticks
    metrics["step_calls_per_second"] = float(steps / metrics["elapsed_seconds"])
    metrics["env_steps_per_second"] = float(env_steps / metrics["elapsed_seconds"])
    metrics["physics_ticks_per_second"] = float(physics_ticks / metrics["elapsed_seconds"])
    return metrics


def _resolve_backend(requested: str) -> str:
    has_gobot_batch = _has_gobot_batch_pool()
    has_batch_env = importlib.util.find_spec("mujoco._batch_env") is not None
    if requested == "gobot" and not has_gobot_batch:
        raise RuntimeError("gobot._MujocoBatchPool is not available; rebuild Gobot with MuJoCo support")
    if requested == "batch_env" and not has_batch_env:
        raise RuntimeError("mujoco._batch_env is not installed; cannot run BatchEnvPool benchmark")
    if requested == "rollout":
        return "rollout"
    if requested == "gobot":
        return "gobot"
    if requested == "batch_env":
        return "batch_env"
    if has_gobot_batch:
        return "gobot"
    return "batch_env" if has_batch_env else "rollout"


def _has_gobot_batch_pool() -> bool:
    try:
        import gobot
    except Exception:
        return False
    return bool(getattr(gobot, "_has_mujoco_batch_pool", False)) and getattr(gobot, "_MujocoBatchPool", None) is not None


def _load_model(xml_path: Path) -> mujoco.MjModel:
    go1_project = None
    if xml_path.parent.name == "xml" and xml_path.parent.parent.name == "assets":
        go1_project = xml_path.parent.parent.parent
    if go1_project is not None:
        return _load_go1_model_with_absolute_meshdir(xml_path, go1_project)

    previous_cwd = Path.cwd()
    try:
        cwd = xml_path.parent
        os.chdir(cwd)
        return mujoco.MjModel.from_xml_path(os.path.relpath(xml_path, cwd))
    finally:
        os.chdir(previous_cwd)


def _load_go1_model_with_absolute_meshdir(xml_path: Path, project_path: Path) -> mujoco.MjModel:
    with _temporary_go1_xml(xml_path, project_path) as scene_xml:
        previous_cwd = Path.cwd()
        try:
            os.chdir(scene_xml.parent)
            return mujoco.MjModel.from_xml_path(scene_xml.name)
        finally:
            os.chdir(previous_cwd)


@contextmanager
def _temporary_go1_xml(xml_path: Path, project_path: Path):
    with tempfile.TemporaryDirectory(prefix="gobot_go1_mjcf_") as tmp:
        tmp_dir = Path(tmp)
        scene_xml = tmp_dir / xml_path.name
        robot_xml = tmp_dir / "go1.xml"
        scene_xml.write_text(xml_path.read_text(encoding="utf-8"), encoding="utf-8")
        robot_text = (project_path / "assets/xml/go1.xml").read_text(encoding="utf-8")
        meshdir = (project_path / "assets").as_posix()
        robot_text = robot_text.replace('meshdir="assets"', f'meshdir="{meshdir}"')
        robot_xml.write_text(robot_text, encoding="utf-8")
        yield scene_xml


@contextmanager
def _xml_path_for_c_loader(xml_path: Path):
    go1_project = None
    if xml_path.parent.name == "xml" and xml_path.parent.parent.name == "assets":
        go1_project = xml_path.parent.parent.parent
    if go1_project is not None:
        with _temporary_go1_xml(xml_path, go1_project) as scene_xml:
            yield scene_xml
        return
    yield xml_path


def _run_gobot_batch(xml_path: Path, ncontrol: int, rng: np.random.Generator, args: argparse.Namespace) -> dict[str, Any]:
    import gobot

    with _xml_path_for_c_loader(xml_path) as c_xml_path:
        pool = gobot._MujocoBatchPool(str(c_xml_path), num_envs=args.num_envs, threads=args.threads, timestep=0.002)
        state = pool.initial_state()
        if ncontrol != pool.ncontrol:
            raise RuntimeError(f"control size mismatch: Python model has {ncontrol}, Gobot pool has {pool.ncontrol}")

        for _ in range(args.warmup_steps):
            control = _make_control(args.num_envs, args.nstep, pool.ncontrol, args.actions, rng)
            state = pool.step(state, control=control, nstep=args.nstep)

        times: list[float] = []
        profile_records: dict[str, list[float]] = {}
        begin_total = time.perf_counter()
        for _ in range(args.steps):
            control = _make_control(args.num_envs, args.nstep, pool.ncontrol, args.actions, rng)
            begin = time.perf_counter()
            state = pool.step(state, control=control, nstep=args.nstep)
            times.append(time.perf_counter() - begin)
            for key, value in pool.step_profile().items():
                profile_records.setdefault(str(key), []).append(float(value))
        elapsed = time.perf_counter() - begin_total
        metrics = _timing_metrics(elapsed, times)
        metrics["timing_records"] = profile_records
        metrics["timing_mean_ms"] = _timing_mean(profile_records)
        metrics["timing_median_ms"] = _timing_median(profile_records)
        return metrics


def _run_batch_env(model: mujoco.MjModel, state: np.ndarray, ncontrol: int, rng: np.random.Generator, args: argparse.Namespace) -> dict[str, Any]:
    from mujoco.batch_env import BatchEnvPool

    pool = BatchEnvPool(model, nbatch=args.num_envs, nthread=args.threads)
    try:
        for _ in range(args.warmup_steps):
            control = _make_control(args.num_envs, args.nstep, ncontrol, args.actions, rng)
            state = pool.step(state, nstep=args.nstep, control=control)

        times: list[float] = []
        begin_total = time.perf_counter()
        for _ in range(args.steps):
            control = _make_control(args.num_envs, args.nstep, ncontrol, args.actions, rng)
            begin = time.perf_counter()
            state = pool.step(state, nstep=args.nstep, control=control)
            times.append(time.perf_counter() - begin)
        elapsed = time.perf_counter() - begin_total
        return _timing_metrics(elapsed, times)
    finally:
        pool.close()


def _run_rollout(model: mujoco.MjModel, state: np.ndarray, ncontrol: int, rng: np.random.Generator, args: argparse.Namespace) -> dict[str, Any]:
    from mujoco.rollout import Rollout

    datas = [mujoco.MjData(model) for _ in range(max(1, int(args.threads)))]
    rollout = Rollout(nthread=max(0, int(args.threads)))
    state_out = np.empty((args.num_envs, args.nstep, state.shape[1]), dtype=np.float64)
    sensordata = np.empty((args.num_envs, args.nstep, model.nsensordata), dtype=np.float64)
    try:
        for _ in range(args.warmup_steps):
            control = _make_control(args.num_envs, args.nstep, ncontrol, args.actions, rng)
            state_out, sensordata = rollout.rollout(
                model,
                datas,
                state,
                control=control,
                nstep=args.nstep,
                state=state_out,
                sensordata=sensordata,
            )
            state = np.ascontiguousarray(state_out[:, -1, :])

        times: list[float] = []
        begin_total = time.perf_counter()
        for _ in range(args.steps):
            control = _make_control(args.num_envs, args.nstep, ncontrol, args.actions, rng)
            begin = time.perf_counter()
            state_out, sensordata = rollout.rollout(
                model,
                datas,
                state,
                control=control,
                nstep=args.nstep,
                state=state_out,
                sensordata=sensordata,
            )
            state = np.ascontiguousarray(state_out[:, -1, :])
            times.append(time.perf_counter() - begin)
        elapsed = time.perf_counter() - begin_total
        return _timing_metrics(elapsed, times)
    finally:
        rollout.close()


def _initial_state(model: mujoco.MjModel, num_envs: int, nstate: int) -> np.ndarray:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    state = np.empty((nstate,), dtype=np.float64)
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return np.repeat(state.reshape(1, -1), num_envs, axis=0).copy()


def _make_control(num_envs: int, nstep: int, ncontrol: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    if ncontrol == 0:
        return np.zeros((num_envs, nstep, 0), dtype=np.float64)
    if mode == "zero":
        return np.zeros((num_envs, nstep, ncontrol), dtype=np.float64)
    return rng.uniform(-1.0, 1.0, size=(num_envs, nstep, ncontrol)).astype(np.float64)


def _timing_metrics(elapsed: float, times: list[float]) -> dict[str, float]:
    return {
        "elapsed_seconds": float(elapsed),
        "mean_call_ms": float(statistics.fmean(times) * 1000.0),
        "min_call_ms": float(min(times) * 1000.0),
        "max_call_ms": float(max(times) * 1000.0),
        "p50_call_ms": _percentile_ms(times, 0.50),
        "p95_call_ms": _percentile_ms(times, 0.95),
    }


def _timing_mean(records: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(np.mean(np.asarray(values, dtype=np.float64))) for key, values in records.items() if values}


def _timing_median(records: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(np.median(np.asarray(values, dtype=np.float64))) for key, values in records.items() if values}


def _percentile_ms(values: list[float], fraction: float) -> float:
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * fraction)))
    return float(sorted_values[index] * 1000.0)


if __name__ == "__main__":
    main()
