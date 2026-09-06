"""Profile actual conveyor physics ticks, including the transition into hand contact."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "apply_forces", "state_sync", "sdk_pre_step", "sdk_islands", "sdk_post_step",
    "island_prepare", "island_newton", "island_queries", "collision_detection",
    "contact_jacobians", "assembly", "linear_setup", "linear_solve", "line_search",
)
TIMING_SEMANTICS = {
    "units": "seconds",
    "percentile": "nearest-rank p95; median p50",
    "solve_time_seconds": "SDK PreStepEcs + StepEcs + PostStepEcs, not just linear solve",
    "wall_stages": ["apply_forces", "state_sync", "sdk_pre_step", "sdk_islands", "sdk_post_step"],
    "parallel_sum": "Inclusive elapsed scopes summed over islands/tasks; may exceed SDK wall time",
    "overlap": [
        "island_newton includes assembly, linear setup/solve and line search",
        "line_search includes trial-state updates and residual/energy assemblies",
        "assembly includes collision detection and contact Jacobians; body assembly can run concurrently",
        "collision_detection includes pipeline synchronization, not exclusively collision CPU work",
        "contact_jacobians includes waiting for articulated Jacobian tasks",
        "linear_solve includes preconditioner construction and iterations",
    ],
    "window": "100 warmup then 500 consecutive physics ticks starting at observed contact; configurable",
    "contact": "Qualifying target hand contact, not hand-to-floor or authored phase labels",
    "penetration": "Backend-reported contact distance, not an independent geometry penetration test",
}


def distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "p50": median(ordered),
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "max": ordered[-1],
        "sum": sum(ordered),
    }


def failure_record(failure: dict[str, Any]) -> dict[str, Any]:
    # Non-finite residuals are evidence of failure, not valid timing samples or zeros.
    def safe(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {k: safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        return value
    return safe(failure)


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"ticks": len(samples)}
    if not samples:
        return result
    result.update(
        first_tick=samples[0]["tick"], last_tick=samples[-1]["tick"],
        phases=dict(Counter(row["phase"] for row in samples)),
        convergence=dict(Counter(row["solver"]["convergence"] for row in samples)),
        qualifying_contact_ticks=sum(row["qualifying_contact"] for row in samples),
    )
    result["contact_fraction"] = result["qualifying_contact_ticks"] / len(samples)
    for key in ("step_once_seconds", "observed_tick_seconds", "command_seconds", "state_view_seconds",
                "max_penetration_meters", "directed_contact_count"):
        result[key] = distribution([row[key] for row in samples])
    for key in ("total_step_time_seconds", "solve_time_seconds", "newton_iterations",
                "line_search_iterations", "linear_iterations", "residual_norm"):
        result[key] = distribution([row["solver"][key] for row in samples])
    result["stage_timings"] = {}
    tables = [{stage["name"]: stage for stage in row["solver"]["stage_timings"]} for row in samples]
    for name in STAGES:
        entries = [table[name] for table in tables]
        result["stage_timings"][name] = {
            "seconds": distribution([stage["time_seconds"] for stage in entries]),
            "calls": distribution([stage["calls"] for stage in entries]),
            "parallel_sum": entries[0]["parallel_sum"],
        }
    return result


class ContactWindow:
    """Keep one consecutive window after real contact, including any later contact loss."""

    def __init__(self, target: str, required_hands: int, warmup_ticks: int, measure_ticks: int):
        if required_hands not in (1, 2) or warmup_ticks < 0 or measure_ticks <= 0:
            raise ValueError("invalid contact window parameters")
        self.target = target
        self.required_hands = required_hands
        self.warmup_ticks = warmup_ticks
        self.measure_ticks = measure_ticks
        self.contact_tick: int | None = None
        self.samples: list[dict[str, Any]] = []
        self.complete = False
        self._simulation_time = 0.0

    def add(self, sample: dict[str, Any]) -> bool:
        if self.complete:
            raise ValueError("contact window already complete")
        if sample["tick"] != len(self.samples) + 1:
            raise ValueError("physics ticks skipped, duplicated or reset")
        simulation_time = sample["simulation_time_seconds"]
        if not math.isfinite(simulation_time) or not math.isclose(
            simulation_time - self._simulation_time, 0.002, rel_tol=1.0e-3, abs_tol=1.0e-7
        ):
            raise ValueError("physics time did not advance by the unchanged 2 ms fixed step")
        solver = sample["solver"]
        if not solver.get("timings_available"):
            raise ValueError("SDK stage profiling is unavailable; rebuild the SDK and Gobot")
        stages = solver["stage_timings"]
        if len(stages) != len(STAGES) or {s["name"] for s in stages} != set(STAGES):
            raise ValueError("SDK stage profiling has missing or duplicate stages")
        values = [solver[k] for k in ("total_step_time_seconds", "solve_time_seconds", "residual_norm",
                                      "newton_iterations", "line_search_iterations", "linear_iterations")]
        values.extend(s[k] for s in stages for k in ("time_seconds", "calls"))
        values.extend(sample[k] for k in ("step_once_seconds", "command_seconds", "state_view_seconds",
                                          "observed_tick_seconds", "max_penetration_meters"))
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("non-finite or negative profiling sample")
        if solver["convergence"] == "diverged":
            raise ValueError("solver diverged")
        sample["qualifying_contact"] = (
            len(set(sample["contacting_hands"][self.target])) >= self.required_hands
        )
        if self.contact_tick is None and sample["qualifying_contact"]:
            self.contact_tick = sample["tick"]
        category = "precontact"
        if self.contact_tick is not None:
            offset = sample["tick"] - self.contact_tick
            category = "contact_warmup" if offset < self.warmup_ticks else "measurement"
            self.complete = offset + 1 >= self.warmup_ticks + self.measure_ticks
        sample["window"] = category
        self.samples.append(sample)
        self._simulation_time = simulation_time
        return not self.complete

    def summary(self) -> dict[str, Any]:
        measured = [row for row in self.samples if row["window"] == "measurement"]
        return {
            "window_complete": self.complete,
            "measurement_contains_contact": any(row["qualifying_contact"] for row in measured),
            "contact_trigger_tick": self.contact_tick,
            "target": self.target,
            "required_hands": self.required_hands,
            "warmup_ticks_requested": self.warmup_ticks,
            "measurement_ticks_requested": self.measure_ticks,
            "scene_complexity": self.samples[0].get("scene_complexity") if self.samples else None,
            "all_ticks": summarize_samples(self.samples),
            **{name: summarize_samples([row for row in self.samples if row["window"] == name])
               for name in ("precontact", "contact_warmup", "measurement")},
            "measured_contact_only": summarize_samples([row for row in measured if row["qualifying_contact"]]),
            "measured_contact_lost": summarize_samples([row for row in measured if not row["qualifying_contact"]]),
            "by_phase": {phase: summarize_samples([row for row in self.samples if row["phase"] == phase])
                         for phase in dict.fromkeys(row["phase"] for row in self.samples)},
        }


def _command(args: list[str], cwd: Path = ROOT) -> str | None:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _revision(path: Path) -> dict[str, Any]:
    diff = _command(["git", "diff", "HEAD", "--"], path)
    return {
        "head": _command(["git", "rev-parse", "HEAD"], path),
        "status": _command(["git", "status", "--short", "--untracked-files=no"], path),
        "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=ROOT / "examples/conveyor_packages/conveyor_packages.jscn")
    parser.add_argument("--quality", choices=("interactive", "accurate"), default="interactive")
    parser.add_argument("--target", choices=("blue_mailer", "yellow_pouch", "carton_small"), default="blue_mailer")
    parser.add_argument("--required-hands", type=int, choices=(1, 2), default=2)
    parser.add_argument("--warmup-ticks", type=int, default=100)
    parser.add_argument("--measure-ticks", type=int, default=500)
    parser.add_argument("--max-ticks", type=int, default=1800)
    parser.add_argument("--max-wall-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    window = ContactWindow(args.target, args.required_hands, args.warmup_ticks, args.measure_ticks)
    if args.max_ticks <= 0 or not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0:
        parser.error("tick and wall-time limits must be positive and finite")
    # This is the normal example runner, including its native controllers and force models.
    sys.path.insert(0, str(ROOT / "examples/conveyor_packages"))
    import conveyor_packages_batch as conveyor

    run_args = conveyor._parser().parse_args([
        "--scene", str(args.scene), "--quality", args.quality, "--steps", str(args.max_ticks),
        "--warmup-steps", "0", "--solver-timings",
    ])
    args.output.mkdir(parents=True, exist_ok=True)
    trace_path = args.output / "ticks.jsonl"
    summary_path = args.output / "summary.json"
    if trace_path.exists() or summary_path.exists():
        parser.error("output already contains a capture; choose a new directory")
    source_paths = [Path(__file__), args.scene, *list((ROOT / "examples/conveyor_packages").glob("*.py"))]
    sdk = ROOT / "3rdparty/project_superdex"
    source_paths.extend(sdk / name for name in (
        "superdex_physics/libraries/mochi/mochi_physics/include/mochi_physics/utils/step_profiling.h",
        "superdex_physics/libraries/mochi/mochi_physics/src/mochi_step_profiling.h",
    ))
    metadata = {
        "schema": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "cpu": _command(["lscpu"]),
        "load_average_at_start": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "gpu": _command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "python": sys.version, "gobot_extension": conveyor.gobot._core.__file__,
        "gobot": _revision(ROOT), "superdex": _revision(sdk),
        "source_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths if p.is_file()},
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "fixed_dt": conveyor.FIXED_DT, "substeps": 1, "execution": "cpu",
        "linear_solver": "auto", "linear_iterations_limit": -1,
        "sdk_worker_threads": 0, "sdk_threading": "single_threaded",
        "record_deformable_contact_forces": True,
        "wall_limit_policy": "checked between physics ticks; cannot interrupt an in-flight SDK call",
        "quality": vars(conveyor.quality_profile(args.quality)),
        "timing_semantics": TIMING_SEMANTICS,
    }
    started = time.perf_counter()
    last_progress = started
    error = None
    timed_out = False
    run_result = None
    failed_step = None
    with trace_path.open("x", encoding="utf-8", buffering=1) as trace:
        trace.write(json.dumps({"metadata": metadata}, allow_nan=False) + "\n")

        def observe(sample: dict[str, Any]) -> bool:
            nonlocal last_progress, timed_out
            keep_running = window.add(sample)
            elapsed = time.perf_counter() - started
            sample["capture_elapsed_seconds"] = elapsed
            trace.write(json.dumps(sample, allow_nan=False) + "\n")
            if time.perf_counter() - last_progress >= 10 or not keep_running:
                print(f"tick={sample['tick']} phase={sample['phase']} window={sample['window']} "
                      f"contact={sample['qualifying_contact']} step_ms={sample['step_once_seconds'] * 1000:.2f} "
                      f"elapsed_s={elapsed:.1f}", flush=True)
                last_progress = time.perf_counter()
            timed_out = keep_running and elapsed >= args.max_wall_seconds
            return keep_running and not timed_out

        def observe_failure(failure: dict[str, Any]) -> None:
            nonlocal failed_step
            failed_step = failure_record(failure)
            trace.write(json.dumps({"failed_step": failed_step}, allow_nan=False) + "\n")

        try:
            run_result = conveyor.run(run_args, tick_observer=observe, failure_observer=observe_failure)
        except (Exception, KeyboardInterrupt) as exc:
            error = f"{type(exc).__name__}: {exc}"
    summary = window.summary()
    summary.update(metadata=metadata, error=error, failed_step=failed_step, timed_out=timed_out, example_result=run_result,
                   elapsed_seconds=time.perf_counter() - started)
    summary["capture_succeeded"] = (
        window.complete and summary["measurement_contains_contact"] and error is None and not timed_out
    )
    summary["incomplete_reason"] = (
        error or ("wall-time limit reached" if timed_out else
                  "target hand contact not reached" if window.contact_tick is None else
                  "no target hand contact during measurement" if window.complete else
                  "tick limit reached before the measurement window completed")
    ) if not summary["capture_succeeded"] else None
    with summary_path.open("x", encoding="utf-8") as out:
        json.dump(summary, out, indent=2, allow_nan=False)
        out.write("\n")
    print(f"Summary: {summary_path}; complete={summary['capture_succeeded']}; error={error}", flush=True)
    return 0 if summary["capture_succeeded"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
