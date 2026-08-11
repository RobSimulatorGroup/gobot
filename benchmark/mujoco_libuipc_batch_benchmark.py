"""Benchmark the explicit MuJoCo Warp + libuipc coupling at fixed capacities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = ROOT / "examples" / "mujoco_libuipc"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

import mujoco_libuipc_batch as batch_example  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment-counts", type=int, nargs="+", default=(4, 64, 256)
    )
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fixed-dt", type=float, default=0.002)
    parser.add_argument("--rigid-substeps", type=int, default=1)
    parser.add_argument("--ipc-substeps", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--module-path", default="")
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--maximum-regression", type=float, default=0.10)
    return parser


def _baseline_throughput(path: Path, environment_count: int) -> float:
    values = json.loads(path.read_text(encoding="utf-8"))
    for result in values.get("results", ()):
        if int(result.get("environments", -1)) == environment_count:
            return float(result["environment_steps_per_second"])
    raise ValueError(
        f"baseline {path} has no result for {environment_count} environments"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    counts = tuple(int(value) for value in args.environment_counts)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("environment counts must be positive")
    if not 0.0 <= args.maximum_regression < 1.0:
        raise ValueError("--maximum-regression must be in [0, 1)")

    results = []
    for environment_count in counts:
        example_args = batch_example._parser().parse_args([])
        example_args.num_envs = environment_count
        example_args.environments_per_shard = min(64, environment_count)
        example_args.steps = args.steps
        example_args.warmup_steps = args.warmup_steps
        example_args.repeats = args.repeats
        example_args.fixed_dt = args.fixed_dt
        example_args.rigid_substeps = args.rigid_substeps
        example_args.ipc_substeps = args.ipc_substeps
        example_args.device = args.device
        example_args.module_path = args.module_path
        results.append(batch_example.run(example_args))

    output: dict[str, Any] = {
        "benchmark_schema_version": 2,
        "results": results,
    }
    if args.baseline_json is not None:
        current_256 = next(
            (
                float(result["environment_steps_per_second"])
                for result in results
                if int(result["environments"]) == 256
            ),
            None,
        )
        if current_256 is None:
            raise ValueError("baseline admission requires a 256-environment run")
        baseline_256 = _baseline_throughput(args.baseline_json, 256)
        ratio = current_256 / baseline_256
        output["baseline_256_environment_steps_per_second"] = baseline_256
        output["throughput_ratio_256"] = ratio
        output["maximum_regression"] = args.maximum_regression
        if ratio < 1.0 - args.maximum_regression:
            raise RuntimeError(
                "256-environment throughput regressed by more than "
                f"{args.maximum_regression:.0%}: ratio={ratio:.4f}"
            )
    return output


def main() -> None:
    print(json.dumps(run(_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
