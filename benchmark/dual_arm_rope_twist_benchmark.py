"""Benchmark dual-arm rope quality profiles and coupler CUDA Graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "dual_arm_rope_twist"
if str(EXAMPLE) not in sys.path:
    sys.path.insert(0, str(EXAMPLE))

import rope_twist_batch as batch  # noqa: E402


PROFILES = {
    "interactive": {
        "coupling_iterations": 1,
        "relaxation_mode": "fixed",
        "newton_max_iterations": 16,
        "line_search_max_iterations": 8,
        "linear_system_tolerance_rate": 1.0e-3,
        "defer_deformable_contact_forces": True,
    },
    "accurate": {
        "coupling_iterations": 2,
        "relaxation_mode": "aitken",
        "newton_max_iterations": 16,
        "line_search_max_iterations": 8,
        "linear_system_tolerance_rate": 1.0e-3,
        "defer_deformable_contact_forces": False,
    },
    "interactive-balanced": {
        "coupling_iterations": 1,
        "relaxation_mode": "fixed",
        "newton_max_iterations": 12,
        "line_search_max_iterations": 6,
        "linear_system_tolerance_rate": 2.0e-3,
        "defer_deformable_contact_forces": True,
    },
    "interactive-fast": {
        "coupling_iterations": 1,
        "relaxation_mode": "fixed",
        "newton_max_iterations": 8,
        "line_search_max_iterations": 4,
        "linear_system_tolerance_rate": 5.0e-3,
        "defer_deformable_contact_forces": True,
    },
}
DEFAULT_PROFILES = ("interactive", "accurate")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-counts", type=int, nargs="+", default=(1, 64))
    parser.add_argument(
        "--environments-per-shard",
        type=int,
        default=0,
        help="fixed shard capacity; 0 uses min(64, environment count)",
    )
    parser.add_argument("--single-environment-steps", type=int, default=500)
    parser.add_argument("--batch-steps", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--maximum-step-latency-seconds", type=float, default=5.0
    )
    parser.add_argument(
        "--profiles",
        choices=tuple(PROFILES),
        nargs="+",
        default=DEFAULT_PROFILES,
    )
    parser.add_argument(
        "--graph-modes",
        choices=("captured", "eager"),
        nargs="+",
        default=("captured", "eager"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--module-path", default="")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.environment_counts or any(
        count <= 0 for count in args.environment_counts
    ):
        raise ValueError("environment counts must be positive")
    if args.environments_per_shard < 0:
        raise ValueError("environments per shard must be non-negative")
    if args.environments_per_shard and any(
        count % args.environments_per_shard
        for count in args.environment_counts
    ):
        raise ValueError(
            "each environment count must be divisible by environments per shard"
        )
    if args.single_environment_steps <= 0 or args.batch_steps <= 0:
        raise ValueError("benchmark step counts must be positive")
    if args.warmup_steps < 0:
        raise ValueError("warmup steps must be non-negative")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.maximum_step_latency_seconds < 0.0:
        raise ValueError("maximum step latency must be non-negative")


def _summarize(
    profile: str,
    graph_mode: str,
    environment_count: int,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    latencies = [
        1000.0 * float(run["elapsed_seconds"]) / int(run["steps"])
        for run in runs
    ]
    throughputs = [float(run["environment_steps_per_second"]) for run in runs]
    completed = all(bool(run["admission_completed"]) for run in runs)
    return {
        "profile": profile,
        "coupler_graph": graph_mode,
        "environments": environment_count,
        "environments_per_shard": int(runs[0]["environments_per_shard"]),
        "requested_steps": int(runs[0]["requested_steps"]),
        "completed_steps": [int(run["steps"]) for run in runs],
        "warmup_steps": int(runs[0]["warmup_steps"]),
        "repeats": len(runs),
        "step_latency_ms_runs": latencies,
        "interface_residual_runs": [
            float(run["interface_residual"]) for run in runs
        ],
        "maximum_grip_slip_meters_runs": [
            float(run["maximum_grip_slip_range_meters"][1])
            for run in runs
        ],
        "admission_completed": completed,
        "admission_abort_reasons": [
            str(run["admission_abort_reason"])
            for run in runs
            if bool(run["admission_aborted"])
        ],
        "maximum_step_latency_seconds": max(
            float(run["maximum_step_latency_seconds"]) for run in runs
        ),
        "median_step_latency_ms": median(latencies),
        "median_environment_steps_per_second": median(throughputs),
        "interface_residual_max": max(
            float(run["interface_residual"]) for run in runs
        ),
        "maximum_attachment_error_meters": max(
            float(run["maximum_attachment_error_range_meters"][1])
            for run in runs
        ),
        "maximum_grip_slip_meters": max(
            float(run["maximum_grip_slip_range_meters"][1]) for run in runs
        ),
        "maximum_rope_vertex_penetration_meters": max(
            float(run["maximum_rope_vertex_penetration_range_meters"][1])
            for run in runs
        ),
        "maximum_coupling_wrench_imbalance_ratio": max(
            float(
                run[
                    "maximum_coupling_wrench_imbalance_ratio_range"
                ][1]
            )
            for run in runs
        ),
        "peak_wrench_newton_meters": max(
            float(run["raw_peak_axial_torque_range_newton_meters"][1])
            for run in runs
        ),
        "exact_contact_wrench": all(
            bool(run["exact_contact_wrench"]) for run in runs
        ),
        "deformable_contact_forces_exported": all(
            bool(run["deformable_contact_forces_exported"])
            for run in runs
        ),
        "coupler_graph_captured": all(
            bool(run["coupler_graph_captured"]) for run in runs
        ),
        "phase_latency_ms": dict(runs[-1]["phase_latency_ms"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    summaries = []
    with tempfile.TemporaryDirectory(
        prefix="gobot-dual-arm-benchmark-"
    ) as temporary:
        workspace_root = Path(temporary)
        for environment_count in args.environment_counts:
            steps = (
                args.single_environment_steps
                if environment_count == 1
                else args.batch_steps
            )
            for profile_name in args.profiles:
                profile = PROFILES[profile_name]
                for graph_mode in args.graph_modes:
                    runs = []
                    for repeat in range(args.repeats):
                        example_args = batch._parser().parse_args([])
                        example_args.num_envs = environment_count
                        example_args.environments_per_shard = (
                            args.environments_per_shard
                            or min(64, environment_count)
                        )
                        example_args.steps = steps
                        example_args.warmup_steps = args.warmup_steps
                        example_args.maximum_step_latency_seconds = (
                            args.maximum_step_latency_seconds
                        )
                        example_args.device = args.device
                        example_args.module_path = args.module_path
                        example_args.drive_mode = batch.SHOWCASE_DRIVE_MODE
                        example_args.no_coupler_graph = graph_mode == "eager"
                        for name, value in profile.items():
                            setattr(example_args, name, value)
                        example_args.workspace = (
                            workspace_root
                            / (
                                f"{profile_name}_{graph_mode}_"
                                f"{environment_count}_{repeat}"
                            )
                        )
                        runs.append(batch.run(example_args))
                    summaries.append(
                        _summarize(
                            profile_name,
                            graph_mode,
                            environment_count,
                            runs,
                        )
                    )

    indexed = {
        (
            result["profile"],
            result["coupler_graph"],
            result["environments"],
        ): result
        for result in summaries
    }
    comparisons = []
    for environment_count in args.environment_counts:
        comparison: dict[str, Any] = {"environments": environment_count}
        interactive = indexed.get(
            ("interactive", "captured", environment_count)
        )
        accurate = indexed.get(("accurate", "captured", environment_count))
        if (
            interactive is not None
            and accurate is not None
            and bool(interactive["admission_completed"])
            and bool(accurate["admission_completed"])
        ):
            speedup = (
                float(accurate["median_step_latency_ms"])
                / float(interactive["median_step_latency_ms"])
            )
            comparison["interactive_speedup_over_accurate"] = speedup
            comparison["interactive_speedup_passes_1_6x"] = speedup >= 1.6
            reference_wrench = float(accurate["peak_wrench_newton_meters"])
            wrench_error = abs(
                float(interactive["peak_wrench_newton_meters"])
                - reference_wrench
            ) / max(reference_wrench, 1.0e-12)
            interactive_gates = {
                "interface_residual": (
                    float(interactive["interface_residual_max"]) <= 6.0e-3
                ),
                "attachment_error": (
                    float(interactive["maximum_attachment_error_meters"])
                    <= 1.0e-4
                ),
                "grip_slip": (
                    float(interactive["maximum_grip_slip_meters"])
                    <= 1.0e-4
                ),
                "peak_wrench": wrench_error <= 0.10,
                "penetration_absolute": (
                    float(
                        interactive[
                            "maximum_rope_vertex_penetration_meters"
                        ]
                    )
                    <= 1.0e-3
                ),
                "penetration_relative": (
                    float(
                        interactive[
                            "maximum_rope_vertex_penetration_meters"
                        ]
                    )
                    <= float(
                        accurate[
                            "maximum_rope_vertex_penetration_meters"
                        ]
                    )
                    + 5.0e-4
                ),
                "exact_contact_wrench": bool(
                    interactive["exact_contact_wrench"]
                ),
            }
            accurate_gates = {
                "interface_residual": (
                    float(accurate["interface_residual_max"]) <= 1.0e-5
                ),
                "coupling_wrench_imbalance": (
                    float(
                        accurate[
                            "maximum_coupling_wrench_imbalance_ratio"
                        ]
                    )
                    < 0.01
                ),
                "exact_contact_wrench": bool(
                    accurate["exact_contact_wrench"]
                ),
            }
            comparison["quality_admission"] = {
                "interactive": {
                    "passes": all(interactive_gates.values()),
                    "gates": interactive_gates,
                    "peak_wrench_relative_error": wrench_error,
                },
                "accurate": {
                    "passes": all(accurate_gates.values()),
                    "gates": accurate_gates,
                },
            }
        if accurate is not None and bool(accurate["admission_completed"]):
            reference_wrench = float(accurate["peak_wrench_newton_meters"])
            candidate_evaluations = []
            eligible_candidates = []
            for candidate_name in (
                "interactive-balanced",
                "interactive-fast",
            ):
                candidate = indexed.get(
                    (candidate_name, "captured", environment_count)
                )
                if candidate is None:
                    continue
                if not bool(candidate["admission_completed"]):
                    candidate_evaluations.append(
                        {
                            "profile": candidate_name,
                            "passes": False,
                            "gates": {"admission_completed": False},
                            "peak_wrench_relative_error": None,
                        }
                    )
                    continue
                wrench_error = abs(
                    float(candidate["peak_wrench_newton_meters"])
                    - reference_wrench
                ) / max(reference_wrench, 1.0e-12)
                gates = {
                    "interface_residual": (
                        float(candidate["interface_residual_max"])
                        <= 6.0e-3
                    ),
                    "attachment_error": (
                        float(candidate["maximum_attachment_error_meters"])
                        <= 1.0e-4
                    ),
                    "grip_slip": (
                        float(candidate["maximum_grip_slip_meters"])
                        <= 1.0e-4
                    ),
                    "peak_wrench": wrench_error <= 0.10,
                    "penetration_absolute": (
                        float(
                            candidate[
                                "maximum_rope_vertex_penetration_meters"
                            ]
                        )
                        <= 1.0e-3
                    ),
                    "penetration_relative": (
                        float(
                            candidate[
                                "maximum_rope_vertex_penetration_meters"
                            ]
                        )
                        <= float(
                            accurate[
                                "maximum_rope_vertex_penetration_meters"
                            ]
                        )
                        + 5.0e-4
                    ),
                    "exact_contact_wrench": bool(
                        candidate["exact_contact_wrench"]
                    ),
                }
                passes = all(gates.values())
                candidate_evaluations.append(
                    {
                        "profile": candidate_name,
                        "passes": passes,
                        "gates": gates,
                        "peak_wrench_relative_error": wrench_error,
                    }
                )
                if passes:
                    eligible_candidates.append(candidate)
            if candidate_evaluations:
                selected = (
                    min(
                        eligible_candidates,
                        key=lambda value: float(
                            value["median_step_latency_ms"]
                        ),
                    )["profile"]
                    if eligible_candidates
                    else "interactive"
                )
                comparison["interactive_solver_tuning"] = {
                    "candidates": candidate_evaluations,
                    "selected_profile": selected,
                    "used_accurate_solver_parameter_fallback": (
                        not eligible_candidates
                    ),
                }
        graph_speedups = {}
        for profile_name in args.profiles:
            captured = indexed.get(
                (profile_name, "captured", environment_count)
            )
            eager = indexed.get((profile_name, "eager", environment_count))
            if captured is not None and eager is not None:
                if not (
                    bool(captured["admission_completed"])
                    and bool(eager["admission_completed"])
                ):
                    continue
                graph_speedups[profile_name] = (
                    float(eager["median_step_latency_ms"])
                    / float(captured["median_step_latency_ms"])
                )
        if graph_speedups:
            comparison["coupler_graph_speedup"] = graph_speedups
        failed = [
            result
            for result in summaries
            if result["environments"] == environment_count
            and not bool(result["admission_completed"])
        ]
        if failed:
            comparison["admission_failures"] = [
                {
                    "profile": result["profile"],
                    "coupler_graph": result["coupler_graph"],
                    "completed_steps": result["completed_steps"],
                    "reasons": result["admission_abort_reasons"],
                }
                for result in failed
            ]
        comparisons.append(comparison)

    return {
        "benchmark_schema_version": 1,
        "scene": "dual_arm_rope_twist",
        "same_process_and_gpu": True,
        "results": summaries,
        "comparisons": comparisons,
    }


def main() -> None:
    print(json.dumps(run(_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
