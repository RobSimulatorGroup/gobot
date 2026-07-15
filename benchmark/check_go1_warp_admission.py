"""Validate a set of Go1 MuJoCo Warp benchmark results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def validate_results(
    results: Iterable[Mapping[str, Any]],
    *,
    required_envs: Sequence[int],
    minimum_throughput: Mapping[int, float],
    minimum_scale_factor: float,
) -> list[str]:
    errors: list[str] = []
    by_env_count: dict[int, Mapping[str, Any]] = {}
    for result in results:
        try:
            env_count = int(result["num_envs"])
        except (KeyError, TypeError, ValueError):
            errors.append("benchmark result has no valid num_envs")
            continue
        if env_count in by_env_count:
            errors.append(f"duplicate benchmark result for {env_count} envs")
            continue
        by_env_count[env_count] = result

    measured: dict[int, float] = {}
    for env_count in required_envs:
        result = by_env_count.get(int(env_count))
        if result is None:
            errors.append(f"missing benchmark result for {env_count} envs")
            continue
        if result.get("sim_backend") != "gobot_mujoco_warp":
            errors.append(f"{env_count} envs: sim_backend is not gobot_mujoco_warp")
        if result.get("cuda_synchronized_interval") is not True:
            errors.append(f"{env_count} envs: measured interval was not CUDA-synchronized")
        memory_profile = result.get("memory_profile")
        if not isinstance(memory_profile, Mapping) or memory_profile.get("graph_capture") is not True:
            errors.append(f"{env_count} envs: CUDA graph capture was not enabled")
        try:
            throughput = float(result["throughput_env_steps_per_s"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{env_count} envs: throughput is missing or invalid")
            continue
        if not math.isfinite(throughput) or throughput <= 0.0:
            errors.append(f"{env_count} envs: throughput must be finite and positive")
            continue
        measured[int(env_count)] = throughput
        required = float(minimum_throughput.get(int(env_count), 0.0))
        if throughput < required:
            errors.append(
                f"{env_count} envs: throughput {throughput:.1f} is below {required:.1f}"
            )

    ordered = [int(env_count) for env_count in required_envs if int(env_count) in measured]
    for previous, current in zip(ordered, ordered[1:], strict=False):
        required = measured[previous] * float(minimum_scale_factor)
        if measured[current] < required:
            errors.append(
                f"{previous}->{current} env scaling is {measured[current] / measured[previous]:.3f}x, "
                f"below {minimum_scale_factor:.3f}x"
            )
    return errors


def _parse_minimum(value: str) -> tuple[int, float]:
    try:
        env_count_text, throughput_text = value.split("=", maxsplit=1)
        env_count = int(env_count_text)
        throughput = float(throughput_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "minimum throughput must use NUM_ENVS=ENV_STEPS_PER_SECOND"
        ) from error
    if env_count <= 0 or throughput < 0.0 or not math.isfinite(throughput):
        raise argparse.ArgumentTypeError("minimum throughput values must be finite and non-negative")
    return env_count, throughput


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--required-envs", nargs="+", type=int, default=(256, 1024, 2048))
    parser.add_argument(
        "--minimum-throughput",
        action="append",
        type=_parse_minimum,
        default=[],
        metavar="NUM_ENVS=ENV_STEPS_PER_SECOND",
    )
    parser.add_argument("--minimum-scale-factor", type=float, default=1.05)
    args = parser.parse_args()
    if args.minimum_scale_factor <= 0.0 or not math.isfinite(args.minimum_scale_factor):
        parser.error("--minimum-scale-factor must be finite and positive")

    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    minimum_throughput = dict(args.minimum_throughput)
    errors = validate_results(
        results,
        required_envs=args.required_envs,
        minimum_throughput=minimum_throughput,
        minimum_scale_factor=args.minimum_scale_factor,
    )

    for result in sorted(results, key=lambda item: int(item.get("num_envs", 0))):
        print(
            f"{int(result['num_envs']):5d} envs: "
            f"{float(result['throughput_env_steps_per_s']):10.1f} env-step/s"
        )
    if errors:
        print("Go1 MuJoCo Warp benchmark admission failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("Go1 MuJoCo Warp benchmark admission passed")


if __name__ == "__main__":
    main()
