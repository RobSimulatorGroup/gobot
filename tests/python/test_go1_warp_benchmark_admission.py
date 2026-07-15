from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.check_go1_warp_admission import validate_results  # noqa: E402


def _result(env_count: int, throughput: float, *, graph_capture: bool = True) -> dict:
    return {
        "num_envs": env_count,
        "sim_backend": "gobot_mujoco_warp",
        "cuda_synchronized_interval": True,
        "throughput_env_steps_per_s": throughput,
        "memory_profile": {"graph_capture": graph_capture},
    }


def main() -> int:
    results = [_result(256, 10_000.0), _result(1024, 22_000.0), _result(2048, 30_000.0)]
    assert validate_results(
        results,
        required_envs=(256, 1024, 2048),
        minimum_throughput={256: 8_000.0, 1024: 16_000.0, 2048: 25_000.0},
        minimum_scale_factor=1.05,
    ) == []

    no_graph = [_result(256, 10_000.0, graph_capture=False)]
    assert any(
        "graph capture" in error
        for error in validate_results(
            no_graph,
            required_envs=(256,),
            minimum_throughput={},
            minimum_scale_factor=1.0,
        )
    )

    slow = [_result(256, 10_000.0), _result(1024, 10_100.0)]
    errors = validate_results(
        slow,
        required_envs=(256, 1024, 2048),
        minimum_throughput={1024: 20_000.0},
        minimum_scale_factor=1.05,
    )
    assert any("below 20000.0" in error for error in errors)
    assert any("256->1024" in error for error in errors)
    assert any("missing benchmark result for 2048" in error for error in errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
