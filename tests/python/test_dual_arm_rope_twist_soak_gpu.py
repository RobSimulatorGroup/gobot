from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile


if os.environ.get("GOBOT_RUN_DUAL_ARM_ROPE_SOAK_GPU_TEST") != "1":
    raise SystemExit(77)


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "dual_arm_rope_twist"
MODULE_PATH = os.environ.get("GOBOT_LIBUIPC_TEST_MODULE_PATH", "")
sys.path.insert(0, str(EXAMPLE))

import rope_twist_batch as batch  # noqa: E402


SOAK_STEPS = 40_000


def test_interactive_single_environment_40000_step_soak() -> None:
    with tempfile.TemporaryDirectory(
        prefix="gobot-rope-twist-soak-gpu-"
    ) as temporary:
        args = batch._parser().parse_args(
            [
                "--num-envs",
                "1",
                "--environments-per-shard",
                "1",
                "--steps",
                str(SOAK_STEPS),
                "--continue-after-cycle-complete",
                "--coupling-iterations",
                "1",
                "--relaxation-mode",
                "fixed",
                "--defer-deformable-contact-forces",
                "--maximum-step-latency-seconds",
                "5",
                "--module-path",
                MODULE_PATH,
                "--workspace",
                temporary,
            ]
        )
        result = batch.run(args)

    summary = {
        name: result[name]
        for name in (
            "steps",
            "requested_steps",
            "admission_completed",
            "admission_abort_reason",
            "maximum_step_latency_seconds",
            "interface_residual",
            "maximum_grip_slip_range_meters",
            "maximum_attachment_error_range_meters",
            "maximum_rope_vertex_penetration_range_meters",
            "maximum_coupling_wrench_imbalance_ratio_range",
        )
    }
    print(json.dumps(summary, sort_keys=True))

    assert result["steps"] == SOAK_STEPS
    assert result["requested_steps"] == SOAK_STEPS
    assert result["admission_completed"]
    assert not result["admission_aborted"]
    assert result["coupler_graph_captured"]
    assert result["exact_contact_wrench"]
    assert not result["deformable_contact_forces_exported"]
    assert result["ipc_position_storage_stable"]
    assert result["qpos_storage_stable"]
    assert result["interface_residual"] <= 6.0e-3
    assert result["maximum_grip_slip_range_meters"][1] <= 1.0e-4
    assert result["maximum_attachment_error_range_meters"][1] <= 1.0e-4
    assert (
        result["maximum_rope_vertex_penetration_range_meters"][1]
        <= 1.0e-3
    )
    assert (
        result["maximum_coupling_wrench_imbalance_ratio_range"][1]
        < 0.01
    )


def main() -> int:
    test_interactive_single_environment_40000_step_soak()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
