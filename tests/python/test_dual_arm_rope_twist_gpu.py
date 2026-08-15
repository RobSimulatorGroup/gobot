from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile


if os.environ.get("GOBOT_RUN_LIBUIPC_BATCH_GPU_TEST") != "1":
    raise SystemExit(77)

from gobot.ipc import LibuipcBatchConfig, LibuipcBatchSolver, LibuipcConfig


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "dual_arm_rope_twist"
MODULE_PATH = os.environ.get("GOBOT_LIBUIPC_TEST_MODULE_PATH", "")
sys.path.insert(0, str(EXAMPLE))
import controllers
import rope_twist_batch as batch


def _arguments(
    workspace: Path,
    *,
    drive_mode: str = controllers.FINITE_TORQUE_DRIVE_MODE,
    steps: int = batch.CYCLE_TICKS,
):
    return batch._parser().parse_args(
        [
            "--num-envs",
            "1",
            "--environments-per-shard",
            "1",
            "--steps",
            str(steps),
            "--drive-mode",
            drive_mode,
            "--module-path",
            MODULE_PATH,
            "--workspace",
            str(workspace),
            "--no-mujoco-graph",
        ]
    )


def test_dual_arm_twist_stalls_without_losing_the_friction_grasps() -> None:
    with tempfile.TemporaryDirectory(prefix="gobot-rope-twist-gpu-") as temporary:
        result = batch.run(_arguments(Path(temporary)))

    trial = result["trials"][0]
    print(
        json.dumps(
            {
                "steps": result["steps"],
                "requested_steps": result["requested_steps"],
                "trial": trial,
            },
            sort_keys=True,
        )
    )
    assert result["task"] == "dual_fr3_three_strand_rope_twist"
    assert result["robot_count"] == 2
    assert result["deformable_strand_count"] == 3
    assert result["attachment_count"] == 6
    assert result["fixture_coupling_count"] == 2
    assert result["grip_mode"] == "mujoco_fixture_friction_contact"
    assert result["integration_scheme"] == "newton_proxy"
    assert result["coupling_iterations"] == 2
    assert result["actual_coupling_iterations"] == 2
    assert result["relaxation_mode"] == "aitken"
    assert result["exact_contact_wrench"]
    assert result["feedback_source"] == "native_contact_wrench"
    assert result["ipc_position_storage_stable"]
    assert result["qpos_storage_stable"]
    assert result["requested_steps"] == batch.CYCLE_TICKS
    assert result["steps"] < result["requested_steps"]
    assert result["stalled_count"] == 1
    assert result["safety_stopped_count"] == 0
    assert result["peak_contact_force_range_newtons"][0] > 0.05
    assert result["maximum_attachment_error_range_meters"][1] < 0.001
    assert result["maximum_shape_deformation_range_meters"][0] > 0.02

    assert trial["stalled"]
    assert not trial["safety_stopped"]
    assert trial["stall_tick"] > (
        batch.TWIST_START_TICK + controllers.STALL_DETECTION_DELAY_TICKS
    )
    assert 15.0 <= trial["stalled_relative_turns"] <= 25.0
    assert trial["peak_actual_relative_turns"] >= trial["stalled_relative_turns"]
    assert trial["raw_peak_axial_torque_newton_meters"] > 0.112
    assert min(
        abs(value)
        for value in trial["stall_wrist_efforts_newton_meters"]
    ) >= 0.112
    assert min(
        abs(value)
        for value in trial["stall_axial_torques_newton_meters"]
    ) >= 0.093
    assert max(
        abs(value)
        for value in trial["stall_wrist_speeds_radians_per_second"]
    ) < 0.10
    assert trial["grip_contact_seen"]
    assert trial["grip_preload_minimum_finger_force_newtons"] > 0.3
    assert trial["maximum_grip_slip_meters"] < 0.010
    assert trial["maximum_grip_rotation_slip_radians"] < 0.05
    assert trial["maximum_attachment_error_meters"] < 0.001
    assert (
        sum(abs(value) for value in trial["strand_winding_turns"]) / 3.0
        > 14.0
    )
    assert abs(
        trial["joint7_positions_radians"][0]
        - trial["joint7_positions_radians"][1]
    ) < 0.75
    assert math.isclose(
        result["wrist_drive_torque_limit_newton_meters"],
        0.125,
        abs_tol=1.0e-8,
    )


def test_showcase_drive_tracks_requested_speed_under_rope_load() -> None:
    steps = 1200
    with tempfile.TemporaryDirectory(
        prefix="gobot-rope-twist-showcase-gpu-"
    ) as temporary:
        result = batch.run(
            _arguments(
                Path(temporary),
                drive_mode=controllers.SHOWCASE_DRIVE_MODE,
                steps=steps,
            )
        )

    trial = result["trials"][0]
    assert result["drive_mode"] == controllers.SHOWCASE_DRIVE_MODE
    assert result["integration_scheme"] == "newton_proxy"
    assert result["actual_coupling_iterations"] == 2
    assert result["steps"] == steps
    assert result["stalled_count"] == 0
    assert math.isclose(
        result["wrist_drive_torque_limit_newton_meters"],
        controllers.WRIST_SHOWCASE_TORQUE_LIMIT,
        abs_tol=1.0e-8,
    )
    assert max(
        abs(abs(value) - controllers.WRIST_TARGET_SPEED)
        for value in trial["joint7_velocities_radians_per_second"]
    ) < 0.01
    assert trial["maximum_grip_slip_meters"] < 0.001
    assert trial["maximum_grip_rotation_slip_radians"] < 0.01
    assert trial["maximum_attachment_error_meters"] < 0.001


def main() -> int:
    availability = LibuipcBatchSolver.availability(
        LibuipcBatchConfig(
            solver=LibuipcConfig(module_path=MODULE_PATH),
            environments_per_shard=1,
        )
    )
    assert availability.available, availability.reason
    test_dual_arm_twist_stalls_without_losing_the_friction_grasps()
    test_showcase_drive_tracks_requested_speed_under_rope_load()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
