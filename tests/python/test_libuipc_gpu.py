from __future__ import annotations

import importlib.util
import os
from pathlib import Path


if os.environ.get("GOBOT_RUN_LIBUIPC_GPU_TEST") != "1":
    raise SystemExit(77)

import numpy as np

import gobot
from gobot.ipc import LibuipcConfig, LibuipcProvider


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples" / "libuipc"
MODULE_PATH = os.environ.get("GOBOT_LIBUIPC_TEST_MODULE_PATH", "")


def _load_play_script():
    spec = importlib.util.spec_from_file_location(
        "gobot_libuipc_gpu_demo", EXAMPLE_ROOT / "libuipc_demo.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _provider(scene_name: str):
    context = gobot.app.create_context()
    context.set_project_path(str(EXAMPLE_ROOT))
    context.load_scene("res://" + scene_name)
    if scene_name == "fr3_brick_grasp.jscn":
        config = LibuipcConfig(
            fixed_time_step=0.01,
            friction_coefficient=0.8,
            contact_activation_distance=5.0e-4,
            contact_resistance=1.0e8,
            module_path=MODULE_PATH,
        )
    else:
        config = LibuipcConfig(
            fixed_time_step=0.01,
            module_path=MODULE_PATH,
        )
    provider = LibuipcProvider.from_context(
        context,
        config=config,
    )
    return context, provider


def _assert_finite(provider: LibuipcProvider) -> np.ndarray:
    positions = np.asarray(provider.arrays["positions"])
    assert positions.dtype == np.float64
    assert np.isfinite(positions).all()
    assert provider.diagnostics["valid"]
    return positions


def _assert_contact_forces(
    provider: LibuipcProvider, positions: np.ndarray
) -> np.ndarray:
    contact_forces = np.asarray(provider.arrays["contact_forces"])
    assert contact_forces.dtype == np.float64
    assert contact_forces.shape == positions.shape
    assert not contact_forces.flags.writeable
    assert np.isfinite(contact_forces).all()
    return contact_forces


def test_fr3_soft_box_grasp_joint_targets() -> None:
    demo = _load_play_script()
    assert demo.FR3_CLOSED_FINGER == 0.0146
    context, provider = _provider("fr3_brick_grasp.jscn")
    held_positions = np.asarray(provider.arrays["positions"])
    try:
        assert not held_positions.flags.writeable
        assert len(provider.joints) == 9
        joint_paths = {
            str(entry["path"]).rsplit("/", 1)[-1]: str(entry["path"])
            for entry in provider.joints
        }
        body_indices = {
            str(entry["path"]).rsplit("/", 1)[-1]: index
            for index, entry in enumerate(provider.affine_bodies)
        }
        initial_positions = _assert_finite(provider).copy()
        _assert_contact_forces(provider, initial_positions)
        initial_transforms = np.array(
            provider.arrays["affine_transforms"], copy=True
        )
        arm = body_indices["fr3_link2"]
        tool = body_indices["fr3_link7"]
        left_finger = body_indices["fr3_leftfinger"]
        right_finger = body_indices["fr3_rightfinger"]

        settled_centers = []
        settled_minima = []
        grasp_relative_heights = []
        airborne_relative_heights = []
        high_hold_heights = []
        high_hold_side_loads = []
        returned_minima = []
        released_centers = []
        released_minima = []
        released_side_loads = []
        peak_transforms = None

        for step in range(1000):
            target_time = float(step) * demo.FIXED_DT
            for name, target in demo._fr3_motion_targets(target_time).items():
                provider.set_joint_target(joint_paths[name], target)
            provider.step()
            positions = _assert_finite(provider)
            contact_forces = _assert_contact_forces(provider, positions)
            transforms = np.asarray(provider.arrays["affine_transforms"])
            state_time = float(step + 1) * demo.FIXED_DT
            center = positions.mean(axis=0)
            minimum_z = float(positions[:, 2].min())
            finger_center_z = 0.5 * float(
                transforms[left_finger, 2, 3]
                + transforms[right_finger, 2, 3]
            )
            relative_height = float(center[2]) - finger_center_z

            middle_y = float(np.median(positions[:, 1]))
            lower_side = positions[:, 1] < middle_y
            upper_side = positions[:, 1] > middle_y
            lower_load = float(
                np.maximum(contact_forces[lower_side, 1], 0.0).sum()
            )
            upper_load = float(
                np.maximum(-contact_forces[upper_side, 1], 0.0).sum()
            )

            if 0.8 <= state_time <= 1.0:
                settled_centers.append(center.copy())
                settled_minima.append(minimum_z)
            if 3.4 <= state_time <= 3.6:
                grasp_relative_heights.append(relative_height)
            if 4.2 <= state_time <= 6.8:
                airborne_relative_heights.append(relative_height)
            if 5.3 <= state_time <= 6.1:
                high_hold_heights.append(float(center[2]))
                high_hold_side_loads.append((lower_load, upper_load))
            if 7.8 <= state_time <= 8.4:
                returned_minima.append(minimum_z)
            if 9.7 <= state_time <= 10.0:
                released_centers.append(center.copy())
                released_minima.append(minimum_z)
                released_side_loads.append((lower_load, upper_load))
            if 5.19 <= state_time <= 5.21:
                peak_transforms = np.array(transforms, copy=True)

        final_positions = _assert_finite(provider)
        _assert_contact_forces(provider, final_positions)
        assert peak_transforms is not None
        baseline_center = np.median(np.asarray(settled_centers), axis=0)
        baseline_minimum = float(np.median(settled_minima))
        grasp_relative_height = float(np.median(grasp_relative_heights))
        high_hold_loads = np.asarray(high_hold_side_loads)
        release_loads = np.asarray(released_side_loads)
        metrics = {
            "arm_rotation": float(np.linalg.norm(
                peak_transforms[arm, :3, :3]
                - initial_transforms[arm, :3, :3]
            )),
            "tool_translation": float(np.linalg.norm(
                peak_transforms[tool, :3, 3] - initial_transforms[tool, :3, 3]
            )),
            "left_finger_translation": float(np.linalg.norm(
                peak_transforms[left_finger, :3, 3]
                - initial_transforms[left_finger, :3, 3]
            )),
            "right_finger_translation": float(np.linalg.norm(
                peak_transforms[right_finger, :3, 3]
                - initial_transforms[right_finger, :3, 3]
            )),
            "high_hold_min_lift": min(high_hold_heights)
            - float(baseline_center[2]),
            "max_downward_slip": max(
                grasp_relative_height - value
                for value in airborne_relative_heights
            ),
            "lower_hold_load_q10": float(np.quantile(high_hold_loads[:, 0], 0.1)),
            "upper_hold_load_q10": float(np.quantile(high_hold_loads[:, 1], 0.1)),
            "max_return_minimum_error": max(
                abs(value - baseline_minimum) for value in returned_minima
            ),
            "max_release_center_error": max(
                abs(float(value[2]) - float(baseline_center[2]))
                for value in released_centers
            ),
            "max_release_minimum_error": max(
                abs(value - baseline_minimum) for value in released_minima
            ),
            "release_lateral_load_q90": float(
                np.quantile(release_loads.max(axis=1), 0.9)
            ),
        }
        assert metrics["arm_rotation"] > 0.05, metrics
        assert metrics["tool_translation"] > 0.04, metrics
        assert metrics["left_finger_translation"] > 0.015, metrics
        assert metrics["right_finger_translation"] > 0.015, metrics
        assert metrics["high_hold_min_lift"] > 0.045, metrics
        assert metrics["max_downward_slip"] < 0.003, metrics
        assert metrics["lower_hold_load_q10"] > 0.18, metrics
        assert metrics["upper_hold_load_q10"] > 0.18, metrics
        assert metrics["max_return_minimum_error"] < 0.0015, metrics
        assert metrics["max_release_center_error"] < 0.003, metrics
        assert metrics["max_release_minimum_error"] < 0.0015, metrics
        assert metrics["release_lateral_load_q90"] < 0.02, metrics
        assert float(final_positions[:, 2].min()) > -0.01, metrics
        assert provider.diagnostics["frame"] == 1000

        provider.reset()
        np.testing.assert_allclose(
            provider.arrays["affine_transforms"], initial_transforms, atol=1.0e-8
        )
        _assert_contact_forces(
            provider, np.asarray(provider.arrays["positions"])
        )
        assert np.count_nonzero(provider.arrays["contact_forces"]) == 0
        assert provider.diagnostics["frame"] == 0
    finally:
        provider.close()
        del context
    assert np.isfinite(held_positions).all()


def test_two_deformable_contact() -> None:
    context, provider = _provider("soft_cube_stack.jscn")
    try:
        provider.step(nsteps=50)
        positions = _assert_finite(provider)
        assert provider.diagnostics["deformable_body_count"] == 2
        assert float(positions[:, 2].min()) > -0.01
    finally:
        provider.close()
        del context


def test_affine_press_target() -> None:
    context, provider = _provider("soft_cube_press.jscn")
    try:
        matches = [
            (index, str(entry["path"]))
            for index, entry in enumerate(provider.affine_bodies)
            if str(entry["path"]).endswith("/press_head")
        ]
        assert len(matches) == 1
        index, path = matches[0]
        initial_positions = _assert_finite(provider).copy()
        initial_target = np.array(
            provider.arrays["affine_transforms"][index], copy=True
        )
        for step in range(40):
            target = initial_target.copy()
            target[2, 3] -= 0.14 * min(1.0, float(step + 1) / 30.0)
            provider.set_affine_target(path, target)
            provider.step()
        final_positions = _assert_finite(provider)
        initial_height = float(np.ptp(initial_positions[:, 2]))
        final_height = float(np.ptp(final_positions[:, 2]))
        assert final_height < initial_height - 0.002
    finally:
        provider.close()
        del context


def main() -> int:
    availability = LibuipcProvider.availability(
        LibuipcConfig(module_path=MODULE_PATH)
    )
    assert availability.available, availability.reason
    test_fr3_soft_box_grasp_joint_targets()
    test_two_deformable_contact()
    test_affine_press_target()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
