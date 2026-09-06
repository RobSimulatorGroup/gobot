from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
try:
    import pytest
except ModuleNotFoundError as error:
    if __name__ != "__main__" or error.name != "pytest":
        raise
    print("SKIP: conveyor acceptance tests require pytest")
    raise SystemExit(77) from error


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "conveyor_packages"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


METRICS = _load("conveyor_acceptance_test_metrics", "conveyor_grasp_metrics.py")


@pytest.fixture(scope="module")
def runner():
    for dependency in ("torch", "mujoco", "scipy", "trimesh"):
        pytest.importorskip(dependency)
    builder = _load("conveyor_acceptance_test_builder", "build_scene.py")
    profile = _load("conveyor_acceptance_test_profile", "conveyor_profile.py")
    with pytest.MonkeyPatch.context() as patches:
        for name, module in (("build_scene", builder), ("conveyor_profile", profile),
                             ("conveyor_grasp_metrics", METRICS)):
            patches.setitem(sys.modules, name, module)
        controller = _load("conveyor_acceptance_test_controller", "conveyor_pinch_controller.py")
        patches.setitem(sys.modules, "conveyor_pinch_controller", controller)
        return _load("conveyor_acceptance_test_runner", "conveyor_mujoco_ipc.py")


def _shell():
    return np.array([[0., 0., .1], [.1, 0., .1], [0., .1, .1]])


def _pinch():
    return np.array([[0., 0., 3.], [0., 0., -1.], [0., 0., -1.], [0., 0., -1.]])


def _measurements(required_steps=4):
    return METRICS.GraspMeasurements(
        initial_shell=_shell(), face_triangles=np.array([[0, 1, 2]]),
        table_height=0., fixed_dt=.05, required_steps=required_steps,
    )


def _observe(metrics, *, shell=None, pinches=None, phase="blue_flip_grip", **kwargs):
    shell = _shell() if shell is None else shell
    fields = dict(
        phase=phase, shell=shell, vertices=shell, velocities=np.zeros_like(shell),
        fingertip_forces=np.stack([_pinch(), _pinch()]) if pinches is None else pinches,
        hand_contact_force=3., proxy_error=0., wrist_rotation_degrees=0.,
    )
    fields.update(kwargs)
    return metrics.observe(**fields)


def _finish(metrics, flipped=True):
    shell = _shell().copy()
    if flipped:
        shell[:, 1] *= -1.
    shell[:, 2] = .002
    for _ in range(2):
        _observe(metrics, shell=shell, pinches=np.zeros((2, 4, 3)),
                 phase="blue_flip_settle", hand_contact_force=0.)


def test_thumb_and_each_other_finger_must_carry_opposing_load():
    assert METRICS.opposed_pinch(_pinch(), .01)[0]
    for finger in range(4):
        forces = _pinch()
        forces[finger] = 0.
        assert not METRICS.opposed_pinch(forces, .01)[0]
    assert not METRICS.opposed_pinch(np.abs(_pinch()), .01)[0]


def test_material_normal_tracks_flip_not_translation_or_pca_sign():
    triangle = np.array([[0, 1, 2]])
    initial = METRICS.material_face_normal(_shell(), triangle)
    translated = METRICS.material_face_normal(_shell() + (2., 3., 4.), triangle)
    assert METRICS.angle_degrees(initial, translated) == pytest.approx(0.)
    flipped = _shell() * (1., -1., -1.)
    assert METRICS.angle_degrees(initial, METRICS.material_face_normal(flipped, triangle)) == pytest.approx(180.)
    with pytest.raises(ValueError, match="collapsed"):
        METRICS.material_face_normal(np.zeros((3, 3)), triangle)


def test_proxy_error_includes_rotation_and_affine_shear():
    target = np.eye(4)[None]
    proxy = target.copy()
    assert METRICS.proxy_surface_error(target, proxy, np.array([.1])) == 0.
    proxy[0, 0, 1] = .02
    assert METRICS.proxy_surface_error(target, proxy, np.array([.1])) == pytest.approx(.002)
    proxy[0, 2, 3] = .001
    assert METRICS.proxy_surface_error(target, proxy, np.array([.1])) == pytest.approx(.003)


def test_complete_contact_driven_flip_and_release_passes():
    metrics = _measurements()
    _observe(metrics)
    _observe(metrics)
    _finish(metrics)
    result = metrics.result(error=None, exact_feedback=True)
    assert result["passed"], result
    assert result["longest_airborne_pinch_seconds"] == pytest.approx(.1)


def test_pick_and_place_without_flip_fails():
    metrics = _measurements()
    _observe(metrics)
    _observe(metrics)
    _finish(metrics, flipped=False)
    result = metrics.result(error=None, exact_feedback=True)
    assert not result["passed"]
    assert "flipped_material_face" in result["failed_gates"]


def test_one_hand_or_nonconsecutive_grip_cannot_pass_airborne_hold():
    metrics = _measurements(required_steps=5)
    _observe(metrics)
    forces = np.stack([_pinch(), np.zeros((4, 3))])
    _observe(metrics, pinches=forces)
    _observe(metrics)
    _finish(metrics)
    assert not metrics.result(error=None, exact_feedback=True)["gates"]["sustained_airborne_pinch"]


@pytest.mark.parametrize("field,value,gate", [
    ("proxy_error", .002, "proxy_tracks_actual_hand"),
    ("wrist_rotation_degrees", 90., "no_wrist_flip"),
])
def test_measured_physics_violations_fail(field, value, gate):
    metrics = _measurements()
    _observe(metrics, **{field: value})
    _observe(metrics)
    _finish(metrics)
    assert not metrics.result(error=None, exact_feedback=True)["gates"][gate]


def test_short_runs_and_solver_failures_never_count_as_success():
    metrics = _measurements()
    _observe(metrics)
    result = metrics.result(error=None, exact_feedback=True)
    assert not result["gates"]["completed"]
    _observe(metrics)
    _finish(metrics)
    assert not metrics.result(error="solver failed", exact_feedback=True)["passed"]
    assert not metrics.result(error=None, exact_feedback=False)["passed"]


def test_nonfinite_state_is_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        _observe(_measurements(), proxy_error=float("nan"))


def test_missing_hand_cannot_pass_bilateral_measurement():
    with pytest.raises(ValueError, match="both hands"):
        _observe(_measurements(), pinches=_pinch()[None])


def test_released_but_moving_package_cannot_pass_settling():
    metrics = _measurements(required_steps=5)
    _observe(metrics)
    _observe(metrics)
    _finish(metrics)
    shell = _shell().copy()
    shell[:, 1] *= -1.
    shell[:, 2] = .002
    _observe(metrics, shell=shell, pinches=np.zeros((2, 4, 3)),
             phase="blue_flip_settle", hand_contact_force=0., velocities=np.ones_like(shell))
    assert not metrics.result(error=None, exact_feedback=True)["gates"]["released_and_settled"]


def test_diagnostics_failure_preserves_original_error(runner):
    class FaultedProvider:
        @property
        def diagnostics(self):
            raise RuntimeError("diagnostics unavailable")

    report = {"error": "original step failure"}
    runner._capture_diagnostics(report, FaultedProvider())
    assert report["error"] == "original step failure"
    assert "diagnostics unavailable" in report["secondary_errors"][0]


def test_runner_import_preserves_provider_config_identity(runner):
    module = importlib.import_module(runner.LibuipcBatchConfig.__module__)
    assert module.LibuipcBatchConfig is runner.LibuipcBatchConfig


def _controller(runner, **settings):
    points = np.asarray([[x, y, .63] for y in (-.55, -.51, -.49, -.42, -.23)
                         for x in (-.03, .07, .17, .25, .35)])
    faces = np.arange(24).reshape(-1, 3)
    poses = np.zeros((2, 2, 22))
    poses[:, :, :6] = runner.HAND_STAGE_BLUE_FLIP_CONTACT_TARGETS
    poses[1, :, 6:] = 1.
    controller = runner.ContactPinchController(
        poses, points, faces, runner.BLUE_SEGMENTS, runner.FIXED_DT,
        runner.PinchControlSettings(**settings),
    )
    return controller, points


def test_contact_control_uses_measured_crest_without_moving_vertices(runner):
    controller, points = _controller(runner)
    controller.tick = controller.boundaries["blue_flip_approach"][0]
    points[:, 2] += .012
    before = points.copy()
    command = controller.command(points)
    np.testing.assert_array_equal(points, before)
    np.testing.assert_array_equal(command[:, 3:6], 0.)
    np.testing.assert_allclose(controller.centers[:, 2], .648)


def test_no_contact_or_one_hand_does_not_start_lifting(runner):
    controller, points = _controller(runner, maximum_wait_steps=60)
    controller.tick = controller.boundaries["blue_flip_grip"][1] - 1
    for _ in range(60):
        controller.command(points)
        controller.observe(np.stack([_pinch(), np.zeros((4, 3))]), 0.)
    assert controller.failure is not None
    assert controller.lift_started_step is None
    assert controller.phase == "blue_flip_grip"
    with pytest.raises(RuntimeError, match="no longer advancing"):
        controller.command(points)


def test_only_sustained_bilateral_load_and_proxy_tracking_unlock_lift(runner):
    controller, _ = _controller(runner)
    controller.tick = controller.boundaries["blue_flip_grip"][1] - 1
    for _ in range(60):
        controller.observe(np.stack([_pinch(), _pinch()]), .002)
    assert controller.lift_started_step is None
    for _ in range(49):
        controller.observe(np.stack([_pinch(), _pinch()]), 0.)
    assert controller.lift_started_step is None
    controller.observe(np.stack([_pinch(), _pinch()]), 0.)
    assert controller.phase == "blue_flip_stabilize"
    assert controller.lift_started_step == 111


def test_contact_control_halts_after_grasp_loss_and_relaxes_overload(runner):
    controller, _ = _controller(runner)
    controller.tick = controller.boundaries["blue_flip_grip"][0]
    controller.closure[:] = .5
    controller.observe(100. * np.stack([_pinch(), _pinch()]), 0.)
    assert np.all(controller.closure < .5)
    controller.tick = controller.boundaries["blue_flip_stabilize"][0]
    for _ in range(10):
        controller.observe(np.zeros((2, 4, 3)), 0.)
    assert "lost during lift" in controller.failure


def test_grasp_point_tracking_is_bounded_and_freezes_for_a_loaded_hand(runner):
    controller, points = _controller(runner)
    controller.tick = controller.boundaries["blue_flip_contact"][0]
    controller.command(points)
    initial = controller.centers.copy()
    moved = points + (0., .2, 0.)
    previous = initial.copy()
    for _ in range(800):
        controller.command(moved)
        delta = np.linalg.norm(controller.centers - previous, axis=1)
        assert np.all(delta <= controller.settings.target_tracking_speed_mps * runner.FIXED_DT + 1.e-12)
        previous = controller.centers.copy()
    np.testing.assert_allclose(np.linalg.norm(controller.centers - initial, axis=1), .06)
    controller.observe(np.stack([_pinch(), np.zeros((4, 3))]), 0.)
    controller.command(points)
    np.testing.assert_array_equal(controller.centers[0], previous[0])
    assert np.linalg.norm(controller.centers[1] - initial[1]) < .06


@pytest.mark.parametrize("settings", [
    {"closing_seconds": 0.}, {"maximum_wait_steps": 1.5}, {"maximum_wait_steps": True},
    {"crest_height_offset_meters": float("nan")},
    {"maximum_fingertip_force_newtons": .1},
])
def test_invalid_contact_control_settings_are_rejected(runner, settings):
    with pytest.raises(ValueError):
        runner.PinchControlSettings(**settings)


def test_contact_control_rejects_invalid_feedback_without_advancing(runner):
    controller, _ = _controller(runner)
    for forces, error in ((np.zeros((1, 4, 3)), 0.), (np.full((2, 4, 3), np.nan), 0.),
                          (np.zeros((2, 4, 3)), -1.)):
        with pytest.raises(ValueError):
            controller.observe(forces, error)
    assert controller.steps == controller.tick == 0


def test_waiting_physics_ticks_cannot_fake_sequence_completion():
    metrics = _measurements()
    _observe(metrics)
    _observe(metrics)
    _finish(metrics)
    assert not metrics.result(error=None, exact_feedback=True, sequence_completed=False)["gates"]["completed"]


def test_commands_are_joint_targets_without_wrist_rotation(runner):
    targets = [np.asarray(pose) for pose in runner.LEAP_BLUE_AIR_PINCH_TARGETS_BY_SIDE]
    offsets = [np.array([0., .02, .01])] * 2
    trajectory = runner.joint_trajectory(runner.ACCEPTANCE_STEPS, targets, offsets)
    assert trajectory.shape == (3020, 2, 22)
    assert np.isfinite(trajectory).all()
    np.testing.assert_array_equal(trajectory[:, :, 3:6], 0.)
    np.testing.assert_allclose(trajectory[0], runner.hand_controls_at_tick(0))
    tick = sum(segment.duration for segment in runner.BLUE_SEGMENTS[:4]) - 1
    for side in range(2):
        np.testing.assert_allclose(trajectory[tick, side, 6:], targets[side])
        expected = np.asarray(runner.hand_controls_at_tick(tick)[side][:3]) + offsets[side]
        np.testing.assert_allclose(trajectory[tick, side, :3], expected)


def test_compile_trial_preserves_authored_assets_and_explicit_two_way_contract(runner):
    scene = EXAMPLE / "conveyor_packages.jscn"
    before = hashlib.sha256(scene.read_bytes()).hexdigest()
    context, artifact, face_triangles = runner.load_trial(scene)
    try:
        assert face_triangles.shape[1] == 3
        assert {body["name"] for body in artifact.ipc.deformable_bodies} == set(runner.PACKAGE_NAMES)
        assert sum(body["vertex_count"] for body in artifact.ipc.deformable_bodies) == 2600
        assert all(not body["kinematic"] for body in artifact.ipc.deformable_bodies)
        assert not artifact.ipc.deformable_attachments
        for name in runner.LEAP_ROBOT_NAMES:
            mappings = [entry for entry in artifact.coupled_bodies if entry.robot_name == name]
            assert {entry.link_name for entry in mappings} == set(runner.LEAP_CONTACT_LINK_NAMES)
            assert all(entry.mode == "TwoWay" and entry.force_scale == entry.torque_scale == 1.
                       for entry in mappings)
        statics = [entry for entry in artifact.coupled_bodies if entry.robot_name not in runner.LEAP_ROBOT_NAMES]
        assert {entry.robot_name for entry in statics} == {"warehouse_frame", "conveyor"}
        assert all(entry.mode == "OneWay" for entry in statics)
        assert artifact.collision_ownership["rigid_rigid"] == "mujoco"
        assert artifact.collision_ownership["deformable_rigid"] == "libuipc"
        assert runner.BLUE_SEGMENTS[-1].phase == "blue_flip_settle"
        assert runner.ACCEPTANCE_STEPS == 3020
        radii = runner.collision_link_radii(artifact)
        assert all(0. < radii[entry.ipc_path] < .5 for entry in artifact.coupled_bodies
                   if entry.robot_name in runner.LEAP_ROBOT_NAMES)
        targets, offsets, fits = runner.fit_pinch_targets(artifact, .0025)
        assert all(len(target) == 16 for target in targets)
        for target in targets:
            np.testing.assert_array_equal(target[[1, 5, 9]], 0.)
            np.testing.assert_allclose(target[:4], target[4:8])
            np.testing.assert_allclose(target[:4], target[8:12])
        assert all(np.linalg.norm(offset) < .03 for offset in offsets)
        assert all(abs(fit["fitted_gap_meters"] - .0025) < 1.0e-6 for fit in fits)
        poses, family_fits = runner.contact_pinch_poses(artifact, .0025)
        assert poses.shape == (17, 2, 22)
        np.testing.assert_array_equal(poses[:, :, 3:6], 0.)
        for opening, opening_fits in zip(np.linspace(.05, .0025, 17), family_fits, strict=True):
            assert all(abs(fit["fitted_gap_meters"] - opening) < 1.e-6 for fit in opening_fits)
            assert all(fit["pad_alignment_error"] < 1.e-4 for fit in opening_fits)
    finally:
        context.clear_scene()
    assert hashlib.sha256(scene.read_bytes()).hexdigest() == before


@pytest.mark.skipif(os.environ.get("GOBOT_RUN_CONVEYOR_IPC_GPU_TEST") != "1",
                    reason="requires MuJoCo Warp and native libuipc CUDA")
def test_real_gpu_smoke_is_finite_but_does_not_pass_acceptance(runner, tmp_path):
    args = runner._parser().parse_args(["--steps", "2", "--report", str(tmp_path / "report.json")])
    result = runner.run(args)
    assert result["error"] is None, result
    assert result["completed_steps"] == 2
    assert not result["passed"]
    assert result["capabilities"]["exact_contact_wrench"]
    assert not result["capabilities"]["graph_capture"]
    assert result["diagnostics"]["libuipc"]["newton_converged"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
