from __future__ import annotations

from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile

import mujoco
import numpy as np
import torch

import gobot
from gobot.rl import CompiledMuJoCoIpcArtifact


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "dual_arm_rope_twist"
SCENE = EXAMPLE / "dual_arm_rope_twist.jscn"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _builder():
    return _load_module(
        "gobot_rope_twist_test_builder", EXAMPLE / "build_scene.py"
    )


@lru_cache(maxsize=1)
def _controllers():
    return _load_module(
        "gobot_rope_twist_test_controllers", EXAMPLE / "controllers.py"
    )


@lru_cache(maxsize=1)
def _artifact() -> CompiledMuJoCoIpcArtifact:
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(EXAMPLE))
        context.load_scene("res://" + SCENE.name)
        return CompiledMuJoCoIpcArtifact.from_context(context)
    finally:
        context.clear_scene()


def _mujoco_id(model, object_type, name: str) -> int:
    value = mujoco.mj_name2id(model, object_type, name)
    assert value >= 0, name
    return int(value)


def test_scene_is_reproducible() -> None:
    builder = _builder()
    with tempfile.TemporaryDirectory(prefix="gobot-rope-twist-scene-") as temporary:
        output = Path(temporary)
        generated = builder.build_scene(output)
        assert generated.read_bytes() == SCENE.read_bytes()
        assert {
            path.name for path in output.iterdir() if path.is_file()
        } == {
            "README.md",
            "build_scene.py",
            "controllers.py",
            "dual_arm_rope_twist.jscn",
            "project.gobot",
            "rope_twist_batch.py",
            "rope_twist_play.py",
        }


def test_scene_compiles_to_friction_grasps_and_attached_soft_strands() -> None:
    builder = _builder()
    artifact = _artifact()
    assert artifact.mujoco.dimensions == {
        "nq": 32,
        "nv": 30,
        "nu": 18,
        "nbody": 23,
        "njoint": 20,
        "ngeom": 24,
        "nsensor": 0,
        "nhfield": 0,
    }
    assert [robot.name for robot in artifact.mujoco.robots] == [
        *builder.ROBOT_NAMES,
        *builder.FIXTURE_ROBOT_NAMES,
    ]
    assert [body["name"] for body in artifact.ipc.deformable_bodies] == list(
        builder.ROPE_NAMES
    )
    assert all(
        int(body["vertex_count"])
        == (builder.ROPE_SEGMENTS + 1) * (builder.ROPE_SIDES + 1)
        and int(body["tetrahedron_count"])
        == builder.ROPE_SEGMENTS * builder.ROPE_SIDES * 3
        and math.isclose(
            float(body["young_modulus"]), builder.ROPE_YOUNG_MODULUS
        )
        for body in artifact.ipc.deformable_bodies
    )
    assert len(artifact.ipc.deformable_attachments) == 6
    assert {
        str(entry["deformable_body_path"]).rsplit("/", 1)[-1]
        for entry in artifact.ipc.deformable_attachments
    } == set(builder.ROPE_NAMES)
    assert {
        int(entry["proxy_index"])
        for entry in artifact.ipc.deformable_attachments
    } == {0, 1}
    assert all(
        len(entry["vertex_indices"]) == builder.ROPE_SIDES + 1
        for entry in artifact.ipc.deformable_attachments
    )
    assert [mapping.mode for mapping in artifact.coupled_bodies] == [
        "TwoWay",
        "TwoWay",
    ]

    model = mujoco.MjModel.from_xml_string(artifact.mujoco.content)
    for robot_name in builder.ROBOT_NAMES:
        joint_name = f"{robot_name}_fr3_joint7"
        joint_id = _mujoco_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        dof = int(model.jnt_dofadr[joint_id])
        control = artifact.mujoco.control_for_joint(joint_name)
        assert control.mode == "velocity"
        assert math.isclose(
            float(model.actuator_gainprm[control.index, 0]),
            builder.WRIST_VELOCITY_GAIN,
        )
        assert np.allclose(
            model.jnt_range[joint_id],
            (-builder.WRIST_ROTATION_LIMIT, builder.WRIST_ROTATION_LIMIT),
        )
        assert math.isclose(
            float(model.dof_damping[dof]), builder.WRIST_PASSIVE_DAMPING
        )
        assert np.allclose(
            model.actuator_ctrlrange[control.index],
            (-builder.WRIST_TARGET_SPEED, builder.WRIST_TARGET_SPEED),
        )
        assert np.allclose(
            model.actuator_forcerange[control.index],
            (
                -builder.WRIST_DRIVE_TORQUE_LIMIT,
                builder.WRIST_DRIVE_TORQUE_LIMIT,
            ),
        )

    for fixture_name in builder.FIXTURE_ROBOT_NAMES:
        free_joint = _mujoco_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"{fixture_name}_fixture_free_joint",
        )
        assert model.jnt_type[free_joint] == mujoco.mjtJoint.mjJNT_FREE
        fixture_geom = _mujoco_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{fixture_name}_fixture_body_collision",
        )
        assert model.geom_type[fixture_geom] == mujoco.mjtGeom.mjGEOM_BOX
        assert math.isclose(
            float(model.geom_friction[fixture_geom, 0]),
            builder.FIXTURE_FRICTION,
        )


def test_both_facing_robots_stretch_outward_and_counterrotate() -> None:
    builder = _builder()
    controllers = _controllers()
    assert math.isclose(
        builder.FR3_FINGER_GRIP_POSITION,
        controllers.FR3_FINGER_GRIP_POSITION,
    )
    artifact = _artifact()
    model = mujoco.MjModel.from_xml_string(artifact.mujoco.content)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tool_ids = [
        _mujoco_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"{robot_name}_{builder.TOOL_LINK_NAME}",
        )
        for robot_name in builder.ROBOT_NAMES
    ]
    initial_x = data.xpos[tool_ids, 0].copy()
    peak_tick = (
        controllers.TWIST_START_TICK + controllers.STRETCH_PERIOD_TICKS // 2
    )
    arm_targets = controllers.nominal_arm_targets(peak_tick)
    for robot_index, robot_name in enumerate(builder.ROBOT_NAMES):
        for joint_index in range(controllers.MOTION_JOINT_COUNT):
            joint_id = _mujoco_id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{robot_name}_fr3_joint{joint_index + 1}",
            )
            data.qpos[model.jnt_qposadr[joint_id]] = arm_targets[
                robot_index, joint_index
            ]
    mujoco.mj_forward(model, data)
    displacement_x = data.xpos[tool_ids, 0] - initial_x
    assert displacement_x[0] < -0.006
    assert displacement_x[1] > 0.006

    assert controllers.nominal_stretch_fraction(
        controllers.TWIST_START_TICK
    ) == 0.0
    assert math.isclose(
        controllers.nominal_stretch_fraction(peak_tick), 1.0
    )
    assert math.isclose(
        controllers.nominal_stretch_fraction(
            controllers.TWIST_START_TICK + controllers.STRETCH_PERIOD_TICKS
        ),
        0.0,
        abs_tol=1.0e-12,
    )
    assert controllers.nominal_wrist_target(
        controllers.TWIST_START_TICK - 1
    ) == 0.0
    assert controllers.nominal_wrist_target(
        controllers.TWIST_START_TICK
    ) == controllers.WRIST_TARGET_SPEED
    # Equal local joint7 signs are opposite rotations about the common world
    # rope axis because the robot bases face each other.
    command = controllers.nominal_joint_targets(peak_tick)
    assert np.all(command[:, controllers.WRIST_INDEX] > 0.0)
    assert np.allclose(command[0], command[1])


def test_gravity_compensation_does_not_hide_rope_reaction() -> None:
    builder = _builder()
    controllers = _controllers()
    artifact = _artifact()
    model = mujoco.MjModel.from_xml_string(artifact.mujoco.content)
    data = mujoco.MjData(model)
    schedule = controllers.gravity_compensation_schedule(
        artifact.mujoco.content, builder.ROBOT_NAMES
    )
    tick = controllers.TWIST_START_TICK + 937
    targets = controllers.nominal_joint_targets(tick)
    wrist_angle = 1.234
    for robot_index, robot_name in enumerate(builder.ROBOT_NAMES):
        qpos_addresses = []
        for joint_name in builder.JOINT_NAMES:
            joint_id = _mujoco_id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{robot_name}_{joint_name}",
            )
            qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        data.qpos[qpos_addresses] = targets[robot_index]
        data.qpos[qpos_addresses[controllers.WRIST_INDEX]] = wrist_angle
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    for robot_index, dof_addresses in enumerate(
        schedule.joint_dof_addresses
    ):
        predicted = (
            schedule.offset[tick, robot_index]
            + schedule.cosine[tick, robot_index] * math.cos(wrist_angle)
            + schedule.sine[tick, robot_index] * math.sin(wrist_angle)
        )
        assert np.allclose(
            predicted,
            data.qfrc_bias[list(dof_addresses)],
            rtol=0.0,
            atol=1.0e-5,
        )


def test_finite_torque_controller_stalls_on_effort_and_speed() -> None:
    controllers = _controllers()
    controller = controllers.BatchedTwistController(
        torch.zeros((2, 2, controllers.JOINT_COUNT), dtype=torch.float32),
        controllers.stall_detection_layout(2),
        fixed_dt=0.002,
    )
    controller.tick = (
        controllers.TWIST_START_TICK
        + controllers.STRETCH_PERIOD_TICKS
        + controllers.STRETCH_PERIOD_TICKS // 2
    )
    wrench = torch.zeros((2, 2, 6), dtype=torch.float32)
    wrench[:, :, 5] = controllers.WRIST_DRIVE_TORQUE_LIMIT * 0.9
    positions = torch.zeros_like(controller.command)
    positions[:, :, controllers.WRIST_INDEX] = 4.0 * math.pi
    velocities = torch.zeros_like(controller.command)
    efforts = torch.full(
        (2, 2), controllers.WRIST_DRIVE_TORQUE_LIMIT, dtype=torch.float32
    )
    for _ in range(controllers.STALL_CONFIRM_TICKS + 50):
        command = controller.step(wrench, positions, velocities, efforts)
    assert bool(controller.stalled.all().item())
    assert bool((controller.stall_tick > 0).all().item())
    assert torch.allclose(
        controller.stalled_relative_rotation,
        torch.full((2,), 8.0 * math.pi),
    )
    assert torch.all(
        controller.stalled_wrist_speed
        <= controllers.STALL_SPEED_THRESHOLD
    )
    assert torch.all(
        controller.stalled_wrist_effort
        >= controllers.WRIST_DRIVE_TORQUE_LIMIT
        * controllers.STALL_TORQUE_FRACTION
    )
    assert torch.all(
        controller.stalled_axial_torque
        >= controllers.WRIST_DRIVE_TORQUE_LIMIT
        * controllers.STALL_REACTION_TORQUE_FRACTION
    )
    # The velocity request stays active. A stalled wrist is therefore a
    # physical torque-limit result, not a controller-authored stop.
    assert torch.allclose(
        command[:, :, controllers.WRIST_INDEX],
        torch.full((2, 2), controllers.WRIST_TARGET_SPEED),
    )
    while not controller.cycle_complete:
        controller.step(wrench, positions, velocities, efforts)
    assert controller.phase == "TorqueStalled"

    unloaded = controllers.BatchedTwistController(
        torch.zeros((1, 2, controllers.JOINT_COUNT), dtype=torch.float32),
        controllers.stall_detection_layout(1),
        fixed_dt=0.002,
    )
    unloaded.tick = (
        controllers.TWIST_START_TICK
        + controllers.STALL_DETECTION_DELAY_TICKS
    )
    unloaded_positions = positions[:1]
    unloaded_velocities = velocities[:1]
    unloaded_efforts = efforts[:1]
    for _ in range(controllers.STALL_CONFIRM_TICKS + 50):
        unloaded.step(
            torch.zeros((1, 2, 6), dtype=torch.float32),
            unloaded_positions,
            unloaded_velocities,
            unloaded_efforts,
        )
    assert not bool(unloaded.stalled.any().item())


def test_wrist_drive_modes_apply_distinct_runtime_torque_limits() -> None:
    controllers = _controllers()
    assert math.isclose(
        controllers.wrist_drive_torque_limit(
            controllers.FINITE_TORQUE_DRIVE_MODE
        ),
        controllers.WRIST_DRIVE_TORQUE_LIMIT,
    )
    assert math.isclose(
        controllers.wrist_drive_torque_limit(controllers.SHOWCASE_DRIVE_MODE),
        controllers.WRIST_SHOWCASE_TORQUE_LIMIT,
    )

    class FakeRigidSolver:
        def __init__(self) -> None:
            self.force_range = torch.zeros((1, 5, 2), dtype=torch.float32)
            self.recompute_count = 0

        def model_array(self, name: str):
            assert name == "actuator_forcerange"
            return self.force_range

        def recompute_constants(self) -> None:
            self.recompute_count += 1

    solver = FakeRigidSolver()
    controllers.configure_wrist_torque_limit(
        solver,
        (1, 3),
        controllers.WRIST_SHOWCASE_TORQUE_LIMIT,
    )
    expected = torch.tensor(
        [
            -controllers.WRIST_SHOWCASE_TORQUE_LIMIT,
            controllers.WRIST_SHOWCASE_TORQUE_LIMIT,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(solver.force_range[0, 1], expected)
    assert torch.allclose(solver.force_range[0, 3], expected)
    assert solver.recompute_count == 1


def test_authored_task_is_visible_and_runtime_driven() -> None:
    builder = _builder()
    scene = json.loads(SCENE.read_text(encoding="utf-8"))
    nodes = scene["__NODES__"]
    root = next(node for node in nodes if node["parent"] == -1)
    assert root["properties"]["script"] == "ExtResource(rope_twist_play_script)"
    assert [node["name"] for node in nodes if node["type"] == "Robot3D"] == [
        *builder.ROBOT_NAMES,
        *builder.FIXTURE_ROBOT_NAMES,
    ]
    assert [
        node["name"] for node in nodes if node["type"] == "DeformableBody3D"
    ] == list(builder.ROPE_NAMES)
    assert len(
        [node for node in nodes if node["type"] == "DeformableAttachment3D"]
    ) == 6
    assert all(
        not node["properties"]["debug_wireframe_visible"]
        and node["properties"]["debug_surface_color"]["alpha"] == 1.0
        for node in nodes
        if node["type"] == "DeformableBody3D"
    )
    assert len(
        [node for node in nodes if node["name"] == "fixture_body_visual"]
    ) == 2
    assert not any(
        "shell" in node["name"] or "cover" in node["name"] for node in nodes
    )
    project = json.loads((EXAMPLE / "project.gobot").read_text(encoding="utf-8"))
    view = project["editor_scene_views"]["res://dual_arm_rope_twist.jscn"]
    assert abs(float(view["eye"][0])) < 1.5
    assert float(view["eye"][1]) > 1.4
    assert float(view["at"][2]) > 0.5

    batch_source = (EXAMPLE / "rope_twist_batch.py").read_text(encoding="utf-8")
    play_source = (EXAMPLE / "rope_twist_play.py").read_text(encoding="utf-8")
    controller_source = (EXAMPLE / "controllers.py").read_text(encoding="utf-8")
    combined = batch_source + play_source + controller_source
    assert play_source.index("import torch") < play_source.index(
        "from gobot.ipc import"
    )
    assert 'provider.arrays["actuator_force"]' in combined
    assert 'rigid_arrays["xfrc_applied"]' in controller_source
    assert 'provider.arrays["ipc_contact_forces"]' in batch_source
    assert "fixture_wrenches_in_tool_frames" in combined
    assert "BatchedGravityCompensator" in combined
    assert "ProviderPlaySession" in play_source
    assert "apply_link_poses" in play_source
    assert "apply_deformable_vertices" in play_source
    assert "DebugArrow" in play_source
    assert "GOBOT_ROPE_TWIST_DRIVE_MODE" in play_source
    assert 'model_array("actuator_forcerange")' in controller_source
    assert "WRIST_SHOWCASE_TORQUE_LIMIT" in controller_source
    assert "examples.mujoco_libuipc" not in combined


def main() -> int:
    test_scene_is_reproducible()
    test_scene_compiles_to_friction_grasps_and_attached_soft_strands()
    test_both_facing_robots_stretch_outward_and_counterrotate()
    test_gravity_compensation_does_not_hide_rope_reaction()
    test_finite_torque_controller_stalls_on_effort_and_speed()
    test_wrist_drive_modes_apply_distinct_runtime_torque_limits()
    test_authored_task_is_visible_and_runtime_driven()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
