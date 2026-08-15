from __future__ import annotations

from functools import lru_cache
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

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
def _play():
    return _load_module(
        "gobot_rope_twist_test_play", EXAMPLE / "rope_twist_play.py"
    )


@lru_cache(maxsize=1)
def _batch():
    example_path = str(EXAMPLE)
    sys.path.insert(0, example_path)
    try:
        return _load_module(
            "gobot_rope_twist_test_batch", EXAMPLE / "rope_twist_batch.py"
        )
    finally:
        sys.path.remove(example_path)


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
        "ngeom": 23,
        "nsensor": 0,
        "nhfield": 0,
    }
    assert [robot.name for robot in artifact.mujoco.robots] == [
        *builder.ROBOT_NAMES,
        *builder.FIXTURE_BODY_NAMES,
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
    couplings = {
        (mapping.robot_name, mapping.link_name): mapping
        for mapping in artifact.coupled_bodies
    }
    expected_modes = {
        **{
            (fixture_name, fixture_name): "TwoWay"
            for fixture_name in builder.FIXTURE_BODY_NAMES
        },
        **{
            (robot_name, link_name): "OneWay"
            for robot_name in builder.ROBOT_NAMES
            for link_name in builder.ROBOT_IPC_PROXY_LINK_NAMES
        },
    }
    assert {
        key: mapping.mode for key, mapping in couplings.items()
    } == expected_modes
    fixture_proxy_indices = {
        couplings[(fixture_name, fixture_name)].ipc_body_index
        for fixture_name in builder.FIXTURE_BODY_NAMES
    }
    assert {
        int(entry["proxy_index"])
        for entry in artifact.ipc.deformable_attachments
    } == fixture_proxy_indices
    assert all(
        len(entry["vertex_indices"]) == builder.ROPE_SIDES + 1
        for entry in artifact.ipc.deformable_attachments
    )
    assert len(artifact.coupled_bodies) == 8
    assert sum(mapping.mode == "OneWay" for mapping in artifact.coupled_bodies) == 6
    assert {
        str(collider["path"]).rsplit("/", 1)[-1]
        for collider in artifact.ipc.static_colliders
    } == {"workcell_floor_collision"}
    for robot in artifact.ipc.robots:
        if robot["name"] not in builder.ROBOT_NAMES:
            continue
        tool_link = next(
            link
            for link in robot["links"]
            if link["name"] == builder.TOOL_LINK_NAME
        )
        collisions = {
            str(shape["path"]).rsplit("/", 1)[-1]: shape
            for shape in tool_link["collision_shapes"]
        }
        assert collisions["fr3_link7_collision"]["disabled"]
        assert collisions["fr3_hand_collision"]["disabled"]
        proxy = collisions[f"{robot['name']}_hand_collision_proxy"]
        assert not proxy["disabled"]
        assert proxy["shape_type"] == "box"

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

    for fixture_name in builder.FIXTURE_BODY_NAMES:
        free_joint = _mujoco_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"{fixture_name}_free_joint",
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


def test_rope_endpoints_use_selected_affine_proxy_frames() -> None:
    controllers = _controllers()
    endpoint_indices = torch.arange(6, dtype=torch.int64).reshape(2, 3, 1)
    local_endpoints = torch.tensor(
        [
            [
                [[0.1, 0.2, 0.3], [0.4, -0.2, 0.1], [-0.1, 0.5, 0.2]],
                [[-0.3, 0.2, 0.4], [0.2, 0.1, -0.5], [0.6, -0.1, 0.2]],
            ]
        ],
        dtype=torch.float64,
    )
    affine_targets = torch.eye(4, dtype=torch.float64).repeat(1, 5, 1, 1)
    rotations = (
        torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        ),
    )
    translations = (
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        torch.tensor([-2.0, 1.0, 0.5], dtype=torch.float64),
    )
    proxy_indices = (4, 1)
    positions = torch.empty((1, 6, 3), dtype=torch.float64)
    for side, proxy_index in enumerate(proxy_indices):
        affine_targets[0, proxy_index, :3, :3] = rotations[side]
        affine_targets[0, proxy_index, :3, 3] = translations[side]
        positions[0, side * 3 : side * 3 + 3] = (
            torch.matmul(local_endpoints[0, side], rotations[side].T)
            + translations[side]
        )

    measured = controllers.rope_endpoints_in_affine_frames(
        positions,
        endpoint_indices,
        affine_targets,
        proxy_indices,
    )
    torch.testing.assert_close(measured, local_endpoints)


def test_box_vertex_penetration_uses_runtime_box_frames() -> None:
    controllers = _controllers()
    positions = torch.tensor(
        (
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            ((10.8, 0.0, 0.0), (13.0, 0.0, 0.0)),
        ),
        dtype=torch.float64,
    )
    transforms = torch.eye(4, dtype=torch.float64).expand(2, 1, 4, 4).clone()
    transforms[1, 0, 0, 3] = 10.0
    sizes = torch.tensor(((2.0, 2.0, 2.0),), dtype=torch.float64)

    penetration = controllers.maximum_box_vertex_penetration(
        positions, transforms, sizes
    )

    torch.testing.assert_close(
        penetration, torch.tensor((1.0, 0.2), dtype=torch.float64)
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


def test_play_builds_bounded_soft_and_grip_contact_force_arrows() -> None:
    play = _play()
    positions = np.zeros((30, 3), dtype=np.float64)
    positions[:, 0] = np.arange(30, dtype=np.float64)
    forces = np.zeros_like(positions)
    forces[:, 2] = np.arange(1, 31, dtype=np.float64)

    arrows = play._contact_force_arrows(
        positions,
        forces,
        color=play.IPC_CONTACT_FORCE_ARROW_COLOR,
        label="rope contact",
        force_scale=0.08,
        max_force_length=0.8,
        max_count=5,
    )
    assert len(arrows) == 5
    assert tuple(arrows[0].start) == (29.0, 0.0, 0.0)
    assert tuple(arrows[0].vector) == (0.0, 0.0, 1.0)
    assert arrows[0].label == "rope contact 30 N"
    assert math.isclose(arrows[0].scale, 0.08 * math.log1p(30.0))

    uncapped_arrows = play._contact_force_arrows(
        positions,
        forces,
        color=play.IPC_CONTACT_FORCE_ARROW_COLOR,
        label="rope contact",
        force_scale=0.08,
        max_force_length=0.8,
    )
    assert len(uncapped_arrows) == len(positions)

    weak_force = np.zeros((1, 3), dtype=np.float64)
    weak_force[0, 0] = 0.01
    weak_arrow = play._contact_force_arrows(
        np.zeros_like(weak_force),
        weak_force,
        color=play.IPC_CONTACT_FORCE_ARROW_COLOR,
        label="rope contact",
        force_scale=0.08,
        max_force_length=0.8,
        min_force_length=play.IPC_CONTACT_FORCE_ARROW_MIN_LENGTH,
    )
    assert len(weak_arrow) == 1
    assert weak_arrow[0].scale == play.IPC_CONTACT_FORCE_ARROW_MIN_LENGTH

    found = np.zeros(30, dtype=bool)
    found[3] = True
    grip_arrows = play._contact_force_arrows(
        positions,
        forces,
        found=found,
        color=play.GRIP_CONTACT_FORCE_ARROW_COLOR,
        label="grip contact",
        force_scale=1.0,
        max_force_length=0.25,
    )
    assert len(grip_arrows) == 1
    assert grip_arrows[0].label == "grip contact 4 N"
    assert grip_arrows[0].scale == 0.25

    specs = play._grip_sensor_specs()
    assert [spec.name for spec in specs] == [
        "left_fixture_grip",
        "right_fixture_grip",
    ]
    assert all(
        spec.fields
        == ("found", "force", "pos", "normal", "tangent")
        for spec in specs
    )

    local_forces = torch.tensor(
        [[10.0, 2.0, 3.0], [10.0, 2.0, -3.0]]
    )
    normals = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    tangents = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    world_forces = play._contact_frame_forces_to_world(
        local_forces, normals, tangents
    )
    assert torch.allclose(
        world_forces,
        torch.tensor([[3.0, 10.0, 2.0], [-3.0, -10.0, -2.0]]),
    )


def test_interactive_scene_and_contact_refresh_cadence() -> None:
    play = _play()

    class Provider:
        frame = 0
        arrays = {"ipc_positions": torch.zeros((1, 2, 3))}
        synchronize_count = 0
        sense_count = 0
        refresh_count = 0

        def synchronize(self) -> None:
            self.synchronize_count += 1

        def sense(self) -> None:
            self.sense_count += 1

        def refresh_deformable_contact_forces(self) -> None:
            self.refresh_count += 1

    class Context:
        draw_contact_forces = False

        def get_physics_debug_settings(self):
            return {
                "draw_contact_forces": self.draw_contact_forces,
                "contact_force_scale": 1.0,
                "contact_force_max_length": 1.0,
            }

    provider = Provider()
    context = Context()
    render_frames = []
    contact_frames = []
    shown_arrows = []
    script = SimpleNamespace(
        provider=provider,
        context=context,
        quality_profile=play.QUALITY_PROFILES["interactive"],
        last_scene_sync_frame=-1,
        last_contact_refresh_frame=-1,
        contact_arrows_enabled=False,
        cached_torque_arrows=["torque"],
        cached_contact_arrows=[],
    )
    script._sync_render_state = lambda positions: render_frames.append(
        provider.frame
    )

    def refresh_contact(settings, positions) -> None:
        del settings, positions
        contact_frames.append(provider.frame)
        script.cached_contact_arrows = ["contact"]

    script._refresh_contact_arrows = refresh_contact
    original_set_debug_arrows = play.set_debug_arrows
    play.set_debug_arrows = lambda arrows: shown_arrows.append(list(arrows))
    try:
        for frame in range(3):
            provider.frame = frame
            play.Script._sync_scene(script)
        assert render_frames == [0, 2]
        assert provider.synchronize_count == 2

        context.draw_contact_forces = True
        for frame in range(3, 8):
            provider.frame = frame
            play.Script._sync_scene(script)
        assert contact_frames == [3, 7]
        assert provider.refresh_count == 2
        assert provider.sense_count == 2
        assert shown_arrows[-1] == ["torque", "contact"]

        context.draw_contact_forces = False
        provider.frame = 8
        play.Script._sync_scene(script)
        assert script.cached_contact_arrows == []
        assert shown_arrows[-1] == ["torque"]
        assert provider.refresh_count == 2
    finally:
        play.set_debug_arrows = original_set_debug_arrows


def test_rope_twist_defaults_to_solver_coupled_proxy() -> None:
    batch = _batch()
    defaults = batch._parser().parse_args([])
    assert not hasattr(defaults, "integration_scheme")
    assert defaults.coupling_iterations == 2
    assert defaults.maximum_step_latency_seconds == 0.0
    assert not defaults.continue_after_cycle_complete
    assert not defaults.defer_deformable_contact_forces
    batch._validate_args(defaults)

    full_soak = batch._parser().parse_args(
        ["--continue-after-cycle-complete"]
    )
    assert full_soak.continue_after_cycle_complete

    unequal_proxy = batch._parser().parse_args(
        ["--rigid-substeps", "2", "--ipc-substeps", "1"]
    )
    try:
        batch._validate_args(unequal_proxy)
    except ValueError as error:
        assert "requires equal rigid and IPC substep counts" in str(error)
    else:
        raise AssertionError("unequal Proxy substeps were accepted")

    negative_latency = batch._parser().parse_args(
        ["--maximum-step-latency-seconds", "-1"]
    )
    try:
        batch._validate_args(negative_latency)
    except ValueError as error:
        assert "maximum-step-latency" in str(error)
    else:
        raise AssertionError("negative step latency limit was accepted")

    play = _play()
    assert play.DEFAULT_QUALITY == "interactive"
    assert play.COUPLING_ITERATIONS == 1
    assert play._coupling_iterations() == 1


def test_rope_twist_editor_quality_profiles_are_distinct() -> None:
    play = _play()
    variables = (
        play.QUALITY_ENVIRONMENT_VARIABLE,
        play.COUPLING_ITERATIONS_ENVIRONMENT_VARIABLE,
    )
    previous = {name: os.environ.get(name) for name in variables}

    class Context:
        project_path = str(Path(tempfile.gettempdir()) / "no-gobot-repository")

    try:
        os.environ.pop(play.COUPLING_ITERATIONS_ENVIRONMENT_VARIABLE, None)
        expected = {
            "interactive": (1, "fixed", 2, 4, 16, 8, 1.0e-3),
            "accurate": (2, "aitken", 1, 1, 16, 8, 1.0e-3),
        }
        for quality, values in expected.items():
            os.environ[play.QUALITY_ENVIRONMENT_VARIABLE] = quality
            profile = play._quality_profile()
            assert (
                play._coupling_iterations(profile),
                profile.relaxation_mode,
                profile.scene_sync_interval,
                profile.contact_refresh_interval,
                profile.newton_max_iterations,
                profile.line_search_max_iterations,
                profile.linear_system_tolerance_rate,
            ) == values
            solver = play._batch_config(Context(), profile)
            assert solver.newton_max_iterations == values[4]
            assert solver.line_search_max_iterations == values[5]
            assert solver.linear_system_tolerance_rate == values[6]
            assert not solver.export_deformable_contact_forces

        os.environ[play.QUALITY_ENVIRONMENT_VARIABLE] = "invalid"
        try:
            play._quality_profile()
        except ValueError as error:
            assert play.QUALITY_ENVIRONMENT_VARIABLE in str(error)
        else:
            raise AssertionError("invalid editor quality profile was accepted")

        os.environ[play.QUALITY_ENVIRONMENT_VARIABLE] = "interactive"
        profile = play._quality_profile()
        os.environ[play.COUPLING_ITERATIONS_ENVIRONMENT_VARIABLE] = "3"
        assert play._coupling_iterations(profile) == 3
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_rope_twist_editor_coupling_iterations_are_validated() -> None:
    play = _play()
    variable = play.COUPLING_ITERATIONS_ENVIRONMENT_VARIABLE
    previous = os.environ.get(variable)
    try:
        os.environ[variable] = "1"
        assert play._coupling_iterations() == 1
        for invalid in ("0", "not-an-integer"):
            os.environ[variable] = invalid
            try:
                play._coupling_iterations()
            except ValueError as error:
                assert variable in str(error)
            else:
                raise AssertionError(
                    f"invalid iteration count {invalid!r} accepted"
                )
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


def test_authored_task_is_visible_and_runtime_driven() -> None:
    builder = _builder()
    scene = json.loads(SCENE.read_text(encoding="utf-8"))
    nodes = scene["__NODES__"]
    root = next(node for node in nodes if node["parent"] == -1)
    assert root["properties"]["script"] == "ExtResource(rope_twist_play_script)"
    assert [node["name"] for node in nodes if node["type"] == "Robot3D"] == [
        *builder.ROBOT_NAMES,
    ]
    assert [node["name"] for node in nodes if node["type"] == "RigidBody3D"] == list(
        builder.FIXTURE_BODY_NAMES
    )
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
    assert 'provider.arrays["ipc_contact_forces"]' in play_source
    assert "get_physics_debug_settings" in play_source
    assert "MuJoCoWarpContactSensorSpec" in play_source
    assert "GOBOT_ROPE_TWIST_DRIVE_MODE" in play_source
    assert "GOBOT_ROPE_TWIST_INTEGRATION_SCHEME" not in play_source
    assert "GOBOT_ROPE_TWIST_COUPLING_ITERATIONS" in play_source
    assert "GOBOT_ROPE_TWIST_QUALITY" in play_source
    assert "frame = self.provider.frame" in play_source
    assert "refresh_deformable_contact_forces" in play_source
    assert 'model_array("actuator_forcerange")' in controller_source
    assert "WRIST_SHOWCASE_TORQUE_LIMIT" in controller_source
    assert "examples.mujoco_libuipc" not in combined


def main() -> int:
    test_scene_is_reproducible()
    test_scene_compiles_to_friction_grasps_and_attached_soft_strands()
    test_both_facing_robots_stretch_outward_and_counterrotate()
    test_gravity_compensation_does_not_hide_rope_reaction()
    test_rope_endpoints_use_selected_affine_proxy_frames()
    test_box_vertex_penetration_uses_runtime_box_frames()
    test_finite_torque_controller_stalls_on_effort_and_speed()
    test_wrist_drive_modes_apply_distinct_runtime_torque_limits()
    test_play_builds_bounded_soft_and_grip_contact_force_arrows()
    test_interactive_scene_and_contact_refresh_cadence()
    test_rope_twist_defaults_to_solver_coupled_proxy()
    test_rope_twist_editor_quality_profiles_are_distinct()
    test_rope_twist_editor_coupling_iterations_are_validated()
    test_authored_task_is_visible_and_runtime_driven()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
