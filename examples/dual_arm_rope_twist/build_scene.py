"""Build the dual-FR3, three-strand rope-twisting scene."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import gobot
import numpy as np


HERE = Path(__file__).resolve().parent
SCENE_NAME = "dual_arm_rope_twist.jscn"
PLAY_SCRIPT_NAME = "rope_twist_play.py"
PLAY_SCRIPT_PATH = "res://" + PLAY_SCRIPT_NAME
PLAY_SCRIPT_RESOURCE_ID = "rope_twist_play_script"

FR3_SOURCE_PROJECT = HERE.parent / "libuipc"
FR3_SOURCE_BUILDER = FR3_SOURCE_PROJECT / "build_demos.py"
FR3_SOURCE_ASSETS = FR3_SOURCE_PROJECT / "assets"
FR3_URDF_RESOURCE = (
    "res://assets/franka_emika_panda/urdf/fr3_franka_hand.urdf"
)

SCENE_ROOT_NAME = "dual_arm_rope_twist"
LEFT_ROBOT_NAME = "left_fr3"
RIGHT_ROBOT_NAME = "right_fr3"
ROBOT_NAMES = (LEFT_ROBOT_NAME, RIGHT_ROBOT_NAME)
LEFT_FIXTURE_BODY_NAME = "left_rope_fixture"
RIGHT_FIXTURE_BODY_NAME = "right_rope_fixture"
FIXTURE_BODY_NAMES = (
    LEFT_FIXTURE_BODY_NAME,
    RIGHT_FIXTURE_BODY_NAME,
)
TOOL_LINK_NAME = "fr3_link7"
ARM_JOINT_NAMES = tuple(f"fr3_joint{index}" for index in range(1, 8))
FINGER_JOINT_NAMES = ("fr3_finger_joint1", "fr3_finger_joint2")
FINGER_LINK_NAMES = ("fr3_leftfinger", "fr3_rightfinger")
ROBOT_IPC_PROXY_LINK_NAMES = (TOOL_LINK_NAME, *FINGER_LINK_NAMES)
JOINT_NAMES = ARM_JOINT_NAMES + FINGER_JOINT_NAMES
ROBOT_LINK_NAMES = tuple(f"fr3_link{index}" for index in range(8)) + (
    *FINGER_LINK_NAMES,
)
ROPE_NAMES = ("rope_coral", "rope_gold", "rope_blue")

# Offline IK solution. Both bases remain upright; the right base is yawed by
# 180 degrees. At q=0 the link7 axes point inward along world +X and -X.
FR3_ROPE_POSE = (
    -0.7433863055,
    -0.3703417127,
    0.7452846650,
    -1.8520545191,
    1.9125360536,
    3.1076412964,
    -0.8808,
)
FR3_FINGER_OPEN_POSITION = 0.0240
FR3_FINGER_GRIP_POSITION = 0.0180
ARM_EFFORT_LIMITS = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
ARM_VELOCITY_LIMITS = (2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26)
ARM_STIFFNESS = (850.0, 850.0, 760.0, 680.0, 220.0, 170.0, 95.0)
ARM_DAMPING = (52.0, 52.0, 46.0, 40.0, 18.0, 15.0, 10.0)
WRIST_ROTATION_LIMIT = 200.0 * math.pi
WRIST_TARGET_SPEED = 1.35
WRIST_DRIVE_TORQUE_LIMIT = 0.020
# A stiff velocity servo holds the requested rate until the explicit actuator
# force limit is reached. The finite-torque batch keeps the authored 0.020 N m
# cap; editor Play can raise only that cap for its continuous showcase mode.
WRIST_VELOCITY_GAIN = 400.0
WRIST_PASSIVE_DAMPING = 0.005
GRIP_PAD_SIZE = (0.080, 0.004, 0.090)
GRIP_PAD_VISUAL_SIZE = (0.080, 0.0020, 0.090)
# At the authored finger opening, each pad just touches the fixture without
# preload. The position drive deliberately over-closes each finger by 6.0 mm
# during Play; the force-limited drive supplies a firm frictional preload. The
# visible pad and fixture are each inset 1.0 mm from their physical faces,
# concealing the solver's compliant contact depth instead of rendering
# interpenetration.
# MuJoCo Warp requires zero geom margin while MULTICCD is enabled.
GRIP_PAD_POSITION = (0.0, -0.002, 0.085)
GRIP_PAD_FRICTION = 5.0
GRIP_AXIAL_STOP_SIZE = (0.050, 0.018, 0.004)
GRIP_VERTICAL_STOP_SIZE = (0.004, 0.018, 0.050)
GRIP_STOP_PRELOAD = 5.0e-4
GRIP_STOP_FRICTION = 0.25
GRIP_STOP_POCKET_CENTER_Z = 0.0846
GRIP_STOP_COLLISION_SEGMENTS = 2
GRIP_STOP_COLLISION_GAP = 1.0e-3
GRIP_CONTACT_OFFSET = 0.0
GRIP_REST_OFFSET = 0.0
GRIP_CONTACT_COMPLIANCE = 1.0e-6
# The imported FR3 hand uses two triangle meshes. A tight primitive proxy is
# both cheaper and substantially better conditioned as a driven libuipc affine
# body, while still blocking the rope across the palm and flange.
HAND_COLLISION_PROXY_SIZE = (0.156, 0.156, 0.120)
HAND_COLLISION_PROXY_POSITION = (0.0, 0.0, 0.113)
FIXTURE_SIZE = (0.050, 0.040, 0.040)
FIXTURE_VISUAL_SIZE = (0.050, 0.0380, 0.0380)
FIXTURE_MASS = 0.080
FIXTURE_FRICTION = 5.0
FIXTURE_CENTER_X = 0.270
FIXTURE_MOUNT_LENGTH = 0.010
ATTACHMENT_STRENGTH_RATE = 5.0e3
WORKCELL_FLOOR_SIZE = (2.20, 1.20, 0.05)
WORKCELL_FLOOR_POSITION = (0.0, 0.0, -0.025)
WORKCELL_FLOOR_FRICTION = 0.8

LEFT_BASE_POSITION = (-0.90, 0.0, 0.0)
RIGHT_BASE_POSITION = (0.90, 0.0, 0.0)
ROPE_CENTER_Z = 0.72001
ROPE_LENGTH = 0.470
ROPE_RADIUS = 0.0040
ROPE_YOUNG_MODULUS = 2.25e5
STRAND_SPACING = 0.0085
_STRAND_OFFSET_RADIUS = STRAND_SPACING / math.sqrt(3.0)
_STRAND_OFFSET_ANGLES = tuple(
    0.5 * math.pi + index * 2.0 * math.pi / 3.0 for index in range(3)
)
# An equilateral cross-section keeps every strand off the bundle axis, so all
# three visibly wind around the common centerline instead of leaving one strand
# nearly straight down the middle.
STRAND_OFFSETS = tuple(
    (
        _STRAND_OFFSET_RADIUS * math.cos(angle),
        _STRAND_OFFSET_RADIUS * math.sin(angle),
    )
    for angle in _STRAND_OFFSET_ANGLES
)
ROPE_SEGMENTS = 72
ROPE_SIDES = 6
_FR3_BUILDER: Any | None = None


def _load_fr3_builder() -> Any:
    global _FR3_BUILDER
    if _FR3_BUILDER is not None:
        return _FR3_BUILDER
    if not FR3_SOURCE_BUILDER.is_file():
        raise FileNotFoundError(
            "the shared FR3 scene builder is missing: " + str(FR3_SOURCE_BUILDER)
        )
    module_name = "gobot_dual_arm_rope_fr3_source"
    spec = importlib.util.spec_from_file_location(module_name, FR3_SOURCE_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load FR3 builder: {FR3_SOURCE_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _FR3_BUILDER = module
    return module


def _nodes_by_name(root: Any) -> dict[str, Any]:
    nodes: dict[str, Any] = {}
    pending = [root]
    while pending:
        node = pending.pop()
        if node.name in nodes:
            raise RuntimeError(f"FR3 source has duplicate node name {node.name!r}")
        nodes[node.name] = node
        pending.extend(node.children)
    return nodes


def _path_from_root(root: Any, node: Any) -> str:
    names: list[str] = []
    current = node
    while current.name != root.name:
        if current.parent is None:
            raise RuntimeError(f"{node.name!r} is not below {root.name!r}")
        names.append(current.name)
        current = current.parent
    return "../" + "/".join(reversed(names))


def _translation(position: tuple[float, float, float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = position
    return transform


def _yaw_transform(
    position: tuple[float, float, float], yaw_radians: float
) -> np.ndarray:
    cosine = math.cos(yaw_radians)
    sine = math.sin(yaw_radians)
    transform = _translation(position)
    transform[:3, :3] = (
        (cosine, -sine, 0.0),
        (sine, cosine, 0.0),
        (0.0, 0.0, 1.0),
    )
    return transform


def _add_box_visual(
    parent: Any,
    name: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float, float],
) -> Any:
    visual = gobot.create_box_visual(name, size, position)
    visual.surface_color = color
    visual.semantic_label = name
    parent.add_child(visual)
    return visual


def _create_fr3_robot(
    root: Any,
    fr3_builder: Any,
    name: str,
    base_transform: np.ndarray,
) -> Any:
    source_root = fr3_builder._fr3_soft_grasp_scene()
    robot = source_root.find("fr3_arm")
    if robot is None:
        raise RuntimeError("shared FR3 scene builder did not create fr3_arm")
    robot.reparent(root)
    robot.name = name
    robot.mode = gobot.RobotMode.Assembly
    robot.source_path = FR3_URDF_RESOURCE
    fr3_builder._set_matrix(robot, base_transform)

    nodes = _nodes_by_name(robot)
    _, joint_specs = fr3_builder._parse_fr3_urdf()
    for index, (joint_name, initial) in enumerate(
        zip(ARM_JOINT_NAMES, FR3_ROPE_POSE, strict=True)
    ):
        joint = nodes[joint_name]
        spec = joint_specs[joint_name]
        if joint_name == "fr3_joint7":
            joint.drive_mode = gobot.JointDriveMode.Velocity
            joint.drive_stiffness = 0.0
            joint.drive_damping = WRIST_VELOCITY_GAIN
            joint.damping = WRIST_PASSIVE_DAMPING
            joint.effort_limit = WRIST_DRIVE_TORQUE_LIMIT
            joint.force_lower_limit = -WRIST_DRIVE_TORQUE_LIMIT
            joint.force_upper_limit = WRIST_DRIVE_TORQUE_LIMIT
        else:
            joint.drive_mode = gobot.JointDriveMode.Position
            joint.drive_stiffness = ARM_STIFFNESS[index]
            joint.drive_damping = ARM_DAMPING[index]
            joint.effort_limit = ARM_EFFORT_LIMITS[index]
            joint.force_lower_limit = -ARM_EFFORT_LIMITS[index]
            joint.force_upper_limit = ARM_EFFORT_LIMITS[index]
        joint.velocity_limit = ARM_VELOCITY_LIMITS[index]
        lower_limit = float(spec["lower"]) - initial
        upper_limit = float(spec["upper"]) - initial
        if joint_name == "fr3_joint7":
            lower_limit = -WRIST_ROTATION_LIMIT
            upper_limit = WRIST_ROTATION_LIMIT
        joint.lower_limit = lower_limit
        joint.upper_limit = upper_limit
        if joint_name == "fr3_joint7":
            joint.control_lower_limit = -WRIST_TARGET_SPEED
            joint.control_upper_limit = WRIST_TARGET_SPEED
        else:
            joint.control_lower_limit = lower_limit
            joint.control_upper_limit = upper_limit
        joint.initial_position = 0.0
        joint.joint_position = 0.0
        child = nodes[f"fr3_link{index + 1}"]
        fr3_builder._set_matrix(
            child,
            fr3_builder._joint_motion_matrix(
                spec["type"], spec["axis"], initial
            ),
        )

    for joint_name in FINGER_JOINT_NAMES:
        joint = nodes[joint_name]
        joint.drive_mode = gobot.JointDriveMode.Position
        joint.drive_stiffness = 1.2e4
        joint.drive_damping = 100.0
        joint.effort_limit = 75.0
        joint.force_lower_limit = -75.0
        joint.force_upper_limit = 75.0
        joint.lower_limit = -FR3_FINGER_OPEN_POSITION
        joint.upper_limit = 0.04 - FR3_FINGER_OPEN_POSITION
        joint.control_lower_limit = -FR3_FINGER_OPEN_POSITION
        joint.control_upper_limit = 0.04 - FR3_FINGER_OPEN_POSITION
        joint.initial_position = 0.0
        joint.joint_position = 0.0
        spec = joint_specs[joint_name]
        child = nodes[str(spec["child"])]
        fr3_builder._set_matrix(
            child,
            fr3_builder._joint_motion_matrix(
                spec["type"], spec["axis"], FR3_FINGER_OPEN_POSITION
            ),
        )

    for side, finger_name in zip(
        ("left", "right"), FINGER_LINK_NAMES, strict=True
    ):
        finger = nodes[finger_name]
        for collision_index in range(1, 5):
            suffix = "" if collision_index == 1 else f"_{collision_index}"
            nodes[f"{finger_name}_collision{suffix}"].disabled = True

        pad_visual = gobot.create_box_visual(
            f"{name}_{side}_rubber_pad", GRIP_PAD_VISUAL_SIZE
        )
        pad_visual.surface_color = (0.035, 0.045, 0.055, 1.0)
        pad_visual.semantic_label = "friction_grip_pad"
        fr3_builder._set_matrix(
            pad_visual, _translation(GRIP_PAD_POSITION)
        )
        finger.add_child(pad_visual)

        pad_collision = gobot.create_box_collision(
            f"{name}_{side}_rubber_pad_collision", GRIP_PAD_SIZE
        )
        pad_collision.visible = False
        pad_collision.physics_material = {
            "sliding_friction": GRIP_PAD_FRICTION,
            "torsional_friction": 0.0,
            "rolling_friction": 0.0,
            "contact_compliance": GRIP_CONTACT_COMPLIANCE,
            "contact_damping": 1.0,
        }
        pad_collision.contact_offset = GRIP_CONTACT_OFFSET
        pad_collision.rest_offset = GRIP_REST_OFFSET
        fr3_builder._set_matrix(
            pad_collision, _translation(GRIP_PAD_POSITION)
        )
        finger.add_child(pad_collision)

        stop_z = 0.5 * FIXTURE_SIZE[0] + 0.5 * GRIP_AXIAL_STOP_SIZE[2]
        stop_z -= GRIP_STOP_PRELOAD
        stop_y = GRIP_PAD_POSITION[1] - 0.5 * GRIP_PAD_SIZE[1]
        stop_y -= 0.5 * GRIP_AXIAL_STOP_SIZE[1]
        stop_x = 0.5 * FIXTURE_SIZE[2] + 0.5 * GRIP_VERTICAL_STOP_SIZE[0]
        stop_x -= GRIP_STOP_PRELOAD
        stop_specs = [
            (
                end,
                GRIP_AXIAL_STOP_SIZE,
                (0.0, stop_y, GRIP_STOP_POCKET_CENTER_Z + sign * stop_z),
            )
            for end, sign in (("inner", -1.0), ("outer", 1.0))
        ]
        stop_specs.extend(
            (
                edge,
                GRIP_VERTICAL_STOP_SIZE,
                (sign * stop_x, stop_y, GRIP_STOP_POCKET_CENTER_Z),
            )
            for edge, sign in (("bottom", -1.0), ("top", 1.0))
        )
        for stop_name, stop_size, stop_position in stop_specs:
            stop_visual = gobot.create_box_visual(
                f"{name}_{side}_{stop_name}_grip_stop",
                stop_size,
                stop_position,
            )
            stop_visual.surface_color = (0.035, 0.045, 0.055, 1.0)
            stop_visual.semantic_label = "keyed_grip_stop"
            finger.add_child(stop_visual)

            # MuJoCo emits only a small fixed contact set for one box pair.
            # Split each long rail into disjoint collision boxes so the
            # pocket remains stiff across its full face without changing its
            # visible shape or the global contact solver settings.
            long_axis = 0 if stop_name in ("inner", "outer") else 2
            segment_size = list(stop_size)
            segment_size[long_axis] = (
                stop_size[long_axis]
                - GRIP_STOP_COLLISION_GAP
                * (GRIP_STOP_COLLISION_SEGMENTS - 1)
            ) / GRIP_STOP_COLLISION_SEGMENTS
            segment_pitch = (
                segment_size[long_axis] + GRIP_STOP_COLLISION_GAP
            )
            segment_center = 0.5 * (GRIP_STOP_COLLISION_SEGMENTS - 1)
            for segment_index in range(GRIP_STOP_COLLISION_SEGMENTS):
                segment_position = list(stop_position)
                segment_position[long_axis] += (
                    segment_index - segment_center
                ) * segment_pitch
                stop_collision = gobot.create_box_collision(
                    f"{name}_{side}_{stop_name}_grip_stop_collision_"
                    f"{segment_index + 1}",
                    tuple(segment_size),
                    tuple(segment_position),
                )
                stop_collision.visible = False
                stop_collision.physics_material = {
                    "sliding_friction": GRIP_STOP_FRICTION,
                    "torsional_friction": 0.0,
                    "rolling_friction": 0.0,
                    "contact_compliance": GRIP_CONTACT_COMPLIANCE,
                    "contact_damping": 1.0,
                }
                stop_collision.contact_offset = GRIP_CONTACT_OFFSET
                stop_collision.rest_offset = GRIP_REST_OFFSET
                finger.add_child(stop_collision)

    for collision_name in ("fr3_link7_collision", "fr3_hand_collision"):
        nodes[collision_name].disabled = True
    hand_collision = gobot.create_box_collision(
        f"{name}_hand_collision_proxy",
        HAND_COLLISION_PROXY_SIZE,
        HAND_COLLISION_PROXY_POSITION,
    )
    hand_collision.visible = False
    hand_collision.physics_material = {
        "sliding_friction": 0.25,
        "torsional_friction": 0.005,
        "rolling_friction": 0.0001,
    }
    nodes[TOOL_LINK_NAME].add_child(hand_collision)

    for link_name in ROBOT_IPC_PROXY_LINK_NAMES:
        coupling = gobot.create_node(
            "PhysicsCoupling", f"robot_{name}_{link_name}_ipc_collision_proxy"
        )
        coupling.target_body_path = _path_from_root(root, nodes[link_name])
        coupling.mode = gobot.PhysicsCouplingMode.OneWay
        root.add_child(coupling)

    # The authored pose is the task-local q=0 for both physics compilers.
    robot.mode = gobot.RobotMode.Motion
    return robot


def _set_box_inertia(
    link: Any, mass: float, size: tuple[float, float, float]
) -> None:
    link.has_inertial = True
    link.mass = mass
    x, y, z = size
    link.inertia_diagonal = (
        mass * (y * y + z * z) / 12.0,
        mass * (x * x + z * z) / 12.0,
        mass * (x * x + y * y) / 12.0,
    )


def _create_fixture_body(
    root: Any,
    body_name: str,
    center_x: float,
) -> None:
    fixture = gobot.create_node("RigidBody3D", body_name)
    fixture.position = (center_x, 0.0, ROPE_CENTER_Z)
    _set_box_inertia(fixture, FIXTURE_MASS, FIXTURE_SIZE)

    body_visual = gobot.create_box_visual(
        "fixture_body_visual", FIXTURE_VISUAL_SIZE
    )
    body_visual.surface_color = (0.08, 0.52, 0.64, 1.0)
    body_visual.semantic_label = "rope_fixture_body"
    fixture.add_child(body_visual)

    body_collision = gobot.create_box_collision(
        "fixture_body_collision", FIXTURE_SIZE
    )
    body_collision.visible = False
    body_collision.physics_material = {
        "sliding_friction": FIXTURE_FRICTION,
        "torsional_friction": 0.0,
        "rolling_friction": 0.0,
        "contact_compliance": GRIP_CONTACT_COMPLIANCE,
        "contact_damping": 1.0,
    }
    body_collision.contact_offset = GRIP_CONTACT_OFFSET
    body_collision.rest_offset = GRIP_REST_OFFSET
    fixture.add_child(body_collision)

    inward_sign = -1.0 if center_x > 0.0 else 1.0
    mount_x = inward_sign * (
        0.5 * FIXTURE_SIZE[0] + 0.5 * FIXTURE_MOUNT_LENGTH
    )
    for rope_name, (offset_y, offset_z) in zip(
        ROPE_NAMES, STRAND_OFFSETS, strict=True
    ):
        mount = gobot.create_box_visual(
            f"{rope_name}_mount",
            (FIXTURE_MOUNT_LENGTH, 0.006, 0.006),
            (mount_x, offset_y, offset_z),
        )
        mount.surface_color = (0.96, 0.78, 0.08, 1.0)
        mount.semantic_label = "rope_fixture_mount"
        fixture.add_child(mount)

    root.add_child(fixture)


def _positive_tetrahedron(
    vertices: list[tuple[float, float, float]],
    indices: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    points = np.asarray([vertices[index] for index in indices], dtype=np.float64)
    volume6 = float(
        (points[1] - points[0])
        @ np.cross(points[2] - points[0], points[3] - points[0])
    )
    if abs(volume6) <= 1.0e-14:
        raise ValueError("rope strand generator produced a degenerate tetrahedron")
    if volume6 < 0.0:
        return (indices[1], indices[0], indices[2], indices[3])
    return indices


def _strand_tetrahedral_mesh(
    cross_section_offset: tuple[float, float],
) -> Any:
    cross_section_vertices = ROPE_SIDES + 1
    center_y, center_z = cross_section_offset
    vertices: list[tuple[float, float, float]] = []
    for section in range(ROPE_SEGMENTS + 1):
        x = -0.5 * ROPE_LENGTH + ROPE_LENGTH * section / ROPE_SEGMENTS
        vertices.append((x, center_y, center_z))
        for side in range(ROPE_SIDES):
            angle = 2.0 * math.pi * side / ROPE_SIDES
            vertices.append(
                (
                    x,
                    center_y + ROPE_RADIUS * math.cos(angle),
                    center_z + ROPE_RADIUS * math.sin(angle),
                )
            )

    tetrahedra: list[tuple[int, int, int, int]] = []
    for section in range(ROPE_SEGMENTS):
        first = section * cross_section_vertices
        second = (section + 1) * cross_section_vertices
        for side in range(ROPE_SIDES):
            next_side = (side + 1) % ROPE_SIDES
            a0 = first
            a1 = first + 1 + side
            a2 = first + 1 + next_side
            b0 = second
            b1 = second + 1 + side
            b2 = second + 1 + next_side
            for tetrahedron in (
                (a0, b0, b1, b2),
                (a0, a1, b1, b2),
                (a0, a1, a2, b2),
            ):
                tetrahedra.append(
                    _positive_tetrahedron(vertices, tetrahedron)
                )

    mesh = gobot.TetrahedralMesh()
    mesh.vertices = vertices
    mesh.tetrahedra = tetrahedra
    mesh.surface_triangles = []
    mesh.validate()
    return mesh


def _create_strand(
    name: str,
    cross_section_offset: tuple[float, float],
    color: tuple[float, float, float, float],
) -> Any:
    body = gobot.create_node("DeformableBody3D", name)
    body.mesh = _strand_tetrahedral_mesh(cross_section_offset)
    body.position = (0.0, 0.0, ROPE_CENTER_Z)
    body.semantic_label = "twisted_rope_strand"
    body.density = 980.0
    body.young_modulus = ROPE_YOUNG_MODULUS
    body.poisson_ratio = 0.36
    body.damping = 0.08
    body.self_collision_enabled = True
    body.collision_layer = 1
    body.collision_mask = 1
    body.debug_surface_color = color
    body.debug_wireframe_visible = False
    return body


def _fixture_path(body_name: str) -> str:
    return f"../{body_name}"


def _add_fixture_coupling(root: Any, body_name: str, side: str) -> None:
    coupling = gobot.create_node(
        "PhysicsCoupling", f"{side}_fixture_coupling"
    )
    coupling.target_body_path = _fixture_path(body_name)
    coupling.mode = gobot.PhysicsCouplingMode.TwoWay
    coupling.force_scale = 1.0
    coupling.torque_scale = 1.0
    root.add_child(coupling)


def _add_rope_attachment(
    root: Any,
    rope_name: str,
    fixture_body_name: str,
    side: str,
) -> None:
    section_size = ROPE_SIDES + 1
    vertex_count = (ROPE_SEGMENTS + 1) * section_size
    if side == "left":
        indices = list(range(section_size))
    else:
        indices = list(range(vertex_count - section_size, vertex_count))
    attachment = gobot.create_node(
        "DeformableAttachment3D", f"{rope_name}_{side}_fixture_mount"
    )
    attachment.deformable_body_path = f"../{rope_name}"
    attachment.rigid_link_path = _fixture_path(fixture_body_name)
    attachment.vertex_indices = indices
    attachment.strength_rate = ATTACHMENT_STRENGTH_RATE
    root.add_child(attachment)


def create_scene() -> Any:
    fr3_builder = _load_fr3_builder()
    root = gobot.create_node("Node3D", SCENE_ROOT_NAME)

    floor = _add_box_visual(
        root,
        "workcell_floor",
        WORKCELL_FLOOR_SIZE,
        WORKCELL_FLOOR_POSITION,
        (0.12, 0.14, 0.16, 1.0),
    )
    floor_collision = gobot.create_box_collision(
        "workcell_floor_collision", WORKCELL_FLOOR_SIZE
    )
    floor_collision.visible = False
    floor_collision.physics_material = {
        "sliding_friction": WORKCELL_FLOOR_FRICTION,
        "torsional_friction": 0.0,
        "rolling_friction": 0.0,
    }
    floor.add_child(floor_collision)
    for side, position in (
        ("left", LEFT_BASE_POSITION),
        ("right", RIGHT_BASE_POSITION),
    ):
        _add_box_visual(
            root,
            f"{side}_robot_pedestal",
            (0.34, 0.34, 0.08),
            (position[0], position[1], 0.04),
            (0.30, 0.33, 0.36, 1.0),
        )

    _create_fr3_robot(
        root,
        fr3_builder,
        LEFT_ROBOT_NAME,
        _yaw_transform(LEFT_BASE_POSITION, 0.0),
    )
    _create_fr3_robot(
        root,
        fr3_builder,
        RIGHT_ROBOT_NAME,
        _yaw_transform(RIGHT_BASE_POSITION, math.pi),
    )
    _create_fixture_body(
        root, LEFT_FIXTURE_BODY_NAME, -FIXTURE_CENTER_X
    )
    _create_fixture_body(
        root, RIGHT_FIXTURE_BODY_NAME, FIXTURE_CENTER_X
    )

    strand_colors = (
        (0.94, 0.22, 0.12, 1.0),
        (0.98, 0.66, 0.08, 1.0),
        (0.10, 0.48, 0.92, 1.0),
    )
    for name, color, offset in zip(
        ROPE_NAMES, strand_colors, STRAND_OFFSETS, strict=True
    ):
        root.add_child(_create_strand(name, offset, color))

    for fixture_body_name, side in (
        (LEFT_FIXTURE_BODY_NAME, "left"),
        (RIGHT_FIXTURE_BODY_NAME, "right"),
    ):
        _add_fixture_coupling(root, fixture_body_name, side)
        for rope_name in ROPE_NAMES:
            _add_rope_attachment(
                root, rope_name, fixture_body_name, side
            )
    return root


def _finalize_scene(scene_path: Path) -> None:
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    resources = scene.get("__EXT_RESOURCES__")
    nodes = scene.get("__NODES__")
    if not isinstance(resources, list) or not isinstance(nodes, list):
        raise RuntimeError("generated rope-twist scene has no resource/node table")
    resources.insert(
        0,
        {
            "__ID__": PLAY_SCRIPT_RESOURCE_ID,
            "__PATH__": PLAY_SCRIPT_PATH,
            "__TYPE__": "PythonScript",
        },
    )
    roots = [entry for entry in nodes if int(entry.get("parent", -2)) == -1]
    if len(roots) != 1:
        raise RuntimeError("generated rope-twist scene has no unique root")
    roots[0].setdefault("properties", {})["script"] = (
        f"ExtResource({PLAY_SCRIPT_RESOURCE_ID})"
    )

    resources.sort(
        key=lambda entry: (
            entry.get("__PATH__") != PLAY_SCRIPT_PATH,
            str(entry.get("__PATH__", "")),
        )
    )
    external_replacements: dict[str, str] = {}
    external_type_counts: dict[str, int] = {}
    for entry in resources:
        old_id = str(entry["__ID__"])
        if entry.get("__PATH__") == PLAY_SCRIPT_PATH:
            new_id = PLAY_SCRIPT_RESOURCE_ID
        else:
            resource_type = str(entry["__TYPE__"]).lower()
            index = external_type_counts.get(resource_type, 0)
            external_type_counts[resource_type] = index + 1
            new_id = f"external_{resource_type}_{index}"
        external_replacements[old_id] = new_id
        entry["__ID__"] = new_id

    subresources = scene.setdefault("__SUB_RESOURCES__", [])
    replacements: dict[str, str] = {}
    type_counts: dict[str, int] = {}
    for entry in subresources:
        resource_type = str(entry["__TYPE__"])
        index = type_counts.get(resource_type, 0)
        type_counts[resource_type] = index + 1
        replacements[str(entry["__ID__"])] = f"{resource_type}_{index}"

    def rewrite(value: Any) -> Any:
        if isinstance(value, str):
            for old, new in external_replacements.items():
                if value == f"ExtResource({old})":
                    return f"ExtResource({new})"
            for old, new in replacements.items():
                if value == f"SubResource({old})":
                    return f"SubResource({new})"
            return value
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    scene = rewrite(scene)
    for entry in scene["__SUB_RESOURCES__"]:
        entry["__ID__"] = replacements[str(entry["__ID__"])]
    scene_path.write_text(
        json.dumps(scene, indent=4, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _stage_project(output_dir: Path) -> None:
    if output_dir == HERE:
        return
    for name in (
        "README.md",
        "build_scene.py",
        "controllers.py",
        "project.gobot",
        "rope_twist_batch.py",
        PLAY_SCRIPT_NAME,
    ):
        source = HERE / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
    shutil.copytree(FR3_SOURCE_ASSETS, output_dir / "assets", dirs_exist_ok=True)


def build_scene(output_dir: Path = HERE) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _stage_project(output_dir)
    gobot.app.context().set_project_path(str(FR3_SOURCE_PROJECT))
    root = create_scene()
    destination = output_dir / SCENE_NAME
    gobot.app.context().set_project_path(str(output_dir))
    gobot.save_scene(root, "res://" + SCENE_NAME)
    _finalize_scene(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    print(build_scene(args.output_dir))


if __name__ == "__main__":
    main()
