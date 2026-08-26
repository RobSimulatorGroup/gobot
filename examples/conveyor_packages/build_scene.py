"""Build the coupled rigid/deformable parcel conveyor scene."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import gobot
import numpy as np


HERE = Path(__file__).resolve().parent
SCENE_NAME = "conveyor_packages.jscn"
PLAY_SCRIPT_NAME = "conveyor_packages_play.py"
PLAY_SCRIPT_PATH = "res://" + PLAY_SCRIPT_NAME
PLAY_SCRIPT_RESOURCE_ID = "conveyor_packages_play_script"

LEAP_ASSET_ROOT = HERE / "assets" / "leap_hand"
LEAP_RESOURCE_ROOT = "res://assets/leap_hand"
HAND_SIDES = ("left", "right")
# Place the anatomical left/right models on their matching world sides so both
# thumbs point inward toward the workpiece.
LEAP_MJCF_BY_SIDE = ("left_hand.xml", "right_hand.xml")
LEAP_ROBOT_NAMES = tuple(f"leap_{side}" for side in HAND_SIDES)
LEAP_FINGER_JOINT_NAMES = (
    "if_mcp",
    "if_rot",
    "if_pip",
    "if_dip",
    "mf_mcp",
    "mf_rot",
    "mf_pip",
    "mf_dip",
    "rf_mcp",
    "rf_rot",
    "rf_pip",
    "rf_dip",
    "th_cmc",
    "th_axl",
    "th_mcp",
    "th_ipl",
)
LEAP_FINGER_CLOSE_TARGETS = (
    1.08,
    0.0,
    0.82,
    0.78,
    1.08,
    0.0,
    0.82,
    0.78,
    1.08,
    0.0,
    0.82,
    0.78,
    0.82,
    0.50,
    0.82,
    0.78,
)
LEAP_CONTACT_LINK_NAMES = (
    "palm",
    "if_bs",
    "if_px",
    "if_md",
    "if_ds",
    "mf_bs",
    "mf_px",
    "mf_md",
    "mf_ds",
    "rf_bs",
    "rf_px",
    "rf_md",
    "rf_ds",
    "th_mp",
    "th_bs",
    "th_px",
    "th_ds",
)
HAND_STAGE_DOF_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
HAND_STAGE_JOINT_NAMES_BY_SIDE = tuple(
    tuple(f"leap_{side}_wrist_{name}" for name in HAND_STAGE_DOF_NAMES)
    for side in HAND_SIDES
)
HAND_STAGE_LINK_NAMES_BY_SIDE = tuple(
    (f"leap_{side}_mount",)
    + tuple(
        f"leap_{side}_stage_{name}" for name in HAND_STAGE_DOF_NAMES[:-1]
    )
    for side in HAND_SIDES
)
HAND_BASE_LINK_NAMES = tuple(names[1] for names in HAND_STAGE_LINK_NAMES_BY_SIDE)
HAND_JOINT_NAMES_BY_SIDE = tuple(
    stage_names + LEAP_FINGER_JOINT_NAMES
    for stage_names in HAND_STAGE_JOINT_NAMES_BY_SIDE
)
HAND_LINK_NAMES_BY_SIDE = tuple(
    stage_names[1:] + LEAP_CONTACT_LINK_NAMES
    for stage_names in HAND_STAGE_LINK_NAMES_BY_SIDE
)
MANIPULATION_STATION_X = 0.16
HAND_ROOT_POSITIONS = (
    # The two Menagerie CAD frames are asymmetric. These roots align their
    # fingertips in world Y/Z while leaving room for the inward-facing thumbs.
    (MANIPULATION_STATION_X - 0.18, 0.0433, 1.0930),
    (MANIPULATION_STATION_X + 0.18, -0.0321, 1.1195),
)
# Menagerie presents the hands palm-up. This proper rotation turns the palms
# downward and points the fingers toward the outfeed (+Y).
HAND_PALM_ALIGNMENT_ROTATION = np.asarray(
    ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    dtype=np.float64,
)
HAND_STAGE_TRANSLATION_RANGE = 1.24
HAND_STAGE_ROTATION_RANGE = math.pi + 0.12
# Roll is intentionally unwrapped across the one-shot manipulation cycle. The
# hands always turn toward the outfeed, so successive flips reach equivalent
# orientations at increasingly negative joint coordinates without a 2*pi
# controller jump between parcels.
HAND_STAGE_ROLL_RANGE = 6.0 * math.pi + 0.12
HAND_STAGE_LINEAR_STIFFNESS = 9000.0
HAND_STAGE_LINEAR_DAMPING = 240.0
HAND_STAGE_ANGULAR_STIFFNESS = 1400.0
HAND_STAGE_ANGULAR_DAMPING = 85.0
HAND_FINGER_STIFFNESS = 36.0
HAND_FINGER_DAMPING = 1.20
HAND_FRICTION = 3.00
LEAP_TIP_BOUNDS = (
    (-0.0112503, -0.0500004, 0.0023530),
    (0.00958777, -0.0195963, 0.0266643),
)
LEAP_THUMB_TIP_BOUNDS = (
    (-0.0112650, -0.0620958, -0.0266430),
    (0.0095930, -0.0315304, -0.00234675),
)
RIGID_COLLISION_LAYER = 0b0001
HAND_COLLISION_LAYER = 0b0010
DEFORMABLE_COLLISION_LAYER = 0b0100
RIGID_PACKAGE_COLLISION_LAYER = 0b1000
RIGID_COLLISION_MASK = (
    RIGID_COLLISION_LAYER
    | DEFORMABLE_COLLISION_LAYER
    | RIGID_PACKAGE_COLLISION_LAYER
)
HAND_COLLISION_MASK = (
    DEFORMABLE_COLLISION_LAYER | RIGID_PACKAGE_COLLISION_LAYER
)
DEFORMABLE_COLLISION_MASK = (
    RIGID_COLLISION_LAYER
    | HAND_COLLISION_LAYER
    | RIGID_PACKAGE_COLLISION_LAYER
)
RIGID_PACKAGE_COLLISION_MASK = (
    RIGID_COLLISION_LAYER
    | HAND_COLLISION_LAYER
    | DEFORMABLE_COLLISION_LAYER
    | RIGID_PACKAGE_COLLISION_LAYER
)

BELT_FRAME_LENGTH = 2.70
BELT_SURFACE_LENGTH = BELT_FRAME_LENGTH
BELT_PROXY_LENGTH = BELT_SURFACE_LENGTH
BELT_WIDTH = 0.72
BELT_THICKNESS = 0.04
BELT_CENTER_Z = 0.54
BELT_TOP_Z = BELT_CENTER_Z + 0.5 * BELT_THICKNESS
BELT_CENTER_X = 0.0
BELT_CENTER_Y = 0.58
WORKTABLE_LENGTH = 2.10
# Preserve the 15 mm handoff gap to the outfeed while extending the sorting
# surface toward the operator. The deeper table supports the full reverse face
# of a pouch after an outward turnover instead of letting a successful flip
# fall past the old front edge.
WORKTABLE_DEPTH = 1.08
WORKTABLE_THICKNESS = 0.08
WORKTABLE_CENTER_X = -0.15
WORKTABLE_CENTER_Y = -0.335
WORKTABLE_TOP_Z = BELT_TOP_Z
# Polished sorting-table laminate against a plastic mailer.  Keeping this below
# the palm's 2.5 coefficient lets the down-facing hand sweep the parcel by
# contact friction while the belt remains stationary.
WORKTABLE_SLIDING_FRICTION = 0.30

RIGID_BOX_SPECS = (
    {
        "name": "carton_small",
        "size": (0.25, 0.20, 0.18),
        "mass": 0.62,
        # Incoming rigid parcel waiting on the left side of the static table.
        "position": (-1.02, 0.015, WORKTABLE_TOP_Z + 0.092),
        "rotation_degrees": (0.0, 0.0, 5.0),
        "color": (0.70, 0.43, 0.20, 1.0),
    },
    {
        "name": "carton_wide",
        "size": (0.31, 0.24, 0.15),
        "mass": 0.84,
        # Keep the rigid carton ahead of the gripped mailer so the interactive
        # x1 profile demonstrates transport instead of a deliberate rear-end
        # impact during braking.
        "position": (0.60, BELT_CENTER_Y + 0.10, BELT_TOP_Z + 0.077),
        "rotation_degrees": (0.0, 0.0, -7.0),
        "color": (0.82, 0.55, 0.25, 1.0),
    },
    {
        "name": "carton_tall",
        "size": (0.22, 0.18, 0.27),
        "mass": 0.76,
        "position": (0.96, BELT_CENTER_Y - 0.11, BELT_TOP_Z + 0.137),
        "rotation_degrees": (0.0, 0.0, 9.0),
        "color": (0.62, 0.34, 0.16, 1.0),
    },
)

SOFT_PACKAGE_SPECS = (
    {
        "name": "soft_mailer_blue",
        "model": "thin_shell",
        "size": (0.44, 0.32, 0.13),
        "position": (
            MANIPULATION_STATION_X,
            -0.375,
            WORKTABLE_TOP_Z + 0.235,
        ),
        "rotation_degrees": (4.0, -6.0, 2.0),
        # libuipc currently has no deformable-to-deformable attachment. Carry
        # most parcel inertia on the closed film so a fingertip grasp moves the
        # package as one object; the light inner core only supports its volume.
        # The 0.35 kg total models a filled poly mailer rather than the earlier
        # 1.12 kg parcel, which could not be rolled realistically by the two
        # palm contacts while the table supports its weight.
        "density": 675.3040236312168,
        "young_modulus": 1.6e5,
        "poisson_ratio": 0.36,
        "damping": 7.0,
        "thickness": 1.2e-3,
        "bending_stiffness": 1.2e-4,
        "cells": (34, 25),
        "color": (0.10, 0.36, 0.64, 1.0),
        "visible": True,
    },
    {
        "name": "soft_mailer_blue_fill",
        "model": "volumetric",
        "size": (0.39, 0.275, 0.110),
        "position": (
            MANIPULATION_STATION_X,
            -0.375,
            # Center the contents inside the asymmetric film cavity. The
            # mailer's top is intentionally fuller than its bottom, so the
            # core sits 10 mm below the shell origin to leave IPC clearance.
            WORKTABLE_TOP_Z + 0.225,
        ),
        "rotation_degrees": (4.0, -6.0, 2.0),
        "density": 5.44371665998382,
        "young_modulus": 7.0e3,
        "poisson_ratio": 0.43,
        "damping": 7.0,
        "cells": (12, 9, 5),
        "color": (0.04, 0.08, 0.12, 0.0),
        "visible": False,
    },
    {
        "name": "soft_pouch_yellow",
        "model": "thin_shell",
        "size": (0.29, 0.205, 0.120),
        # The second mailer starts above the upstream table and settles under
        # its own weight. Its visible film is finer than the blue mailer and
        # encloses a separate soft fill body, avoiding the old solid trapezoid
        # silhouette while retaining real volume during the later turnover.
        "position": (
            -0.42,
            0.055,
            WORKTABLE_TOP_Z + 0.250,
        ),
        "rotation_degrees": (2.0, -5.0, 8.0),
        "density": 1189.6129109172923,
        "young_modulus": 1.2e5,
        "poisson_ratio": 0.38,
        "damping": 6.0,
        "thickness": 1.1e-3,
        # The fine 36 x 27 film needs enough edge bending resistance to keep
        # the unsupported heat-sealed flange flat instead of rolling outward.
        "bending_stiffness": 1.6e-4,
        "cells": (36, 27),
        "color": (0.88, 0.56, 0.08, 1.0),
        "visible": True,
    },
    {
        "name": "soft_pouch_yellow_fill",
        "model": "volumetric",
        "size": (0.255, 0.175, 0.095),
        "position": (
            -0.42,
            0.055,
            WORKTABLE_TOP_Z + 0.242,
        ),
        "rotation_degrees": (2.0, -5.0, 8.0),
        "density": 13.65133343681454,
        "young_modulus": 5.0e3,
        "poisson_ratio": 0.44,
        "damping": 6.0,
        # Extra through-thickness layers resolve a rounded pillow profile:
        # the contents bulge at mid-height and taper toward both film sheets.
        "cells": (14, 11, 8),
        "side_rounding": 0.08,
        "color": (0.08, 0.06, 0.02, 0.0),
        "visible": False,
    },
)


def _nodes_by_name(root: Any) -> dict[str, Any]:
    nodes: dict[str, Any] = {}
    pending = [root]
    while pending:
        node = pending.pop()
        if node.name in nodes:
            raise RuntimeError(
                f"source scene has duplicate node name {node.name!r}"
            )
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


def _axis_angle_quaternion(
    axis: tuple[float, float, float], angle: float
) -> tuple[float, float, float, float]:
    normalized = np.asarray(axis, dtype=np.float64)
    length = float(np.linalg.norm(normalized))
    if length <= 1.0e-12:
        raise ValueError("revolute joint axis must be non-zero")
    normalized /= length
    sine = math.sin(0.5 * angle)
    return (
        math.cos(0.5 * angle),
        float(normalized[0]) * sine,
        float(normalized[1]) * sine,
        float(normalized[2]) * sine,
    )


def _axis_angle_matrix(
    axis: tuple[float, float, float], angle: float
) -> np.ndarray:
    normalized = np.asarray(axis, dtype=np.float64)
    normalized /= np.linalg.norm(normalized)
    x, y, z = normalized
    cosine = math.cos(angle)
    sine = math.sin(angle)
    one_minus_cosine = 1.0 - cosine
    return np.asarray(
        (
            (
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ),
            (
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ),
            (
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ),
        ),
        dtype=np.float64,
    )


def _rigid_matrix(
    position: tuple[float, float, float], rotation: np.ndarray
) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)
    return matrix


def _euler_degrees_matrix(rotation_degrees: tuple[float, float, float]) -> np.ndarray:
    x, y, z = (math.radians(float(value)) for value in rotation_degrees)
    rotation_x = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, math.cos(x), -math.sin(x)),
            (0.0, math.sin(x), math.cos(x)),
        ),
        dtype=np.float64,
    )
    rotation_y = np.asarray(
        (
            (math.cos(y), 0.0, math.sin(y)),
            (0.0, 1.0, 0.0),
            (-math.sin(y), 0.0, math.cos(y)),
        ),
        dtype=np.float64,
    )
    rotation_z = np.asarray(
        (
            (math.cos(z), -math.sin(z), 0.0),
            (math.sin(z), math.cos(z), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return rotation_z @ rotation_y @ rotation_x


def _node_local_rigid_matrix(node: Any) -> np.ndarray:
    return _rigid_matrix(
        tuple(float(value) for value in node.position),
        _euler_degrees_matrix(
            tuple(float(value) for value in node.rotation_degrees)
        ),
    )


def _matrix_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return (
            0.25 * scale,
            float(rotation[2, 1] - rotation[1, 2]) / scale,
            float(rotation[0, 2] - rotation[2, 0]) / scale,
            float(rotation[1, 0] - rotation[0, 1]) / scale,
        )

    diagonal = np.diag(rotation)
    index = int(np.argmax(diagonal))
    if index == 0:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        return (
            float(rotation[2, 1] - rotation[1, 2]) / scale,
            0.25 * scale,
            float(rotation[0, 1] + rotation[1, 0]) / scale,
            float(rotation[0, 2] + rotation[2, 0]) / scale,
        )
    if index == 1:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        return (
            float(rotation[0, 2] - rotation[2, 0]) / scale,
            float(rotation[0, 1] + rotation[1, 0]) / scale,
            0.25 * scale,
            float(rotation[1, 2] + rotation[2, 1]) / scale,
        )
    scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
    return (
        float(rotation[1, 0] - rotation[0, 1]) / scale,
        float(rotation[0, 2] + rotation[2, 0]) / scale,
        float(rotation[1, 2] + rotation[2, 1]) / scale,
        0.25 * scale,
    )


def _set_box_inertia(
    link: Any, mass: float, size: tuple[float, float, float]
) -> None:
    size_x, size_y, size_z = size
    link.has_inertial = True
    link.mass = mass
    link.inertia_diagonal = (
        mass * (size_y * size_y + size_z * size_z) / 12.0,
        mass * (size_x * size_x + size_z * size_z) / 12.0,
        mass * (size_x * size_x + size_y * size_y) / 12.0,
    )


def _add_visual(
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


def _add_oriented_visual(
    parent: Any,
    name: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    rotation_degrees: tuple[float, float, float],
    color: tuple[float, float, float, float],
) -> Any:
    visual = _add_visual(parent, name, size, position, color)
    visual.rotation_degrees = rotation_degrees
    return visual


def _add_collision(
    parent: Any,
    name: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    *,
    sliding_friction: float,
    orientation: tuple[float, float, float, float] | None = None,
    collision_layer: int = RIGID_COLLISION_LAYER,
    collision_mask: int = RIGID_COLLISION_MASK,
) -> Any:
    collision = gobot.create_box_collision(name, size, position)
    if orientation is not None:
        collision.set_transform(position, orientation)
    collision.visible = False
    collision.collision_layer = collision_layer
    collision.collision_mask = collision_mask
    collision.physics_material = {
        "sliding_friction": sliding_friction,
        "torsional_friction": 0.004,
        "rolling_friction": 0.0002,
        "contact_compliance": 0.0,
        "contact_damping": 1.0,
    }
    # MuJoCo Warp's MULTICCD path requires zero geom margin. libuipc uses its
    # own contact_activation_distance, so deformable contact remains buffered.
    collision.contact_offset = 0.0
    collision.rest_offset = 0.0
    parent.add_child(collision)
    return collision


def _add_box_geometry(
    parent: Any,
    name: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float, float],
    *,
    sliding_friction: float,
    collision_layer: int = RIGID_COLLISION_LAYER,
    collision_mask: int = RIGID_COLLISION_MASK,
) -> None:
    _add_visual(parent, name + "_visual", size, position, color)
    _add_collision(
        parent,
        name + "_collision",
        size,
        position,
        sliding_friction=sliding_friction,
        collision_layer=collision_layer,
        collision_mask=collision_mask,
    )


def _set_virtual_link_inertia(link: Any) -> None:
    link.has_inertial = True
    link.mass = 0.02
    link.inertia_diagonal = (2.0e-5, 2.0e-5, 2.0e-5)


def _configure_stage_joint(joint: Any, dof_name: str) -> None:
    linear = dof_name in {"x", "y", "z"}
    joint.joint_type = (
        gobot.JointType.Prismatic if linear else gobot.JointType.Revolute
    )
    joint.axis = {
        "x": (1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
        "roll": (1.0, 0.0, 0.0),
        "pitch": (0.0, 1.0, 0.0),
        "yaw": (0.0, 0.0, 1.0),
    }[dof_name]
    limit = (
        HAND_STAGE_TRANSLATION_RANGE
        if linear
        else (
            HAND_STAGE_ROLL_RANGE
            if dof_name == "roll"
            else HAND_STAGE_ROTATION_RANGE
        )
    )
    joint.lower_limit = -limit
    joint.upper_limit = limit
    joint.control_lower_limit = -limit
    joint.control_upper_limit = limit
    joint.velocity_limit = 1.8 if linear else 7.0
    joint.effort_limit = 1800.0 if linear else 420.0
    joint.force_lower_limit = -joint.effort_limit
    joint.force_upper_limit = joint.effort_limit
    joint.armature = 0.01 if linear else 0.004
    joint.damping = 8.0 if linear else 0.25
    joint.drive_mode = gobot.JointDriveMode.Position
    joint.drive_stiffness = (
        HAND_STAGE_LINEAR_STIFFNESS
        if linear
        else HAND_STAGE_ANGULAR_STIFFNESS
    )
    joint.drive_damping = (
        HAND_STAGE_LINEAR_DAMPING
        if linear
        else HAND_STAGE_ANGULAR_DAMPING
    )


def _box_from_bounds(
    bounds: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    lower = np.asarray(bounds[0], dtype=np.float64)
    upper = np.asarray(bounds[1], dtype=np.float64)
    return (
        tuple(float(value) for value in upper - lower),
        tuple(float(value) for value in 0.5 * (lower + upper)),
    )


def _configure_leap_model(nodes: dict[str, Any], robot_name: str) -> None:
    for node in nodes.values():
        if node.type_name == "MeshInstance3D":
            node.semantic_label = robot_name + "_dexterous_hand"
        elif node.type_name == "CollisionShape3D":
            node.disabled = False
            node.visible = False
            node.collision_layer = HAND_COLLISION_LAYER
            node.collision_mask = HAND_COLLISION_MASK
            node.contact_offset = 0.0
            node.rest_offset = 0.0
            node.physics_material = {
                "sliding_friction": HAND_FRICTION,
                "torsional_friction": 0.006,
                "rolling_friction": 0.0002,
                "contact_compliance": 0.0,
                "contact_damping": 1.0,
            }

    for joint_name in LEAP_FINGER_JOINT_NAMES:
        joint = nodes[joint_name]
        joint.drive_mode = gobot.JointDriveMode.Position
        joint.drive_stiffness = HAND_FINGER_STIFFNESS
        joint.drive_damping = HAND_FINGER_DAMPING
        joint.damping = 0.03
        joint.armature = 0.002
        joint.force_lower_limit = -12.0
        joint.force_upper_limit = 12.0
        joint.initial_position = 0.0
        joint.joint_position = 0.0

    regular_tip_size, regular_tip_center = _box_from_bounds(LEAP_TIP_BOUNDS)
    for finger_name in ("if", "mf", "rf"):
        _add_collision(
            nodes[f"{finger_name}_ds"],
            f"{robot_name}_{finger_name}_tip_collision",
            regular_tip_size,
            regular_tip_center,
            sliding_friction=HAND_FRICTION,
            collision_layer=HAND_COLLISION_LAYER,
            collision_mask=HAND_COLLISION_MASK,
        )
    thumb_tip_size, thumb_tip_center = _box_from_bounds(
        LEAP_THUMB_TIP_BOUNDS
    )
    _add_collision(
        nodes["th_ds"],
        f"{robot_name}_th_tip_collision",
        thumb_tip_size,
        thumb_tip_center,
        sliding_friction=HAND_FRICTION,
        collision_layer=HAND_COLLISION_LAYER,
        collision_mask=HAND_COLLISION_MASK,
    )


def _create_leap_hand(root: Any, side: str, side_index: int) -> None:
    robot_name = LEAP_ROBOT_NAMES[side_index]
    resource_name = LEAP_MJCF_BY_SIDE[side_index]
    resource_path = f"{LEAP_RESOURCE_ROOT}/{resource_name}"
    source_scene = gobot.load_scene(resource_path)
    source_nodes = _nodes_by_name(source_scene.root)
    palm = source_nodes["palm"]
    _configure_leap_model(source_nodes, robot_name)

    robot = gobot.create_node("Robot3D", robot_name)
    robot.mode = gobot.RobotMode.Motion
    robot.source_path = resource_path
    robot.position = HAND_ROOT_POSITIONS[side_index]
    robot.semantic_label = "floating_leap_hand"

    stage_links = HAND_STAGE_LINK_NAMES_BY_SIDE[side_index]
    stage_joints = HAND_STAGE_JOINT_NAMES_BY_SIDE[side_index]
    mount = gobot.create_node("Link3D", stage_links[0])
    mount.role = gobot.LinkRole.VirtualRoot
    robot.add_child(mount)
    current_link = mount
    for dof_index, (dof_name, joint_name) in enumerate(
        zip(HAND_STAGE_DOF_NAMES, stage_joints, strict=True)
    ):
        joint = gobot.create_node("Joint3D", joint_name)
        _configure_stage_joint(joint, dof_name)
        joint.parent_link = current_link.name
        current_link.add_child(joint)
        if dof_index + 1 < len(stage_links):
            child = gobot.create_node(
                "Link3D", stage_links[dof_index + 1]
            )
            _set_virtual_link_inertia(child)
            joint.child_link = child.name
            joint.add_child(child)
            current_link = child
        else:
            joint.child_link = palm.name
            palm.reparent(joint)

    aligned_palm = _rigid_matrix(
        (0.0, 0.0, 0.0), HAND_PALM_ALIGNMENT_ROTATION
    ) @ _node_local_rigid_matrix(palm)
    palm.set_transform(
        tuple(float(value) for value in aligned_palm[:3, 3]),
        _matrix_quaternion(aligned_palm[:3, :3]),
    )
    root.add_child(robot)

    robot_nodes = _nodes_by_name(robot)
    for link_name in LEAP_CONTACT_LINK_NAMES:
        _add_coupling(
            root,
            f"{robot_name}_{link_name}_coupling",
            _path_from_root(root, robot_nodes[link_name]),
            gobot.PhysicsCouplingMode.OneWay,
        )


def _create_leap_hands(root: Any) -> None:
    for side_index, side in enumerate(HAND_SIDES):
        _create_leap_hand(root, side, side_index)


def _positive_tetrahedron(
    vertices: list[tuple[float, float, float]],
    indices: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    points = np.asarray([vertices[index] for index in indices])
    signed_volume = float(
        (points[1] - points[0])
        @ np.cross(points[2] - points[0], points[3] - points[0])
    )
    if abs(signed_volume) <= 1.0e-14:
        raise ValueError("soft package mesh contains a degenerate tetrahedron")
    if signed_volume < 0.0:
        return (indices[1], indices[0], indices[2], indices[3])
    return indices


def _soft_package_mesh(
    size: tuple[float, float, float],
    cells: tuple[int, int, int] = (6, 4, 3),
    *,
    side_rounding: float = 0.0,
) -> Any:
    cells_x, cells_y, cells_z = cells
    if min(*size, cells_x, cells_y, cells_z) <= 0:
        raise ValueError("soft package dimensions and cells must be positive")
    if not 0.0 <= side_rounding < 0.5:
        raise ValueError("soft package side rounding must be in [0, 0.5)")
    half_x, half_y, half_z = (0.5 * float(value) for value in size)
    nx = cells_x + 1
    ny = cells_y + 1

    def vertex_index(ix: int, iy: int, iz: int) -> int:
        return iz * nx * ny + iy * nx + ix

    vertices: list[tuple[float, float, float]] = []
    for iz in range(cells_z + 1):
        w = 2.0 * iz / cells_z - 1.0
        for iy in range(cells_y + 1):
            v = 2.0 * iy / cells_y - 1.0
            for ix in range(cells_x + 1):
                u = 2.0 * ix / cells_x - 1.0
                center_x = max(0.0, 1.0 - u * u)
                center_y = max(0.0, 1.0 - v * v)
                if side_rounding > 0.0:
                    # Map the regular grid to a rounded pillow volume. The
                    # middle layers carry the full footprint, while the top
                    # and bottom layers draw inward. Height also fades toward
                    # the sealed perimeter, leaving a broad, inflated center.
                    layer_side_scale = 1.0 - side_rounding * w * w
                    x = (
                        half_x
                        * u
                        * (1.0 - side_rounding * (1.0 - center_y))
                        * layer_side_scale
                    )
                    y = (
                        half_y
                        * v
                        * (1.0 - side_rounding * (1.0 - center_x))
                        * layer_side_scale
                    )
                    center_fraction = math.sqrt(center_x * center_y)
                    # Match the asymmetric film cavity: the contents have a
                    # shallow underside and most of their loft above it. This
                    # preserves clearance from both sheets at initialization.
                    bottom = -(
                        0.04 + 0.15 * center_fraction
                    ) * size[2]
                    top = (
                        0.44 + 0.40 * center_fraction
                    ) * size[2]
                    layer = 0.5 * (w + 1.0)
                    z = bottom + layer * (top - bottom)
                    vertices.append((x, y, z))
                    continue
                # Keep a rounded rectangular footprint, but form the package
                # from asymmetric top and bottom sheets.  A symmetric solid
                # pillow leaves a bowl-shaped underside when it bridges the
                # table and belt; a real filled mailer has a broad, shallow
                # contact patch and most of its loft above that patch.
                x = half_x * u * (0.94 + 0.06 * center_y)
                y = half_y * v * (0.94 + 0.06 * center_x)
                distance_from_seam = max(
                    0.0, min(1.0 - abs(u), 1.0 - abs(v))
                )
                content_fraction = min(1.0, distance_from_seam / 0.16)
                content_fraction = content_fraction * content_fraction * (
                    3.0 - 2.0 * content_fraction
                )
                crown = 1.0 - 0.05 * (0.55 * u * u + 0.45 * v * v)
                edge_band = 4.0 * content_fraction * (
                    1.0 - content_fraction
                )
                wrinkle = (
                    0.025
                    * size[2]
                    * edge_band
                    * math.sin(5.0 * math.pi * u + 3.0 * math.pi * v)
                )
                center_crease = (
                    0.025
                    * size[2]
                    * math.exp(-55.0 * u * u)
                    * max(0.0, 1.0 - 1.35 * abs(v))
                    * content_fraction
                )
                seam_half_thickness = 0.018 * size[2]
                bottom = (
                    -seam_half_thickness
                    - 0.14 * size[2] * content_fraction * crown
                    + 0.20 * wrinkle
                )
                top = (
                    seam_half_thickness
                    + 0.82 * size[2] * content_fraction * crown
                    + wrinkle
                    - center_crease
                )
                layer = 0.5 * (w + 1.0)
                z = bottom + layer * (top - bottom)
                vertices.append((x, y, z))

    tetrahedra: list[tuple[int, int, int, int]] = []
    for iz in range(cells_z):
        for iy in range(cells_y):
            for ix in range(cells_x):
                v000 = vertex_index(ix, iy, iz)
                v100 = vertex_index(ix + 1, iy, iz)
                v010 = vertex_index(ix, iy + 1, iz)
                v110 = vertex_index(ix + 1, iy + 1, iz)
                v001 = vertex_index(ix, iy, iz + 1)
                v101 = vertex_index(ix + 1, iy, iz + 1)
                v011 = vertex_index(ix, iy + 1, iz + 1)
                v111 = vertex_index(ix + 1, iy + 1, iz + 1)
                for tetrahedron in (
                    (v000, v100, v110, v111),
                    (v000, v110, v010, v111),
                    (v000, v010, v011, v111),
                    (v000, v011, v001, v111),
                    (v000, v001, v101, v111),
                    (v000, v101, v100, v111),
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


def _soft_mailer_shell_mesh(
    size: tuple[float, float, float],
    cells: tuple[int, int] = (18, 13),
) -> Any:
    cells_x, cells_y = cells
    if min(*size, cells_x, cells_y) <= 0:
        raise ValueError("soft mailer shell dimensions and cells must be positive")
    half_x, half_y, height = (
        0.5 * float(size[0]),
        0.5 * float(size[1]),
        float(size[2]),
    )
    nx = cells_x + 1
    ny = cells_y + 1
    layer_stride = nx * ny

    def vertex_index(layer: int, ix: int, iy: int) -> int:
        return layer * layer_stride + iy * nx + ix

    vertices: list[tuple[float, float, float]] = []
    for layer in range(2):
        for iy in range(ny):
            v = 2.0 * iy / cells_y - 1.0
            for ix in range(nx):
                u = 2.0 * ix / cells_x - 1.0
                center_x = max(0.0, 1.0 - u * u)
                center_y = max(0.0, 1.0 - v * v)
                x = half_x * u * (0.93 + 0.07 * center_y)
                y = half_y * v * (0.93 + 0.07 * center_x)
                seam_distance = max(
                    0.0, min(1.0 - abs(u), 1.0 - abs(v))
                )
                fill = min(1.0, seam_distance / 0.24)
                fill = fill * fill * (3.0 - 2.0 * fill)
                edge_band = 4.0 * fill * (1.0 - fill)
                # A heat-sealed perimeter starts coplanar. Film irregularity
                # belongs on the shoulder just inside that flange; placing the
                # largest opposite-signed wave directly on the free seam made
                # it curl into a tube as soon as gravity loaded the shell.
                seam_wave = (
                    0.003
                    * height
                    * edge_band
                    * math.sin(4.0 * math.pi * u - 3.0 * math.pi * v)
                )
                if layer == 0:
                    bottom_wrinkle = (
                        0.018
                        * height
                        * edge_band
                        * math.sin(5.0 * math.pi * u + 2.0 * math.pi * v)
                    )
                    z = (
                        -0.025 * height
                        - 0.245 * height * fill
                        + bottom_wrinkle
                        - seam_wave
                    )
                else:
                    diagonal_crease = (
                        0.085
                        * height
                        * math.exp(-90.0 * (u + 0.48 * v - 0.18) ** 2)
                        * fill
                    )
                    cross_crease = (
                        0.050
                        * height
                        * math.exp(-110.0 * (u - 0.60 * v + 0.28) ** 2)
                        * fill
                    )
                    top_wrinkle = (
                        0.040
                        * height
                        * edge_band
                        * math.sin(5.0 * math.pi * u + 3.0 * math.pi * v)
                    )
                    crown = 0.96 - 0.07 * (0.55 * u * u + 0.45 * v * v)
                    z = (
                        0.025 * height
                        + 0.785 * height * fill * crown
                        + top_wrinkle
                        - diagonal_crease
                        - cross_crease
                        + seam_wave
                    )
                vertices.append((x, y, z))

    triangles: list[tuple[int, int, int]] = []
    for iy in range(cells_y):
        for ix in range(cells_x):
            bottom_00 = vertex_index(0, ix, iy)
            bottom_10 = vertex_index(0, ix + 1, iy)
            bottom_01 = vertex_index(0, ix, iy + 1)
            bottom_11 = vertex_index(0, ix + 1, iy + 1)
            top_00 = vertex_index(1, ix, iy)
            top_10 = vertex_index(1, ix + 1, iy)
            top_01 = vertex_index(1, ix, iy + 1)
            top_11 = vertex_index(1, ix + 1, iy + 1)
            # Top faces +Z; bottom faces -Z.
            triangles.extend(
                (
                    (top_00, top_10, top_11),
                    (top_00, top_11, top_01),
                    (bottom_00, bottom_11, bottom_10),
                    (bottom_00, bottom_01, bottom_11),
                )
            )

    perimeter = (
        tuple((ix, 0) for ix in range(nx))
        + tuple((cells_x, iy) for iy in range(1, ny))
        + tuple((ix, cells_y) for ix in range(cells_x - 1, -1, -1))
        + tuple((0, iy) for iy in range(cells_y - 1, 0, -1))
    )
    for edge_index, (ix, iy) in enumerate(perimeter):
        next_ix, next_iy = perimeter[(edge_index + 1) % len(perimeter)]
        bottom_a = vertex_index(0, ix, iy)
        bottom_b = vertex_index(0, next_ix, next_iy)
        top_a = vertex_index(1, ix, iy)
        top_b = vertex_index(1, next_ix, next_iy)
        triangles.extend(
            ((bottom_a, bottom_b, top_b), (bottom_a, top_b, top_a))
        )

    mesh = gobot.SurfaceMesh()
    mesh.vertices = vertices
    mesh.triangles = triangles
    mesh.validate()
    return mesh


def _create_warehouse_frame(root: Any) -> None:
    robot = gobot.create_node("Robot3D", "warehouse_frame")
    robot.mode = gobot.RobotMode.Assembly
    frame = gobot.create_node("Link3D", "frame")
    _set_box_inertia(frame, 300.0, (3.0, 1.0, 0.8))

    _add_box_geometry(
        frame,
        "factory_floor",
        (6.0, 4.0, 0.08),
        (0.0, 0.0, -0.04),
        (0.30, 0.33, 0.36, 1.0),
        sliding_friction=0.85,
    )

    _add_box_geometry(
        frame,
        "worktable_surface",
        (WORKTABLE_LENGTH, WORKTABLE_DEPTH, WORKTABLE_THICKNESS),
        (
            WORKTABLE_CENTER_X,
            WORKTABLE_CENTER_Y,
            WORKTABLE_TOP_Z - 0.5 * WORKTABLE_THICKNESS,
        ),
        (0.40, 0.43, 0.44, 1.0),
        sliding_friction=WORKTABLE_SLIDING_FRICTION,
    )
    _add_visual(
        frame,
        "incoming_package_lane",
        (0.82, 0.42, 0.006),
        (-0.70, 0.015, WORKTABLE_TOP_Z + 0.003),
        (0.29, 0.32, 0.33, 1.0),
    )
    worktable_near_y = WORKTABLE_CENTER_Y - 0.5 * WORKTABLE_DEPTH
    worktable_far_y = WORKTABLE_CENTER_Y + 0.5 * WORKTABLE_DEPTH
    _add_visual(
        frame,
        "worktable_front_apron",
        (WORKTABLE_LENGTH, 0.055, 0.25),
        (WORKTABLE_CENTER_X, worktable_near_y + 0.02, 0.405),
        (0.19, 0.22, 0.24, 1.0),
    )
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            _add_visual(
                frame,
                "worktable_leg_"
                f"{'left' if x_sign < 0.0 else 'right'}_"
                f"{'near' if y_sign < 0.0 else 'far'}",
                (0.09, 0.09, 0.48),
                (
                    WORKTABLE_CENTER_X
                    + x_sign * (0.5 * WORKTABLE_LENGTH - 0.14),
                    WORKTABLE_CENTER_Y + y_sign * 0.24,
                    0.24,
                ),
                (0.22, 0.25, 0.27, 1.0),
            )

    rail_offset = 0.5 * BELT_WIDTH + 0.045
    near_rail_y = BELT_CENTER_Y - rail_offset
    far_rail_y = BELT_CENTER_Y + rail_offset
    belt_start = BELT_CENTER_X - 0.5 * BELT_FRAME_LENGTH
    belt_end = BELT_CENTER_X + 0.5 * BELT_FRAME_LENGTH
    table_start = WORKTABLE_CENTER_X - 0.5 * WORKTABLE_LENGTH
    table_end = WORKTABLE_CENTER_X + 0.5 * WORKTABLE_LENGTH
    rail_segments = (
        (
            "near_upstream",
            table_start - belt_start,
            0.5 * (belt_start + table_start),
            near_rail_y,
        ),
        (
            "near_downstream",
            belt_end - table_end,
            0.5 * (table_end + belt_end),
            near_rail_y,
        ),
        (
            "far_upstream",
            0.5 * BELT_FRAME_LENGTH,
            BELT_CENTER_X - 0.25 * BELT_FRAME_LENGTH,
            far_rail_y,
        ),
        (
            "far_downstream",
            0.5 * BELT_FRAME_LENGTH,
            BELT_CENTER_X + 0.25 * BELT_FRAME_LENGTH,
            far_rail_y,
        ),
    )
    for name, length, center_x, center_y in rail_segments:
        _add_box_geometry(
            frame,
            f"{name}_guide_rail",
            (length, 0.07, 0.17),
            (center_x, center_y, BELT_TOP_Z + 0.085),
            (0.17, 0.21, 0.24, 1.0),
            sliding_friction=0.65,
        )
    _add_box_geometry(
        frame,
        "end_stop",
        (0.07, BELT_WIDTH + 0.16, 0.30),
        (belt_end, BELT_CENTER_Y, BELT_TOP_Z + 0.15),
        (0.84, 0.24, 0.12, 1.0),
        sliding_friction=0.75,
    )

    for x in (-1.05, 1.05):
        for y in (
            BELT_CENTER_Y - 0.38,
            BELT_CENTER_Y + 0.38,
        ):
            _add_visual(
                frame,
                f"support_{'left' if x < 0 else 'right'}_"
                f"{'near' if y < BELT_CENTER_Y else 'far'}",
                (0.12, 0.12, 0.52),
                (x, y, 0.26),
                (0.24, 0.28, 0.31, 1.0),
            )
    _add_visual(
        frame,
        "lower_crossbeam",
        (BELT_FRAME_LENGTH, 0.10, 0.10),
        (BELT_CENTER_X, BELT_CENTER_Y, 0.20),
        (0.21, 0.25, 0.28, 1.0),
    )

    scanner_x = 1.03
    for side, sign in (("near", -1.0), ("far", 1.0)):
        _add_visual(
            frame,
            f"scanner_{side}_post",
            (0.055, 0.055, 0.72),
            (
                scanner_x,
                BELT_CENTER_Y + sign * 0.49,
                BELT_TOP_Z + 0.36,
            ),
            (0.08, 0.50, 0.62, 1.0),
        )
    _add_visual(
        frame,
        "scanner_crossbar",
        (0.065, 1.04, 0.065),
        (scanner_x, BELT_CENTER_Y, BELT_TOP_Z + 0.72),
        (0.08, 0.50, 0.62, 1.0),
    )
    _add_visual(
        frame,
        "scanner_camera",
        (0.12, 0.12, 0.08),
        (scanner_x, BELT_CENTER_Y, BELT_TOP_Z + 0.66),
        (0.04, 0.05, 0.06, 1.0),
    )

    robot.add_child(frame)
    root.add_child(robot)


def _create_conveyor(root: Any) -> None:
    conveyor = gobot.create_node("Robot3D", "conveyor")
    conveyor.mode = gobot.RobotMode.Assembly

    belt = gobot.create_node("Link3D", "belt_surface")
    belt.position = (BELT_CENTER_X, BELT_CENTER_Y, BELT_CENTER_Z)
    _set_box_inertia(
        belt,
        80.0,
        (BELT_SURFACE_LENGTH, BELT_WIDTH, BELT_THICKNESS),
    )
    _add_visual(
        belt,
        "moving_belt_visual",
        (BELT_SURFACE_LENGTH, BELT_WIDTH, BELT_THICKNESS),
        (0.0, 0.0, 0.0),
        (0.055, 0.075, 0.085, 1.0),
    )
    _add_collision(
        belt,
        "moving_belt_collision",
        (BELT_PROXY_LENGTH, BELT_WIDTH, BELT_THICKNESS),
        (0.0, 0.0, 0.0),
        # Both solvers keep belt contact normal-only. Explicit velocity-field
        # forces provide all traction, matching Newton's conveyor-force model.
        sliding_friction=1.0e-5,
    )
    marker_pitch = 0.22
    marker_count = int(math.ceil(BELT_SURFACE_LENGTH / marker_pitch))
    marker_start = -0.5 * (marker_count - 1) * marker_pitch
    for index in range(marker_count):
        _add_visual(
            belt,
            f"belt_marker_{index:02d}",
            (0.014, BELT_WIDTH - 0.035, 0.002),
            (marker_start + index * marker_pitch, 0.0, 0.021),
            (0.20, 0.24, 0.26, 1.0),
        )
    for side, sign in (("near", -1.0), ("far", 1.0)):
        _add_visual(
            belt,
            f"belt_{side}_safety_stripe",
            (BELT_SURFACE_LENGTH, 0.012, 0.003),
            (0.0, sign * (0.5 * BELT_WIDTH - 0.012), 0.022),
            (0.94, 0.70, 0.10, 1.0),
        )

    conveyor.add_child(belt)
    root.add_child(conveyor)


def _create_carton(root: Any, spec: dict[str, Any]) -> None:
    name = str(spec["name"])
    size = tuple(float(value) for value in spec["size"])
    body = gobot.create_node("RigidBody3D", name)
    body.position = spec["position"]
    body.rotation_degrees = spec["rotation_degrees"]
    body.semantic_label = "rigid_shipping_carton"
    _set_box_inertia(body, float(spec["mass"]), size)
    _add_box_geometry(
        body,
        name,
        size,
        (0.0, 0.0, 0.0),
        spec["color"],
        sliding_friction=0.82,
        collision_layer=RIGID_PACKAGE_COLLISION_LAYER,
        collision_mask=RIGID_PACKAGE_COLLISION_MASK,
    )

    tape_width = min(0.055, 0.22 * size[1])
    _add_visual(
        body,
        name + "_packing_tape",
        (size[0] + 0.002, tape_width, 0.004),
        (0.0, 0.0, 0.5 * size[2] + 0.002),
        (0.91, 0.78, 0.48, 1.0),
    )
    _add_visual(
        body,
        name + "_shipping_label",
        (0.11, 0.004, min(0.07, 0.55 * size[2])),
        (0.0, 0.5 * size[1] + 0.002, 0.02),
        (0.88, 0.91, 0.90, 1.0),
    )
    root.add_child(body)


def _create_soft_package(root: Any, spec: dict[str, Any]) -> None:
    body = gobot.create_node("DeformableBody3D", spec["name"])
    if spec.get("model", "volumetric") == "thin_shell":
        body.model = gobot.DeformableBodyModel.ThinShell
        body.surface_mesh = _soft_mailer_shell_mesh(
            spec["size"], spec["cells"]
        )
        body.thickness = spec["thickness"]
        body.bending_stiffness = spec["bending_stiffness"]
        body.self_collision_enabled = True
    else:
        body.model = gobot.DeformableBodyModel.Volumetric
        body.mesh = _soft_package_mesh(
            spec["size"],
            spec["cells"],
            side_rounding=float(spec.get("side_rounding", 0.0)),
        )
        body.self_collision_enabled = False
    body.position = spec["position"]
    body.rotation_degrees = spec["rotation_degrees"]
    body.density = spec["density"]
    body.young_modulus = spec["young_modulus"]
    body.poisson_ratio = spec["poisson_ratio"]
    body.damping = spec["damping"]
    # The symmetric filter matrix keeps the moving hand proxies away from the
    # rigid table while preserving hand-package and table-package contact.
    body.collision_layer = DEFORMABLE_COLLISION_LAYER
    body.collision_mask = DEFORMABLE_COLLISION_MASK
    body.debug_surface_color = spec["color"]
    body.debug_wireframe_visible = False
    body.visible = bool(spec.get("visible", True))
    body.semantic_label = "deformable_shipping_package"
    root.add_child(body)


def _add_coupling(
    root: Any,
    name: str,
    target_body_path: str,
    mode: Any,
    *,
    force_scale: float = 1.0,
    torque_scale: float = 1.0,
) -> None:
    coupling = gobot.create_node("PhysicsCoupling", name)
    coupling.target_body_path = target_body_path
    coupling.mode = mode
    coupling.force_scale = force_scale
    coupling.torque_scale = torque_scale
    root.add_child(coupling)


def create_scene() -> Any:
    root = gobot.create_node("Node3D", "conveyor_packages")
    _create_warehouse_frame(root)
    _create_conveyor(root)
    _create_leap_hands(root)
    for spec in RIGID_BOX_SPECS:
        _create_carton(root, spec)
    for spec in SOFT_PACKAGE_SPECS:
        _create_soft_package(root, spec)

    _add_coupling(
        root,
        "belt_surface_coupling",
        "../conveyor/belt_surface",
        gobot.PhysicsCouplingMode.OneWay,
    )
    _add_coupling(
        root,
        "warehouse_frame_coupling",
        "../warehouse_frame/frame",
        gobot.PhysicsCouplingMode.OneWay,
    )
    for spec in RIGID_BOX_SPECS:
        name = str(spec["name"])
        _add_coupling(
            root,
            name + "_coupling",
            "../" + name,
            gobot.PhysicsCouplingMode.TwoWay,
        )
    return root


def _finalize_scene(scene_path: Path) -> None:
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    nodes = scene.get("__NODES__", [])
    resources = scene.get("__EXT_RESOURCES__", [])
    if not isinstance(nodes, list) or not isinstance(resources, list):
        raise RuntimeError("generated conveyor scene has no node/resource table")

    roots = [entry for entry in nodes if int(entry.get("parent", -2)) == -1]
    if len(roots) != 1:
        raise RuntimeError("generated conveyor scene has no unique root")
    resources.insert(
        0,
        {
            "__ID__": PLAY_SCRIPT_RESOURCE_ID,
            "__PATH__": PLAY_SCRIPT_PATH,
            "__TYPE__": "PythonScript",
        },
    )
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

    subresources = scene.get("__SUB_RESOURCES__", [])
    if not isinstance(subresources, list):
        raise RuntimeError("generated conveyor scene has no subresource table")
    type_counts: dict[str, int] = {}
    replacements: dict[str, str] = {}
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
    source_assets = HERE / "assets"
    missing_hands = [
        path
        for path in (
            LEAP_ASSET_ROOT / "left_hand.xml",
            LEAP_ASSET_ROOT / "right_hand.xml",
        )
        if not path.is_file()
    ]
    if missing_hands:
        raise FileNotFoundError(f"LEAP Hand assets are missing: {missing_hands}")
    asset_link = output_dir / "assets"
    if output_dir != HERE and not asset_link.exists():
        relative_target = os.path.relpath(source_assets, output_dir)
        asset_link.symlink_to(relative_target, target_is_directory=True)
    elif asset_link.resolve() != source_assets.resolve():
        raise RuntimeError(
            f"conveyor asset path does not reference {source_assets}: "
            f"{asset_link}"
        )
    if output_dir == HERE:
        return
    for name in (
        "README.md",
        "build_scene.py",
        "conveyor_forces.py",
        "conveyor_packages_batch.py",
        "conveyor_packages_play.py",
        "conveyor_profile.py",
        "project.gobot",
    ):
        source = HERE / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)


def build_scene(output_dir: Path = HERE) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _stage_project(output_dir)
    gobot.app.context().set_project_path(str(output_dir))
    destination = output_dir / SCENE_NAME
    gobot.save_scene(create_scene(), "res://" + SCENE_NAME)
    _finalize_scene(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    print(build_scene(args.output_dir))


if __name__ == "__main__":
    main()
