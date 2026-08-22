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

OPENARM_ASSET_ROOT = HERE / "assets" / "openarm_description"
OPENARM_URDF_RESOURCE = (
    "res://assets/openarm_description/openarm_v20_bimanual.urdf"
)
ALLEGRO_RESOURCE_ROOT = "res://assets/wonik_allegro"
OPENARM_ROBOT_NAME = "openarm_bimanual"
ALLEGRO_HAND_SCALE = 1.50
ALLEGRO_HAND_WRIST_OFFSET = 0.120
ALLEGRO_FINGER_POSE = (0.0, 0.24, 0.34, 0.26)
ALLEGRO_THUMB_POSE = (0.52, 0.18, 0.30, 0.24)
# These bounds mirror the Allegro meshes shipped with this example.  Adjacent
# segments are merged into two oriented proxies per finger.  The outer shell
# absorbs the external-proxy coupling tolerance so the rendered fingertips do
# not become visible through a deformable during release.
ALLEGRO_VISUAL_PROXY_MARGIN = 0.006
ALLEGRO_VISUAL_PROXY_GROUPS = (
    ("root", ("base", "proximal"), "proximal"),
    ("tip", ("medial", "distal", "tip"), "medial"),
)
ALLEGRO_FINGER_MESH_BOUNDS = {
    "base": ((-0.0098, -0.01345, 0.0), (0.0098, 0.01345, 0.0220)),
    "proximal": ((-0.0098, -0.01355, -0.0088), (0.0098, 0.01345, 0.0628)),
    "medial": ((-0.0098, -0.01355, -0.0063), (0.0098, 0.01345, 0.0472)),
    "distal": ((-0.0098, -0.01345, -0.0063), (0.0098, 0.01345, 0.0157)),
    "tip": ((-0.0120, -0.0120, -0.0110), (0.0120, 0.0120, 0.0120)),
}
ALLEGRO_THUMB_MESH_BOUNDS = {
    "proximal": ((-0.0098, -0.01345, -0.000168), (0.0098, 0.01345, 0.021833)),
    "medial": ((-0.0098, -0.01345, -0.0088), (0.0098, 0.01355, 0.0577)),
    "distal": ((-0.0098, -0.01345, -0.0088), (0.0098, 0.01355, 0.0313)),
    "tip": ((-0.0120, -0.0120, -0.0110), (0.0120, 0.0120, 0.0120)),
}
ALLEGRO_THUMB_BASE_MESH_BOUNDS = {
    "left": ((-0.0392, -0.0080, -0.0415), (0.0055, 0.0260, 0.0080)),
    "right": ((-0.0392, -0.0080, -0.0080), (0.0055, 0.0260, 0.0415)),
}
# One pad spans the scaled Allegro palm and fingers.  The longer footprint keeps
# the visible down-facing hand in contact throughout the sweep instead of
# shearing only the mailer's leading edge and then slipping off it.
HAND_PUSH_PAD_SIZE = (0.180, 0.025, 0.260)
HAND_DOWN_ANGLE_DEGREES = 60.0
_HAND_DOWN = math.sin(math.radians(HAND_DOWN_ANGLE_DEGREES))
_HAND_FORWARD = math.cos(math.radians(HAND_DOWN_ANGLE_DEGREES))
HAND_PUSH_PAD_REFERENCE_ROTATION = np.asarray(
    (
        (0.0, -1.0, 0.0),
        (_HAND_DOWN, 0.0, -_HAND_FORWARD),
        (_HAND_FORWARD, 0.0, _HAND_DOWN),
    ),
    dtype=np.float64,
)
HAND_PRESS_ANGLE_DEGREES = 90.0
_HAND_PRESS_DOWN = math.sin(math.radians(HAND_PRESS_ANGLE_DEGREES))
_HAND_PRESS_FORWARD = math.cos(math.radians(HAND_PRESS_ANGLE_DEGREES))
HAND_PUSH_PAD_WORLD_ROTATION = np.asarray(
    (
        (1.0, 0.0, 0.0),
        (0.0, _HAND_PRESS_FORWARD, _HAND_PRESS_DOWN),
        (0.0, -_HAND_PRESS_DOWN, _HAND_PRESS_FORWARD),
    ),
    dtype=np.float64,
)
HAND_PUSH_PAD_LOCAL_ROTATION = (
    HAND_PUSH_PAD_REFERENCE_ROTATION.T @ HAND_PUSH_PAD_WORLD_ROTATION
)
HAND_PUSH_PAD_WORLD_OFFSETS = (
    (0.10, 0.080, -0.110),
    (-0.10, 0.080, -0.110),
)
HAND_VISUAL_WORLD_ADJUSTMENT = np.asarray(
    (
        (-1.0, 0.0, 0.0),
        (0.0, 0.5, -math.sqrt(3.0) / 2.0),
        (0.0, -math.sqrt(3.0) / 2.0, -0.5),
    ),
    dtype=np.float64,
)
HAND_VISUAL_LOCAL_ADJUSTMENT = (
    HAND_PUSH_PAD_REFERENCE_ROTATION.T
    @ HAND_VISUAL_WORLD_ADJUSTMENT
    @ HAND_PUSH_PAD_REFERENCE_ROTATION
)
RIGID_COLLISION_LAYER = 0b0001
HAND_COLLISION_LAYER = 0b0010
DEFORMABLE_COLLISION_LAYER = 0b0100
RIGID_COLLISION_MASK = RIGID_COLLISION_LAYER | DEFORMABLE_COLLISION_LAYER
HAND_COLLISION_MASK = DEFORMABLE_COLLISION_LAYER
DEFORMABLE_COLLISION_MASK = RIGID_COLLISION_LAYER | HAND_COLLISION_LAYER
HAND_PUSH_PAD_LOCAL_OFFSETS = tuple(
    tuple(
        float(value)
        for value in HAND_PUSH_PAD_REFERENCE_ROTATION.T
        @ np.asarray(offset, dtype=np.float64)
    )
    for offset in HAND_PUSH_PAD_WORLD_OFFSETS
)
# The heel/curled-finger edge of each down-facing hand catches the rear wall of
# the parcel.  A normal contact at this edge translates a volumetric mailer;
# relying on top-surface friction alone only shears the FEM volume before it
# elastically snaps back.
HAND_SWEEP_EDGE_SIZE = (0.180, 0.025, 0.105)
HAND_SWEEP_EDGE_WORLD_OFFSETS = (
    (0.10, -0.080, -0.200),
    (-0.10, -0.080, -0.200),
)
HAND_SWEEP_EDGE_LOCAL_OFFSETS = tuple(
    tuple(
        float(value)
        for value in HAND_PUSH_PAD_REFERENCE_ROTATION.T
        @ np.asarray(offset, dtype=np.float64)
    )
    for offset in HAND_SWEEP_EDGE_WORLD_OFFSETS
)
HAND_SWEEP_EDGE_LOCAL_ROTATION = HAND_PUSH_PAD_REFERENCE_ROTATION.T
ARM_SIDES = ("left", "right")
ARM_BASE_LINK_NAMES = tuple(
    f"openarm_{side}_base_link" for side in ARM_SIDES
)
ARM_JOINT_NAMES_BY_SIDE = tuple(
    tuple(f"openarm_{side}_joint{index}" for index in range(1, 8))
    + tuple(f"openarm_{side}_finger_joint{index}" for index in range(1, 3))
    for side in ARM_SIDES
)
ARM_LINK_NAMES_BY_SIDE = tuple(
    (f"openarm_{side}_base_link",)
    + tuple(f"openarm_{side}_link{index}" for index in range(1, 7))
    + (
        f"openarm_{side}_ee_base_link",
        f"openarm_{side}_ee_link1",
        f"openarm_{side}_ee_link2",
    )
    for side in ARM_SIDES
)
OPENARM_PROXY_LINK_NAMES = tuple(
    link_name
    for side in ARM_SIDES
    for link_name in (
        f"openarm_{side}_ee_base_link",
        f"openarm_{side}_ee_link1",
        f"openarm_{side}_ee_link2",
    )
)
MANIPULATION_STATION_X = 0.16
# Keep the shoulder bridge behind the work surface while placing the palms over
# the rear half of the target mailer at the grip pose.  The previous -0.34 m
# placement left the down-facing palms behind the package, so most of the arm
# sweep happened before frictional contact began.
OPENARM_ROOT_POSITION = (MANIPULATION_STATION_X, -0.19, 0.05)
OPENARM_ROOT_YAW_DEGREES = 90.0
# Offline IK keeps both elbows raised behind the static sorting table while the
# hands frame the blue mailer. The robot root is yawed 90 degrees so its shared
# shoulder follows the conveyor. The wrists sweep the package from the table
# onto the front belt before retracting.
ARM_INITIAL_POSES = (
    (
        -0.4248239998980243,
        -1.0884326790574228,
        1.4229884665739385,
        2.247245233442501,
        -1.5707899999999997,
        -0.0755760882846291,
        1.1553548821422193,
    ),
    (
        0.4248295822718003,
        1.0884330176982384,
        -1.4229946651516743,
        2.2472452224918924,
        1.5707899999999997,
        0.07557404197213391,
        -1.1553539501786063,
    ),
)
ARM_JOINT_AXES_BY_SIDE = (
    (
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
    (
        (0.0, -1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
)
FINGER_JOINT_AXES = ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
FINGER_OPEN_POSITIONS = (0.72, -0.72)
ARM_STIFFNESS = (320.0, 300.0, 230.0, 210.0, 85.0, 70.0, 42.0)
ARM_DAMPING = (30.0, 28.0, 23.0, 21.0, 9.0, 7.0, 5.0)
GRIPPER_STIFFNESS = 150.0
GRIPPER_DAMPING = 8.0
GRIPPER_FRICTION = 2.5

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
WORKTABLE_DEPTH = 0.66
WORKTABLE_THICKNESS = 0.08
WORKTABLE_CENTER_X = -0.15
WORKTABLE_CENTER_Y = -0.125
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
        "position": (-0.86, 0.015, WORKTABLE_TOP_Z + 0.092),
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
        "size": (0.44, 0.32, 0.10),
        "position": (
            MANIPULATION_STATION_X,
            0.045,
            WORKTABLE_TOP_Z + 0.235,
        ),
        "rotation_degrees": (4.0, -6.0, 2.0),
        # The numerical film is thicker than real polyethylene so IPC keeps a
        # robust contact shell. Its low density keeps the skin much lighter
        # than the hidden contents.
        "density": 260.0,
        "young_modulus": 1.8e5,
        "poisson_ratio": 0.36,
        "damping": 7.0,
        "thickness": 1.2e-3,
        "bending_stiffness": 2.0e-4,
        "cells": (18, 13),
        "color": (0.10, 0.36, 0.64, 1.0),
        "visible": True,
    },
    {
        "name": "soft_mailer_blue_fill",
        "model": "volumetric",
        "size": (0.35, 0.235, 0.060),
        "position": (
            MANIPULATION_STATION_X,
            0.045,
            # Center the contents inside the asymmetric film cavity. The
            # mailer's top is intentionally fuller than its bottom, so the
            # core sits 10 mm below the shell origin to leave IPC clearance.
            WORKTABLE_TOP_Z + 0.225,
        ),
        "rotation_degrees": (4.0, -6.0, 2.0),
        "density": 295.5586864819189,
        "young_modulus": 8.0e3,
        "poisson_ratio": 0.43,
        "damping": 7.0,
        "cells": (8, 6, 3),
        "color": (0.04, 0.08, 0.12, 0.0),
        "visible": False,
    },
    {
        "name": "soft_pouch_yellow",
        "model": "volumetric",
        "size": (0.30, 0.23, 0.09),
        # This second deformable package queues upstream on the left side of
        # the static sorting table rather than starting on the conveyor.
        "position": (
            -0.42,
            0.055,
            WORKTABLE_TOP_Z + 0.016,
        ),
        "rotation_degrees": (0.0, 0.0, 8.0),
        "density": 208.72358069160808,
        "young_modulus": 3.5e4,
        "poisson_ratio": 0.43,
        "damping": 0.70,
        "cells": (9, 7, 4),
        "color": (0.88, 0.56, 0.08, 1.0),
        "visible": True,
    },
)


def _nodes_by_name(root: Any) -> dict[str, Any]:
    nodes: dict[str, Any] = {}
    pending = [root]
    while pending:
        node = pending.pop()
        if node.name in nodes:
            raise RuntimeError(
                f"OpenArm source has duplicate node name {node.name!r}"
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


def _bake_revolute_joint(
    nodes: dict[str, Any],
    joint_name: str,
    initial_position: float,
    stiffness: float,
    damping: float,
    effort_floor: float = 1.0,
) -> None:
    joint = nodes[joint_name]
    lower_limit = float(joint.lower_limit) - initial_position
    upper_limit = float(joint.upper_limit) - initial_position
    effort_limit = max(float(joint.effort_limit), float(effort_floor))
    joint.drive_mode = gobot.JointDriveMode.Position
    joint.drive_stiffness = stiffness
    joint.drive_damping = damping
    joint.damping = 0.02
    joint.lower_limit = lower_limit
    joint.upper_limit = upper_limit
    joint.control_lower_limit = lower_limit
    joint.control_upper_limit = upper_limit
    joint.force_lower_limit = -effort_limit
    joint.force_upper_limit = effort_limit
    joint.initial_position = 0.0
    joint.joint_position = 0.0
    child = nodes[str(joint.child_link)]
    child.set_transform(
        (0.0, 0.0, 0.0),
        _axis_angle_quaternion(tuple(joint.axis), initial_position),
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
) -> None:
    _add_visual(parent, name + "_visual", size, position, color)
    _add_collision(
        parent,
        name + "_collision",
        size,
        position,
        sliding_friction=sliding_friction,
    )


def _new_visual_frame(
    parent: Any,
    name: str,
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    scale: float = 1.0,
) -> Any:
    frame = gobot.create_node("Node3D", name)
    frame.set_transform(position, orientation)
    frame.scale = (scale, scale, scale)
    parent.add_child(frame)
    return frame


def _reparent_allegro_visual(
    source_nodes: dict[str, Any],
    source_link_name: str,
    parent: Any,
    target_name: str,
    color: tuple[float, float, float, float],
) -> Any:
    visuals = [
        child
        for child in source_nodes[source_link_name].children
        if child.type_name == "MeshInstance3D"
    ]
    if len(visuals) != 1:
        raise RuntimeError(
            f"Allegro link {source_link_name!r} must have exactly one visual"
        )
    visual = visuals[0]
    visual.reparent(parent)
    visual.name = target_name
    visual.surface_color = color
    visual.semantic_label = target_name
    return visual


def _allegro_visual_proxy_geometry(
    visual: Any,
    link_from_hand: np.ndarray,
    hand_from_segment: np.ndarray,
    mesh_bounds: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    visual_scale = np.asarray(visual.scale, dtype=np.float64)
    if not np.allclose(visual_scale, 1.0, atol=1.0e-12):
        raise RuntimeError(
            f"Allegro visual {visual.name!r} has unsupported local scale"
        )

    hand_from_visual = hand_from_segment @ _node_local_rigid_matrix(visual)
    link_rotation = link_from_hand[:3, :3] @ hand_from_visual[:3, :3]
    link_translation = link_from_hand[:3, 3] + link_from_hand[
        :3, :3
    ] @ (ALLEGRO_HAND_SCALE * hand_from_visual[:3, 3])

    lower = np.asarray(mesh_bounds[0], dtype=np.float64)
    upper = np.asarray(mesh_bounds[1], dtype=np.float64)
    center = 0.5 * (lower + upper)
    size = ALLEGRO_HAND_SCALE * (upper - lower)
    position = link_translation + link_rotation @ (
        ALLEGRO_HAND_SCALE * center
    )
    return position, link_rotation, size


def _add_merged_allegro_visual_proxy(
    driven_link: Any,
    target_name: str,
    geometries: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
    reference_index: int,
) -> None:
    if not geometries or not 0 <= reference_index < len(geometries):
        raise ValueError("Allegro proxy group requires a valid reference")
    signs = np.asarray(
        tuple(
            (x_sign, y_sign, z_sign)
            for x_sign in (-1.0, 1.0)
            for y_sign in (-1.0, 1.0)
            for z_sign in (-1.0, 1.0)
        ),
        dtype=np.float64,
    )
    corners = []
    for position, rotation, size in geometries:
        corners.append(position + (rotation @ (0.5 * size * signs).T).T)
    points = np.concatenate(tuple(corners), axis=0)
    rotation = geometries[reference_index][1]
    aligned = (rotation.T @ points.T).T
    lower = aligned.min(axis=0)
    upper = aligned.max(axis=0)
    position = rotation @ (0.5 * (lower + upper))
    size = upper - lower + 2.0 * ALLEGRO_VISUAL_PROXY_MARGIN
    _add_collision(
        driven_link,
        target_name + "_proxy",
        tuple(float(value) for value in size),
        tuple(float(value) for value in position),
        sliding_friction=GRIPPER_FRICTION,
        orientation=_matrix_quaternion(rotation),
        collision_layer=HAND_COLLISION_LAYER,
        collision_mask=HAND_COLLISION_MASK,
    )


def _add_fixed_allegro_joint(
    source_nodes: dict[str, Any],
    parent: Any,
    target_prefix: str,
    source_joint_name: str,
    position: float,
) -> Any:
    source_joint = source_nodes[source_joint_name]
    origin = gobot.create_node(
        "Node3D", f"{target_prefix}_{source_joint_name}_origin"
    )
    origin.position = tuple(float(value) for value in source_joint.position)
    origin.rotation_degrees = tuple(
        float(value) for value in source_joint.rotation_degrees
    )
    parent.add_child(origin)
    pose = _new_visual_frame(
        origin,
        f"{target_prefix}_{source_joint_name}_pose",
        orientation=_axis_angle_quaternion(
            tuple(float(value) for value in source_joint.axis), position
        ),
    )
    return pose


def _add_allegro_finger_visuals(
    source_nodes: dict[str, Any],
    parent: Any,
    driven_link: Any,
    link_from_hand: np.ndarray,
    side: str,
    finger_prefix: str,
) -> None:
    target_prefix = f"openarm_{side}_allegro_{finger_prefix}"
    segment_names = ("base", "proximal", "medial", "distal")
    current = parent
    hand_from_segment = np.eye(4, dtype=np.float64)
    proxy_geometries: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for joint_index, (segment_name, position) in enumerate(
        zip(segment_names, ALLEGRO_FINGER_POSE, strict=True)
    ):
        source_joint = source_nodes[f"{finger_prefix}j{joint_index}"]
        hand_from_segment = (
            hand_from_segment
            @ _node_local_rigid_matrix(source_joint)
            @ _rigid_matrix(
                (0.0, 0.0, 0.0),
                _axis_angle_matrix(
                    tuple(float(value) for value in source_joint.axis),
                    position,
                ),
            )
        )
        current = _add_fixed_allegro_joint(
            source_nodes,
            current,
            target_prefix,
            f"{finger_prefix}j{joint_index}",
            position,
        )
        visual = _reparent_allegro_visual(
            source_nodes,
            f"{finger_prefix}_{segment_name}",
            current,
            f"{target_prefix}_{segment_name}",
            (0.055, 0.060, 0.065, 1.0),
        )
        proxy_geometries[segment_name] = _allegro_visual_proxy_geometry(
            visual,
            link_from_hand,
            hand_from_segment,
            ALLEGRO_FINGER_MESH_BOUNDS[segment_name],
        )
    visual = _reparent_allegro_visual(
        source_nodes,
        f"{finger_prefix}_tip",
        current,
        f"{target_prefix}_tip",
        (0.52, 0.55, 0.56, 1.0),
    )
    proxy_geometries["tip"] = _allegro_visual_proxy_geometry(
        visual,
        link_from_hand,
        hand_from_segment,
        ALLEGRO_FINGER_MESH_BOUNDS["tip"],
    )
    for group_name, segment_group, reference_segment in (
        ALLEGRO_VISUAL_PROXY_GROUPS
    ):
        _add_merged_allegro_visual_proxy(
            driven_link,
            f"{target_prefix}_{group_name}",
            tuple(proxy_geometries[name] for name in segment_group),
            segment_group.index(reference_segment),
        )


def _add_allegro_thumb_visuals(
    source_nodes: dict[str, Any],
    parent: Any,
    driven_link: Any,
    link_from_hand: np.ndarray,
    side: str,
) -> None:
    target_prefix = f"openarm_{side}_allegro_thumb"
    segment_names = ("base", "proximal", "medial", "distal")
    current = parent
    hand_from_segment = np.eye(4, dtype=np.float64)
    proxy_geometries: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for joint_index, (segment_name, position) in enumerate(
        zip(segment_names, ALLEGRO_THUMB_POSE, strict=True)
    ):
        source_joint = source_nodes[f"thj{joint_index}"]
        hand_from_segment = (
            hand_from_segment
            @ _node_local_rigid_matrix(source_joint)
            @ _rigid_matrix(
                (0.0, 0.0, 0.0),
                _axis_angle_matrix(
                    tuple(float(value) for value in source_joint.axis),
                    position,
                ),
            )
        )
        current = _add_fixed_allegro_joint(
            source_nodes,
            current,
            target_prefix,
            f"thj{joint_index}",
            position,
        )
        visual = _reparent_allegro_visual(
            source_nodes,
            f"th_{segment_name}",
            current,
            f"{target_prefix}_{segment_name}",
            (0.055, 0.060, 0.065, 1.0),
        )
        mesh_bounds = (
            ALLEGRO_THUMB_BASE_MESH_BOUNDS[side]
            if segment_name == "base"
            else ALLEGRO_THUMB_MESH_BOUNDS[segment_name]
        )
        proxy_geometries[segment_name] = _allegro_visual_proxy_geometry(
            visual,
            link_from_hand,
            hand_from_segment,
            mesh_bounds,
        )
    visual = _reparent_allegro_visual(
        source_nodes,
        "th_tip",
        current,
        f"{target_prefix}_tip",
        (0.52, 0.55, 0.56, 1.0),
    )
    proxy_geometries["tip"] = _allegro_visual_proxy_geometry(
        visual,
        link_from_hand,
        hand_from_segment,
        ALLEGRO_THUMB_MESH_BOUNDS["tip"],
    )
    for group_name, segment_group, reference_segment in (
        ALLEGRO_VISUAL_PROXY_GROUPS
    ):
        _add_merged_allegro_visual_proxy(
            driven_link,
            f"{target_prefix}_{group_name}",
            tuple(proxy_geometries[name] for name in segment_group),
            segment_group.index(reference_segment),
        )


def _add_robot_hand(
    nodes: dict[str, Any], side: str, side_index: int
) -> None:
    palm = nodes[f"openarm_{side}_ee_base_link"]
    for suffix in ("ee_base_link", "ee_link1", "ee_link2"):
        imported_visual = nodes.get(f"openarm_{side}_{suffix}_visual")
        if imported_visual is not None:
            imported_visual.visible = False

    source_scene = gobot.load_scene(
        f"{ALLEGRO_RESOURCE_ROOT}/{side}_hand.xml"
    )
    source_nodes = _nodes_by_name(source_scene.root)
    alignment_offset = (0.0, 0.0, -ALLEGRO_HAND_WRIST_OFFSET)
    source_alignment = _axis_angle_matrix((1.0, 0.0, 0.0), math.pi)
    alignment_rotation = HAND_VISUAL_LOCAL_ADJUSTMENT @ source_alignment
    alignment = _rigid_matrix(alignment_offset, alignment_rotation)
    palm_frame = _new_visual_frame(
        palm,
        f"openarm_{side}_allegro_hand",
        position=alignment_offset,
        orientation=_matrix_quaternion(alignment_rotation),
        scale=ALLEGRO_HAND_SCALE,
    )
    _reparent_allegro_visual(
        source_nodes,
        "palm",
        palm_frame,
        f"openarm_{side}_allegro_palm",
        (0.055, 0.060, 0.065, 1.0),
    )

    for driven_link_index, group_name in ((1, "fingers"), (2, "thumb")):
        driven_link = nodes[f"openarm_{side}_ee_link{driven_link_index}"]
        source_joint = nodes[
            f"openarm_{side}_finger_joint{driven_link_index}"
        ]
        link_rotation = _axis_angle_matrix(
            tuple(float(value) for value in source_joint.axis),
            FINGER_OPEN_POSITIONS[side_index],
        )
        link_reference = _rigid_matrix(
            tuple(float(value) for value in source_joint.position),
            link_rotation,
        )
        relative = np.linalg.inv(link_reference) @ alignment
        group_frame = _new_visual_frame(
            driven_link,
            f"openarm_{side}_allegro_{group_name}_group",
            position=tuple(float(value) for value in relative[:3, 3]),
            orientation=_matrix_quaternion(relative[:3, :3]),
            scale=ALLEGRO_HAND_SCALE,
        )
        if group_name == "fingers":
            for finger_prefix in ("ff", "mf", "rf"):
                _add_allegro_finger_visuals(
                    source_nodes,
                    group_frame,
                    driven_link,
                    relative,
                    side,
                    finger_prefix,
                )
        else:
            _add_allegro_thumb_visuals(
                source_nodes,
                group_frame,
                driven_link,
                relative,
                side,
            )

    _add_collision(
        palm,
        f"openarm_{side}_palm_proxy",
        HAND_PUSH_PAD_SIZE,
        HAND_PUSH_PAD_LOCAL_OFFSETS[side_index],
        sliding_friction=GRIPPER_FRICTION,
        orientation=_matrix_quaternion(HAND_PUSH_PAD_LOCAL_ROTATION),
        collision_layer=HAND_COLLISION_LAYER,
        collision_mask=HAND_COLLISION_MASK,
    )
    _add_collision(
        palm,
        f"openarm_{side}_sweep_edge_proxy",
        HAND_SWEEP_EDGE_SIZE,
        HAND_SWEEP_EDGE_LOCAL_OFFSETS[side_index],
        sliding_friction=GRIPPER_FRICTION,
        orientation=_matrix_quaternion(HAND_SWEEP_EDGE_LOCAL_ROTATION),
        collision_layer=HAND_COLLISION_LAYER,
        collision_mask=HAND_COLLISION_MASK,
    )
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
) -> Any:
    cells_x, cells_y, cells_z = cells
    if min(*size, cells_x, cells_y, cells_z) <= 0:
        raise ValueError("soft package dimensions and cells must be positive")
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
                seam_wave = (
                    0.012
                    * height
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
                        - 0.285 * height * fill
                        + bottom_wrinkle
                        - (1.0 - fill) * seam_wave
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
                        + 0.625 * height * fill * crown
                        + top_wrinkle
                        - diagonal_crease
                        - cross_crease
                        + (1.0 - fill) * seam_wave
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
    _add_visual(
        frame,
        "manual_sorting_zone",
        (0.62, 0.48, 0.006),
        (MANIPULATION_STATION_X, 0.015, WORKTABLE_TOP_Z + 0.003),
        (0.46, 0.48, 0.48, 1.0),
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
    _add_visual(
        frame,
        "worktable_transfer_lip",
        (WORKTABLE_LENGTH - 0.04, 0.035, 0.012),
        (WORKTABLE_CENTER_X, worktable_far_y + 0.0025, 0.566),
        (0.72, 0.75, 0.76, 1.0),
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


def _create_openarm_station(root: Any) -> None:
    urdf_path = OPENARM_ASSET_ROOT / "openarm_v20_bimanual.urdf"
    if not urdf_path.is_file():
        raise FileNotFoundError(f"OpenArm asset is missing: {urdf_path}")
    source_scene = gobot.load_scene(OPENARM_URDF_RESOURCE)
    source_root = source_scene.root
    robot = gobot.create_node("Robot3D", OPENARM_ROBOT_NAME)
    for child in tuple(source_root.children):
        child.reparent(robot)
    root.add_child(robot)
    robot.mode = gobot.RobotMode.Motion
    robot.source_path = OPENARM_URDF_RESOURCE
    robot.position = OPENARM_ROOT_POSITION
    robot.rotation_degrees = (0.0, 0.0, OPENARM_ROOT_YAW_DEGREES)
    robot.semantic_label = "integrated_bimanual_package_manipulator"

    nodes = _nodes_by_name(robot)
    for node in nodes.values():
        if (
            node.type_name == "CollisionShape3D"
            and node.name != "openarm_body_link0_collision"
        ):
            # Triangle collision meshes remain useful authoring assets, but a
            # handful of primitive hand proxies is much cheaper for IPC.
            node.disabled = True

    for side_index, side in enumerate(ARM_SIDES):
        for joint_index, (joint_name, initial_position) in enumerate(
            zip(
                ARM_JOINT_NAMES_BY_SIDE[side_index][:7],
                ARM_INITIAL_POSES[side_index],
                strict=True,
            )
        ):
            _bake_revolute_joint(
                nodes,
                joint_name,
                initial_position,
                ARM_STIFFNESS[joint_index],
                ARM_DAMPING[joint_index],
            )
        for joint_name in ARM_JOINT_NAMES_BY_SIDE[side_index][-2:]:
            _bake_revolute_joint(
                nodes,
                joint_name,
                FINGER_OPEN_POSITIONS[side_index],
                GRIPPER_STIFFNESS,
                GRIPPER_DAMPING,
            )

        _add_robot_hand(nodes, side, side_index)

    # A raised shared shoulder, camera mast, and sensor head make the robot read
    # as one human-scale bimanual station rather than two pedestal arms.
    _add_visual(
        robot,
        "openarm_base_pedestal",
        (0.48, 0.40, 0.10),
        (0.0, 0.0, -0.05),
        (0.18, 0.21, 0.23, 1.0),
    )
    _add_visual(
        robot,
        "openarm_base_plate",
        (0.44, 0.36, 0.08),
        (0.0, 0.0, 0.04),
        (0.22, 0.25, 0.27, 1.0),
    )
    _add_visual(
        robot,
        "openarm_shoulder_shroud",
        (0.12, 0.44, 0.10),
        (0.0, 0.0, 0.94),
        (0.68, 0.70, 0.71, 1.0),
    )
    _add_visual(
        robot,
        "openarm_camera_mast",
        (0.055, 0.055, 0.43),
        (-0.055, 0.0, 1.19),
        (0.34, 0.37, 0.39, 1.0),
    )
    _add_visual(
        robot,
        "openarm_sensor_head",
        (0.12, 0.25, 0.065),
        (0.055, 0.0, 1.41),
        (0.72, 0.74, 0.75, 1.0),
    )
    _add_visual(
        robot,
        "openarm_sensor_lens",
        (0.012, 0.15, 0.035),
        (0.119, 0.0, 1.41),
        (0.025, 0.035, 0.045, 1.0),
    )

    for link_name in OPENARM_PROXY_LINK_NAMES:
        _add_coupling(
            root,
            link_name + "_coupling",
            _path_from_root(root, nodes[link_name]),
            gobot.PhysicsCouplingMode.OneWay,
        )


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
        body.mesh = _soft_package_mesh(spec["size"], spec["cells"])
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
    _create_openarm_station(root)
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

    # Imported scene roots retain their source SceneState during packing, which
    # restores child transforms to the URDF zero pose. Reapply the deliberately
    # baked IK rotations to the serialized child links.
    baked_rotations: dict[str, tuple[float, float, float]] = {}
    for side_index in range(len(ARM_SIDES)):
        child_names = ARM_LINK_NAMES_BY_SIDE[side_index][1:]
        positions = (
            *ARM_INITIAL_POSES[side_index],
            FINGER_OPEN_POSITIONS[side_index],
            FINGER_OPEN_POSITIONS[side_index],
        )
        axes = (*ARM_JOINT_AXES_BY_SIDE[side_index], *FINGER_JOINT_AXES)
        for child_name, position, axis in zip(
            child_names, positions, axes, strict=True
        ):
            baked_rotations[child_name] = tuple(
                math.degrees(position) * component for component in axis
            )
    for entry in nodes:
        rotation = baked_rotations.get(str(entry.get("name", "")))
        if rotation is None:
            continue
        entry.setdefault("properties", {})["rotation_degrees"] = {
            "matrix_data": {
                "cols": 1,
                "rows": 3,
                "storage": list(rotation),
            }
        }

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
    if not (OPENARM_ASSET_ROOT / "openarm_v20_bimanual.urdf").is_file():
        raise FileNotFoundError(f"OpenArm assets are missing: {OPENARM_ASSET_ROOT}")
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
