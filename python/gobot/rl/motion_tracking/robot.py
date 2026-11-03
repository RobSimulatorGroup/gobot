# Adapted from Motphys/Uni-SuperMimic, faf7818e8abac3ec6abb8f00e6e904d3577f800b.
# SPDX-License-Identifier: Apache-2.0
"""Unitree G1 29-DoF robot constants for SuperMimic."""

from __future__ import annotations

import re

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0


def _kp(armature: float) -> float:
    return armature * NATURAL_FREQ**2


def _kd(armature: float) -> float:
    return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip

TRACKED_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]  # fmt: skip

COLLISION_GEOM_NAMES = tuple(
    f"{body}_collision" if count == 1 else f"{body}_collision_{index}"
    for body, count in (
        ("pelvis_contour_link", 1),
        ("left_hip_pitch_link", 1),
        ("left_hip_roll_link", 1),
        ("left_hip_yaw_link", 1),
        ("left_knee_link", 1),
        ("left_ankle_pitch_link", 1),
        ("left_ankle_roll_link", 4),
        ("right_hip_pitch_link", 1),
        ("right_hip_roll_link", 1),
        ("right_hip_yaw_link", 1),
        ("right_knee_link", 1),
        ("right_ankle_pitch_link", 1),
        ("right_ankle_roll_link", 4),
        ("torso_link", 1),
        ("logo_link", 1),
        ("head_link", 1),
        ("left_shoulder_pitch_link", 1),
        ("left_shoulder_roll_link", 1),
        ("left_shoulder_yaw_link", 1),
        ("left_elbow_link", 1),
        ("left_wrist_roll_link", 1),
        ("left_wrist_pitch_link", 1),
        ("left_wrist_yaw_link", 1),
        ("right_shoulder_pitch_link", 1),
        ("right_shoulder_roll_link", 1),
        ("right_shoulder_yaw_link", 1),
        ("right_elbow_link", 1),
        ("right_wrist_roll_link", 1),
        ("right_wrist_pitch_link", 1),
        ("right_wrist_yaw_link", 1),
    )
    for index in range(1, count + 1)
)

_M5020 = (25.0, 37.0, ARMATURE_5020)
_M4010 = (5.0, 22.0, ARMATURE_4010)
JOINT_MOTORS = {
    "hip_pitch": (88.0, 32.0, ARMATURE_7520_14),
    "hip_roll": (139.0, 20.0, ARMATURE_7520_22),
    "hip_yaw": (88.0, 32.0, ARMATURE_7520_14),
    "knee": (139.0, 20.0, ARMATURE_7520_22),
    "ankle_pitch": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "ankle_roll": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "waist_yaw": (88.0, 32.0, ARMATURE_7520_14),
    "waist_roll": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "waist_pitch": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "shoulder_pitch": _M5020,
    "shoulder_roll": _M5020,
    "shoulder_yaw": _M5020,
    "elbow": _M5020,
    "wrist_roll": _M5020,
    "wrist_pitch": _M4010,
    "wrist_yaw": _M4010,
}

DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}


def motor_of(joint_name: str) -> tuple[float, float, float]:
    key = joint_name.replace("left_", "").replace("right_", "").replace("_joint", "")
    return JOINT_MOTORS[key]


def action_scale(joint_name: str) -> float:
    effort, _, armature = motor_of(joint_name)
    return 0.25 * effort / _kp(armature)


def default_joint_pos_list() -> list[float]:
    positions = []
    for name in G1_JOINT_NAMES:
        value = 0.0
        for pattern, position in DEFAULT_JOINT_POS.items():
            if re.fullmatch(pattern, name):
                value = position
        positions.append(value)
    return positions
