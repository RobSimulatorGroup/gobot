"""Pinned topology plus a dependency-free reader for Newton's G1 policy YAML."""

from __future__ import annotations

import ast
from pathlib import Path

PHYSICS_DT = 0.005
POLICY_DECIMATION = 4
POLICY_DT = PHYSICS_DT * POLICY_DECIMATION
ACTION_SCALE = 0.5
BASE_LINK = "pelvis"
BASE_POSE_XYZW = (0.0, 0.0, 0.76, 0.0, 0.0, 0.7071, 0.7071)

JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
)

LINK_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_hand_thumb_0_link",
    "left_hand_thumb_1_link",
    "left_hand_thumb_2_link",
    "left_hand_middle_0_link",
    "left_hand_middle_1_link",
    "left_hand_index_0_link",
    "left_hand_index_1_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_hand_thumb_0_link",
    "right_hand_thumb_1_link",
    "right_hand_thumb_2_link",
    "right_hand_middle_0_link",
    "right_hand_middle_1_link",
    "right_hand_index_0_link",
    "right_hand_index_1_link",
)

DEFAULT_JOINT_POSITION = (
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
)

JOINT_STIFFNESS = (
    200.0, 150.0, 150.0, 200.0, 20.0, 20.0,
    200.0, 150.0, 150.0, 200.0, 20.0, 20.0,
    300.0, 300.0, 300.0,
    40.0, 40.0, 40.0, 40.0, 20.0, 20.0, 20.0,
    10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
    40.0, 40.0, 40.0, 40.0, 20.0, 20.0, 20.0,
    10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
)

JOINT_DAMPING = (
    5.0, 5.0, 5.0, 5.0, 2.0, 2.0,
    5.0, 5.0, 5.0, 5.0, 2.0, 2.0,
    5.0, 5.0, 5.0,
    10.0, 10.0, 10.0, 10.0, 0.5, 0.5, 0.5,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    10.0, 10.0, 10.0, 10.0, 0.5, 0.5, 0.5,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
)

JOINT_ARMATURE = (0.1,) * len(JOINT_NAMES)
OBSERVATION_DIM = 12 + 3 * len(JOINT_NAMES)
ACTION_DIM = len(JOINT_NAMES)


def load_native_policy_contract(path: str | Path) -> dict[str, object]:
    """Read the small YAML subset used by Newton's released policy contract."""

    document: dict[str, object] = {}
    active_list: str | None = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("  - "):
            if active_list is None:
                raise ValueError(f"unexpected YAML list item in {path}: {line!r}")
            value = ast.literal_eval(line[4:].strip())
            cast_list = document[active_list]
            assert isinstance(cast_list, list)
            cast_list.append(value)
            continue
        if line[0].isspace() or ":" not in line:
            raise ValueError(f"unsupported policy YAML syntax in {path}: {line!r}")
        key, raw_value = line.split(":", 1)
        raw_value = raw_value.strip()
        if not raw_value:
            document[key] = []
            active_list = key
        else:
            document[key] = ast.literal_eval(raw_value)
            active_list = None

    required = (
        "num_dofs",
        "action_scale",
        "mjw_joint_names",
        "mjw_joint_pos",
        "mjw_joint_stiffness",
        "mjw_joint_damping",
        "mjw_joint_armature",
    )
    missing = [key for key in required if key not in document]
    if missing:
        raise ValueError(f"policy YAML is missing: {', '.join(missing)}")
    num_dofs = document["num_dofs"]
    if not isinstance(num_dofs, int) or isinstance(num_dofs, bool) or num_dofs <= 0:
        raise ValueError("policy num_dofs must be a positive integer")
    for key in required[2:]:
        values = document[key]
        if not isinstance(values, list) or len(values) != num_dofs:
            raise ValueError(f"policy {key} must contain {num_dofs} values")
        document[key] = tuple(values)
    return document
