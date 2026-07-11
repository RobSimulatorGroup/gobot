"""Unitree Go1 articulation profile owned by the Go1 example."""

from __future__ import annotations


GO1_JOINT_NAMES: tuple[str, ...] = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)

GO1_DEFAULT_JOINT_POS: tuple[float, ...] = (
    0.1,
    0.9,
    -1.8,
    -0.1,
    0.9,
    -1.8,
    0.1,
    0.9,
    -1.8,
    -0.1,
    0.9,
    -1.8,
)
GO1_DEFAULT_BASE_POSITION: tuple[float, float, float] = (0.0, 0.0, 0.278)
GO1_FOOT_NAMES: tuple[str, ...] = ("FR", "FL", "RR", "RL")
GO1_FOOT_LINK_NAMES: tuple[str, ...] = ("FR_calf", "FL_calf", "RR_calf", "RL_calf")


def _joint_values(hip_thigh: float, calf: float) -> tuple[float, ...]:
    return tuple(
        calf if "calf" in joint_name else hip_thigh
        for joint_name in GO1_JOINT_NAMES
    )


GO1_KP: tuple[float, ...] = _joint_values(15.895242654143557, 35.764295971822996)
GO1_KD: tuple[float, ...] = _joint_values(1.0119225760208341, 2.2768257960468765)
GO1_ARMATURE: tuple[float, ...] = _joint_values(0.000111842 * 6.0**2, 0.000111842 * 9.0**2)
GO1_EFFORT_LIMIT: tuple[float, ...] = _joint_values(23.7, 35.55)
GO1_VELOCITY_LIMIT: tuple[float, ...] = _joint_values(30.1, 20.06)


__all__ = [
    "GO1_ARMATURE",
    "GO1_DEFAULT_BASE_POSITION",
    "GO1_DEFAULT_JOINT_POS",
    "GO1_EFFORT_LIMIT",
    "GO1_FOOT_LINK_NAMES",
    "GO1_FOOT_NAMES",
    "GO1_JOINT_NAMES",
    "GO1_KD",
    "GO1_KP",
    "GO1_VELOCITY_LIMIT",
]
