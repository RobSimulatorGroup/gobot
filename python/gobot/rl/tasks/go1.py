"""Canonical Go1 articulation and controller profile."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping


GO1_TASK_NAME = "go1_rough_velocity"
GO1_TASK_VERSION = "go1_rough_velocity_numpy_v2"
GO1_PHYSICS_DT = 0.005
GO1_DECIMATION = 4


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

GO1_MUJOCO_SOLVER_SETTINGS: Mapping[str, Any] = MappingProxyType(
    {
        "solver": "Newton",
        "integrator": "ImplicitFast",
        "cone": "Elliptic",
        "jacobian": "Auto",
        "iterations": 10,
        "line_search_iterations": 20,
        "convex_collision_iterations": 500,
        "tolerance": 1.0e-8,
        "line_search_tolerance": 0.01,
        "convex_collision_tolerance": 1.0e-6,
        "impedance_ratio": 10.0,
    }
)

__all__ = [
    "GO1_ARMATURE",
    "GO1_DECIMATION",
    "GO1_DEFAULT_BASE_POSITION",
    "GO1_DEFAULT_JOINT_POS",
    "GO1_EFFORT_LIMIT",
    "GO1_FOOT_LINK_NAMES",
    "GO1_FOOT_NAMES",
    "GO1_JOINT_NAMES",
    "GO1_KD",
    "GO1_KP",
    "GO1_MUJOCO_SOLVER_SETTINGS",
    "GO1_PHYSICS_DT",
    "GO1_TASK_NAME",
    "GO1_TASK_VERSION",
    "GO1_VELOCITY_LIMIT",
]
