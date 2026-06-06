"""Gobot-native velocity locomotion task."""

from __future__ import annotations

from .cfg import (
    UniformVelocityCommandCfg,
    UniformVelocityCommandRanges,
    VelocityDomainRandomizationCfg,
    VelocityIllegalContactCfg,
    VelocityObservationCfg,
    VelocityRewardCfg,
    VelocityStage,
    VelocityTaskCfg,
    VelocityTerrainNormalUprightCfg,
    rsl_rl_train_cfg,
    unitree_g1_flat_velocity_cfg,
    unitree_g1_rough_velocity_cfg,
    unitree_go1_flat_velocity_cfg,
    unitree_go1_rough_velocity_cfg,
    velocity_task_cfg,
)
from .env import GobotVelocityEnv

__all__ = [
    "GobotVelocityEnv",
    "UniformVelocityCommandCfg",
    "UniformVelocityCommandRanges",
    "VelocityDomainRandomizationCfg",
    "VelocityIllegalContactCfg",
    "VelocityObservationCfg",
    "VelocityRewardCfg",
    "VelocityStage",
    "VelocityTaskCfg",
    "VelocityTerrainNormalUprightCfg",
    "rsl_rl_train_cfg",
    "unitree_g1_flat_velocity_cfg",
    "unitree_g1_rough_velocity_cfg",
    "unitree_go1_flat_velocity_cfg",
    "unitree_go1_rough_velocity_cfg",
    "velocity_task_cfg",
]
