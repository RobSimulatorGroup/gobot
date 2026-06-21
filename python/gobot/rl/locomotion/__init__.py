"""Reusable locomotion helpers for Gobot RL examples and tasks."""

from __future__ import annotations

from gobot.rl.runtime import (
    LocomotionBatchSpec,
    NativeLocomotionBatchBackend,
)

from .base import LocomotionBatchEnv, LocomotionControlCfg, LocomotionNoiseCfg
from .command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
    UniformVelocityCommandRanges,
    VelocityCommandStage,
)
from .domain_randomization import LocomotionDomainRandomization, LocomotionDomainRandomizationCfg
from .observation import (
    ObservationField,
    ObservationSchema,
    VELOCITY_OBS_SCHEMA_VERSION,
    build_velocity_actor_observation,
    build_velocity_critic_observation,
    log_contact_forces,
    velocity_actor_observation_schema,
    velocity_critic_observation_schema,
)
from .reward import (
    LocomotionRewardContext,
    RewardTerm,
    action_rate_l2,
    dispatch_reward_terms,
    tracking_xy_velocity,
    tracking_yaw_velocity,
    upright_reward,
)
from .terrain import HeightScan, TerrainSampler, TerrainSpawn

__all__ = [
    "HeightScan",
    "LocomotionBatchEnv",
    "LocomotionBatchSpec",
    "LocomotionControlCfg",
    "LocomotionDomainRandomization",
    "LocomotionDomainRandomizationCfg",
    "LocomotionNoiseCfg",
    "LocomotionRewardContext",
    "NativeLocomotionBatchBackend",
    "ObservationField",
    "ObservationSchema",
    "RewardTerm",
    "TerrainSampler",
    "TerrainSpawn",
    "UniformVelocityCommand",
    "UniformVelocityCommandCfg",
    "UniformVelocityCommandRanges",
    "VELOCITY_OBS_SCHEMA_VERSION",
    "VelocityCommandStage",
    "action_rate_l2",
    "build_velocity_actor_observation",
    "build_velocity_critic_observation",
    "dispatch_reward_terms",
    "log_contact_forces",
    "tracking_xy_velocity",
    "tracking_yaw_velocity",
    "upright_reward",
    "velocity_actor_observation_schema",
    "velocity_critic_observation_schema",
]
