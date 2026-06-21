"""Reusable locomotion helpers for Gobot RL examples and tasks."""

from __future__ import annotations

from gobot.rl.runtime import (
    LocomotionBatchSpec,
    NativeLocomotionBatchBackend,
    NativeLocomotionBatchState,
)

from .base import LocomotionBatchEnv, LocomotionControlCfg, LocomotionNoiseCfg
from .command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
    UniformVelocityCommandRanges,
    VelocityCommandStage,
)
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
from .terrain import TerrainSampler

__all__ = [
    "LocomotionBatchEnv",
    "LocomotionBatchSpec",
    "LocomotionControlCfg",
    "LocomotionNoiseCfg",
    "NativeLocomotionBatchBackend",
    "NativeLocomotionBatchState",
    "ObservationField",
    "ObservationSchema",
    "TerrainSampler",
    "UniformVelocityCommand",
    "UniformVelocityCommandCfg",
    "UniformVelocityCommandRanges",
    "VELOCITY_OBS_SCHEMA_VERSION",
    "VelocityCommandStage",
    "build_velocity_actor_observation",
    "build_velocity_critic_observation",
    "log_contact_forces",
    "velocity_actor_observation_schema",
    "velocity_critic_observation_schema",
]
