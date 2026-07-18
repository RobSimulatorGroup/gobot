"""Reusable locomotion helpers for Gobot RL examples and tasks."""

from __future__ import annotations

from gobot.rl.runtime import (
    LocomotionBatchSpec,
    NativeLocomotionBatchBackend,
)

from .base import LocomotionBatchEnv, LocomotionControlCfg, LocomotionNoiseCfg
from .command import (
    UniformVelocityCommandCfg,
    UniformVelocityCommandRanges,
    VelocityCommandStage,
)
from .observation import (
    VELOCITY_OBS_SCHEMA_VERSION,
    build_velocity_actor_observation,
    velocity_actor_observation_schema,
    velocity_critic_observation_schema,
)

__all__ = [
    "LocomotionBatchEnv",
    "LocomotionBatchSpec",
    "LocomotionControlCfg",
    "LocomotionNoiseCfg",
    "NativeLocomotionBatchBackend",
    "UniformVelocityCommandCfg",
    "UniformVelocityCommandRanges",
    "VELOCITY_OBS_SCHEMA_VERSION",
    "VelocityCommandStage",
    "build_velocity_actor_observation",
    "velocity_actor_observation_schema",
    "velocity_critic_observation_schema",
]
