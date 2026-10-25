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

from .config import FootCfg, LocomotionRobotCfg, VelocityTaskCfg

def make_velocity_env(cfg, *, backend="mujoco-cpu", num_envs=64, device=None, seed=42, **kwargs):
    """Create a typed velocity environment from an authored scene and robot profile."""
    if backend == "mujoco-cpu":
        from .cpu_env import CpuVelocityEnv
        return CpuVelocityEnv(cfg, num_envs=num_envs, device=device or "cpu", seed=seed, **kwargs)
    if backend == "mujoco-warp":
        from .warp_env import WarpVelocityEnv
        return WarpVelocityEnv(cfg, num_envs=num_envs, device=device or "cuda:0", seed=seed, **kwargs)
    raise ValueError(f"unsupported velocity backend {backend!r}")

__all__ += ["FootCfg", "LocomotionRobotCfg", "VelocityTaskCfg", "make_velocity_env"]
