"""Velocity command sampling and command curriculum runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Protocol, Sequence

import numpy as np

from .math import _as_vec, _quat, _quat_to_yaw, _rotate_vec_by_quat_inv, _wrap_to_pi


@dataclass
class UniformVelocityCommandRanges:
    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: tuple[float, float] = (-1.0, 1.0)
    ang_vel_z: tuple[float, float] = (-0.5, 0.5)
    heading: tuple[float, float] | None = (-math.pi, math.pi)


@dataclass
class UniformVelocityCommandCfg:
    name: str = "twist"
    resampling_time_range: tuple[float, float] = (3.0, 8.0)
    rel_standing_envs: float = 0.1
    rel_heading_envs: float = 0.3
    rel_world_envs: float = 0.0
    rel_forward_envs: float = 0.2
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    init_velocity_prob: float = 0.0
    ranges: UniformVelocityCommandRanges = field(default_factory=UniformVelocityCommandRanges)


@dataclass
class VelocityCommandStage:
    step: int
    lin_vel_x: tuple[float, float] | None = None
    lin_vel_y: tuple[float, float] | None = None
    ang_vel_z: tuple[float, float] | None = None


class VelocityCommandEnv(Protocol):
    num_envs: int
    step_dt: float
    _rng: np.random.Generator


class UniformVelocityCommand:
    def __init__(self, cfg: UniformVelocityCommandCfg, env: VelocityCommandEnv) -> None:
        self.cfg = cfg
        self.env = env
        self.command_b = np.zeros((env.num_envs, 3), dtype=np.float32)
        self.command_w = np.zeros((env.num_envs, 3), dtype=np.float32)
        self.heading_target = np.zeros(env.num_envs, dtype=np.float32)
        self.heading_error = np.zeros(env.num_envs, dtype=np.float32)
        self.is_heading_env = np.zeros(env.num_envs, dtype=bool)
        self.is_standing_env = np.zeros(env.num_envs, dtype=bool)
        self.is_world_env = np.zeros(env.num_envs, dtype=bool)
        self.is_forward_env = np.zeros(env.num_envs, dtype=bool)
        self.time_left = np.zeros(env.num_envs, dtype=np.float32)
        self.metrics: dict[str, np.ndarray] = {
            "error_vel_xy": np.zeros(env.num_envs, dtype=np.float32),
            "error_vel_yaw": np.zeros(env.num_envs, dtype=np.float32),
        }

    def reset(self, env_ids: np.ndarray) -> None:
        self._resample(env_ids)
        low, high = self.cfg.resampling_time_range
        self.time_left[env_ids] = self.env._rng.uniform(low, high, size=env_ids.shape).astype(np.float32)
        self.metrics["error_vel_xy"][env_ids] = 0.0
        self.metrics["error_vel_yaw"][env_ids] = 0.0

    def compute(self, dt: float, states: Sequence[Any | None]) -> None:
        self.time_left -= float(dt)
        resample_ids = np.flatnonzero(self.time_left <= 0.0).astype(np.int64)
        if resample_ids.size:
            self.reset(resample_ids)
        self._update_heading_and_world_commands(states)
        max_command_step = max(self.cfg.resampling_time_range[1] / max(self.env.step_dt, 1.0e-9), 1.0)
        for env_id, state in enumerate(states):
            if state is None:
                continue
            base_lin_vel_b = _rotate_vec_by_quat_inv(_as_vec(state.base.get("linear_velocity"), 3), _quat(state.base))
            base_ang_vel_b = _rotate_vec_by_quat_inv(_as_vec(state.base.get("angular_velocity"), 3), _quat(state.base))
            self.metrics["error_vel_xy"][env_id] += np.linalg.norm(self.command_b[env_id, :2] - base_lin_vel_b[:2]) / max_command_step
            self.metrics["error_vel_yaw"][env_id] += abs(self.command_b[env_id, 2] - base_ang_vel_b[2]) / max_command_step

    def _resample(self, env_ids: np.ndarray) -> None:
        ranges = self.cfg.ranges
        self.command_b[env_ids, 0] = self.env._rng.uniform(*ranges.lin_vel_x, size=env_ids.shape)
        self.command_b[env_ids, 1] = self.env._rng.uniform(*ranges.lin_vel_y, size=env_ids.shape)
        self.command_b[env_ids, 2] = self.env._rng.uniform(*ranges.ang_vel_z, size=env_ids.shape)
        if self.cfg.heading_command:
            if ranges.heading is None:
                raise ValueError("heading_command=True requires heading range")
            self.heading_target[env_ids] = self.env._rng.uniform(*ranges.heading, size=env_ids.shape)
            self.is_heading_env[env_ids] = self.env._rng.uniform(0.0, 1.0, size=env_ids.shape) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = self.env._rng.uniform(0.0, 1.0, size=env_ids.shape) <= self.cfg.rel_standing_envs
        self.is_world_env[env_ids] = self.env._rng.uniform(0.0, 1.0, size=env_ids.shape) <= self.cfg.rel_world_envs
        self.command_w[env_ids] = self.command_b[env_ids]
        self.is_forward_env[env_ids] = self.env._rng.uniform(0.0, 1.0, size=env_ids.shape) <= self.cfg.rel_forward_envs
        forward_ids = env_ids[self.is_forward_env[env_ids]]
        if forward_ids.size:
            self.command_b[forward_ids, 0] = np.maximum(np.abs(self.command_b[forward_ids, 0]), 0.3)
            self.command_b[forward_ids, 1] = 0.0
            self.command_b[forward_ids, 2] = 0.0

    def _update_heading_and_world_commands(self, states: Sequence[Any | None]) -> None:
        for env_id, state in enumerate(states):
            if state is None:
                continue
            heading = _quat_to_yaw(_quat(state.base))
            if self.cfg.heading_command and self.is_heading_env[env_id]:
                self.heading_error[env_id] = _wrap_to_pi(float(self.heading_target[env_id]) - heading)
                lo, hi = self.cfg.ranges.ang_vel_z
                self.command_b[env_id, 2] = np.clip(self.cfg.heading_control_stiffness * self.heading_error[env_id], lo, hi)
            if self.is_world_env[env_id]:
                vx_w, vy_w = self.command_w[env_id, :2]
                self.command_b[env_id, 0] = math.cos(heading) * vx_w + math.sin(heading) * vy_w
                self.command_b[env_id, 1] = -math.sin(heading) * vx_w + math.cos(heading) * vy_w
        self.command_b[self.is_standing_env] = 0.0
        self.command_w[self.is_standing_env] = 0.0


__all__ = [
    "UniformVelocityCommand",
    "UniformVelocityCommandCfg",
    "UniformVelocityCommandRanges",
    "VelocityCommandStage",
]
