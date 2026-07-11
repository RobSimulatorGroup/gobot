"""Velocity command helpers for Gobot locomotion tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Commands:
    vel_limit: list[list[float]] = field(default_factory=lambda: [[-0.6, -0.4, -0.8], [1.0, 0.4, 0.8]])
    resampling_time: float = 0.0
    heading_command: bool = False
    heading_range: list[float] = field(default_factory=lambda: [-3.14, 3.14])
    heading_control_stiffness: float = 0.5
    rel_standing_envs: float = 0.0


def sample_velocity_commands(
    rng: np.random.Generator,
    num_samples: int,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    return np.asarray(rng.uniform(low=low, high=high, size=(num_samples, 3)), dtype=np.float32)


def zero_small_xy_commands(commands: np.ndarray, *, threshold: float = 0.2) -> None:
    moving = np.linalg.norm(commands[:, :2], axis=1) > threshold
    commands[:, :2] *= moving[:, None]


def sample_heading_commands(env: Any, num_samples: int) -> np.ndarray:
    heading_range = np.asarray(env.cfg.commands.heading_range, dtype=np.float32)
    if heading_range.shape != (2,):
        raise ValueError(f"commands.heading_range must have shape (2,), got {heading_range.shape}")
    low, high = float(np.min(heading_range)), float(np.max(heading_range))
    return np.asarray(np.random.uniform(low, high, size=(num_samples,)), dtype=np.float32)


def _yaw_from_quat(base_quat: np.ndarray) -> np.ndarray:
    q = np.asarray(base_quat, dtype=np.float32).reshape(-1, 4)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).astype(np.float32)


def _wrap_to_pi(value: np.ndarray) -> np.ndarray:
    return ((np.asarray(value, dtype=np.float32) + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def apply_heading_yaw_feedback(
    commands: np.ndarray,
    base_quat: np.ndarray,
    heading_commands: np.ndarray,
    *,
    stiffness: float,
    clip: float = 2.0,
) -> None:
    heading = _yaw_from_quat(base_quat)
    commands[:, 2] = np.clip(stiffness * _wrap_to_pi(heading_commands - heading), -clip, clip)
