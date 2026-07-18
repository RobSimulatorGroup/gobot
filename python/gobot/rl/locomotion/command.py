"""Velocity command configuration shared by locomotion backends."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class UniformVelocityCommandRanges:
    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: tuple[float, float] = (-1.0, 1.0)
    ang_vel_z: tuple[float, float] = (-0.5, 0.5)
    heading: tuple[float, float] | None = (-math.pi, math.pi)


@dataclass
class UniformVelocityCommandCfg:
    resampling_time_range: tuple[float, float] = (3.0, 8.0)
    rel_standing_envs: float = 0.1
    rel_heading_envs: float = 0.3
    rel_world_envs: float = 0.0
    rel_forward_envs: float = 0.2
    rel_run_envs: float = 0.0
    run_velocity_x: tuple[float, float] = (1.5, 2.5)
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    zero_small_xy_threshold: float = 0.0
    ranges: UniformVelocityCommandRanges = field(default_factory=UniformVelocityCommandRanges)


@dataclass
class VelocityCommandStage:
    step: int
    lin_vel_x: tuple[float, float] | None = None
    lin_vel_y: tuple[float, float] | None = None
    ang_vel_z: tuple[float, float] | None = None
    run_velocity_x: tuple[float, float] | None = None


__all__ = [
    "UniformVelocityCommandCfg",
    "UniformVelocityCommandRanges",
    "VelocityCommandStage",
]
