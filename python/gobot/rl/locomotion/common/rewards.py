"""Shared NumPy reward dispatch helpers for locomotion tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RewardContext:
    info: dict[str, Any]
    linvel: np.ndarray
    gyro: np.ndarray
    dof_pos: np.ndarray
    num_envs: int = 0
    default_angles: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    tracking_sigma: float = 0.25
    base_height_target: float = 0.0
    base_height: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    gravity: np.ndarray | None = None
    dof_vel: np.ndarray | None = None
    pose_weights: np.ndarray | None = None
    joint_range: np.ndarray | None = None
    linvel_yaw: np.ndarray | None = None


RewardFn = Callable[[RewardContext], np.ndarray]


def run_reward_dispatch(
    *,
    scales: Mapping[str, float],
    fns: Mapping[str, RewardFn],
    ctx: RewardContext,
    info: dict[str, Any],
    enable_log: bool,
    ctrl_dt: float,
) -> np.ndarray:
    reward = np.zeros((ctx.num_envs,), dtype=np.float32)
    log: dict[str, float] = {}
    for name, scale in scales.items():
        if scale == 0.0:
            continue
        fn = fns.get(name)
        if fn is None:
            raise KeyError(f"unknown reward term {name!r}")
        value = np.asarray(fn(ctx), dtype=np.float32).reshape(ctx.num_envs)
        weighted = value * float(scale) * float(ctrl_dt)
        reward += weighted
        if enable_log:
            log[f"reward/{name}"] = float(np.mean(weighted))
    if enable_log:
        info.setdefault("log", {}).update(log)
    return reward
