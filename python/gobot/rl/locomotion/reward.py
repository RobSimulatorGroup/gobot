"""Reward dispatch helpers for locomotion environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


@dataclass
class LocomotionRewardContext:
    """Common arrays supplied to vectorized locomotion reward terms."""

    dt: float
    command: np.ndarray
    base_lin_vel_b: np.ndarray
    base_ang_vel_b: np.ndarray
    projected_gravity: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    actions: np.ndarray
    previous_actions: np.ndarray
    foot_height: np.ndarray | None = None
    foot_contact: np.ndarray | None = None


RewardTerm = Callable[[LocomotionRewardContext], np.ndarray]


def dispatch_reward_terms(
    context: LocomotionRewardContext,
    terms: Mapping[str, RewardTerm],
    weights: Mapping[str, float],
    *,
    scale_by_dt: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Evaluate named reward terms and return total reward plus logs."""

    env_count = int(context.actions.shape[0])
    total = np.zeros((env_count,), dtype=np.float32)
    logged: dict[str, np.ndarray] = {}
    scale = float(context.dt) if scale_by_dt else 1.0
    for name, term in terms.items():
        weight = float(weights.get(name, 0.0))
        values = np.asarray(term(context), dtype=np.float32).reshape(env_count)
        weighted = values * weight * scale
        logged[name] = weighted
        total += weighted
    return total, logged


def tracking_xy_velocity(std: float) -> RewardTerm:
    std2 = max(float(std) ** 2, 1.0e-6)

    def _term(context: LocomotionRewardContext) -> np.ndarray:
        error = np.sum(np.square(context.command[:, :2] - context.base_lin_vel_b[:, :2]), axis=1)
        return np.exp(-error / std2).astype(np.float32)

    return _term


def tracking_yaw_velocity(std: float) -> RewardTerm:
    std2 = max(float(std) ** 2, 1.0e-6)

    def _term(context: LocomotionRewardContext) -> np.ndarray:
        error = np.square(context.command[:, 2] - context.base_ang_vel_b[:, 2])
        return np.exp(-error / std2).astype(np.float32)

    return _term


def upright_reward(std: float) -> RewardTerm:
    std2 = max(float(std) ** 2, 1.0e-6)

    def _term(context: LocomotionRewardContext) -> np.ndarray:
        error = np.sum(np.square(context.projected_gravity[:, :2]), axis=1)
        return np.exp(-error / std2).astype(np.float32)

    return _term


def action_rate_l2(context: LocomotionRewardContext) -> np.ndarray:
    return np.sum(np.square(context.actions - context.previous_actions), axis=1).astype(np.float32)


__all__ = [
    "LocomotionRewardContext",
    "RewardTerm",
    "action_rate_l2",
    "dispatch_reward_terms",
    "tracking_xy_velocity",
    "tracking_yaw_velocity",
    "upright_reward",
]
