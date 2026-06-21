"""Locomotion domain-randomization payload helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


@dataclass
class LocomotionDomainRandomizationCfg:
    """Backend-neutral reset randomization settings.

    The hot stepping path should consume arrays produced from this config, not
    parse asset files or scene nodes.
    """

    enabled: bool = True
    encoder_bias_range: tuple[float, float] = (0.0, 0.0)
    reset_lin_vel_ranges: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    reset_ang_vel_ranges: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    friction_range: tuple[float, float] | None = None
    mass_scale_range: tuple[float, float] | None = None
    gain_scale_range: tuple[float, float] | None = None


class LocomotionDomainRandomization:
    """Samples reset-time payloads for locomotion environments."""

    def __init__(self, cfg: LocomotionDomainRandomizationCfg, *, num_actions: int) -> None:
        self.cfg = cfg
        self.num_actions = int(num_actions)

    def reset_payload(
        self,
        env_ids: Sequence[int] | np.ndarray,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        count = int(env_ids.size)
        payload = {
            "env_ids": env_ids,
            "base_linear_velocity": np.zeros((count, 3), dtype=np.float32),
            "base_angular_velocity": np.zeros((count, 3), dtype=np.float32),
            "encoder_bias": np.zeros((count, self.num_actions), dtype=np.float32),
        }
        if count == 0 or not self.cfg.enabled:
            return payload
        payload["base_linear_velocity"] = _sample_vec_ranges(
            rng,
            self.cfg.reset_lin_vel_ranges,
            ("x", "y", "z"),
            count,
        )
        payload["base_angular_velocity"] = _sample_vec_ranges(
            rng,
            self.cfg.reset_ang_vel_ranges,
            ("x", "y", "z"),
            count,
        )
        lo, hi = self.cfg.encoder_bias_range
        if self.num_actions:
            payload["encoder_bias"] = rng.uniform(lo, hi, size=(count, self.num_actions)).astype(np.float32)
        if self.cfg.friction_range is not None:
            payload["friction"] = rng.uniform(*self.cfg.friction_range, size=count).astype(np.float32)
        if self.cfg.mass_scale_range is not None:
            payload["mass_scale"] = rng.uniform(*self.cfg.mass_scale_range, size=count).astype(np.float32)
        if self.cfg.gain_scale_range is not None:
            payload["gain_scale"] = rng.uniform(*self.cfg.gain_scale_range, size=count).astype(np.float32)
        return payload


def _sample_vec_ranges(
    rng: np.random.Generator,
    ranges: Mapping[str, tuple[float, float]],
    names: Sequence[str],
    count: int,
) -> np.ndarray:
    values = np.zeros((count, len(names)), dtype=np.float32)
    for axis, name in enumerate(names):
        lo, hi = ranges.get(name, (0.0, 0.0))
        values[:, axis] = rng.uniform(float(lo), float(hi), size=count).astype(np.float32)
    return values


__all__ = ["LocomotionDomainRandomization", "LocomotionDomainRandomizationCfg"]
