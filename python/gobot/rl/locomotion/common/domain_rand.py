"""Common locomotion domain randomization config."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DomainRandConfig:
    randomize_friction: bool = True
    friction_range: list[float] = field(default_factory=lambda: [0.6, 1.4])
    randomize_base_mass: bool = True
    added_mass_range: list[float] = field(default_factory=lambda: [-1.0, 3.0])
    randomize_kp: bool = False
    kp_multiplier_range: list[float] = field(default_factory=lambda: [1.0, 1.0])
    randomize_kd: bool = False
    kd_multiplier_range: list[float] = field(default_factory=lambda: [1.0, 1.0])
    push_body_name: str = ""
    push_interval_s: float = 0.0
    push_velocity_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
