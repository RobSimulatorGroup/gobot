"""Height scan config shared by locomotion tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HeightScanConfig:
    sensor_name: str | None = None
    max_distance: float = 5.0
    enabled: bool = False
