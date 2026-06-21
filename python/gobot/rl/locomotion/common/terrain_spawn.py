"""Terrain spawn config shared by locomotion tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TerrainSpawnConfig:
    curriculum: bool = False
    spawn_jitter: float = 0.0
    difficulty_radius: float = 1.0
