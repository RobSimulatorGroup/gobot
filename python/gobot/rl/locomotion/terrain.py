"""Terrain height and spawn helpers for locomotion tasks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .math import _json_vec


@dataclass
class HeightScan:
    """Normalizes terrain height-scan arrays for observation groups."""

    dim: int
    max_distance: float = 5.0
    enabled: bool = True

    def normalize(self, values: np.ndarray | None) -> np.ndarray:
        if not self.enabled:
            return np.zeros((0,), dtype=np.float32)
        if self.dim <= 0:
            if values is None:
                return np.zeros((0,), dtype=np.float32)
            raise RuntimeError("height scan is configured with zero channels")
        if values is None:
            raise RuntimeError("height scan is enabled but no values were provided")
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1:] != (self.dim,):
            raise RuntimeError(f"height scan expected trailing dimension {self.dim}, got {array.shape}")
        return array / max(float(self.max_distance), 1.0e-6)

    def terrain_normal(self, points: np.ndarray, hits: np.ndarray | None = None, *, min_hits: int = 3) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if points.shape[-1] != 3:
            raise ValueError(f"height scan points must have last dimension 3, got {points.shape}")
        flat = points.reshape(-1, 3)
        if hits is not None:
            mask = np.asarray(hits, dtype=bool).reshape(-1)
            flat = flat[mask[: flat.shape[0]]]
        flat = flat[np.all(np.isfinite(flat), axis=1)]
        if flat.shape[0] < max(3, int(min_hits)):
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        centered = flat.astype(np.float64) - np.mean(flat.astype(np.float64), axis=0)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        normal = vh[-1]
        if normal[2] < 0.0:
            normal = -normal
        length = np.linalg.norm(normal)
        if length <= 1.0e-6 or not np.all(np.isfinite(normal)):
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        return (normal / length).astype(np.float32)


class TerrainSpawn:
    """Spawn origin sampling with a simple promote/demote curriculum."""

    def __init__(
        self,
        origins: np.ndarray,
        *,
        levels: np.ndarray | None = None,
        curriculum: bool = True,
        warmup_index: int | None = None,
    ) -> None:
        origins = np.asarray(origins, dtype=np.float32)
        if origins.ndim != 2 or origins.shape[1] != 3 or origins.shape[0] == 0:
            origins = np.zeros((1, 3), dtype=np.float32)
        self.origins = origins
        if levels is None:
            levels = np.zeros((origins.shape[0],), dtype=np.float32)
        self.levels = np.asarray(levels, dtype=np.float32).reshape(-1)
        if self.levels.shape != (origins.shape[0],):
            raise ValueError("terrain spawn levels must match origins")
        self.curriculum = bool(curriculum)
        self.order = np.argsort(self.levels, kind="stable")
        self.warmup_index = int(np.argmin(np.linalg.norm(origins[:, :2], axis=1)) if warmup_index is None else warmup_index)

    def sample_index(
        self,
        rng: np.random.Generator,
        *,
        progress: float,
        limit: float,
        warmup_progress: float = 0.10,
    ) -> int:
        if not self.curriculum:
            return int(rng.integers(0, self.origins.shape[0]))
        if float(progress) < warmup_progress and float(limit) <= 0.0:
            return self.warmup_index
        allowed_level = max(float(np.clip(progress, 0.0, 1.0)), float(limit))
        candidates = np.flatnonzero(self.levels <= allowed_level + 1.0e-6)
        if candidates.size == 0:
            candidates = self.order[:1]
        if self.warmup_index not in candidates:
            candidates = np.concatenate([candidates, np.asarray([self.warmup_index], dtype=np.int64)])
        return int(candidates[int(rng.integers(0, len(candidates)))])

    def update_limit(
        self,
        current: float,
        *,
        reset_reason: int,
        survival: float,
        distance: float,
        expected_distance: float,
    ) -> float:
        step = 1.0 / max(float(self.order.size - 1), 1.0)
        level = float(current)
        if int(reset_reason) == 2 or survival > 0.75 or distance > expected_distance:
            level += step
        elif int(reset_reason) == 1 and survival < 0.25:
            level -= step
        return float(np.clip(level, 0.0, 1.0))


class TerrainSampler:
    """Small height-query fallback for reset placement."""

    def __init__(self, scene_path: str | Path, grid_resolution: float = 0.08) -> None:
        self._boxes: list[dict[str, Any]] = []
        self._heightfields: list[dict[str, Any]] = []
        self._grid_resolution = float(grid_resolution)
        self._grid_origin = np.zeros(2, dtype=np.float64)
        self._grid_heights: np.ndarray | None = None
        path = Path(scene_path)
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        terrain_node = next((node for node in data.get("__NODES__", []) if node.get("type") == "Terrain3D"), None)
        if terrain_node is None:
            return
        properties = terrain_node.get("properties", {})
        for box in properties.get("boxes", []):
            self._boxes.append(
                {
                    "center": _json_vec(box.get("center"), 3),
                    "size": _json_vec(box.get("size"), 3),
                    "rotation": _json_vec(box.get("rotation_degrees"), 3),
                }
            )
        for heightfield in properties.get("heightfields", []):
            rows = int(heightfield.get("rows", 0))
            cols = int(heightfield.get("cols", 0))
            heights = np.asarray(heightfield.get("heights", []), dtype=np.float64)
            if rows <= 1 or cols <= 1 or heights.size != rows * cols:
                continue
            self._heightfields.append(
                {
                    "center": _json_vec(heightfield.get("center"), 3),
                    "size": _json_vec(heightfield.get("size"), 2),
                    "rows": rows,
                    "cols": cols,
                    "heights": heights.reshape(rows, cols),
                    "z_offset": float(heightfield.get("z_offset", 0.0)),
                }
            )
        self._build_height_grid()

    def height_at(self, x: float, y: float) -> float:
        if self._grid_heights is not None:
            return self._grid_height_at(x, y)
        height = -np.inf
        for box in self._boxes:
            candidate = self._box_height(box, x, y)
            if candidate is not None:
                height = max(height, candidate)
        for heightfield in self._heightfields:
            candidate = self._heightfield_height(heightfield, x, y)
            if candidate is not None:
                height = max(height, candidate)
        return float(height if np.isfinite(height) else 0.0)

    def heights_at(self, xy: np.ndarray) -> np.ndarray:
        """Return terrain heights for points with shape ``(..., 2)``."""
        points = np.asarray(xy, dtype=np.float64)
        if points.shape[-1] != 2:
            raise ValueError(f"xy must have last dimension 2, got {points.shape}")
        flat = points.reshape(-1, 2)
        if self._grid_heights is not None:
            heights = self._grid_heights_at(flat[:, 0], flat[:, 1])
        else:
            heights = np.asarray([self.height_at(float(x), float(y)) for x, y in flat], dtype=np.float64)
        return heights.reshape(points.shape[:-1]).astype(np.float32)

    def bounds(self) -> tuple[float, float, float, float] | None:
        """Return the loaded terrain XY bounds as ``(min_x, min_y, max_x, max_y)``."""
        return self._terrain_bounds()

    def _build_height_grid(self) -> None:
        bounds = self._terrain_bounds()
        if bounds is None:
            return
        min_x, min_y, max_x, max_y = bounds
        padding = max(self._grid_resolution * 2.0, 0.25)
        xs = np.arange(min_x - padding, max_x + padding + self._grid_resolution * 0.5, self._grid_resolution)
        ys = np.arange(min_y - padding, max_y + padding + self._grid_resolution * 0.5, self._grid_resolution)
        if xs.size < 2 or ys.size < 2:
            return
        grid_x, grid_y = np.meshgrid(xs, ys)
        heights = np.full(grid_x.shape, -np.inf, dtype=np.float64)
        for box in self._boxes:
            candidate = self._box_height_grid(box, grid_x, grid_y)
            if candidate is not None:
                heights = np.maximum(heights, candidate)
        for heightfield in self._heightfields:
            candidate = self._heightfield_height_grid(heightfield, grid_x, grid_y)
            if candidate is not None:
                heights = np.maximum(heights, candidate)
        heights[~np.isfinite(heights)] = 0.0
        self._grid_origin = np.array([xs[0], ys[0]], dtype=np.float64)
        self._grid_heights = heights

    def _terrain_bounds(self) -> tuple[float, float, float, float] | None:
        bounds = []
        for box in self._boxes:
            center = box["center"]
            size = box["size"]
            bounds.append((center[0] - size[0] * 0.5, center[1] - size[1] * 0.5, center[0] + size[0] * 0.5, center[1] + size[1] * 0.5))
        for heightfield in self._heightfields:
            center = heightfield["center"]
            size = heightfield["size"]
            bounds.append((center[0] - size[0] * 0.5, center[1] - size[1] * 0.5, center[0] + size[0] * 0.5, center[1] + size[1] * 0.5))
        if not bounds:
            return None
        return (
            min(bound[0] for bound in bounds),
            min(bound[1] for bound in bounds),
            max(bound[2] for bound in bounds),
            max(bound[3] for bound in bounds),
        )

    def _grid_height_at(self, x: float, y: float) -> float:
        assert self._grid_heights is not None
        return float(self._grid_heights_at(np.asarray([x]), np.asarray([y]))[0])

    def _grid_heights_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert self._grid_heights is not None
        u = (np.asarray(x, dtype=np.float64) - self._grid_origin[0]) / self._grid_resolution
        v = (np.asarray(y, dtype=np.float64) - self._grid_origin[1]) / self._grid_resolution
        rows, cols = self._grid_heights.shape
        inside = (u >= 0.0) & (v >= 0.0) & (u <= cols - 1) & (v <= rows - 1)
        c0 = np.clip(np.floor(u).astype(np.int64), 0, cols - 1)
        r0 = np.clip(np.floor(v).astype(np.int64), 0, rows - 1)
        c1 = np.minimum(c0 + 1, cols - 1)
        r1 = np.minimum(r0 + 1, rows - 1)
        fu = u - c0
        fv = v - r0
        h00 = self._grid_heights[r0, c0]
        h10 = self._grid_heights[r0, c1]
        h01 = self._grid_heights[r1, c0]
        h11 = self._grid_heights[r1, c1]
        sampled = (h00 * (1.0 - fu) + h10 * fu) * (1.0 - fv) + (h01 * (1.0 - fu) + h11 * fu) * fv
        return np.where(inside, sampled, 0.0)

    @staticmethod
    def _box_height(box: Mapping[str, Any], x: float, y: float) -> float | None:
        center = box["center"]
        size = box["size"]
        rotation = box["rotation"]
        yaw = -np.deg2rad(rotation[2])
        local_x = x - center[0]
        local_y = y - center[1]
        rx = np.cos(yaw) * local_x - np.sin(yaw) * local_y
        ry = np.sin(yaw) * local_x + np.cos(yaw) * local_y
        if abs(rx) > size[0] * 0.5 or abs(ry) > size[1] * 0.5:
            return None
        return float(center[2] + size[2] * 0.5)

    @staticmethod
    def _box_height_grid(box: Mapping[str, Any], grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray | None:
        center = box["center"]
        size = box["size"]
        rotation = box["rotation"]
        yaw = -np.deg2rad(rotation[2])
        local_x = grid_x - center[0]
        local_y = grid_y - center[1]
        rx = np.cos(yaw) * local_x - np.sin(yaw) * local_y
        ry = np.sin(yaw) * local_x + np.cos(yaw) * local_y
        mask = (np.abs(rx) <= size[0] * 0.5) & (np.abs(ry) <= size[1] * 0.5)
        if not np.any(mask):
            return None
        return np.where(mask, center[2] + size[2] * 0.5, -np.inf)

    @staticmethod
    def _heightfield_height(heightfield: Mapping[str, Any], x: float, y: float) -> float | None:
        center = heightfield["center"]
        size = heightfield["size"]
        local_x = x - center[0]
        local_y = y - center[1]
        if abs(local_x) > size[0] * 0.5 or abs(local_y) > size[1] * 0.5:
            return None
        cols = heightfield["cols"]
        rows = heightfield["rows"]
        u = (local_x / size[0] + 0.5) * (cols - 1)
        v = (local_y / size[1] + 0.5) * (rows - 1)
        c0 = int(np.clip(np.floor(u), 0, cols - 1))
        r0 = int(np.clip(np.floor(v), 0, rows - 1))
        c1 = min(c0 + 1, cols - 1)
        r1 = min(r0 + 1, rows - 1)
        fu = float(u - c0)
        fv = float(v - r0)
        heights = heightfield["heights"]
        h00 = heights[r0, c0]
        h10 = heights[r0, c1]
        h01 = heights[r1, c0]
        h11 = heights[r1, c1]
        return float(center[2] + heightfield["z_offset"] + (h00 * (1.0 - fu) + h10 * fu) * (1.0 - fv) + (h01 * (1.0 - fu) + h11 * fu) * fv)

    @staticmethod
    def _heightfield_height_grid(heightfield: Mapping[str, Any], grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray | None:
        center = heightfield["center"]
        size = heightfield["size"]
        local_x = grid_x - center[0]
        local_y = grid_y - center[1]
        mask = (np.abs(local_x) <= size[0] * 0.5) & (np.abs(local_y) <= size[1] * 0.5)
        if not np.any(mask):
            return None
        cols = heightfield["cols"]
        rows = heightfield["rows"]
        u = (local_x / size[0] + 0.5) * (cols - 1)
        v = (local_y / size[1] + 0.5) * (rows - 1)
        c0 = np.clip(np.floor(u).astype(np.int64), 0, cols - 1)
        r0 = np.clip(np.floor(v).astype(np.int64), 0, rows - 1)
        c1 = np.minimum(c0 + 1, cols - 1)
        r1 = np.minimum(r0 + 1, rows - 1)
        fu = u - c0
        fv = v - r0
        heights = heightfield["heights"]
        h00 = heights[r0, c0]
        h10 = heights[r0, c1]
        h01 = heights[r1, c0]
        h11 = heights[r1, c1]
        sampled = center[2] + heightfield["z_offset"] + (h00 * (1.0 - fu) + h10 * fu) * (1.0 - fv) + (h01 * (1.0 - fu) + h11 * fu) * fv
        return np.where(mask, sampled, -np.inf)


__all__ = ["HeightScan", "TerrainSampler", "TerrainSpawn"]
