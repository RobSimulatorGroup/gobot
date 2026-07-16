"""Checkpoint helpers for Go1 terrain curriculum assignments."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


TERRAIN_CURRICULUM_STATE_VERSION = 1


def build_terrain_curriculum_state(
    levels: Any,
    terrain_types: Any,
    *,
    rows: int,
    cols: int,
) -> dict[str, Any]:
    levels_np = np.asarray(levels, dtype=np.int64).reshape(-1)
    types_np = np.asarray(terrain_types, dtype=np.int64).reshape(-1)
    _validate_assignments(levels_np, types_np, rows=rows, cols=cols)
    histogram = np.zeros((int(cols), int(rows)), dtype=np.int64)
    np.add.at(histogram, (types_np, levels_np), 1)
    return {
        "version": TERRAIN_CURRICULUM_STATE_VERSION,
        "rows": int(rows),
        "cols": int(cols),
        "num_envs": int(levels_np.size),
        "levels": levels_np.copy(),
        "types": types_np.copy(),
        "level_histogram_by_type": histogram,
    }


def restore_terrain_curriculum_assignments(
    state: Mapping[str, Any],
    current_levels: Any,
    current_types: Any,
    *,
    rows: int,
    cols: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    version = int(state.get("version", 0))
    if version != TERRAIN_CURRICULUM_STATE_VERSION:
        raise RuntimeError(f"unsupported terrain curriculum checkpoint version {version}")
    saved_rows = int(state.get("rows", -1))
    saved_cols = int(state.get("cols", -1))
    if (saved_rows, saved_cols) != (int(rows), int(cols)):
        raise RuntimeError(
            "terrain curriculum checkpoint grid does not match the current scene: "
            f"saved={saved_rows}x{saved_cols}, current={int(rows)}x{int(cols)}"
        )

    current_levels_np = np.asarray(current_levels, dtype=np.int64).reshape(-1)
    current_types_np = np.asarray(current_types, dtype=np.int64).reshape(-1)
    _validate_assignments(current_levels_np, current_types_np, rows=rows, cols=cols)

    saved_levels = np.asarray(state.get("levels", ()), dtype=np.int64).reshape(-1)
    saved_types = np.asarray(state.get("types", ()), dtype=np.int64).reshape(-1)
    if saved_levels.size == current_levels_np.size and saved_types.size == current_types_np.size:
        _validate_assignments(saved_levels, saved_types, rows=rows, cols=cols)
        return saved_levels.copy(), saved_types.copy(), True

    histogram = np.asarray(state.get("level_histogram_by_type", ()), dtype=np.int64)
    if histogram.shape != (int(cols), int(rows)):
        if saved_levels.size == 0 or saved_levels.size != saved_types.size:
            raise RuntimeError("terrain curriculum checkpoint has no usable assignment histogram")
        _validate_assignments(saved_levels, saved_types, rows=rows, cols=cols)
        histogram = np.zeros((int(cols), int(rows)), dtype=np.int64)
        np.add.at(histogram, (saved_types, saved_levels), 1)

    restored_levels = current_levels_np.copy()
    for terrain_type in range(int(cols)):
        target_ids = np.flatnonzero(current_types_np == terrain_type)
        if target_ids.size == 0:
            continue
        counts = _resample_histogram_counts(histogram[terrain_type], int(target_ids.size))
        values = np.repeat(np.arange(int(rows), dtype=np.int64), counts)
        # Keep environment index and difficulty from becoming monotonically correlated.
        np.random.default_rng(0x6B07 + terrain_type).shuffle(values)
        restored_levels[target_ids] = values
    return restored_levels, current_types_np.copy(), False


def _resample_histogram_counts(histogram: np.ndarray, count: int) -> np.ndarray:
    histogram = np.asarray(histogram, dtype=np.float64).reshape(-1)
    if np.any(histogram < 0.0):
        raise RuntimeError("terrain curriculum checkpoint histogram contains negative counts")
    if not np.any(histogram > 0.0):
        histogram = np.ones_like(histogram)
    ideal = histogram / float(histogram.sum()) * int(count)
    counts = np.floor(ideal).astype(np.int64)
    remainder = int(count) - int(counts.sum())
    if remainder > 0:
        counts[np.argsort(-(ideal - counts))[:remainder]] += 1
    return counts


def _validate_assignments(
    levels: np.ndarray,
    terrain_types: np.ndarray,
    *,
    rows: int,
    cols: int,
) -> None:
    if int(rows) <= 0 or int(cols) <= 0:
        raise RuntimeError(f"terrain curriculum grid must be positive, got {rows}x{cols}")
    if levels.size != terrain_types.size:
        raise RuntimeError(
            "terrain curriculum level/type assignment lengths differ: "
            f"{levels.size} != {terrain_types.size}"
        )
    if np.any((levels < 0) | (levels >= int(rows))):
        raise RuntimeError("terrain curriculum checkpoint contains an out-of-range level")
    if np.any((terrain_types < 0) | (terrain_types >= int(cols))):
        raise RuntimeError("terrain curriculum checkpoint contains an out-of-range terrain type")


__all__ = [
    "TERRAIN_CURRICULUM_STATE_VERSION",
    "build_terrain_curriculum_state",
    "restore_terrain_curriculum_assignments",
]
