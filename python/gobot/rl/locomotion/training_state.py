"""Checkpoint helpers for locomotion terrain curriculum assignments."""

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
    return {
        "version": TERRAIN_CURRICULUM_STATE_VERSION,
        "rows": int(rows),
        "cols": int(cols),
        "num_envs": int(levels_np.size),
        "levels": levels_np.copy(),
        "types": types_np.copy(),
    }


def restore_terrain_curriculum_assignments(
    state: Mapping[str, Any],
    current_levels: Any,
    current_types: Any,
    *,
    rows: int,
    cols: int,
) -> tuple[np.ndarray, np.ndarray]:
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
    if saved_levels.size != current_levels_np.size or saved_types.size != current_types_np.size:
        raise RuntimeError("terrain curriculum checkpoint environment count does not match")
    _validate_assignments(saved_levels, saved_types, rows=rows, cols=cols)
    return saved_levels.copy(), saved_types.copy()


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
