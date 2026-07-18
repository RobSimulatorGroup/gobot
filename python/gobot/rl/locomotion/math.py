"""Small math and scene-tree helpers shared by locomotion workflows."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def find_node_by_name(node: Any, name: str) -> Any | None:
    if node is None:
        return None
    if getattr(node, "name", None) == name:
        return node
    for child in getattr(node, "children", ()):
        found = find_node_by_name(child, name)
        if found is not None:
            return found
    return None


def quaternion_to_yaw(quaternion: Sequence[float] | np.ndarray) -> float:
    w, x, y, z = quaternion
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def quaternion_from_yaw(yaw: float) -> np.ndarray:
    half_yaw = yaw * 0.5
    return np.array(
        [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
        dtype=np.float32,
    )


__all__ = [
    "find_node_by_name",
    "quaternion_from_yaw",
    "quaternion_to_yaw",
]
