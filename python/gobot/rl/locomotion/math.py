"""Math and scene-value helpers for locomotion tasks."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def _json_vec(value: object, size: int) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("matrix_data", {}).get("storage", [])
    vector = np.zeros(size, dtype=np.float64)
    array = np.asarray(value if value is not None else [], dtype=np.float64).reshape(-1)
    vector[: min(size, array.size)] = array[:size]
    return vector


def _find_node_by_name(node: Any, name: str):
    if node is None:
        return None
    if getattr(node, "name", None) == name:
        return node
    for child in getattr(node, "children", []):
        found = _find_node_by_name(child, name)
        if found is not None:
            return found
    return None


def _as_vec(value: object, size: int) -> np.ndarray:
    if isinstance(value, Mapping) and "position" in value:
        value = value.get("position")
    array = np.asarray(value if value is not None else [], dtype=np.float32).reshape(-1)
    out = np.zeros(size, dtype=np.float32)
    out[: min(size, array.size)] = array[:size]
    return out


def _quat(base_or_quat: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
    if isinstance(base_or_quat, Mapping):
        transform = base_or_quat.get("global_transform", {})
        value = transform.get("quaternion", [1.0, 0.0, 0.0, 0.0])
    else:
        value = base_or_quat
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size < 4:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return array[:4]


def _quat_to_rp(q: np.ndarray | Sequence[float]) -> tuple[float, float]:
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return float(roll), float(pitch)


def _quat_to_yaw(q: np.ndarray | Sequence[float]) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _quat_from_yaw(yaw: float) -> np.ndarray:
    half_yaw = yaw * 0.5
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float32)


def _rotate_vec_by_quat_inv(v: np.ndarray, q: np.ndarray | Sequence[float]) -> np.ndarray:
    w, x, y, z = q
    return _quat_rotate(v, np.array([w, -x, -y, -z], dtype=np.float32))


def _quat_rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    xyz = np.array([x, y, z], dtype=np.float32)
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def _wrap_to_pi(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


__all__ = [
    "_as_vec",
    "_find_node_by_name",
    "_json_vec",
    "_quat",
    "_quat_from_yaw",
    "_quat_rotate",
    "_quat_to_rp",
    "_quat_to_yaw",
    "_rotate_vec_by_quat_inv",
    "_wrap_to_pi",
]
