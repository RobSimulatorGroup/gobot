"""Rendering helpers for Gobot Python workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from . import _core


Vector3Like = Sequence[float]
ColorLike = Sequence[float]


@dataclass(frozen=True)
class DebugArrow:
    start: Vector3Like
    vector: Vector3Like
    color: ColorLike = (1.0, 1.0, 1.0, 1.0)
    scale: float = 1.0
    label: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "start": tuple(float(value) for value in self.start),
            "vector": tuple(float(value) for value in self.vector),
            "color": tuple(float(value) for value in self.color),
            "scale": float(self.scale),
            "label": self.label,
        }


DebugArrowLike = DebugArrow | Mapping[str, Any]


def capture_rgb(
    *,
    root: Any | None = None,
    width: int = 640,
    height: int = 480,
    eye: Vector3Like = (2.4, -3.0, 1.6),
    target: Vector3Like = (0.0, 0.0, 0.5),
    up: Vector3Like = (0.0, 0.0, 1.0),
    fov_y: float = 60.0,
    z_near: float = 0.05,
    z_far: float = 200.0,
    debug_arrows: Sequence[DebugArrowLike] | None = None,
) -> np.ndarray:
    """Render a scene to an RGB uint8 image.

    Experimental helper used by training video capture until Gobot has a
    first-class runtime scene owner for headless rendering.
    """

    return _core._capture_rgb(
        root,
        width,
        height,
        eye,
        target,
        up,
        fov_y,
        z_near,
        z_far,
        _debug_arrows_to_core(debug_arrows),
    )


def _debug_arrows_to_core(debug_arrows: Sequence[DebugArrowLike] | None) -> list[dict[str, object]] | None:
    if debug_arrows is None:
        return None
    result: list[dict[str, object]] = []
    for arrow in debug_arrows:
        if isinstance(arrow, DebugArrow):
            result.append(arrow.to_dict())
        else:
            result.append(dict(arrow))
    return result


def set_debug_arrows(debug_arrows: Sequence[DebugArrowLike]) -> None:
    _core._set_debug_arrows(_debug_arrows_to_core(debug_arrows) or [])


def clear_debug_arrows() -> None:
    _core._clear_debug_arrows()


def _shutdown_headless_render_context() -> None:
    _core._shutdown_headless_render_context()


__all__ = ["DebugArrow", "capture_rgb", "clear_debug_arrows", "set_debug_arrows"]
