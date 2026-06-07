"""Rendering helpers for Gobot Python workflows."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import _core


Vector3Like = Sequence[float]


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
) -> np.ndarray:
    """Render a scene to an RGB uint8 image.

    Experimental helper used by training video capture until Gobot has a
    first-class runtime scene owner for headless rendering.
    """

    return _core._capture_rgb(root, width, height, eye, target, up, fov_y, z_near, z_far)


__all__ = ["capture_rgb"]
