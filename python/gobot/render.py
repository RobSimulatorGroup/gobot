"""Rendering helpers for Gobot Python workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    import numpy as np

    from ._core import RenderBuffer, RenderFrame, RenderProduct

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

_NATIVE_RENDER_TYPES = {"RenderBuffer", "RenderFrame", "RenderProduct"}


def __getattr__(name: str) -> Any:
    if name in _NATIVE_RENDER_TYPES:
        try:
            return getattr(_core, name)
        except AttributeError as error:
            raise AttributeError(
                f"gobot.render.{name} requires a Gobot native module with RenderProduct support"
            ) from error
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class CameraSensor:
    """Synchronous pinhole camera backed by a reusable render product."""

    def __init__(
        self,
        camera: Any,
        *,
        root: Any | None = None,
        width: int = 640,
        height: int = 480,
        outputs: Sequence[str] = ("rgb",),
        device: str = "auto",
        mode: str = "minimal",
        frame_slots: int = 3,
    ) -> None:
        self.camera = camera
        self.root = root
        sensor_type = getattr(_core, "_CameraSensor", None)
        if sensor_type is None:
            raise RuntimeError(
                "gobot.render.CameraSensor requires a Gobot native module with RenderProduct support"
            )
        self._impl = sensor_type(
            camera,
            root,
            width,
            height,
            tuple(outputs),
            device,
            mode,
            frame_slots,
        )

    @property
    def render_product(self) -> RenderProduct:
        return self._impl.render_product

    def capture(self) -> RenderFrame:
        return self._impl.capture()


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
    """Capture an RGB uint8 image through a one-output CPU render product."""

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


__all__ = [
    "CameraSensor",
    "DebugArrow",
    "RenderBuffer",
    "RenderFrame",
    "RenderProduct",
    "capture_rgb",
    "clear_debug_arrows",
    "set_debug_arrows",
]
