from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

Vector3Like = Sequence[float]
ColorLike = Sequence[float]

@dataclass(frozen=True)
class DebugArrow:
    start: Vector3Like
    vector: Vector3Like
    color: ColorLike = (1.0, 1.0, 1.0, 1.0)
    scale: float = 1.0
    label: str = ""

    def to_dict(self) -> dict[str, object]: ...

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
) -> np.ndarray: ...

def set_debug_arrows(debug_arrows: Sequence[DebugArrowLike]) -> None: ...
def clear_debug_arrows() -> None: ...
