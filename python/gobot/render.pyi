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

class RenderBuffer:
    @property
    def name(self) -> str: ...
    @property
    def dtype(self) -> str: ...
    @property
    def device(self) -> str: ...
    @property
    def shape(self) -> list[int]: ...
    @property
    def nbytes(self) -> int: ...
    def numpy(self) -> np.ndarray: ...
    def __dlpack__(self, *, stream: int | None = None): ...
    def __dlpack_device__(self) -> tuple[int, int]: ...

class RenderFrame:
    @property
    def frame_index(self) -> int: ...
    @property
    def instance_id_to_path(self) -> dict[int, str]: ...
    @property
    def semantic_id_to_label(self) -> dict[int, str]: ...
    def keys(self) -> list[str]: ...
    def __getitem__(self, output: str) -> RenderBuffer: ...
    def __contains__(self, output: object) -> bool: ...
    def __len__(self) -> int: ...

class RenderProduct:
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def device(self) -> str: ...

class CameraSensor:
    camera: Any
    root: Any | None
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
    ) -> None: ...
    @property
    def render_product(self) -> RenderProduct: ...
    def capture(self) -> RenderFrame: ...

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
