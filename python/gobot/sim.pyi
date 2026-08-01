from collections.abc import Callable
from typing import Any

from ._core import JointControllerGains

class ProviderPlaySession:
    context: Any
    provider: Any
    fixed_dt: float
    max_sub_steps: int
    close_provider: bool
    def __init__(
        self,
        context: Any,
        provider: Any,
        *,
        fixed_dt: float,
        max_sub_steps: int = ...,
        before_step: Callable[[float], None] | None = ...,
        reset: Callable[[], None] | None = ...,
        sync_scene: Callable[[], None] | None = ...,
        close_provider: bool = ...,
    ) -> None: ...
    @property
    def running(self) -> bool: ...
    def start(self) -> ProviderPlaySession: ...
    def reset(self) -> None: ...
    def sync_scene(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> ProviderPlaySession: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool: ...

__all__: list[str]
