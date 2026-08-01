"""Backend-neutral simulation helpers."""

from __future__ import annotations

from collections.abc import Callable
import math
import operator
from typing import Any

from ._core import JointControllerGains


class ProviderPlaySession:
    """Connect a Python physics provider to Gobot's fixed-step Play lifecycle."""

    def __init__(
        self,
        context: Any,
        provider: Any,
        *,
        fixed_dt: float,
        max_sub_steps: int = 8,
        before_step: Callable[[float], None] | None = None,
        reset: Callable[[], None] | None = None,
        sync_scene: Callable[[], None] | None = None,
        close_provider: bool = True,
    ) -> None:
        normalized_fixed_dt = float(fixed_dt)
        if not math.isfinite(normalized_fixed_dt) or normalized_fixed_dt <= 0.0:
            raise ValueError("fixed_dt must be finite and greater than zero")
        try:
            normalized_max_sub_steps = operator.index(max_sub_steps)
        except TypeError as error:
            raise TypeError("max_sub_steps must be an integer") from error
        if isinstance(max_sub_steps, bool) or normalized_max_sub_steps <= 0:
            raise ValueError("max_sub_steps must be a positive integer")
        if not callable(getattr(provider, "step", None)):
            raise TypeError("provider must define step(...)")
        self.context = context
        self.provider = provider
        self.fixed_dt = normalized_fixed_dt
        self.max_sub_steps = normalized_max_sub_steps
        self.before_step = before_step
        self.reset_callback = reset
        self.sync_callback = sync_scene
        self.close_provider = bool(close_provider)
        self._token: int | None = None
        self._closed = False
        self._provider_closed = False

    @property
    def running(self) -> bool:
        return self._token is not None and not self._closed

    def start(self) -> "ProviderPlaySession":
        if self._closed:
            raise RuntimeError("provider play session is closed")
        if self._token is not None:
            return self
        begin = getattr(self.context, "_begin_external_simulation", None)
        if not callable(begin):
            raise RuntimeError("this Gobot build does not support external simulation sessions")
        self._token = int(
            begin(
                self._step,
                self._reset,
                self._sync_scene,
                self._close_from_engine,
                self.fixed_dt,
                self.max_sub_steps,
            )
        )
        return self

    def reset(self) -> None:
        if not self.running:
            raise RuntimeError("provider play session is not running")
        reset_external = getattr(self.context, "_reset_external_simulation", None)
        if not callable(reset_external) or not reset_external(self._token):
            raise RuntimeError("external simulation session reset failed")

    def sync_scene(self) -> None:
        if not self.running:
            raise RuntimeError("provider play session is not running")
        sync_external = getattr(self.context, "_sync_external_simulation", None)
        if not callable(sync_external) or not sync_external(self._token):
            raise RuntimeError("external simulation scene sync failed")

    def close(self) -> None:
        if self._closed:
            return
        token, self._token = self._token, None
        try:
            if token is not None:
                end = getattr(self.context, "_end_external_simulation", None)
                if callable(end):
                    end(token)
        finally:
            self._close_from_engine()

    def __enter__(self) -> "ProviderPlaySession":
        return self.start()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.close()
        return False

    def _step(self, fixed_dt: float) -> None:
        if self._closed:
            raise RuntimeError("provider play session is closed")
        if self.before_step is not None:
            self.before_step(float(fixed_dt))
        self.provider.step(nsteps=1)

    def _reset(self) -> None:
        if self.reset_callback is None:
            raise RuntimeError("provider play session has no reset callback")
        self.reset_callback()

    def _sync_scene(self) -> None:
        if self.sync_callback is not None:
            self.sync_callback()

    def _close_from_engine(self) -> None:
        if self._provider_closed:
            self._closed = True
            self._token = None
            return
        self._closed = True
        self._token = None
        if self.close_provider:
            self._provider_closed = True
            self.provider.close()


__all__ = ["JointControllerGains", "ProviderPlaySession"]
