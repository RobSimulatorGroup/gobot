"""Backend-neutral simulation helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import math
import operator
from typing import Any

from ._core import JointControllerGains


class ProviderUnavailableError(RuntimeError):
    """Raised when an explicitly requested simulation provider is unavailable."""


@dataclass(frozen=True)
class ProviderCapabilities:
    """Backend-neutral capabilities reported by an external simulation provider."""

    name: str
    device: str
    device_native: bool
    graph_capture: bool
    masked_reset: bool
    fixed_capacity: bool
    runtime_checkpoint: bool = False
    exact_contact_wrench: bool = False
    sensor_batch: bool = False
    solver_substeps: bool = False
    graph_capture_reason: str = ""
    reset_scope: str = "full"


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
        self._status = "Starting"

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
        self._publish_diagnostics()
        return self

    def set_status(self, status: str) -> None:
        """Publish a human-readable provider state without touching device data."""

        normalized = str(status).strip()
        if not normalized:
            raise ValueError("provider session status must not be empty")
        self._status = normalized
        if self.running:
            self._publish_diagnostics()

    def _publish_diagnostics(self) -> None:
        publish = getattr(self.context, "_set_external_simulation_diagnostics", None)
        if not callable(publish) or self._token is None:
            return
        capabilities = getattr(self.provider, "capabilities", None)
        capacities = getattr(self.provider, "capacities", {})
        if callable(capacities):
            capacities = capacities()
        graph_enabled = bool(getattr(capabilities, "graph_capture", False))
        graph_captured = bool(getattr(self.provider, "graph_captured", graph_enabled))
        if graph_captured:
            graph_status = "Captured"
        elif graph_enabled:
            graph_status = "Pending"
        else:
            graph_status = "Disabled"
        publish(
            self._token,
            {
                "provider_name": str(getattr(capabilities, "name", type(self.provider).__name__)),
                "device": str(getattr(capabilities, "device", "")),
                "environment_count": int(getattr(self.provider, "num_envs", 0)),
                "controlled_joint_count": int(
                    getattr(self.provider, "_last_robot_view_joint_count", 0)
                ),
                "capacities": json.dumps(dict(capacities), sort_keys=True, separators=(",", ":")),
                "graph_status": graph_status,
                "status": self._status,
            },
        )

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


__all__ = [
    "JointControllerGains",
    "ProviderCapabilities",
    "ProviderPlaySession",
    "ProviderUnavailableError",
]
