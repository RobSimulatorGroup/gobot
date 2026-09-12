"""Data-only Python provider adapter to the C++ simulation session contract."""

from __future__ import annotations

import math
import operator
import threading
from typing import Any

from .._core import _SimulationExecutor, _SimulationSession, SimulationMicrostepResult


class SimulationStepError(RuntimeError):
    """A failed step retaining the exact completed progress for every environment."""

    def __init__(self, result: Any):
        self.result = result
        messages = dict.fromkeys(env.error for env in result.environments if env.error)
        super().__init__("; ".join(messages) or "Simulation step failed")


class _ProviderExecutor(_SimulationExecutor):
    def __init__(self, provider: Any, controller: Any = None):
        super().__init__()
        self.provider = provider
        self.controller = controller
        self.actions = None

    def advance(self, dt: float):
        actions, self.actions = self.actions, None
        if self.controller is not None:
            actions = self.controller.step(dt)
        advance = getattr(self.provider, "advance_microstep", None)
        if advance is not None:
            return advance(dt, actions)
        # Providers without substeps advance one full step. An exception is
        # intentionally allowed through: C++ marks unknown advancement invalid.
        if actions is None:
            self.provider.step(nsteps=1)
        else:
            self.provider.step(actions, nsteps=1)
        result = []
        for _ in range(self.provider.num_envs):
            value = SimulationMicrostepResult()
            value.completed = True
            value.advanced_time = dt
            result.append(value)
        return result

    def reset(self, environments):
        selected = set(environments)
        self.provider.reset([env in selected for env in range(self.provider.num_envs)])
        self.actions = None
        if self.controller is not None:
            self.controller.reset(tuple(environments))

    def close(self):
        self.actions = None
        self.provider.close()


class SimulationSession:
    """One owner thread, one provider, one fixed-step clock implementation.

    Create the provider on the same thread as this session. Editor execution
    uses a worker-side factory; a live provider or SceneTree must not cross the
    thread boundary. ``step`` returns progress, while states remain available
    through provider views. No scene synchronization or CPU snapshot is implicit.
    """

    def __init__(self, provider: Any, *, controller: Any = None):
        self._owner = threading.get_ident()
        self._provider = provider
        dt = getattr(provider, "fixed_time_step", None)
        if dt is None:
            dt = getattr(provider, "_fixed_time_step", None)
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("provider must expose a finite positive fixed_time_step")
        configured_substeps = getattr(provider, "session_substeps", 1)
        substeps = operator.index(configured_substeps)
        if isinstance(configured_substeps, bool) or substeps <= 0:
            raise ValueError("session_substeps must be a positive integer")
        self._executor = _ProviderExecutor(provider, controller)
        self._session = _SimulationSession(self._executor, provider.num_envs, dt / substeps, substeps)
        self.last_result = None

    def _require_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("SimulationSession must be used on its owner thread")

    def step(self, actions=None, *, nsteps: int = 1, raise_on_failure: bool = True):
        self._require_owner()
        count = operator.index(nsteps)
        if isinstance(nsteps, bool) or count <= 0:
            raise ValueError("nsteps must be a positive integer")
        self._executor.actions = actions
        try:
            result = self._session.step(count)
        finally:
            self._executor.actions = None
        self.last_result = result
        if not result.completed and raise_on_failure:
            raise SimulationStepError(result)
        return result

    def reset(self, environments=None):
        self._require_owner()
        if environments is None:
            indices = []
        else:
            indices = []
            for value in environments:
                index = operator.index(value)
                if isinstance(value, bool) or index < 0:
                    raise ValueError("environment indices must be non-negative integers")
                indices.append(index)
            if not indices:
                return  # An explicit empty selection is a no-op, not full reset.
        self._session.reset(indices)
        self.last_result = None

    @property
    def clocks(self):
        self._require_owner()
        return self._session.clocks

    @property
    def fixed_time_step(self):
        return self._session.fixed_time_step

    def close(self):
        self._require_owner()
        self._session.close()

    def __enter__(self):
        self._require_owner()
        return self

    def __exit__(self, *_):
        self.close()
