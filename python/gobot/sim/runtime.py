"""Worker-owned physics providers using Gobot's native bounded mailbox.

Compile scene artifacts on the scene owner thread, then pass their JSON data
to an importable factory. A factory must return a provider or RuntimeComponents;
it must not capture/access a live Node, SceneTree, AppContext or editor callback.
"""
from __future__ import annotations

import atexit
import base64
from dataclasses import dataclass
import importlib
import json
import math
import operator
import threading
from typing import Any, Mapping

from .._core import _SimulationDataWorker, SimulationCompletion, SimulationRuntimeInfo
from .session import _ProviderExecutor


def _pack_configuration(value):
    # Artifacts contain binary mesh blobs. Encode data explicitly, never pickle
    # live Python objects; wrap every container so authored keys cannot collide
    # with the transport's byte tag.
    if isinstance(value, bytes):
        return {"kind": "bytes", "value": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("runtime configuration keys must be strings")
        return {"kind": "map", "value": {key: _pack_configuration(item) for key, item in value.items()}}
    if isinstance(value, (tuple, list)):
        return {"kind": "list", "value": [_pack_configuration(item) for item in value]}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"runtime configuration must contain data, got {type(value).__name__}")


def _unpack_configuration(value):
    if not isinstance(value, dict):
        return value
    kind, payload = value["kind"], value["value"]
    if kind == "bytes":
        return base64.b64decode(payload, validate=True)
    if kind == "map":
        return {key: _unpack_configuration(item) for key, item in payload.items()}
    if kind == "list":
        return [_unpack_configuration(item) for item in payload]
    raise ValueError("unknown runtime configuration data type")


@dataclass(frozen=True)
class RuntimeComponents:
    """Created by the factory on the runtime thread, never on the scene thread."""
    provider: Any
    controller: Any = None
    output: Any = None


@dataclass(frozen=True, init=False)
class SimulationRuntimeSpec:
    """Immutable, data-only recipe; no Python callable or live provider crosses threads."""
    recipe: str

    def __init__(self, factory: str, arguments: Mapping[str, Any]):
        module, separator, function = factory.partition(":")
        if not separator or not function.isidentifier() or not all(part.isidentifier() for part in module.split(".")):
            raise ValueError("factory must be an importable 'module:function' name")
        payload = json.dumps({"factory": factory, "arguments": _pack_configuration(dict(arguments))}, allow_nan=False)
        object.__setattr__(self, "recipe", payload)


def _indices(values):
    result = []
    for value in values:
        index = operator.index(value)
        if isinstance(value, bool) or index < 0:
            raise ValueError("environment indices must be non-negative integers")
        result.append(index)
    if len(set(result)) != len(result):
        raise ValueError("environment indices must be unique")
    return result


def _subscription(fields, environments, max_hz):
    fields = tuple(fields)
    if any(not isinstance(field, str) or not field for field in fields) or len(set(fields)) != len(fields):
        raise ValueError("snapshot fields must be unique non-empty names")
    rate = float(max_hz)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("snapshot max_hz must be finite and positive")
    return fields, _indices(environments), 1. / rate


# Retired workers stay alive until their SDK teardown finishes. Dropping the
# Python Play object after Stop must not implicitly join the UI thread.
_workers: set[Any] = set()
_retired: set[Any] = set()


def _reap_retired():
    for worker in tuple(_retired):
        if not worker.pending:
            worker.shutdown()
            _retired.remove(worker)
            _workers.discard(worker)


def _shutdown_all():
    for worker in tuple(_workers):
        worker.retire()
    for worker in tuple(_workers):
        worker.shutdown()  # C++ releases the GIL while joining.
    _workers.clear()
    _retired.clear()


atexit.register(_shutdown_all)


class AsyncSimulationSession:
    """Nonblocking install/step/reset/stop with explicit completed-state polling.

    ``submit`` accepts a mapping of named numeric command arrays. Only one
    outstanding request/completion is allowed; false means the caller should
    poll before retrying. Snapshot buffers are read-only and retain their frame
    when wrapped with ``numpy.asarray(snapshot.buffer(name))``.
    """

    def __init__(self, spec: SimulationRuntimeSpec, *, fields=(), environments=(0,), max_hz=60.):
        if not isinstance(spec, SimulationRuntimeSpec):
            raise TypeError("spec must be a SimulationRuntimeSpec")
        selected = _subscription(fields, environments, max_hz)
        _reap_retired()
        self._owner = threading.get_ident()
        self._epoch = 1
        self._closed = False
        self._info = None
        self._pending_subscription = None
        self._worker = _SimulationDataWorker()
        _workers.add(self._worker)
        if not self._worker.install(spec.recipe, self._epoch, *selected):
            self.shutdown()
            raise RuntimeError("could not submit simulation installation")

    def _require_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("AsyncSimulationSession must be used on its owner thread")

    def _require_open(self):
        self._require_owner()
        if self._closed:
            raise RuntimeError("asynchronous simulation session is closed")

    @property
    def ready(self) -> bool:
        self._require_owner()
        return not self._closed and self._pending_subscription is None and self._worker.ready

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def info(self) -> SimulationRuntimeInfo | None:
        """Read-only installation metadata; None while the factory is running."""
        self._require_owner()
        return self._info

    def submit(self, commands=None, *, nsteps=1) -> bool:
        self._require_open()
        count = operator.index(nsteps)
        if isinstance(nsteps, bool) or count <= 0:
            raise ValueError("nsteps must be a positive integer")
        if not self._worker.ready:
            return False
        return self._worker.step({} if commands is None else dict(commands), count, self._epoch)

    def poll(self) -> SimulationCompletion | None:
        self._require_owner()
        _reap_retired()
        if self._closed:
            return None
        result = self._worker.poll()
        if result is None or result.epoch != self._epoch:
            return None
        if result.info.environment_count:
            self._info = result.info
        self._flush_subscription()
        return result

    def reset(self, environments=None) -> bool:
        self._require_open()
        selected = [] if environments is None else _indices(environments)
        if environments is not None and not selected:
            return True  # Explicit empty mask is a no-op.
        next_epoch = self._epoch + 1
        accepted = self._worker.reset(selected, next_epoch)
        if accepted:
            self._epoch = next_epoch
        return accepted

    def subscribe(self, fields, *, environments=(0,), max_hz=60.) -> bool:
        """Coalesce output changes and apply them after the in-flight completion.

        A subscription never discards physical advancement. Only the most recent
        pending selection is retained, independent of how often the UI changes it.
        """
        self._require_open()
        self._pending_subscription = _subscription(fields, environments, max_hz)
        self._flush_subscription()
        return True

    def _flush_subscription(self):
        if self._pending_subscription is not None and self._worker.ready:
            if self._worker.subscribe(*self._pending_subscription, self._epoch):
                self._pending_subscription = None

    def close(self) -> None:
        self._require_owner()
        if self._closed:
            return
        self._closed = True
        self._pending_subscription = None
        self._epoch += 1
        self._worker.retire()
        _retired.add(self._worker)

    def shutdown(self) -> None:
        """Join for headless/process shutdown. Editor Stop uses nonblocking close()."""
        self.close()
        self._worker.shutdown()
        _retired.discard(self._worker)
        _workers.discard(self._worker)

    def __del__(self):
        worker = getattr(self, "_worker", None)
        if worker is not None and not getattr(self, "_closed", True):
            worker.retire()
            _retired.add(worker)


class _DataProviderExecutor(_ProviderExecutor):
    def __init__(self, components):
        super().__init__(components.provider, components.controller)
        self.output = components.output
        self.environment_count = operator.index(self.provider.num_envs)
        configured_substeps = getattr(self.provider, "session_substeps", 1)
        self.substeps = operator.index(configured_substeps)
        if isinstance(configured_substeps, bool) or self.substeps <= 0:
            raise ValueError("session_substeps must be a positive integer")
        self.microstep_dt = float(self.provider.fixed_time_step) / self.substeps

    def describe(self):
        capabilities = getattr(self.provider, "capabilities", None)
        capacities = getattr(self.provider, "capacities", {})
        if callable(capacities):
            capacities = capacities()
        enabled = bool(getattr(capabilities, "graph_capture", False))
        captured = bool(getattr(self.provider, "graph_captured", False))
        return {
            "provider_name": str(getattr(capabilities, "name", type(self.provider).__name__)),
            "device": str(getattr(capabilities, "device", "")),
            "graph_status": "Captured" if captured else "Pending" if enabled else "Disabled",
            "controlled_joint_count": int(getattr(self.provider, "_last_robot_view_joint_count", 0)),
            "capacities": dict(capacities),
        }

    def apply_commands(self, commands):
        if not commands:
            return
        if self.controller is not None:
            self.controller.apply_commands(commands)
        elif set(commands) == {"actions"}:
            self.actions = commands["actions"]
        else:
            raise ValueError("provider without a controller accepts only the 'actions' command")

    def snapshot(self, fields, environments):
        if not fields:
            return {}
        if self.output is not None:
            return self.output.snapshot(fields, environments)
        capture = getattr(self.provider, "snapshot_arrays", None)
        if capture is not None:
            return capture(fields, environments)
        return _copy_selected_arrays(self.provider.arrays, fields, environments, self.environment_count)


def _copy_selected_arrays(arrays, fields, environments, environment_count):
    import numpy as np

    result = {}
    for name in fields:
        array = arrays[name]
        if array.shape[0] != environment_count:
            raise ValueError(f"{name}: snapshot arrays must have a leading environment axis")
        selected = array[list(environments)]
        if hasattr(selected, "detach"):
            selected = selected.detach().cpu().numpy()
        result[name] = np.asarray(selected)
    return result


def _create_executor(recipe):
    value = json.loads(recipe)
    module, function = value["factory"].split(":")
    result = getattr(importlib.import_module(module), function)(**_unpack_configuration(value["arguments"]))
    components = result if isinstance(result, RuntimeComponents) else RuntimeComponents(result)
    try:
        return _DataProviderExecutor(components)
    except Exception:
        components.provider.close()
        raise
