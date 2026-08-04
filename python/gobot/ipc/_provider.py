"""Warp IPC provider lifecycle and configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import operator
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from gobot.rl.providers.base import (
    BatchPhysicsProvider,
    BatchProviderCapabilities,
    GraphInvalidatedError,
    ProviderUnavailableError,
    RobotBatchSpec,
)

from ._artifact import CompiledIpcSceneArtifact, validate_ipc_artifact
from ._views import (
    DeformableBatchSpec,
    DeformableBatchView,
    TactileBatchSpec,
    TactileBatchView,
)


@dataclass(frozen=True)
class WarpIpcConfig:
    """Fixed-capacity configuration for one Warp IPC provider session."""

    device: str = "cuda:0"
    fixed_time_step: float = 0.002
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    barrier_distance: float = 1.0e-3
    barrier_stiffness: float = 1.0e5
    kinematic_stiffness: float = 1.0e5
    friction_coefficient: float = 0.5
    ccd_tolerance: float = 1.0e-6
    newton_iterations: int = 20
    cg_iterations: int = 100
    pt_capacity: int = 131072
    ee_capacity: int = 131072
    hessian_capacity: int = 1048576
    capture_graphs: bool = True

    def __post_init__(self) -> None:
        device = str(self.device)
        if not device:
            raise ValueError("Warp IPC device must not be empty")
        gravity = tuple(float(value) for value in self.gravity)
        if len(gravity) != 3 or not all(math.isfinite(value) for value in gravity):
            raise ValueError("Warp IPC gravity must contain three finite values")
        for name in (
            "fixed_time_step",
            "barrier_distance",
            "barrier_stiffness",
            "kinematic_stiffness",
            "ccd_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Warp IPC {name} must be finite and positive")
        friction = float(self.friction_coefficient)
        if not math.isfinite(friction) or friction < 0.0:
            raise ValueError("Warp IPC friction_coefficient must be finite and non-negative")
        for name in ("newton_iterations", "cg_iterations"):
            original = getattr(self, name)
            try:
                value = operator.index(original)
            except TypeError as error:
                raise TypeError(f"Warp IPC {name} must be an integer") from error
            if isinstance(original, bool) or value <= 0:
                raise ValueError(f"Warp IPC {name} must be a positive integer")
            object.__setattr__(self, name, value)
        for name in ("pt_capacity", "ee_capacity", "hessian_capacity"):
            original = getattr(self, name)
            try:
                value = operator.index(original)
            except TypeError as error:
                raise TypeError(f"Warp IPC {name} must be an integer") from error
            if isinstance(original, bool) or value < 0:
                raise ValueError(f"Warp IPC {name} must be a non-negative integer")
            object.__setattr__(self, name, value)
        if not isinstance(self.capture_graphs, bool):
            raise TypeError("Warp IPC capture_graphs must be a bool")
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "gravity", gravity)
        object.__setattr__(self, "fixed_time_step", float(self.fixed_time_step))
        object.__setattr__(self, "barrier_distance", float(self.barrier_distance))
        object.__setattr__(self, "barrier_stiffness", float(self.barrier_stiffness))
        object.__setattr__(self, "kinematic_stiffness", float(self.kinematic_stiffness))
        object.__setattr__(self, "friction_coefficient", friction)
        object.__setattr__(self, "ccd_tolerance", float(self.ccd_tolerance))


@dataclass(frozen=True)
class _Availability:
    available: bool
    reason: str = ""


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _storage_pointer(value: Any) -> int:
    data_ptr = getattr(value, "data_ptr", None)
    if callable(data_ptr):
        return int(data_ptr())
    pointer = getattr(value, "ptr", None)
    if pointer is not None:
        return int(pointer)
    return id(value)


def _storage_signature(
    arrays: Mapping[str, Any],
) -> tuple[tuple[str, int, tuple[int, ...], str, str], ...]:
    signature = []
    for name, value in sorted(arrays.items()):
        shape = tuple(int(item) for item in getattr(value, "shape", ()))
        dtype = str(
            getattr(value, "dtype", getattr(getattr(value, "array", None), "dtype", ""))
        ).lower()
        device = str(getattr(value, "device", ""))
        signature.append((str(name), _storage_pointer(value), shape, dtype, device))
    return tuple(signature)


def _resolve_manifest_entries(
    values: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    description: str,
) -> tuple[Mapping[str, Any], ...]:
    resolved = []
    for requested in names:
        name = str(requested)
        path_matches = [value for value in values if str(value.get("path", "")) == name]
        matches = path_matches or [
            value for value in values if str(value.get("name", "")) == name
        ]
        if not matches:
            raise KeyError(f"compiled IPC artifact has no {description} {name!r}")
        if len(matches) != 1:
            paths = sorted(str(value.get("path", "")) for value in matches)
            raise ValueError(
                f"compiled IPC artifact {description} name {name!r} is ambiguous; "
                f"use one of the full paths {paths}"
            )
        resolved.append(matches[0])
    return tuple(resolved)


class WarpIpcProvider(BatchPhysicsProvider):
    """Persistent fixed-capacity session consuming only a compiled IPC artifact.

    The numerical runtime is loaded lazily. This module deliberately does not
    import Torch, Warp, or Newton while defining the public contracts.
    """

    def __init__(
        self,
        artifact: Mapping[str, Any] | CompiledIpcSceneArtifact,
        *,
        num_envs: int,
        config: WarpIpcConfig | None = None,
        device: str | None = None,
        capture_graphs: bool | None = None,
        _session: Any | None = None,
    ) -> None:
        try:
            resolved_num_envs = operator.index(num_envs)
        except TypeError as error:
            raise TypeError("num_envs must be an integer") from error
        if isinstance(num_envs, bool) or resolved_num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if config is not None and not isinstance(config, WarpIpcConfig):
            raise TypeError("config must be a WarpIpcConfig")
        resolved_config = config or WarpIpcConfig()
        overrides: dict[str, Any] = {}
        if device is not None:
            overrides["device"] = str(device)
        if capture_graphs is not None:
            if not isinstance(capture_graphs, bool):
                raise TypeError("capture_graphs must be a bool")
            overrides["capture_graphs"] = capture_graphs
        if overrides:
            resolved_config = replace(resolved_config, **overrides)

        self.artifact = validate_ipc_artifact(artifact)
        self.config = resolved_config
        self._num_envs = resolved_num_envs
        self._closed = False
        self._generation = 1
        self._step_count = 0
        self._last_robot_view_joint_count = 0
        fingerprint_value = {
            "artifact": self.artifact.digest,
            "config": asdict(self.config),
            "num_envs": self._num_envs,
            "provider": "warp-ipc",
            "schema_version": 1,
        }
        self._runtime_fingerprint = "sha256:" + hashlib.sha256(
            json.dumps(
                fingerprint_value, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()

        self._session = _session if _session is not None else self._create_session()
        raw_session_envs = getattr(self._session, "num_envs", self._num_envs)
        try:
            session_envs = operator.index(raw_session_envs)
        except TypeError as error:
            self._close_session()
            raise RuntimeError(
                "Warp IPC runtime returned a non-integer environment count"
            ) from error
        if isinstance(raw_session_envs, bool):
            self._close_session()
            raise RuntimeError("Warp IPC runtime returned an invalid environment count")
        if session_envs != self._num_envs:
            self._close_session()
            raise RuntimeError(
                f"Warp IPC runtime environment count mismatch: {session_envs}/{self._num_envs}"
            )
        arrays = getattr(self._session, "arrays", None)
        if callable(arrays):
            arrays = arrays()
        if not isinstance(arrays, Mapping):
            self._close_session()
            raise RuntimeError("Warp IPC runtime did not expose a stable array mapping")
        self._arrays = MappingProxyType(dict(arrays))
        self._storage_signature = _storage_signature(self._arrays)
        if self.config.capture_graphs and not self.graph_captured:
            self._close_session()
            raise ProviderUnavailableError(
                "Warp IPC graph capture was explicitly requested but the runtime did not capture a tick"
            )

    @classmethod
    def from_context(cls, context: Any, **kwargs: Any) -> "WarpIpcProvider":
        compile_artifact = getattr(context, "compile_ipc_scene_artifact", None)
        if not callable(compile_artifact):
            raise RuntimeError("Gobot AppContext has no compile-only IPC scene artifact API")
        return cls(compile_artifact(), **kwargs)

    @staticmethod
    def availability() -> _Availability:
        missing = [
            module
            for module in ("newton", "warp")
            if importlib.util.find_spec(module) is None
        ]
        if missing:
            return _Availability(False, "missing Python package(s): " + ", ".join(missing))
        versions = {
            "newton": _distribution_version("newton"),
            "warp-lang": _distribution_version("warp-lang"),
        }
        if versions["newton"] != "1.4.0" or versions["warp-lang"] != "1.15.0":
            return _Availability(
                False,
                "Warp IPC requires newton==1.4.0 and warp-lang==1.15.0; "
                f"found newton={versions['newton']!r}, warp-lang={versions['warp-lang']!r}",
            )
        runtime = importlib.import_module("gobot.ipc._runtime")
        runtime_availability = getattr(runtime, "availability", None)
        if callable(runtime_availability):
            available, reason = runtime_availability()
            if not available:
                return _Availability(False, str(reason))
        return _Availability(True)

    def _create_session(self) -> Any:
        availability = self.availability()
        if not availability.available:
            raise ProviderUnavailableError(
                "Warp IPC requested but its runtime is unavailable: "
                f"{availability.reason}"
            )
        try:
            runtime = importlib.import_module("gobot.ipc._runtime")
            return runtime.create_session(self.artifact, self.config, self._num_envs)
        except ProviderUnavailableError:
            raise
        except Exception as error:
            raise ProviderUnavailableError(
                "Warp IPC dependencies are installed but the runtime could not start: "
                f"{type(error).__name__}: {error}"
            ) from error

    @property
    def capabilities(self) -> BatchProviderCapabilities:
        return BatchProviderCapabilities(
            name="Warp IPC",
            device=self.config.device,
            device_native=True,
            graph_capture=self.config.capture_graphs,
            masked_reset=True,
            fixed_capacity=True,
        )

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def fixed_time_step(self) -> float:
        return self.config.fixed_time_step

    @property
    def runtime_fingerprint(self) -> str:
        return self._runtime_fingerprint

    @property
    def graph_captured(self) -> bool:
        return bool(getattr(self._session, "graph_captured", False))

    @property
    def capacities(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                "pt": self.config.pt_capacity,
                "ee": self.config.ee_capacity,
                "hessian": self.config.hessian_capacity,
                "newton_iterations": self.config.newton_iterations,
                "cg_iterations": self.config.cg_iterations,
            }
        )

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        runtime_values = getattr(self._session, "diagnostics", {})
        if callable(runtime_values):
            runtime_values = runtime_values()
        values = dict(runtime_values) if isinstance(runtime_values, Mapping) else {}
        values.update(
            {
                "provider": "Warp IPC",
                "device": self.config.device,
                "num_envs": self._num_envs,
                "graph_captured": self.graph_captured,
                "step_count": self._step_count,
            }
        )
        return MappingProxyType(values)

    @property
    def arrays(self) -> Mapping[str, Any]:
        self._require_open()
        self._validate_storage()
        return self._arrays

    def create_deformable_view(
        self,
        spec: DeformableBatchSpec | None = None,
        *,
        body_names: Sequence[str] | None = None,
    ) -> DeformableBatchView:
        self._require_open()
        if spec is not None and body_names is not None:
            raise TypeError("pass either DeformableBatchSpec or body_names, not both")
        if spec is None:
            if body_names is None:
                raise TypeError("body_names are required")
            spec = DeformableBatchSpec(tuple(body_names))
        entries = _resolve_manifest_entries(
            self.artifact.deformable_bodies, spec.body_names, "deformable body"
        )
        adapter = self._session.create_deformable_view_adapter(spec, entries)
        return DeformableBatchView(
            self,
            spec,
            adapter,
            vertex_counts=tuple(int(entry["vertex_count"]) for entry in entries),
        )

    def create_tactile_view(
        self,
        spec: TactileBatchSpec | None = None,
        *,
        sensor_names: Sequence[str] | None = None,
    ) -> TactileBatchView:
        self._require_open()
        if spec is not None and sensor_names is not None:
            raise TypeError("pass either TactileBatchSpec or sensor_names, not both")
        if spec is None:
            if sensor_names is None:
                raise TypeError("sensor_names are required")
            spec = TactileBatchSpec(tuple(sensor_names))
        entries = _resolve_manifest_entries(
            self.artifact.tactile_sensors, spec.sensor_names, "tactile sensor"
        )
        first = entries[0]
        expected = (
            tuple(int(value) for value in first["resolution"]),
            str(first.get("gel_topology_sha256", first["gel_mesh_blob"])),
            int(first["gel_vertex_count"]),
            len(first["marker_positions"]),
        )
        for entry in entries[1:]:
            actual = (
                tuple(int(value) for value in entry["resolution"]),
                str(entry.get("gel_topology_sha256", entry["gel_mesh_blob"])),
                int(entry["gel_vertex_count"]),
                len(entry["marker_positions"]),
            )
            if actual != expected:
                raise ValueError(
                    "all sensors in one tactile view must have identical resolution, "
                    "gel topology, and marker count"
                )
        adapter = self._session.create_tactile_view_adapter(spec, entries)
        return TactileBatchView(
            self,
            spec,
            adapter,
            resolution=expected[0],
            gel_vertex_count=expected[2],
            marker_count=expected[3],
        )

    def _create_robot_view_adapter(self, spec: RobotBatchSpec) -> Any:
        self._require_open()
        create_adapter = getattr(self._session, "create_robot_view_adapter", None)
        if not callable(create_adapter):
            raise NotImplementedError("this Warp IPC runtime does not provide robot batch views")
        return create_adapter(spec)

    def step(self, actions: Any | None = None, *, nsteps: int = 1) -> Mapping[str, Any]:
        self._require_open()
        try:
            resolved_nsteps = operator.index(nsteps)
        except TypeError as error:
            raise TypeError("nsteps must be an integer") from error
        if isinstance(nsteps, bool) or resolved_nsteps <= 0:
            raise ValueError("nsteps must be a positive integer")
        self._validate_storage()
        self._session.step(actions, nsteps=resolved_nsteps)
        self._step_count += resolved_nsteps
        self._validate_storage()
        return self._arrays

    def reset(self, reset_mask: Any, **state: Any) -> Mapping[str, Any]:
        self._require_open()
        self._validate_storage()
        self._session.reset(reset_mask, **state)
        self._validate_storage()
        return self._arrays

    def close(self) -> None:
        if self._closed:
            return
        self._close_session()
        self._closed = True
        self._generation += 1

    def _close_session(self) -> None:
        close = getattr(self._session, "close", None)
        if callable(close):
            close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Warp IPC provider is closed")

    def _validate_storage(self) -> None:
        arrays = getattr(self._session, "arrays", None)
        if callable(arrays):
            arrays = arrays()
        if not isinstance(arrays, Mapping):
            raise GraphInvalidatedError("Warp IPC runtime replaced its array mapping")
        current = _storage_signature(arrays)
        if current != self._storage_signature:
            raise GraphInvalidatedError(
                "Warp IPC storage changed after provider creation; rebuild the provider and graph"
            )


__all__ = ["WarpIpcConfig", "WarpIpcProvider"]
