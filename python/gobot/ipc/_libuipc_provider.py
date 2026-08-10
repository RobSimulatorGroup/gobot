"""Native libuipc provider backed by Gobot's private C++ solver module."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import operator
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from gobot import _core
from gobot.sim import (
    ProviderCapabilities,
    ProviderUnavailableError,
)

from ._artifact import (
    CompiledIpcSceneArtifact,
    validate_ipc_artifact,
)


@dataclass(frozen=True)
class LibuipcConfig:
    """Configuration for one native libuipc CUDA session."""

    fixed_time_step: float = 0.01
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    friction_coefficient: float = 0.5
    contact_activation_distance: float = 0.01
    contact_resistance: float = 1.0e9
    affine_stiffness: float = 1.0e8
    kinematic_strength: float = 100.0
    device_index: int = 0
    workspace: str = ""
    backend_module_directory: str = ""
    module_path: str = ""

    def __post_init__(self) -> None:
        gravity = tuple(float(value) for value in self.gravity)
        if len(gravity) != 3 or not all(math.isfinite(value) for value in gravity):
            raise ValueError("libuipc gravity must contain three finite values")
        for name in (
            "fixed_time_step",
            "contact_activation_distance",
            "contact_resistance",
            "affine_stiffness",
            "kinematic_strength",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"libuipc {name} must be finite and positive")
            object.__setattr__(self, name, value)
        friction = float(self.friction_coefficient)
        if not math.isfinite(friction) or friction < 0.0:
            raise ValueError(
                "libuipc friction_coefficient must be finite and non-negative"
            )
        try:
            device_index = operator.index(self.device_index)
        except TypeError as error:
            raise TypeError("libuipc device_index must be an integer") from error
        if isinstance(self.device_index, bool) or device_index < 0:
            raise ValueError("libuipc device_index must be non-negative")
        object.__setattr__(self, "gravity", gravity)
        object.__setattr__(self, "friction_coefficient", friction)
        object.__setattr__(self, "device_index", device_index)
        for name in ("workspace", "backend_module_directory", "module_path"):
            object.__setattr__(self, name, str(getattr(self, name)))

    def solver_mapping(self) -> Mapping[str, Any]:
        values = asdict(self)
        values.pop("module_path")
        return values


@dataclass(frozen=True)
class LibuipcProviderAvailability:
    available: bool
    reason: str = ""


def _normalized_body_info(values: Any, description: str) -> tuple[Mapping[str, Any], ...]:
    try:
        entries = tuple(dict(value) for value in values)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"libuipc returned invalid {description} metadata") from error
    paths: list[str] = []
    for entry in entries:
        path = str(entry.get("path", ""))
        offset = entry.get("element_offset")
        count = entry.get("element_count")
        if (
            not path
            or isinstance(offset, bool)
            or isinstance(count, bool)
            or not isinstance(offset, int)
            or not isinstance(count, int)
            or offset < 0
            or count <= 0
        ):
            raise RuntimeError(f"libuipc returned invalid {description} metadata")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise RuntimeError(f"libuipc returned duplicate {description} paths")
    return entries


def _affine_transforms_to_poses(transforms: Any) -> Any:
    import numpy as np

    values = np.asarray(transforms, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4):
        raise RuntimeError("libuipc affine transforms must have shape [count,4,4]")
    if not np.isfinite(values).all():
        raise RuntimeError("libuipc affine transforms contain non-finite values")

    poses = np.empty((len(values), 7), dtype=np.float32)
    poses[:, :3] = values[:, :3, 3]
    for index, linear in enumerate(values[:, :3, :3]):
        left, _, right = np.linalg.svd(linear)
        if np.linalg.det(left @ right) < 0.0:
            left[:, -1] *= -1.0
        rotation = left @ right
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            quaternion = np.asarray(
                (
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    0.25 * scale,
                )
            )
        else:
            axis = int(np.argmax(np.diag(rotation)))
            first = axis
            second = (axis + 1) % 3
            third = (axis + 2) % 3
            scale = math.sqrt(
                max(
                    0.0,
                    1.0
                    + rotation[first, first]
                    - rotation[second, second]
                    - rotation[third, third],
                )
            ) * 2.0
            if scale <= np.finfo(np.float64).eps:
                raise RuntimeError("libuipc affine transform has an invalid rotation")
            quaternion = np.zeros(4, dtype=np.float64)
            quaternion[first] = 0.25 * scale
            quaternion[second] = (
                rotation[second, first] + rotation[first, second]
            ) / scale
            quaternion[third] = (
                rotation[third, first] + rotation[first, third]
            ) / scale
            quaternion[3] = (
                rotation[third, second] - rotation[second, third]
            ) / scale
        quaternion /= np.linalg.norm(quaternion)
        poses[index, 3:] = quaternion
    return poses


class LibuipcProvider:
    """Single-environment provider using the native libuipc solver module."""

    accepts_device_actions = False

    def __enter__(self) -> "LibuipcProvider":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.close()
        return False

    def __init__(
        self,
        artifact: Mapping[str, Any] | CompiledIpcSceneArtifact,
        *,
        config: LibuipcConfig | None = None,
        _session: Any | None = None,
    ) -> None:
        if config is not None and not isinstance(config, LibuipcConfig):
            raise TypeError("config must be a LibuipcConfig")
        self.artifact = validate_ipc_artifact(artifact)
        self.config = config or LibuipcConfig()
        if self.artifact.tactile_sensors:
            raise ValueError(
                "libuipc does not yet consume Gobot tactile sensor artifacts"
            )
        self._closed = False
        self._generation = 1
        self._bound_context: Any | None = None
        self._bound_bodies: tuple[Any, ...] | None = None
        self._bound_affine_links: tuple[Any, ...] | None = None
        self._joints = tuple(
            joint
            for robot in self.artifact.robots
            for joint in robot["joints"]
        )
        self._last_robot_view_joint_count = len(self._joints)
        self._session = _session if _session is not None else self._create_session()
        try:
            self._deformable_bodies = _normalized_body_info(
                self._session.deformable_bodies, "deformable body"
            )
            self._affine_bodies = _normalized_body_info(
                self._session.affine_bodies, "affine body"
            )
            self._validate_session_layout()
            self._arrays = MappingProxyType(
                {
                    "positions": self._session.positions,
                    "velocities": self._session.velocities,
                    "contact_forces": self._session.deformable_contact_forces,
                    "affine_transforms": self._session.affine_transforms,
                }
            )
            self._validate_arrays()
        except Exception:
            self._close_session()
            raise
        fingerprint = {
            "artifact": self.artifact.digest,
            "config": asdict(self.config),
            "provider": "libuipc",
            "schema_version": 2,
        }
        self._runtime_fingerprint = "sha256:" + hashlib.sha256(
            json.dumps(
                fingerprint, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()

    @classmethod
    def availability(
        cls, config: LibuipcConfig | None = None
    ) -> LibuipcProviderAvailability:
        resolved = config or LibuipcConfig()
        session_type = getattr(_core, "_IpcSolverSession", None)
        if session_type is None:
            return LibuipcProviderAvailability(
                False, "this Gobot build has no native IPC solver bindings"
            )
        try:
            available = bool(
                session_type.is_module_available(resolved.module_path)
            )
        except Exception as error:
            return LibuipcProviderAvailability(False, str(error))
        if not available:
            module = resolved.module_path or "the bundled libuipc solver module"
            return LibuipcProviderAvailability(False, f"cannot load {module}")
        return LibuipcProviderAvailability(True)

    @classmethod
    def from_context(cls, context: Any, **kwargs: Any) -> "LibuipcProvider":
        compile_artifact = getattr(context, "compile_ipc_scene_artifact", None)
        if not callable(compile_artifact):
            raise RuntimeError(
                "this Gobot build does not expose IPC scene compilation"
            )
        return cls(compile_artifact(), **kwargs)

    def _create_session(self) -> Any:
        availability = self.availability(self.config)
        if not availability.available:
            raise ProviderUnavailableError(availability.reason)
        return _core._IpcSolverSession(
            dict(self.artifact.to_mapping()),
            dict(self.config.solver_mapping()),
            self.config.module_path,
        )

    def _validate_session_layout(self) -> None:
        expected_deformables = {
            str(entry["path"]): int(entry["vertex_count"])
            for entry in self.artifact.deformable_bodies
        }
        actual_deformables = {
            str(entry["path"]): int(entry["element_count"])
            for entry in self._deformable_bodies
        }
        if actual_deformables != expected_deformables:
            raise RuntimeError(
                "libuipc deformable body layout does not match the compiled artifact"
            )

        expected_affine_paths = {
            str(link["path"])
            for robot in self.artifact.robots
            for link in robot["links"]
            if any(
                not bool(shape.get("disabled", False))
                for shape in link["collision_shapes"]
            )
        }
        actual_affine_paths = {
            str(entry["path"]) for entry in self._affine_bodies
        }
        if actual_affine_paths != expected_affine_paths:
            raise RuntimeError(
                "libuipc affine body layout does not match the compiled artifact"
            )

    def _validate_arrays(self) -> None:
        total_vertices = sum(
            int(entry["element_count"]) for entry in self._deformable_bodies
        )
        expected_shapes = {
            "positions": (total_vertices, 3),
            "velocities": (total_vertices, 3),
            "contact_forces": (total_vertices, 3),
            "affine_transforms": (len(self._affine_bodies), 4, 4),
        }
        for name, shape in expected_shapes.items():
            value = self._arrays[name]
            actual = tuple(int(component) for component in getattr(value, "shape", ()))
            if actual != shape:
                raise RuntimeError(
                    f"libuipc {name} array has shape {actual}, expected {shape}"
                )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="libuipc",
            device=f"cuda:{self.config.device_index}",
            device_native=True,
            graph_capture=False,
            masked_reset=False,
            fixed_capacity=True,
        )

    @property
    def num_envs(self) -> int:
        return 1

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
        return False

    @property
    def capacities(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                "deformable_bodies": len(self._deformable_bodies),
                "deformable_vertices": sum(
                    int(entry["element_count"])
                    for entry in self._deformable_bodies
                ),
                "affine_bodies": len(self._affine_bodies),
                "joints": len(self._joints),
            }
        )

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        self._require_open()
        values = dict(self._session.diagnostics)
        values.update(
            {
                "provider": "libuipc",
                "device": f"cuda:{self.config.device_index}",
                "num_envs": 1,
                "graph_captured": False,
            }
        )
        return MappingProxyType(values)

    @property
    def arrays(self) -> Mapping[str, Any]:
        self._require_open()
        return self._arrays

    @property
    def deformable_bodies(self) -> tuple[Mapping[str, Any], ...]:
        return self._deformable_bodies

    @property
    def affine_bodies(self) -> tuple[Mapping[str, Any], ...]:
        return self._affine_bodies

    @property
    def joints(self) -> tuple[Mapping[str, Any], ...]:
        return self._joints

    def step(
        self, actions: Any | None = None, *, nsteps: int = 1
    ) -> Mapping[str, Any]:
        self._require_open()
        if actions is not None:
            raise TypeError("libuipc accepts affine or joint targets, not action tensors")
        try:
            count = operator.index(nsteps)
        except TypeError as error:
            raise TypeError("libuipc step count must be an integer") from error
        if isinstance(nsteps, bool) or count <= 0:
            raise ValueError("libuipc step count must be positive")
        self._session.step(count)
        return self.diagnostics

    def reset(
        self, reset_mask: Any | None = None, **state: Any
    ) -> Mapping[str, Any]:
        self._require_open()
        if state:
            raise TypeError("libuipc reset does not accept custom state")
        if reset_mask is not None:
            try:
                import numpy as np

                mask = np.asarray(reset_mask, dtype=bool)
            except Exception as error:
                raise TypeError("libuipc reset mask must be boolean") from error
            if mask.size != 1 or not bool(mask.reshape(-1)[0]):
                raise ValueError(
                    "libuipc has one environment and supports only a full reset"
                )
        self._session.reset()
        return self.diagnostics

    def set_affine_target(self, path: str, transform: Any) -> None:
        self._require_open()
        requested = str(path)
        if requested not in {
            str(entry["path"]) for entry in self._affine_bodies
        }:
            raise KeyError(f"libuipc scene has no affine body {requested!r}")
        self._session.set_affine_target(requested, transform)

    def set_joint_target(self, path: str, position: float) -> None:
        self._require_open()
        requested = str(path)
        if requested not in {str(entry["path"]) for entry in self._joints}:
            raise KeyError(f"libuipc scene has no joint {requested!r}")
        value = float(position)
        if not math.isfinite(value):
            raise ValueError("libuipc joint target must be finite")
        self._session.set_joint_target(requested, value)

    def bind_scene(
        self,
        context: Any,
        bodies: Sequence[Any],
        affine_links: Sequence[Any] | None = None,
    ) -> "LibuipcProvider":
        resolved = tuple(bodies)
        if len(resolved) != len(self._deformable_bodies):
            raise ValueError(
                f"libuipc scene sync requires {len(self._deformable_bodies)} "
                f"deformable bodies, got {len(resolved)}"
            )
        self._bound_context = context
        self._bound_bodies = resolved
        if affine_links is not None:
            resolved_links = tuple(affine_links)
            if len(resolved_links) != len(self._affine_bodies):
                raise ValueError(
                    f"libuipc scene sync requires {len(self._affine_bodies)} "
                    f"affine links, got {len(resolved_links)}"
                )
            self._bound_affine_links = resolved_links
        return self

    def sync_scene(
        self,
        context: Any | None = None,
        bodies: Sequence[Any] | None = None,
        affine_links: Sequence[Any] | None = None,
    ) -> None:
        self._require_open()
        resolved_context = self._bound_context if context is None else context
        resolved_bodies = self._bound_bodies if bodies is None else tuple(bodies)
        resolved_links = (
            self._bound_affine_links
            if affine_links is None
            else tuple(affine_links)
        )
        if resolved_context is None or resolved_bodies is None:
            raise RuntimeError("libuipc provider has no bound scene context and bodies")
        if len(resolved_bodies) != len(self._deformable_bodies):
            raise ValueError(
                f"libuipc scene sync requires {len(self._deformable_bodies)} "
                f"deformable bodies, got {len(resolved_bodies)}"
            )

        import numpy as np

        counts = tuple(
            int(entry["element_count"]) for entry in self._deformable_bodies
        )
        padded = np.zeros((len(counts), max(counts), 3), dtype=np.float32)
        positions = np.asarray(self._arrays["positions"])
        for body_index, (entry, count) in enumerate(
            zip(self._deformable_bodies, counts, strict=True)
        ):
            offset = int(entry["element_offset"])
            padded[body_index, :count] = positions[offset : offset + count]
        apply_vertices = getattr(
            resolved_context, "apply_deformable_vertices", None
        )
        if not callable(apply_vertices):
            apply_vertices = getattr(
                resolved_context, "_apply_deformable_vertex_batch", None
            )
        if not callable(apply_vertices):
            raise RuntimeError(
                "this Gobot build does not expose deformable scene synchronization"
            )
        apply_vertices(resolved_bodies, padded, counts)

        if resolved_links is not None:
            if len(resolved_links) != len(self._affine_bodies):
                raise ValueError(
                    f"libuipc scene sync requires {len(self._affine_bodies)} "
                    f"affine links, got {len(resolved_links)}"
                )
            apply_poses = getattr(resolved_context, "apply_link_poses", None)
            if not callable(apply_poses):
                apply_poses = getattr(
                    resolved_context, "_apply_link_pose_batch", None
                )
            if not callable(apply_poses):
                raise RuntimeError(
                    "this Gobot build does not expose affine scene synchronization"
                )
            apply_poses(
                resolved_links,
                _affine_transforms_to_poses(self._arrays["affine_transforms"]),
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._generation += 1
        self._bound_context = None
        self._bound_bodies = None
        self._bound_affine_links = None
        self._close_session()

    def _close_session(self) -> None:
        session = getattr(self, "_session", None)
        self._session = None
        if hasattr(self, "_arrays"):
            self._arrays = MappingProxyType({})
        if session is not None:
            session.close()

    def _require_open(self) -> None:
        if self._closed or self._session is None:
            raise RuntimeError("libuipc provider is closed")


__all__ = [
    "LibuipcConfig",
    "LibuipcProvider",
    "LibuipcProviderAvailability",
]
