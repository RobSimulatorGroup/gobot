"""Backend-neutral batched views for IPC deformables and tactile sensors."""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Any, Mapping, Sequence

from gobot.rl.providers.base import GraphInvalidatedError


def _tensor_pointer(value: Any) -> int:
    data_ptr = getattr(value, "data_ptr", None)
    if callable(data_ptr):
        return int(data_ptr())
    pointer = getattr(value, "ptr", None)
    if pointer is not None:
        return int(pointer)
    return id(value)


def _tensor_dtype(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        dtype = getattr(getattr(value, "array", None), "dtype", None)
    return str(dtype).lower()


def _validate_tensor(value: Any, expected_shape: tuple[int, ...], description: str) -> None:
    shape = tuple(int(item) for item in getattr(value, "shape", ()))
    if shape != expected_shape:
        raise RuntimeError(
            f"Warp IPC {description} tensor has shape {shape}, expected {expected_shape}"
        )
    if "float32" not in _tensor_dtype(value):
        raise RuntimeError(f"Warp IPC {description} tensor must use float32 storage")


def _state_storage_signature(
    state: Any, field_names: Sequence[str]
) -> tuple[tuple[str, int, tuple[int, ...], str], ...]:
    return tuple(
        (
            name,
            _tensor_pointer(getattr(state, name)),
            tuple(int(item) for item in getattr(getattr(state, name), "shape", ())),
            _tensor_dtype(getattr(state, name)),
        )
        for name in field_names
    )


def _validate_unique_names(values: Sequence[str], description: str) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if not names or any(not value for value in names):
        raise ValueError(f"{description} names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError(f"{description} names must be unique")
    return names


class _ArtifactBoundView:
    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self._generation = getattr(provider, "generation", None)
        self._artifact_digest = getattr(getattr(provider, "artifact", None), "digest", None)

    def _validate(self) -> None:
        if getattr(self._provider, "generation", None) != self._generation:
            raise RuntimeError("IPC batch view is stale because its provider was closed or rebuilt")
        digest = getattr(getattr(self._provider, "artifact", None), "digest", None)
        if digest != self._artifact_digest:
            raise RuntimeError("IPC batch view cannot be used with a different compiled artifact")


@dataclass(frozen=True)
class DeformableBatchSpec:
    body_names: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "body_names",
            _validate_unique_names(self.body_names, "deformable body"),
        )


@dataclass(frozen=True)
class DeformableBatchState:
    """Stable float32 device tensors in world coordinates."""

    position: Any
    velocity: Any
    contact_force: Any


class DeformableBatchView(_ArtifactBoundView):
    def __init__(
        self,
        provider: Any,
        spec: DeformableBatchSpec,
        adapter: Any,
        *,
        vertex_counts: Sequence[int],
    ) -> None:
        super().__init__(provider)
        self.spec = spec
        self._adapter = adapter
        self._vertex_counts = tuple(int(value) for value in vertex_counts)
        if len(self._vertex_counts) != len(spec.body_names) or any(
            value <= 0 for value in self._vertex_counts
        ):
            raise ValueError("deformable view vertex counts do not match its bodies")
        self._state: DeformableBatchState | None = None
        self._state_storage_signature: tuple[
            tuple[str, int, tuple[int, ...], str], ...
        ] | None = None
        self._scene_context: Any | None = None
        self._scene_bodies: tuple[Any, ...] | None = None

    @property
    def vertex_counts(self) -> tuple[int, ...]:
        return self._vertex_counts

    @property
    def max_vertex_count(self) -> int:
        return max(self._vertex_counts)

    def read_state(self) -> DeformableBatchState:
        self._validate()
        previous = self._state
        updated = self._adapter.read_state(previous)
        expected_shape = (
            int(self._provider.num_envs),
            len(self.spec.body_names),
            self.max_vertex_count,
            3,
        )
        for field_name in ("position", "velocity", "contact_force"):
            _validate_tensor(
                getattr(updated, field_name), expected_shape, f"deformable {field_name}"
            )
        signature = _state_storage_signature(
            updated, ("position", "velocity", "contact_force")
        )
        if self._state_storage_signature is None:
            self._state_storage_signature = signature
        elif signature != self._state_storage_signature:
            raise GraphInvalidatedError(
                "Warp IPC deformable state storage changed after its first read"
            )
        self._state = updated
        return self._state

    def set_kinematic_targets(self, targets: Any, *, target_mask: Any | None = None) -> None:
        self._validate()
        self._adapter.set_kinematic_targets(targets, target_mask=target_mask)

    def reset(self, reset_mask: Any, **state: Any) -> Mapping[str, Any]:
        self._validate()
        return self._adapter.reset(reset_mask, **state)

    def bind_scene(self, context: Any, bodies: Sequence[Any]) -> "DeformableBatchView":
        if len(bodies) != len(self.spec.body_names):
            raise ValueError(
                f"deformable scene sync requires {len(self.spec.body_names)} bodies, "
                f"got {len(bodies)}"
            )
        self._scene_context = context
        self._scene_bodies = tuple(bodies)
        return self

    def sync_scene(
        self,
        context: Any | None = None,
        bodies: Sequence[Any] | None = None,
        *,
        env_index: int = 0,
    ) -> None:
        """Explicitly read back and synchronize one selected environment."""

        self._validate()
        resolved_context = self._scene_context if context is None else context
        resolved_bodies = self._scene_bodies if bodies is None else tuple(bodies)
        if resolved_context is None or resolved_bodies is None:
            raise RuntimeError("deformable batch view has no bound scene context and bodies")
        if len(resolved_bodies) != len(self.spec.body_names):
            raise ValueError(
                f"deformable scene sync requires {len(self.spec.body_names)} bodies, "
                f"got {len(resolved_bodies)}"
            )
        try:
            index = operator.index(env_index)
        except TypeError as error:
            raise TypeError("scene sync environment index must be an integer") from error
        if isinstance(env_index, bool):
            raise TypeError("scene sync environment index must be an integer")
        if not 0 <= index < int(self._provider.num_envs):
            raise IndexError(
                f"scene sync environment index {index} is outside [0, {self._provider.num_envs})"
            )
        positions = self.read_state().position[index].detach().cpu().numpy()
        apply_vertices = getattr(resolved_context, "apply_deformable_vertices", None)
        if not callable(apply_vertices):
            apply_vertices = getattr(resolved_context, "_apply_deformable_vertex_batch", None)
        if not callable(apply_vertices):
            raise RuntimeError(
                "this Gobot build does not expose AppContext.apply_deformable_vertices"
            )
        apply_vertices(resolved_bodies, positions, self._vertex_counts)


@dataclass(frozen=True)
class TactileBatchSpec:
    sensor_names: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sensor_names",
            _validate_unique_names(self.sensor_names, "tactile sensor"),
        )


@dataclass(frozen=True)
class TactileBatchState:
    """Stable float32 tensors updated only when the caller explicitly renders."""

    rgb: Any
    depth: Any
    normal: Any
    marker_position: Any
    marker_flow: Any
    contact_force: Any
    contact_wrench: Any


class TactileBatchView(_ArtifactBoundView):
    def __init__(
        self,
        provider: Any,
        spec: TactileBatchSpec,
        adapter: Any,
        *,
        resolution: tuple[int, int],
        gel_vertex_count: int,
        marker_count: int,
    ) -> None:
        super().__init__(provider)
        self.spec = spec
        self._adapter = adapter
        self._resolution = (int(resolution[0]), int(resolution[1]))
        self._gel_vertex_count = int(gel_vertex_count)
        self._marker_count = int(marker_count)
        if any(value <= 0 for value in self._resolution) or self._gel_vertex_count <= 0:
            raise ValueError("tactile view has invalid image or gel dimensions")
        if self._marker_count < 0:
            raise ValueError("tactile view marker count cannot be negative")
        self._state: TactileBatchState | None = None
        self._state_storage_signature: tuple[
            tuple[str, int, tuple[int, ...], str], ...
        ] | None = None

    @property
    def resolution(self) -> tuple[int, int]:
        """Image resolution as ``(height, width)``."""

        return self._resolution

    @property
    def gel_vertex_count(self) -> int:
        return self._gel_vertex_count

    @property
    def marker_count(self) -> int:
        return self._marker_count

    def read_state(self) -> TactileBatchState:
        """Return current tensors without triggering tactile rendering."""

        self._validate()
        previous = self._state
        updated = self._adapter.read_state(previous)
        num_envs = int(self._provider.num_envs)
        sensor_count = len(self.spec.sensor_names)
        height, width = self._resolution
        expected_shapes = {
            "rgb": (num_envs, sensor_count, height, width, 3),
            "depth": (num_envs, sensor_count, height, width),
            "normal": (num_envs, sensor_count, height, width, 3),
            "marker_position": (num_envs, sensor_count, self._marker_count, 2),
            "marker_flow": (num_envs, sensor_count, self._marker_count, 2),
            "contact_force": (
                num_envs,
                sensor_count,
                self._gel_vertex_count,
                3,
            ),
            "contact_wrench": (num_envs, sensor_count, 6),
        }
        field_names = tuple(expected_shapes)
        for field_name, expected_shape in expected_shapes.items():
            _validate_tensor(
                getattr(updated, field_name), expected_shape, f"tactile {field_name}"
            )
        signature = _state_storage_signature(updated, field_names)
        if self._state_storage_signature is None:
            self._state_storage_signature = signature
        elif signature != self._state_storage_signature:
            raise GraphInvalidatedError(
                "Warp IPC tactile state storage changed after its first read"
            )
        self._state = updated
        return self._state

    def render(self) -> TactileBatchState:
        """Explicitly update RGB, depth, normal, marker, and contact outputs."""

        self._validate()
        self._adapter.render()
        return self.read_state()

    def reset(self, reset_mask: Any, **state: Any) -> Mapping[str, Any]:
        self._validate()
        return self._adapter.reset(reset_mask, **state)


__all__ = [
    "DeformableBatchSpec",
    "DeformableBatchState",
    "DeformableBatchView",
    "TactileBatchSpec",
    "TactileBatchState",
    "TactileBatchView",
]
