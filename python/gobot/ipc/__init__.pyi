from __future__ import annotations

from typing import Any, Mapping, Sequence

from gobot.rl.providers import BatchPhysicsProvider, BatchProviderCapabilities

class CompiledIpcSceneArtifact:
    schema_version: int
    producer: str
    producer_version: str
    format: str
    manifest: str
    manifest_sha256: str
    blobs: Mapping[str, bytes]
    def __init__(
        self,
        schema_version: int,
        producer: str,
        producer_version: str,
        format: str,
        manifest: str,
        manifest_sha256: str,
        blobs: Mapping[str, bytes] | Sequence[Mapping[str, Any]],
    ) -> None: ...
    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CompiledIpcSceneArtifact: ...
    def to_mapping(self) -> Mapping[str, Any]: ...
    @property
    def digest(self) -> str: ...
    @property
    def manifest_data(self) -> Mapping[str, Any]: ...
    @property
    def deformable_bodies(self) -> tuple[Mapping[str, Any], ...]: ...
    @property
    def tactile_sensors(self) -> tuple[Mapping[str, Any], ...]: ...
    @property
    def robots(self) -> tuple[Mapping[str, Any], ...]: ...

class WarpIpcConfig:
    device: str
    fixed_time_step: float
    gravity: tuple[float, float, float]
    barrier_distance: float
    barrier_stiffness: float
    kinematic_stiffness: float
    friction_coefficient: float
    ccd_tolerance: float
    newton_iterations: int
    cg_iterations: int
    pt_capacity: int
    ee_capacity: int
    hessian_capacity: int
    capture_graphs: bool
    def __init__(
        self,
        device: str = "cuda:0",
        fixed_time_step: float = 0.002,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        barrier_distance: float = 0.001,
        barrier_stiffness: float = 100000.0,
        kinematic_stiffness: float = 100000.0,
        friction_coefficient: float = 0.5,
        ccd_tolerance: float = 0.000001,
        newton_iterations: int = 20,
        cg_iterations: int = 100,
        pt_capacity: int = 131072,
        ee_capacity: int = 131072,
        hessian_capacity: int = 1048576,
        capture_graphs: bool = True,
    ) -> None: ...

class DeformableBatchSpec:
    body_names: tuple[str, ...]
    def __init__(self, body_names: tuple[str, ...]) -> None: ...

class DeformableBatchState:
    position: Any
    velocity: Any
    contact_force: Any
    def __init__(self, position: Any, velocity: Any, contact_force: Any) -> None: ...

class DeformableBatchView:
    spec: DeformableBatchSpec
    @property
    def vertex_counts(self) -> tuple[int, ...]: ...
    @property
    def max_vertex_count(self) -> int: ...
    def read_state(self) -> DeformableBatchState: ...
    def set_kinematic_targets(self, targets: Any, *, target_mask: Any | None = None) -> None: ...
    def reset(self, reset_mask: Any, **state: Any) -> Mapping[str, Any]: ...
    def bind_scene(self, context: Any, bodies: Sequence[Any]) -> DeformableBatchView: ...
    def sync_scene(
        self,
        context: Any | None = None,
        bodies: Sequence[Any] | None = None,
        *,
        env_index: int = 0,
    ) -> None: ...

class TactileBatchSpec:
    sensor_names: tuple[str, ...]
    def __init__(self, sensor_names: tuple[str, ...]) -> None: ...

class TactileBatchState:
    rgb: Any
    depth: Any
    normal: Any
    marker_position: Any
    marker_flow: Any
    contact_force: Any
    contact_wrench: Any
    def __init__(
        self,
        rgb: Any,
        depth: Any,
        normal: Any,
        marker_position: Any,
        marker_flow: Any,
        contact_force: Any,
        contact_wrench: Any,
    ) -> None: ...

class TactileBatchView:
    spec: TactileBatchSpec
    @property
    def resolution(self) -> tuple[int, int]: ...
    @property
    def gel_vertex_count(self) -> int: ...
    @property
    def marker_count(self) -> int: ...
    def read_state(self) -> TactileBatchState: ...
    def render(self) -> TactileBatchState: ...
    def reset(self, reset_mask: Any, **state: Any) -> Mapping[str, Any]: ...

class WarpIpcProvider(BatchPhysicsProvider):
    artifact: CompiledIpcSceneArtifact
    config: WarpIpcConfig
    def __init__(
        self,
        artifact: Mapping[str, Any] | CompiledIpcSceneArtifact,
        *,
        num_envs: int,
        config: WarpIpcConfig | None = None,
        device: str | None = None,
        capture_graphs: bool | None = None,
        _session: Any | None = None,
    ) -> None: ...
    @classmethod
    def from_context(cls, context: Any, **kwargs: Any) -> WarpIpcProvider: ...
    @property
    def capabilities(self) -> BatchProviderCapabilities: ...
    @property
    def num_envs(self) -> int: ...
    @property
    def generation(self) -> int: ...
    @property
    def fixed_time_step(self) -> float: ...
    @property
    def runtime_fingerprint(self) -> str: ...
    @property
    def graph_captured(self) -> bool: ...
    @property
    def capacities(self) -> Mapping[str, int]: ...
    @property
    def diagnostics(self) -> Mapping[str, Any]: ...
    @property
    def arrays(self) -> Mapping[str, Any]: ...
    def create_deformable_view(
        self,
        spec: DeformableBatchSpec | None = None,
        *,
        body_names: Sequence[str] | None = None,
    ) -> DeformableBatchView: ...
    def create_tactile_view(
        self,
        spec: TactileBatchSpec | None = None,
        *,
        sensor_names: Sequence[str] | None = None,
    ) -> TactileBatchView: ...
    def step(self, actions: Any | None = None, *, nsteps: int = 1) -> Mapping[str, Any]: ...
    def reset(self, reset_mask: Any, **state: Any) -> Mapping[str, Any]: ...
    def close(self) -> None: ...

__all__: list[str]
