from __future__ import annotations

from typing import Any, Mapping, Sequence

from gobot.sim import ProviderCapabilities


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


class LibuipcConfig:
    fixed_time_step: float
    gravity: tuple[float, float, float]
    friction_coefficient: float
    contact_activation_distance: float
    contact_resistance: float
    affine_stiffness: float
    kinematic_strength: float
    device_index: int
    workspace: str
    backend_module_directory: str
    module_path: str
    def __init__(
        self,
        fixed_time_step: float = 0.01,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        friction_coefficient: float = 0.5,
        contact_activation_distance: float = 0.01,
        contact_resistance: float = 1000000000.0,
        affine_stiffness: float = 100000000.0,
        kinematic_strength: float = 100.0,
        device_index: int = 0,
        workspace: str = "",
        backend_module_directory: str = "",
        module_path: str = "",
    ) -> None: ...
    def solver_mapping(self) -> Mapping[str, Any]: ...


class LibuipcBatchConfig:
    solver: LibuipcConfig
    environments_per_shard: int
    contact_constitution: str
    newton_max_iterations: int
    line_search_max_iterations: int
    linear_system_tolerance_rate: float
    export_deformable_state: bool
    export_affine_state: bool
    export_deformable_contact_forces: bool
    def __init__(
        self,
        solver: LibuipcConfig = ...,
        environments_per_shard: int = 64,
        contact_constitution: str = "ipc",
        al_ipc_mu_scale_fem: float = 50000000.0,
        al_ipc_mu_scale_abd: float = 100000.0,
        al_ipc_toi_threshold: float = 0.1,
        al_ipc_alpha_lower_bound: float = 0.000001,
        al_ipc_decay_factor: float = 0.3,
        newton_max_iterations: int = 16,
        line_search_max_iterations: int = 8,
        linear_system_tolerance_rate: float = 0.001,
        export_deformable_state: bool = True,
        export_affine_state: bool = True,
        export_deformable_contact_forces: bool = True,
    ) -> None: ...
    def solver_mapping(self, num_envs: int) -> Mapping[str, Any]: ...


class LibuipcProviderAvailability:
    available: bool
    reason: str
    def __init__(self, available: bool, reason: str = "") -> None: ...


class LibuipcBatchSolver:
    artifact: CompiledIpcSceneArtifact
    config: LibuipcBatchConfig
    accepts_device_targets: bool
    wrench_source: str
    def __init__(
        self,
        artifact: Mapping[str, Any] | CompiledIpcSceneArtifact,
        *,
        num_envs: int,
        config: LibuipcBatchConfig | None = None,
        device: str | None = None,
        _session: Any | None = None,
        _torch: Any | None = None,
    ) -> None: ...
    def __enter__(self) -> LibuipcBatchSolver: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool: ...
    @classmethod
    def availability(
        cls, config: LibuipcBatchConfig | None = None
    ) -> LibuipcProviderAvailability: ...
    @classmethod
    def from_context(cls, context: Any, **kwargs: Any) -> LibuipcBatchSolver: ...
    @property
    def capabilities(self) -> ProviderCapabilities: ...
    @property
    def num_envs(self) -> int: ...
    @property
    def generation(self) -> int: ...
    @property
    def frame(self) -> int: ...
    @property
    def fixed_time_step(self) -> float: ...
    @property
    def gravity(self) -> tuple[float, float, float]: ...
    @property
    def runtime_fingerprint(self) -> str: ...
    @property
    def graph_captured(self) -> bool: ...
    @property
    def shard_count(self) -> int: ...
    @property
    def arrays(self) -> Mapping[str, Any]: ...
    @property
    def deformable_bodies(self) -> tuple[Mapping[str, Any], ...]: ...
    @property
    def affine_bodies(self) -> tuple[Mapping[str, Any], ...]: ...
    @property
    def capacities(self) -> Mapping[str, int]: ...
    @property
    def diagnostics(self) -> Mapping[str, Any]: ...
    def set_affine_targets(self, targets: Any) -> None: ...
    def step(self, *, nsteps: int = 1) -> Mapping[str, Any]: ...
    def refresh_state(self) -> Mapping[str, Any]: ...
    def refresh_deformable_contact_forces(self) -> Any: ...
    def reset(self, reset_mask: Any | None = None) -> Mapping[str, Any]: ...
    def synchronize(self) -> None: ...
    def close(self) -> None: ...


class LibuipcProvider:
    artifact: CompiledIpcSceneArtifact
    config: LibuipcConfig
    accepts_device_actions: bool
    def __init__(
        self,
        artifact: Mapping[str, Any] | CompiledIpcSceneArtifact,
        *,
        config: LibuipcConfig | None = None,
        _session: Any | None = None,
    ) -> None: ...
    def __enter__(self) -> LibuipcProvider: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool: ...
    @classmethod
    def availability(
        cls, config: LibuipcConfig | None = None
    ) -> LibuipcProviderAvailability: ...
    @classmethod
    def from_context(cls, context: Any, **kwargs: Any) -> LibuipcProvider: ...
    @property
    def capabilities(self) -> ProviderCapabilities: ...
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
    @property
    def deformable_bodies(self) -> tuple[Mapping[str, Any], ...]: ...
    @property
    def affine_bodies(self) -> tuple[Mapping[str, Any], ...]: ...
    @property
    def joints(self) -> tuple[Mapping[str, Any], ...]: ...
    def step(
        self, actions: Any | None = None, *, nsteps: int = 1
    ) -> Mapping[str, Any]: ...
    def reset(
        self, reset_mask: Any | None = None, **state: Any
    ) -> Mapping[str, Any]: ...
    def set_affine_target(self, path: str, transform: Any) -> None: ...
    def set_joint_target(self, path: str, position: float) -> None: ...
    def bind_scene(
        self,
        context: Any,
        bodies: Sequence[Any],
        affine_links: Sequence[Any] | None = None,
    ) -> LibuipcProvider: ...
    def sync_scene(
        self,
        context: Any | None = None,
        bodies: Sequence[Any] | None = None,
        affine_links: Sequence[Any] | None = None,
    ) -> None: ...
    def close(self) -> None: ...


__all__: list[str]
