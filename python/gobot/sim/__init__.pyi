from collections.abc import Callable
from typing import Any

from .._core import (
    JointControllerGains, SimulationCompletion, SimulationEnvironmentClock,
    SimulationEnvironmentProgress, SimulationMicrostepResult, SimulationRuntimeInfo,
    SimulationSnapshot, SimulationStepResult,
)
from .session import SimulationSession, SimulationStepError
from .runtime import AsyncSimulationSession, SimulationRuntimeSpec, RuntimeComponents
from .scene_snapshot import SceneStateOutput, SceneSnapshotSync
from .providers import (
    BatchPhysicsProvider, CompiledSceneArtifact, CompiledMuJoCoIpcArtifact,
    MuJoCoWarpProvider, MuJoCoIpcProvider, MuJoCoIpcConfig,
    RobotBatchSpec, RobotBatchState, RobotBatchView,
    SensorBatchSpec, SensorBatchState, SensorBatchView,
)

class ProviderUnavailableError(RuntimeError): ...

class ProviderCapabilities:
    name: str
    device: str
    device_native: bool
    graph_capture: bool
    masked_reset: bool
    fixed_capacity: bool
    def __init__(
        self,
        name: str,
        device: str,
        device_native: bool,
        graph_capture: bool,
        masked_reset: bool,
        fixed_capacity: bool,
    ) -> None: ...

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
        before_step: Callable[[float], Any] | None = ...,
        reset: Callable[[], None] | None = ...,
        sync_scene: Callable[..., None] | None = ...,
        close_provider: bool = ...,
    ) -> None: ...
    @property
    def running(self) -> bool: ...
    def start(self) -> ProviderPlaySession: ...
    def set_status(self, status: str) -> None: ...
    def reset(self) -> None: ...
    def sync_scene(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> ProviderPlaySession: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool: ...

__all__: list[str]
