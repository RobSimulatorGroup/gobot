"""Optional batched simulation providers."""

from importlib import import_module

from .base import (
    BatchPhysicsProvider,
    BatchProviderCapabilities,
    CompiledControlTopology,
    CompiledRobotTopology,
    CompiledSceneArtifact,
    GraphInvalidatedError,
    ProviderUnavailableError,
    RobotBatchSpec,
    RobotBatchState,
    RobotBatchView,
    SimulationCapacityError,
)
from .mujoco_warp import (
    MuJoCoWarpContactSensorSpec,
    MuJoCoWarpProvider,
    MuJoCoWarpProviderAvailability,
    MuJoCoWarpRaycastSensorSpec,
    MuJoCoWarpRobotLayout,
)
from .newton import (
    NewtonModelConfig,
    NewtonProvider,
    NewtonProviderAvailability,
    NewtonRobotLayout,
)

_IPC_EXPORTS = {
    "CompiledIpcSceneArtifact",
    "DeformableBatchSpec",
    "DeformableBatchState",
    "DeformableBatchView",
    "TactileBatchSpec",
    "TactileBatchState",
    "TactileBatchView",
    "WarpIpcConfig",
    "WarpIpcProvider",
}


def __getattr__(name: str):
    if name in _IPC_EXPORTS:
        value = getattr(import_module("gobot.ipc"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BatchPhysicsProvider",
    "BatchProviderCapabilities",
    "CompiledControlTopology",
    "CompiledRobotTopology",
    "CompiledSceneArtifact",
    "GraphInvalidatedError",
    "CompiledIpcSceneArtifact",
    "DeformableBatchSpec",
    "DeformableBatchState",
    "DeformableBatchView",
    "MuJoCoWarpContactSensorSpec",
    "MuJoCoWarpProvider",
    "MuJoCoWarpProviderAvailability",
    "MuJoCoWarpRaycastSensorSpec",
    "MuJoCoWarpRobotLayout",
    "NewtonModelConfig",
    "NewtonProvider",
    "NewtonProviderAvailability",
    "NewtonRobotLayout",
    "ProviderUnavailableError",
    "RobotBatchSpec",
    "RobotBatchState",
    "RobotBatchView",
    "SimulationCapacityError",
    "TactileBatchSpec",
    "TactileBatchState",
    "TactileBatchView",
    "WarpIpcConfig",
    "WarpIpcProvider",
]
