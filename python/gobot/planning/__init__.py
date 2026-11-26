"""Unified C++-backed motion planning for Gobot scenes (Coal CPU & MuJoCo Warp GPU)."""

from .backends import CoalBackend, CollisionBackend, MuJoCoWarpBackend, make_backend
from .gpu_batch import GpuBatchBackend, GpuBatchConfig, GpuBatchOptimizer, GpuBatchStats
from .planner import MotionPlanner
from .retime import retime_trajectory, slice_trajectory_suffix
from .types import (
    BatchIKResult,
    BatchPlanResult,
    CartesianPlanConfig,
    CartesianPlanResult,
    CartesianWaypoint,
    IKResult,
    JointName,
    Link,
    LinkSphere,
    MotionPlanRequest,
    PathConstraint,
    PlanDiagnostics,
    PlanningGroup,
    PlanResult,
    PoseConstraint,
    SolverConfig,
    SyncResult,
    Trajectory,
)
from .world import PlanningWorld

__all__ = [
    "BatchIKResult",
    "BatchPlanResult",
    "CartesianPlanConfig",
    "CartesianPlanResult",
    "CartesianWaypoint",
    "CoalBackend",
    "CollisionBackend",
    "GpuBatchBackend",
    "GpuBatchConfig",
    "GpuBatchOptimizer",
    "GpuBatchStats",
    "IKResult",
    "JointName",
    "Link",
    "LinkSphere",
    "MotionPlanner",
    "MotionPlanRequest",
    "MuJoCoWarpBackend",
    "PathConstraint",
    "PlanDiagnostics",
    "PlanResult",
    "PlanningGroup",
    "PlanningWorld",
    "PoseConstraint",
    "SolverConfig",
    "SyncResult",
    "Trajectory",
    "make_backend",
    "retime_trajectory",
    "slice_trajectory_suffix",
]
