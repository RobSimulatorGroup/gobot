"""Declarative data types for Gobot motion planning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math

import numpy as np

Link = tuple[str, str]
JointName = tuple[str, str]


def _normalize_tol(value: float | Sequence[float]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(3, float(arr), dtype=np.float64)
    if arr.shape != (3,) or np.isnan(arr).any() or (arr < 0).any():
        raise ValueError("tolerance must be a nonnegative scalar or 3-vector (inf marks a free axis)")
    return arr


@dataclass(frozen=True)
class PoseConstraint:
    """World-space pose constraint with per-axis tolerances (inf = free axis)."""

    link: Link
    position: Sequence[float] | np.ndarray | None = None
    rotation: Sequence[Sequence[float]] | np.ndarray | None = None
    position_tolerance: float | Sequence[float] = 1e-4
    rotation_tolerance: float | Sequence[float] = 1e-3
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.position is None and self.rotation is None:
            raise ValueError("PoseConstraint requires position, rotation, or both")
        _normalize_tol(self.position_tolerance)
        _normalize_tol(self.rotation_tolerance)
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("weight must be finite and positive")


@dataclass(frozen=True)
class PathConstraint:
    """Apply a PoseConstraint across multiple (or all) trajectory waypoints."""

    constraint: PoseConstraint
    timesteps: slice | Sequence[int] | None = None


@dataclass(frozen=True)
class PlanningGroup:
    """Named subset of movable joints to optimize while holding other joints fixed."""

    name: str
    active_joints: tuple[JointName, ...]


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for hybrid global/local trajectory planning and time-scaling."""

    waypoints: int = 16
    margin: float = 0.002
    world_margin: float | None = None
    collision_substeps: int = 8
    max_segment_length: float | None = None
    iterations: int = 40
    stop_at_endpoints: bool = False
    first_feasible: bool = True
    use_global_seeding: bool = True
    num_seeds: int = 1
    rrt_timeout_s: float = 0.35
    rrt_step_size: float = 0.25
    rrt_max_iterations: int = 4000
    rrt_shortcut_iters: int = 64
    velocity_scale: float = 1.0
    acceleration_scale: float = 1.0
    jerk_scale: float = 1.0
    spline_order: int = 3
    strict_ccd: bool = False
    dt: float | None = None
    retime: bool = True

    def __post_init__(self) -> None:
        if self.spline_order not in (3, 5):
            raise ValueError("spline_order must be 3 (cubic Hermite) or 5 (quintic Hermite)")
        if not math.isfinite(self.velocity_scale) or self.velocity_scale <= 0.0:
            raise ValueError("velocity_scale must be finite and positive")
        if not math.isfinite(self.acceleration_scale) or self.acceleration_scale <= 0.0:
            raise ValueError("acceleration_scale must be finite and positive")
        if not math.isfinite(self.jerk_scale) or self.jerk_scale <= 0.0:
            raise ValueError("jerk_scale must be finite and positive")
        if self.dt is not None and (not math.isfinite(self.dt) or self.dt <= 0.0):
            raise ValueError("dt must be finite and positive")

    @property
    def effective_world_margin(self) -> float:
        return self.margin if self.world_margin is None else self.world_margin


JointConfig = Sequence[float] | np.ndarray
GoalSpec = JointConfig | Sequence[PoseConstraint] | Sequence[Sequence[PoseConstraint]]


@dataclass(frozen=True)
class MotionPlanRequest:
    """Declarative request for single- or multi-goal motion planning."""

    start: JointConfig | Sequence[PoseConstraint]
    goal: GoalSpec
    waypoint_targets: Mapping[int, PoseConstraint | JointConfig] | None = None
    path_constraints: Sequence[PathConstraint] = ()
    seed: np.ndarray | None = None
    start_seed: JointConfig | None = None
    goal_seed: JointConfig | None = None
    group: str | PlanningGroup | Sequence[JointName] | None = None
    config: SolverConfig = field(default_factory=SolverConfig)


@dataclass(frozen=True)
class Trajectory:
    """Time-parameterized joint trajectory supporting C^1 cubic (order 3) and C^2 quintic (order 5) Hermite sampling."""

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    dt: float | None = None
    spline_order: int = 3

    def __post_init__(self) -> None:
        if self.spline_order not in (3, 5):
            raise ValueError("spline_order must be 3 (cubic Hermite) or 5 (quintic Hermite)")

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0]) if len(self.times) else 0.0

    @property
    def num_waypoints(self) -> int:
        return int(self.positions.shape[0])

    def _eval(self, t: float | np.ndarray, *, return_jerk: bool):
        from gobot import _coal_native as _native

        scalar = np.ndim(t) == 0
        ts = np.ascontiguousarray(np.atleast_1d(np.asarray(t, dtype=np.float64)))
        out = _native.evaluate_trajectory(
            np.ascontiguousarray(self.times, dtype=np.float64),
            np.ascontiguousarray(self.positions, dtype=np.float64),
            np.ascontiguousarray(self.velocities, dtype=np.float64),
            np.ascontiguousarray(self.accelerations, dtype=np.float64),
            ts,
            int(self.spline_order),
            bool(return_jerk),
        )
        if scalar:
            return tuple(arr[0] for arr in out)
        return out

    def sample(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate (position, velocity, acceleration) at time(s) t via the shared C++ spline evaluator."""
        return self._eval(t, return_jerk=False)

    def sample_with_jerk(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate (position, velocity, acceleration, jerk) at time(s) t via the shared C++ spline evaluator."""
        return self._eval(t, return_jerk=True)


@dataclass(frozen=True)
class PlanDiagnostics:
    """Timing breakdown, solver counts, and collision witness diagnostics."""

    total_ms: float = 0.0
    ik_ms: float = 0.0
    seed_ms: float = 0.0
    sqp_ms: float = 0.0
    validate_ms: float = 0.0
    retime_ms: float = 0.0
    qp_calls: int = 0
    sqp_iterations: int = 0
    seed_strategy: str = "straight"
    witnesses: np.ndarray | None = None


@dataclass(frozen=True)
class IKResult:
    positions: np.ndarray | None
    reason: str | None = None

    @property
    def success(self) -> bool:
        return self.positions is not None


@dataclass(frozen=True)
class BatchIKResult:
    positions: np.ndarray
    success: np.ndarray
    reasons: tuple[str | None, ...]


@dataclass(frozen=True)
class SyncResult:
    """Summary of an incremental PlanningWorld synchronization."""

    version: int
    topology_version: int
    changed: bool
    poses_updated: int = 0
    shapes_added: int = 0
    shapes_removed: int = 0
    shapes_replaced: int = 0
    robot_bases_updated: int = 0
    export_ms: float = 0.0
    diff_ms: float = 0.0
    commit_ms: float = 0.0


@dataclass(frozen=True)
class PlanResult:
    trajectory: Trajectory | None
    initial_positions: np.ndarray | None = None
    goal_index: int = 0
    reason: str | None = None
    diagnostics: PlanDiagnostics = field(default_factory=PlanDiagnostics)
    world_version: int = 0

    @property
    def success(self) -> bool:
        return self.trajectory is not None

    @property
    def positions(self) -> np.ndarray | None:
        return None if self.trajectory is None else self.trajectory.positions

    @property
    def dt(self) -> float | None:
        return None if self.trajectory is None else self.trajectory.dt


@dataclass(frozen=True)
class BatchPlanResult:
    results: tuple[PlanResult, ...]
    positions: np.ndarray
    success: np.ndarray
    reasons: tuple[str | None, ...]
    initial_positions: np.ndarray
    world_version: int = 0
