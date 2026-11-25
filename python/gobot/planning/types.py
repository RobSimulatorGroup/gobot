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
    dt: float | None = None
    retime: bool = True

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
    """Time-parameterized joint trajectory supporting continuous cubic Hermite sampling."""

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    dt: float | None = None

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0]) if len(self.times) else 0.0

    @property
    def num_waypoints(self) -> int:
        return int(self.positions.shape[0])

    def sample(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate (position, velocity, acceleration) at time(s) t via cubic Hermite splines."""
        scalar = np.ndim(t) == 0
        ts = np.atleast_1d(np.asarray(t, dtype=np.float64))
        ts = np.clip(ts, self.times[0], self.times[-1])
        idx = np.searchsorted(self.times, ts, side="right") - 1
        idx = np.clip(idx, 0, len(self.times) - 2)
        t0 = self.times[idx]
        h = np.maximum(self.times[idx + 1] - t0, 1e-12)
        u = ((ts - t0) / h)[:, None]
        h_col = h[:, None]

        p0, p1 = self.positions[idx], self.positions[idx + 1]
        v0, v1 = self.velocities[idx], self.velocities[idx + 1]

        u2 = u * u
        u3 = u2 * u
        h00 = 2 * u3 - 3 * u2 + 1
        h10 = u3 - 2 * u2 + u
        h01 = -2 * u3 + 3 * u2
        h11 = u3 - u2
        pos = h00 * p0 + h10 * h_col * v0 + h01 * p1 + h11 * h_col * v1

        dh00 = (6 * u2 - 6 * u) / h_col
        dh10 = 3 * u2 - 4 * u + 1
        dh01 = (-6 * u2 + 6 * u) / h_col
        dh11 = 3 * u2 - 2 * u
        vel = dh00 * p0 + dh10 * v0 + dh01 * p1 + dh11 * v1

        ddh00 = (12 * u - 6) / (h_col * h_col)
        ddh10 = (6 * u - 4) / h_col
        ddh01 = (-12 * u + 6) / (h_col * h_col)
        ddh11 = (6 * u - 2) / h_col
        acc = ddh00 * p0 + ddh10 * v0 + ddh01 * p1 + ddh11 * v1

        if scalar:
            return pos[0], vel[0], acc[0]
        return pos, vel, acc


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
class PlanResult:
    trajectory: Trajectory | None
    initial_positions: np.ndarray | None = None
    goal_index: int = 0
    reason: str | None = None
    diagnostics: PlanDiagnostics = field(default_factory=PlanDiagnostics)

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
