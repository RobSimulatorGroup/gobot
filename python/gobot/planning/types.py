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


def _validate_manifold_constraints(constraints: Sequence[PoseConstraint]) -> None:
    """Validate that manifold constraints use strictly positive tolerances (> 0 or +inf)."""
    for c in constraints:
        if not isinstance(c, PoseConstraint):
            raise TypeError("manifold_constraints entries must be PoseConstraint instances")
        if c.position is not None:
            pt = _normalize_tol(c.position_tolerance)
            if (pt <= 0.0).any():
                raise ValueError(
                    "manifold_constraints require strictly positive position_tolerance (> 0 or inf); "
                    "zero tolerance is not supported"
                )
        if c.rotation is not None:
            rt = _normalize_tol(c.rotation_tolerance)
            if (rt <= 0.0).any():
                raise ValueError(
                    "manifold_constraints require strictly positive rotation_tolerance (> 0 or inf); "
                    "zero tolerance is not supported"
                )


@dataclass(frozen=True)
class MotionPlanRequest:
    """Declarative request for single- or multi-goal motion planning.

    Note: ``manifold_constraints`` must use strictly positive tolerances (``> 0`` or ``inf``);
    zero tolerances are rejected at request construction.
    """

    start: JointConfig | Sequence[PoseConstraint]
    goal: GoalSpec
    waypoint_targets: Mapping[int, PoseConstraint | JointConfig] | None = None
    path_constraints: Sequence[PathConstraint] = ()
    seed: np.ndarray | None = None
    start_seed: JointConfig | None = None
    goal_seed: JointConfig | None = None
    group: str | PlanningGroup | Sequence[JointName] | None = None
    config: SolverConfig = field(default_factory=SolverConfig)
    manifold_constraints: Sequence[PoseConstraint] = ()
    start_velocity: JointConfig | None = None
    start_acceleration: JointConfig | None = None

    def __post_init__(self) -> None:
        if self.manifold_constraints:
            _validate_manifold_constraints(self.manifold_constraints)
        if self.start_acceleration is not None and self.config.spline_order != 5:
            raise ValueError("start_acceleration is only supported when config.spline_order == 5")


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
    timed_out: bool = False


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
    """A successful trajectory maps request slots to knot rows via `waypoint_indices`.

    On a dense BiRRT fallback, free interior slots refer to the nearest knots
    at uniform joint-space arc-length intervals between the endpoints.
    """

    trajectory: Trajectory | None
    initial_positions: np.ndarray | None = None
    goal_index: int = 0
    reason: str | None = None
    diagnostics: PlanDiagnostics = field(default_factory=PlanDiagnostics)
    world_version: int = 0
    waypoint_indices: np.ndarray | None = None

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
    """Batch of individually verified spline trajectories.

    For playback, use `results[i].trajectory.sample(t)`. The `positions` array
    holds the spline knots for inspection, padded by repeating each last knot.
    Use `results[i].trajectory.num_waypoints` for the original row length.
    Linear interpolation of these arrays requires separate polyline validation.
    """

    results: tuple[PlanResult, ...]
    positions: np.ndarray
    success: np.ndarray
    reasons: tuple[str | None, ...]
    initial_positions: np.ndarray
    world_version: int = 0


@dataclass(frozen=True)
class CartesianWaypoint:
    """Target waypoint for linear (MoveL) or directed circular arc (MoveC) Cartesian path planning."""

    position: Sequence[float] | np.ndarray
    rotation: Sequence[Sequence[float]] | np.ndarray | None = None
    motion_type: str = "linear"
    via_position: Sequence[float] | np.ndarray | None = None
    blend_radius: float = 0.0

    def __post_init__(self) -> None:
        if self.motion_type not in ("linear", "circular"):
            raise ValueError("motion_type must be 'linear' or 'circular'")
        pos = np.asarray(self.position, dtype=np.float64)
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise ValueError("CartesianWaypoint position must be a finite 3-vector")
        if self.rotation is not None:
            rot = np.asarray(self.rotation, dtype=np.float64)
            if rot.shape != (3, 3) or not np.isfinite(rot).all():
                raise ValueError("CartesianWaypoint rotation must be a finite 3x3 matrix")
        if self.motion_type == "circular":
            if self.via_position is None:
                raise ValueError("circular CartesianWaypoint requires via_position")
            via = np.asarray(self.via_position, dtype=np.float64)
            if via.shape != (3,) or not np.isfinite(via).all():
                raise ValueError("CartesianWaypoint via_position must be a finite 3-vector")
        elif self.via_position is not None:
            raise ValueError("via_position is only valid when motion_type='circular'")
        if not math.isfinite(self.blend_radius) or self.blend_radius < 0.0:
            raise ValueError("blend_radius must be finite and nonnegative")


@dataclass(frozen=True)
class CartesianPlanConfig:
    """Configuration for Cartesian TCP path planning, continuity IK, retiming, and continuous verification."""

    max_step: float = 0.01
    max_rot_step: float = 0.05
    max_pos_error: float = 1e-3
    max_rot_error: float = 1e-2
    jump_threshold_revolute: float = 0.5
    jump_threshold_prismatic: float = 0.2
    max_subdivision_depth: int = 4
    ik_iterations: int = 30
    ik_damping: float = 1e-3
    avoid_collisions: bool = True
    margin: float = 0.002
    world_margin: float | None = None
    strict_ccd: bool = False
    collision_substeps: int = 4
    spline_order: int = 3
    velocity_scale: float = 1.0
    acceleration_scale: float = 1.0
    jerk_scale: float = 1.0
    stop_at_endpoints: bool = True
    hold_orientation: bool = False
    verify_original: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_step) or self.max_step <= 0.0:
            raise ValueError("max_step must be finite and positive")
        if not math.isfinite(self.max_rot_step) or self.max_rot_step <= 0.0:
            raise ValueError("max_rot_step must be finite and positive")
        if not math.isfinite(self.max_pos_error) or self.max_pos_error <= 0.0:
            raise ValueError("max_pos_error must be finite and positive")
        if not math.isfinite(self.max_rot_error) or self.max_rot_error <= 0.0:
            raise ValueError("max_rot_error must be finite and positive")
        if math.isnan(self.jump_threshold_revolute) or self.jump_threshold_revolute <= 0.0:
            raise ValueError("jump_threshold_revolute must be positive (or +inf)")
        if math.isnan(self.jump_threshold_prismatic) or self.jump_threshold_prismatic <= 0.0:
            raise ValueError("jump_threshold_prismatic must be positive (or +inf)")
        if self.max_subdivision_depth < 0:
            raise ValueError("max_subdivision_depth must be >= 0")
        if self.ik_iterations < 1:
            raise ValueError("ik_iterations must be >= 1")
        if not math.isfinite(self.ik_damping) or self.ik_damping < 0.0:
            raise ValueError("ik_damping must be finite and nonnegative")
        if not math.isfinite(self.margin) or self.margin < 0.0:
            raise ValueError("margin must be finite and nonnegative")
        if self.world_margin is not None and (not math.isfinite(self.world_margin) or self.world_margin < 0.0):
            raise ValueError("world_margin must be finite and nonnegative")
        if self.collision_substeps < 1:
            raise ValueError("collision_substeps must be >= 1")
        if self.spline_order not in (3, 5):
            raise ValueError("spline_order must be 3 (cubic Hermite) or 5 (quintic Hermite)")
        if not math.isfinite(self.velocity_scale) or self.velocity_scale <= 0.0:
            raise ValueError("velocity_scale must be finite and positive")
        if not math.isfinite(self.acceleration_scale) or self.acceleration_scale <= 0.0:
            raise ValueError("acceleration_scale must be finite and positive")
        if not math.isfinite(self.jerk_scale) or self.jerk_scale <= 0.0:
            raise ValueError("jerk_scale must be finite and positive")

    @property
    def effective_world_margin(self) -> float:
        return self.margin if self.world_margin is None else self.world_margin


@dataclass(frozen=True)
class CartesianPlanResult:
    """Verified executable Cartesian path result with monotone progress fraction."""

    trajectory: Trajectory | None
    fraction: float = 0.0
    completed_waypoints: int = 0
    max_pos_error: float = 0.0
    max_rot_error: float = 0.0
    reason: str | None = None
    diagnostics: PlanDiagnostics = field(default_factory=PlanDiagnostics)
    world_version: int = 0

    @property
    def success(self) -> bool:
        return self.trajectory is not None and self.fraction >= 1.0 - 1e-9 and self.reason is None

    @property
    def positions(self) -> np.ndarray | None:
        return None if self.trajectory is None else self.trajectory.positions


@dataclass(frozen=True)
class LinkSphere:
    """Link-local collision sphere used for fast O(1) ESDF obstacle avoidance."""

    link: Link | str
    shape_name: str
    center: np.ndarray
    radius: float
    collision_layer: int = 1
    collision_mask: int = 1
