"""Shared C++-backed trajectory retiming and continuous bounds verification (cubic & C^2 quintic Hermite)."""

from __future__ import annotations

import numpy as np

from .types import Trajectory

try:
    from gobot import _coal_native as _native
except ImportError as error:
    raise ImportError("gobot.planning requires a Gobot build with GOB_BUILD_COAL=ON") from error


def check_trajectory_bounds(
    traj: Trajectory,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    *,
    check_velocity: bool = True,
    check_acceleration: bool = True,
    tol: float = 1e-5,
    lower_limits: np.ndarray | None = None,
    upper_limits: np.ndarray | None = None,
    jerk_limits: np.ndarray | None = None,
    check_jerk: bool = True,
) -> tuple[bool, str | None, list[int]]:
    """Verify continuous joint position, velocity, acceleration, and (for order 5) jerk bounds.

    Delegates to the shared C++ spline verifier using exact closed-form cubic/acceleration/jerk
    extrema and certified Bernstein branch-and-bound envelopes for quintic position and velocity.
    """
    ok, reason, bad_segs = _native.check_trajectory_bounds(
        np.ascontiguousarray(traj.times, dtype=np.float64),
        np.ascontiguousarray(traj.positions, dtype=np.float64),
        np.ascontiguousarray(traj.velocities, dtype=np.float64),
        np.ascontiguousarray(traj.accelerations, dtype=np.float64),
        np.ascontiguousarray(velocity_limits, dtype=np.float64),
        np.ascontiguousarray(acceleration_limits, dtype=np.float64),
        bool(check_velocity),
        bool(check_acceleration),
        float(tol),
        None if lower_limits is None else np.ascontiguousarray(lower_limits, dtype=np.float64),
        None if upper_limits is None else np.ascontiguousarray(upper_limits, dtype=np.float64),
        None if jerk_limits is None else np.ascontiguousarray(jerk_limits, dtype=np.float64),
        bool(check_jerk),
        int(traj.spline_order),
    )
    return bool(ok), reason, list(bad_segs)


def retime_trajectory(
    positions: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    *,
    stop_at_endpoints: bool = True,
    dt: float | None = None,
    knot_velocity_scales: np.ndarray | None = None,
    jerk_limits: np.ndarray | None = None,
    spline_order: int = 3,
    start_velocity: np.ndarray | None = None,
    start_acceleration: np.ndarray | None = None,
    lower_limits: np.ndarray | None = None,
    upper_limits: np.ndarray | None = None,
) -> Trajectory:
    """Assign timestamps, velocities, and accelerations respecting cubic (order 3) or C^2 quintic (order 5) bounds.

    Delegates to the shared C++ retiming engine:
    - Order 3 (default): exact cubic Hermite TOPP with closed-form acceleration interval jumping and
      optional fixed ``start_velocity`` at ``t = 0``.
    - Order 5 (C^2 quintic): respects joint velocity, acceleration, and jerk limits with optional
      fixed ``start_velocity`` and ``start_acceleration`` at ``t = 0``.
    """
    q = np.ascontiguousarray(positions, dtype=np.float64)
    if q.ndim != 2 or len(q) < 2:
        raise ValueError("positions must have shape [T>=2, J]")
    if start_acceleration is not None and int(spline_order) != 5:
        raise ValueError("start_acceleration is only supported for spline_order == 5")
    d = _native.retime_trajectory(
        q,
        np.ascontiguousarray(velocity_limits, dtype=np.float64),
        np.ascontiguousarray(acceleration_limits, dtype=np.float64),
        bool(stop_at_endpoints),
        None if dt is None else float(dt),
        None if knot_velocity_scales is None else np.ascontiguousarray(knot_velocity_scales, dtype=np.float64),
        None if jerk_limits is None else np.ascontiguousarray(jerk_limits, dtype=np.float64),
        int(spline_order),
        None if start_velocity is None else np.ascontiguousarray(start_velocity, dtype=np.float64),
        None if start_acceleration is None else np.ascontiguousarray(start_acceleration, dtype=np.float64),
        None if lower_limits is None else np.ascontiguousarray(lower_limits, dtype=np.float64),
        None if upper_limits is None else np.ascontiguousarray(upper_limits, dtype=np.float64),
    )
    return Trajectory(
        times=d["times"],
        positions=d["positions"],
        velocities=d["velocities"],
        accelerations=d["accelerations"],
        dt=d["dt"],
        spline_order=int(d["spline_order"]),
    )


def slice_trajectory_suffix(traj: Trajectory, t_anchor: float) -> Trajectory:
    """Slice an exact Hermite suffix of ``traj`` starting at ``t_anchor`` (rebased to ``times[0] == 0.0``)."""
    d = _native.slice_trajectory_suffix(
        np.ascontiguousarray(traj.times, dtype=np.float64),
        np.ascontiguousarray(traj.positions, dtype=np.float64),
        np.ascontiguousarray(traj.velocities, dtype=np.float64),
        np.ascontiguousarray(traj.accelerations, dtype=np.float64),
        float(t_anchor),
        int(traj.spline_order),
        None if traj.dt is None else float(traj.dt),
    )
    return Trajectory(
        times=d["times"],
        positions=d["positions"],
        velocities=d["velocities"],
        accelerations=d["accelerations"],
        dt=d["dt"],
        spline_order=int(d["spline_order"]),
    )
