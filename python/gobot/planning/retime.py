"""Time-optimal velocity/acceleration parameterization (TOPP) for geometric joint paths."""

from __future__ import annotations

import numpy as np

from .types import Trajectory


def retime_trajectory(
    positions: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    *,
    stop_at_endpoints: bool = True,
    dt: float | None = None,
) -> Trajectory:
    """Assign timestamps, velocities, and accelerations respecting joint limits.

    If dt is provided, builds a uniform-step Trajectory with central/endpoint
    finite-difference velocities and accelerations. Otherwise runs a forward-backward
    time-optimal path velocity parameterization along arc length s.
    """
    q = np.asarray(positions, dtype=np.float64)
    if q.ndim != 2 or len(q) < 2:
        raise ValueError("positions must have shape [T>=2, J]")
    T, J = q.shape

    if dt is not None:
        times = np.arange(T, dtype=np.float64) * float(dt)
        vel = np.zeros_like(q)
        acc = np.zeros_like(q)
        if T >= 2:
            vel[1:-1] = (q[2:] - q[:-2]) / (2.0 * dt)
            if not stop_at_endpoints:
                vel[0] = (q[1] - q[0]) / dt
                vel[-1] = (q[-1] - q[-2]) / dt
            acc[1:-1] = (q[2:] - 2.0 * q[1:-1] + q[:-2]) / (dt * dt)
            acc[0] = (vel[1] - vel[0]) / dt
            acc[-1] = (vel[-1] - vel[-2]) / dt
        return Trajectory(times=times, positions=q.copy(), velocities=vel, accelerations=acc, dt=float(dt))

    vmax = np.asarray(velocity_limits, dtype=np.float64)
    amax = np.asarray(acceleration_limits, dtype=np.float64)
    safe_vmax = np.where(np.isfinite(vmax) & (vmax > 0), vmax, 1e6)
    safe_amax = np.where(np.isfinite(amax) & (amax > 0), amax, 1e6)

    # Arc-length parameterization along piecewise-linear segments.
    dq = np.diff(q, axis=0)  # (T-1, J)
    ds = np.linalg.norm(dq, axis=-1)  # (T-1,)
    # Tangent vector q'(s) on each segment.
    nonzero = ds > 1e-12
    qp = np.zeros_like(dq)
    qp[nonzero] = dq[nonzero] / ds[nonzero, None]

    # Per-segment maximum path speed s_dot from velocity limits: |q'_j| * s_dot <= vmax_j.
    seg_sdot_max = np.full(T - 1, np.inf, dtype=np.float64)
    for k in range(T - 1):
        if not nonzero[k]:
            seg_sdot_max[k] = 0.0
            continue
        abs_qp = np.abs(qp[k])
        mask = abs_qp > 1e-12
        if mask.any():
            seg_sdot_max[k] = float(np.min(safe_vmax[mask] / abs_qp[mask]))

    # Knot velocity upper bounds from adjacent segments + corner curvature limits.
    sdot = np.full(T, np.inf, dtype=np.float64)
    sdot[0] = 0.0 if stop_at_endpoints else seg_sdot_max[0]
    sdot[-1] = 0.0 if stop_at_endpoints else seg_sdot_max[-1]
    for i in range(1, T - 1):
        sdot[i] = min(seg_sdot_max[i - 1], seg_sdot_max[i])
        if nonzero[i - 1] and nonzero[i]:
            ds_avg = 0.5 * (ds[i - 1] + ds[i])
            # Approximate second derivative q''(s) at knot i.
            qpp = (qp[i] - qp[i - 1]) / max(ds_avg, 1e-9)
            abs_qpp = np.abs(qpp)
            mask = abs_qpp > 1e-12
            if mask.any():
                # Centripetal acceleration bound: |q''_j| * s_dot^2 <= amax_j.
                sdot_curv = float(np.sqrt(np.min(safe_amax[mask] / abs_qpp[mask])))
                sdot[i] = min(sdot[i], sdot_curv)

    # Per-segment maximum path acceleration |s_ddot| from |q'_j| * |s_ddot| <= amax_j.
    seg_sddot_max = np.full(T - 1, 1e6, dtype=np.float64)
    for k in range(T - 1):
        if not nonzero[k]:
            continue
        abs_qp = np.abs(qp[k])
        mask = abs_qp > 1e-12
        if mask.any():
            seg_sddot_max[k] = float(np.min(safe_amax[mask] / abs_qp[mask]))

    # Forward pass: v_{i+1}^2 <= v_i^2 + 2 * a_max * ds_i.
    for i in range(T - 1):
        if not nonzero[i]:
            sdot[i + 1] = min(sdot[i + 1], sdot[i])
            continue
        v_reach = np.sqrt(sdot[i] * sdot[i] + 2.0 * seg_sddot_max[i] * ds[i])
        sdot[i + 1] = min(sdot[i + 1], v_reach)

    # Backward pass: v_i^2 <= v_{i+1}^2 + 2 * a_max * ds_i.
    for i in range(T - 2, -1, -1):
        if not nonzero[i]:
            sdot[i] = min(sdot[i], sdot[i + 1])
            continue
        v_reach = np.sqrt(sdot[i + 1] * sdot[i + 1] + 2.0 * seg_sddot_max[i] * ds[i])
        sdot[i] = min(sdot[i], v_reach)

    # Integrate timestamps with constant acceleration on each segment.
    times = np.zeros(T, dtype=np.float64)
    for i in range(T - 1):
        if not nonzero[i]:
            times[i + 1] = times[i] + 1e-4
            continue
        v_avg = 0.5 * (sdot[i] + sdot[i + 1])
        if v_avg > 1e-9:
            dt_i = ds[i] / v_avg
        else:
            # Both ends stationary on a single-segment move: triangular profile.
            dt_i = 2.0 * np.sqrt(ds[i] / max(seg_sddot_max[i], 1e-6))
        # Ensure per-joint average velocity never exceeds vmax.
        min_dt_vel = float(np.max(np.abs(dq[i]) / safe_vmax))
        times[i + 1] = times[i] + max(dt_i, min_dt_vel, 1e-4)

    # Joint velocities and accelerations at knots.
    vel = np.zeros_like(q)
    for i in range(1, T - 1):
        tangent = 0.5 * (qp[i - 1] + qp[i]) if (nonzero[i - 1] and nonzero[i]) else (qp[i] if nonzero[i] else qp[i - 1])
        nrm = float(np.linalg.norm(tangent))
        if nrm > 1e-12:
            tangent = tangent / nrm
        vel[i] = tangent * sdot[i]
        # Clamp strictly to joint velocity limits.
        ratio = np.max(np.abs(vel[i]) / safe_vmax)
        if ratio > 1.0:
            vel[i] /= ratio
    if not stop_at_endpoints:
        vel[0] = qp[0] * sdot[0]
        vel[-1] = qp[-1] * sdot[-1]

    acc = np.zeros_like(q)
    dt_seg = np.maximum(np.diff(times), 1e-9)[:, None]
    seg_acc = np.diff(vel, axis=0) / dt_seg
    acc[0] = seg_acc[0]
    acc[-1] = seg_acc[-1]
    if T > 2:
        acc[1:-1] = 0.5 * (seg_acc[:-1] + seg_acc[1:])

    return Trajectory(times=times, positions=q.copy(), velocities=vel, accelerations=acc, dt=None)
