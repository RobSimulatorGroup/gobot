"""Time-optimal velocity/acceleration parameterization (TOPP) for cubic Hermite trajectories."""

from __future__ import annotations

import numpy as np

from .types import Trajectory


def _hermite_segment_peaks(
    d: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact per-joint max |v(t)| and max |a(t)| on a cubic Hermite segment over t in [0, h]."""
    inv_h = 1.0 / h
    six_d_h = 6.0 * d * inv_h
    c1 = six_d_h - 4.0 * v0 - 2.0 * v1
    c2 = -six_d_h + 3.0 * (v0 + v1)

    # Acceleration a(u) = (c1 + 2*c2*u) / h is linear in u in [0, 1]; extrema lie at u=0 and u=1.
    a0 = c1 * inv_h
    a1 = (c1 + 2.0 * c2) * inv_h
    max_a = np.maximum(np.abs(a0), np.abs(a1))

    # Velocity v(u) = v0 + c1*u + c2*u^2 is quadratic; check endpoints and interior stationary point.
    max_v = np.maximum(np.abs(v0), np.abs(v1))
    nz = np.abs(c2) > 1e-12
    if nz.any():
        u_star = np.zeros_like(c1)
        u_star[nz] = -c1[nz] / (2.0 * c2[nz])
        interior = nz & (u_star > 0.0) & (u_star < 1.0)
        if interior.any():
            v_star = v0[interior] - (c1[interior] * c1[interior]) / (4.0 * c2[interior])
            max_v[interior] = np.maximum(max_v[interior], np.abs(v_star))

    return max_v, max_a


def _hermite_segment_feasible(
    d: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    h: float,
    vmax: np.ndarray,
    amax: np.ndarray,
    tol: float = 1e-9,
) -> bool:
    max_v, max_a = _hermite_segment_peaks(d, v0, v1, h)
    return bool((max_v <= vmax * (1.0 + tol)).all() and (max_a <= amax * (1.0 + tol)).all())


def check_trajectory_bounds(
    traj: Trajectory,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    *,
    check_velocity: bool = True,
    check_acceleration: bool = True,
    tol: float = 1e-5,
) -> tuple[bool, str | None, list[int]]:
    """Exactly verify continuous cubic Hermite |v(t)| <= vmax and |a(t)| <= amax across all segments."""
    q = np.asarray(traj.positions, dtype=np.float64)
    v = np.asarray(traj.velocities, dtype=np.float64)
    dt_arr = np.diff(traj.times)
    dq = np.diff(q, axis=0)
    vmax = np.asarray(velocity_limits, dtype=np.float64)
    amax = np.asarray(acceleration_limits, dtype=np.float64)
    safe_vmax = np.where(np.isfinite(vmax) & (vmax > 0), vmax, np.inf)
    safe_amax = np.where(np.isfinite(amax) & (amax > 0), amax, np.inf)

    bad_segs: list[int] = []
    first_reason: str | None = None
    for k in range(len(dq)):
        h = float(dt_arr[k])
        if h <= 1e-12:
            continue
        max_v, max_a = _hermite_segment_peaks(dq[k], v[k], v[k + 1], h)
        if check_velocity and (max_v > safe_vmax * (1.0 + tol)).any():
            bad_segs.append(k)
            if first_reason is None:
                ratio = float(np.max(max_v / safe_vmax))
                first_reason = f"Hermite curve violates joint velocity limits on segment {k} ({ratio:.2f}x limit)"
        elif check_acceleration and (max_a > safe_amax * (1.0 + tol)).any():
            bad_segs.append(k)
            if first_reason is None:
                ratio = float(np.max(max_a / safe_amax))
                first_reason = f"Hermite curve violates joint acceleration limits on segment {k} ({ratio:.2f}x limit)"
    return len(bad_segs) == 0, first_reason, bad_segs


def _zero_vel_hermite_dt(d: np.ndarray, vmax: np.ndarray, amax: np.ndarray) -> float:
    """Exact minimum duration for a cubic Hermite segment with v0 = v1 = 0."""
    abs_d = np.abs(d)
    if float(abs_d.max()) <= 1e-12:
        return 1e-4
    h_v = float(np.max(1.5 * abs_d / vmax))
    h_a = float(np.max(np.sqrt(6.0 * abs_d / amax)))
    return max(h_v, h_a, 1e-4)


def _min_feasible_hermite_dt(
    d: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    h_init: float,
    vmax: np.ndarray,
    amax: np.ndarray,
) -> float:
    """Find a tight feasible duration h where the cubic Hermite segment satisfies vmax and amax."""
    h_zero = _zero_vel_hermite_dt(d, vmax, amax)
    min_dt_vel = float(np.max(np.abs(d) / vmax))
    h_start = max(float(h_init), min_dt_vel, 1e-4)

    six_d = 6.0 * d
    v42 = 4.0 * v0 + 2.0 * v1
    v24 = 2.0 * v0 + 4.0 * v1
    v3 = 3.0 * (v0 + v1)
    v_end = np.maximum(np.abs(v0), np.abs(v1))
    vmax_tol = vmax * (1.0 + 1e-9)
    amax_tol = amax * (1.0 + 1e-9)

    def feasible(h: float) -> bool:
        inv_h = 1.0 / h
        sdh = six_d * inv_h
        ah = amax_tol * h
        c1 = sdh - v42
        if (np.abs(c1) > ah).any() or (np.abs(v24 - sdh) > ah).any():
            return False
        if (v_end > vmax_tol).any():
            return False
        c2 = v3 - sdh
        nz = np.abs(c2) > 1e-12
        if nz.any():
            u_star = -c1[nz] / (2.0 * c2[nz])
            inside = (u_star > 0.0) & (u_star < 1.0)
            if inside.any():
                c1_i = c1[nz][inside]
                c2_i = c2[nz][inside]
                v_star = v0[nz][inside] - (c1_i * c1_i) / (4.0 * c2_i)
                if (np.abs(v_star) > vmax_tol[nz][inside]).any():
                    return False
        return True

    # Fast check around h_start (including slightly shorter durations near cruise speed).
    for scale in (1.0, 0.80, 0.90, 1.15, 0.65, 1.30):
        cand = max(h_start * scale, min_dt_vel, 1e-4)
        if feasible(cand):
            if scale == 1.0 and cand <= min_dt_vel * 1.02:
                return cand
            lo = max(min_dt_vel, 1e-4)
            hi = cand
            if feasible(lo):
                return lo
            for _ in range(8):
                mid = 0.5 * (lo + hi)
                if feasible(mid):
                    hi = mid
                else:
                    lo = mid
            return hi

    h_lo = h_start
    h_hi = max(h_lo * 1.5, h_zero)
    for _ in range(20):
        if feasible(h_hi):
            break
        h_lo = h_hi
        h_hi *= 2.0

    for _ in range(10):
        mid = 0.5 * (h_lo + h_hi)
        if feasible(mid):
            h_hi = mid
        else:
            h_lo = mid
    return h_hi


def retime_trajectory(
    positions: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    *,
    stop_at_endpoints: bool = True,
    dt: float | None = None,
    knot_velocity_scales: np.ndarray | None = None,
) -> Trajectory:
    """Assign timestamps, velocities, and accelerations respecting cubic Hermite bounds.

    When ``dt`` is ``None``, computes a time-optimal path parameterization (TOPP) and then
    solves exact cubic Hermite segment durations so that continuous sampling via
    ``Trajectory.sample(t)`` strictly satisfies ``|v(t)| <= velocity_limits`` and
    ``|a(t)| <= acceleration_limits`` for all ``t``.

    When ``dt`` is provided, builds a uniform-step trajectory with endpoint-clamped knot
    velocities scaled by optional ``knot_velocity_scales``.
    """
    q = np.asarray(positions, dtype=np.float64)
    if q.ndim != 2 or len(q) < 2:
        raise ValueError("positions must have shape [T>=2, J]")
    T, J = q.shape

    vmax = np.asarray(velocity_limits, dtype=np.float64)
    amax = np.asarray(acceleration_limits, dtype=np.float64)
    safe_vmax = np.where(np.isfinite(vmax) & (vmax > 0), vmax, 1e6)
    safe_amax = np.where(np.isfinite(amax) & (amax > 0), amax, 1e6)

    if knot_velocity_scales is None:
        k_scales = np.ones(T, dtype=np.float64)
    else:
        k_scales = np.clip(np.asarray(knot_velocity_scales, dtype=np.float64), 0.0, 1.0)
        if k_scales.shape != (T,):
            raise ValueError(f"knot_velocity_scales must have shape ({T},)")

    dq = np.diff(q, axis=0)  # (T-1, J)
    ds = np.linalg.norm(dq, axis=-1)  # (T-1,)
    nonzero = ds > 1e-12

    if dt is not None:
        dt_f = float(dt)
        times = np.arange(T, dtype=np.float64) * dt_f
        vel = np.zeros_like(q)
        if T >= 2:
            if T > 2:
                vel[1:-1] = (q[2:] - q[:-2]) / (2.0 * dt_f)
            if not stop_at_endpoints:
                vel[0] = (q[1] - q[0]) / dt_f
                vel[-1] = (q[-1] - q[-2]) / dt_f
        vel *= k_scales[:, None]
        # Zero velocities adjacent to zero-length segments (e.g., repeated stop_at_endpoints knots).
        for k in range(T - 1):
            if not nonzero[k]:
                vel[k] = 0.0
                vel[k + 1] = 0.0
        # Clamp knot velocities to velocity limits.
        v_ratio = np.max(np.abs(vel) / safe_vmax[None, :], axis=-1)
        over = v_ratio > 1.0
        if over.any():
            vel[over] /= v_ratio[over, None]

        # When no external collision damping is active, project free knot velocities into the
        # exact fixed-dt cubic Hermite acceleration-feasible interval only if any segment violates limits.
        if (k_scales >= 1.0 - 1e-12).all() and T >= 2:
            needs_proj = any(
                nonzero[k] and not _hermite_segment_feasible(dq[k], vel[k], vel[k + 1], dt_f, safe_vmax, safe_amax, tol=1e-5)
                for k in range(T - 1)
            )
            if needs_proj:
                free_lo = 1 if stop_at_endpoints else 0
                free_hi = T - 1 if stop_at_endpoints else T
                ah = safe_amax * dt_f
                for _ in range(8):
                    for i in range(free_lo, free_hi):
                        if (i > 0 and not nonzero[i - 1]) or (i < T - 1 and not nonzero[i]):
                            continue
                        lo = -safe_vmax.copy()
                        hi = safe_vmax.copy()
                        if i > 0:
                            c = 6.0 * dq[i - 1] / dt_f
                            v_prev = vel[i - 1]
                            lo = np.maximum(lo, np.maximum((c - 4.0 * v_prev - ah) * 0.5, (c - 2.0 * v_prev - ah) * 0.25))
                            hi = np.minimum(hi, np.minimum((c - 4.0 * v_prev + ah) * 0.5, (c - 2.0 * v_prev + ah) * 0.25))
                        if i < T - 1:
                            c = 6.0 * dq[i] / dt_f
                            v_next = vel[i + 1]
                            lo = np.maximum(lo, np.maximum((c - 2.0 * v_next - ah) * 0.25, (c - 4.0 * v_next - ah) * 0.5))
                            hi = np.minimum(hi, np.minimum((c - 2.0 * v_next + ah) * 0.25, (c - 4.0 * v_next + ah) * 0.5))
                        feasible = lo <= hi
                        if feasible.any():
                            vel[i, feasible] = np.clip(vel[i, feasible], lo[feasible], hi[feasible])

        acc = np.zeros_like(q)
        for k in range(T - 1):
            a0 = (6.0 * dq[k] / dt_f - 4.0 * vel[k] - 2.0 * vel[k + 1]) / dt_f
            a1 = (-6.0 * dq[k] / dt_f + 2.0 * vel[k] + 4.0 * vel[k + 1]) / dt_f
            acc[k] = a0 if k == 0 else 0.5 * (acc[k] + a0)
            acc[k + 1] = a1
        return Trajectory(times=times, positions=q.copy(), velocities=vel, accelerations=acc, dt=dt_f)

    # Tangent vector q'(s) on each segment.
    qp = np.zeros_like(dq)
    qp[nonzero] = dq[nonzero] / ds[nonzero, None]

    # Per-segment maximum path speed and acceleration along arc length.
    seg_sdot_max = np.full(T - 1, np.inf, dtype=np.float64)
    seg_sddot_max = np.full(T - 1, 1e6, dtype=np.float64)
    for k in range(T - 1):
        if not nonzero[k]:
            seg_sdot_max[k] = 0.0
            continue
        abs_qp = np.abs(qp[k])
        mask = abs_qp > 1e-12
        if mask.any():
            seg_sdot_max[k] = float(np.min(safe_vmax[mask] / abs_qp[mask]))
            seg_sddot_max[k] = float(np.min(safe_amax[mask] / abs_qp[mask]))

    # Knot speed upper bounds from adjacent segments, curvature, and caller velocity scales.
    sdot = np.full(T, np.inf, dtype=np.float64)
    sdot[0] = 0.0 if stop_at_endpoints else seg_sdot_max[0]
    sdot[-1] = 0.0 if stop_at_endpoints else seg_sdot_max[-1]
    for i in range(1, T - 1):
        sdot[i] = min(seg_sdot_max[i - 1], seg_sdot_max[i])
        if nonzero[i - 1] and nonzero[i]:
            ds_avg = 0.5 * (ds[i - 1] + ds[i])
            qpp = (qp[i] - qp[i - 1]) / max(ds_avg, 1e-9)
            abs_qpp = np.abs(qpp)
            mask = abs_qpp > 1e-12
            if mask.any():
                sdot_curv = float(np.sqrt(np.min(safe_amax[mask] / abs_qpp[mask])))
                sdot[i] = min(sdot[i], sdot_curv)
    sdot *= k_scales

    # Forward-backward TOPP pass along arc length.
    for i in range(T - 1):
        if not nonzero[i]:
            sdot[i + 1] = min(sdot[i + 1], sdot[i])
            continue
        v_reach = np.sqrt(sdot[i] * sdot[i] + 2.0 * seg_sddot_max[i] * ds[i])
        sdot[i + 1] = min(sdot[i + 1], v_reach)

    for i in range(T - 2, -1, -1):
        if not nonzero[i]:
            sdot[i] = min(sdot[i], sdot[i + 1])
            continue
        v_reach = np.sqrt(sdot[i + 1] * sdot[i + 1] + 2.0 * seg_sddot_max[i] * ds[i])
        sdot[i] = min(sdot[i], v_reach)

    # Compute knot velocity vectors along blended unit tangents, clamped strictly to vmax.
    vel = np.zeros_like(q)
    for i in range(1, T - 1):
        if not nonzero[i - 1] or not nonzero[i]:
            vel[i] = 0.0
            continue
        tangent = 0.5 * (qp[i - 1] + qp[i])
        nrm = float(np.linalg.norm(tangent))
        if nrm > 1e-12:
            tangent = tangent / nrm
            vel[i] = tangent * sdot[i]
            ratio = float(np.max(np.abs(vel[i]) / safe_vmax))
            if ratio > 1.0:
                vel[i] /= ratio
    if not stop_at_endpoints:
        if nonzero[0]:
            vel[0] = qp[0] * sdot[0]
            ratio = float(np.max(np.abs(vel[0]) / safe_vmax))
            if ratio > 1.0:
                vel[0] /= ratio
        if nonzero[-1]:
            vel[-1] = qp[-1] * sdot[-1]
            ratio = float(np.max(np.abs(vel[-1]) / safe_vmax))
            if ratio > 1.0:
                vel[-1] /= ratio

    # Solve exact cubic Hermite segment durations. If high corner velocities would force a
    # segment to take longer than a rest-to-rest Hermite move, damp those knot velocities.
    seg_dt = np.full(T - 1, 1e-4, dtype=np.float64)
    zero_dt = np.array([_zero_vel_hermite_dt(dq[k], safe_vmax, safe_amax) if nonzero[k] else 1e-4
                        for k in range(T - 1)], dtype=np.float64)

    for attempt in range(6):
        damped_any = False
        for k in range(T - 1):
            if not nonzero[k]:
                seg_dt[k] = 1e-4
                continue
            v_avg = 0.5 * (float(np.linalg.norm(vel[k])) + float(np.linalg.norm(vel[k + 1])))
            h_est = ds[k] / v_avg if v_avg > 1e-9 else zero_dt[k]
            min_dt_vel = float(np.max(np.abs(dq[k]) / safe_vmax))
            h_k = _min_feasible_hermite_dt(dq[k], vel[k], vel[k + 1], max(h_est, min_dt_vel), safe_vmax, safe_amax)
            # If excessive corner velocity forces a large slowdown/detour, damp adjacent knots.
            if h_k > 1.15 * zero_dt[k] and (np.any(vel[k] != 0.0) or np.any(vel[k + 1] != 0.0)):
                factor = 0.0 if attempt == 5 else 0.6
                vel[k] *= factor
                vel[k + 1] *= factor
                damped_any = True
            seg_dt[k] = h_k
        if not damped_any:
            break

    # Final exact pass from fresh duration estimates so damped segments shrink tightly to zero_dt.
    for k in range(T - 1):
        if not nonzero[k]:
            vel[k] = 0.0
            vel[k + 1] = 0.0
            seg_dt[k] = 1e-4
            continue
        v_avg = 0.5 * (float(np.linalg.norm(vel[k])) + float(np.linalg.norm(vel[k + 1])))
        h_est = ds[k] / v_avg if v_avg > 1e-9 else zero_dt[k]
        min_dt_vel = float(np.max(np.abs(dq[k]) / safe_vmax))
        seg_dt[k] = _min_feasible_hermite_dt(dq[k], vel[k], vel[k + 1], max(h_est, min_dt_vel), safe_vmax, safe_amax)

    times = np.zeros(T, dtype=np.float64)
    times[1:] = np.cumsum(seg_dt)

    # Record consistent Hermite endpoint accelerations at knots.
    acc = np.zeros_like(q)
    for k in range(T - 1):
        h = seg_dt[k]
        a0 = (6.0 * dq[k] / h - 4.0 * vel[k] - 2.0 * vel[k + 1]) / h
        a1 = (-6.0 * dq[k] / h + 2.0 * vel[k] + 4.0 * vel[k + 1]) / h
        acc[k] = a0 if k == 0 else 0.5 * (acc[k] + a0)
        acc[k + 1] = a1

    return Trajectory(times=times, positions=q.copy(), velocities=vel, accelerations=acc, dt=None)
