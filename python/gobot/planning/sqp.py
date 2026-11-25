"""Pure NumPy/SciPy/OSQP trust-region SQP with parallel batch execution and seed racing."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import math
import os
from time import perf_counter

import numpy as np
from scipy import sparse

from .backends import CollisionBackend
from .world import PlanningWorld

try:
    import osqp
except ImportError as error:
    raise ImportError("gobot.planning requires the gobot[planning-sqp] extra (osqp)") from error


_COLLISION_LINEARIZATION_BUFFER_M = 0.05


@dataclass
class SQPJobSpec:
    initial: np.ndarray                         # (T, J)
    fixed: int                                  # 1 or 2
    dt_vel: float | None                        # None -> omit hard velocity constraint (use retiming)
    dt_smooth: float                            # dt used for smoothness Hessian scaling
    active_mask: np.ndarray                     # (J,) bool
    pose_constraints: dict[int, list]           # waypoint index -> list[PoseConstraintSpec]
    joint_targets: dict[int, np.ndarray]        # waypoint index -> (J,)
    margin: float
    world_margin: float
    collision_substeps: int
    max_segment_length: float | None
    iterations: int
    first_feasible: bool
    velocity_scale: float = 1.0
    seed_strategy: str = "straight"
    goal_index: int = 0


@dataclass
class SQPJobResult:
    positions: np.ndarray | None
    initial_positions: np.ndarray
    reason: str | None = None
    qp_calls: int = 0
    sqp_iterations: int = 0
    sqp_ms: float = 0.0
    validate_ms: float = 0.0
    witnesses: np.ndarray | None = None
    cost: float = math.inf
    seed_strategy: str = "straight"
    goal_index: int = 0


@dataclass
class _SegmentEval:
    segment: np.ndarray              # (R,)
    value: np.ndarray                # (R,)
    jacobian: np.ndarray | None      # (R, 2, J)
    frames: np.ndarray | None        # (S, N_shapes, 3, 4)
    substeps: int
    convex_ok: bool
    witnesses: np.ndarray | None = None


def _grid_substeps(q: np.ndarray, minimum: int, max_length: float | None) -> int:
    if max_length is None:
        return minimum
    longest = float(np.linalg.norm(np.diff(q, axis=0), axis=-1).max())
    return max(minimum, math.ceil(longest / max_length))


def _qp_step(hessian: sparse.csc_matrix, gradient: np.ndarray, values: np.ndarray,
             jacobian: sparse.csr_matrix, lower: np.ndarray, upper: np.ndarray,
             penalty: float, cache: dict | None = None) -> np.ndarray | None:
    size = gradient.size
    if cache is not None and "qp" in cache:
        qp, l_vec, u_vec = cache["qp"], cache["l"], cache["u"]
        l_vec[:size] = lower
        u_vec[:size] = upper
        qp.update(l=l_vec, u=u_vec)
        result = qp.solve(raise_error=False)
        if result.info.status in ("solved", "solved inaccurate") and result.x is not None:
            step = result.x[:size]
            if np.all(np.isfinite(step)):
                return np.clip(step, lower, upper)
        return None

    if len(values):
        worst = values + jacobian.maximum(0) @ upper + jacobian.minimum(0) @ lower
        active = worst > 0.0
        values, jacobian = values[active], jacobian[active]
    count = len(values)
    scale = max(1.0, penalty / 10.0)
    qp = osqp.OSQP()
    if count == 0:
        P = sparse.triu(hessian, format="csc") / scale
        q_vec = gradient / scale
        matrix = sparse.eye(size, format="csc")
        l_vec = lower.copy()
        u_vec = upper.copy()
        polish = True
    else:
        zero = sparse.csc_matrix((count, count))
        identity = sparse.eye(count, format="csc")
        P = sparse.block_diag([sparse.triu(hessian), zero], format="csc") / scale
        q_vec = np.r_[gradient / scale, np.full(count, penalty / scale)]
        matrix = sparse.vstack([
            sparse.eye(size + count, format="csc"),
            sparse.hstack([jacobian, -identity]),
        ], format="csc")
        l_vec = np.r_[lower, np.zeros(count), np.full(count, -np.inf)]
        u_vec = np.r_[upper, np.full(count, np.inf), -values]
        polish = count <= 64
    qp.setup(P=P, q=q_vec, A=matrix, l=l_vec, u=u_vec,
             verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=4000,
             polishing=polish, warm_starting=True)
    if cache is not None:
        cache["qp"] = qp
        cache["l"] = l_vec
        cache["u"] = u_vec
    result = qp.solve(raise_error=False)
    if result.info.status in ("solved", "solved inaccurate") and result.x is not None:
        step = result.x[:size]
        if np.all(np.isfinite(step)):
            return np.clip(step, lower, upper)
    return None


def _waypoint_rows(derivatives: np.ndarray, starts: np.ndarray, fixed: int, waypoints: int) -> sparse.csr_matrix:
    rows, width, joints = derivatives.shape
    size = (waypoints - 2 * fixed) * joints
    if rows == 0:
        return sparse.csr_matrix((0, size), dtype=np.float64)
    columns = (starts[:, None] + np.arange(width) - fixed) * joints
    columns = columns[:, :, None] + np.arange(joints)
    valid = (columns >= 0) & (columns < size) & (derivatives != 0.0)
    row_ids = np.broadcast_to(np.arange(rows)[:, None, None], derivatives.shape)
    return sparse.csr_matrix((derivatives[valid], (row_ids[valid], columns[valid])), shape=(rows, size))


def _assemble_segment_eval(
    world: PlanningWorld,
    pairs: np.ndarray,
    substeps: int,
    dt_vel: float | None,
    vel_scale: float,
    linearize: bool,
    c_rows: np.ndarray,
    c_res: np.ndarray,
    c_grad: np.ndarray | None,
    c_seg_ok: np.ndarray,
    frames: np.ndarray,
    witnesses: np.ndarray | None,
) -> _SegmentEval:
    n_segs, _, joints = pairs.shape
    seg_ok = c_seg_ok.copy()
    segments, values, derivatives = [], [], []

    if dt_vel is not None:
        vmax = world.velocity_limits * vel_scale
        speed = (pairs[:, 1] - pairs[:, 0]) / dt_vel
        seg_ok &= (np.abs(speed) <= vmax * (1.0 + 1e-6)).all(axis=-1)
        speed_vals = np.concatenate((speed - vmax, -speed - vmax), axis=-1)
        s_rows, s_cols = np.nonzero(np.isfinite(speed_vals))
        if len(s_rows):
            segments.append(s_rows.astype(np.int64))
            values.append(speed_vals[s_rows, s_cols])
            if linearize:
                eye = np.eye(joints, dtype=np.float64) / dt_vel
                sj = np.stack((-eye, eye), axis=1)
                derivatives.append(np.concatenate((sj, -sj), axis=0)[s_cols])

    if len(c_rows) or not segments:
        segments.append(c_rows // substeps)
        values.append(c_res)
        if linearize:
            if len(c_rows) and c_grad is not None:
                t = ((c_rows % substeps).astype(np.float64) / substeps)[:, None]
                derivatives.append(np.stack(((1.0 - t) * c_grad, t * c_grad), axis=1))
            else:
                derivatives.append(np.empty((0, 2, joints), dtype=np.float64))

    if len(segments) == 1:
        seg, val = segments[0], values[0]
        jac = derivatives[0] if linearize else None
    else:
        merged_seg = np.concatenate(segments)
        order = np.argsort(merged_seg, kind="stable")
        seg = merged_seg[order]
        val = np.concatenate(values)[order]
        jac = np.concatenate(derivatives, axis=0)[order] if linearize else None
    return _SegmentEval(seg, val, jac, frames, substeps, bool(seg_ok.all()), witnesses)


def _sqp_job(world: PlanningWorld, backend: CollisionBackend, spec: SQPJobSpec):
    """Generator driving one trajectory job, yielding collision batch requests."""
    t_job_start = perf_counter()
    nat = world._native
    q_init = spec.initial.copy()
    waypoints, joints = q_init.shape
    fixed = spec.fixed
    free = slice(fixed, waypoints - fixed)
    lower, upper = world.lower_limits, world.upper_limits
    margins = world.pair_margins(spec.margin, spec.world_margin)
    vmax = world.velocity_limits * spec.velocity_scale

    # Precompute fixed goal frame once for fast validation reuse.
    (goal_frame,) = nat.compute_frames(np.ascontiguousarray(q_init[-1:]), False)

    # Per-job failure caches for convex and original mesh checks.
    prev_convex: tuple[int, int, int] | None = None
    prev_orig: tuple[int, int, int] | None = None
    failure: str | None = None
    validate_time = 0.0
    qp_calls = 0
    sqp_iters = 0
    last_witnesses: np.ndarray | None = None

    def check_frames(frames: np.ndarray, substeps: int, verify_original: bool) -> tuple[int, int] | None:
        nonlocal prev_convex, prev_orig
        prev = prev_orig if verify_original else prev_convex
        if prev is not None:
            s, p, last_sub = prev
            if last_sub == substeps and s < len(frames):
                hit = backend.first_violation(
                    frames, np.array([s], dtype=np.int64), np.array([p], dtype=np.int64),
                    margins, verify_original,
                )
                if hit is not None:
                    return hit
        rows, pids = nat.screen_aabbs(frames, margins, verify_original, substeps)
        hit = backend.first_violation(frames, rows, pids, margins, verify_original) if len(rows) else None
        if verify_original:
            prev_orig = (int(hit[0]), int(hit[1]), substeps) if hit is not None else None
        else:
            prev_convex = (int(hit[0]), int(hit[1]), substeps) if hit is not None else None
        return hit

    validation_substeps = 2 * spec.collision_substeps

    def check_targets(positions: np.ndarray) -> tuple[bool, str | None]:
        for idx, target in spec.joint_targets.items():
            if np.max(np.abs(positions[idx] - target)) > 1e-4:
                return False, f"joint position tolerance at waypoint {idx}"
        for idx, specs in spec.pose_constraints.items():
            _, _, ok = nat.eval_pose_constraints(positions[idx], specs, False)
            if not ok:
                return False, f"pose tolerance at waypoint {idx}"
        return True, None

    def feasible(positions: np.ndarray, seg_eval: _SegmentEval | None = None) -> bool:
        nonlocal failure, validate_time
        t0 = perf_counter()
        try:
            if not np.isfinite(positions).all() or (positions < lower).any() or (positions > upper).any():
                failure = "trajectory violates joint limits"
                return False
            if spec.dt_vel is not None:
                if (np.abs(np.diff(positions, axis=0)) / spec.dt_vel > vmax * (1.0 + 1e-6)).any():
                    failure = "trajectory violates joint velocity limits"
                    return False
            t_ok, t_reason = check_targets(positions)
            if not t_ok:
                failure = t_reason
                return False
            if world.num_pairs == 0:
                failure = None
                return True
            substeps = _grid_substeps(positions, validation_substeps, spec.max_segment_length)
            skip_convex = False
            if seg_eval is not None and seg_eval.frames is not None and seg_eval.substeps == substeps:
                frames = np.concatenate((seg_eval.frames, goal_frame), axis=0)
                skip_convex = seg_eval.convex_ok
            else:
                u = (np.arange(substeps, dtype=np.float64) / substeps)[:, None]
                between = (1.0 - u)[None, :, :] * positions[:-1, None, :] + u[None, :, :] * positions[1:, None, :]
                samples = np.concatenate((between.reshape(-1, joints), positions[-1:]), axis=0)
                (frames,) = nat.compute_frames(np.ascontiguousarray(samples), False)
            if not skip_convex:
                hit = check_frames(frames, substeps, False)
                if hit is not None:
                    seg = min(int(hit[0]) // substeps, waypoints - 2)
                    failure = f"collision clearance on segment {seg}, pair {world.collision_pairs[int(hit[1])]}"
                    return False
            if world.has_mesh:
                hit = check_frames(frames, substeps, True)
                if hit is not None:
                    seg = min(int(hit[0]) // substeps, waypoints - 2)
                    failure = f"original mesh clearance on segment {seg}, pair {world.collision_pairs[int(hit[1])]}"
                    return False
            failure = None
            return True
        finally:
            validate_time += (perf_counter() - t0) * 1000.0

    # Fast check if zero interior free waypoints or seed is already feasible.
    if waypoints <= 2 * fixed:
        ok = feasible(q_init)
        total_ms = (perf_counter() - t_job_start) * 1000.0
        return SQPJobResult(
            q_init.copy() if ok else None, q_init, None if ok else failure,
            0, 0, total_ms - validate_time, validate_time, None,
            0.0 if ok else math.inf, spec.seed_strategy, spec.goal_index,
        )

    # Smoothness objective: first and second joint differences.
    eye = np.eye(waypoints, dtype=np.float64)
    d1 = np.diff(eye, axis=0)
    d2 = np.diff(eye, n=2, axis=0)
    matrix = np.concatenate((d1, 0.1 * d2 / (spec.dt_smooth ** 2)), axis=0)
    smooth_jac = sparse.kron(sparse.csr_matrix(matrix[:, free]), sparse.eye(joints), format="csr")
    hessian = (smooth_jac.T @ smooth_jac).tocsc()

    # Build pinned mask for inactive planning-group joints and matched interior joint targets.
    n_free_wp = waypoints - 2 * fixed
    pinned_mask = np.zeros((n_free_wp, joints), dtype=np.bool_)
    if not spec.active_mask.all():
        pinned_mask[:, ~spec.active_mask] = True
    unpinned_joint_targets: dict[int, np.ndarray] = {}
    for idx, target in spec.joint_targets.items():
        if fixed <= idx < waypoints - fixed and np.array_equal(q_init[idx], target):
            pinned_mask[idx - fixed, :] = True
        else:
            unpinned_joint_targets[idx] = target
    pinned_flat = pinned_mask.reshape(-1) if pinned_mask.any() else None

    def eval_targets(path: np.ndarray, linearize: bool):
        res_blocks, jac_blocks = [], []
        all_ok = True
        for idx, target in unpinned_joint_targets.items():
            r = path[idx] - target
            if np.max(np.abs(r)) > 1e-4:
                all_ok = False
            res_blocks.append(r)
            if linearize:
                jac_blocks.append((idx, np.eye(joints, dtype=np.float64)))
        for idx, specs in spec.pose_constraints.items():
            r, jc, ok = nat.eval_pose_constraints(path[idx], specs, linearize)
            if not ok:
                all_ok = False
            res_blocks.append(r)
            if linearize:
                jac_blocks.append((idx, jc))
        return res_blocks, jac_blocks, all_ok

    def run_attempt(seed_q: np.ndarray, base_substeps: int, segment_minimum: bool):
        nonlocal qp_calls, sqp_iters, last_witnesses
        q = seed_q.copy()
        initial_feasible = feasible(q)
        if spec.first_feasible and initial_feasible:
            return q, float(0.5 * np.sum((matrix @ q) ** 2)), None
        if initial_feasible and fixed == 1 and not spec.pose_constraints and not spec.joint_targets:
            frac = np.linspace(0.0, 1.0, waypoints, dtype=np.float64)[1:-1, None]
            if np.allclose(q[free], (1.0 - frac) * q[0] + frac * q[-1], atol=1e-12, rtol=0.0):
                return q, float(0.5 * np.sum((matrix @ q) ** 2)), None

        best: np.ndarray | None = q.copy() if initial_feasible else None
        best_cost = float(0.5 * np.sum((matrix @ q) ** 2)) if initial_feasible else math.inf
        init_viol: float | None = None
        min_viol = math.inf
        min_viol_q: np.ndarray | None = None
        radius, penalty = 0.1, 10.0

        for _ in range(spec.iterations):
            sqp_iters += 1
            substeps = _grid_substeps(q, base_substeps, spec.max_segment_length)
            pairs = np.stack((q[:-1], q[1:]), axis=1)
            seg_eval: _SegmentEval = yield (
                substeps, True, segment_minimum, pairs, margins, spec.dt_vel, spec.velocity_scale,
            )
            if seg_eval.witnesses is not None and len(seg_eval.witnesses):
                last_witnesses = seg_eval.witnesses
            cart_res, cart_jacs, cart_ok = eval_targets(q, True)

            blocks = [_waypoint_rows(seg_eval.jacobian, seg_eval.segment, fixed, waypoints)]
            val_list = [seg_eval.value]
            for (idx, jc), r in zip(cart_jacs, cart_res):
                blk = _waypoint_rows(jc[:, None, :], np.full(len(r), idx, dtype=np.int64), fixed, waypoints)
                blocks.extend((blk, -blk))
                val_list.extend((r, -r))
            value = np.concatenate(val_list) if len(val_list) > 1 else val_list[0]
            jac = sparse.vstack(blocks, format="csr") if len(blocks) > 1 else blocks[0]

            residual = matrix @ q
            cost = float(0.5 * np.sum(residual * residual))
            gradient = smooth_jac.T @ residual.reshape(-1)
            cur_viol = float(np.maximum(seg_eval.value, 0.0).sum() + sum(np.abs(r).sum() for r in cart_res))
            if init_viol is None:
                init_viol = cur_viol
            if cur_viol < min_viol:
                min_viol = cur_viol
                min_viol_q = q.copy()

            qp_cache: dict = {}
            predicted = 0.0
            while radius >= 1e-4:
                lo = np.maximum(lower - q[free], -radius).reshape(-1)
                hi = np.minimum(upper - q[free], radius).reshape(-1)
                if pinned_flat is not None:
                    lo[pinned_flat] = 0.0
                    hi[pinned_flat] = 0.0
                qp_calls += 1
                step = _qp_step(hessian, gradient, value, jac, lo, hi, penalty, qp_cache)
                if step is None:
                    radius *= 0.25
                    continue
                lin_viol = float(np.maximum(value + jac @ step, 0.0).sum()) if len(value) else 0.0
                predicted = penalty * (cur_viol - lin_viol) - float(step @ gradient) - 0.5 * float(step @ (hessian @ step))
                if predicted <= 1e-9:
                    break
                trial = q.copy()
                trial[free] = np.clip(trial[free] + step.reshape(n_free_wp, joints), lower, upper)
                trial_pairs = np.stack((trial[:-1], trial[1:]), axis=1)
                trial_seg: _SegmentEval = yield (
                    substeps, False, segment_minimum, trial_pairs, margins, spec.dt_vel, spec.velocity_scale,
                )
                trial_res, _, trial_cart_ok = eval_targets(trial, False)
                trial_cost = float(0.5 * np.sum((matrix @ trial) ** 2))
                trial_viol = float(np.maximum(trial_seg.value, 0.0).sum() + sum(np.abs(r).sum() for r in trial_res))
                actual = cost - trial_cost + penalty * (cur_viol - trial_viol)
                if actual >= 0.25 * predicted:
                    q = trial
                    radius *= 1.5
                    if trial_viol < min_viol:
                        min_viol = trial_viol
                        min_viol_q = q.copy()
                    if trial_cost < best_cost and trial_seg.convex_ok and trial_cart_ok:
                        if feasible(q, trial_seg):
                            if spec.first_feasible:
                                return q, trial_cost, None
                            best, best_cost = q.copy(), trial_cost
                    break
                radius *= 0.25
            else:
                predicted = 0.0

            if predicted <= 1e-9:
                if best is not None and np.array_equal(q, best):
                    break
                if seg_eval.convex_ok and cart_ok and feasible(q, seg_eval):
                    if cost < best_cost:
                        best, best_cost = q.copy(), cost
                    if spec.first_feasible:
                        return q, cost, None
                    break
                if penalty >= 1e5:
                    break
                penalty *= 10.0
                radius = 0.1

        warm = min_viol_q if (best is None and min_viol_q is not None and init_viol is not None and min_viol < init_viol) else None
        return best, best_cost, warm

    attempts = [(spec.collision_substeps, False), (validation_substeps, False)]
    if spec.first_feasible and world.num_pairs > 0:
        attempts.insert(0, (validation_substeps, True))

    seed_cur = q_init
    sol: np.ndarray | None = None
    sol_cost = math.inf
    for substeps, seg_min in attempts:
        sol, sol_cost, warm = yield from run_attempt(seed_cur, substeps, seg_min)
        if sol is not None or world.num_pairs == 0:
            break
        if warm is not None:
            seed_cur = warm

    total_ms = (perf_counter() - t_job_start) * 1000.0
    return SQPJobResult(
        sol, q_init, None if sol is not None else (failure or "trajectory optimization found no feasible path"),
        qp_calls, sqp_iters, max(0.0, total_ms - validate_time), validate_time,
        last_witnesses, sol_cost, spec.seed_strategy, spec.goal_index,
    )


def _eval_merged_requests(world: PlanningWorld, backend: CollisionBackend, requests: list) -> dict[int, _SegmentEval]:
    """Merge heterogeneous SQP requests into ONE FK + ONE narrow-phase collision call."""
    if len(requests) == 1:
        row, substeps, linearize, seg_min, pairs, margins, dt_vel, vel_scale = requests[0]
        buffer = _COLLISION_LINEARIZATION_BUFFER_M if linearize else 0.0
        u = (np.arange(substeps, dtype=np.float64) / substeps)[:, None]
        samples = ((1.0 - u)[None, :, :] * pairs[:, :1, :] + u[None, :, :] * pairs[:, 1:, :]).reshape(-1, pairs.shape[-1])
        opt_m = margins + 1e-4 if seg_min else margins
        ev = backend.eval_rows(samples, opt_m, margins, buffer, linearize, substeps, seg_min)
        return {
            row: _assemble_segment_eval(
                world, pairs, substeps, dt_vel, vel_scale, linearize,
                ev.rows, ev.residuals, ev.gradients, ev.seg_ok, ev.frames, ev.witnesses,
            )
        }

    nat = world._native
    joints = world.num_joints
    need_joints = any(req[2] for req in requests)
    u_cache: dict[int, np.ndarray] = {}
    sample_blocks = []
    meta = []
    s_off = 0
    for row, substeps, linearize, seg_min, pairs, margins, dt_vel, vel_scale in requests:
        u = u_cache.get(substeps)
        if u is None:
            u = (np.arange(substeps, dtype=np.float64) / substeps)[:, None]
            u_cache[substeps] = u
        samples = ((1.0 - u)[None, :, :] * pairs[:, :1, :] + u[None, :, :] * pairs[:, 1:, :]).reshape(-1, joints)
        n_s = len(samples)
        sample_blocks.append(samples)
        meta.append((row, substeps, linearize, seg_min, pairs, margins, dt_vel, vel_scale, s_off, n_s))
        s_off += n_s

    all_samples = np.ascontiguousarray(np.concatenate(sample_blocks, axis=0), dtype=np.float64)
    fk_out = nat.compute_frames(all_samples, need_joints)
    if need_joints:
        all_frames, all_jorig, all_jaxis = fk_out
    else:
        (all_frames,) = fk_out
        all_jorig = all_jaxis = None

    # Screen AABBs per request slice, merge all candidates into ONE narrow-phase call.
    cand_rows, cand_pairs, spans = [], [], []
    c_off = 0
    for row, substeps, linearize, seg_min, pairs, margins, dt_vel, vel_scale, off, cnt in meta:
        buffer = _COLLISION_LINEARIZATION_BUFFER_M if linearize else 0.0
        opt_m = margins + 1e-4 if seg_min else margins
        if world.num_pairs > 0:
            r_loc, p_loc = nat.screen_aabbs(all_frames[off:off + cnt], opt_m + buffer, False, substeps)
        else:
            r_loc = p_loc = np.empty(0, dtype=np.int64)
        n_c = len(r_loc)
        spans.append((c_off, n_c))
        if n_c:
            cand_rows.append(r_loc + off)
            cand_pairs.append(p_loc)
            c_off += n_c

    if c_off:
        m_rows = np.concatenate(cand_rows) if len(cand_rows) > 1 else cand_rows[0]
        m_pairs = np.concatenate(cand_pairs) if len(cand_pairs) > 1 else cand_pairs[0]
        m_data = backend.query_indexed_convex(all_frames, m_rows, m_pairs)
    else:
        m_rows = m_pairs = np.empty(0, dtype=np.int64)
        m_data = np.empty((0, 10), dtype=np.float64)

    replies: dict[int, _SegmentEval] = {}
    for (row, substeps, linearize, seg_min, pairs, margins, dt_vel, vel_scale, off, cnt), (co, cc) in zip(meta, spans):
        buffer = _COLLISION_LINEARIZATION_BUFFER_M if linearize else 0.0
        opt_m = margins + 1e-4 if seg_min else margins
        n_segs = len(pairs)
        seg_ok = np.ones(n_segs, dtype=np.bool_)
        c_rows_n = np.empty(0, dtype=np.int64)
        c_res_n = np.empty(0, dtype=np.float64)
        c_grad: np.ndarray | None = np.empty((0, joints), dtype=np.float64) if linearize else None
        c_wit: np.ndarray | None = np.empty((0, 6), dtype=np.float64) if linearize else None

        if cc:
            r_glob = m_rows[co:co + cc]
            r_loc = r_glob - off
            p_loc = m_pairs[co:co + cc]
            d_loc = m_data[co:co + cc]
            req_min = np.maximum(0.0, margins[p_loc] - 1e-6)
            failed = ~(np.isfinite(d_loc[:, 0]) & (d_loc[:, 0] >= req_min))
            if failed.any():
                seg_ok[r_loc[failed] // substeps] = False
            res = opt_m[p_loc] - d_loc[:, 0]
            near = np.flatnonzero(res > -buffer)
            if seg_min and len(near):
                group = (r_loc[near] // substeps) * len(opt_m) + p_loc[near]
                order = np.lexsort((-res[near], group))
                first = np.r_[True, np.diff(group[order]) != 0]
                near = near[np.sort(order[first])]
            c_rows_n = r_loc[near]
            c_res_n = res[near]
            if linearize and len(near):
                c_grad, c_wit = nat.compute_analytic_gradients(
                    all_frames, all_jorig, all_jaxis, r_glob[near], p_loc[near], d_loc[near],
                )

        replies[row] = _assemble_segment_eval(
            world, pairs, substeps, dt_vel, vel_scale, linearize,
            c_rows_n, c_res_n, c_grad, seg_ok, all_frames[off:off + cnt], c_wit,
        )
    return replies


def run_sqp_batch(
    world: PlanningWorld,
    backend: CollisionBackend,
    specs: list[SQPJobSpec],
    *,
    race: bool = False,
) -> list[SQPJobResult | None]:
    """Execute SQP jobs with multi-core OSQP/validation and merged collision batches.

    When race=True, returns immediately as soon as any job finds a feasible trajectory.
    """
    n = len(specs)
    jobs = [_sqp_job(world, backend, spec) for spec in specs]
    results: list[SQPJobResult | None] = [None] * n
    replies: dict[int, object] = dict.fromkeys(range(n))
    workers = min(n, max(1, os.cpu_count() or 1))
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

    def advance(row: int, reply):
        try:
            req = jobs[row].send(reply)
        except StopIteration as done:
            return row, True, done.value
        return row, False, req

    try:
        while replies:
            requests = []
            if pool is not None and len(replies) > 1:
                futures = [pool.submit(advance, row, reply) for row, reply in replies.items()]
                outcomes = [f.result() for f in futures]
            else:
                outcomes = [advance(row, reply) for row, reply in replies.items()]
            won = False
            for row, finished, payload in outcomes:
                if finished:
                    results[row] = payload
                    if race and payload is not None and payload.positions is not None:
                        won = True
                else:
                    requests.append((row, *payload))
            if won:
                for j in jobs:
                    j.close()
                break
            if not requests:
                break
            replies = _eval_merged_requests(world, backend, requests)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    return results
