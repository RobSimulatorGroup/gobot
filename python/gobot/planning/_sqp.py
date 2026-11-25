"""CPU trust-region SQP for a quadratic joint objective and nonlinear inequalities."""

from functools import partial
import threading

import numpy as np
import torch
from scipy import sparse
from torch.func import jacfwd

try:
    import osqp
except ImportError as error:
    raise ImportError("CPU SQP requires the gobot[planning-sqp] extra") from error


_COLLISION_LINEARIZATION_BUFFER_M = 0.05
_AD_LOCK = threading.Lock()


def _qp_step(hessian, gradient, values, jacobian, lower, upper, penalty, cache=None):
    """Minimize the quadratic model plus an L1 cost on linearized violations."""
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

    # Eliminate inequalities guaranteed to hold anywhere in this trust box.
    worst = values + jacobian.maximum(0) @ upper + jacobian.minimum(0) @ lower
    active = worst > 0
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
        # A nonnegative slack per inequality keeps the QP feasible at a colliding seed.
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


def _waypoint_rows(derivatives, starts, fixed, waypoints):
    """Place contiguous local joint derivatives into the free waypoint columns."""
    data = derivatives.double().numpy()
    rows, width, joints = data.shape
    columns = (starts.numpy()[:, None] + np.arange(width) - fixed) * joints
    columns = columns[:, :, None] + np.arange(joints)
    size = (waypoints - 2 * fixed) * joints
    valid = (columns >= 0) & (columns < size) & (data != 0)
    row_ids = np.broadcast_to(np.arange(rows)[:, None, None], data.shape)
    return sparse.csr_matrix((data[valid], (row_ids[valid], columns[valid])), shape=(rows, size))


def _could_be_feasible(segment, cart):
    if getattr(segment, "convex_ok", None) is False:
        return False
    for residual in cart:
        if residual.numel() and not bool((residual.abs() <= 1e-4).all()):
            return False
    return True


def _check_feasible(feasible, q, segment=None):
    if segment is not None and getattr(feasible, "accepts_segment", False):
        return feasible(q, segment)
    return feasible(q)


def optimize_sqp(initial, fixed, lower, upper, dt, substeps_for, targets,
                 feasible, iterations, *, first_feasible=False, segment_minimum=False):
    """Convexify segment and Cartesian terms, shrink or expand a box on joint increments."""
    free = slice(fixed, -fixed)
    q = initial.clone()
    waypoints, joints = q.shape
    initial_feasible = feasible(q)
    if first_feasible and initial_feasible:
        return q
    if initial_feasible and fixed == 1:
        # With fixed endpoints, equal increments minimize sum(|dq|²) and give zero acceleration.
        fractions = torch.linspace(0, 1, waypoints, dtype=q.dtype, device=q.device)[1:-1, None]
        if torch.equal(q[free], torch.lerp(q[0], q[-1], fractions)):
            return q

    # Interior joint targets that already match the seed are enforced exactly via
    # zero box bounds instead of linearized equality rows and slack variables.
    active_targets = []
    pinned_mask = np.zeros((waypoints - 2 * fixed, joints), dtype=bool)
    has_pinned = False
    for index, error in targets:
        if (isinstance(error, partial) and error.func is torch.sub and "other" in error.keywords
                and fixed <= index < waypoints - fixed and torch.equal(q[index], error.keywords["other"])):
            pinned_mask[index - fixed, :] = True
            has_pinned = True
        else:
            active_targets.append((index, error))
    pinned_flat = pinned_mask.reshape(-1) if has_pinned else None

    eye = torch.eye(waypoints, dtype=q.dtype, device=q.device)
    matrix = torch.cat((eye.diff(dim=0), 0.1 * eye.diff(n=2, dim=0) / dt**2))
    smooth_jac = sparse.kron(sparse.csr_matrix(matrix[:, free].double().numpy()),
                             sparse.eye(joints), format="csr")
    hessian = smooth_jac.T @ smooth_jac
    best = None
    best_cost = float("inf")
    initial_violation = None
    min_violation = float("inf")
    min_violation_q = None

    def evaluate(path, substeps, linearize=False):
        pairs = torch.stack((path[:-1], path[1:]), dim=1)
        segment = yield substeps, linearize, segment_minimum, pairs
        cart = [error(path[index]) for index, error in active_targets]
        return segment, cart

    def violation(segment, cart):
        return segment.value.clamp_min(0).sum() + sum(error.abs().sum() for error in cart)

    def rows(segment, cart):
        blocks = [_waypoint_rows(segment.jacobian, segment.segment, fixed, len(q))]
        values = [segment.value]
        for (index, error), value in zip(active_targets, cart, strict=True):
            with _AD_LOCK:
                jac = jacfwd(error)(q[index])
            block = _waypoint_rows(jac[:, None, :], torch.full((len(value),), index), fixed, len(q))
            blocks.extend((block, -block))
            values.extend((value, -value))
        return torch.cat(values).double().numpy(), sparse.vstack(blocks, format="csr")

    if initial_feasible:
        best, best_cost = q.clone(), float(0.5 * (matrix @ q).square().sum())
    radius, penalty = 0.1, 10.0
    for _ in range(iterations):
        # Compare the exact merit and local model on the same sampling grid.
        substeps = substeps_for(q)
        segment, cart = yield from evaluate(q, substeps, linearize=True)
        value, jac = rows(segment, cart)
        residual = matrix @ q
        cost = float(0.5 * residual.square().sum())
        gradient = smooth_jac.T @ residual.flatten().double().numpy()
        current_violation = float(violation(segment, cart))
        if initial_violation is None:
            initial_violation = current_violation
        if current_violation < min_violation:
            min_violation = current_violation
            min_violation_q = q.clone()
        qp_cache = {}
        while radius >= 1e-4:
            lo = (lower - q[free]).clamp_min(-radius).flatten().double().numpy()
            hi = (upper - q[free]).clamp_max(radius).flatten().double().numpy()
            if pinned_flat is not None:
                lo[pinned_flat] = 0.0
                hi[pinned_flat] = 0.0
            step = _qp_step(hessian, gradient, value, jac, lo, hi, penalty, qp_cache)
            if step is None:
                radius *= 0.25
                continue
            predicted = (penalty * (current_violation - np.maximum(value + jac @ step, 0).sum())
                         - step @ gradient - 0.5 * step @ hessian @ step)
            if predicted <= 1e-9:
                break
            trial = q.clone()
            trial[free] += torch.as_tensor(step, dtype=q.dtype).reshape_as(q[free])
            trial[free] = trial[free].clamp(lower, upper)
            trial_segment, trial_cart = yield from evaluate(trial, substeps)
            trial_cost = float(0.5 * (matrix @ trial).square().sum())
            trial_violation = float(violation(trial_segment, trial_cart))
            actual = cost - trial_cost + penalty * (current_violation - trial_violation)
            if actual >= 0.25 * predicted:
                q = trial
                radius *= 1.5
                if trial_violation < min_violation:
                    min_violation = trial_violation
                    min_violation_q = q.clone()
                if trial_cost < best_cost and _could_be_feasible(trial_segment, trial_cart):
                    if _check_feasible(feasible, q, trial_segment):
                        if first_feasible:
                            return q
                        best, best_cost = q.clone(), trial_cost
                break
            radius *= 0.25
        else:
            predicted = 0.0
        if predicted <= 1e-9:
            if best is not None and torch.equal(q, best):
                break
            if _could_be_feasible(segment, cart) and _check_feasible(feasible, q, segment):
                if cost < best_cost:
                    best, best_cost = q.clone(), cost
                if first_feasible:
                    return q
                break
            if penalty >= 1e5:
                break
            penalty *= 10
            radius = 0.1
    if best is None and min_violation_q is not None and initial_violation is not None and min_violation < initial_violation:
        try:
            feasible.warm_start = min_violation_q
        except AttributeError:
            pass
    return best
