"""CPU trust-region SQP for a quadratic joint objective and nonlinear inequalities."""

import numpy as np
import torch
from scipy import sparse
from torch.func import jacfwd, vmap

try:
    import osqp
except ImportError as error:
    raise ImportError("CPU SQP requires the gobot[planning-sqp] extra") from error


def _qp_step(hessian, gradient, values, jacobian, lower, upper, penalty):
    """Minimize the quadratic model plus an L1 cost on linearized violations."""
    size = gradient.size
    # Eliminate inequalities guaranteed to hold anywhere in this trust box.
    worst = values + jacobian.maximum(0) @ upper + jacobian.minimum(0) @ lower
    active = worst > 0
    values, jacobian = values[active], jacobian[active]
    count = len(values)
    # A nonnegative slack per inequality keeps the QP feasible at a colliding seed.
    zero = sparse.csc_matrix((count, count))
    identity = sparse.eye(count, format="csc")
    matrix = sparse.vstack([
        sparse.eye(size + count, format="csc"),
        sparse.hstack([jacobian, -identity]),
    ], format="csc")
    qp = osqp.OSQP()
    qp.setup(P=sparse.block_diag([sparse.triu(hessian), zero], format="csc"),
             q=np.r_[gradient, np.full(count, penalty)], A=matrix,
             l=np.r_[lower, np.zeros(count), np.full(count, -np.inf)],
             u=np.r_[upper, np.full(count, np.inf), -values],
             verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=20000, polishing=True)
    result = qp.solve(raise_error=False)
    if result.info.status == "solved":
        return result.x[:size]
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


def optimize_sqp(initial, fixed, lower, upper, matrix, constraints_for, targets,
                 feasible, iterations):
    """Convexify segment and Cartesian terms, shrink or expand a box on joint increments."""
    free = slice(fixed, -fixed)
    q = initial.clone()
    joints = q.shape[1]
    smooth_jac = sparse.kron(sparse.csr_matrix(matrix[:, free].double().numpy()),
                             sparse.eye(joints), format="csr")
    hessian = smooth_jac.T @ smooth_jac
    best = None
    best_cost = float("inf")

    def evaluate(path, segment_constraints):
        pairs = torch.stack((path[:-1], path[1:]), dim=1)
        segment = segment_constraints(pairs)
        cart = [error(path[index]) for index, error in targets]
        return pairs, segment, cart

    def violation(segment, cart):
        return segment.clamp_min(0).sum() + sum(error.abs().sum() for error in cart)

    def rows(pairs, segment, cart, segment_constraints):
        # Components far from the contact/velocity threshold contribute zero merit.
        # Trial evaluation still includes all pairs, including newly active ones.
        near = segment > -0.05
        near[:, :2 * joints] = torch.isfinite(segment[:, :2 * joints])
        indices = near.any(dim=-1).nonzero().flatten()
        blocks, values = [], []
        if len(indices):
            local = vmap(jacfwd(segment_constraints))(pairs[indices])
            selected, component = near[indices].nonzero(as_tuple=True)
            segment_ids = indices[selected]
            blocks.append(_waypoint_rows(local[selected, component], segment_ids, fixed, len(q)))
            values.append(segment[segment_ids, component])
        for (index, error), value in zip(targets, cart, strict=True):
            jac = jacfwd(error)(q[index])
            block = _waypoint_rows(jac[:, None, :], torch.full((len(value),), index), fixed, len(q))
            blocks.extend((block, -block))
            values.extend((value, -value))
        if not values:
            return np.empty(0), sparse.csr_matrix((0, q[free].numel()))
        return torch.cat(values).double().numpy(), sparse.vstack(blocks, format="csr")

    if feasible(q):
        best, best_cost = q.clone(), float(0.5 * (matrix @ q).square().sum())
    radius, penalty = 0.1, 10.0
    for _ in range(iterations):
        # Compare the exact merit and local model on the same sampling grid.
        segment_constraints = constraints_for(q)
        pairs, segment, cart = evaluate(q, segment_constraints)
        value, jac = rows(pairs, segment, cart, segment_constraints)
        residual = matrix @ q
        cost = float(0.5 * residual.square().sum())
        gradient = smooth_jac.T @ residual.flatten().double().numpy()
        current_violation = float(violation(segment, cart))
        while radius >= 1e-4:
            lo = (lower - q[free]).clamp_min(-radius).flatten().double().numpy()
            hi = (upper - q[free]).clamp_max(radius).flatten().double().numpy()
            step = _qp_step(hessian, gradient, value, jac, lo, hi, penalty)
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
            _, trial_segment, trial_cart = evaluate(trial, segment_constraints)
            trial_cost = float(0.5 * (matrix @ trial).square().sum())
            actual = cost - trial_cost + penalty * (current_violation - float(violation(trial_segment, trial_cart)))
            if actual >= 0.25 * predicted:
                q = trial
                radius *= 1.5
                if trial_cost < best_cost and feasible(q):
                    best, best_cost = q.clone(), trial_cost
                break
            radius *= 0.25
        else:
            predicted = 0.0
        if predicted <= 1e-9:
            if feasible(q) or penalty >= 1e5:
                break
            penalty *= 10
            radius = 0.1
    return best
