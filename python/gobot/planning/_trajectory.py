"""Least-squares terms for piecewise-linear joint trajectories."""

import math
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.func import jacfwd, jacrev, vmap

from ._solver import augmented, solve


def grid_substeps(q, minimum, max_length):
    if max_length is None:
        return minimum
    longest = torch.linalg.vector_norm(q.diff(dim=0), dim=-1).max().item()
    return max(minimum, math.ceil(longest / max_length))


def _sample_fractions(positions, count):
    return torch.arange(count, dtype=positions.dtype, device=positions.device)[:, None] / count


def interpolation(positions, substeps):
    samples = torch.lerp(positions[:-1, None, :], positions[1:, None, :], _sample_fractions(positions, substeps))
    return torch.cat((samples.flatten(0, 1), positions[-1:]))


def acceleration_stencil(eye, dt):
    """Five-point central acceleration at every waypoint with a full stencil."""
    return (-eye[4:] + 16 * eye[3:-1] - 30 * eye[2:-2] + 16 * eye[1:-3] - eye[:-4]) / (12 * dt**2)


def collision_penalty(distance, margin):
    near = (margin - distance).clamp_min(0).square() / (2 * (margin + 1e-6))
    return torch.where(distance < 0, margin * 0.5 - distance, near)


def make_segment_constraints(scene, dt, margin, world_margin):
    """Shared segment kernel for trajectories in the same scene."""
    margins = scene.pair_margins(margin, world_margin)

    def segment_inequalities(pair, fractions):
        start, end = pair.unbind(-2)
        speed = (end - start) / dt
        samples = torch.lerp(start.unsqueeze(-2), end.unsqueeze(-2), fractions)
        distances = scene.distances(samples)
        constraints = torch.cat((
            speed - scene.speed, -speed - scene.speed, (margins - distances).flatten(-2),
            world_margin - scene.swept_world_distances(start, end),
        ), dim=-1)
        return constraints, distances

    return segment_inequalities


@dataclass
class SegmentRows:
    segment: torch.Tensor
    value: torch.Tensor
    jacobian: torch.Tensor | None

    def slice(self, first, count):
        take = (self.segment >= first) & (self.segment < first + count)
        jac = None if self.jacobian is None else self.jacobian[take]
        return SegmentRows(self.segment[take] - first, self.value[take], jac)


def _selected_derivatives(function, pairs, active):
    """Different segments can select different sweep or primitive constraints."""
    lengths = active.sum(-1)
    blocks, segment_ids = [], []
    for count in lengths[lengths > 0].unique().tolist():
        rows = (lengths == count).nonzero().flatten()
        columns = active[rows].nonzero()[:, 1].reshape(-1, count)

        def selected(pair, indices):
            return function(pair).index_select(-1, indices)

        jac = vmap(jacrev(selected))(pairs[rows], columns)
        blocks.append(jac.flatten(0, 1))
        segment_ids.append(rows.repeat_interleave(count))
    if not blocks:
        return pairs.new_empty((0, *pairs.shape[1:]))
    return torch.cat(blocks)[torch.argsort(torch.cat(segment_ids), stable=True)]


def sqp_segment_rows(scene, dt, margins, world_margin, segment_inequalities, pairs, fractions, linearize,
                     segment_minimum=False):
    from ._sqp import _COLLISION_LINEARIZATION_BUFFER_M

    # Rows outside this window have zero contribution to the L1 violation.
    buffer = _COLLISION_LINEARIZATION_BUFFER_M if linearize else 0.0
    joints = pairs.size(-1)
    if not scene.distance_models:
        def constraints(pair):
            return segment_inequalities(pair, fractions)[0]

        values = constraints(pairs)
        active = values > -buffer
        active[:, :2 * joints] = torch.isfinite(values[:, :2 * joints])
        rows, cols = active.nonzero(as_tuple=True)
        jac = _selected_derivatives(constraints, pairs, active) if linearize else None
        return SegmentRows(rows, values[rows, cols], jac)

    speed = (pairs[:, 1] - pairs[:, 0]) / dt
    speed_values = torch.cat((speed - scene.speed, -speed - scene.speed), dim=-1)
    rows, cols = torch.isfinite(speed_values).nonzero(as_tuple=True)
    segments, values = [rows], [speed_values[rows, cols]]
    derivatives = []
    if linearize:
        eye = torch.eye(joints, dtype=pairs.dtype) / dt
        speed_jac = torch.stack((-eye, eye), dim=1)
        derivatives.append(torch.cat((speed_jac, -speed_jac))[cols])

    model, = scene.distance_models
    if model.labels:
        samples = torch.lerp(pairs[:, :1], pairs[:, 1:], fractions).flatten(0, 1)
        # Aim slightly inside the feasible region on the first low-latency attempt.
        # Full-row retries and final validation use the requested clearances.
        if segment_minimum:
            margins = margins + 1e-4
        rows, residual, gradient = model.constraint_rows(scene.poses, samples, margins, buffer, linearize,
                                                         group_size=len(fractions), segment_minimum=segment_minimum)
        segments.append(rows // len(fractions))
        values.append(residual)
        if gradient is not None:
            t = fractions[rows % len(fractions)]
            derivatives.append(torch.stack(((1 - t) * gradient, t * gradient), dim=1))

    if scene.swept_pairs:
        def swept(pair):
            return world_margin - scene.swept_world_distances(pair[..., 0, :], pair[..., 1, :])

        sweep = swept(pairs)
        active = sweep > -buffer
        rows, cols = active.nonzero(as_tuple=True)
        segments.append(rows)
        values.append(sweep[rows, cols])
        if linearize:
            derivatives.append(_selected_derivatives(swept, pairs, active))
    # Match the segment-major order of the dense constraints, including ties.
    segments = torch.cat(segments)
    order = torch.argsort(segments, stable=True)
    jac = torch.cat(derivatives)[order] if linearize else None
    return SegmentRows(segments[order], torch.cat(values)[order], jac)


def run_trajectories(jobs, evaluate):
    """Advance independent SQP jobs together at their segment evaluation requests."""
    results = [None] * len(jobs)
    replies = dict.fromkeys(range(len(jobs)))
    while replies:
        groups = defaultdict(list)
        for row, reply in replies.items():
            try:
                substeps, derivative, segment_minimum, pairs = jobs[row].send(reply)
            except StopIteration as done:
                results[row] = done.value
            else:
                groups[substeps, derivative, segment_minimum].append((row, pairs))
        replies = {}
        for (substeps, derivative, segment_minimum), requests in groups.items():
            pairs = [group for _, group in requests]
            merged = torch.cat(pairs) if len(pairs) > 1 else pairs[0]
            fractions = _sample_fractions(merged, substeps)

            values = evaluate(merged, fractions, derivative, segment_minimum)
            offset = 0
            for row, group in requests:
                replies[row] = values.slice(offset, len(group)) if len(requests) > 1 else values
                offset += len(group)
    return results


def optimize(scene, initial, dt, margin, substeps, iterations, feasible, fixed, solver,
             targets, max_length, segment_inequalities, segment_minimum=False):
    if solver in ("sqp", "sqp_fast"):
        from ._sqp import optimize_sqp

        return (yield from optimize_sqp(initial, fixed, scene.lower, scene.upper, dt,
                                        lambda q: grid_substeps(q, substeps, max_length),
                                        targets, feasible, iterations,
                                        first_feasible=solver == "sqp_fast", segment_minimum=segment_minimum))

    waypoints, joints = initial.shape
    eye = torch.eye(waypoints, dtype=initial.dtype, device=initial.device)
    free = slice(fixed, -fixed)
    fractions = _sample_fractions(initial, grid_substeps(initial, substeps, max_length))

    def trajectory(interior):
        return torch.cat((initial[:fixed], interior, initial[-fixed:]))

    def segment_terms(pair):
        constraints, distances = segment_inequalities(pair, fractions)
        soft = 5 * collision_penalty(distances[..., 0, :scene.self_pair_count], margin)
        return soft, constraints

    matrix = torch.cat((0.01 * eye, eye.diff(dim=0), 0.1 * acceleration_stencil(eye, dt)))
    offset = torch.zeros_like(matrix @ initial)
    offset[:waypoints] = 0.01 * scene.initial

    joint_eye = torch.eye(joints, dtype=initial.dtype, device=initial.device)
    smooth_jac = torch.kron(matrix[:, free].contiguous(), joint_eye)
    smooth_hessian = smooth_jac.T @ smooth_jac

    def segments(q):
        return torch.stack((q[:-1], q[1:]), dim=1)

    def terms(interior):
        q = trajectory(interior)
        soft, constraints = segment_terms(segments(q))
        # The omitted final waypoint self cost is constant since its joints are fixed.
        return torch.cat(((matrix @ q - offset).flatten(), soft.flatten())), constraints.flatten()

    def local_residual(pair, dual, penalty):
        soft, constraints = segment_terms(pair)
        residual = torch.cat((soft, augmented(constraints, dual, penalty)), dim=-1)
        return residual, residual

    local_jac = vmap(jacfwd(local_residual, has_aux=True), in_dims=(0, 0, None))

    def linearize(interior, multiplier, penalty):
        q = trajectory(interior)
        pairs = segments(q)
        soft, constraints = segment_terms(pairs)
        dual = multiplier.reshape_as(constraints)
        active = (soft != 0).any(-1) | ((constraints + dual / penalty) > 0).any(-1)
        gradient = smooth_jac.T @ (matrix @ q - offset).flatten()
        if not active.any():
            return smooth_hessian, gradient
        # Each nonlinear factor touches exactly two waypoints. Accumulate its
        # small Jacobian blocks directly into the normal equations.
        indices = active.nonzero().flatten()
        jac, residual = local_jac(pairs[indices], dual[indices], penalty)
        jac = jac.flatten(-2)
        transpose = jac.transpose(-1, -2)
        h = (transpose @ jac).reshape(-1, 2, joints, 2, joints)
        g = (transpose @ residual.unsqueeze(-1)).reshape(-1, 2, joints)
        hessian = q.new_zeros(waypoints, joints, waypoints, joints)
        local_gradient = torch.zeros_like(q)
        for a in range(2):
            local_gradient.index_add_(0, indices + a, g[:, a])
            for b in range(2):
                hessian[indices + a, :, indices + b, :] += h[:, a, :, b, :]
        return (smooth_hessian + hessian[free, :, free, :].reshape_as(smooth_hessian),
                gradient + local_gradient[free].flatten())

    interior = solve(initial[free], scene.lower, scene.upper, terms, lambda x: feasible(trajectory(x)),
                     iterations, linearize)
    return trajectory(interior) if interior is not None else None
