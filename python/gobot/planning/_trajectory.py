"""Least-squares terms for piecewise-linear joint trajectories."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os

import numpy as np
import torch
from torch.func import jacfwd, jacrev, vmap

from ._geometry import clearance_mask
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
    poses: tuple[torch.Tensor, torch.Tensor] | None = None
    substeps: int = 0
    segment_ok: torch.Tensor | None = None

    @property
    def convex_ok(self) -> bool | None:
        if self.segment_ok is None:
            return None
        return bool(self.segment_ok.all())

    def slice(self, first, count):
        take = (self.segment >= first) & (self.segment < first + count)
        jac = None if self.jacobian is None else self.jacobian[take]
        poses = None
        if self.poses is not None and self.substeps > 0:
            p, r = self.poses
            lo, hi = first * self.substeps, (first + count) * self.substeps
            poses = (p[lo:hi], r[lo:hi])
        seg_ok = None if self.segment_ok is None else self.segment_ok[first:first + count]
        return SegmentRows(self.segment[take] - first, self.value[take], jac, poses, self.substeps, seg_ok)


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


def _pack_segment_rows(segments, values, derivatives, linearize, poses, substeps, seg_ok):
    if len(segments) == 1:
        seg = segments[0]
        val = values[0]
        jac = derivatives[0] if linearize else None
        return SegmentRows(seg, val, jac, poses, substeps, seg_ok)
    # Match the segment-major order of the dense constraints, including ties.
    merged_seg = torch.cat(segments)
    order = torch.argsort(merged_seg, stable=True)
    jac = torch.cat(derivatives)[order] if linearize else None
    return SegmentRows(merged_seg[order], torch.cat(values)[order], jac, poses, substeps, seg_ok)


def sqp_segment_rows(scene, dt, margins, world_margin, segment_inequalities, pairs, fractions, linearize,
                     segment_minimum=False):
    from ._sqp import _COLLISION_LINEARIZATION_BUFFER_M

    # Rows outside this window have zero contribution to the L1 violation.
    buffer = _COLLISION_LINEARIZATION_BUFFER_M if linearize else 0.0
    joints = pairs.size(-1)
    n_segs = len(pairs)
    substeps = len(fractions)
    if not scene.distance_models:
        def constraints(pair):
            return segment_inequalities(pair, fractions)[0]

        values = constraints(pairs)
        speed_tol = scene.speed * 1e-6
        speed_bad = (values[:, :2 * joints] > torch.cat((speed_tol, speed_tol))).any(-1)
        col_bad = (values[:, 2 * joints:] > 1e-6).any(-1) if values.size(-1) > 2 * joints else False
        seg_ok = ~(speed_bad | col_bad)
        active = values > -buffer
        active[:, :2 * joints] = torch.isfinite(values[:, :2 * joints])
        rows, cols = active.nonzero(as_tuple=True)
        jac = _selected_derivatives(constraints, pairs, active) if linearize else None
        return SegmentRows(rows, values[rows, cols], jac, None, substeps, seg_ok)

    speed = (pairs[:, 1] - pairs[:, 0]) / dt
    seg_ok = (speed.abs() <= scene.speed * (1 + 1e-6)).all(dim=-1)
    speed_values = torch.cat((speed - scene.speed, -speed - scene.speed), dim=-1)
    s_rows, s_cols = torch.isfinite(speed_values).nonzero(as_tuple=True)
    segments, values, derivatives = [], [], []
    if len(s_rows):
        segments.append(s_rows)
        values.append(speed_values[s_rows, s_cols])
        if linearize:
            eye = torch.eye(joints, dtype=pairs.dtype) / dt
            speed_jac = torch.stack((-eye, eye), dim=1)
            derivatives.append(torch.cat((speed_jac, -speed_jac))[s_cols])

    model, = scene.distance_models
    poses_pr = None
    if model.labels:
        samples = torch.lerp(pairs[:, :1], pairs[:, 1:], fractions).flatten(0, 1)
        # Aim slightly inside the feasible region on the first low-latency attempt.
        # Full-row retries and final validation use the requested clearances.
        opt_margins = margins + 1e-4 if segment_minimum else margins
        rows, residual, gradient, poses_pr, col_ok = model.constraint_rows(
            scene.poses, samples, opt_margins, buffer, linearize,
            group_size=substeps, segment_minimum=segment_minimum,
            scene=scene, return_extra=True, check_margins=margins,
        )
        seg_ok = seg_ok & col_ok
        if len(rows) or not segments:
            segments.append(rows // substeps)
            values.append(residual)
            if linearize:
                if gradient is not None:
                    t = fractions[rows % substeps]
                    derivatives.append(torch.stack(((1 - t) * gradient, t * gradient), dim=1))
                else:
                    derivatives.append(pairs.new_empty((0, 2, joints)))

    if scene.swept_pairs:
        def swept(pair):
            return world_margin - scene.swept_world_distances(pair[..., 0, :], pair[..., 1, :])

        sweep = swept(pairs)
        seg_ok = seg_ok & clearance_mask(world_margin - sweep, world_margin).all(dim=-1)
        active = sweep > -buffer
        w_rows, w_cols = active.nonzero(as_tuple=True)
        if len(w_rows):
            segments.append(w_rows)
            values.append(sweep[w_rows, w_cols])
            if linearize:
                derivatives.append(_selected_derivatives(swept, pairs, active))

    if not segments:
        empty_i = torch.empty((0,), dtype=torch.long, device=pairs.device)
        empty_v = pairs.new_empty((0,))
        empty_j = pairs.new_empty((0, 2, joints)) if linearize else None
        return SegmentRows(empty_i, empty_v, empty_j, poses_pr, substeps, seg_ok)
    return _pack_segment_rows(segments, values, derivatives, linearize, poses_pr, substeps, seg_ok)


def sqp_segment_rows_merged(scene, dt, margins, world_margin, segment_inequalities, requests):
    """Evaluate heterogeneous SQP requests in one merged FK and narrow-phase batch."""
    from ._sqp import _COLLISION_LINEARIZATION_BUFFER_M

    if not scene.distance_models or not scene.distance_models[0].labels:
        # Fallback for torch backend or collision-free scenes.
        replies = {}
        groups = defaultdict(list)
        for row, substeps, linearize, seg_min, pairs in requests:
            groups[substeps, linearize, seg_min].append((row, pairs))
        for (substeps, linearize, seg_min), items in groups.items():
            pairs_list = [p for _, p in items]
            merged = torch.cat(pairs_list) if len(pairs_list) > 1 else pairs_list[0]
            fractions = _sample_fractions(merged, substeps)
            values = sqp_segment_rows(scene, dt, margins, world_margin, segment_inequalities,
                                      merged, fractions, linearize, seg_min)
            offset = 0
            for row, group in items:
                replies[row] = values.slice(offset, len(group)) if len(items) > 1 else values
                offset += len(group)
        return replies

    model, = scene.distance_models
    need_joints = any(linearize for _, _, linearize, _, _ in requests)
    frac_cache = {}
    sample_blocks = []
    meta = []
    sample_offset = 0
    for row, substeps, linearize, seg_min, pairs in requests:
        fractions = frac_cache.get(substeps)
        if fractions is None:
            fractions = _sample_fractions(pairs, substeps)
            frac_cache[substeps] = fractions
        samples = torch.lerp(pairs[:, :1], pairs[:, 1:], fractions).flatten(0, 1)
        n_samples = len(samples)
        sample_blocks.append(samples)
        meta.append((row, substeps, linearize, seg_min, pairs, fractions, sample_offset, n_samples))
        sample_offset += n_samples

    all_samples = torch.cat(sample_blocks) if len(sample_blocks) > 1 else sample_blocks[0]
    if need_joints:
        all_p, all_r, all_o, all_a = scene.poses_and_joints(all_samples)
    else:
        all_p, all_r = scene.poses(all_samples)
        all_o = all_a = None
    all_frames = model.world_frames(all_p, all_r)
    all_frames_np = np.ascontiguousarray(all_frames.detach().numpy(), dtype=np.float64)

    # Run fast C++ AABB broadphase per request slice, then combine into ONE narrow-phase query.
    thr_cache = {}
    cand_rows, cand_pairs, cand_spans = [], [], []
    cand_offset = 0
    for row, substeps, linearize, seg_min, pairs, fractions, s_off, s_cnt in meta:
        buffer = _COLLISION_LINEARIZATION_BUFFER_M if linearize else 0.0
        key = (buffer, seg_min)
        thr_np = thr_cache.get(key)
        if thr_np is None:
            opt_m = margins + 1e-4 if seg_min else margins
            thr_np = np.ascontiguousarray((opt_m + buffer).detach().numpy(), dtype=np.float64)
            thr_cache[key] = thr_np
        r_loc, p_loc = model.bounds[False].candidates(
            all_frames_np[s_off:s_off + s_cnt], model.moving, model._endpoints_np,
            model.planes, thr_np, substeps,
        )
        n_c = len(r_loc)
        cand_spans.append((cand_offset, n_c))
        if n_c:
            cand_rows.append(r_loc + s_off)
            cand_pairs.append(p_loc)
            cand_offset += n_c

    if cand_offset:
        merged_rows = np.concatenate(cand_rows) if len(cand_rows) > 1 else cand_rows[0]
        merged_pairs = np.concatenate(cand_pairs) if len(cand_pairs) > 1 else cand_pairs[0]
        merged_data = torch.from_numpy(
            model._selected_query(all_frames_np, merged_rows, merged_pairs, False)
        ).to(all_frames.dtype)
        merged_rows_t = torch.from_numpy(merged_rows)
        merged_pairs_t = torch.from_numpy(merged_pairs)
    else:
        merged_data = all_frames.new_empty((0, 10))
        merged_rows_t = torch.empty((0,), dtype=torch.long)
        merged_pairs_t = torch.empty((0,), dtype=torch.long)

    # Assemble per-request SegmentRows (analytic gradient only for linearize=True).
    replies = {}
    joints = all_samples.size(-1)
    eye_cache = None
    for (row, substeps, linearize, seg_min, pairs, fractions, s_off, s_cnt), (c_off, c_cnt) in zip(meta, cand_spans):
        buffer = _COLLISION_LINEARIZATION_BUFFER_M if linearize else 0.0
        n_segs = len(pairs)
        speed = (pairs[:, 1] - pairs[:, 0]) / dt
        seg_ok = (speed.abs() <= scene.speed * (1 + 1e-6)).all(dim=-1)
        speed_values = torch.cat((speed - scene.speed, -speed - scene.speed), dim=-1)
        s_rows, s_cols = torch.isfinite(speed_values).nonzero(as_tuple=True)
        segments, values, derivatives = [], [], []
        if len(s_rows):
            segments.append(s_rows)
            values.append(speed_values[s_rows, s_cols])
            if linearize:
                if eye_cache is None:
                    eye = torch.eye(joints, dtype=pairs.dtype) / dt
                    sj = torch.stack((-eye, eye), dim=1)
                    eye_cache = torch.cat((sj, -sj))
                derivatives.append(eye_cache[s_cols])

        opt_margins = margins + 1e-4 if seg_min else margins
        if c_cnt:
            c_rows_glob = merged_rows_t[c_off:c_off + c_cnt]
            c_rows = c_rows_glob - s_off
            c_ids = merged_pairs_t[c_off:c_off + c_cnt]
            c_data = merged_data[c_off:c_off + c_cnt]
            failed = ~clearance_mask(c_data[:, 0], margins[c_ids])
            if failed.any():
                seg_ok[c_rows[failed] // substeps] = False
            residual = opt_margins[c_ids] - c_data[:, 0]
            near = (residual > -buffer).nonzero().flatten()
            if seg_min and len(near):
                group = ((c_rows[near] // substeps) * len(opt_margins) + c_ids[near]).numpy()
                order = np.lexsort((-residual[near].numpy(), group))
                first = np.r_[True, np.diff(group[order]) != 0]
                near = near[np.sort(order[first])]
            c_rows_n = c_rows[near]
            c_ids_n = c_ids[near]
            c_res_n = residual[near]
            if len(c_rows_n) or not segments:
                segments.append(c_rows_n // substeps)
                values.append(c_res_n)
                if linearize:
                    if len(c_rows_n):
                        grad = model._analytic_gradient(
                            scene, all_frames, c_rows_glob[near], c_ids_n, c_data[near], all_o, all_a,
                        )
                        t = fractions[c_rows_n % substeps]
                        derivatives.append(torch.stack(((1 - t) * grad, t * grad), dim=1))
                    else:
                        derivatives.append(pairs.new_empty((0, 2, joints)))
        elif not segments:
            segments.append(torch.empty((0,), dtype=torch.long, device=pairs.device))
            values.append(pairs.new_empty((0,)))
            if linearize:
                derivatives.append(pairs.new_empty((0, 2, joints)))

        if scene.swept_pairs:
            def swept(pair):
                return world_margin - scene.swept_world_distances(pair[..., 0, :], pair[..., 1, :])

            sweep = swept(pairs)
            seg_ok = seg_ok & clearance_mask(world_margin - sweep, world_margin).all(dim=-1)
            active = sweep > -buffer
            w_rows, w_cols = active.nonzero(as_tuple=True)
            if len(w_rows):
                segments.append(w_rows)
                values.append(sweep[w_rows, w_cols])
                if linearize:
                    derivatives.append(_selected_derivatives(swept, pairs, active))

        poses_slice = (all_p[s_off:s_off + s_cnt], all_r[s_off:s_off + s_cnt])
        replies[row] = _pack_segment_rows(segments, values, derivatives, linearize, poses_slice, substeps, seg_ok)
    return replies


def run_trajectories(jobs, evaluate, *, merged_evaluate=None):
    """Advance independent SQP jobs together with parallel QP/validation and merged collision batches."""
    results = [None] * len(jobs)
    replies = dict.fromkeys(range(len(jobs)))
    workers = min(len(jobs), max(1, os.cpu_count() or 1))
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

    def advance(row, reply):
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
            for row, finished, payload in outcomes:
                if finished:
                    results[row] = payload
                else:
                    substeps, derivative, segment_minimum, pairs = payload
                    requests.append((row, substeps, derivative, segment_minimum, pairs))
            replies = {}
            if not requests:
                break
            if len(requests) == 1:
                row, substeps, derivative, segment_minimum, pairs = requests[0]
                fractions = _sample_fractions(pairs, substeps)
                replies[row] = evaluate(pairs, fractions, derivative, segment_minimum)
            elif merged_evaluate is not None:
                replies = merged_evaluate(requests)
            else:
                groups = defaultdict(list)
                for row, substeps, derivative, segment_minimum, pairs in requests:
                    groups[substeps, derivative, segment_minimum].append((row, pairs))
                for (substeps, derivative, segment_minimum), items in groups.items():
                    pairs_list = [group for _, group in items]
                    merged = torch.cat(pairs_list) if len(pairs_list) > 1 else pairs_list[0]
                    fractions = _sample_fractions(merged, substeps)
                    values = evaluate(merged, fractions, derivative, segment_minimum)
                    offset = 0
                    for row, group in items:
                        replies[row] = values.slice(offset, len(group)) if len(items) > 1 else values
                        offset += len(group)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
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
