"""Least-squares terms for piecewise-linear joint trajectories."""

import torch
from torch.func import jacfwd, vmap

from ._solver import augmented, solve


def interpolation(positions, substeps):
    fraction = torch.arange(substeps, dtype=positions.dtype, device=positions.device) / substeps
    samples = torch.lerp(positions[:-1, None, :], positions[1:, None, :], fraction[:, None])
    return torch.cat((samples.flatten(0, 1), positions[-1:]))


def acceleration_stencil(eye, dt):
    """Five-point central acceleration at every waypoint with a full stencil."""
    return (-eye[4:] + 16 * eye[3:-1] - 30 * eye[2:-2] + 16 * eye[1:-3] - eye[:-4]) / (12 * dt**2)


def collision_penalty(distance, margin):
    near = (margin - distance).clamp_min(0).square() / (2 * (margin + 1e-6))
    return torch.where(distance < 0, margin * 0.5 - distance, near)


def optimize(scene, initial, dt, margin, world_margin, substeps, iterations, feasible, fixed):
    waypoints, joints = initial.shape
    free = slice(fixed, -fixed)
    eye = torch.eye(waypoints, dtype=initial.dtype, device=initial.device)
    matrix = torch.cat((0.01 * eye, eye.diff(dim=0), 0.1 * acceleration_stencil(eye, dt)))
    offset = torch.zeros_like(matrix @ initial)
    offset[:waypoints] = 0.01 * scene.initial
    joint_eye = torch.eye(joints, dtype=initial.dtype, device=initial.device)
    smooth_jac = torch.kron(matrix[:, free].contiguous(), joint_eye)
    smooth_hessian = smooth_jac.T @ smooth_jac
    fractions = torch.arange(substeps, dtype=initial.dtype, device=initial.device)[:, None] / substeps
    margins = scene.pair_margins(margin, world_margin)

    def trajectory(interior):
        return torch.cat((initial[:fixed], interior, initial[-fixed:]))

    def segment_terms(pair):
        start, end = pair.unbind(-2)
        speed = (end - start) / dt
        samples = torch.lerp(start.unsqueeze(-2), end.unsqueeze(-2), fractions)
        distances = scene.distances(samples)
        soft = 5 * collision_penalty(distances[..., 0, :scene.self_pair_count], margin)
        constraints = torch.cat((
            speed - scene.speed, -speed - scene.speed, (margins - distances).flatten(-2),
            world_margin - scene.swept_world_distances(start, end),
        ), dim=-1)
        return soft, constraints

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
