"""Projected Levenberg–Marquardt with an inequality augmented Lagrangian."""

import torch
from torch.func import jacfwd


def augmented(inequality, multiplier, penalty):
    return (inequality + multiplier / penalty).clamp_min(0) * penalty**0.5


def solve_rows(initial, lower, upper, terms, feasible, iterations, linearize=None):
    """Solve independent least-squares problems along the first dimension."""
    x = initial.clamp(lower, upper)
    residual, inequality = terms(x)
    multiplier = torch.zeros_like(inequality)
    penalty = 10.0
    tolerance = 2 * torch.finfo(x.dtype).eps
    x_feasible = feasible(x)
    best = x
    best_cost = torch.where(x_feasible, residual.square().sum(-1), torch.inf)
    finished = torch.zeros_like(best_cost, dtype=torch.bool)
    eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)

    def merit_cost(res, ineq):
        return res.square().sum(-1) + augmented(ineq, multiplier, penalty).square().sum(-1)

    if linearize is None:
        # Perturb every independent row in the same joint direction. This
        # yields B local Jacobians (B, residuals, joints) in one AD transform.
        delta = torch.zeros_like(x[0])

        def linearize(value, dual, weight):
            def shifted(shift):
                res, con = terms(value + shift)
                r = torch.cat((res, augmented(con, dual, weight)), dim=-1)
                return r, r
            jac, r = jacfwd(shifted, has_aux=True)(delta)
            transpose = jac.transpose(-1, -2)
            return transpose @ jac, (transpose @ r.unsqueeze(-1)).squeeze(-1)

    for _ in range(6 if inequality.numel() else 1):
        damping = torch.full_like(best_cost, 1e-3)
        cost = merit_cost(residual, inequality)
        running = ~finished
        for _ in range(iterations):
            hessian, gradient = linearize(x, multiplier, penalty)
            finite = torch.isfinite(hessian).all(dim=(-1, -2)) & torch.isfinite(gradient).all(dim=-1)
            active = ((x <= lower) & (gradient > 0)) | ((x >= upper) & (gradient < 0))
            free = ~active & finite[:, None]
            hessian = torch.where(free.unsqueeze(-1) & free.unsqueeze(-2), hessian, 0)
            gradient = torch.where(free, gradient, 0)
            running = running & finite & (gradient.abs().amax(-1) > tolerance)
            if not running.any():
                break
            diagonal = hessian.diagonal(dim1=-2, dim2=-1).clamp_min(1)
            pending = running
            accepted_any = torch.zeros_like(running)
            for _ in range(8):
                system = hessian + torch.diag_embed(damping[:, None] * diagonal)
                factor, info = torch.linalg.cholesky_ex(system)
                safe_factor = torch.where((info == 0)[:, None, None], factor, eye)
                step = torch.cholesky_solve(-gradient.unsqueeze(-1), safe_factor)
                trial = (x + step.squeeze(-1)).clamp(lower, upper)
                trial_res, trial_ineq = terms(trial)
                trial_cost = merit_cost(trial_res, trial_ineq)
                accept = pending & (info == 0) & (trial_cost < cost)
                if accept.any():
                    x = torch.where(accept[:, None], trial, x)
                    residual = torch.where(accept[:, None], trial_res, residual)
                    inequality = torch.where(accept[:, None], trial_ineq, inequality)
                    cost = torch.where(accept, trial_cost, cost)
                    accepted_any = accepted_any | accept
                pending = pending & ~accept
                damping = torch.where(accept, (damping / 3).clamp_min(1e-7), damping)
                if not pending.any():
                    break
                damping = torch.where(pending, damping * 10, damping)
            running = running & ~pending
            if accepted_any.any():
                x_feasible = torch.where(accepted_any, feasible(x), x_feasible)
                objective = residual.square().sum(-1)
                improved = running & x_feasible & (objective < best_cost)
                best = torch.where(improved[:, None], x, best)
                best_cost = torch.where(improved, objective, best_cost)
            if not running.any():
                break
        finished = finished | x_feasible
        if finished.all():
            break
        multiplier = (multiplier + penalty * inequality).clamp_min(0)
        penalty *= 10
    return best, torch.isfinite(best_cost)


def solve(initial, lower, upper, terms, feasible, iterations, linearize):
    """Adapt the trajectory's structured normal equations to one solver row."""
    def row_terms(row):
        residual, inequality = terms(row.reshape_as(initial))
        return residual.reshape(1, -1), inequality.reshape(1, -1)

    def row_linearize(row, dual, penalty):
        hessian, gradient = linearize(row.reshape_as(initial), dual[0], penalty)
        return hessian[None], gradient[None]

    def row_feasible(row):
        return torch.as_tensor([feasible(row.reshape_as(initial))], device=row.device)

    best, success = solve_rows(initial.reshape(1, -1), lower.expand_as(initial).reshape(1, -1),
                               upper.expand_as(initial).reshape(1, -1), row_terms, row_feasible,
                               iterations, row_linearize)
    return best.reshape_as(initial) if success.item() else None
