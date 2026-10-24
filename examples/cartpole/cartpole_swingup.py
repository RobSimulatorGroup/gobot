# SPDX-License-Identifier: Apache-2.0
"""Single-pole iLQR swing-up using MuJoCo's native CPU Batch API.

Adapted from Kevin Zakka's Apache-2.0 mjbatch cartpole_swingup example.
The model is compiled from Gobot's cartpole.jscn; no parallel MJCF asset is kept.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

if __package__:
    from . import cartpole_task as task
else:
    import cartpole_task as task

EPS = 1e-6
ALPHAS = 0.5 ** np.arange(9)
W_CART, W_SPEED, W_SPIN, W_CTRL, TERMINAL = 0.1, 0.01, 0.01, 0.2, 100.0


class PlanningCancelled(Exception):
    """The editor stopped or reset this solve."""


@dataclass(frozen=True)
class Plan:
    states: np.ndarray
    actions: np.ndarray
    gains: np.ndarray
    hold_gain: np.ndarray
    cost: float
    iterations: int

    def force(self, knot: int, state: np.ndarray) -> float:
        x = np.asarray(state, dtype=np.float64).copy()
        if x.shape != (4,) or not np.isfinite(x).all():
            raise ValueError("Cartpole state must contain four finite values")
        if knot < len(self.actions):
            error = x - self.states[knot]
            error[1] = task.wrap(error[1])
            action = self.actions[knot] + self.gains[knot] @ error
        else:
            x[1] = task.wrap(x[1])
            action = self.hold_gain @ x
        return float(np.clip(action[0], -1.0, 1.0)) * task.FORCE_LIMIT


def cost(t, x, u):
    state = (1.0 - np.cos(x[..., 1]) + W_CART * x[..., 0] ** 2
             + W_SPEED * x[..., 2] ** 2 + W_SPIN * x[..., 3] ** 2)
    return (TERMINAL if t == task.HORIZON else 1.0) * state + W_CTRL * u[..., 0] ** 2


def expand(xs, us):
    weights = np.array([2 * W_CART, 0.0, 2 * W_SPEED, 2 * W_SPIN])
    lx = weights * xs
    diagonal = np.tile(weights, (len(xs), 1))
    lx[:, 1] = np.sin(xs[:, 1])
    diagonal[:, 1] = np.cos(xs[:, 1])
    lxx = diagonal[:, :, None] * np.eye(4)
    lx[-1] *= TERMINAL
    lxx[-1] *= TERMINAL
    return lx, lxx, 2 * W_CTRL * us, np.full((len(us), 1, 1), 2 * W_CTRL)


def backward(A, B, lx, lxx, lu, luu, lo, hi, mu):
    """Box-constrained iLQR pass; the one-dimensional QP has an exact solution."""
    count = len(lu)
    vx, vxx = lx[-1].copy(), lxx[-1].copy()
    k, gains = np.zeros((count, 1)), np.zeros((count, 1, 4))
    for t in reversed(range(count)):
        qx, qu = lx[t] + A[t].T @ vx, lu[t] + B[t].T @ vx
        qxx = lxx[t] + A[t].T @ vxx @ A[t]
        quu = luu[t] + B[t].T @ vxx @ B[t]
        qux = B[t].T @ vxx @ A[t]
        reg = vxx + mu * np.eye(4)
        quu_reg = luu[t] + B[t].T @ reg @ B[t]
        qux_reg = B[t].T @ reg @ A[t]
        curvature = quu_reg[0, 0]
        if not np.isfinite(curvature) or curvature <= 0.0:
            return None
        optimum = -qu[0] / curvature
        k[t, 0] = np.clip(optimum, lo[t, 0], hi[t, 0])
        if lo[t, 0] < optimum < hi[t, 0]:
            gains[t] = -qux_reg / curvature
        K = gains[t]
        vx = qx + K.T @ quu @ k[t] + K.T @ qu + qux.T @ k[t]
        vxx = qxx + K.T @ quu @ K + K.T @ qux + qux.T @ K
        vxx = 0.5 * (vxx + vxx.T)
    return k, gains


class Planner:
    """Owns numerical rollouts; it never accesses a live Gobot scene or editor."""

    def __init__(self, artifact, *, workers=0, cancelled: Callable[[], bool] = lambda: False):
        import mujoco
        from mujoco.batch import Batch

        if isinstance(workers, bool) or int(workers) != workers or workers < 0:
            raise ValueError("workers must be a non-negative integer")
        self.cancelled = cancelled
        self.model = mujoco.MjModel.from_xml_string(artifact["content"])
        self.model.opt.timestep = task.FIXED_DT
        m = self.model
        if (m.nq, m.nv, m.nu) != (2, 2, 1) or list(m.jnt_type) != [mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE]:
            raise ValueError("Expected one cart slide, one pole hinge and one cart actuator")
        if not np.allclose(m.actuator_ctrlrange[0], [-task.FORCE_LIMIT, task.FORCE_LIMIT]):
            raise ValueError("Compiled cart force range does not match the playback contract")
        self.batch = Batch(m, nsim=task.HORIZON * 6, nthread=int(workers))
        self.line = Batch(m, nsim=len(ALPHAS), nthread=int(workers))
        self.line_failed = np.zeros(len(ALPHAS), dtype=bool)
        self.fields = {id(b): tuple(b.bind(f) for f in ("qpos", "qvel", "ctrl", "qacc_warmstart"))
                       for b in (self.batch, self.line)}

    def check_cancelled(self):
        if self.cancelled():
            raise PlanningCancelled()

    def transition(self, x, u, *, line=False):
        self.check_cancelled()
        b = self.line if line else self.batch
        qpos, qvel, ctrl, warm = self.fields[id(b)]
        qpos[:], qvel[:] = x[:, :2], x[:, 2:]
        ctrl[:], warm[:] = task.FORCE_LIMIT * u, 0.0
        from mujoco.batch import SimulationError

        try:
            b.step(nstep=task.SUBSTEPS)
        except SimulationError as error:
            if not line:
                raise
            # A failed candidate must not discard the other line-search rows.
            self.line_failed[error.ids] = True
        return np.concatenate((qpos, qvel), axis=1)

    def linearize(self, xs, us):
        count, columns = task.HORIZON, 6
        x, u = np.repeat(xs[:-1], columns, axis=0), np.repeat(us, columns, axis=0)
        delta = EPS * np.tile(np.eye(columns)[:, 1:], (count, 1))
        delta[:, 4:] *= np.where(u + EPS > 1.0, -1.0, 1.0)
        x += delta[:, :4]
        before = self.batch.warning.copy()
        out = self.transition(x, u + delta[:, 4:]).reshape(count, columns, 4)
        if not np.isfinite(out).all() or np.any(self.batch.warning != before):
            raise RuntimeError("Invalid simulation while linearizing the cartpole trajectory")
        base = np.repeat(out[:, 0], columns - 1, axis=0)
        h = delta.reshape(count, columns, -1)[:, 1:].sum(axis=2).ravel()
        jac = (out[:, 1:].reshape(-1, 4) - base) / h[:, None]
        jac = np.swapaxes(jac.reshape(count, 5, 4), 1, 2)
        return jac[:, :, :4], jac[:, :, 4:]

    def rollout(self, x0, us, gains=None):
        count = len(ALPHAS)
        x, before = np.tile(x0, (count, 1)), self.line.warning.copy()
        self.line_failed.fill(False)
        xs = np.empty((task.HORIZON + 1, count, 4))
        actions, total = np.empty((task.HORIZON, count, 1)), np.zeros(count)
        xs[0] = x
        for t in range(task.HORIZON):
            u = np.tile(us[t], (count, 1))
            if gains is not None:
                nominal, k, K = gains
                u += ALPHAS[:, None] * k[t] + (x - nominal[t]) @ K[t].T
            u = np.clip(u, -1.0, 1.0)
            total += cost(t, x, u)
            x = self.transition(x, u, line=True)
            xs[t + 1], actions[t] = x, u
        total += cost(task.HORIZON, x, np.zeros((count, 1)))
        invalid = self.line_failed | ~np.isfinite(xs).all(axis=(0, 2)) | ~np.isfinite(total)
        invalid |= np.any(self.line.warning != before, axis=(1, 2))
        total[invalid] = np.inf
        return xs, actions, total

    def solve(self, *, seed=0, iterations=300, progress=lambda message: None):
        if iterations < 1:
            raise ValueError("iterations must be positive")
        us = np.clip(0.1 * np.random.default_rng(seed).standard_normal((task.HORIZON, 1)), -1.0, 1.0)
        candidates, _, totals = self.rollout(task.INITIAL_STATE, us)
        xs, total = candidates[:, 0], float(totals[0])
        if not np.isfinite(total):
            raise RuntimeError("Initial cartpole trajectory is invalid")
        mu = 1.0
        derivatives = (*self.linearize(xs, us), *expand(xs, us), -1.0 - us, 1.0 - us)
        for iteration in range(iterations):
            self.check_cancelled()
            accepted, stop = False, False
            sweep = backward(*derivatives, mu=mu)
            if sweep is not None:
                k, K = sweep
                candidates, actions, totals = self.rollout(task.INITIAL_STATE, us, (xs, k, K))
                best = int(np.argmin(totals))
                if totals[best] < total:
                    drop = total - float(totals[best])
                    xs, us, total = candidates[:, best], actions[:, best], float(totals[best])
                    mu = max(mu / 10.0, 1e-6)
                    accepted, stop = True, drop < 1e-4 * max(total, 1e-12)
            if not accepted:
                mu *= 10.0
                stop = mu > 1e6
            progress(f"iLQR {iteration + 1}/{iterations}: cost={total:.4f}, regularization={mu:.0e}")
            if accepted:
                derivatives = (*self.linearize(xs, us), *expand(xs, us), -1.0 - us, 1.0 - us)
            if stop:
                break
        final = backward(*derivatives, mu=mu)
        # The accepted trajectory can still have negative state curvature.
        # Recompute its feedback with sufficient regularization, rather than
        # using gains around the preceding trajectory or an invalid factorization.
        while final is None and mu <= 1e6:
            mu *= 10.0
            final = backward(*derivatives, mu=mu)
        top = np.zeros((task.HORIZON + 1, 4)), np.zeros((task.HORIZON, 1))
        hold = backward(*self.linearize(*top), *expand(*top),
                        np.full((task.HORIZON, 1), -1.0), np.full((task.HORIZON, 1), 1.0), mu=0.0)
        if final is None or hold is None:
            raise RuntimeError("Cartpole feedback gains could not be computed")
        plan = Plan(xs.copy(), us.copy(), final[1], hold[1][0], total, iteration + 1)
        if abs(task.wrap(xs[-1, 1])) >= 0.1 or abs(xs[-1, 0]) >= 0.2:
            raise RuntimeError(f"Cartpole solve did not reach upright: x={xs[-1, 0]:.3f}, tilt={task.wrap(xs[-1, 1]):.3f}")
        if np.max(np.abs(xs[:, 0])) >= 2.4:
            raise RuntimeError("Cartpole plan exceeds the rail limits")
        return plan

    def close(self):
        self.fields.clear()
        self.batch = self.line = None


def solve(artifact, *, workers=0, seed=0, iterations=300, progress=lambda message: None,
          cancelled=lambda: False):
    planner = Planner(artifact, workers=workers, cancelled=cancelled)
    try:
        return planner.solve(seed=seed, iterations=iterations, progress=progress)
    finally:
        planner.close()


def verify_playback(context, joints, plan, *, hold_seconds=10.0):
    task.reset(joints)
    samples = []
    total_ticks = task.HORIZON * task.SUBSTEPS + round(hold_seconds / task.FIXED_DT)
    force = 0.0
    for tick in range(total_ticks):
        if tick % task.SUBSTEPS == 0:
            force = plan.force(tick // task.SUBSTEPS, task.state(joints))
        joints[0].set_effort_target(force)
        context.step_once()
        x = task.state(joints)
        if not np.isfinite(x).all() or abs(x[0]) >= 2.4:
            raise RuntimeError("Cartpole playback became invalid or exceeded the rail")
        samples.append(x)
    samples = np.asarray(samples)
    tip = samples[task.HORIZON * task.SUBSTEPS - 1]
    if abs(task.wrap(tip[1])) >= 0.1 or abs(tip[0]) >= 0.2:
        raise RuntimeError("Gobot playback did not reach the planned upright state")
    tail = samples[-round(2.0 / task.FIXED_DT):]
    if np.max(np.abs(task.wrap(tail[:, 1]))) >= 0.05 or np.max(np.abs(tail[:, 0])) >= 0.05:
        raise RuntimeError("Cartpole failed the final two-second balance check")
    return samples


def main():
    import gobot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="accepted for parity; this entry point is always headless")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(Path(__file__).resolve().parent))
        context.load_scene("res://cartpole.jscn")
        joints = task.configure_context(context)
        artifact = context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
        plan = solve(artifact, workers=args.workers, seed=args.seed, iterations=args.iterations, progress=print)
        context.build_world(gobot.PhysicsBackendType.MuJoCoCpu)
        states = verify_playback(context, joints, plan)
        print(f"done: cost={plan.cost:.4f}, cart={states[-1, 0]:+.6f} m, tilt={task.wrap(states[-1, 1]):+.6f} rad")
    finally:
        context.clear_scene()


if __name__ == "__main__":
    main()
