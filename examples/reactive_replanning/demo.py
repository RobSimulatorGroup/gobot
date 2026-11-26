"""Closed-loop 20 Hz reactive replanning demo for 7-DOF Franka Panda with a moving obstacle.

Evaluates scheduled switch-time continuity in a discrete-time controller simulation (q, dq[, ddq]), world_version snapshot
consistency before commit, strict_ccd verification, and per-path (reused_prev / warm_start /
global_fallback) plus full-tick (perception + replan + handoff) latency distributions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

import gobot
from gobot.planning import (
    MotionPlanner,
    MotionPlanRequest,
    PlanningWorld,
    PoseConstraint,
    SolverConfig,
    Trajectory,
)


def _load_panda_world(mode: str) -> tuple[PlanningWorld, np.ndarray, np.ndarray, np.ndarray]:
    project = Path(__file__).resolve().parents[1] / "panda_trajopt"
    ctx = gobot.app.context()
    prev = ctx.project_path
    try:
        ctx.set_project_path(str(project))
        ctx.load_scene("res://world.jscn")
        snap = ctx.compile_physics_scene_snapshot()
        root = ctx.root
        start_xyz = np.asarray(root.find("Start").position, dtype=np.float64)
        goal_xyz = np.asarray(root.find("Goal").position, dtype=np.float64)
    finally:
        ctx.clear_scene()
        if prev:
            ctx.set_project_path(prev)

    mounts = (("panda_link0", "panda_link2"), ("panda_link4", "panda_link6"), ("panda_link7", "panda_hand"))
    world = PlanningWorld.from_snapshot(
        snap,
        {"panda": "panda_link0"},
        ignored_link_pairs=[(("panda", a), ("panda", b)) for a, b in mounts],
    )
    # Remove static central pillar so we can drive a dynamic moving obstacle cleanly.
    for name in list(world._native.shape_names()):
        if "pillar" in name.lower() or "obstacle" in name.lower():
            world.remove_obstacle(name)
    if mode == "esdf":
        world.set_collision_enabled(("panda", "panda_link0"), "esdf", False)
    return world, start_xyz, goal_xyz, np.diag([-1.0, 1.0, -1.0])


def _obstacle_cloud(center: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    xs = np.linspace(center[0] - half_extents[0], center[0] + half_extents[0], 4)
    ys = np.linspace(center[1] - half_extents[1], center[1] + half_extents[1], 4)
    zs = np.linspace(center[2] - half_extents[2], center[2] + half_extents[2], 12)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)


def _pct(vals: list[float], p: float) -> float:
    return float(np.percentile(vals, p)) if vals else 0.0


def _execution_interval(traj: Trajectory, start: float, end: float) -> Trajectory:
    """Restrict the command spline to this tick, retaining its interior knots and derivatives."""
    inside = traj.times[(traj.times > start) & (traj.times < end)]
    times = np.r_[start, inside, end]
    q, dq, ddq = traj.sample(times)
    return Trajectory(times - start, q, dq, ddq, spline_order=traj.spline_order)


def run_demo(
    *,
    mode: str = "coal",
    hz: float = 20.0,
    spline_order: int = 3,
    strict_ccd: bool = True,
    max_ticks: int = 120,
    verbose: bool = True,
) -> dict:
    world, start_xyz, goal_xyz, down = _load_panda_world(mode)
    planner = MotionPlanner(world, backend="coal")
    hand = ("panda", "panda_hand")
    seed = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, -2.3], dtype=np.float64)

    world_margin = 0.015 if mode == "esdf" else 0.01
    ik_s = planner.solve_ik(
        [PoseConstraint(hand, start_xyz, down, position_tolerance=1e-4, rotation_tolerance=1e-2)],
        seed=seed, margin=0.002, world_margin=world_margin,
    )
    ik_g = planner.solve_ik(
        [PoseConstraint(hand, goal_xyz, down, position_tolerance=1e-4, rotation_tolerance=1e-2)],
        seed=seed, margin=0.002, world_margin=world_margin,
    )
    if not (ik_s.success and ik_g.success):
        raise RuntimeError(f"Initial IK failed: start={ik_s.reason}, goal={ik_g.reason}")
    q_start = ik_s.positions
    q_goal = ik_g.positions

    mid_xy = 0.5 * (start_xyz[:2] + goal_xyz[:2])
    obs_half = np.array([0.04, 0.04, 0.22], dtype=np.float64)
    obs_z = 0.42

    def obstacle_center(t_sim: float) -> np.ndarray:
        # Start clear of the initial trajectory, then smoothly sweep into the central corridor
        # so the moving arm must reactively replan around it without teleporting into link margins.
        phase = np.clip((t_sim - 0.15) / 0.70, 0.0, 1.0)
        smooth = phase * phase * (3.0 - 2.0 * phase)
        offset_x = 0.28 * (1.0 - smooth)
        offset_y = 0.24 * (1.0 - smooth)
        return np.array([mid_xy[0] + offset_x, mid_xy[1] + offset_y, obs_z], dtype=np.float64)

    def update_perception(center: np.ndarray) -> float:
        t0 = perf_counter()
        if mode == "coal":
            if "moving_pillar" not in world._native.shape_names():
                world.add_box("moving_pillar", tuple(2.0 * obs_half), center)
            else:
                world.update_obstacle_pose("moving_pillar", center)
        else:
            cloud = _obstacle_cloud(center, obs_half)
            world.set_point_cloud(
                cloud,
                resolution=0.03,
                origin=(mid_xy[0] - 0.55, mid_xy[1] - 0.55, 0.05),
                shape=(38, 38, 28),
                point_radius=0.005,
            )
        return (perf_counter() - t0) * 1000.0

    update_perception(obstacle_center(0.0))

    dt_tick = 1.0 / hz
    budget_s = 0.85 * dt_tick  # leave headroom inside the 50 ms (20 Hz) control period
    cfg = SolverConfig(
        waypoints=14,
        margin=0.002,
        world_margin=world_margin,
        collision_substeps=6,
        iterations=30,
        stop_at_endpoints=True,
        first_feasible=True,
        use_global_seeding=True,
        num_seeds=1,
        rrt_timeout_s=0.03,
        velocity_scale=0.65,
        acceleration_scale=0.65,
        jerk_scale=0.65,
        spline_order=spline_order,
        strict_ccd=strict_ccd,
    )

    init_res = planner.plan(MotionPlanRequest(start=q_start, goal=q_goal, config=cfg))
    if not init_res.success or init_res.trajectory is None:
        raise RuntimeError(f"Initial plan failed: {init_res.reason}")

    active_traj: Trajectory = init_res.trajectory
    t_on_active = 0.0
    t_sim = 0.0

    path_times: dict[str, list[float]] = defaultdict(list)
    tick_totals_ms: list[float] = []
    perception_ms_list: list[float] = []
    replan_ms_list: list[float] = []
    n_ticks = 0
    n_success = 0
    n_timeout = 0
    n_overrun = 0
    n_stale_revalidated = 0
    max_pos_jump = 0.0
    max_vel_jump = 0.0
    max_acc_jump = 0.0
    reached_goal = False

    for _ in range(max_ticks):
        n_ticks += 1
        t_tick0 = perf_counter()

        # 1. Perception update for current control tick.
        obs_pos = obstacle_center(t_sim)
        p_ms = update_perception(obs_pos)
        perception_ms_list.append(p_ms)

        # While planning, the controller follows the old spline to the next scheduled switch.
        t_switch = min(t_on_active + dt_tick, float(active_traj.times[-1]))
        prefix = _execution_interval(active_traj, t_on_active, t_switch)
        ok_prefix, _ = world.validate_trajectory(
            prefix, margin=cfg.margin, world_margin=cfg.effective_world_margin,
            velocity_scale=cfg.velocity_scale, acceleration_scale=cfg.acceleration_scale,
            jerk_scale=cfg.jerk_scale, strict_ccd=cfg.strict_ccd,
        )
        if not ok_prefix:
            tick_totals_ms.append((perf_counter() - t_tick0) * 1000.0)
            n_overrun += int(tick_totals_ms[-1] > dt_tick * 1000.0)
            break
        q_cmd, dq_cmd, ddq_cmd = active_traj.sample(t_switch)
        t_r0 = perf_counter()
        res = planner.replan(
            q_cmd,
            dq_cmd,
            prev_traj=active_traj,
            goal=q_goal,
            ddq_now=ddq_cmd if spline_order == 5 else None,
            t_on_prev=t_switch,
            reuse_if_valid=True,
            time_budget_s=budget_s,
            config=cfg,
        )
        r_ms = (perf_counter() - t_r0) * 1000.0
        replan_ms_list.append(r_ms)

        # 4. Handoff validation: verify world_version consistency before committing.
        committed = False
        if res.success and res.trajectory is not None:
            if res.world_version != world.version:
                n_stale_revalidated += 1
                checked_version = world.version
                ok_reval, _ = world.validate_trajectory(
                    res.trajectory,
                    margin=cfg.margin,
                    world_margin=cfg.effective_world_margin,
                    strict_ccd=cfg.strict_ccd, velocity_scale=cfg.velocity_scale,
                    acceleration_scale=cfg.acceleration_scale, jerk_scale=cfg.jerk_scale,
                )
                ok_reval = ok_reval and checked_version == world.version
            else:
                ok_reval = True

            if ok_reval:
                q0_new, dq0_new, ddq0_new = res.trajectory.sample(0.0)
                # Evaluate the running command at handoff. Discard a result that missed the switch.
                q_switch, dq_switch, ddq_switch = prefix.sample(prefix.duration)
                if perf_counter() - t_tick0 <= dt_tick:
                    max_pos_jump = max(max_pos_jump, float(np.max(np.abs(q0_new - q_switch))))
                    max_vel_jump = max(max_vel_jump, float(np.max(np.abs(dq0_new - dq_switch))))
                    if spline_order == 5:
                        max_acc_jump = max(max_acc_jump, float(np.max(np.abs(ddq0_new - ddq_switch))))
                    active_traj = res.trajectory
                    t_on_active = 0.0
                    committed = True
                    n_success += 1
                    strat = res.diagnostics.seed_strategy
                    bucket = (
                        "reused_prev"
                        if strat == "reused_prev"
                        else ("warm_start" if strat == "warm_start" else "global_fallback")
                    )
                    path_times[bucket].append(r_ms)
        else:
            if res.diagnostics.timed_out:
                n_timeout += 1

        full_tick_ms = (perf_counter() - t_tick0) * 1000.0
        tick_totals_ms.append(full_tick_ms)
        if full_tick_ms > dt_tick * 1000.0:
            n_overrun += 1

        if not committed:
            t_on_active = t_switch
        t_sim += dt_tick

        q_after, dq_after, _ = active_traj.sample(t_on_active)
        if np.linalg.norm(q_after - q_goal) < 1e-3 and np.linalg.norm(dq_after) < 1e-3:
            reached_goal = True
            break

    summary = {
        "mode": mode,
        "hz": hz,
        "spline_order": spline_order,
        "strict_ccd": strict_ccd,
        "ticks": n_ticks,
        "reached_goal": reached_goal,
        "success_rate": n_success / max(1, n_ticks),
        "timeout_rate": n_timeout / max(1, n_ticks),
        "overrun_rate": n_overrun / max(1, n_ticks),
        "stale_revalidated": n_stale_revalidated,
        "max_pos_jump": max_pos_jump,
        "max_vel_jump": max_vel_jump,
        "max_acc_jump": max_acc_jump,
        "perception_p50_ms": _pct(perception_ms_list, 50),
        "perception_p95_ms": _pct(perception_ms_list, 95),
        "full_tick_p50_ms": _pct(tick_totals_ms, 50),
        "full_tick_p95_ms": _pct(tick_totals_ms, 95),
        "paths": {
            k: {"count": len(v), "p50_ms": _pct(v, 50), "p95_ms": _pct(v, 95)}
            for k, v in path_times.items()
        },
    }
    if verbose:
        print(f"=== Reactive Replanning Demo ({mode.upper()}, {hz:.0f} Hz, order={spline_order}, strict_ccd={strict_ccd}) ===")
        print(
            f"Ticks: {n_ticks} | Reached goal: {reached_goal} | "
            f"Success: {summary['success_rate']*100:.1f}% | "
            f"Budget timeout: {summary['timeout_rate']*100:.1f}% | "
            f"Tick overrun (>{dt_tick*1000:.1f} ms): {summary['overrun_rate']*100:.1f}%"
        )
        print(
            f"Switch continuity: max |dq_pos|={max_pos_jump:.2e} rad, "
            f"max |dq_vel|={max_vel_jump:.2e} rad/s"
            + (f", max |dq_acc|={max_acc_jump:.2e} rad/s^2" if spline_order == 5 else "")
        )
        print(
            f"Perception update: P50={summary['perception_p50_ms']:.2f} ms, "
            f"P95={summary['perception_p95_ms']:.2f} ms | "
            f"Full tick (perception + replan + handoff): "
            f"P50={summary['full_tick_p50_ms']:.2f} ms, P95={summary['full_tick_p95_ms']:.2f} ms"
        )
        for name in ("reused_prev", "warm_start", "global_fallback"):
            st = summary["paths"].get(name)
            if st:
                print(f"  - {name:15s}: n={st['count']:3d}, P50={st['p50_ms']:.2f} ms, P95={st['p95_ms']:.2f} ms")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop 20 Hz Panda reactive replanning benchmark/demo")
    parser.add_argument("--mode", choices=("coal", "esdf", "both"), default="both")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--spline-order", type=int, choices=(3, 5), default=3)
    parser.add_argument("--no-strict-ccd", action="store_true")
    parser.add_argument("--max-ticks", type=int, default=120)
    args = parser.parse_args()

    modes = ("coal", "esdf") if args.mode == "both" else (args.mode,)
    for m in modes:
        run_demo(
            mode=m,
            hz=args.hz,
            spline_order=args.spline_order,
            strict_ccd=not args.no_strict_ccd,
            max_ticks=args.max_ticks,
            verbose=True,
        )


if __name__ == "__main__":
    main()
