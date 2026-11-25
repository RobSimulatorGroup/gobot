"""Unified MotionPlanner with C++ IK, BiRRT global seeding, Batch Racing, and time-scaling."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import numpy as np

from .backends import CollisionBackend, make_backend
from .retime import check_trajectory_bounds, retime_trajectory
from .sqp import SQPJobSpec, run_sqp_batch
from .types import (
    BatchIKResult,
    BatchPlanResult,
    IKResult,
    Link,
    MotionPlanRequest,
    PathConstraint,
    PlanDiagnostics,
    PlanningGroup,
    PlanResult,
    PoseConstraint,
    SolverConfig,
)
from .world import PlanningWorld


def _is_pose_sequence(obj) -> bool:
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, np.ndarray)) and len(obj) > 0 and isinstance(obj[0], PoseConstraint)


def _is_goal_set(obj) -> bool:
    if not isinstance(obj, Sequence) or isinstance(obj, (str, bytes, np.ndarray)) or not obj:
        return False
    first = obj[0]
    if _is_pose_sequence(first):
        return True
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)) and len(first) > 0 and not isinstance(first[0], (int, float, np.number)):
        return False
    if isinstance(first, np.ndarray) and first.ndim == 1 and len(obj) >= 1:
        return True
    return False


def _linear_seed(start: np.ndarray, goal: np.ndarray, waypoints: int, fixed: int,
                 joint_anchors: dict[int, np.ndarray]) -> np.ndarray:
    anchors = [(fixed - 1, start), *sorted(joint_anchors.items()), (waypoints - fixed, goal)]
    pieces = []
    for (first, a), (last, b) in zip(anchors, anchors[1:]):
        frac = np.linspace(0.0, 1.0, last - first + 1, dtype=np.float64)[:-1, None]
        pieces.append((1.0 - frac) * a + frac * b)
    interior = np.concatenate(pieces, axis=0)[1:] if pieces else np.empty((0, len(start)), dtype=np.float64)
    return np.concatenate((np.repeat(start[None, :], fixed, axis=0),
                           interior,
                           np.repeat(goal[None, :], fixed, axis=0)), axis=0)


def _expand_path_constraints(world: PlanningWorld, waypoints: int, fixed: int,
                             waypoint_targets, path_constraints: Sequence[PathConstraint]):
    pose_map: dict[int, list] = {}
    joint_map: dict[int, np.ndarray] = {}
    for idx, target in (waypoint_targets or {}).items():
        if not isinstance(idx, int) or isinstance(idx, bool) or not (0 <= idx < waypoints):
            raise ValueError(f"waypoint target index {idx!r} out of range [0, {waypoints})")
        if isinstance(target, PoseConstraint):
            pose_map.setdefault(idx, []).append(world.compile_pose_constraint(target))
        else:
            jt = world.configuration(target)
            if (jt < world.lower_limits).any() or (jt > world.upper_limits).any():
                raise ValueError(f"joint target at waypoint {idx} violates joint limits")
            joint_map[idx] = jt
    for pc in path_constraints:
        compiled = world.compile_pose_constraint(pc.constraint)
        if pc.timesteps is None:
            indices = range(fixed, waypoints - fixed)
        elif isinstance(pc.timesteps, slice):
            indices = range(*pc.timesteps.indices(waypoints))
        else:
            indices = pc.timesteps
        for idx in indices:
            if not (0 <= idx < waypoints):
                raise ValueError(f"path constraint timestep {idx} out of range")
            pose_map.setdefault(int(idx), []).append(compiled)
    return pose_map, joint_map


class MotionPlanner:
    """High-level single and batch motion planner backed by C++ PlanningWorld and Coal/Warp."""

    def __init__(self, world: PlanningWorld, *, backend: str = "coal") -> None:
        self.world = world
        self.backend: CollisionBackend = make_backend(world, backend)

    def solve_ik(
        self,
        targets: Sequence[PoseConstraint],
        *,
        seed: Sequence[float] | np.ndarray | None = None,
        group: str | PlanningGroup | Sequence[tuple[str, str]] | None = None,
        margin: float = 0.002,
        world_margin: float | None = None,
        iterations: int = 40,
        num_seeds: int = 1,
        verify_original: bool = True,
        rng_seed: int = 0,
    ) -> IKResult:
        """Solve inverse kinematics for one or more PoseConstraints in C++ (with optional multi-seed restarts)."""
        if not targets:
            raise ValueError("provide at least one PoseConstraint")
        compiled = [self.world.compile_pose_constraint(t) for t in targets]
        mask = self.world.group_mask(group)
        wm = margin if world_margin is None else world_margin
        base_seed = self.world.initial_positions.copy() if seed is None else self.world.configuration(seed)
        seeds = [base_seed]
        if num_seeds > 1:
            rng = np.random.default_rng(rng_seed)
            lo = np.where(np.isfinite(self.world.lower_limits), self.world.lower_limits, base_seed - np.pi)
            hi = np.where(np.isfinite(self.world.upper_limits), self.world.upper_limits, base_seed + np.pi)
            for _ in range(num_seeds - 1):
                s = base_seed.copy()
                s[mask] = rng.uniform(lo[mask], hi[mask])
                seeds.append(s)
        problems = [(compiled, s) for s in seeds]
        q_out, ok_out, reasons = self.world._native.solve_ik_batch(
            problems, mask, float(margin), float(wm), int(iterations), bool(verify_original),
        )
        valid = np.flatnonzero(ok_out)
        if not len(valid):
            return IKResult(None, reasons[0] if reasons else "IK failed")
        # Pick the feasible solution closest to base_seed.
        dists = np.linalg.norm(q_out[valid] - base_seed[None, :], axis=-1)
        best = valid[int(np.argmin(dists))]
        return IKResult(q_out[best].copy(), None)

    def solve_ik_batch(
        self,
        target_sets: Sequence[Sequence[PoseConstraint]],
        *,
        seeds: Sequence[Sequence[float] | np.ndarray] | np.ndarray | None = None,
        group: str | PlanningGroup | Sequence[tuple[str, str]] | None = None,
        margin: float = 0.002,
        world_margin: float | None = None,
        iterations: int = 40,
        verify_original: bool = True,
    ) -> BatchIKResult:
        """Solve B independent IK problems in parallel in C++."""
        B = len(target_sets)
        if B == 0:
            raise ValueError("provide at least one IK target set")
        mask = self.world.group_mask(group)
        wm = margin if world_margin is None else world_margin
        if seeds is None:
            seed_arr = np.repeat(self.world.initial_positions[None, :], B, axis=0)
        else:
            s_np = np.asarray(seeds, dtype=np.float64)
            seed_arr = np.repeat(self.world.configuration(s_np)[None, :], B, axis=0) if s_np.ndim == 1 else self.world.configurations(s_np)
            if len(seed_arr) != B:
                raise ValueError("seeds batch size must match target_sets")
        problems = [([self.world.compile_pose_constraint(t) for t in ts], seed_arr[i])
                    for i, ts in enumerate(target_sets)]
        q_out, ok_out, reasons = self.world._native.solve_ik_batch(
            problems, mask, float(margin), float(wm), int(iterations), bool(verify_original),
        )
        return BatchIKResult(q_out, ok_out, tuple(reasons))

    def _resolve_endpoints(self, req: MotionPlanRequest, mask: np.ndarray):
        cfg = req.config
        wm = cfg.effective_world_margin
        t0 = perf_counter()
        # Resolve start.
        if _is_pose_sequence(req.start):
            s_ik = self.solve_ik(req.start, seed=req.start_seed, group=req.group,
                                 margin=cfg.margin, world_margin=wm, num_seeds=max(1, cfg.num_seeds))
            if not s_ik.success:
                return None, None, f"start IK failed: {s_ik.reason}", (perf_counter() - t0) * 1000.0
            start_q = s_ik.positions
        else:
            start_q = self.world.configuration(req.start)

        # Resolve goal (single joint config, single PoseConstraint list, or GoalSet).
        goal_candidates: list[tuple[int, np.ndarray]] = []
        if _is_pose_sequence(req.goal):
            g_seed = start_q if req.goal_seed is None else self.world.configuration(req.goal_seed)
            g_ik = self.solve_ik(req.goal, seed=g_seed, group=req.group,
                                 margin=cfg.margin, world_margin=wm, num_seeds=max(1, cfg.num_seeds))
            if not g_ik.success:
                return start_q, None, f"goal IK failed: {g_ik.reason}", (perf_counter() - t0) * 1000.0
            goal_candidates.append((0, g_ik.positions))
        elif isinstance(req.goal, Sequence) and not isinstance(req.goal, (str, bytes, np.ndarray)) and len(req.goal) > 0 and _is_pose_sequence(req.goal[0]):
            # GoalSet of pose target sets -> parallel C++ batch IK!
            g_seed = start_q if req.goal_seed is None else self.world.configuration(req.goal_seed)
            ik_batch = self.solve_ik_batch(req.goal, seeds=g_seed, group=req.group,
                                           margin=cfg.margin, world_margin=wm)
            valid = np.flatnonzero(ik_batch.success)
            if not len(valid):
                return start_q, None, f"all {len(req.goal)} GoalSet candidates failed IK", (perf_counter() - t0) * 1000.0
            # Sort feasible goals by joint-space distance from start.
            dists = np.linalg.norm(ik_batch.positions[valid] - start_q[None, :], axis=-1)
            for idx in valid[np.argsort(dists)]:
                goal_candidates.append((int(idx), ik_batch.positions[int(idx)].copy()))
        elif isinstance(req.goal, Sequence) and not isinstance(req.goal, (str, bytes, np.ndarray)) and len(req.goal) > 0 and not isinstance(req.goal[0], (int, float, np.number)):
            # GoalSet of joint configurations.
            for idx, gq in enumerate(req.goal):
                goal_candidates.append((idx, self.world.configuration(gq)))
            goal_candidates.sort(key=lambda item: float(np.linalg.norm(item[1] - start_q)))
        else:
            goal_candidates.append((0, self.world.configuration(req.goal)))

        ik_ms = (perf_counter() - t0) * 1000.0
        # Ensure inactive joints in goal match start.
        if not mask.all():
            for i, (gid, gq) in enumerate(goal_candidates):
                gq_fixed = gq.copy()
                gq_fixed[~mask] = start_q[~mask]
                goal_candidates[i] = (gid, gq_fixed)
        return start_q, goal_candidates, None, ik_ms

    def _validate_endpoints(self, start_q: np.ndarray, goal_q: np.ndarray, cfg: SolverConfig, fixed: int) -> str | None:
        ends = np.stack((start_q, goal_q), axis=0)
        if (ends < self.world.lower_limits).any() or (ends > self.world.upper_limits).any():
            return "start or goal violates joint limits"
        if cfg.dt is not None:
            moving = cfg.waypoints - 2 * fixed + 1
            vmax = self.world.velocity_limits * cfg.velocity_scale
            if (np.abs(goal_q - start_q) / (moving * cfg.dt) > vmax * (1.0 + 1e-6)).any():
                return "joint velocity limits require a longer duration"
        if self.world.num_pairs > 0:
            margins = self.world.pair_margins(cfg.margin, cfg.effective_world_margin)
            (frames,) = self.world._native.compute_frames(np.ascontiguousarray(ends), False)
            for verify in ((False, True) if self.world.has_mesh else (False,)):
                rows, pids = self.world._native.screen_aabbs(frames, margins, verify, 1)
                if len(rows):
                    hit = self.backend.first_violation(frames, rows, pids, margins, verify)
                    if hit is not None:
                        seg = 0 if int(hit[0]) == 0 else cfg.waypoints - 2
                        return f"endpoint collision on segment {seg}, pair {self.world.collision_pairs[int(hit[1])]}"
        return None

    def _build_seed_jobs(
        self,
        start_q: np.ndarray,
        goal_candidates: list[tuple[int, np.ndarray]],
        req: MotionPlanRequest,
        mask: np.ndarray,
        pose_map: dict[int, list],
        joint_map: dict[int, np.ndarray],
        *,
        force_rrt: bool = False,
    ) -> tuple[list[SQPJobSpec], float]:
        cfg = req.config
        fixed = 2 if cfg.stop_at_endpoints else 1
        dt_vel = cfg.dt
        dt_smooth = cfg.dt if cfg.dt is not None else 0.25
        interior_anchors = {k: v for k, v in joint_map.items() if fixed <= k < cfg.waypoints - fixed}
        t_seed = perf_counter()
        specs: list[SQPJobSpec] = []

        def make_spec(init_q: np.ndarray, strategy: str, goal_idx: int) -> SQPJobSpec:
            return SQPJobSpec(
                initial=init_q, fixed=fixed, dt_vel=dt_vel, dt_smooth=dt_smooth,
                active_mask=mask, pose_constraints=pose_map, joint_targets=joint_map,
                margin=cfg.margin, world_margin=cfg.effective_world_margin,
                collision_substeps=cfg.collision_substeps, max_segment_length=cfg.max_segment_length,
                iterations=cfg.iterations, first_feasible=cfg.first_feasible,
                velocity_scale=cfg.velocity_scale, seed_strategy=strategy, goal_index=goal_idx,
            )

        if req.seed is not None and not force_rrt:
            seed = np.asarray(req.seed, dtype=np.float64)
            if seed.shape != (cfg.waypoints, self.world.num_joints) or not np.isfinite(seed).all():
                raise ValueError(f"seed must have shape {(cfg.waypoints, self.world.num_joints)}")
            q0 = seed.copy()
            q0[:fixed] = start_q
            q0[-fixed:] = goal_candidates[0][1]
            q0[fixed:-fixed] = np.clip(q0[fixed:-fixed], self.world.lower_limits, self.world.upper_limits)
            if not mask.all():
                q0[:, ~mask] = start_q[~mask]
            specs.append(make_spec(q0, "user_seed", goal_candidates[0][0]))
            return specs, (perf_counter() - t_seed) * 1000.0

        # Add straight-line seeds for top goal candidates.
        if not force_rrt:
            for gid, gq in goal_candidates[:max(1, cfg.num_seeds)]:
                q0 = _linear_seed(start_q, gq, cfg.waypoints, fixed, interior_anchors)
                specs.append(make_spec(q0, "straight", gid))

        # Generate C++ BiRRT global seeds when requested or in fallback mode.
        need_rrt = force_rrt or (cfg.use_global_seeding and cfg.num_seeds > 1)
        if need_rrt and not interior_anchors:
            n_rrt = max(1, cfg.num_seeds) if force_rrt else max(1, cfg.num_seeds - len(specs))
            rrt_starts = np.repeat(start_q[None, :], n_rrt, axis=0)
            rrt_goals = np.stack([goal_candidates[i % len(goal_candidates)][1] for i in range(n_rrt)], axis=0)
            rrt_gids = [goal_candidates[i % len(goal_candidates)][0] for i in range(n_rrt)]
            rrt_seeds = [1001 + i * 7919 for i in range(n_rrt)]
            inner_wp = cfg.waypoints - 2 * (fixed - 1)
            paths, ok = self.world._native.plan_birrt_batch(
                rrt_starts, rrt_goals, mask, float(cfg.margin), float(cfg.effective_world_margin),
                float(cfg.rrt_step_size), int(cfg.rrt_max_iterations), float(cfg.rrt_timeout_s),
                rrt_seeds, bool(self.world.has_mesh), int(cfg.rrt_shortcut_iters), int(inner_wp),
            )
            for i in range(n_rrt):
                if ok[i]:
                    path = paths[i]
                    if fixed == 2:
                        path = np.concatenate((start_q[None, :], path, rrt_goals[i:i + 1]), axis=0)
                    specs.append(make_spec(path, "birrt", rrt_gids[i]))

        # Fill remaining seed slots with perturbed midpoint seeds if requested.
        if not force_rrt and len(specs) < cfg.num_seeds and not interior_anchors:
            rng = np.random.default_rng(42)
            lo = np.where(np.isfinite(self.world.lower_limits), self.world.lower_limits, start_q - 1.5)
            hi = np.where(np.isfinite(self.world.upper_limits), self.world.upper_limits, start_q + 1.5)
            mid_idx = cfg.waypoints // 2
            while len(specs) < cfg.num_seeds:
                gid, gq = goal_candidates[len(specs) % len(goal_candidates)]
                mid = 0.5 * (start_q + gq)
                mid[mask] = np.clip(mid[mask] + rng.normal(0.0, 0.35, size=int(mask.sum())), lo[mask], hi[mask])
                q0 = _linear_seed(start_q, gq, cfg.waypoints, fixed, {mid_idx: mid})
                specs.append(make_spec(q0, "perturbed", gid))

        return specs, (perf_counter() - t_seed) * 1000.0

    def _finalize_trajectory(self, positions: np.ndarray, cfg: SolverConfig) -> tuple[object | None, float, str | None]:
        t0 = perf_counter()
        vmax = self.world.velocity_limits * cfg.velocity_scale
        amax = self.world.acceleration_limits * cfg.acceleration_scale
        dt_arg = cfg.dt if cfg.dt is not None else (None if cfg.retime else 0.1)

        T = len(positions)
        substeps = max(int(cfg.collision_substeps), 4)
        if cfg.max_segment_length is not None:
            longest = float(np.linalg.norm(np.diff(positions, axis=0), axis=-1).max())
            substeps = max(substeps, int(np.ceil(longest / cfg.max_segment_length)))
        u = np.arange(substeps, dtype=np.float64) / substeps
        margins = self.world.pair_margins(cfg.margin, cfg.effective_world_margin) if self.world.num_pairs > 0 else None

        knot_scales = np.ones(T, dtype=np.float64)
        last_error: str | None = None
        for attempt in range(8):
            traj = retime_trajectory(
                positions, vmax, amax,
                stop_at_endpoints=cfg.stop_at_endpoints,
                dt=dt_arg,
                knot_velocity_scales=knot_scales,
            )
            # 1. Exact continuous cubic Hermite velocity & acceleration limit verification.
            ok_bounds, bound_err, bound_bad_segs = check_trajectory_bounds(
                traj, vmax, amax, check_velocity=True, check_acceleration=True,
            )
            if not ok_bounds:
                last_error = bound_err
                # For fixed dt, damping knot velocities toward zero generally increases peak
                # acceleration (up to 6*|dq|/dt^2), so reject immediately once infeasible.
                if dt_arg is not None:
                    return None, (perf_counter() - t0) * 1000.0, last_error

            # 2. Sample the actual continuous cubic Hermite curve per segment and verify limits + clearance.
            t_seg0 = traj.times[:-1, None]
            t_seg1 = traj.times[1:, None]
            ts = np.concatenate(((t_seg0 + u[None, :] * (t_seg1 - t_seg0)).reshape(-1), traj.times[-1:]))
            samples, _, _ = traj.sample(ts)

            bad_segs: set[int] = set(bound_bad_segs)
            lim_bad = (
                (samples < self.world.lower_limits - 1e-6) | (samples > self.world.upper_limits + 1e-6)
            ).any(axis=-1)
            if lim_bad.any():
                for idx in np.flatnonzero(lim_bad):
                    bad_segs.add(min(int(idx) // substeps, T - 2))
                last_error = "retimed Hermite curve violates joint limits"

            if not bad_segs and margins is not None:
                (frames,) = self.world._native.compute_frames(np.ascontiguousarray(samples), False)
                for verify in ((False, True) if self.world.has_mesh else (False,)):
                    rows, pids = self.world._native.screen_aabbs(frames, margins, verify, substeps)
                    if len(rows):
                        hit = self.backend.first_violation(frames, rows, pids, margins, verify)
                        if hit is not None:
                            kind = "mesh" if verify else "convex"
                            seg0 = min(int(hit[0]) // substeps, T - 2)
                            last_error = (
                                f"retimed Hermite {kind} clearance failed on segment {seg0}, "
                                f"pair {self.world.collision_pairs[int(hit[1])]}"
                            )
                            # Identify all violating segments in one pass so multi-corner paths
                            # converge in at most 2 damping rounds instead of one segment per round.
                            dists = self.world._native.query_distances_indexed(frames, rows, pids, verify)[:, 0]
                            viol = dists < np.maximum(0.0, margins[pids] - 1e-6)
                            for r_idx in rows[viol]:
                                bad_segs.add(min(int(r_idx) // substeps, T - 2))
                            bad_segs.add(seg0)
                            break

            if not bad_segs:
                return traj, (perf_counter() - t0) * 1000.0, None

            # Damp knot velocities around offending segments; at zero knot velocity the cubic
            # Hermite segment collapses geometrically onto the verified straight segment p_i -> p_{i+1}.
            changed = False
            for seg in bad_segs:
                knots = (seg, seg + 1)
                if dt_arg is not None and not cfg.stop_at_endpoints:
                    # Endpoint velocities on fixed-dt open endpoints are already collinear with
                    # their adjacent segment; avoid zeroing them needlessly.
                    knots = tuple(k for k in knots if 0 < k < T - 1) or knots
                for knot in knots:
                    old = knot_scales[knot]
                    if old > 0.0:
                        knot_scales[knot] = 0.0 if (attempt >= 2 or old <= 0.25) else old * 0.5
                        changed = True
            if not changed:
                if dt_arg is None and np.any(knot_scales > 0.0):
                    knot_scales[:] = 0.0
                else:
                    return None, (perf_counter() - t0) * 1000.0, last_error

        return None, (perf_counter() - t0) * 1000.0, last_error

    def plan(self, request: MotionPlanRequest) -> PlanResult:
        """Plan a trajectory for a single declarative MotionPlanRequest."""
        t_start = perf_counter()
        cfg = request.config
        fixed = 2 if cfg.stop_at_endpoints else 1
        if cfg.waypoints < 2 * fixed:
            raise ValueError(f"waypoints must be >= {2 * fixed}")
        mask = self.world.group_mask(request.group)
        pose_map, joint_map = _expand_path_constraints(
            self.world, cfg.waypoints, fixed, request.waypoint_targets, request.path_constraints,
        )
        start_q, goal_candidates, ep_err, ik_ms = self._resolve_endpoints(request, mask)
        if ep_err is not None or goal_candidates is None:
            return PlanResult(
                None, None, 0, ep_err,
                PlanDiagnostics(total_ms=(perf_counter() - t_start) * 1000.0, ik_ms=ik_ms),
            )

        # Keep only valid endpoints.
        valid_goals = []
        last_ep_err = None
        for gid, gq in goal_candidates:
            err = self._validate_endpoints(start_q, gq, cfg, fixed)
            if err is None:
                valid_goals.append((gid, gq))
            else:
                last_ep_err = err
        if not valid_goals:
            return PlanResult(
                None, None, 0, last_ep_err,
                PlanDiagnostics(total_ms=(perf_counter() - t_start) * 1000.0, ik_ms=ik_ms),
            )

        specs, seed_ms = self._build_seed_jobs(start_q, valid_goals, request, mask, pose_map, joint_map)
        raw_results = run_sqp_batch(self.world, self.backend, specs, race=(len(specs) > 1 and cfg.first_feasible))
        succeeded = [r for r in raw_results if r is not None and r.positions is not None]

        # Global BiRRT fallback when local single-seed SQP gets stuck in a local minimum.
        if not succeeded and cfg.use_global_seeding and cfg.num_seeds == 1 and request.seed is None:
            rrt_specs, rrt_seed_ms = self._build_seed_jobs(
                start_q, valid_goals, request, mask, pose_map, joint_map, force_rrt=True,
            )
            seed_ms += rrt_seed_ms
            if rrt_specs:
                rrt_results = run_sqp_batch(self.world, self.backend, rrt_specs, race=cfg.first_feasible)
                succeeded = [r for r in rrt_results if r is not None and r.positions is not None]
                if succeeded:
                    raw_results = rrt_results

        if not succeeded:
            first = next((r for r in raw_results if r is not None), None)
            diag = PlanDiagnostics(
                total_ms=(perf_counter() - t_start) * 1000.0, ik_ms=ik_ms, seed_ms=seed_ms,
                sqp_ms=0.0 if first is None else first.sqp_ms,
                validate_ms=0.0 if first is None else first.validate_ms,
                qp_calls=0 if first is None else first.qp_calls,
                sqp_iterations=0 if first is None else first.sqp_iterations,
                seed_strategy="none" if first is None else first.seed_strategy,
                witnesses=None if first is None else first.witnesses,
            )
            return PlanResult(
                None, None if first is None else first.initial_positions, 0,
                "trajectory optimization failed" if first is None else first.reason, diag,
            )

        winner = min(succeeded, key=lambda r: r.cost)
        traj, retime_ms, retime_err = self._finalize_trajectory(winner.positions, cfg)
        diag = PlanDiagnostics(
            total_ms=(perf_counter() - t_start) * 1000.0,
            ik_ms=ik_ms, seed_ms=seed_ms, sqp_ms=winner.sqp_ms,
            validate_ms=winner.validate_ms, retime_ms=retime_ms,
            qp_calls=winner.qp_calls, sqp_iterations=winner.sqp_iterations,
            seed_strategy=winner.seed_strategy, witnesses=winner.witnesses,
        )
        if traj is None:
            return PlanResult(None, winner.initial_positions, winner.goal_index, retime_err, diag)
        return PlanResult(traj, winner.initial_positions, winner.goal_index, None, diag)

    def plan_batch(self, requests: Sequence[MotionPlanRequest], *, chunk_size: int = 8) -> BatchPlanResult:
        """Plan B independent MotionPlanRequests with merged collision batches and parallel OSQP."""
        if not requests:
            raise ValueError("provide at least one MotionPlanRequest")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        B = len(requests)
        waypoints = requests[0].config.waypoints
        all_results: list[PlanResult | None] = [None] * B

        for offset in range(0, B, chunk_size):
            chunk = requests[offset:min(offset + chunk_size, B)]
            active_indices: list[int] = []
            active_specs: list[SQPJobSpec] = []
            active_ik_ms: list[float] = []
            active_seed_ms: list[float] = []
            chunk_t0 = perf_counter()

            for local_i, req in enumerate(chunk):
                row_i = offset + local_i
                cfg = req.config
                fixed = 2 if cfg.stop_at_endpoints else 1
                mask = self.world.group_mask(req.group)
                pose_map, joint_map = _expand_path_constraints(
                    self.world, cfg.waypoints, fixed, req.waypoint_targets, req.path_constraints,
                )
                start_q, goal_candidates, ep_err, ik_ms = self._resolve_endpoints(req, mask)
                if ep_err is not None or goal_candidates is None:
                    all_results[row_i] = PlanResult(None, None, 0, ep_err, PlanDiagnostics(ik_ms=ik_ms))
                    continue
                valid_goals = [(gid, gq) for gid, gq in goal_candidates
                               if self._validate_endpoints(start_q, gq, cfg, fixed) is None]
                if not valid_goals:
                    err = self._validate_endpoints(start_q, goal_candidates[0][1], cfg, fixed)
                    all_results[row_i] = PlanResult(None, None, 0, err, PlanDiagnostics(ik_ms=ik_ms))
                    continue
                specs, seed_ms = self._build_seed_jobs(start_q, valid_goals[:1], req, mask, pose_map, joint_map)
                active_indices.append(row_i)
                active_specs.append(specs[0])
                active_ik_ms.append(ik_ms)
                active_seed_ms.append(seed_ms)

            if active_specs:
                sqp_out = run_sqp_batch(self.world, self.backend, active_specs, race=False)
                # Optional BiRRT fallback for failed rows in the chunk.
                for idx_in_chunk, (row_i, sqp_res, ik_ms, seed_ms) in enumerate(
                    zip(active_indices, sqp_out, active_ik_ms, active_seed_ms)
                ):
                    req = requests[row_i]
                    cfg = req.config
                    if (sqp_res is None or sqp_res.positions is None) and cfg.use_global_seeding and req.seed is None:
                        all_results[row_i] = self.plan(req)
                        continue
                    if sqp_res is None or sqp_res.positions is None:
                        all_results[row_i] = PlanResult(
                            None, None if sqp_res is None else sqp_res.initial_positions, 0,
                            "trajectory optimization failed" if sqp_res is None else sqp_res.reason,
                            PlanDiagnostics(
                                total_ms=(perf_counter() - chunk_t0) * 1000.0, ik_ms=ik_ms, seed_ms=seed_ms,
                                sqp_ms=0.0 if sqp_res is None else sqp_res.sqp_ms,
                                validate_ms=0.0 if sqp_res is None else sqp_res.validate_ms,
                                qp_calls=0 if sqp_res is None else sqp_res.qp_calls,
                                sqp_iterations=0 if sqp_res is None else sqp_res.sqp_iterations,
                                witnesses=None if sqp_res is None else sqp_res.witnesses,
                            ),
                        )
                    else:
                        traj, retime_ms, retime_err = self._finalize_trajectory(sqp_res.positions, cfg)
                        all_results[row_i] = PlanResult(
                            traj, sqp_res.initial_positions, sqp_res.goal_index, retime_err,
                            PlanDiagnostics(
                                total_ms=(perf_counter() - chunk_t0) * 1000.0, ik_ms=ik_ms, seed_ms=seed_ms,
                                sqp_ms=sqp_res.sqp_ms, validate_ms=sqp_res.validate_ms, retime_ms=retime_ms,
                                qp_calls=sqp_res.qp_calls, sqp_iterations=sqp_res.sqp_iterations,
                                seed_strategy=sqp_res.seed_strategy, witnesses=sqp_res.witnesses,
                            ),
                        )

        final_results = tuple(r for r in all_results if r is not None)
        J = self.world.num_joints
        empty = np.full((waypoints, J), np.nan, dtype=np.float64)
        positions = np.stack([r.positions if r.success and r.positions.shape[0] == waypoints else empty for r in final_results])
        init_pos = np.stack([r.initial_positions if r.initial_positions is not None and r.initial_positions.shape[0] == waypoints else empty
                             for r in final_results])
        success = np.asarray([r.success for r in final_results], dtype=np.bool_)
        reasons = tuple(r.reason for r in final_results)
        return BatchPlanResult(final_results, positions, success, reasons, init_pos)

    def debug_draw(
        self,
        debug_draw,
        result: PlanResult,
        *,
        link: Link | None = None,
        samples: int = 64,
        draw_witnesses: bool = True,
        max_witnesses: int = 6,
        seed_color: tuple[float, float, float, float] = (1.0, 0.35, 0.67, 1.0),
        plan_color: tuple[float, float, float, float] = (0.10, 0.95, 0.90, 1.0),
        witness_color: tuple[float, float, float, float] = (1.0, 0.25, 0.15, 1.0),
    ) -> list:
        """Render seed path, smooth optimized trajectory, and closest collision witnesses.

        Works with both ``gobot.render`` (``DebugArrow`` + ``set_debug_arrows``) and any
        debug-draw object exposing ``draw_line(start, end, color)``.
        """
        segments: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], bool]] = []
        if link is not None and result.initial_positions is not None:
            pts, _ = self.world.link_pose(result.initial_positions, link)
            for a, b in zip(pts[:-1], pts[1:]):
                if np.any(a != b):
                    segments.append((a, b, seed_color, False))
        if link is not None and result.trajectory is not None:
            traj = result.trajectory
            if samples > traj.num_waypoints and traj.duration > 0.0:
                ts = np.linspace(traj.times[0], traj.times[-1], samples)
                q_vis, _, _ = traj.sample(ts)
            else:
                q_vis = traj.positions
            pts, _ = self.world.link_pose(q_vis, link)
            for a, b in zip(pts[:-1], pts[1:]):
                if np.any(a != b):
                    segments.append((a, b, plan_color, False))
        if draw_witnesses and max_witnesses > 0 and result.diagnostics.witnesses is not None:
            rows = np.asarray(result.diagnostics.witnesses, dtype=np.float64)
            if rows.ndim == 2 and len(rows) > 0:
                dists = np.linalg.norm(rows[:, 3:6] - rows[:, :3], axis=-1)
                order = np.argsort(dists)
                picked: list[np.ndarray] = []
                for idx in order:
                    if dists[idx] <= 1e-6:
                        continue
                    mid = 0.5 * (rows[idx, :3] + rows[idx, 3:6])
                    if any(np.linalg.norm(mid - p) < 0.06 for p in picked):
                        continue
                    picked.append(mid)
                    segments.append((rows[idx, :3], rows[idx, 3:6], witness_color, False))
                    if len(picked) >= max_witnesses:
                        break

        if hasattr(debug_draw, "set_debug_arrows") and hasattr(debug_draw, "DebugArrow"):
            arrows = [
                debug_draw.DebugArrow(a.tolist(), (b - a).tolist(), color=color, show_head=head)
                for a, b, color, head in segments
            ]
            debug_draw.set_debug_arrows(arrows)
            return arrows

        for a, b, color, _ in segments:
            debug_draw.draw_line(a.tolist(), b.tolist(), color)
        return segments
