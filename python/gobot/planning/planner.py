"""Thin Python wrapper over C++ gobot_planning MotionPlanner (Coal) and experimental MuJoCo Warp backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter

import numpy as np

from gobot import _coal_native

from .backends import CollisionBackend, make_backend
from .sqp import SQPJobSpec, run_sqp_batch
from .types import (
    BatchIKResult,
    BatchPlanResult,
    CartesianPlanConfig,
    CartesianPlanResult,
    CartesianWaypoint,
    GoalSpec,
    IKResult,
    JointConfig,
    Link,
    MotionPlanRequest,
    PathConstraint,
    PlanDiagnostics,
    PlanningGroup,
    PlanResult,
    PoseConstraint,
    SolverConfig,
    Trajectory,
    _validate_manifold_constraints,
)
from .world import PlanningWorld


def _is_pose_sequence(obj) -> bool:
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, np.ndarray)) and len(obj) > 0 and isinstance(obj[0], PoseConstraint)


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


def _expand_path_constraints(world: PlanningWorld, waypoints: int, fixed_start: int,
                             waypoint_targets, path_constraints: Sequence[PathConstraint],
                             fixed_end: int | None = None):
    if fixed_end is None:
        fixed_end = fixed_start
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
            indices = range(fixed_start, waypoints - fixed_end)
        elif isinstance(pc.timesteps, slice):
            indices = range(*pc.timesteps.indices(waypoints))
        else:
            indices = pc.timesteps
        for idx in indices:
            if not (0 <= idx < waypoints):
                raise ValueError(f"path constraint timestep {idx} out of range")
            pose_map.setdefault(int(idx), []).append(compiled)
    return pose_map, joint_map


def _config_to_native_dict(cfg: SolverConfig) -> dict:
    return {
        "waypoints": int(cfg.waypoints),
        "margin": float(cfg.margin),
        "world_margin": None if cfg.world_margin is None else float(cfg.world_margin),
        "collision_substeps": int(cfg.collision_substeps),
        "max_segment_length": None if cfg.max_segment_length is None else float(cfg.max_segment_length),
        "iterations": int(cfg.iterations),
        "stop_at_endpoints": bool(cfg.stop_at_endpoints),
        "first_feasible": bool(cfg.first_feasible),
        "use_global_seeding": bool(cfg.use_global_seeding),
        "num_seeds": int(cfg.num_seeds),
        "rrt_timeout_s": float(cfg.rrt_timeout_s),
        "rrt_step_size": float(cfg.rrt_step_size),
        "rrt_max_iterations": int(cfg.rrt_max_iterations),
        "rrt_shortcut_iters": int(cfg.rrt_shortcut_iters),
        "velocity_scale": float(cfg.velocity_scale),
        "acceleration_scale": float(cfg.acceleration_scale),
        "jerk_scale": float(cfg.jerk_scale),
        "spline_order": int(cfg.spline_order),
        "strict_ccd": bool(cfg.strict_ccd),
        "dt": None if cfg.dt is None else float(cfg.dt),
        "retime": bool(cfg.retime),
    }


def _trajectory_from_native(d: dict | None) -> Trajectory | None:
    if d is None:
        return None
    return Trajectory(
        times=d["times"],
        positions=d["positions"],
        velocities=d["velocities"],
        accelerations=d["accelerations"],
        dt=d["dt"],
        spline_order=int(d.get("spline_order", 3)),
    )


def _trajectory_to_native_dict(traj: Trajectory) -> dict:
    return {
        "times": np.ascontiguousarray(traj.times, dtype=np.float64),
        "positions": np.ascontiguousarray(traj.positions, dtype=np.float64),
        "velocities": np.ascontiguousarray(traj.velocities, dtype=np.float64),
        "accelerations": np.ascontiguousarray(traj.accelerations, dtype=np.float64),
        "dt": None if traj.dt is None else float(traj.dt),
        "spline_order": int(traj.spline_order),
    }


def _joint_vector(world: PlanningWorld, values: JointConfig, name: str) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if arr.shape != (world.num_joints,):
        raise ValueError(f"{name} must have shape ({world.num_joints},), got {arr.shape}")
    return arr


def _plan_result_from_native(d: dict) -> PlanResult:
    diag_d = d["diagnostics"]
    diag = PlanDiagnostics(
        total_ms=float(diag_d["total_ms"]),
        ik_ms=float(diag_d["ik_ms"]),
        seed_ms=float(diag_d["seed_ms"]),
        sqp_ms=float(diag_d["sqp_ms"]),
        validate_ms=float(diag_d["validate_ms"]),
        retime_ms=float(diag_d["retime_ms"]),
        qp_calls=int(diag_d["qp_calls"]),
        sqp_iterations=int(diag_d["sqp_iterations"]),
        seed_strategy=str(diag_d["seed_strategy"]),
        witnesses=diag_d["witnesses"],
        timed_out=bool(diag_d.get("timed_out", False)),
    )
    return PlanResult(
        trajectory=_trajectory_from_native(d["trajectory"]),
        initial_positions=d["initial_positions"],
        goal_index=int(d["goal_index"]),
        reason=d["reason"],
        diagnostics=diag,
        world_version=int(d.get("world_version", 0)),
        waypoint_indices=d.get("waypoint_indices"),
    )


def _request_to_native_dict(world: PlanningWorld, req: MotionPlanRequest) -> dict:
    cfg = req.config
    has_start_bc = (req.start_velocity is not None) or (req.start_acceleration is not None)
    fixed_start = 1 if has_start_bc else (2 if cfg.stop_at_endpoints else 1)
    fixed_end = 2 if cfg.stop_at_endpoints else 1
    if cfg.waypoints < fixed_start + fixed_end:
        raise ValueError(f"waypoints must be >= {fixed_start + fixed_end}")
    mask = world.group_mask(req.group)
    pose_map, joint_map = _expand_path_constraints(
        world, cfg.waypoints, fixed_start, req.waypoint_targets, req.path_constraints, fixed_end=fixed_end,
    )

    compiled_manifold = [world.compile_pose_constraint(c) for c in req.manifold_constraints]
    out: dict = {
        "config": _config_to_native_dict(cfg),
        "pose_constraints": pose_map,
        "manifold_constraints": compiled_manifold,
        "joint_targets": {k: np.ascontiguousarray(v, dtype=np.float64) for k, v in joint_map.items()},
        "active_mask": np.ascontiguousarray(mask, dtype=np.bool_),
        "start_velocity": None if req.start_velocity is None else _joint_vector(world, req.start_velocity, "start_velocity"),
        "start_acceleration": None if req.start_acceleration is None else _joint_vector(world, req.start_acceleration, "start_acceleration"),
    }
    if _is_pose_sequence(req.start):
        out["start_is_pose"] = True
        out["start_poses"] = [world.compile_pose_constraint(t) for t in req.start]
    else:
        out["start_is_pose"] = False
        out["start_q"] = world.configuration(req.start)
    out["start_seed"] = None if req.start_seed is None else world.configuration(req.start_seed)

    if _is_pose_sequence(req.goal):
        out["goal_is_pose"] = True
        out["goal_pose_sets"] = [[world.compile_pose_constraint(t) for t in req.goal]]
    elif (isinstance(req.goal, Sequence) and not isinstance(req.goal, (str, bytes, np.ndarray))
          and len(req.goal) > 0 and _is_pose_sequence(req.goal[0])):
        out["goal_is_pose"] = True
        out["goal_pose_sets"] = [[world.compile_pose_constraint(t) for t in ts] for ts in req.goal]
    elif (isinstance(req.goal, Sequence) and not isinstance(req.goal, (str, bytes, np.ndarray))
          and len(req.goal) > 0 and not isinstance(req.goal[0], (int, float, np.number))):
        out["goal_is_pose"] = False
        out["goal_qs"] = [world.configuration(gq) for gq in req.goal]
    else:
        out["goal_is_pose"] = False
        out["goal_qs"] = [world.configuration(req.goal)]
    out["goal_seed"] = None if req.goal_seed is None else world.configuration(req.goal_seed)

    if req.seed is not None:
        seed = np.asarray(req.seed, dtype=np.float64)
        if seed.shape != (cfg.waypoints, world.num_joints) or not np.isfinite(seed).all():
            raise ValueError(f"seed must have shape {(cfg.waypoints, world.num_joints)}")
        out["seed"] = np.ascontiguousarray(seed, dtype=np.float64)
    else:
        out["seed"] = None
    return out


class MotionPlanner:
    """High-level single and batch motion planner backed by standalone C++ gobot_planning."""

    def __init__(self, world: PlanningWorld, *, backend: str = "coal") -> None:
        self.world = world
        self.backend: CollisionBackend = make_backend(world, backend)
        self._native_planner = _coal_native.MotionPlanner(world._native)

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

    def _finalize_trajectory(
        self,
        positions: np.ndarray,
        cfg: SolverConfig,
        *,
        snap=None,
        manifold_constraints=None,
        active_mask=None,
    ) -> tuple[Trajectory | None, float, str | None]:
        """Single unified C++ Hermite retiming, continuous CCD & continuous manifold verification path."""
        target = self.world._native if snap is None else snap
        traj_dict, retime_ms, err = target.finalize_trajectory(
            np.ascontiguousarray(positions, dtype=np.float64),
            _config_to_native_dict(cfg),
            manifold_constraints,
            active_mask,
        )
        return _trajectory_from_native(traj_dict), float(retime_ms), err

    def plan(self, request: MotionPlanRequest) -> PlanResult:
        """Plan a trajectory for a single declarative MotionPlanRequest."""
        if self.backend.name == "coal":
            raw = self._native_planner.plan(_request_to_native_dict(self.world, request))
            return _plan_result_from_native(raw)
        return self._plan_warp_single(request)

    def replan(
        self,
        q_now: JointConfig,
        dq_now: JointConfig,
        prev_traj: Trajectory | None = None,
        goal: GoalSpec | None = None,
        *,
        ddq_now: JointConfig | None = None,
        t_on_prev: float | None = None,
        reuse_if_valid: bool = True,
        time_budget_s: float | None = None,
        waypoint_targets: Mapping[int, PoseConstraint | JointConfig] | None = None,
        path_constraints: Sequence[PathConstraint] = (),
        manifold_constraints: Sequence[PoseConstraint] = (),
        goal_seed: JointConfig | None = None,
        group: str | PlanningGroup | Sequence[tuple[str, str]] | None = None,
        config: SolverConfig | None = None,
    ) -> PlanResult:
        """Reactively replan from a live moving state ``(q_now, dq_now[, ddq_now])``.

        Hierarchy on one pinned ``WorldSnapshot``:
        1. Exact Hermite suffix reuse from ``prev_traj`` at ``t_on_prev`` (or unambiguous anchor match)
           when goal, constraints, timing, spline order, and measured state all match and atomic validation passes.
        2. Shifted warm-start SQP seeded from ``prev_traj`` with exact start-boundary enforcement.
        3. Time-budgeted global BiRRT/CBiRRT fallback preserving the exact start boundary.

        When ``goal is None``, the goal defaults to ``prev_traj.positions[-1]``.
        """
        cfg = SolverConfig() if config is None else config
        if ddq_now is not None and cfg.spline_order != 5:
            raise ValueError("ddq_now (start_acceleration) is only supported when config.spline_order == 5")
        if prev_traj is None and goal is None:
            raise ValueError("replan requires either goal or prev_traj")
        if manifold_constraints:
            _validate_manifold_constraints(manifold_constraints)
        fixed_start = 1
        fixed_end = 2 if cfg.stop_at_endpoints else 1
        if cfg.waypoints < fixed_start + fixed_end:
            raise ValueError(f"waypoints must be >= {fixed_start + fixed_end}")

        mask = self.world.group_mask(group)
        pose_map, joint_map = _expand_path_constraints(
            self.world, cfg.waypoints, fixed_start, waypoint_targets, path_constraints, fixed_end=fixed_end,
        )
        compiled_manifold = [self.world.compile_pose_constraint(c) for c in manifold_constraints]

        req_dict: dict = {
            "config": _config_to_native_dict(cfg),
            "q_now": self.world.configuration(q_now),
            "dq_now": _joint_vector(self.world, dq_now, "dq_now"),
            "ddq_now": None if ddq_now is None else _joint_vector(self.world, ddq_now, "ddq_now"),
            "prev_traj": None if prev_traj is None else _trajectory_to_native_dict(prev_traj),
            "t_on_prev": None if t_on_prev is None else float(t_on_prev),
            "reuse_if_valid": bool(reuse_if_valid),
            "time_budget_s": 0.0 if time_budget_s is None else float(time_budget_s),
            "pose_constraints": pose_map,
            "manifold_constraints": compiled_manifold,
            "joint_targets": {k: np.ascontiguousarray(v, dtype=np.float64) for k, v in joint_map.items()},
            "active_mask": np.ascontiguousarray(mask, dtype=np.bool_),
            "goal_seed": None if goal_seed is None else self.world.configuration(goal_seed),
        }
        if goal is None:
            req_dict["has_goal"] = False
        else:
            req_dict["has_goal"] = True
            if _is_pose_sequence(goal):
                req_dict["goal_is_pose"] = True
                req_dict["goal_pose_sets"] = [[self.world.compile_pose_constraint(t) for t in goal]]
            elif (isinstance(goal, Sequence) and not isinstance(goal, (str, bytes, np.ndarray))
                  and len(goal) > 0 and _is_pose_sequence(goal[0])):
                req_dict["goal_is_pose"] = True
                req_dict["goal_pose_sets"] = [[self.world.compile_pose_constraint(t) for t in ts] for ts in goal]
            elif (isinstance(goal, Sequence) and not isinstance(goal, (str, bytes, np.ndarray))
                  and len(goal) > 0 and not isinstance(goal[0], (int, float, np.number))):
                req_dict["goal_is_pose"] = False
                req_dict["goal_qs"] = [self.world.configuration(gq) for gq in goal]
            else:
                req_dict["goal_is_pose"] = False
                req_dict["goal_qs"] = [self.world.configuration(goal)]

        raw = self._native_planner.replan(req_dict)
        return _plan_result_from_native(raw)

    def plan_batch(self, requests: Sequence[MotionPlanRequest], *, chunk_size: int = 64) -> BatchPlanResult:
        """Plan independent requests. Use each successful result's trajectory for playback."""
        if not requests:
            raise ValueError("provide at least one MotionPlanRequest")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        waypoints = requests[0].config.waypoints

        if self.backend.name == "coal":
            native_reqs = [_request_to_native_dict(self.world, req) for req in requests]
            raw_list = self._native_planner.plan_batch(native_reqs, int(chunk_size))
            final_results = tuple(_plan_result_from_native(item) for item in raw_list)
        else:
            final_results = self._plan_warp_batch(requests, chunk_size=chunk_size)

        J = self.world.num_joints
        def stack_knots(rows: Sequence[np.ndarray | None]) -> np.ndarray:
            width = max([waypoints] + [len(row) for row in rows if row is not None])
            packed = np.full((len(rows), width, J), np.nan, dtype=np.float64)
            for i, row in enumerate(rows):
                if row is not None and len(row):
                    packed[i, :len(row)] = row
                    packed[i, len(row):] = row[-1]
            return packed

        positions = stack_knots([r.positions for r in final_results])
        init_pos = stack_knots([r.initial_positions for r in final_results])
        success = np.asarray([r.success for r in final_results], dtype=np.bool_)
        reasons = tuple(r.reason for r in final_results)
        world_ver = final_results[0].world_version if final_results else self.world.version
        return BatchPlanResult(final_results, positions, success, reasons, init_pos, world_version=world_ver)

    def plan_cartesian_path(
        self,
        start: JointConfig,
        waypoints: Sequence[CartesianWaypoint],
        *,
        link: Link,
        tcp_offset_position: Sequence[float] | np.ndarray | None = None,
        tcp_offset_rotation: Sequence[Sequence[float]] | np.ndarray | None = None,
        group: str | PlanningGroup | Sequence[tuple[str, str]] | None = None,
        config: CartesianPlanConfig | None = None,
    ) -> CartesianPlanResult:
        """Plan a verified linear/circular-arc Cartesian TCP path in C++ against an immutable WorldSnapshot."""
        if not waypoints:
            raise ValueError("provide at least one CartesianWaypoint")
        if link not in self.world.links:
            raise KeyError(f"unknown link {link!r}")
        cfg = CartesianPlanConfig() if config is None else config
        start_q = self.world.configuration(start)
        mask = self.world.group_mask(group)

        tcp_pos = None
        if tcp_offset_position is not None:
            tcp_pos = np.asarray(tcp_offset_position, dtype=np.float64)
            if tcp_pos.shape != (3,) or not np.isfinite(tcp_pos).all():
                raise ValueError("tcp_offset_position must be a finite 3-vector")
        tcp_rot = None
        if tcp_offset_rotation is not None:
            tcp_rot = np.asarray(tcp_offset_rotation, dtype=np.float64)
            if tcp_rot.shape != (3, 3) or not np.isfinite(tcp_rot).all():
                raise ValueError("tcp_offset_rotation must be a finite 3x3 matrix")

        native_wps = []
        for wp in waypoints:
            if not isinstance(wp, CartesianWaypoint):
                raise TypeError(f"expected CartesianWaypoint, got {type(wp).__name__}")
            native_wps.append({
                "motion_type": str(wp.motion_type),
                "position": np.asarray(wp.position, dtype=np.float64),
                "rotation": None if wp.rotation is None else np.asarray(wp.rotation, dtype=np.float64),
                "via_position": None if wp.via_position is None else np.asarray(wp.via_position, dtype=np.float64),
                "blend_radius": float(wp.blend_radius),
            })

        req_dict = {
            "start_q": np.ascontiguousarray(start_q, dtype=np.float64),
            "link_id": int(self.world.links[link]),
            "tcp_offset_position": tcp_pos,
            "tcp_offset_rotation": tcp_rot,
            "active_mask": np.ascontiguousarray(mask, dtype=np.bool_),
            "waypoints": native_wps,
            "config": {
                "max_step": float(cfg.max_step),
                "max_rot_step": float(cfg.max_rot_step),
                "max_pos_error": float(cfg.max_pos_error),
                "max_rot_error": float(cfg.max_rot_error),
                "jump_threshold_revolute": float(cfg.jump_threshold_revolute),
                "jump_threshold_prismatic": float(cfg.jump_threshold_prismatic),
                "max_subdivision_depth": int(cfg.max_subdivision_depth),
                "ik_iterations": int(cfg.ik_iterations),
                "ik_damping": float(cfg.ik_damping),
                "avoid_collisions": bool(cfg.avoid_collisions),
                "margin": float(cfg.margin),
                "world_margin": None if cfg.world_margin is None else float(cfg.world_margin),
                "strict_ccd": bool(cfg.strict_ccd),
                "collision_substeps": int(cfg.collision_substeps),
                "spline_order": int(cfg.spline_order),
                "velocity_scale": float(cfg.velocity_scale),
                "acceleration_scale": float(cfg.acceleration_scale),
                "jerk_scale": float(cfg.jerk_scale),
                "stop_at_endpoints": bool(cfg.stop_at_endpoints),
                "hold_orientation": bool(cfg.hold_orientation),
                "verify_original": bool(cfg.verify_original),
            },
        }
        raw = self._native_planner.plan_cartesian_path(req_dict)
        diag_d = raw["diagnostics"]
        diag = PlanDiagnostics(
            total_ms=float(diag_d["total_ms"]),
            ik_ms=float(diag_d["ik_ms"]),
            seed_ms=float(diag_d["seed_ms"]),
            sqp_ms=float(diag_d["sqp_ms"]),
            validate_ms=float(diag_d["validate_ms"]),
            retime_ms=float(diag_d["retime_ms"]),
            qp_calls=int(diag_d["qp_calls"]),
            sqp_iterations=int(diag_d["sqp_iterations"]),
            seed_strategy=str(diag_d["seed_strategy"]),
            witnesses=diag_d["witnesses"],
        )
        return CartesianPlanResult(
            trajectory=_trajectory_from_native(raw["trajectory"]),
            fraction=float(raw["fraction"]),
            completed_waypoints=int(raw["completed_waypoints"]),
            max_pos_error=float(raw["max_pos_error"]),
            max_rot_error=float(raw["max_rot_error"]),
            reason=raw["reason"],
            diagnostics=diag,
            world_version=int(raw["world_version"]),
        )

    # --- Experimental MuJoCo Warp path (shares C++ IK, BiRRT, QP, and unified C++ finalize_trajectory) ---

    def _resolve_endpoints(self, req: MotionPlanRequest, mask: np.ndarray, *, snap=None):
        cfg = req.config
        wm = cfg.effective_world_margin
        t0 = perf_counter()
        target = self.world._native if snap is None else snap

        def solve_single_ik(targets, seed_val, num_seeds):
            compiled = [self.world.compile_pose_constraint(t) for t in targets]
            base_seed = self.world.initial_positions.copy() if seed_val is None else self.world.configuration(seed_val)
            seeds = [base_seed]
            if num_seeds > 1:
                rng = np.random.default_rng(0)
                lo = np.where(np.isfinite(self.world.lower_limits), self.world.lower_limits, base_seed - np.pi)
                hi = np.where(np.isfinite(self.world.upper_limits), self.world.upper_limits, base_seed + np.pi)
                for _ in range(num_seeds - 1):
                    s = base_seed.copy()
                    s[mask] = rng.uniform(lo[mask], hi[mask])
                    seeds.append(s)
            q_out, ok_out, reasons = target.solve_ik_batch(
                [(compiled, s) for s in seeds], mask, float(cfg.margin), float(wm), 40, True,
            )
            valid = np.flatnonzero(ok_out)
            if not len(valid):
                return IKResult(None, reasons[0] if reasons else "IK failed")
            best = valid[int(np.argmin(np.linalg.norm(q_out[valid] - base_seed[None, :], axis=-1)))]
            return IKResult(q_out[best].copy(), None)

        if _is_pose_sequence(req.start):
            s_ik = solve_single_ik(req.start, req.start_seed, max(1, cfg.num_seeds))
            if not s_ik.success:
                return None, None, f"start IK failed: {s_ik.reason}", (perf_counter() - t0) * 1000.0
            start_q = s_ik.positions
        else:
            start_q = self.world.configuration(req.start)

        goal_candidates: list[tuple[int, np.ndarray]] = []
        if _is_pose_sequence(req.goal):
            g_seed = start_q if req.goal_seed is None else self.world.configuration(req.goal_seed)
            g_ik = solve_single_ik(req.goal, g_seed, max(1, cfg.num_seeds))
            if not g_ik.success:
                return start_q, None, f"goal IK failed: {g_ik.reason}", (perf_counter() - t0) * 1000.0
            goal_candidates.append((0, g_ik.positions))
        elif (isinstance(req.goal, Sequence) and not isinstance(req.goal, (str, bytes, np.ndarray))
              and len(req.goal) > 0 and _is_pose_sequence(req.goal[0])):
            g_seed = start_q if req.goal_seed is None else self.world.configuration(req.goal_seed)
            problems = [([self.world.compile_pose_constraint(t) for t in ts], g_seed) for ts in req.goal]
            q_out, ok_out, _ = target.solve_ik_batch(problems, mask, float(cfg.margin), float(wm), 40, True)
            valid = np.flatnonzero(ok_out)
            if not len(valid):
                return start_q, None, f"all {len(req.goal)} GoalSet candidates failed IK", (perf_counter() - t0) * 1000.0
            dists = np.linalg.norm(q_out[valid] - start_q[None, :], axis=-1)
            for idx in valid[np.argsort(dists)]:
                goal_candidates.append((int(idx), q_out[int(idx)].copy()))
        elif (isinstance(req.goal, Sequence) and not isinstance(req.goal, (str, bytes, np.ndarray))
              and len(req.goal) > 0 and not isinstance(req.goal[0], (int, float, np.number))):
            for idx, gq in enumerate(req.goal):
                goal_candidates.append((idx, self.world.configuration(gq)))
            goal_candidates.sort(key=lambda item: float(np.linalg.norm(item[1] - start_q)))
        else:
            goal_candidates.append((0, self.world.configuration(req.goal)))

        ik_ms = (perf_counter() - t0) * 1000.0
        if not mask.all():
            for i, (gid, gq) in enumerate(goal_candidates):
                gq_fixed = gq.copy()
                gq_fixed[~mask] = start_q[~mask]
                goal_candidates[i] = (gid, gq_fixed)
        return start_q, goal_candidates, None, ik_ms

    def _validate_endpoints(self, start_q: np.ndarray, goal_q: np.ndarray, cfg: SolverConfig, fixed: int, *, snap) -> str | None:
        ends = np.stack((start_q, goal_q), axis=0)
        if (ends < self.world.lower_limits).any() or (ends > self.world.upper_limits).any():
            return "start or goal violates joint limits"
        if cfg.dt is not None:
            moving = cfg.waypoints - 2 * fixed + 1
            vmax = self.world.velocity_limits * cfg.velocity_scale
            if (np.abs(goal_q - start_q) / (moving * cfg.dt) > vmax * (1.0 + 1e-6)).any():
                return "joint velocity limits require a longer duration"
        if snap.num_pairs > 0:
            margins = snap.pair_margins(float(cfg.margin), float(cfg.effective_world_margin))
            pair_names = tuple(snap.pair_shape_names())
            (frames,) = snap.compute_frames(np.ascontiguousarray(ends), False)
            for verify in ((False, True) if snap.has_mesh else (False,)):
                rows, pids = snap.screen_aabbs(frames, margins, verify, 1)
                if len(rows):
                    hit = self.backend.first_violation(frames, rows, pids, margins, verify, snap=snap)
                    if hit is not None:
                        seg = 0 if int(hit[0]) == 0 else cfg.waypoints - 2
                        return f"endpoint collision on segment {seg}, pair {pair_names[int(hit[1])]}"
        return None

    def _build_seed_jobs(self, start_q, goal_candidates, req, mask, pose_map, joint_map, *, snap, force_rrt=False):
        cfg = req.config
        fixed = 2 if cfg.stop_at_endpoints else 1
        dt_smooth = cfg.dt if cfg.dt is not None else 0.25
        interior_anchors = {k: v for k, v in joint_map.items() if fixed <= k < cfg.waypoints - fixed}
        t_seed = perf_counter()
        specs: list[SQPJobSpec] = []

        def make_spec(init_q, strategy, goal_idx):
            return SQPJobSpec(
                initial=init_q, fixed=fixed, dt_vel=cfg.dt, dt_smooth=dt_smooth,
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

        if not force_rrt:
            for gid, gq in goal_candidates[:max(1, cfg.num_seeds)]:
                specs.append(make_spec(_linear_seed(start_q, gq, cfg.waypoints, fixed, interior_anchors), "straight", gid))

        need_rrt = force_rrt or (cfg.use_global_seeding and cfg.num_seeds > 1)
        if need_rrt and not interior_anchors:
            n_rrt = max(1, cfg.num_seeds) if force_rrt else max(1, cfg.num_seeds - len(specs))
            rrt_starts = np.repeat(start_q[None, :], n_rrt, axis=0)
            rrt_goals = np.stack([goal_candidates[i % len(goal_candidates)][1] for i in range(n_rrt)], axis=0)
            rrt_gids = [goal_candidates[i % len(goal_candidates)][0] for i in range(n_rrt)]
            rrt_seeds = [1001 + i * 7919 for i in range(n_rrt)]
            inner_wp = cfg.waypoints - 2 * (fixed - 1)
            paths, ok = snap.plan_birrt_batch(
                rrt_starts, rrt_goals, mask, float(cfg.margin), float(cfg.effective_world_margin),
                float(cfg.rrt_step_size), int(cfg.rrt_max_iterations), float(cfg.rrt_timeout_s),
                rrt_seeds, bool(snap.has_mesh), int(cfg.rrt_shortcut_iters), int(inner_wp),
            )
            for i in range(n_rrt):
                if ok[i]:
                    path = paths[i]
                    if fixed == 2:
                        path = np.concatenate((start_q[None, :], path, rrt_goals[i:i + 1]), axis=0)
                    specs.append(make_spec(path, "birrt", rrt_gids[i]))

        if not force_rrt and len(specs) < cfg.num_seeds and not interior_anchors:
            rng = np.random.default_rng(42)
            lo = np.where(np.isfinite(self.world.lower_limits), self.world.lower_limits, start_q - 1.5)
            hi = np.where(np.isfinite(self.world.upper_limits), self.world.upper_limits, start_q + 1.5)
            mid_idx = cfg.waypoints // 2
            while len(specs) < cfg.num_seeds:
                gid, gq = goal_candidates[len(specs) % len(goal_candidates)]
                mid = 0.5 * (start_q + gq)
                mid[mask] = np.clip(mid[mask] + rng.normal(0.0, 0.35, size=int(mask.sum())), lo[mask], hi[mask])
                specs.append(make_spec(_linear_seed(start_q, gq, cfg.waypoints, fixed, {mid_idx: mid}), "perturbed", gid))
        return specs, (perf_counter() - t_seed) * 1000.0

    def _plan_warp_single(self, request: MotionPlanRequest, *, snap=None) -> PlanResult:
        pinned = self.world._native.snapshot() if snap is None else snap
        if (request.manifold_constraints or pinned.has_esdf
                or request.start_velocity is not None or request.start_acceleration is not None):
            # Constrained global manifold planning, CPU voxel ESDF sphere-cover planning, and
            # fixed start-boundary planning execute natively in C++ against the exact pinned snapshot.
            req_dict = _request_to_native_dict(self.world, request)
            raw = self._native_planner.plan_with_snapshot(pinned, req_dict)
            return _plan_result_from_native(raw)
        t_start = perf_counter()
        cfg = request.config
        fixed = 2 if cfg.stop_at_endpoints else 1
        if cfg.waypoints < 2 * fixed:
            raise ValueError(f"waypoints must be >= {2 * fixed}")
        mask = self.world.group_mask(request.group)
        pose_map, joint_map = _expand_path_constraints(
            self.world, cfg.waypoints, fixed, request.waypoint_targets, request.path_constraints,
        )
        pinned_ver = int(pinned.version)
        start_q, goal_candidates, ep_err, ik_ms = self._resolve_endpoints(request, mask, snap=pinned)
        if ep_err is not None or goal_candidates is None:
            return PlanResult(None, None, 0, ep_err,
                              PlanDiagnostics(total_ms=(perf_counter() - t_start) * 1000.0, ik_ms=ik_ms),
                              world_version=pinned_ver)
        valid_goals = []
        last_ep_err = None
        for gid, gq in goal_candidates:
            err = self._validate_endpoints(start_q, gq, cfg, fixed, snap=pinned)
            if err is None:
                valid_goals.append((gid, gq))
            else:
                last_ep_err = err
        if not valid_goals:
            return PlanResult(None, None, 0, last_ep_err,
                              PlanDiagnostics(total_ms=(perf_counter() - t_start) * 1000.0, ik_ms=ik_ms),
                              world_version=pinned_ver)

        specs, seed_ms = self._build_seed_jobs(start_q, valid_goals, request, mask, pose_map, joint_map, snap=pinned)
        raw_results = run_sqp_batch(self.world, self.backend, specs, race=(len(specs) > 1 and cfg.first_feasible), snap=pinned)
        succeeded = [r for r in raw_results if r is not None and r.positions is not None]
        if not succeeded and cfg.use_global_seeding and cfg.num_seeds == 1 and request.seed is None:
            rrt_specs, rrt_seed_ms = self._build_seed_jobs(start_q, valid_goals, request, mask, pose_map, joint_map, snap=pinned, force_rrt=True)
            seed_ms += rrt_seed_ms
            if rrt_specs:
                rrt_results = run_sqp_batch(self.world, self.backend, rrt_specs, race=cfg.first_feasible, snap=pinned)
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
            return PlanResult(None, None if first is None else first.initial_positions, 0,
                              "trajectory optimization failed" if first is None else first.reason, diag,
                              world_version=pinned_ver)

        winner = min(succeeded, key=lambda r: r.cost)
        traj, retime_ms, retime_err = self._finalize_trajectory(winner.positions, cfg, snap=pinned)
        diag = PlanDiagnostics(
            total_ms=(perf_counter() - t_start) * 1000.0, ik_ms=ik_ms, seed_ms=seed_ms,
            sqp_ms=winner.sqp_ms, validate_ms=winner.validate_ms, retime_ms=retime_ms,
            qp_calls=winner.qp_calls, sqp_iterations=winner.sqp_iterations,
            seed_strategy=winner.seed_strategy, witnesses=winner.witnesses,
        )
        return PlanResult(
            traj, winner.initial_positions, winner.goal_index, retime_err, diag,
            world_version=pinned_ver,
            waypoint_indices=None if traj is None else np.arange(traj.positions.shape[0], dtype=np.int64),
        )

    def _plan_warp_batch(self, requests: Sequence[MotionPlanRequest], *, chunk_size: int) -> tuple[PlanResult, ...]:
        B = len(requests)
        pinned = self.world._native.snapshot()
        pinned_ver = int(pinned.version)
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
                if (req.manifold_constraints or pinned.has_esdf
                        or req.start_velocity is not None or req.start_acceleration is not None):
                    all_results[row_i] = self._plan_warp_single(req, snap=pinned)
                    continue
                cfg = req.config
                fixed = 2 if cfg.stop_at_endpoints else 1
                mask = self.world.group_mask(req.group)
                pose_map, joint_map = _expand_path_constraints(
                    self.world, cfg.waypoints, fixed, req.waypoint_targets, req.path_constraints,
                )
                start_q, goal_candidates, ep_err, ik_ms = self._resolve_endpoints(req, mask, snap=pinned)
                if ep_err is not None or goal_candidates is None:
                    all_results[row_i] = PlanResult(None, None, 0, ep_err, PlanDiagnostics(ik_ms=ik_ms),
                                                    world_version=pinned_ver)
                    continue
                valid_goals = [(gid, gq) for gid, gq in goal_candidates
                               if self._validate_endpoints(start_q, gq, cfg, fixed, snap=pinned) is None]
                if not valid_goals:
                    err = self._validate_endpoints(start_q, goal_candidates[0][1], cfg, fixed, snap=pinned)
                    all_results[row_i] = PlanResult(None, None, 0, err, PlanDiagnostics(ik_ms=ik_ms),
                                                    world_version=pinned_ver)
                    continue
                specs, seed_ms = self._build_seed_jobs(start_q, valid_goals[:1], req, mask, pose_map, joint_map, snap=pinned)
                active_indices.append(row_i)
                active_specs.append(specs[0])
                active_ik_ms.append(ik_ms)
                active_seed_ms.append(seed_ms)

            if active_specs:
                sqp_out = run_sqp_batch(self.world, self.backend, active_specs, race=False, snap=pinned)
                for row_i, sqp_res, ik_ms, seed_ms in zip(active_indices, sqp_out, active_ik_ms, active_seed_ms):
                    req = requests[row_i]
                    cfg = req.config
                    if (sqp_res is None or sqp_res.positions is None) and cfg.use_global_seeding and req.seed is None:
                        all_results[row_i] = self._plan_warp_single(req, snap=pinned)
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
                            world_version=pinned_ver,
                        )
                    else:
                        traj, retime_ms, retime_err = self._finalize_trajectory(sqp_res.positions, cfg, snap=pinned)
                        all_results[row_i] = PlanResult(
                            traj, sqp_res.initial_positions, sqp_res.goal_index, retime_err,
                            PlanDiagnostics(
                                total_ms=(perf_counter() - chunk_t0) * 1000.0, ik_ms=ik_ms, seed_ms=seed_ms,
                                sqp_ms=sqp_res.sqp_ms, validate_ms=sqp_res.validate_ms, retime_ms=retime_ms,
                                qp_calls=sqp_res.qp_calls, sqp_iterations=sqp_res.sqp_iterations,
                                seed_strategy=sqp_res.seed_strategy, witnesses=sqp_res.witnesses,
                            ),
                            world_version=pinned_ver,
                            waypoint_indices=None if traj is None else np.arange(traj.positions.shape[0], dtype=np.int64),
                        )
        return tuple(r for r in all_results if r is not None)

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
        """Render seed path, smooth optimized trajectory, and closest collision witnesses."""
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
