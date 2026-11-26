"""End-to-end benchmark comparing 16-thread CPU Coal vs GPU-resident Batch L-BFGS (Phase 3.3).

Measures on the same pinned WorldSnapshot, same requests, and same strict_ccd settings:
  1. Pure C++ TBB batch verification ceiling on known-feasible trajectories (Amdahl limit).
  2. Container Unloading 6-DOF (static multi-box scene, margin=1mm, world_margin=14mm)
     across B = 1, 16, 64, 128 (matching historical B=128 comparison point).
  3. Panda 7-DOF + 64^3 3D ESDF point-cloud/voxel obstacle field across B = 1, 16, 64, 256, 512, 1024.
  4. Single-query multi-seed parallel exploration (B = 1, K = 64).

Primary metric: verified trajectories/second (strictly passing full request + independent validate_trajectory).
Reports cold JIT/capture separately from warm runs; warm timing includes H2D/D2H, IK, GPU opt, C++ verify, fallback.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

import gobot
from gobot.planning import (
    GpuBatchConfig,
    MotionPlanner,
    MotionPlanRequest,
    PlanningWorld,
    PoseConstraint,
    SolverConfig,
)


def _independent_verify(
    world: PlanningWorld,
    req: MotionPlanRequest,
    res,
    *,
    margin: float,
    world_margin: float,
    substeps: int,
    strict_ccd: bool,
) -> bool:
    if not res.success or res.trajectory is None:
        return False
    traj = res.trajectory
    # Both benchmark scenes request a single joint-space start and goal.
    if res.goal_index != 0 or res.world_version != world.version:
        return False
    for actual, expected in ((traj.positions[0], req.start), (traj.positions[-1], req.goal)):
        if not np.allclose(actual, world.configuration(expected), atol=1e-8, rtol=0):
            return False
    ok, _ = world.validate_trajectory(
        traj,
        margin=margin,
        world_margin=world_margin,
        collision_substeps=substeps,
        check_acceleration=True,
        strict_ccd=strict_ccd,
    )
    if not ok:
        return False
    if req.waypoint_targets:
        mapping = res.waypoint_indices
        if mapping is None or len(mapping) != req.config.waypoints:
            return False
        if mapping[0] != 0 or mapping[-1] != traj.num_waypoints - 1 or np.any(np.diff(mapping) < 0):
            return False
        for idx, target in req.waypoint_targets.items():
            q = traj.positions[mapping[idx]]
            if isinstance(target, PoseConstraint):
                _, _, reached = world._native.eval_pose_constraints(q, [world.compile_pose_constraint(target)], False)
                if not reached:
                    return False
            elif not np.allclose(q, world.configuration(target), atol=1e-4, rtol=0):
                return False
    return True


def _measure_cpp_verify_ceiling(
    planner_coal: MotionPlanner,
    world: PlanningWorld,
    requests: list[MotionPlanRequest],
    feasible_positions: list[np.ndarray],
) -> float:
    from gobot.planning.planner import _request_to_native_dict

    snap = world._native.snapshot()
    items = [
        {
            "request": _request_to_native_dict(world, req),
            "candidates": [{"positions": np.ascontiguousarray(pos, dtype=np.float64), "goal_index": 0}],
        }
        for req, pos in zip(requests, feasible_positions)
    ]
    # Warmup + 3 timed runs.
    planner_coal._native_planner.finalize_candidates_batch(snap, items, False)
    ts = []
    for _ in range(3):
        t0 = perf_counter()
        out = planner_coal._native_planner.finalize_candidates_batch(snap, items, False)
        ts.append((perf_counter() - t0) * 1000.0)
        assert all(x["result"]["trajectory"] is not None for x in out), "known-feasible batch failed verify"
    return median(ts)


def run_container_benchmark(sizes: list[int], repeats: int, strict_ccd: bool, num_seeds: int) -> None:
    ex_dir = Path(__file__).resolve().parent.parent / "container_unloading"
    sys.path.insert(0, str(ex_dir))
    from plan import PAYLOAD, ROBOT, SAMPLES_PER_SEGMENT, _arm_seed

    ctx = gobot.app.context()
    ctx.set_project_path(str(ex_dir))
    ctx.load_scene("res://world.jscn")
    try:
        snapshot = ctx.compile_physics_scene_snapshot()
        pick_node, drop_node = ctx.root.find("Pick"), ctx.root.find("Drop")
        world = PlanningWorld.from_snapshot(snapshot, {ROBOT: "base_link"})
        coal = MotionPlanner(world, backend="coal")

        def ep_ik(marker):
            xyz = np.asarray(marker.position, dtype=np.float64)
            rot = Rotation.from_euler("xyz", marker.rotation_degrees, degrees=True).as_matrix()
            r = coal.solve_ik([PoseConstraint(PAYLOAD, xyz, rot)], seed=_arm_seed(xyz), margin=0.001, world_margin=0.014)
            if not r.success:
                raise RuntimeError(r.reason)
            return r.positions

        start = ep_ik(pick_node)
        goal = ep_ik(drop_node)
    finally:
        ctx.clear_scene()

    substeps = SAMPLES_PER_SEGMENT // 2
    ref_cfg = SolverConfig(
        waypoints=14, dt=None, retime=True, margin=0.001, world_margin=0.014,
        collision_substeps=5, iterations=50, first_feasible=False, strict_ccd=strict_ccd,
    )
    reference = coal.plan(MotionPlanRequest(start=start, goal=goal, config=ref_cfg))
    if not reference.success:
        raise RuntimeError(f"container reference failed: {reference.reason}")

    cfg = SolverConfig(
        waypoints=14, dt=None, retime=True, margin=0.001, world_margin=0.014,
        collision_substeps=substeps, iterations=50, first_feasible=True, strict_ccd=strict_ccd,
    )

    gpu_cfg_pure = GpuBatchConfig(
        num_seeds=num_seeds, top_k_verify=4, cpu_fallback_on_fail=False, use_cuda_graph=True,
    )
    gpu_cfg_fb = GpuBatchConfig(
        num_seeds=num_seeds, top_k_verify=4, cpu_fallback_on_fail=True, use_cuda_graph=True,
    )
    gpu_pure = MotionPlanner(world, backend="gpu_batch", gpu_config=gpu_cfg_pure)
    gpu_fb = MotionPlanner(world, backend="gpu_batch", gpu_config=gpu_cfg_fb)

    print(f"\n=== Container Unloading 6-DOF (strict_ccd={strict_ccd}, substeps={substeps}, K={num_seeds}) ===", flush=True)
    # Cold JIT warmup on B=1
    t_cold0 = perf_counter()
    gpu_pure.plan(MotionPlanRequest(start=start, goal=goal, waypoint_targets={6: reference.positions[6]}, config=cfg))
    print(f"GPU cold JIT + first capture (B=1): {(perf_counter() - t_cold0) * 1000.0:.1f} ms", flush=True)

    for B in sizes:
        offsets = np.arange(B, dtype=np.float64) / max(B - 1, 1)
        starts = np.repeat(start[None, :], B, axis=0)
        goals = np.repeat(goal[None, :], B, axis=0)
        starts[:, 0] += 0.003 * offsets
        goals[:, 0] -= 0.015 * offsets
        via = np.repeat(reference.positions[6:7], B, axis=0)
        via[:, 0] += 0.006 * offsets
        requests = [
            MotionPlanRequest(start=starts[i], goal=goals[i], waypoint_targets={6: via[i]}, config=cfg)
            for i in range(B)
        ]

        # 1. Coal baseline
        coal_times = []
        coal_res = None
        for _ in range(repeats):
            t0 = perf_counter()
            coal_res = coal.plan_batch(requests, chunk_size=max(64, B))
            coal_times.append((perf_counter() - t0) * 1000.0)
        coal_ver = sum(_independent_verify(world, req, r, margin=0.001, world_margin=0.014, substeps=SAMPLES_PER_SEGMENT, strict_ccd=strict_ccd)
                       for req, r in zip(requests, coal_res.results))
        coal_med = median(coal_times)
        coal_tps = coal_ver / (coal_med / 1000.0)

        # Measure pure C++ batch verification ceiling on the coal-solved feasible trajectories
        ver_ceil_ms = _measure_cpp_verify_ceiling(
            coal, world,
            [req for req, r in zip(requests, coal_res.results) if r.success],
            [r.positions for r in coal_res.results if r.success],
        )

        # 2. GPU pure (no CPU fallback) & GPU + fallback
        for label, planner in (("gpu_pure", gpu_pure), ("gpu+fb", gpu_fb)):
            # Warmup capture for this B
            t_cap0 = perf_counter()
            planner.plan_batch(requests, chunk_size=max(64, B))
            cap_wall = (perf_counter() - t_cap0) * 1000.0
            cap_stats = planner.backend.last_stats

            times = []
            res = None
            stats_list = []
            for _ in range(repeats):
                t0 = perf_counter()
                res = planner.plan_batch(requests, chunk_size=max(64, B))
                times.append((perf_counter() - t0) * 1000.0)
                stats_list.append(planner.backend.last_stats)
            ver_cnt = sum(_independent_verify(world, req, r, margin=0.001, world_margin=0.014, substeps=SAMPLES_PER_SEGMENT, strict_ccd=strict_ccd)
                          for req, r in zip(requests, res.results))
            med_ms = median(times)
            p95_ms = float(np.percentile(times, 95))
            tps = ver_cnt / (med_ms / 1000.0)
            st = stats_list[-1]
            print(
                f"  B={B:4d} {label:8s}: {med_ms:7.1f} ms (P95 {p95_ms:6.1f} ms) | {tps:7.1f} ver/s "
                f"| solved {int(res.success.sum())}/{B} ver {ver_cnt:4d}/{B} | top1={st.top1_passed}/{B} top4={st.topm_passed}/{B} fb={st.cpu_fallback_count} "
                f"| gpu={st.gpu_opt_ms:.1f}ms ver_cpp_est={st.verify_ms:.1f}ms fb_cpp_est={st.fallback_ms:.1f}ms "
                f"(cap={cap_stats.graph_capture_ms:.1f}ms, first={cap_wall:.1f}ms)",
                flush=True,
            )
        print(
            f"  B={B:4d} {'coal':8s}: {coal_med:7.1f} ms | {coal_tps:7.1f} ver/s "
            f"| solved {int(coal_res.success.sum())}/{B} ver {coal_ver:4d}/{B} | cpp_verify_ceiling={ver_ceil_ms:.1f}ms",
            flush=True,
        )


def run_panda_esdf_benchmark(sizes: list[int], repeats: int, strict_ccd: bool, num_seeds: int) -> None:
    ex_dir = Path(__file__).resolve().parent.parent / "panda_trajopt"
    ROBOT = "panda"
    HAND = (ROBOT, "panda_hand")
    IGNORED_MOUNTS = (
        ("panda_link0", "panda_link2"),
        ("panda_link4", "panda_link6"),
        ("panda_link7", "panda_hand"),
    )
    DOWN = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    START_SEED = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, -2.3], dtype=np.float64)

    ctx = gobot.app.context()
    ctx.set_project_path(str(ex_dir))
    ctx.load_scene("res://world.jscn")
    try:
        snapshot = ctx.compile_physics_scene_snapshot()
        start_pos = np.asarray(ctx.root.find("Start").position, dtype=np.float64)
        goal_pos = np.asarray(ctx.root.find("Goal").position, dtype=np.float64)
        world = PlanningWorld.from_snapshot(
            snapshot, {ROBOT: "panda_link0"},
            ignored_link_pairs=[((ROBOT, a), (ROBOT, b)) for a, b in IGNORED_MOUNTS],
        )
    finally:
        ctx.clear_scene()

    # Build a 64^3 3D ESDF with a central pillar obstacle between Start and Goal.
    nx = ny = nz = 64
    res_m = 0.025
    origin = np.array([-0.4, -0.8, 0.0], dtype=np.float64)
    occ = np.zeros((nx, ny, nz), dtype=np.uint8)
    mid_xy = 0.5 * (start_pos[:2] + goal_pos[:2])
    ix0, ix1 = int((mid_xy[0] - 0.06 - origin[0]) / res_m), int((mid_xy[0] + 0.06 - origin[0]) / res_m)
    iy0, iy1 = int((mid_xy[1] - 0.06 - origin[1]) / res_m), int((mid_xy[1] + 0.06 - origin[1]) / res_m)
    iz0, iz1 = int((0.12 - origin[2]) / res_m), int((0.68 - origin[2]) / res_m)
    occ[max(0, ix0):min(nx, ix1), max(0, iy0):min(ny, iy1), max(0, iz0):min(nz, iz1)] = 1
    world.set_occupancy_grid(occ, origin=origin, resolution=res_m, inflation_radius=0.005)

    coal = MotionPlanner(world, backend="coal")
    ik_s = coal.solve_ik([PoseConstraint(HAND, start_pos, DOWN)], seed=START_SEED, margin=0.002, world_margin=0.015)
    ik_g = coal.solve_ik([PoseConstraint(HAND, goal_pos, DOWN)], seed=START_SEED, margin=0.002, world_margin=0.015)
    if not ik_s.success or not ik_g.success:
        raise RuntimeError(f"Panda IK failed: start={ik_s.reason} goal={ik_g.reason}")
    q_left, q_right = ik_s.positions, ik_g.positions

    gpu_pure = MotionPlanner(
        world, backend="gpu_batch",
        gpu_config=GpuBatchConfig(num_seeds=num_seeds, top_k_verify=4, cpu_fallback_on_fail=False, use_cuda_graph=True),
    )
    gpu_fb = MotionPlanner(
        world, backend="gpu_batch",
        gpu_config=GpuBatchConfig(num_seeds=num_seeds, top_k_verify=4, cpu_fallback_on_fail=True, use_cuda_graph=True),
    )

    cfg = SolverConfig(
        waypoints=16, dt=None, retime=True, margin=0.002, world_margin=0.012,
        collision_substeps=4, iterations=40, first_feasible=True, strict_ccd=strict_ccd,
    )

    print(f"\n=== Panda 7-DOF + 64^3 3D ESDF (strict_ccd={strict_ccd}, K={num_seeds}) ===", flush=True)
    gpu_pure.plan(MotionPlanRequest(start=q_left, goal=q_right, config=cfg))

    rng = np.random.default_rng(12345)
    snap_p = world._native.snapshot()
    margins_p = snap_p.pair_margins(0.002, 0.012)
    for B in sizes:
        starts_list = [q_left.copy()]
        goals_list = [q_right.copy()]
        while len(starts_list) < B:
            qs = np.clip(q_left + rng.uniform(-0.15, 0.15, size=7), world.lower_limits + 0.05, world.upper_limits - 0.05)
            qg = np.clip(q_right + rng.uniform(-0.15, 0.15, size=7), world.lower_limits + 0.05, world.upper_limits - 0.05)
            if (world.collision_distances(qs[None, :]).min() >= 0.015
                    and world.collision_distances(qg[None, :]).min() >= 0.015
                    and world.esdf_distances(qs[None, :]).min() >= 0.015
                    and world.esdf_distances(qg[None, :]).min() >= 0.015):
                starts_list.append(qs)
                goals_list.append(qg)
        requests = [MotionPlanRequest(start=starts_list[i], goal=goals_list[i], config=cfg) for i in range(B)]

        coal_repeats = 1 if B >= 256 else repeats
        coal_times = []
        coal_res = None
        for _ in range(coal_repeats):
            t0 = perf_counter()
            coal_res = coal.plan_batch(requests, chunk_size=min(B, 256))
            coal_times.append((perf_counter() - t0) * 1000.0)
        coal_ver = sum(_independent_verify(world, req, r, margin=0.002, world_margin=0.012, substeps=4, strict_ccd=strict_ccd)
                       for req, r in zip(requests, coal_res.results))
        coal_med = median(coal_times)
        coal_tps = coal_ver / (coal_med / 1000.0)

        ver_ceil_ms = _measure_cpp_verify_ceiling(
            coal, world,
            [req for req, r in zip(requests, coal_res.results) if r.success],
            [r.positions for r in coal_res.results if r.success],
        )

        for label, planner in (("gpu_pure", gpu_pure), ("gpu+fb", gpu_fb)):
            planner.plan_batch(requests, chunk_size=min(B, 512))
            times = []
            res = None
            stats_list = []
            for _ in range(repeats):
                t0 = perf_counter()
                res = planner.plan_batch(requests, chunk_size=min(B, 512))
                times.append((perf_counter() - t0) * 1000.0)
                stats_list.append(planner.backend.last_stats)
            ver_cnt = sum(_independent_verify(world, req, r, margin=0.002, world_margin=0.012, substeps=4, strict_ccd=strict_ccd)
                          for req, r in zip(requests, res.results))
            med_ms = median(times)
            p95_ms = float(np.percentile(times, 95))
            tps = ver_cnt / (med_ms / 1000.0)
            st = stats_list[-1]
            print(
                f"  B={B:4d} {label:8s}: {med_ms:7.1f} ms (P95 {p95_ms:6.1f} ms) | {tps:7.1f} ver/s "
                f"| solved {int(res.success.sum())}/{B} ver {ver_cnt:4d}/{B} | top1={st.top1_passed}/{B} top4={st.topm_passed}/{B} fb={st.cpu_fallback_count} "
                f"| gpu={st.gpu_opt_ms:.1f}ms ver_cpp_est={st.verify_ms:.1f}ms fb_cpp_est={st.fallback_ms:.1f}ms",
                flush=True,
            )
        print(
            f"  B={B:4d} {'coal':8s}: {coal_med:7.1f} ms | {coal_tps:7.1f} ver/s "
            f"| solved {int(coal_res.success.sum())}/{B} ver {coal_ver:4d}/{B} | cpp_verify_ceiling={ver_ceil_ms:.1f}ms",
            flush=True,
        )


    # Also measure single-query multi-seed regime (B=1, K=64) separately from high-batch regime.
    p_ms = MotionPlanner(
        world, backend="gpu_batch",
        gpu_config=GpuBatchConfig(num_seeds=64, top_k_verify=4, cpu_fallback_on_fail=False, use_cuda_graph=True),
    )
    req_one = MotionPlanRequest(start=q_left, goal=q_right, config=cfg)
    t_cap0 = perf_counter()
    p_ms.plan(req_one)
    ms_cap_ms = (perf_counter() - t_cap0) * 1000.0
    ms_times = []
    r_one = None
    for _ in range(repeats):
        t0 = perf_counter()
        r_one = p_ms.plan(req_one)
        ms_times.append((perf_counter() - t0) * 1000.0)
    ms_ok = _independent_verify(world, req_one, r_one, margin=0.002, world_margin=0.012, substeps=4, strict_ccd=strict_ccd)
    ms_st = p_ms.backend.last_stats
    print(
        f"  B=   1 K=64    : {median(ms_times):7.1f} ms (P95 {float(np.percentile(ms_times, 95)):6.1f} ms) "
        f"| ver {int(ms_ok)}/1 | gpu={ms_st.gpu_opt_ms:.1f}ms ver_cpp={ms_st.verify_ms:.1f}ms (first={ms_cap_ms:.1f}ms)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-sizes", type=int, nargs="+", default=[1, 16, 64, 128])
    parser.add_argument("--panda-sizes", type=int, nargs="+", default=[1, 16, 64, 256, 512, 1024])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--strict-ccd", action="store_true", default=False)
    parser.add_argument("--only", choices=("all", "container", "panda"), default="all")
    args = parser.parse_args()

    if args.only in ("all", "container"):
        run_container_benchmark(args.container_sizes, args.repeats, args.strict_ccd, args.seeds)
    if args.only in ("all", "panda"):
        run_panda_esdf_benchmark(args.panda_sizes, args.repeats, args.strict_ccd, args.seeds)


if __name__ == "__main__":
    main()
