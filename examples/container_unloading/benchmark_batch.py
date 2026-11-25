"""Benchmark C++ batch IK, short transfer, and Coal vs. MuJoCo Warp batch TrajOpt on FANUC M-10iA."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

import gobot
from gobot.planning import (
    MotionPlanner,
    MotionPlanRequest,
    PlanningWorld,
    PoseConstraint,
    SolverConfig,
)
from plan import DT, PAYLOAD, ROBOT, SAMPLES_PER_SEGMENT, _arm_seed


def endpoint_ik(planner: MotionPlanner, marker) -> np.ndarray:
    xyz = np.asarray(marker.position, dtype=np.float64)
    rotation = Rotation.from_euler("xyz", marker.rotation_degrees, degrees=True).as_matrix()
    result = planner.solve_ik(
        [PoseConstraint(PAYLOAD, xyz, rotation)],
        seed=_arm_seed(xyz),
        margin=0.001,
        world_margin=0.014,
    )
    if not result.success:
        raise RuntimeError(result.reason)
    return result.positions


def verify(world: PlanningWorld, path: np.ndarray, samples: np.ndarray, payload_pairs: list[int]) -> tuple[bool, float]:
    """Independently query every convex and authored pair, outside the planner timer."""
    distances = world.collision_distances(samples, verify_original=False)
    margins = world.pair_margins(0.001, 0.014)
    valid = bool((distances >= np.maximum(0.0, margins - 1e-6)).all())
    if world.has_mesh:
        original = world.collision_distances(samples, verify_original=True)
        valid &= bool((original >= np.maximum(0.0, margins - 1e-6)).all())
    ok_val, _ = world.validate_trajectory(
        path, dt=DT, margin=0.001, world_margin=0.014, collision_substeps=SAMPLES_PER_SEGMENT,
    )
    return valid and ok_val, float(distances[:, payload_pairs].min()) * 1000.0


def benchmark_ik_batch(planner: MotionPlanner, pick_pos, pick_rot, sizes: list[int]) -> None:
    base_seed = np.asarray(_arm_seed(pick_pos), dtype=np.float64)
    for b in sizes:
        offsets = np.linspace(-0.015, 0.015, b) if b > 1 else np.array([0.0])
        targets = [[PoseConstraint(PAYLOAD, pick_pos + np.array([0.0, dy, 0.0]), pick_rot)] for dy in offsets]
        seeds = np.repeat(base_seed[None, :], b, axis=0)
        # Warmup + 5 timed runs.
        planner.solve_ik_batch(targets, seeds=seeds, margin=0.001, world_margin=0.014)
        times = []
        for _ in range(5):
            t0 = perf_counter()
            res = planner.solve_ik_batch(targets, seeds=seeds, margin=0.001, world_margin=0.014)
            times.append((perf_counter() - t0) * 1000.0)
        print(
            f"C++ Batch IK B={b:2d}: median {median(times):.2f} ms "
            f"({median(times) / b:.2f} ms/solve), solved {int(res.success.sum())}/{b}",
            flush=True,
        )


def benchmark_short_transfer(planner: MotionPlanner, start: np.ndarray, first_feasible: bool) -> None:
    goal = start.copy()
    goal[0] -= 0.1
    cfg = SolverConfig(
        waypoints=14, dt=DT, margin=0.001, world_margin=0.014,
        collision_substeps=5, iterations=50, first_feasible=first_feasible,
    )
    req = MotionPlanRequest(start=start, goal=goal, config=cfg)
    times = []
    for _ in range(12):
        begin = perf_counter()
        result = planner.plan(req)
        times.append(perf_counter() - begin)
        if not result.success or not np.array_equal(result.positions, result.initial_positions):
            raise RuntimeError(f"short transfer failed to accept straight path: {result.reason}")
    print(f"Short transfer (zero-QP): {1000.0 * median(times[1:]):.2f} ms median (11 warm runs)", flush=True)


def run(sizes: list[int], repeats: int, substeps: int, first_feasible: bool) -> None:
    context = gobot.app.context()
    context.set_project_path(str(Path(__file__).resolve().parent))
    context.load_scene("res://world.jscn")
    try:
        snapshot = context.compile_physics_scene_snapshot()
        pick_node, drop_node = context.root.find("Pick"), context.root.find("Drop")
        pick_pos = np.asarray(pick_node.position, dtype=np.float64)
        pick_rot = Rotation.from_euler("xyz", pick_node.rotation_degrees, degrees=True).as_matrix()
        world = PlanningWorld.from_snapshot(snapshot, {ROBOT: "base_link"})
        planners = {name: MotionPlanner(world, backend=name) for name in ("coal", "mujoco_warp")}
        coal = planners["coal"]
        start = endpoint_ik(coal, pick_node)
        goal = endpoint_ik(coal, drop_node)
    finally:
        context.clear_scene()

    mode = "sqp_fast" if first_feasible else "sqp"
    print(
        f"{mode} | {world.num_pairs} shape pairs | {substeps} substeps | "
        f"Warp device: {wp.get_device()} | {repeats} timed runs",
        flush=True,
    )
    benchmark_ik_batch(coal, pick_pos, pick_rot, sorted(set(sizes) | {1, 16}))
    benchmark_short_transfer(coal, goal, first_feasible)

    ref_cfg = SolverConfig(
        waypoints=14, dt=DT, margin=0.001, world_margin=0.014,
        collision_substeps=5, iterations=50, first_feasible=False,
    )
    reference = coal.plan(MotionPlanRequest(start=start, goal=goal, config=ref_cfg))
    if not reference.success:
        raise RuntimeError(reference.reason)

    cfg = SolverConfig(
        waypoints=14, dt=DT, margin=0.001, world_margin=0.014,
        collision_substeps=substeps, iterations=50, first_feasible=first_feasible,
    )
    payload_pairs = [
        i for i, (a, b) in enumerate(world.collision_pairs)
        if a.startswith(f"{ROBOT}/{PAYLOAD[1]}/") and not b.startswith(f"{ROBOT}/")
    ]
    u = (np.arange(SAMPLES_PER_SEGMENT, dtype=np.float64) / SAMPLES_PER_SEGMENT)[:, None]

    # Warmup Warp kernels outside timed loop.
    planners["mujoco_warp"].plan(
        MotionPlanRequest(start=start, goal=goal, waypoint_targets={6: reference.positions[6]}, config=cfg)
    )

    for size in sizes:
        offsets = np.arange(size, dtype=np.float64) / max(size - 1, 1)
        starts = np.repeat(start[None, :], size, axis=0)
        goals = np.repeat(goal[None, :], size, axis=0)
        starts[:, 0] += 0.003 * offsets
        goals[:, 0] -= 0.015 * offsets
        via = np.repeat(reference.positions[6:7], size, axis=0)
        via[:, 0] += 0.006 * offsets
        requests = [
            MotionPlanRequest(start=starts[i], goal=goals[i], waypoint_targets={6: via[i]}, config=cfg)
            for i in range(size)
        ]
        for name, planner in planners.items():
            times = []
            for _ in range(repeats):
                begin = perf_counter()
                result = planner.plan_batch(requests)
                times.append(perf_counter() - begin)
            clearances, verified = [], 0
            for row, path in enumerate(result.positions):
                if not result.success[row]:
                    print(f"  row {row}: {result.reasons[row]}")
                    continue
                between = (1.0 - u)[None, :, :] * path[:-1, None, :] + u[None, :, :] * path[1:, None, :]
                samples = np.concatenate((between.reshape(-1, world.num_joints), path[-1:]), axis=0)
                valid, clearance = verify(world, path, samples, payload_pairs)
                clearances.append(clearance)
                verified += valid
            avg_qp = float(np.mean([r.diagnostics.qp_calls for r in result.results]))
            med_s = median(times)
            print(
                f"B={size:2d} {name:11s}: median {med_s:.3f}s ({1000.0 * med_s / size:5.1f} ms/traj) "
                f"{np.round(times, 3)}, avg_qp={avg_qp:.1f}, "
                f"solved {int(result.success.sum())}/{size}, verified {verified}/{size}, "
                f"min carton clearance {min(clearances, default=float('nan')):.2f} mm",
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--collision-substeps", type=int, default=SAMPLES_PER_SEGMENT // 2)
    parser.add_argument("--solver", choices=("sqp", "sqp_fast"), default="sqp_fast")
    args = parser.parse_args()
    run(args.sizes, args.repeats, args.collision_substeps, args.solver == "sqp_fast")
