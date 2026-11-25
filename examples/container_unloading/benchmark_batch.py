"""Compare independent pick/via/drop trajectories on the same FANUC snapshot."""

import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

import gobot
from gobot.planning import MotionPlanner, MotionPlanRequest, PlanningWorld, PoseConstraint, SolverConfig
from plan import DT, PAYLOAD, ROBOT, SAMPLES_PER_SEGMENT


def endpoint(planner, marker):
    xyz = marker.position
    rotation = Rotation.from_euler("xyz", marker.rotation_degrees, degrees=True).as_matrix()
    seed = [np.arctan2(xyz[1], xyz[0]), 0.12, -0.50, 0, 0.65, 0]
    result = planner.solve_ik([PoseConstraint(PAYLOAD, xyz, rotation)], seed=seed, margin=0.001, world_margin=0.014)
    if not result.success:
        raise RuntimeError(result.reason)
    return result.positions


def verify(world, path, samples, payload_pairs):
    """Independently query every convex and authored pair, outside the planner timer."""
    distances = world.collision_distances(samples, verify_original=False)
    margins = world.pair_margins(0.001, 0.014)
    valid = bool((distances >= np.maximum(0.0, margins - 1e-6)).all())
    if world.has_mesh:
        original = world.collision_distances(samples, verify_original=True)
        valid &= bool((original >= np.maximum(0.0, margins - 1e-6)).all())
    ok_val, _ = world.validate_trajectory(path, dt=DT, margin=0.001, world_margin=0.014,
                                          collision_substeps=SAMPLES_PER_SEGMENT)
    return valid and ok_val, float(distances[:, payload_pairs].min()) * 1000.0


def short_transfer(planner, start, first_feasible):
    goal = start.copy()
    goal[0] -= 0.1
    cfg = SolverConfig(waypoints=14, dt=DT, margin=0.001, world_margin=0.014,
                       collision_substeps=5, iterations=50, first_feasible=first_feasible)
    req = MotionPlanRequest(start=start, goal=goal, config=cfg)
    times = []
    for _ in range(12):
        begin = perf_counter()
        result = planner.plan(req)
        times.append(perf_counter() - begin)
        if not result.success or not np.array_equal(result.positions, result.initial_positions):
            raise RuntimeError(f"short transfer failed to accept straight path: {result.reason}")
    print(f"Short transfer: {1000 * median(times[1:]):.2f} ms median (11 warm runs)", flush=True)


def run(sizes, repeats, substeps, first_feasible):
    context = gobot.app.context()
    context.set_project_path(str(Path(__file__).resolve().parent))
    context.load_scene("res://world.jscn")
    snapshot = context.compile_physics_scene_snapshot()
    world = PlanningWorld.from_snapshot(snapshot, {ROBOT: "base_link"})
    planners = {name: MotionPlanner(world, backend=name) for name in ("coal", "mujoco_warp")}
    coal = planners["coal"]
    start, goal = [endpoint(coal, context.root.find(name)) for name in ("Pick", "Drop")]
    context.clear_scene()

    short_transfer(coal, goal, first_feasible)
    ref_cfg = SolverConfig(waypoints=14, dt=DT, margin=0.001, world_margin=0.014,
                           collision_substeps=5, iterations=50, first_feasible=False)
    reference = coal.plan(MotionPlanRequest(start=start, goal=goal, config=ref_cfg))
    if not reference.success:
        raise RuntimeError(reference.reason)

    cfg = SolverConfig(waypoints=14, dt=DT, margin=0.001, world_margin=0.014,
                       collision_substeps=substeps, iterations=50, first_feasible=first_feasible)
    mode = "sqp_fast" if first_feasible else "sqp"
    print(f"{mode}, {world.num_pairs} shape pairs, {substeps} substeps, Warp device {wp.get_device()}, "
          f"{repeats} timed runs", flush=True)
    payload_pairs = [i for i, (a, b) in enumerate(world.collision_pairs)
                     if a.startswith(f"{ROBOT}/{PAYLOAD[1]}/") and not b.startswith(f"{ROBOT}/")]
    u = (np.arange(SAMPLES_PER_SEGMENT, dtype=np.float64) / SAMPLES_PER_SEGMENT)[:, None]

    # Warmup Warp kernels.
    planners["mujoco_warp"].plan(MotionPlanRequest(start=start, goal=goal, waypoint_targets={6: reference.positions[6]}, config=cfg))

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
            print(f"B={size} {name}: median {median(times):.3f}s {np.round(times, 3)}, "
                  f"solved {int(result.success.sum())}/{size}, verified {verified}/{size}, "
                  f"min carton clearance {min(clearances, default=float('nan')):.2f} mm", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--collision-substeps", type=int, default=SAMPLES_PER_SEGMENT // 2)
    parser.add_argument("--solver", choices=("sqp", "sqp_fast"), default="sqp_fast")
    args = parser.parse_args()
    run(args.sizes, args.repeats, args.collision_substeps, args.solver == "sqp_fast")
