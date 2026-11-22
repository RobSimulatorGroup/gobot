"""Compare independent pick/via/drop trajectories on the same FANUC snapshot."""

import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

import gobot
from gobot.planning import Planner, PlannerBatch, PoseTarget
from gobot.planning._geometry import clearance_mask
from plan import DT, PAYLOAD, ROBOT, SAMPLES_PER_SEGMENT


def endpoint(planner, marker):
    xyz = marker.position
    rotation = Rotation.from_euler("xyz", marker.rotation_degrees, degrees=True).as_matrix()
    seed = [np.arctan2(xyz[1], xyz[0]), 0.12, -0.50, 0, 0.65, 0]
    result = planner.solve_ik([PoseTarget(PAYLOAD, xyz, rotation)], seed=seed, margin=0.001, world_margin=0.014)
    if not result.success:
        raise RuntimeError(result.reason)
    return result.positions


def verify(coal, path, samples, payload_pairs):
    """Independently query every convex and authored pair, outside the planner timer."""
    scene = coal._scene
    p, r = scene.poses(samples)
    distances = scene.pose_distances(p, r)
    margins = scene.pair_margins(0.001, 0.014)
    valid = clearance_mask(distances, margins).all().item()
    for model in scene.validation_models:
        original = model.distances(p, r, verify=True)
        threshold = original.new_full((len(model.names),), 0.014)
        threshold[:model.self_count] = 0.001
        valid &= clearance_mask(original, threshold).all().item()
    valid &= coal.validate_trajectory(path, dt=DT, margin=0.001, world_margin=0.014,
                                      collision_substeps=SAMPLES_PER_SEGMENT)
    return valid, float(distances[:, payload_pairs].min()) * 1000


def short_transfer(planner, start):
    goal = start.clone()
    goal[0] -= 0.1
    options = dict(dt=DT, waypoints=14, solver="sqp", margin=0.001, world_margin=0.014,
                   collision_substeps=5, iterations=50)
    times = []
    for _ in range(12):
        begin = perf_counter()
        result = planner.solve_trajopt(start, goal, **options)
        times.append(perf_counter() - begin)
        if not result.success or not torch.equal(result.positions, result.initial_positions):
            raise RuntimeError(f"short transfer failed to accept the straight path: {result.reason}")
    print(f"Short transfer: {1000 * median(times[1:]):.2f} ms median (11 warm runs)", flush=True)


def run(sizes, repeats, substeps):
    torch.set_num_threads(1)
    context = gobot.app.context()
    context.set_project_path(str(Path(__file__).resolve().parent))
    context.load_scene("res://world.jscn")
    snapshot = context.compile_physics_scene_snapshot()
    planners = {name: Planner(snapshot, {ROBOT: "base_link"}, collision_backend=name)
                for name in ("coal", "mujoco_warp")}
    coal = planners["coal"]
    start, goal = [endpoint(coal, context.root.find(name)) for name in ("Pick", "Drop")]
    context.clear_scene()
    options = dict(dt=DT, waypoints=14, margin=0.001, world_margin=0.014, collision_substeps=5, iterations=50)
    short_transfer(coal, goal)
    reference = coal.solve_trajopt(start, goal, solver="sqp", **options)
    if not reference.success:
        raise RuntimeError(reference.reason)
    options["collision_substeps"] = substeps
    print(f"{len(coal.collision_pairs)} shape pairs, {substeps} substeps, Warp device {wp.get_device()}, "
          f"{repeats} timed runs", flush=True)
    payload_pairs = [i for i, (a, b) in enumerate(coal.collision_pairs) if a == PAYLOAD and not b[0]]
    fractions = torch.arange(SAMPLES_PER_SEGMENT, dtype=start.dtype)[:, None] / SAMPLES_PER_SEGMENT
    for size in sizes:
        # Keep task zero identical across sizes. Vary pickup, joint via point and drop for every other row.
        offsets = torch.arange(size, dtype=start.dtype) / max(size - 1, 1)
        starts, goals = start.repeat(size, 1), goal.repeat(size, 1)
        starts[:, 0] += 0.003 * offsets
        goals[:, 0] -= 0.015 * offsets
        via = reference.positions[6].repeat(size, 1)
        via[:, 0] += 0.006 * offsets
        targets = [{6: q} for q in via]
        # Time the same colliding seed configurations on both backends, including FK and transfers.
        pick_fractions = torch.linspace(0, 1, 7, dtype=start.dtype)[:-1, None]
        drop_fractions = torch.linspace(0, 1, 8, dtype=start.dtype)[:, None]
        seed = torch.cat((torch.lerp(starts[:, None], via[:, None], pick_fractions),
                          torch.lerp(via[:, None], goals[:, None], drop_fractions)), dim=1)
        between = torch.lerp(seed[:, :-1, None], seed[:, 1:, None], fractions).flatten(1, 2)
        seed_samples = torch.cat((between, goals[:, None]), dim=1)
        reference_distances = coal.collision_distances(seed_samples)
        for name, planner in planners.items():
            batch = PlannerBatch(planner)
            planner.collision_distances(starts)
            query_times = []
            for _ in range(3):
                begin = perf_counter()
                distances = planner.collision_distances(seed_samples)
                query_times.append(perf_counter() - begin)
            max_error = (distances - reference_distances).abs().max().item()
            print(f"B={size} {name}: {size * seed_samples.shape[1]} identical configurations (FK + distances) "
                  f"in {median(query_times):.3f}s, max deviation from Coal {1000 * max_error:.3f} mm", flush=True)
            times = []
            for _ in range(repeats):
                begin = perf_counter()
                result = batch.solve_trajopt(starts, goals, waypoint_targets=targets, **options)
                times.append(perf_counter() - begin)
            clearances, verified = [], 0
            for row, path in enumerate(result.positions):
                if not result.success[row]:
                    print(f"  row {row}: {result.reasons[row]}")
                    continue
                between = torch.lerp(path[:-1, None], path[1:, None], fractions).flatten(0, 1)
                samples = torch.cat((between, path[-1:]))
                valid, clearance = verify(coal, path, samples, payload_pairs)
                clearances.append(clearance)
                verified += valid
            print(f"B={size} {name}: median {median(times):.3f}s {np.round(times, 3)}, "
                  f"solved {int(result.success.sum())}/{size}, verified {verified}/{size}, "
                  f"min carton clearance {min(clearances, default=float('nan')):.2f} mm", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--collision-substeps", type=int, default=SAMPLES_PER_SEGMENT // 2)
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.repeats < 1 or args.collision_substeps < 1:
        parser.error("sizes, repeats and collision substeps must be positive")
    run(args.sizes, args.repeats, args.collision_substeps)
