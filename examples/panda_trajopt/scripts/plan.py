"""Plan the Panda motion from a Gobot snapshot in a separate CPU process."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from gobot.planning import MotionPlanner, MotionPlanRequest, PlanningWorld, PoseConstraint, SolverConfig


def solve(snapshot, start, goal, inspection):
    mounts = (("panda_link0", "panda_link2"), ("panda_link4", "panda_link6"),
              ("panda_link7", "panda_hand"))
    world = PlanningWorld.from_snapshot(
        snapshot, {"panda": "panda_link0"},
        ignored_link_pairs=[(("panda", a), ("panda", b)) for a, b in mounts],
    )
    planner = MotionPlanner(world, backend="coal")
    hand = ("panda", "panda_hand")
    down = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
    seed = np.array([0, -0.5, 0, -2, 0, 1.5, -2.3], dtype=np.float64)
    endpoints = []
    for name, position in (("Start", start), ("Goal", goal)):
        ik = planner.solve_ik(
            [PoseConstraint(hand, position, down)],
            seed=seed, margin=0.002, world_margin=0.025,
        )
        if not ik.success:
            raise ValueError(f"{name} IK failed: {ik.reason}")
        seed = ik.positions
        endpoints.append(seed)

    inspection_index = 8
    inspection_rotation = Rotation.from_euler("z", 25, degrees=True).as_matrix() @ down
    inspection_target = PoseConstraint(hand, inspection, inspection_rotation,
                                       position_tolerance=1e-4, rotation_tolerance=1e-4)
    cfg = SolverConfig(
        waypoints=16, dt=0.14, margin=0.002, world_margin=0.025,
        collision_substeps=3, max_segment_length=0.075,
        stop_at_endpoints=True, iterations=40, first_feasible=False,
    )
    plan = planner.plan(MotionPlanRequest(
        start=endpoints[0], goal=endpoints[1],
        waypoint_targets={inspection_index: inspection_target}, config=cfg,
    ))
    if not plan.success:
        raise ValueError(plan.reason)
    valid, reason = world.validate_trajectory(
        plan.positions, dt=plan.dt, collision_substeps=24, margin=0.002, world_margin=0.020,
    )
    if not valid:
        raise ValueError(f"Original geometry failed 24-substep check: {reason}")
    position, rotation = world.link_pose(plan.positions[inspection_index], hand)
    position_error = float(np.linalg.norm(position - inspection))
    angle_error = float(Rotation.from_matrix(inspection_rotation.T @ rotation).magnitude())

    steps = 24
    q = np.stack((plan.initial_positions, plan.positions), axis=0)
    fraction = (np.arange(steps, dtype=np.float64) / steps)[:, None]
    between = (1.0 - fraction)[None, None, :, :] * q[:, :-1, None, :] + fraction[None, None, :, :] * q[:, 1:, None, :]
    samples = np.concatenate((between.reshape(2, -1, world.num_joints), q[:, -1:]), axis=1)
    world_pairs = [i for i, (_, other) in enumerate(world.collision_pairs) if not other.startswith("panda/")]
    clearance = world.collision_distances(samples)[:, :, world_pairs].min(axis=1)
    hits = sorted({world.collision_pairs[world_pairs[i]][1].split("/")[-2]
                   for i in np.flatnonzero(clearance[0] < 0)})
    names = [f"panda_link{i}" for i in range(9)] + [hand[1]]
    poses = []
    for name in names:
        pos, rot = world.link_pose(samples, ("panda", name))
        quat = Rotation.from_matrix(rot.reshape(-1, 3, 3)).as_quat().reshape(*pos.shape[:-1], 4)
        poses.append(np.concatenate((pos, quat), axis=-1))
    poses = np.stack(poses, axis=2).astype(np.float32)
    return dict(links=names, poses=poses, dt=plan.dt / steps,
                inspection_error=position_error, inspection_angle=angle_error,
                seed_hits=", ".join(hits), world_clearance=float(clearance[1].min()),
                paths=poses[:, ::steps, names.index(hand[1]), :3])


if __name__ == "__main__":
    request, result = map(Path, sys.argv[1:])
    try:
        response = solve(**json.loads(request.read_text()))
    except Exception as error:
        response = {"error": str(error)}
    np.savez(result, **response)
