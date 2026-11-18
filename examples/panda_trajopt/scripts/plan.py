"""Plan the Panda motion from a Gobot snapshot in a separate CPU process."""

import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from gobot.planning import Planner, PoseTarget


def solve(snapshot, start, goal):
    # These shells meet at a shared joint center (links 1 and 5 have no
    # translation between their axes), or at the fixed wrist-hand mounting.
    mounts = (("panda_link0", "panda_link2"), ("panda_link4", "panda_link6"),
              ("panda_link7", "panda_hand"))
    planner = Planner(snapshot, {"panda": "panda_link0"}, collision_backend="coal",
                      ignored_link_pairs=[(("panda", a), ("panda", b)) for a, b in mounts])
    hand = ("panda", "panda_hand")
    down = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
    seed = [0, -0.5, 0, -2, 0, 1.5, -2.3]
    endpoints = []
    for name, position in (("Start", start), ("Goal", goal)):
        ik = planner.solve_ik([PoseTarget(hand, position, down)], seed=seed, margin=0.002, world_margin=0.025)
        if not ik.success:
            raise ValueError(f"{name} IK failed: {ik.reason}")
        seed = ik.positions
        endpoints.append(seed)
    plan = planner.solve_trajopt(*endpoints, waypoints=16, dt=0.14, margin=0.002, world_margin=0.025,
                                 collision_substeps=2, stop_at_endpoints=True)
    if not plan.success:
        raise ValueError(plan.reason)
    # Reserve 5 mm between the optimization target and the finer 20 mm acceptance threshold.
    if not planner.validate_trajectory(plan.positions, dt=plan.dt, collision_substeps=24,
                                       margin=0.002, world_margin=0.020):
        raise ValueError("Original geometry failed the 24-substep trajectory clearance check")
    # Both tracks share the same waypoint clock. Interpolate joints before FK.
    steps = 24
    q = torch.stack((plan.initial_positions, plan.positions))
    fraction = torch.arange(steps, dtype=q.dtype)[:, None] / steps
    between = torch.lerp(q[:, :-1, None], q[:, 1:, None], fraction)
    samples = torch.cat((between.flatten(1, 2), q[:, -1:]), dim=1)
    world = [i for i, (_, other) in enumerate(planner.collision_pairs) if not other[0]]
    clearance = planner.collision_distances(samples)[:, :, world].amin(dim=1)
    hits = sorted({planner.collision_pairs[world[i]][1][1].split("/")[-2]
                   for i in (clearance[0] < 0).nonzero().flatten().tolist()})
    names = [f"panda_link{i}" for i in range(9)] + [hand[1]]
    poses = []
    for name in names:
        position, rotation = planner.link_pose(samples, ("panda", name))
        quaternion = Rotation.from_matrix(rotation.numpy().reshape(-1, 3, 3)).as_quat()
        quaternion = quaternion.reshape(*position.shape[:-1], 4)
        poses.append(np.concatenate((position.numpy(), quaternion), axis=-1))
    # tracks × frames × links × (xyz, xyzw)
    poses = np.stack(poses, axis=2).astype(np.float32)
    return dict(links=names, poses=poses, dt=plan.dt / steps,
                seed_hits=", ".join(hits), world_clearance=clearance[1].min().item(),
                paths=poses[:, ::steps, names.index(hand[1]), :3])


if __name__ == "__main__":
    # These small CPU linear systems run more efficiently on a single thread.
    torch.set_num_threads(1)
    request, result = map(Path, sys.argv[1:])
    try:
        response = solve(**json.loads(request.read_text()))
    except Exception as error:
        response = {"error": str(error)}
    np.savez(result, **response)
