"""Plan the Panda motion from a Gobot snapshot in a separate CPU process."""

import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from gobot.planning import Planner, PoseTarget


def solve(snapshot, start, goal):
    planner = Planner(snapshot, {"panda": "panda_link0"})
    hand = ("panda", "panda_hand")
    down = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
    plan = planner.plan_poses(
        [PoseTarget(hand, start, down)], [PoseTarget(hand, goal, down)],
        start_seed=[0, -0.5, 0, -2, 0, 1.5, -2.3],
        waypoints=50, dt=0.02, collision_substeps=2,
    )
    if not plan.success:
        raise ValueError(plan.reason)
    # Both tracks share the same waypoint clock. Interpolate joints before FK.
    steps = 5
    q = torch.stack((plan.initial_positions, plan.positions))
    fraction = torch.arange(steps, dtype=q.dtype)[:, None] / steps
    between = torch.lerp(q[:, :-1, None], q[:, 1:, None], fraction)
    samples = torch.cat((between.flatten(1, 2), q[:, -1:]), dim=1)
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
