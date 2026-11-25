"""Plan the loaded part of the container unload against the editor snapshot."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from gobot.planning import MotionPlanner, MotionPlanRequest, PlanningWorld, PoseConstraint, SolverConfig


ROBOT = "M10iA"
PAYLOAD = (ROBOT, "Parcel")
DT = 0.23
SAMPLES_PER_SEGMENT = 24


def conveyor_motion(snapshot, drop_pose):
    """World translations to seat the carton and carry it to the last supported position."""
    deck = next(s for s in snapshot["loose_shapes"] if s["name"] == "RollerConveyor_RollerEnvelopeCollider")
    robot = next(r for r in snapshot["robots"] if r["name"] == ROBOT)
    parcel = next(link for link in robot["links"] if link["name"] == PAYLOAD[1])
    box = next(s for s in parcel["shapes"] if s["name"] == "Parcel_CartonCollider")
    deck_pose = np.asarray(deck["transform"]["matrix"])
    axes = deck_pose[:3, :3]
    center = axes.T @ (drop_pose[:3] - deck_pose[:3, 3])
    carton_axes = axes.T @ Rotation.from_quat(drop_pose[3:]).as_matrix()
    half_carton = np.abs(carton_axes) @ (np.asarray(box["size"]) / 2)
    half_deck = np.asarray(deck["size"]) / 2
    gap = center[2] - half_carton[2] - half_deck[2]
    if (np.any(np.abs(center[:2]) + half_carton[:2] > half_deck[:2])
            or abs(carton_axes[2, 2] - 1) > 1e-4 or not 0 <= gap <= 0.05):
        return np.empty((0, 3))
    travel = center[1] + half_deck[1] - half_carton[1]
    return np.stack((-gap * axes[:, 2], -travel * axes[:, 1]))


def solve(snapshot, pick, drop):
    world = PlanningWorld.from_snapshot(snapshot, {ROBOT: "base_link"})
    planner = MotionPlanner(world, backend="coal")
    ends = []
    for label, (position, angles) in (("Pick", pick), ("Drop", drop)):
        rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        yaw = np.arctan2(position[1], position[0])
        seed = [yaw, 0.12, -0.50, 0.0, 0.65, 0.0]
        result = planner.solve_ik(
            [PoseConstraint(PAYLOAD, position, rotation)],
            seed=seed, margin=0.001, world_margin=0.014,
        )
        if not result.success:
            raise ValueError(f"{label} pose failed IK: {result.reason}")
        ends.append(result.positions)

    cfg = SolverConfig(
        waypoints=14, dt=DT, margin=0.001, world_margin=0.014,
        collision_substeps=SAMPLES_PER_SEGMENT // 2, iterations=50, first_feasible=True,
    )
    result = planner.plan(MotionPlanRequest(start=ends[0], goal=ends[1], config=cfg))
    if not result.success:
        raise ValueError(result.reason)

    paths = np.stack((result.initial_positions, result.positions), axis=0)
    u = np.arange(SAMPLES_PER_SEGMENT, dtype=np.float64)[:, None] / SAMPLES_PER_SEGMENT
    between = (1.0 - u)[None, None, :, :] * paths[:, :-1, None, :] + u[None, None, :, :] * paths[:, 1:, None, :]
    samples = np.concatenate((between.reshape(2, -1, world.num_joints), paths[:, -1:]), axis=1)
    payload_pairs = [i for i, (a, b) in enumerate(world.collision_pairs)
                     if a.startswith(f"{ROBOT}/{PAYLOAD[1]}/") and not b.startswith(f"{ROBOT}/")]
    distances = world.collision_distances(samples)[..., payload_pairs]
    min_by_pair = distances.min(axis=1)
    nearest = [world.collision_pairs[payload_pairs[int(i)]][1] for i in min_by_pair.argmin(axis=1)]

    names = ["base_link", *(f"link_{i}" for i in range(1, 7)), PAYLOAD[1]]
    tracks = []
    for name in names:
        pos, rot = world.link_pose(samples, (ROBOT, name))
        quat = Rotation.from_matrix(rot.reshape(-1, 3, 3)).as_quat().reshape(*pos.shape[:-1], 4)
        tracks.append(np.concatenate((pos, quat), axis=-1))
    poses = np.stack(tracks, axis=2).astype(np.float32)
    return dict(names=names, poses=poses, handoff=conveyor_motion(snapshot, poses[1, -1, -1]),
                dt=DT / SAMPLES_PER_SEGMENT, joints=paths,
                carton_clearance=min_by_pair.min(axis=1), nearest=nearest)


if __name__ == "__main__":
    request, output = map(Path, sys.argv[1:])
    try:
        response = solve(**json.loads(request.read_text()))
    except Exception as error:
        response = {"error": str(error)}
    np.savez_compressed(output, **response)
