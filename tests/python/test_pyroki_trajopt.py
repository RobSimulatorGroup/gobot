"""Torch trajectory regressions based on PyRoki's 07_trajopt Panda wall example."""

from copy import deepcopy

import mujoco
import numpy as np
import pytest
import torch

from gobot.planning import Capsule, PoseTarget
from gobot.planning._trajectory import acceleration_stencil
from panda_planning import PANDA_URDF, check_pose, make_planner, panda


@pytest.mark.parametrize("dt", [0.02, 0.15])
def test_five_point_acceleration(dt):
    t = torch.arange(9, dtype=torch.float64) * dt
    matrix = acceleration_stencil(torch.eye(len(t), dtype=t.dtype), dt)
    torch.testing.assert_close(matrix @ t**4, 12 * t[2:-2]**2, atol=1e-10, rtol=1e-10)


def test_panda_over_wall(panda, make_planner):
    snapshot = deepcopy(panda)
    root_sphere = snapshot["robots"][0]["links"][0]["shapes"][0]
    wall_center = np.array([0.5, 0., 0.2])
    wall_size = np.array([0.4, 0.1, 0.4])
    frame = np.eye(4)
    frame[:3, 3] = wall_center
    wall = dict(root_sphere, type=1, name="wall", path="wall", transform={"matrix": frame}, size=wall_size)
    ground = dict(root_sphere, type=6, name="ground", path="ground", transform={"matrix": np.eye(4)},
                  collision_layer=4, collision_mask=1)
    # The base is bolted to the ground. All moving proxies interact with the ground.
    root_sphere["collision_layer"] = 2
    snapshot["loose_shapes"] = [wall, ground]
    planner = make_planner(snapshot, {"panda": "panda_link0"})
    link = ("panda", "panda_hand")
    down = torch.diag(torch.tensor([-1., 1., -1.], dtype=torch.float64))
    start = PoseTarget(link, [0.5, -0.35, 0.2], down)
    goal = PoseTarget(link, [0.5, 0.35, 0.2], down)
    seed = [0., -0.5, 0., -2., 0., 1.5, -2.3]
    dt = 0.02
    plan = planner.plan_poses([start], [goal], dt=dt, waypoints=50, start_seed=seed, collision_substeps=2)
    assert plan.success, plan.reason
    q = plan.positions
    assert q.shape == (50, 7)
    torch.testing.assert_close(q[0], q[1], atol=0, rtol=0)
    torch.testing.assert_close(q[-1], q[-2], atol=0, rtol=0)
    initial = plan.initial_positions
    fractions = torch.linspace(0, 1, len(q) - 2, dtype=q.dtype)[:, None]
    torch.testing.assert_close(initial[1:-1], torch.lerp(q[0], q[-1], fractions))
    torch.testing.assert_close(initial[[0, -1]], q[[0, -1]])
    wall_pairs = [i for i, (_, obstacle) in enumerate(planner.collision_pairs) if obstacle == ("", "wall")]
    assert planner.collision_distances(initial)[:, wall_pairs].min() < 0
    reference = mujoco.MjModel.from_xml_path(str(PANDA_URDF))
    for joints, target in ((q[0], start), (q[-1], goal)):
        check_pose(reference, planner, joints, link, q.new_tensor(target.position), down)

    # Recover proxy centers independently from MuJoCo FK and compute their
    # distances to the box and floor at 80 samples per waypoint interval.
    data = mujoco.MjData(reference)
    dense = torch.lerp(q[:-1, None], q[1:, None], torch.arange(80, dtype=q.dtype)[:, None] / 80)
    centers = []
    for sample in dense.flatten(0, 1).numpy():
        data.qpos[:] = sample
        mujoco.mj_kinematics(reference, data)
        centers.append(data.geom_xpos.copy())
    centers = np.stack(centers)
    faces = np.abs(centers - wall_center) - wall_size * 0.5
    clearance = np.linalg.norm(np.maximum(faces, 0), axis=-1) + np.minimum(faces.max(axis=-1), 0)
    radii = reference.geom_size[:, 0]
    assert (clearance - radii).min() >= 0.05 - 1e-6
    moving = reference.geom_bodyid != reference.body("panda_link0").id
    assert (centers[:, moving, 2] - radii[moving]).min() >= 0.05 - 1e-6
    vmax = q.new_tensor([j["velocity_limit"] for j in snapshot["robots"][0]["joints"] if j["type"] == 1])
    assert (q.diff(dim=0).abs() <= dt * vmax * (1 + 1e-6)).all()


def test_capsule_sweep_catches_obstacle_between_samples(panda, make_planner):
    snapshot = deepcopy(panda)
    joints = torch.tensor([[0., -0.5, 0., -2., 0., 1.5, 0.8],
                           [0.9, -0.5, 0., -2., 0., 1.5, 0.8]], dtype=torch.float64)
    link = ("panda", "panda_hand")
    base_planner = make_planner(snapshot, {"panda": "panda_link0"})
    origins, _ = base_planner.link_pose(joints, link)
    frame = np.eye(4)
    frame[:3, 3] = origins.mean(dim=0).numpy()
    obstacle = dict(snapshot["robots"][0]["links"][0]["shapes"][0], name="post", path="post",
                    radius=0.005, transform={"matrix": frame})
    snapshot["loose_shapes"] = [obstacle]
    # Only the middle of this capsule's sweep hits the post.
    proxies = {("panda", "panda_link0"): [], ("panda", "panda_link4"): [],
               link: [Capsule((0, 0, -0.2), (0, 0, 0.2), 0.01)]}
    planner = make_planner(snapshot, {"panda": "panda_link0"}, link_capsules=proxies)
    assert (planner.collision_distances(joints) > 0.02).all()
    assert not planner.validate_trajectory(joints, dt=1., collision_substeps=1)
    assert not planner.solve_trajopt(*joints, dt=1., waypoints=2, collision_substeps=1).success
