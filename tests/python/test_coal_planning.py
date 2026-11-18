"""Coal's witness convention, Torch transforms and authored-mesh acceptance checks."""

from copy import deepcopy
from itertools import product

import numpy as np
import pytest
import torch
from torch.func import jacfwd, vmap

from gobot._kinematics import axis_rotation
from gobot.planning import Capsule, Planner, PlannerBatch, PoseTarget
from panda_planning import panda

native = pytest.importorskip("gobot._coal_native", exc_type=ModuleNotFoundError)
from gobot.planning._coal import _Distance

DTYPE = torch.float64


@pytest.mark.parametrize("separation", [1.0, 0.6])
def test_signed_sphere_witnesses(separation):
    a, b = [native.Shape(str(radius), dict(type=2, radius=radius)) for radius in (0.5, 0.3)]
    frames = np.tile(np.eye(4)[:3], (1, 1, 2, 1, 1))
    frames[0, 0, 1, 0, 3] = separation
    forward = native.Pairs([(a, b)]).query(frames, False)[0, 0]
    reverse = native.Pairs([(b, a)]).query(frames[:, :, ::-1], False)[0, 0]
    np.testing.assert_allclose(forward, [separation - 0.8, 0.5, 0, 0, -0.3, 0, 0, -1, 0, 0])
    np.testing.assert_allclose(reverse, np.r_[forward[0], forward[4:7], forward[1:4], -forward[7:]])


@pytest.mark.parametrize("bx", [0.9, 0.46])
def test_two_rotating_bodies_forward_reverse_and_batched_queries(bx):
    box = native.Shape("offset box", dict(type=1, size=np.array([0.4, 0.6, 0.8])))
    sphere = native.Shape("offset sphere", dict(type=2, radius=0.3))
    pairs = native.Pairs([(box, sphere)])

    class Counter:
        calls = 0

        def query(self, frames, verify, clearance):
            self.calls += 1
            return pairs.query(frames, verify, clearance)

    counter = Counter()
    def distance(q):
        axis = q.new_tensor([0, 0, 1])
        ra, rb = axis_rotation(axis, q[0]), axis_rotation(axis, q[1])
        pa = q.new_tensor([0.1, 0.12, 0]) + ra @ q.new_tensor([0.01, 0.02, 0])
        pb = q.new_tensor([bx, 0.20, 0.03]) + rb @ q.new_tensor([0.01, 0.08, 0])
        frames = torch.cat((torch.stack((ra, rb)), torch.stack((pa, pb)).unsqueeze(-1)), -1)
        return _Distance.apply(frames.unsqueeze(0), counter)[0].squeeze(0)

    q = torch.tensor([0.22, -0.15], dtype=DTYPE, requires_grad=True)
    batch = torch.stack((q, q + 0.01))
    with torch.no_grad():
        batched_jacobian = vmap(jacfwd(distance))(batch)
    assert counter.calls == 1
    value = distance(q)
    assert (value > 0).item() == (bx > 0.5)
    reverse, = torch.autograd.grad(value, q)
    finite = torch.stack([(distance(q + h) - distance(q - h)) / 2e-6 for h in torch.eye(2, dtype=DTYPE) * 1e-6])
    torch.testing.assert_close(reverse, finite, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(batched_jacobian, torch.stack([jacfwd(distance)(row) for row in batch]))
    torch.testing.assert_close(batched_jacobian[0], reverse)


def test_fingertip_original_mesh_rejects_clear_proxy(panda):
    snapshot = deepcopy(panda)
    base = {"panda": "panda_link0"}
    hand = ("panda", "panda_hand")
    links = snapshot["robots"][0]["links"]
    tip = next(link for link in links if link["name"] == hand[1])
    mesh = tip["shapes"][0]
    for link in links:
        link["shapes"] = []
    tip["shapes"] = [mesh]
    # A thin rectangular finger extending along its own z axis.
    vertices = np.array(list(product([-0.009, 0.009], [-0.01, 0.01], [0.05, 0.15])))
    mesh.update(type=5, name="finger", path="finger", vertices=vertices,
                indices=[0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1,
                         2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3],
                transform=deepcopy(tip["global_transform"]))
    q = torch.tensor([0., -0.5, 0., -2., 0., 1.5, 0.8], dtype=DTYPE)
    start, end = q.clone(), q.clone()
    start[0], end[0] = -0.4, 0.4
    p, r = Planner(snapshot, base).link_pose(q, hand)
    obstacle = np.eye(4)
    obstacle[:3, 3] = p + r @ q.new_tensor([0, 0, 0.1])
    snapshot["loose_shapes"] = [dict(mesh, type=1, path="thin_wall", size=[0.05, 0.002, 0.05],
                                     transform={"matrix": obstacle})]

    exact = Planner(snapshot, base, collision_backend="coal")
    assert exact.collision_distances(q).item() < 0
    assert (exact.collision_distances(torch.stack((start, end))) > 0.02).all()
    proxy = Capsule((0, 0, 0), (0, 0, 0), 0.005)
    planner = Planner(snapshot, base, link_capsules={hand: [proxy]}, collision_backend="coal")
    assert planner.collision_distances(q).item() > 0.02
    path = torch.stack((start, end))
    assert planner.validate_trajectory(path, dt=1, margin=0, collision_substeps=1)
    assert not planner.validate_trajectory(path, dt=1, margin=0, collision_substeps=2)
    result = planner.solve_trajopt(start, end, waypoints=2, dt=1, margin=0, collision_substeps=1)
    assert not result.success
    assert all(word in result.reason for word in ("segment 0", "finger", "thin_wall"))
    samples = torch.stack((start, q))
    positions, rotations = planner.link_pose(samples, hand)
    ik = PlannerBatch(planner).solve_ik([PoseTarget(hand, positions, rotations)], seed=samples, iterations=1)
    assert ik.success.tolist() == [True, False]
    assert ik.reasons[0] is None and "finger" in ik.reasons[1] and "thin_wall" in ik.reasons[1]


def test_two_robot_fk_distance_gradient(panda):
    snapshot = deepcopy(panda)
    left = snapshot["robots"][0]
    right = deepcopy(left)
    right["name"] = "right"
    snapshot["robots"].append(right)
    for robot in (left, right):
        for link in robot["links"]:
            if link["name"] != "panda_hand":
                link["shapes"] = []
    planner = Planner(snapshot, {left["name"]: "panda_link0", "right": "panda_link0"}, collision_backend="coal")
    seed = torch.tensor([0., -0.5, 0., -2., 0., 1.5, 0.8], dtype=DTYPE)
    q = torch.cat((seed, seed + seed.new_tensor([0.15, 0.03, 0, 0, -0.02, 0.01, 0]))).requires_grad_()
    jac, = jacfwd(planner.collision_distances)(q)
    reverse, = torch.autograd.grad(planner.collision_distances(q).sum(), q)
    perturbations = 1e-6 * torch.eye(len(q), dtype=q.dtype)
    finite = (planner.collision_distances(q + perturbations) - planner.collision_distances(q - perturbations)) / 2e-6
    assert jac[:7].norm() > 0.1 and jac[7:].norm() > 0.1
    torch.testing.assert_close(jac, reverse)
    torch.testing.assert_close(jac, finite[:, 0], atol=1e-7, rtol=1e-6)


def test_planar_mesh_keeps_zero_thickness(panda):
    snapshot = deepcopy(panda)
    robot = snapshot["robots"][0]
    hand = next(link for link in robot["links"] if link["name"] == "panda_hand")
    sphere = hand["shapes"][0]
    for link in robot["links"]:
        if link is not hand:
            link["shapes"] = []
    seed = torch.tensor([0., -0.5, 0., -2., 0., 1.5, 0.8], dtype=DTYPE)
    planner = Planner(snapshot, {"panda": "panda_link0"})
    origin, _ = planner.link_pose(seed, ("panda", "panda_hand"))
    plane = np.eye(4)
    plane[:3, 3] = origin - seed.new_tensor([0, 0, 0.1])
    snapshot["loose_shapes"] = [dict(sphere, type=5, path="sheet", transform={"matrix": plane},
        vertices=[[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], indices=[0, 1, 2, 0, 2, 3])]
    planner = Planner(snapshot, {"panda": "panda_link0"}, collision_backend="coal")
    expected = 0.1 - sphere["radius"]
    assert planner.collision_distances(seed).item() == pytest.approx(expected, abs=1e-6)
    assert planner.validate_trajectory(torch.stack((seed, seed)), dt=1, margin=expected - 1e-5)


def test_editor_workcell_optimizes_between_convex_fixtures():
    from pathlib import Path
    import runpy
    import gobot

    project = Path(__file__).resolve().parents[2] / "examples" / "panda_trajopt"
    context = gobot.app.context()
    previous = context.project_path
    try:
        context.set_project_path(str(project))
        context.load_scene("res://world.jscn")
        snapshot = context.compile_physics_scene_snapshot()
        root = context.root
        start = root.find("Start").position
        goal = root.find("Goal").position
    finally:
        context.clear_scene()
        if previous:
            context.set_project_path(previous)

    parts = [shape for link in snapshot["robots"][0]["links"] for shape in link["shapes"]]
    fixtures = [shape for shape in snapshot["loose_shapes"] if shape["type"] == 5]
    assert len(parts) == 11 and all(shape["type"] == 5 and shape["convex"] for shape in parts)
    assert len(fixtures) == 9 and all(shape["convex"] for shape in fixtures)
    solve = runpy.run_path(str(project / "scripts" / "plan.py"))["solve"]
    output = solve(snapshot, start, goal)
    assert set(output["seed_hits"].split(", ")) == {"CenterFixture", "EntryWedge", "ExitWedge"}
    assert output["world_clearance"] >= 0.02
    assert output["poses"].shape[:2] == (2, 361)
