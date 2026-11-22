"""Coal's witness convention, Torch transforms and authored-mesh acceptance checks."""

from copy import deepcopy
from itertools import product
from pathlib import Path

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


@pytest.mark.parametrize("original", [False, True])
def test_segment_screening_and_rechecking_a_previous_collision(original):
    from gobot.planning._coal import CollisionModel, shape

    def sphere(name, body, x):
        spec = dict(type=2, radius=0.1, collision_layer=1, collision_mask=1)
        return shape(spec, ("", name), body, torch.tensor([x, 0, 0], dtype=DTYPE), torch.eye(3, dtype=DTYPE))

    plane = shape(dict(type=6, collision_layer=1, collision_mask=1), ("", "floor"), None,
                  torch.tensor([0, 0, -20], dtype=DTYPE), torch.eye(3, dtype=DTYPE))
    model = CollisionModel([sphere("tip", 0, 0), sphere("left", None, -2), sphere("right", None, 2), plane],
                           lambda a, b: True)
    margins = torch.tensor([0, 0.25, 0], dtype=DTYPE)
    p = torch.zeros((9, 1, 3), dtype=DTYPE)
    r = torch.eye(3, dtype=DTYPE).expand(9, 1, 3, 3)
    p[1, 0, 0], p[-1, 0, 0] = -2.0, 2.4
    frames = model.world_frames(p, r)
    for samples in (frames[:-1], frames):
        flat = model.nearby(samples, margins, original)
        grouped = model.nearby(samples, margins, original, group_size=4)
        for found, expected in zip(grouped, flat, strict=True):
            torch.testing.assert_close(found, expected)
        # The far halfspace stays a candidate at every pose.
        assert int((grouped[1] == 2).sum()) == len(samples)

    check = model.trajectory_check(margins, original)
    assert check(p, r, 4) == (1, model.names[0])
    assert check(p, r, 4) == (1, model.names[0])
    p[1, 0, 0] = 0
    assert check(p, r, 4) == (8, model.names[1])
    # After changing the grid, only the interior of the shorter path collides.
    p[2, 0, 0] = -2.05
    assert check(p[:5], r[:5], 2) == (2, model.names[0])
    p[2, 0, 0] = 0
    assert check(p[:5], r[:5], 2) is None


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
    assert not planner.validate_trajectory(path, dt=1, margin=0, collision_substeps=1,
                                           longest_valid_segment_length=0.2)
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
    from scipy.spatial.transform import Rotation

    pytest.importorskip("osqp")

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
        inspection = root.find("Inspection").position
    finally:
        context.clear_scene()
        if previous:
            context.set_project_path(previous)

    parts = [shape for link in snapshot["robots"][0]["links"] for shape in link["shapes"]]
    fixtures = [shape for shape in snapshot["loose_shapes"] if shape["type"] == 5]
    assert len(parts) == 11 and all(shape["type"] == 5 and shape["convex"] for shape in parts)
    assert len(fixtures) == 9 and all(shape["convex"] for shape in fixtures)
    solve = runpy.run_path(str(project / "scripts" / "plan.py"))["solve"]
    output = solve(snapshot, start, goal, inspection)
    assert set(output["seed_hits"].split(", ")) == {"CenterFixture", "EntryWedge", "ExitWedge", "LowerJaw"}
    inspection_frame = 8 * 24
    tip = output["links"].index("panda_hand")
    seed, optimized = output["poses"][:, inspection_frame, tip]
    assert np.linalg.norm(seed[:3] - inspection) > 0.05
    np.testing.assert_allclose(optimized[:3], inspection, rtol=0, atol=1e-4)
    expected = Rotation.from_euler("z", 25, degrees=True).as_matrix() @ np.diag([-1., 1., -1.])
    assert Rotation.from_matrix(expected.T @ Rotation.from_quat(optimized[3:]).as_matrix()).magnitude() < 1e-4
    assert output["world_clearance"] >= 0.02
    assert output["poses"].shape[:2] == (2, 361)


def test_container_unloading_retracts_the_payload_before_turning():
    from pathlib import Path
    import runpy
    import gobot
    from scipy.spatial.transform import Rotation

    pytest.importorskip("osqp")
    project = Path(__file__).resolve().parents[2] / "examples" / "container_unloading"
    context = gobot.app.context()
    previous = context.project_path
    try:
        context.set_project_path(str(project))
        context.load_scene("res://world.jscn")
        snapshot = context.compile_physics_scene_snapshot()
        pick, drop = (context.root.find(name) for name in ("Pick", "Drop"))
        targets = [(node.position, node.rotation_degrees) for node in (pick, drop)]
    finally:
        context.clear_scene()
        if previous:
            context.set_project_path(previous)

    # Check rotated links, the payload hull, and the original container meshes
    # against exhaustive narrow phase over a batch with both clear and colliding pairs.
    from gobot.planning._geometry import clearance_mask

    scene = Planner(snapshot, {"M10iA": "base_link"}, collision_backend="coal")._scene
    q = scene.initial + torch.randn((64, len(scene.initial)), dtype=scene.dtype,
                                    generator=torch.Generator().manual_seed(17)) * 0.65
    poses = scene.poses(q)
    margins = scene.pair_margins(0.001, 0.014)
    for original, models in ((False, scene.distance_models), (True, scene.validation_models)):
        model, = models
        exhaustive = clearance_mask(model.distances(*poses, verify=original), margins)
        assert exhaustive.any() and not exhaustive.all()
        assert torch.equal(model.clearance_mask(*poses, margins, original), exhaustive)

    output = runpy.run_path(str(project / "plan.py"))["solve"](snapshot, *targets)
    seed, optimized = output["poses"][:, :, output["names"].index("Parcel")]
    seed_distance, optimized_distance = output["carton_clearance"]
    assert seed_distance < -0.20 and "/Port3/" in output["nearest"][0]
    assert optimized_distance >= 0.014 - 1e-6
    np.testing.assert_allclose(seed[[0, -1]], optimized[[0, -1]], atol=1e-6)
    for pose, (position, angles) in zip(optimized[[0, -1]], targets, strict=True):
        np.testing.assert_allclose(pose[:3], position, atol=1e-4)
        expected = Rotation.from_euler("xyz", angles, degrees=True)
        assert (expected.inv() * Rotation.from_quat(pose[3:])).magnitude() < 1e-4
    # At the first 50 mm of lateral travel the carton center has left the rack.
    first_turn = np.flatnonzero(np.abs(optimized[:, 1] - optimized[0, 1]) > 0.05)[0]
    assert optimized[0, 0] - optimized[first_turn, 0] > 0.35
    settle, carry = output["handoff"]
    np.testing.assert_allclose(settle, [0, 0, -0.017], atol=1e-4)
    # The rollers take the released box through the doorway, toward world -X.
    np.testing.assert_allclose(carry, [-2.46, 0, 0], atol=1e-4)
    shapes = {s["name"]: s for s in snapshot["loose_shapes"]}
    assert {"PortWall_LiningCollider", "StarboardWall_LiningCollider", "ShippingContainer_RoofCollider",
            "LeftDoor_LeafCollider", "RightDoor_LeafCollider"} <= shapes.keys()
    threshold_x = np.asarray(shapes["DoorFrame_SillCollider"]["transform"]["matrix"])[0, 3]
    release_x = optimized[-1, 0]
    assert release_x + carry[0] < threshold_x < release_x


def test_mujoco_warp_convex_distance_and_rotation_jacobian():
    pytest.importorskip("mujoco_warp")
    from gobot.planning._coal import shape
    from gobot.planning._mujoco_warp import WarpPairs

    # Offset mesh origin and an off-axis rotating sphere exercise both witness frames.
    hull = np.asarray(list(product([-0.2, 0.2], [-0.3, 0.3], [-0.4, 0.4])))
    common = dict(collision_layer=1, collision_mask=1)
    box = shape(dict(common, type=5, vertices=hull + [0.05, 0, 0],
                     indices=[0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1,
                              2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]),
                ("robot", "hull"), 0, torch.zeros(3, dtype=DTYPE), torch.eye(3, dtype=DTYPE))
    ball = shape(dict(common, type=2, radius=0.18), ("robot", "sphere"), 1,
                 torch.zeros(3, dtype=DTYPE), torch.eye(3, dtype=DTYPE))
    cpu = native.Pairs([(box.geometry, ball.geometry), (ball.geometry, box.geometry)])
    warp = WarpPairs([box, ball], [(0, 1), (1, 0)])

    def distance(q, pairs):
        axis = q.new_tensor([0, 0, 1])
        rot = torch.stack((axis_rotation(axis, q[0]), axis_rotation(axis, q[1])))
        origin = torch.stack((q.new_tensor([0, 0.03, 0]), rot[1] @ q.new_tensor([0, 0.08, 0])
                              + torch.stack((q[2], q[2] * 0, q[2] * 0))))
        frames = torch.cat((rot, origin.unsqueeze(-1)), dim=-1)
        return _Distance.apply(torch.stack((frames, frames.flip(0))), pairs)[0]

    q = torch.tensor([[0.12, -0.07, 0.38], [-0.1, 0.18, 0.58]], dtype=DTYPE)
    value = vmap(lambda x: distance(x, warp))(q)
    assert value[0, 0] < 0 < value[1, 0]
    torch.testing.assert_close(value, vmap(lambda x: distance(x, cpu))(q), atol=2e-6, rtol=0)
    expected = vmap(jacfwd(lambda x: distance(x, cpu)))(q)
    torch.testing.assert_close(vmap(jacfwd(lambda x: distance(x, warp)))(q), expected, atol=2e-4, rtol=0)
    x = q[0].requires_grad_()
    gradient, = torch.autograd.grad(distance(x, warp)[0], x)
    torch.testing.assert_close(gradient, expected[0, 0], atol=2e-4, rtol=0)


def test_warp_distance_on_long_container_door_jamb():
    pytest.importorskip("mujoco_warp")
    import gobot

    context = gobot.app.context()
    previous = context.project_path
    try:
        context.set_project_path(str(Path(__file__).resolve().parents[2] / "examples/container_unloading"))
        context.load_scene("res://world.jscn")
        snapshot = context.compile_physics_scene_snapshot()
    finally:
        context.clear_scene()
        if previous:
            context.set_project_path(previous)
    coal, warp = [Planner(snapshot, {"M10iA": "base_link"}, collision_backend=backend)
                  for backend in ("coal", "mujoco_warp")]
    q = coal.initial_positions
    q[0] += 0.4
    q[1] += 0.2
    # A finite float32 simplex witness used to miss the link_4 / right jamb gap by 95 mm.
    torch.testing.assert_close(warp.collision_distances(q), coal.collision_distances(q), atol=1e-4, rtol=0)
