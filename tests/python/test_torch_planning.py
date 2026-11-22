"""Exercise the public Torch planner against Gobot physics snapshots."""

from copy import deepcopy
import math

import pytest
import torch

import gobot

from gobot.planning._geometry import box_distance, capsule_distance, rotation_log


@pytest.fixture
def snapshot(tmp_path):
    source = tmp_path / "planning.xml"
    source.write_text("""
    <mujoco model="planning">
      <compiler angle="radian"/>
      <worldbody>
        <geom name="obstacle" type="box" pos="0 -0.06 0.5" size="0.16 0.15 0.15"/>
        <geom name="ground" type="plane" size="2 2 0.05"/>
        <body name="base" pos="0 0 0.5">
          <body name="left_x">
            <inertial pos="0 0 0" mass="0.1" diaginertia="0.01 0.01 0.01"/>
            <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
            <body name="left_tip">
              <joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
              <geom type="sphere" size="0.045"/>
            </body>
          </body>
          <body name="right_arm" pos="0 0.7 0">
            <inertial pos="0 0 0" mass="0.1" diaginertia="0.01 0.01 0.01"/>
            <joint name="hinge" axis="0 0 1"/>
            <body name="right_tip" pos="0.25 0 0">
              <geom type="capsule" size="0.035 0.04"/>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """)
    gobot.set_project_path(str(tmp_path))
    gobot.import_mjcf_scene(str(source), "res://planning.jscn", name="planning")
    context = gobot.app.context()
    context.set_project_path(str(tmp_path))
    context.load_scene("res://planning.jscn")
    result = context.compile_physics_scene_snapshot()
    # MJCF has no per-joint angular velocity-limit field.
    for robot in result["robots"]:
        for joint in robot["joints"]:
            if joint["name"] == "x":
                joint["velocity_limit"] = 2.0
    yield result
    context.clear_scene()


def test_pose_ik_with_two_targets_and_scene_collision(snapshot):
    planner = gobot.planning.Planner(snapshot, {"planning": "base"})
    assert planner.joint_names == (("planning", "x"), ("planning", "y"), ("planning", "hinge"))
    left, right = ("planning", "left_tip"), ("planning", "right_tip")
    q = torch.tensor([0.62, 0.2, 2.9], dtype=torch.float64)
    left_position, _ = planner.link_pose(q, left)
    right_position, right_rotation = planner.link_pose(q, right)
    targets = (gobot.planning.PoseTarget(left, left_position),
               gobot.planning.PoseTarget(right, right_position, right_rotation))

    result = planner.solve_ik(targets, seed=torch.tensor([0.5, 0.2, 0.0]))
    assert result.success, result.reason
    torch.testing.assert_close(result.positions, q, atol=1e-4, rtol=0)
    assert planner.collision_distances(result.positions).min() >= 0.02 - 1e-6

    unreachable = planner.solve_ik([gobot.planning.PoseTarget(left, (4.0, 0.0, 0.5))])
    assert not unreachable.success
    assert unreachable.positions is None


def test_trajopt_detours_and_preserves_endpoints(snapshot):
    planner = gobot.planning.Planner(snapshot, {"planning": "base"})
    start = torch.tensor([-0.75, 0.0, 0.0], dtype=torch.float64)
    goal = torch.tensor([0.75, 0.0, 0.0], dtype=torch.float64)
    assert not planner.validate_trajectory(torch.stack((start, goal)), dt=1.0)
    assert not planner.solve_trajopt(start, goal, dt=1.0, waypoints=2).success
    assert "velocity" in planner.solve_trajopt(start, goal, dt=0.01).reason

    plan = planner.solve_trajopt(start, goal, waypoints=9, dt=0.2)
    assert plan.success, plan.reason
    torch.testing.assert_close(plan.positions[0], start, atol=0, rtol=0)
    torch.testing.assert_close(plan.positions[-1], goal, atol=0, rtol=0)
    assert planner.validate_trajectory(plan.positions, dt=plan.dt, collision_substeps=64)
    assert plan.positions[:, 1].max() > 0.15

    link = ("planning", "left_tip")
    endpoint_targets = [
        [gobot.planning.PoseTarget(link, planner.link_pose(q, link)[0])] for q in (start, goal)
    ]
    via_ik = planner.plan_poses(*endpoint_targets, dt=0.2, waypoints=9,
                                start_seed=start, goal_seed=goal, seed=plan.positions, collision_substeps=32)
    assert via_ik.success, via_ik.reason
    assert planner.validate_trajectory(via_ik.positions, dt=via_ik.dt, collision_substeps=64)


def test_sqp_recovers_fixed_time_joint_cost_minimum(snapshot):
    pytest.importorskip("osqp")
    planner = gobot.planning.Planner(snapshot, {"planning": "base"})
    start = torch.tensor([-0.6, 0.35, 0.], dtype=torch.float64)
    goal = torch.tensor([0.6, 0.35, 0.], dtype=torch.float64)
    # Equal increments minimize the upstream zero-target JointVelEq sum with
    # fixed endpoints, and simultaneously drive the JointAccEq cost to zero.
    optimum = torch.lerp(start, goal, torch.linspace(0, 1, 7, dtype=start.dtype)[:, None])
    seed = optimum.clone()
    seed[2:5, 1] += 0.12
    assert planner.validate_trajectory(seed, dt=0.5)
    path = planner.solve_trajopt(start, goal, dt=0.5, waypoints=7, seed=seed, solver="sqp")
    assert path.success, path.reason
    torch.testing.assert_close(path.positions, optimum, atol=1e-5, rtol=0)
    first = planner.solve_trajopt(start, goal, dt=0.5, waypoints=7, seed=seed, solver="sqp_fast")
    assert first.success and torch.equal(first.positions, seed)


def test_sqp_skips_qp_only_for_valid_linear_trajectory(snapshot, monkeypatch):
    pytest.importorskip("osqp")
    from gobot.planning import _sqp

    planner = gobot.planning.Planner(snapshot, {"planning": "base"})
    start = torch.tensor([-0.6, 0.35, 0.], dtype=torch.float64)
    goal = torch.tensor([0.6, 0.35, 0.], dtype=torch.float64)
    settings = dict(dt=0.5, waypoints=7, solver="sqp")
    qp_calls = []
    solve_qp = _sqp._qp_step

    def counted_step(*args):
        qp_calls.append(1)
        return solve_qp(*args)

    monkeypatch.setattr(_sqp, "_qp_step", counted_step)
    clear = planner.solve_trajopt(start, goal, **settings)
    assert clear.success and torch.equal(clear.positions, clear.initial_positions)
    assert not qp_calls

    # Keep both endpoints clear but move the joint-space line through the box.
    start[1] = goal[1] = 0.0
    detour = planner.solve_trajopt(start, goal, **settings)
    assert qp_calls
    assert detour.success, detour.reason
    assert planner.validate_trajectory(detour.positions, dt=0.5, collision_substeps=32)


def test_sqp_intermediate_pose_and_unreachable_target(snapshot):
    pytest.importorskip("osqp")
    planner = gobot.planning.Planner(snapshot, {"planning": "base"})
    start = torch.tensor([-0.75, 0.35, 0.], dtype=torch.float64)
    goal = torch.tensor([0.75, 0.35, 0.], dtype=torch.float64)
    link = ("planning", "left_tip")
    target = gobot.planning.PoseTarget(link, [0.12, 0.48, 0.5])
    options = dict(dt=0.4, waypoints=7, solver="sqp", collision_substeps=2)
    path = planner.solve_trajopt(start, goal, waypoint_targets={3: target}, **options)
    assert path.success, path.reason
    torch.testing.assert_close(path.positions[[0, -1]], torch.stack((start, goal)), atol=0, rtol=0)
    # The two slides are aligned with the world x and y axes.
    torch.testing.assert_close(path.positions[3, :2], start.new_tensor(target.position[:2]),
                               atol=1e-4, rtol=0)
    assert planner.validate_trajectory(path.positions, dt=0.4)

    impossible = gobot.planning.PoseTarget(link, [0.12, 0.48, 1.5])
    path = planner.solve_trajopt(start, goal, waypoint_targets={3: impossible}, **options)
    assert not path.success and "waypoint 3" in path.reason


def test_inter_arm_and_cross_robot_collision(snapshot):
    planner = gobot.planning.Planner(snapshot, {"planning": "base"})
    q = torch.tensor([0.25, 0.7, 0.0], dtype=torch.float64)
    distances = planner.collision_distances(q)
    index = int(distances.argmin())
    assert distances[index] < 0
    assert set(planner.collision_pairs[index]) == {("planning", "left_tip"), ("planning", "right_tip")}
    ignore = [(("planning", "left_tip"), ("planning", "right_tip"))]
    filtered = gobot.planning.Planner(snapshot, {"planning": "base"}, ignored_link_pairs=ignore)
    assert filtered.collision_distances(q).min() > 0.02

    other = deepcopy(snapshot["robots"][0])
    other["name"] = "other"
    snapshot["robots"].append(other)
    coupled = gobot.planning.Planner(snapshot, {"planning": "base", "other": "base"})
    assert ("other", "x") in coupled.joint_names
    assert coupled.collision_distances(torch.tensor([0.6, 0.7, 0.0, 0.6, 0.7, 0.0])).min() < 0


def test_multiple_base_links_in_one_robot(snapshot):
    robot = snapshot["robots"][0]
    original = gobot.planning.Planner(snapshot, {"planning": "base"})
    # Some imported robots have independent arm bases in one Robot3D.
    robot["links"].append(dict(deepcopy(robot["links"][0]), name="second_base"))
    next(j for j in robot["joints"] if j["name"] == "hinge")["parent_link"] = "second_base"
    planner = gobot.planning.Planner(snapshot, {"planning": ("base", "second_base")})
    assert planner.joint_names == original.joint_names
    q = torch.tensor([0.25, 0.7, 0.0], dtype=torch.float64)
    torch.testing.assert_close(planner.collision_distances(q), original.collision_distances(q))


def test_link_local_capsule_override(snapshot):
    tip = ("planning", "left_tip")
    position = (0.0, 0.25, 0.0)
    proxy = gobot.planning.Capsule(position, position, 0.045)
    planner = gobot.planning.Planner(snapshot, {"planning": "base"}, link_capsules={tip: [proxy]})
    q = torch.zeros(3, dtype=torch.float64)
    assert planner.collision_distances(q).min() > 0.02
    jacobian = torch.func.jacfwd(planner.collision_distances)(q)
    assert torch.isfinite(jacobian).all()
    assert (jacobian[:, 1].abs() > 0.1).any()


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_degenerate_and_crossing_capsules(dtype):
    def tensor(value):
        return torch.tensor(value, dtype=dtype)

    a, b, radius = tensor([-1, 0, 0]), tensor([1, 0, 0]), tensor(0.1)
    c, d = tensor([0, -1, 0.3]), tensor([0, 1, 0.3])
    torch.testing.assert_close(capsule_distance(a, b, radius, c, d, radius), tensor(0.1))
    torch.testing.assert_close(capsule_distance(a, b, radius, c, c, radius), tensor(math.sqrt(1.09) - 0.2))
    parallel = capsule_distance(a, b, radius, a + c, b + c, radius)
    torch.testing.assert_close(parallel, tensor(math.sqrt(1.09) - 0.2))

    center, rotation, half_size = tensor([0, 0, 0]), torch.eye(3, dtype=dtype), tensor([0.2, 0.2, 0.2])
    torch.testing.assert_close(box_distance(a, b, radius, center, rotation, half_size), tensor(-0.3))
    a, b = a + tensor([0, 0.4, 0]), b + tensor([0, 0.4, 0])
    torch.testing.assert_close(box_distance(a, b, radius, center, rotation, half_size), tensor(0.1))


def test_rotation_residual_has_finite_forward_jacobian():
    from gobot._kinematics import axis_rotation

    axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)

    def residual(angle):
        return rotation_log(axis_rotation(axis, angle))

    for angle in (0.0, math.pi - 1e-7, math.pi + 1e-7):
        value = torch.tensor(angle, dtype=axis.dtype)
        expected = math.remainder(angle, 2 * math.pi)
        torch.testing.assert_close(residual(value), axis * expected)
        torch.testing.assert_close(torch.func.jacfwd(residual)(value), axis)


@pytest.mark.parametrize("backend", ["coal", "mujoco_warp"])
@pytest.mark.parametrize("solver", ["sqp", "sqp_fast"])
def test_batch_trajopt_independent_waypoints_and_failures(snapshot, backend, solver):
    pytest.importorskip("osqp")
    pytest.importorskip("gobot._coal_native")
    if backend == "mujoco_warp":
        pytest.importorskip("mujoco_warp")
    planner = gobot.planning.Planner(snapshot, {"planning": "base"}, collision_backend=backend)
    start = torch.tensor([[-0.75, 0.35, 0.], [-0.65, 0.42, 0.], [-0.6, 0.32, 0.], [0., 0., 0.]],
                         dtype=torch.float64)
    goal = start.clone()
    goal[:, 0] = torch.tensor([0.75, 0.68, 0.8, 0.7])
    link = ("planning", "left_tip")
    targets = [{2: start.new_tensor([-0.18, 0.49, 0.1])},
               {4: gobot.planning.PoseTarget(link, [0.1, 0.4, 0.5])},
               {3: gobot.planning.PoseTarget(link, [0.1, 0.4, 1.5])}, {}]
    settings = dict(dt=0.45, waypoints=7, collision_substeps=2, iterations=12, solver=solver)
    result = gobot.planning.PlannerBatch(planner).solve_trajopt(start, goal, waypoint_targets=targets, **settings)
    assert result.success.tolist() == [True, True, False, False], result.reasons
    assert "waypoint 3" in result.reasons[2] and "collision" in result.reasons[3]
    assert result.positions[2:].isnan().all()
    assert result.initial_positions[2].isfinite().all() and result.initial_positions[3].isnan().all()
    for i in range(2):
        solo = planner.solve_trajopt(start[i], goal[i], waypoint_targets=targets[i], **settings)
        assert solo.success, solo.reason
        torch.testing.assert_close(result.positions[i], solo.positions, atol=1e-5, rtol=0)
        assert planner.validate_trajectory(result.positions[i], dt=result.dt, collision_substeps=16)
    torch.testing.assert_close(result.positions[0, 2], targets[0][2], atol=1e-4, rtol=0)
    torch.testing.assert_close(planner.link_pose(result.positions[1, 4], link)[0],
                               start.new_tensor(targets[1][4].position), atol=1e-4, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_coal_broadphase_matches_exhaustive_clearance(snapshot, dtype):
    pytest.importorskip("gobot._coal_native")
    from gobot.planning._geometry import clearance_mask

    scene = gobot.planning.Planner(snapshot, {"planning": "base"}, dtype=dtype,
                                  collision_backend="coal")._scene
    model, = scene.distance_models
    # Inter-arm penetration, box penetration, a box clearance at the threshold,
    # and well-separated configurations. The halfspace is unbounded.
    q = scene.tensor([[0.25, 0.7, 0.0], [0, 0, 0], [0.225, -0.06, 0.3], [-0.8, 0.4, -1.0]])
    p, r = scene.poses(q)
    for margin in (0.0, 0.02, 0.020002):
        margins = scene.pair_margins(0.001, margin)
        for original in (True, False):
            distances = model.distances(p, r, verify=original)
            exact = clearance_mask(distances, margins)
            assert exact.any() and not exact.all()
            assert torch.equal(model.clearance_mask(p, r, margins, original), exact)
    rows, ids, data = model.nearby(model.world_frames(p, r), margins)
    assert 0 < len(ids) < len(q) * len(margins)
    torch.testing.assert_close(data[:, 0], scene.distances(q)[rows, ids])


@pytest.mark.parametrize("backend", ["coal", "mujoco_warp"])
def test_sparse_segment_rows_match_dense_jacobian(snapshot, backend):
    pytest.importorskip("gobot._coal_native")
    pytest.importorskip("osqp")
    if backend == "mujoco_warp":
        pytest.importorskip("mujoco_warp")
    from torch.func import jacfwd, vmap
    from gobot.planning._trajectory import make_segment_constraints, sqp_segment_rows

    scene = gobot.planning.Planner(snapshot, {"planning": "base"}, collision_backend=backend)._scene
    margin, world_margin, dt = 0.001, 0.014, 0.5
    margins = scene.pair_margins(margin, world_margin)
    constraints = make_segment_constraints(scene, dt, margin, world_margin)
    q = scene.tensor([[-0.6, -0.06, 0.2], [0.19, -0.07, 0.42], [0.24, 0.71, 0.18], [0.5, 0.6, 0.4]])
    segments = torch.stack((q[:-1], q[1:]), dim=1)
    fractions = scene.tensor([[0.0], [0.5]])
    dense = constraints(segments, fractions)[0]
    jac = vmap(jacfwd(lambda pair: constraints(pair, fractions)[0]))(segments)
    for linearize in (True, False):
        selected = dense > (-0.05 if linearize else 0)
        selected[:, :2 * q.size(-1)] = torch.isfinite(dense[:, :2 * q.size(-1)])
        segment, component = selected.nonzero(as_tuple=True)
        packed = sqp_segment_rows(scene, dt, margins, world_margin, constraints, segments, fractions, linearize)
        assert torch.equal(packed.segment, segment)
        torch.testing.assert_close(packed.value, dense[segment, component])
        if linearize:
            torch.testing.assert_close(packed.jacobian, jac[segment, component], atol=1e-7, rtol=1e-6)
        else:
            assert packed.jacobian is None
            torch.testing.assert_close(packed.value.clamp_min(0).sum(), dense.clamp_min(0).sum())


def test_sqp_fast_uses_closest_sample_per_pair_and_segment(snapshot):
    pytest.importorskip("gobot._coal_native")
    from torch.func import jacfwd, vmap

    scene = gobot.planning.Planner(snapshot, {"planning": "base"}, collision_backend="coal")._scene
    model, = scene.distance_models
    margins = scene.pair_margins(0.001, 0.014)
    q = scene.tensor([[-0.6, -0.06, 0.2], [0.19, -0.07, 0.42], [0.24, 0.71, 0.18]])
    fractions = torch.arange(6, dtype=q.dtype)[:, None] / 6
    samples = torch.lerp(q[:-1, None], q[1:, None], fractions).flatten(0, 1)
    maximum, closest = (margins - scene.distances(samples)).unflatten(0, (2, 6)).max(dim=1)
    jac = -vmap(jacfwd(scene.distances))(samples)
    for linearize in (False, True):
        buffer = 0.05 if linearize else 0.0
        segment, pair = (maximum > -buffer).nonzero(as_tuple=True)
        sample = 6 * segment + closest[segment, pair]
        order = (sample * len(margins) + pair).argsort()
        sample, pair, segment = sample[order], pair[order], segment[order]
        rows, values, gradient = model.constraint_rows(scene.poses, samples, margins, buffer, linearize,
                                                       group_size=6, segment_minimum=True)
        assert len(sample) and torch.equal(rows, sample)
        torch.testing.assert_close(values, maximum[segment, pair])
        if linearize:
            torch.testing.assert_close(gradient, jac[sample, pair])


def test_sqp_fast_retries_with_full_rows(snapshot, monkeypatch):
    pytest.importorskip("gobot._coal_native")
    pytest.importorskip("osqp")
    from gobot import planning

    planner = planning.Planner(snapshot, {"planning": "base"}, collision_backend="coal")
    start, goal = planner._scene.tensor([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]])
    states = []
    optimize = planning.optimize

    def exhaust_first_attempt(*args):
        states.append((args[4], args[-1]))
        if args[-1]:
            return None
        return (yield from optimize(*args))

    monkeypatch.setattr(planning, "optimize", exhaust_first_attempt)
    path = planner.solve_trajopt(start, goal, waypoints=7, dt=0.5, collision_substeps=4, solver="sqp_fast")
    assert states[:2] == [(8, True), (4, False)]
    assert path.success, path.reason
    assert planner.validate_trajectory(path.positions, dt=0.5, collision_substeps=8)
