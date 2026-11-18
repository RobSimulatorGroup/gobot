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
