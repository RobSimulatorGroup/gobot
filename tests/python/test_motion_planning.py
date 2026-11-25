"""Comprehensive tests for Gobot's unified C++-backed motion planning architecture."""

from __future__ import annotations

import math
from pathlib import Path
import runpy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import gobot
from gobot.planning import (
    MotionPlanner,
    MotionPlanRequest,
    PathConstraint,
    PlanningWorld,
    PoseConstraint,
    SolverConfig,
    Trajectory,
    retime_trajectory,
)

pytest.importorskip("gobot._coal_native", exc_type=ModuleNotFoundError)
pytest.importorskip("osqp")


@pytest.fixture
def simple_snapshot(tmp_path):
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
    ctx = gobot.app.context()
    ctx.set_project_path(str(tmp_path))
    ctx.load_scene("res://planning.jscn")
    snap = ctx.compile_physics_scene_snapshot()
    for robot in snap["robots"]:
        for joint in robot["joints"]:
            if joint["name"] == "x":
                joint["velocity_limit"] = 2.0
    yield snap
    ctx.clear_scene()


@pytest.fixture
def panda_snapshot():
    project = Path(__file__).resolve().parents[2] / "examples" / "panda_trajopt"
    ctx = gobot.app.context()
    prev = ctx.project_path
    try:
        ctx.set_project_path(str(project))
        ctx.load_scene("res://world.jscn")
        snap = ctx.compile_physics_scene_snapshot()
        root = ctx.root
        markers = (root.find("Start").position, root.find("Goal").position, root.find("Inspection").position)
    finally:
        ctx.clear_scene()
        if prev:
            ctx.set_project_path(prev)
    return snap, markers


def test_mutable_world_dynamic_obstacles_attach_detach_and_acm(simple_snapshot):
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    planner = MotionPlanner(world, backend="coal")
    q_clear = np.array([-0.6, 0.35, 0.0])
    assert world.collision_distances(q_clear).min() > 0.05

    # 1. Add dynamic obstacle right on top of left_tip -> collision detected.
    tip_pos, _ = world.link_pose(q_clear, ("planning", "left_tip"))
    world.add_box("dyn_box", (0.2, 0.2, 0.2), tip_pos)
    assert world.collision_distances(q_clear).min() < 0.0

    # 2. O(1) obstacle pose update moves it away -> collision clears.
    v_before = world.topology_version
    world.update_obstacle_pose("dyn_box", tip_pos + np.array([0.0, 0.5, 0.0]))
    assert world.topology_version == v_before  # No topology/pair rebuild!
    assert world.collision_distances(q_clear).min() > 0.05

    # 3. Attach obstacle to left_tip -> moves with link, collides with static obstacle when swept.
    world.attach_object("dyn_box", ("planning", "left_tip"), local_position=(0.0, -0.15, 0.0))
    # Moving left_tip near y=0.2 causes attached box (at y=0.05) to collide with static obstacle.
    q_hit = np.array([0.0, 0.20, 0.0])
    assert world.collision_distances(q_hit).min() < 0.0

    # 4. Detach at q_clear returns box to world at its current world pose.
    world.detach_object("dyn_box", q_clear)
    world.remove_obstacle("dyn_box")
    assert world.collision_distances(q_clear).min() > 0.05

    # 5. Mutable ACM toggles collision between left_tip and right_tip.
    q_self = np.array([0.25, 0.7, 0.0])
    assert world.collision_distances(q_self).min() < 0.0
    world.set_collision_enabled(("planning", "left_tip"), ("planning", "right_tip"), False)
    assert world.collision_distances(q_self).min() > 0.02
    world.set_collision_enabled(("planning", "left_tip"), ("planning", "right_tip"), True)
    assert world.collision_distances(q_self).min() < 0.0

    # 6. Mutable robot base pose shifts link world position.
    p0, _ = world.link_pose(q_clear, ("planning", "left_tip"))
    world.set_robot_base_pose("planning", (0.5, 0.0, 0.5))
    p1, _ = world.link_pose(q_clear, ("planning", "left_tip"))
    np.testing.assert_allclose(p1 - p0, [0.5, 0.0, 0.0], atol=1e-10)
    assert planner.world is world


def test_planning_group_locks_inactive_joints(simple_snapshot):
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    world.define_group("left_only", [("planning", "x"), ("planning", "y")])
    planner = MotionPlanner(world, backend="coal")

    start = np.array([-0.7, 0.0, 0.42])
    goal = np.array([0.7, 0.0, 0.42])
    res = planner.plan(MotionPlanRequest(
        start=start, goal=goal, group="left_only",
        config=SolverConfig(waypoints=10, margin=0.02, world_margin=0.02),
    ))
    assert res.success, res.reason
    # Inactive hinge joint stays strictly locked at 0.42 across the entire trajectory.
    np.testing.assert_allclose(res.positions[:, 2], 0.42, atol=1e-12)
    ok, reason = world.validate_trajectory(res.positions, margin=0.02, world_margin=0.02)
    assert ok, reason


def test_cpp_ik_free_rotation_axis_and_goal_set(panda_snapshot):
    snap, (start_xyz, goal_xyz, _) = panda_snapshot
    mounts = (("panda_link0", "panda_link2"), ("panda_link4", "panda_link6"), ("panda_link7", "panda_hand"))
    world = PlanningWorld.from_snapshot(
        snap, {"panda": "panda_link0"},
        ignored_link_pairs=[(("panda", a), ("panda", b)) for a, b in mounts],
    )
    planner = MotionPlanner(world, backend="coal")
    hand = ("panda", "panda_hand")
    down = np.diag([-1.0, 1.0, -1.0])
    seed = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, -2.3])

    # Free Z rotation axis (suction cup / cylindrical grasp): align local Z with world -Z at any yaw.
    free_z = PoseConstraint(hand, start_xyz, down, position_tolerance=1e-4, rotation_tolerance=(1e-3, 1e-3, math.inf))
    ik = planner.solve_ik([free_z], seed=seed, margin=0.002, world_margin=0.025)
    assert ik.success, ik.reason
    pos, rot = world.link_pose(ik.positions, hand)
    np.testing.assert_allclose(pos, start_xyz, atol=1e-4)
    # Local Z-axis must align with world -Z (dot >= cos(1e-3)).
    assert float(np.dot(rot[:, 2], down[:, 2])) >= math.cos(1e-3) - 1e-6

    # GoalSet planning: unreachable candidate + two valid candidates -> picks feasible one.
    unreachable = [PoseConstraint(hand, [5.0, 5.0, 5.0], down)]
    valid_a = [PoseConstraint(hand, goal_xyz, down)]
    valid_b = [PoseConstraint(hand, goal_xyz + np.array([0.0, -0.05, 0.04]), down)]
    res = planner.plan(MotionPlanRequest(
        start=ik.positions,
        goal=[unreachable, valid_a, valid_b],
        config=SolverConfig(waypoints=12, margin=0.002, world_margin=0.02, first_feasible=True),
    ))
    assert res.success, res.reason
    assert res.goal_index in (1, 2)
    assert res.diagnostics.total_ms > 0
    assert res.trajectory.duration > 0


def test_full_trajectory_path_constraint(simple_snapshot):
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    planner = MotionPlanner(world, backend="coal")
    tip = ("planning", "left_tip")

    start = np.array([-0.7, 0.35, 0.0])
    goal = np.array([0.7, 0.35, 0.0])
    # Constrain y == 0.42 across all interior waypoints while leaving x and z free!
    corridor = PathConstraint(
        PoseConstraint(tip, position=(0.0, 0.42, 0.5), position_tolerance=(math.inf, 1e-4, math.inf)),
    )
    res = planner.plan(MotionPlanRequest(
        start=start, goal=goal, path_constraints=[corridor],
        config=SolverConfig(waypoints=9, margin=0.02, world_margin=0.02, first_feasible=True),
    ))
    assert res.success, res.reason
    pts, _ = world.link_pose(res.positions[1:-1], tip)
    np.testing.assert_allclose(pts[:, 1], 0.42, atol=1e-4)


def test_birrt_global_seeding_and_batch_racing(simple_snapshot):
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    # Build a U-shaped trap around the straight-line seed so pure local SQP would get blocked.
    world.add_box("trap_front", (0.08, 0.50, 0.30), (0.0, 0.0, 0.5))
    world.add_box("trap_top", (0.50, 0.08, 0.30), (0.0, 0.25, 0.5))
    planner = MotionPlanner(world, backend="coal")

    start = np.array([-0.5, 0.0, 0.0])
    goal = np.array([0.5, 0.0, 0.0])
    # Multi-seed racing with C++ BiRRT global seeding escapes the trap around y < -0.25.
    res = planner.plan(MotionPlanRequest(
        start=start, goal=goal,
        config=SolverConfig(
            waypoints=12, margin=0.01, world_margin=0.01,
            use_global_seeding=True, num_seeds=4, first_feasible=True, rrt_timeout_s=0.5,
        ),
    ))
    assert res.success, res.reason
    ok, reason = world.validate_trajectory(res.positions, margin=0.01, world_margin=0.01)
    assert ok, reason


def test_retime_trajectory_respects_velocity_and_acceleration_limits():
    # Minimal single-segment exact bound test: q=[0, 1], vmax=1, amax=1.
    simple = retime_trajectory(np.array([[0.0], [1.0]]), np.array([1.0]), np.array([1.0]), stop_at_endpoints=True)
    ts_simple = np.linspace(0.0, simple.duration, 1001)
    ps, vs, as_ = simple.sample(ts_simple)
    assert np.max(np.abs(vs)) <= 1.0 + 1e-6
    assert np.max(np.abs(as_)) <= 1.0 + 1e-6
    np.testing.assert_allclose(ps[0], [0.0], atol=1e-12)
    np.testing.assert_allclose(ps[-1], [1.0], atol=1e-12)

    # Multi-waypoint path with sharp corner.
    q = np.linspace([0.0, -1.0, 0.5], [1.5, 0.8, -0.5], 12)
    q[4:8, 1] += 0.35
    vmax = np.array([1.2, 1.5, 1.0])
    amax = np.array([3.0, 4.0, 2.5])
    traj = retime_trajectory(q, vmax, amax, stop_at_endpoints=True)
    assert traj.duration > 0.0
    np.testing.assert_allclose(traj.velocities[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(traj.velocities[-1], 0.0, atol=1e-12)
    assert (np.abs(traj.velocities) <= vmax * (1.0 + 1e-6)).all()
    # Continuous Hermite sampling strictly satisfies velocity and acceleration limits everywhere.
    p_mid, v_mid, a_mid = traj.sample(np.linspace(0.0, traj.duration, 2001))
    assert p_mid.shape == (2001, 3) and v_mid.shape == (2001, 3) and a_mid.shape == (2001, 3)
    assert np.isfinite(p_mid).all() and np.isfinite(v_mid).all() and np.isfinite(a_mid).all()
    assert (np.abs(v_mid) <= vmax * (1.0 + 1e-5)).all(), np.max(np.abs(v_mid) / vmax)
    assert (np.abs(a_mid) <= amax * (1.0 + 1e-5)).all(), np.max(np.abs(a_mid) / amax)
    np.testing.assert_allclose(p_mid[0], q[0], atol=1e-10)
    np.testing.assert_allclose(p_mid[-1], q[-1], atol=1e-10)


def test_touch_links_does_not_disable_robot_self_collision_and_restores_on_detach(simple_snapshot):
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    # Configuration where left_tip (sphere r=0.045) and right_tip (capsule r=0.035) penetrate deeply (~80 mm).
    q_penetrate = np.array([0.25, 0.70, 0.0])
    d_before = float(world.collision_distances(q_penetrate).min())
    assert d_before < -0.05

    # Attach an object to left_tip with touch_links=[right_tip]:
    # Only the attached object <-> right_tip pair may be ignored, NOT left_tip <-> right_tip!
    tip_pos, _ = world.link_pose(q_penetrate, ("planning", "left_tip"))
    world.add_box("held_part", (0.04, 0.04, 0.04), tip_pos)
    world.attach_object("held_part", ("planning", "left_tip"), touch_links=[("planning", "right_tip")])

    # Attached object itself should have no pair with left_tip or right_tip.
    pair_names = world.collision_pairs
    assert not any("held_part" in a and ("left_tip" in b or "right_tip" in b) for a, b in pair_names)
    assert not any("held_part" in b and ("left_tip" in a or "right_tip" in a) for a, b in pair_names)
    # Robot self-collision between left_tip and right_tip must still be detected!
    d_attached = float(world.collision_distances(q_penetrate).min())
    assert d_attached < -0.05
    ok_attached, _ = world.validate_trajectory(np.stack((q_penetrate, q_penetrate)), margin=0.005, world_margin=0.005)
    assert not ok_attached

    # Detach and remove: self-collision detection remains intact.
    world.detach_object("held_part", q_penetrate)
    world.remove_obstacle("held_part")
    d_after = float(world.collision_distances(q_penetrate).min())
    assert d_after < -0.05
    ok_after, _ = world.validate_trajectory(np.stack((q_penetrate, q_penetrate)), margin=0.005, world_margin=0.005)
    assert not ok_after


def test_hermite_curve_clearance_and_corner_overshoot_guard(simple_snapshot):
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    planner = MotionPlanner(world, backend="coal")
    # Plan around the central box with a tight corridor; verify the ACTUAL continuous Hermite curve.
    cfg = SolverConfig(
        waypoints=10, dt=None, retime=True, margin=0.01, world_margin=0.02,
        collision_substeps=8, stop_at_endpoints=True, first_feasible=True,
    )
    res = planner.plan(MotionPlanRequest(
        start=np.array([-0.55, 0.0, 0.0]),
        goal=np.array([0.55, 0.0, 0.0]),
        config=cfg,
    ))
    assert res.success, res.reason
    ok, reason = world.validate_trajectory(
        res.trajectory, margin=0.01, world_margin=0.02, collision_substeps=16, check_acceleration=True,
    )
    assert ok, reason
    ts = np.linspace(0.0, res.trajectory.duration, 401)
    q_s, v_s, a_s = res.trajectory.sample(ts)
    margins = world.pair_margins(0.01, 0.02)
    dists = world.collision_distances(q_s)
    assert (dists >= margins[None, :] - 1e-4).all(), float((dists - margins[None, :]).min())
    assert (np.abs(v_s) <= world.velocity_limits * (1.0 + 1e-5)).all()
    assert (np.abs(a_s) <= world.acceleration_limits * (1.0 + 1e-5)).all()


def test_fixed_dt_rejects_infeasible_velocity_or_acceleration_bounds(simple_snapshot):
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    planner = MotionPlanner(world, backend="coal")
    # Reproduces dt=0.4 on a 0.8 m slide with vmax=2.0, amax=10.0, stop_at_endpoints=True:
    # polyline average speed is 2.0 (passes polyline check), but cubic Hermite reaches v=3.0, a=30.0.
    start = np.array([-0.4, 0.35, 0.0])
    goal = np.array([0.4, 0.35, 0.0])
    infeasible = planner.plan(MotionPlanRequest(
        start=start, goal=goal,
        config=SolverConfig(
            waypoints=4, dt=0.4, stop_at_endpoints=True,
            margin=0.01, world_margin=0.01, use_global_seeding=False,
        ),
    ))
    assert not infeasible.success
    assert infeasible.reason is not None and ("velocity" in infeasible.reason or "acceleration" in infeasible.reason)

    # With sufficient fixed dt (or open endpoints with smooth travel), the Hermite curve passes all limits.
    feasible = planner.plan(MotionPlanRequest(
        start=start, goal=goal,
        config=SolverConfig(
            waypoints=4, dt=0.8, stop_at_endpoints=True,
            margin=0.01, world_margin=0.01, use_global_seeding=False,
        ),
    ))
    assert feasible.success, feasible.reason
    ok, reason = world.validate_trajectory(feasible.trajectory, margin=0.01, world_margin=0.01, check_acceleration=True)
    assert ok, reason
    ts = np.linspace(0.0, feasible.trajectory.duration, 401)
    _, vf, af = feasible.trajectory.sample(ts)
    assert (np.abs(vf) <= world.velocity_limits * (1.0 + 1e-5)).all()
    assert (np.abs(af) <= world.acceleration_limits * (1.0 + 1e-5)).all()


def test_analytic_collision_and_pose_jacobians_match_finite_differences(panda_snapshot):
    snap, (start_xyz, _, _) = panda_snapshot
    mounts = (("panda_link0", "panda_link2"), ("panda_link4", "panda_link6"), ("panda_link7", "panda_hand"))
    world = PlanningWorld.from_snapshot(
        snap, {"panda": "panda_link0"},
        ignored_link_pairs=[(("panda", a), ("panda", b)) for a, b in mounts],
    )
    q = np.array([0.1, -0.45, 0.15, -1.85, 0.1, 1.45, -1.9], dtype=np.float64)
    eps = 1e-6

    # 1. Check C++ analytic collision residual gradients vs. central finite differences.
    margins = world.pair_margins(0.002, 0.025)
    rows, pids, _, grads, _, _, _ = world._native.eval_collision_rows(
        q[None, :], margins, margins, 0.08, True, 1, False,
    )
    assert len(rows) > 0
    for k in range(min(len(rows), 8)):
        pid = int(pids[k])
        g_ana = grads[k]  # gradient of residual = margin - distance
        g_fd = np.zeros_like(q)
        for j in range(world.num_joints):
            dq = np.zeros_like(q)
            dq[j] = eps
            dp = float(world.collision_distances(q + dq)[pid])
            dm = float(world.collision_distances(q - dq)[pid])
            g_fd[j] = -(dp - dm) / (2.0 * eps)
        np.testing.assert_allclose(g_ana, g_fd, atol=2e-3, rtol=2e-2)

    # 2. Check C++ analytic pose constraint Jacobians (full 6D and free-Z) vs. central finite differences.
    hand = ("panda", "panda_hand")
    down = np.diag([-1.0, 1.0, -1.0])
    specs = [
        world.compile_pose_constraint(PoseConstraint(hand, start_xyz, down)),
        world.compile_pose_constraint(PoseConstraint(hand, start_xyz, down, rotation_tolerance=(1e-3, 1e-3, math.inf))),
    ]
    _, J_ana, _ = world._native.eval_pose_constraints(q, specs, True)
    J_fd = np.zeros_like(J_ana)
    for j in range(world.num_joints):
        dq = np.zeros_like(q)
        dq[j] = eps
        rp, _, _ = world._native.eval_pose_constraints(q + dq, specs, False)
        rm, _, _ = world._native.eval_pose_constraints(q - dq, specs, False)
        J_fd[:, j] = (rp - rm) / (2.0 * eps)
    np.testing.assert_allclose(J_ana, J_fd, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("backend", ["coal", "mujoco_warp"])
def test_batch_planning_and_backend_parity_with_dynamic_mutation(simple_snapshot, backend):
    if backend == "mujoco_warp":
        pytest.importorskip("mujoco_warp")
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    planner = MotionPlanner(world, backend=backend)
    tip = ("planning", "left_tip")
    cfg = SolverConfig(waypoints=8, dt=0.45, margin=0.01, world_margin=0.015,
                       collision_substeps=4, iterations=20, first_feasible=True, use_global_seeding=False)
    requests = [
        MotionPlanRequest(np.array([-0.75, 0.35, 0.0]), np.array([0.75, 0.35, 0.0]),
                          waypoint_targets={2: np.array([-0.18, 0.49, 0.1])}, config=cfg),
        MotionPlanRequest(np.array([-0.65, 0.42, 0.0]), np.array([0.68, 0.42, 0.0]),
                          waypoint_targets={4: PoseConstraint(tip, [0.1, 0.4, 0.5])}, config=cfg),
        MotionPlanRequest(np.array([0.0, 0.0, 0.0]), np.array([0.7, 0.0, 0.0]), config=cfg),  # endpoint collision
    ]
    res = planner.plan_batch(requests)
    assert res.success.tolist() == [True, True, False], res.reasons
    assert "collision" in (res.reasons[2] or "")
    np.testing.assert_allclose(res.positions[0, 2], [-0.18, 0.49, 0.1], atol=1e-4)
    p4, _ = world.link_pose(res.positions[1, 4], tip)
    np.testing.assert_allclose(p4, [0.1, 0.4, 0.5], atol=1e-4)

    # Mutate world topology and verify backend stays synchronized.
    world.add_sphere("extra_ball", 0.06, (0.0, 0.42, 0.5))
    res2 = planner.plan(MotionPlanRequest(np.array([-0.7, 0.42, 0.0]), np.array([0.7, 0.42, 0.0]), config=cfg))
    assert res2.success, res2.reason
    ok, reason = world.validate_trajectory(res2.positions, dt=0.45, margin=0.01, world_margin=0.015)
    assert ok, reason


def test_real_examples_panda_and_container_unloading():
    # 1. Panda workcell inspection task.
    panda_dir = Path(__file__).resolve().parents[2] / "examples" / "panda_trajopt"
    ctx = gobot.app.context()
    prev = ctx.project_path
    try:
        ctx.set_project_path(str(panda_dir))
        ctx.load_scene("res://world.jscn")
        snap = ctx.compile_physics_scene_snapshot()
        root = ctx.root
        args = (root.find("Start").position, root.find("Goal").position, root.find("Inspection").position)
    finally:
        ctx.clear_scene()
        if prev:
            ctx.set_project_path(prev)
    out = runpy.run_path(str(panda_dir / "scripts" / "plan.py"))["solve"](snap, *args)
    assert out["inspection_error"] < 1e-4
    assert out["inspection_angle"] < 1e-4
    assert out["world_clearance"] >= 0.02
    assert out["result"].trajectory.duration > 0.0
    ts_p = np.linspace(0.0, out["result"].trajectory.duration, 401)
    _, vp, ap = out["result"].trajectory.sample(ts_p)
    assert (np.abs(vp) <= out["world"].velocity_limits * 0.6 * (1.0 + 1e-5)).all()
    assert (np.abs(ap) <= out["world"].acceleration_limits * 0.5 * (1.0 + 1e-5)).all()

    # 2. Container unloading FANUC M-10iA task.
    cont_dir = Path(__file__).resolve().parents[2] / "examples" / "container_unloading"
    try:
        ctx.set_project_path(str(cont_dir))
        ctx.load_scene("res://world.jscn")
        snap_c = ctx.compile_physics_scene_snapshot()
        targets = [(ctx.root.find(name).position, ctx.root.find(name).rotation_degrees) for name in ("Pick", "Drop")]
    finally:
        ctx.clear_scene()
        if prev:
            ctx.set_project_path(prev)
    out_c = runpy.run_path(str(cont_dir / "plan.py"))["solve"](snap_c, *targets)
    assert out_c["carton_clearance"][0] < -0.20
    assert out_c["carton_clearance"][1] >= 0.014 - 1e-6
    assert out_c["result"].trajectory.duration > 0.0
    assert len(out_c["handoff"]) == 2
    # Verify actual continuous Hermite playback curve passes convex, original mesh, velocity & acceleration limits.
    ok_c, reason_c = out_c["world"].validate_trajectory(
        out_c["result"].trajectory, margin=0.001, world_margin=0.014,
        collision_substeps=24, velocity_scale=0.65, acceleration_scale=0.6, check_acceleration=True,
    )
    assert ok_c, reason_c
    ts_c = np.linspace(0.0, out_c["result"].trajectory.duration, 313)
    qc, vc, ac = out_c["result"].trajectory.sample(ts_c)
    payload_pairs = [i for i, (a, b) in enumerate(out_c["world"].collision_pairs)
                     if a.startswith("M10iA/Parcel/") and not b.startswith("M10iA/")]
    assert float(out_c["world"].collision_distances(qc)[..., payload_pairs].min()) >= 0.014 - 1e-5
    assert float(out_c["world"].collision_distances(qc, verify_original=True)[..., payload_pairs].min()) >= 0.014 - 1e-5
    assert (np.abs(vc) <= out_c["world"].velocity_limits * 0.65 * (1.0 + 1e-5)).all()
    assert (np.abs(ac) <= out_c["world"].acceleration_limits * 0.6 * (1.0 + 1e-5)).all()


def test_debug_draw_helper(simple_snapshot):
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    planner = MotionPlanner(world, backend="coal")
    res = planner.plan(MotionPlanRequest(
        start=np.array([-0.7, 0.0, 0.0]), goal=np.array([0.7, 0.0, 0.0]),
        config=SolverConfig(waypoints=10, margin=0.02, world_margin=0.02),
    ))
    assert res.success, res.reason
    drawn = []

    class DummyDraw:
        def draw_line(self, a, b, color):
            drawn.append((a, b, color))

    planner.debug_draw(DummyDraw(), res, link=("planning", "left_tip"), samples=10)
    assert len(drawn) >= 18

    arrows_out = []

    class DummyRender:
        class DebugArrow:
            def __init__(self, start, vector, color=(1, 1, 1, 1), show_head=True):
                self.start, self.vector, self.color, self.show_head = start, vector, color, show_head

        def set_debug_arrows(self, arrows):
            arrows_out.extend(arrows)

    planner.debug_draw(DummyRender(), res, link=("planning", "left_tip"), samples=32)
    assert len(arrows_out) >= 40


def test_cpp_l1_box_qp_kkt_and_matches_osqp():
    """Verify C++ ProxAL-SSN L1-hinge box QP solver matches OSQP to high precision and supports warm-start."""
    osqp = pytest.importorskip("osqp")
    from scipy import sparse
    from gobot import _coal_native

    rng = np.random.default_rng(7)
    for n, m, penalty in ((12, 8, 10.0), (48, 32, 100.0), (72, 60, 1e4)):
        M = rng.normal(size=(n, n))
        H = M.T @ M + 0.5 * np.eye(n)
        g = rng.normal(size=n)
        A = rng.normal(size=(m, n)) * 0.4
        v = rng.normal(size=m) * 0.1
        lo = -np.full(n, 0.15)
        hi = np.full(n, 0.15)
        # Pin a subset of variables to test pinned-variable elimination.
        lo[0] = hi[0] = 0.0
        lo[n // 2] = hi[n // 2] = 0.02

        x_cpp, solved, _, _, prim_res, dual_res = _coal_native.solve_l1_box_qp(
            H, g, A, v, lo, hi, penalty, 1e-6, 1e-6,
        )
        assert solved
        assert np.all(np.isfinite(x_cpp))
        assert (x_cpp >= lo - 1e-9).all() and (x_cpp <= hi + 1e-9).all()
        assert abs(x_cpp[0] - 0.0) < 1e-12
        assert abs(x_cpp[n // 2] - 0.02) < 1e-12
        assert prim_res < 1e-5
        assert dual_res < 1e-5

        # Reference solve via OSQP.
        scale = max(1.0, penalty / 10.0)
        P = sparse.block_diag([sparse.triu(sparse.csc_matrix(H)), sparse.csc_matrix((m, m))], format="csc") / scale
        q_vec = np.r_[g / scale, np.full(m, penalty / scale)]
        A_mat = sparse.vstack([
            sparse.eye(n + m, format="csc"),
            sparse.hstack([sparse.csc_matrix(A), -sparse.eye(m, format="csc")]),
        ], format="csc")
        l_vec = np.r_[lo, np.zeros(m), np.full(m, -np.inf)]
        u_vec = np.r_[hi, np.full(m, np.inf), -v]
        prob = osqp.OSQP()
        prob.setup(P=P, q=q_vec, A=A_mat, l=l_vec, u=u_vec, verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=10000, polishing=True)
        res = prob.solve()
        x_ref = np.clip(res.x[:n], lo, hi)

        def obj(x):
            return 0.5 * float(x @ (H @ x)) + float(g @ x) + penalty * float(np.maximum(0.0, v + A @ x).sum())

        obj_cpp, obj_ref = obj(x_cpp), obj(x_ref)
        assert np.linalg.norm(x_cpp - x_ref) <= 1e-4, f"solution mismatch: {np.linalg.norm(x_cpp - x_ref)}"
        assert abs(obj_cpp - obj_ref) / scale <= 1e-3 * max(1.0, abs(obj_ref) / scale), f"obj mismatch: {obj_cpp} vs {obj_ref}"

    # Verify warm-started L1BoxQPSolver across shrinking trust-region radii.
    solver = _coal_native.L1BoxQPSolver()
    solver.setup(H, g, A, v, 50.0)
    for radius in (0.2, 0.05, 0.01):
        lo_r = np.full(n, -radius)
        hi_r = np.full(n, radius)
        step = solver.solve(lo_r, hi_r, 1e-5, 1e-5)
        assert step is not None and np.all(np.isfinite(step))
        assert (step >= lo_r - 1e-9).all() and (step <= hi_r + 1e-9).all()


def test_conservative_advancement_ccd_catches_inter_sample_tunneling(simple_snapshot):
    """Conservative Advancement CCD must detect thin obstacles strictly between discrete samples."""
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    # Place a thin blade centered at x=0.0, y=0.0, z=0.5 directly between start x=-0.6 and goal x=+0.6.
    world.add_box("thin_blade", (0.04, 0.25, 0.30), (0.0, 0.0, 0.5))
    q_path = np.array([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=np.float64)
    margins = world.pair_margins(0.005, 0.01)

    # At substeps=1, discrete samples are ONLY endpoints q_path[0] and q_path[1], both far from x=0.
    (frames,) = world._native.compute_frames(np.ascontiguousarray(q_path), False)
    rows, pids = world._native.screen_aabbs(frames, margins, False, 1)
    assert world._native.first_violation_indexed(frames, rows, pids, margins, False) is None

    # Conservative Advancement must catch the blade at u ~ 0.5 despite zero discrete endpoint violations.
    hit = world._native.first_violation_continuous(q_path, 1, frames, margins, False, 2e-3)
    assert hit is not None
    seg, u_hit, _, dist = hit
    assert seg == 0 and 0.25 < u_hit < 0.75 and dist < 0.01

    # End-to-end planner with collision_substeps=1 must avoid tunneling through the thin blade.
    planner = MotionPlanner(world, backend="coal")
    res = planner.plan(MotionPlanRequest(
        start=q_path[0], goal=q_path[1],
        config=SolverConfig(waypoints=6, margin=0.005, world_margin=0.01, collision_substeps=1, iterations=40),
    ))
    assert res.success, res.reason
    ts = np.linspace(0.0, res.trajectory.duration, 201)
    qs, _, _ = res.trajectory.sample(ts)
    assert float(world.collision_distances(qs).min()) >= 0.005 - 1e-5


def test_missing_acceleration_limit_emits_warning_and_hermite_ccd(simple_snapshot):
    """from_snapshot must warn when acceleration_limit defaults, and Hermite CCD must catch curved-spline hits."""
    with pytest.warns(UserWarning, match="missing acceleration_limit"):
        world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})

    # Place an obstacle directly on the revolute arc of right_tip (radius 0.25 around (0, 0.7, 0.5)):
    # At q_hinge = 0 -> tip at (0.25, 0.7, 0.5); at q_hinge = pi -> tip at (-0.25, 0.7, 0.5);
    # mid-arc passes through (0.0, 0.95, 0.5).
    world.add_box("arc_post", (0.06, 0.06, 0.25), (0.0, 0.95, 0.5))
    q_path = np.array([[-0.6, 0.0, 0.0], [-0.6, 0.0, math.pi]], dtype=np.float64)
    traj = retime_trajectory(q_path, world.velocity_limits, world.acceleration_limits, stop_at_endpoints=True)
    ok, reason = world.validate_trajectory(traj, margin=0.005, world_margin=0.01, collision_substeps=1)
    assert not ok
    assert reason is not None and "continuous" in reason


def test_concurrent_world_mutation_during_planning(simple_snapshot):
    """Mutating PlanningWorld on another thread during plan_batch must be thread-safe via CoW snapshots."""
    import threading
    from gobot import _coal_native

    assert _coal_native.abi_version() >= 1
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    world.add_box("mover", (0.10, 0.10, 0.20), (0.0, 0.30, 0.5))
    planner = MotionPlanner(world, backend="coal")

    stop = threading.Event()
    errors: list[Exception] = []

    def mutator():
        step = 0
        try:
            while not stop.is_set():
                y = 0.28 + 0.06 * math.sin(0.2 * step)
                world.update_obstacle_pose("mover", (0.0, y, 0.5))
                if step % 6 == 0:
                    world.add_sphere("tmp_s", 0.03, (0.85, -0.85, 0.5))
                    world.remove_obstacle("tmp_s")
                step += 1
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=mutator, daemon=True)
    t.start()
    try:
        cfg = SolverConfig(waypoints=10, margin=0.01, world_margin=0.015, collision_substeps=4)
        reqs = [
            MotionPlanRequest(np.array([-0.6, 0.35 + 0.02 * i, 0.0]), np.array([0.6, 0.35 + 0.02 * i, 0.0]), config=cfg)
            for i in range(6)
        ]
        for _ in range(4):
            batch = planner.plan_batch(reqs)
            assert batch.success.all(), batch.reasons
    finally:
        stop.set()
        t.join(timeout=2.0)
    assert not errors, f"mutator thread raised: {errors}"


def test_unbounded_prismatic_hermite_ccd_catches_mid_extension(tmp_path):
    """Unbounded prismatic joint extending mid-segment on a Hermite curve must not underestimate CCD bound."""
    source = tmp_path / "rp_arm.xml"
    source.write_text("""
    <mujoco model="rp_arm">
      <compiler angle="radian"/>
      <worldbody>
        <body name="base" pos="0 0 0">
          <body name="link1">
            <inertial pos="0 0 0" mass="0.1" diaginertia="0.01 0.01 0.01"/>
            <joint name="j_rev" type="hinge" axis="0 0 1" limited="false"/>
            <body name="tip">
              <inertial pos="0 0 0" mass="0.1" diaginertia="0.01 0.01 0.01"/>
              <joint name="j_pris" type="slide" axis="1 0 0" limited="false"/>
              <geom type="sphere" size="0.05"/>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """)
    gobot.set_project_path(str(tmp_path))
    gobot.import_mjcf_scene(str(source), "res://rp_arm.jscn", name="rp_arm")
    ctx = gobot.app.context()
    ctx.set_project_path(str(tmp_path))
    ctx.load_scene("res://rp_arm.jscn")
    try:
        snap = ctx.compile_physics_scene_snapshot()
    finally:
        ctx.clear_scene()
    for robot in snap["robots"]:
        for joint in robot["joints"]:
            joint["velocity_limit"] = 10.0
            joint["acceleration_limit"] = 50.0
    world = PlanningWorld.from_snapshot(snap, {"rp_arm": "base"})
    assert not np.isfinite(world.lower_limits).any() and not np.isfinite(world.upper_limits).any()
    # At u = 0.5: q_rev = pi/2, q_pris = 1.0 m -> tip at (0, 1.0, 0), r = 0.05 -> 10 mm penetration into box at y=1.06.
    world.add_box("post", (0.04, 0.04, 0.20), (0.0, 1.06, 0.0))

    traj = Trajectory(
        times=np.array([0.0, 1.0], dtype=np.float64),
        positions=np.array([[0.0, 0.0], [math.pi, 0.0]], dtype=np.float64),
        velocities=np.array([[0.0, 4.0], [0.0, -4.0]], dtype=np.float64),
        accelerations=np.zeros((2, 2), dtype=np.float64),
    )
    for sub in (1, 2, 4, 8, 16, 24):
        ok, err = world.validate_trajectory(traj, margin=0.0, world_margin=0.0, collision_substeps=sub)
        assert not ok, f"CCD missed 10 mm mid-segment Hermite penetration at substeps={sub}"
        assert err is not None and "clearance" in err


def test_validate_trajectory_snapshot_isolation_and_frame_shape_guard(simple_snapshot):
    """validate_trajectory must pin one immutable snapshot and reject mismatched frames shapes safely."""
    import threading

    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    # Trajectory along x in [-0.5, 0.5], y = 0.35, z = 0.5 (tip sphere r=0.05).
    q_path = np.array([[-0.5, 0.35, 0.0], [0.5, 0.35, 0.0]], dtype=np.float64)

    # State A: obs_a blocks left half (x=-0.25), obs_b parked far away.
    # State B: obs_a parked far away, obs_b blocks right half (x=+0.25).
    # In EVERY valid snapshot (A or B), penetration is ~65 mm, so validate_trajectory must NEVER pass!
    world.add_box("obs_a", (0.14, 0.14, 0.20), (-0.25, 0.35, 0.5))
    world.add_box("obs_b", (0.14, 0.14, 0.20), (2.50, 2.50, 0.5))

    # Defense-in-depth: frames built before adding a shape must fail shape check on newer world,
    # while remaining valid on the pinned WorldSnapshot.
    pinned = world._native.snapshot()
    (old_frames,) = pinned.compute_frames(q_path, False)
    world.add_sphere("extra_obs", 0.05, (2.0, -2.0, 0.5))
    with pytest.raises(ValueError, match="frames shape mismatch"):
        world._native.screen_aabbs(old_frames, world.pair_margins(0.0, 0.0), False, 1)
    # Same frames still work cleanly against the pinned snapshot:
    pinned.screen_aabbs(old_frames, pinned.pair_margins(0.0, 0.0), False, 1)
    world.remove_obstacle("extra_obs")

    stop = threading.Event()
    errors: list[Exception] = []

    def mutator():
        toggle = False
        try:
            while not stop.is_set():
                toggle = not toggle
                if toggle:
                    world.update_obstacle_pose("obs_b", (0.25, 0.35, 0.5))
                    world.update_obstacle_pose("obs_a", (2.50, 2.50, 0.5))
                    world.add_sphere("dyn_tmp", 0.03, (2.0, 2.0, 0.5))
                else:
                    world.update_obstacle_pose("obs_a", (-0.25, 0.35, 0.5))
                    world.update_obstacle_pose("obs_b", (2.50, 2.50, 0.5))
                    world.remove_obstacle("dyn_tmp")
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=mutator, daemon=True)
    t.start()
    try:
        for _ in range(150):
            ok, err = world.validate_trajectory(q_path, margin=0.0, world_margin=0.0, collision_substeps=4)
            assert not ok, "validate_trajectory falsely passed due to torn snapshot read!"
            assert err is not None
    finally:
        stop.set()
        t.join(timeout=2.0)
    assert not errors, f"mutator thread raised: {errors}"


def _make_tf(pos, rot=None):
    mat = np.eye(4, dtype=np.float64)
    if rot is not None:
        mat[:3, :3] = np.asarray(rot, dtype=np.float64)
    mat[:3, 3] = np.asarray(pos, dtype=np.float64)
    return {"matrix": mat.tolist()}


def test_sync_from_snapshot_full_acceptance(simple_snapshot, monkeypatch):
    """Comprehensive acceptance test for PlanningWorld.sync_from_snapshot."""
    import copy
    import gobot.planning.world as world_mod

    # Add two box obstacles + one triangle-mesh obstacle with stable_ids into a mutable snapshot copy.
    cube_verts = np.array([
        [-0.05, -0.05, -0.05], [0.05, -0.05, -0.05], [0.05, 0.05, -0.05], [-0.05, 0.05, -0.05],
        [-0.05, -0.05,  0.05], [0.05, -0.05,  0.05], [0.05, 0.05,  0.05], [-0.05, 0.05,  0.05],
    ], dtype=np.float64)
    cube_idx = np.array([
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6,
        1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3,
    ], dtype=np.int64)

    snap0 = copy.deepcopy(simple_snapshot)
    snap0["loose_shapes"] = [
        {
            "name": "box_a", "path": "World/box_a", "stable_id": "sid_box_a",
            "type": 1, "size": [0.12, 0.12, 0.20], "transform": _make_tf([0.40, 0.25, 0.50]),
            "collision_layer": 1, "collision_mask": 1, "disabled": False,
        },
        {
            "name": "box_b", "path": "World/box_b", "stable_id": "sid_box_b",
            "type": 1, "size": [0.10, 0.10, 0.10], "transform": _make_tf([-0.40, 0.25, 0.50]),
            "collision_layer": 1, "collision_mask": 1, "disabled": False,
        },
        {
            "name": "mesh_c", "path": "World/mesh_c", "stable_id": "sid_mesh_c",
            "type": 5, "vertices": cube_verts.tolist(), "indices": cube_idx.tolist(),
            "geometry_version": 1,
            "transform": _make_tf([0.0, 0.60, 0.50]),
            "collision_layer": 1, "collision_mask": 1, "disabled": False,
        },
    ]

    world = PlanningWorld.from_snapshot(snap0, {"planning": "base"})
    world.define_group("xy_only", [("planning", "x"), ("planning", "y")])

    # Instrument _make_native_shape to prove pure pose/filter syncs NEVER rebuild convex hulls or BVHs.
    build_calls = []
    orig_make = world_mod._make_native_shape

    def counting_make(spec, label):
        build_calls.append(label)
        return orig_make(spec, label)

    monkeypatch.setattr(world_mod, "_make_native_shape", counting_make)

    # 1. No-op sync: must NOT publish a new version or rebuild geometry.
    v_init = world.version
    tv_init = world.topology_version
    res_noop = world.sync_from_snapshot(snap0)
    assert not res_noop.changed
    assert res_noop.version == v_init
    assert res_noop.topology_version == tv_init
    assert build_calls == []

    # 2. Pure pose update + robot base move + rename/reparent (same stable_id):
    #    exactly ONE new snapshot version, ZERO topology_version bump, ZERO geometry rebuilds!
    snap1 = copy.deepcopy(snap0)
    snap1["loose_shapes"][0]["transform"] = _make_tf([0.20, 0.35, 0.50])
    snap1["loose_shapes"][0]["name"] = "box_a_renamed"
    snap1["loose_shapes"][0]["path"] = "World/Group/box_a_renamed"
    snap1["loose_shapes"][2]["transform"] = _make_tf([0.10, 0.55, 0.50])
    # Shift robot base + all robot links & joints rigidly by (+0.05, 0, 0).
    shift = np.eye(4, dtype=np.float64)
    shift[0, 3] = 0.05
    for rob in snap1["robots"]:
        if rob["name"] == "planning":
            for lk in rob["links"]:
                lk["global_transform"]["matrix"] = (shift @ np.asarray(lk["global_transform"]["matrix"])).tolist()
                for sh in lk.get("shapes", ()):
                    sh["transform"]["matrix"] = (shift @ np.asarray(sh["transform"]["matrix"])).tolist()
            for jt in rob["joints"]:
                jt["global_transform"]["matrix"] = (shift @ np.asarray(jt["global_transform"]["matrix"])).tolist()

    res_pose = world.sync_from_snapshot(snap1)
    assert res_pose.changed
    assert res_pose.version == v_init + 1
    # box_a's path/link_key changed -> link_key metadata updated without rebuilding geometry/BVH.
    assert res_pose.topology_version == tv_init + 1
    assert res_pose.poses_updated == 1
    assert res_pose.robot_bases_updated == 1
    assert res_pose.shapes_added == 0
    assert res_pose.shapes_removed == 0
    assert res_pose.shapes_replaced == 1
    assert build_calls == [], f"Pure pose/rename sync rebuilt geometry: {build_calls}"

    # Verify equivalence against full rebuild at same configuration.
    q_test = np.array([0.15, 0.20, 0.40], dtype=np.float64)
    rebuilt1 = PlanningWorld.from_snapshot(snap1, {"planning": "base"})
    pos_inc, rot_inc = world.forward_kinematics(q_test)
    pos_full, rot_full = rebuilt1.forward_kinematics(q_test)
    assert np.allclose(pos_inc, pos_full, atol=1e-12)
    assert np.allclose(rot_inc, rot_full, atol=1e-12)
    assert np.allclose(np.sort(world.collision_distances(q_test)), np.sort(rebuilt1.collision_distances(q_test)), atol=1e-12)

    # 3. Filter-only change (collision_mask -> 0): must NOT rebuild mesh geometry, MUST update active pairs.
    build_calls.clear()
    snap2 = copy.deepcopy(snap1)
    snap2["loose_shapes"][2]["collision_layer"] = 0
    snap2["loose_shapes"][2]["collision_mask"] = 0
    res_filter = world.sync_from_snapshot(snap2)
    assert res_filter.changed
    assert res_filter.shapes_replaced == 1
    assert res_filter.topology_version == tv_init + 2
    assert build_calls == [], f"Filter-only sync rebuilt mesh geometry: {build_calls}"
    rebuilt2 = PlanningWorld.from_snapshot(snap2, {"planning": "base"})
    assert world.num_pairs == rebuilt2.num_pairs
    assert np.allclose(np.sort(world.collision_distances(q_test)), np.sort(rebuilt2.collision_distances(q_test)), atol=1e-12)

    # 4. Add + disable (remove) + geometry replace (size + mesh geometry_version):
    build_calls.clear()
    snap3 = copy.deepcopy(snap2)
    snap3["loose_shapes"][0]["size"] = [0.24, 0.12, 0.20]   # geometry replace
    snap3["loose_shapes"][1]["disabled"] = True              # remove
    snap3["loose_shapes"][2]["collision_layer"] = 1
    snap3["loose_shapes"][2]["collision_mask"] = 1
    snap3["loose_shapes"][2]["vertices"] = (cube_verts * 1.5).tolist()
    snap3["loose_shapes"][2]["geometry_version"] = 2         # geometry replace
    snap3["loose_shapes"].append({
        "name": "sphere_d", "path": "World/sphere_d", "stable_id": "sid_sphere_d",
        "type": 2, "radius": 0.08, "transform": _make_tf([0.0, -0.45, 0.50]),
        "collision_layer": 1, "collision_mask": 1, "disabled": False,
    })
    res_topo = world.sync_from_snapshot(snap3)
    assert res_topo.changed
    assert res_topo.shapes_added == 1
    assert res_topo.shapes_removed == 1
    assert res_topo.shapes_replaced == 2
    assert len(build_calls) == 3  # 2 geometry replacements + 1 addition
    rebuilt3 = PlanningWorld.from_snapshot(snap3, {"planning": "base"})
    assert world.num_pairs == rebuilt3.num_pairs
    assert np.allclose(np.sort(world.collision_distances(q_test)), np.sort(rebuilt3.collision_distances(q_test)), atol=1e-12)

    # 5. Attached object preservation + User ACM override + PlanningGroup preservation:
    #    Attach World/sphere_d to left_tip, disable World/box_a via user ACM, then move sphere_d in snapshot!
    world.attach_object("World/sphere_d", ("planning", "left_tip"), local_position=(0.0, 0.0, 0.10))
    world.set_collision_enabled(("planning", "left_tip"), "World/box_a", False)
    pairs_before = world.num_pairs

    snap4 = copy.deepcopy(snap3)
    # Stale scene moves sphere_d far away (or even deletes it) -> attached payload must NOT be touched or duplicated!
    snap4["loose_shapes"][-1]["transform"] = _make_tf([5.0, 5.0, 5.0])
    v_before_att = world.version
    res_att = world.sync_from_snapshot(snap4)
    assert not res_att.changed
    assert res_att.version == v_before_att
    assert world.num_pairs == pairs_before
    # PlanningGroup still intact and inactive joint stays part of robot (frozen, not static obstacle).
    mask_xy = world.group_mask("xy_only")
    assert mask_xy.tolist() == [True, True, False]

    # Detach sphere_d -> next sync reconciles its world pose to snap4!
    world.detach_object("World/sphere_d", q_test)
    res_det = world.sync_from_snapshot(snap4)
    assert res_det.changed
    assert res_det.poses_updated == 1

    # 6. Atomic failure: robot topology change or invalid transform must throw and leave world completely unchanged!
    v_safe = world.version
    tv_safe = world.topology_version
    dists_safe = world.collision_distances(q_test).copy()

    snap_bad_topo = copy.deepcopy(snap4)
    snap_bad_topo["loose_shapes"][0]["transform"] = _make_tf([9.0, 9.0, 9.0])
    for rob in snap_bad_topo["robots"]:
        if rob["name"] == "planning":
            rob["joints"].pop()  # remove joint -> topology change!
    with pytest.raises(ValueError, match="robot topology changed"):
        world.sync_from_snapshot(snap_bad_topo)
    assert world.version == v_safe
    assert world.topology_version == tv_safe
    assert np.allclose(world.collision_distances(q_test), dists_safe, atol=1e-12)

    # 7. PlanResult & BatchPlanResult record exact world_version.
    planner = MotionPlanner(world, backend="coal")
    req = MotionPlanRequest(
        start=np.array([0.0, 0.0, 0.0]),
        goal=np.array([0.2, 0.1, 0.0]),
        group="xy_only",
        config=SolverConfig(waypoints=8, margin=0.002, world_margin=0.005),
    )
    pres = planner.plan(req)
    assert pres.success
    assert pres.world_version == world.version
    bres = planner.plan_batch([req, req])
    assert bres.world_version == world.version
    assert all(r.world_version == world.version for r in bres.results)


def test_sync_from_scene_live_engine_context(tmp_path):
    """Verify PlanningWorld.from_scene and sync_from_scene against a live Gobot EngineContext."""
    xml = """<mujoco model="sync_scene_test">
      <worldbody>
        <body name="base" pos="0 0 0">
          <body name="slider" pos="0 0 0.2">
            <joint name="jx" type="slide" axis="1 0 0" range="-1 1"/>
            <geom name="tip" type="sphere" size="0.05"/>
          </body>
        </body>
        <geom name="wall" type="box" pos="0.5 0 0.2" size="0.05 0.1 0.1"/>
      </worldbody>
    </mujoco>"""
    xml_path = tmp_path / "sync_scene_test.xml"
    xml_path.write_text(xml, encoding="utf-8")

    ctx = gobot.app.context()
    prev = ctx.project_path
    try:
        gobot.set_project_path(str(tmp_path))
        gobot.import_mjcf_scene(str(xml_path), "res://sync_scene_test.jscn", name="sync_scene_test")
        ctx.set_project_path(str(tmp_path))
        ctx.load_scene("res://sync_scene_test.jscn")

        world = PlanningWorld.from_scene(ctx, {"sync_scene_test": "base"})
        v0 = world.version

        # No scene edits -> no-op sync.
        r0 = world.sync_from_scene(ctx)
        assert not r0.changed
        assert r0.version == v0
        assert r0.export_ms >= 0.0

        # Move the wall obstacle node directly in the live Gobot Scene!
        root = ctx.root
        wall_node = None
        stack = [root]
        while stack:
            cur = stack.pop()
            if "wall" in cur.name.lower():
                wall_node = cur
                break
            stack.extend(cur.children)
        assert wall_node is not None, "could not find wall node in imported scene"
        old_pos = list(wall_node.position)
        wall_node.position = [old_pos[0] + 0.30, old_pos[1], old_pos[2]]

        r1 = world.sync_from_scene(ctx)
        assert r1.changed
        assert r1.version == v0 + 1
        assert r1.poses_updated == 1
        assert r1.shapes_added == 0 and r1.shapes_removed == 0 and r1.shapes_replaced == 0

        # Distance at q=0 must match a fresh PlanningWorld.from_scene(ctx, ...).
        q0 = np.array([0.0], dtype=np.float64)
        fresh = PlanningWorld.from_scene(ctx, {"sync_scene_test": "base"})
        assert np.allclose(world.collision_distances(q0), fresh.collision_distances(q0), atol=1e-12)
    finally:
        ctx.clear_scene()
        if prev:
            ctx.set_project_path(prev)


def test_quintic_c2_spline_jerk_limits_and_counterexamples(simple_snapshot):
    """Verify shared C^2 quintic spline evaluation, global time scaling, jerk limits, and counterexamples."""
    from gobot.planning.retime import check_trajectory_bounds

    # 1. Standalone C^2 quintic retiming with global time scaling & stop_at_endpoints=True.
    wp = np.array([
        [0.0, 0.0, 0.0],
        [0.2, 0.15, 0.3],
        [0.45, 0.25, -0.2],
        [0.6, 0.10, 0.1],
    ], dtype=np.float64)
    vmax = np.array([1.5, 1.5, 2.0])
    amax = np.array([4.0, 4.0, 5.0])
    jmax = np.array([15.0, 15.0, 20.0])

    traj_loose = retime_trajectory(wp, vmax, amax, stop_at_endpoints=True, spline_order=5)
    traj_jerk = retime_trajectory(wp, vmax, amax, stop_at_endpoints=True, jerk_limits=jmax, spline_order=5)
    assert traj_jerk.spline_order == 5
    assert traj_jerk.duration >= traj_loose.duration

    # Verify global scaling preserves exact geometric path q(s) for normalized progress s in [0, 1]!
    s = np.linspace(0.0, 1.0, 201)
    p_loose, _, _ = traj_loose.sample(s * traj_loose.duration)
    p_jerk, v_jerk, a_jerk, j_jerk = traj_jerk.sample_with_jerk(s * traj_jerk.duration)
    assert np.allclose(p_loose, p_jerk, atol=1e-10), "Global time scaling altered geometric path!"

    # Verify strict v, a, j bounds everywhere and stop_at_endpoints v=a=0.
    assert np.allclose(v_jerk[0], 0.0, atol=1e-10)
    assert np.allclose(a_jerk[0], 0.0, atol=1e-10)
    assert np.allclose(v_jerk[-1], 0.0, atol=1e-10)
    assert np.allclose(a_jerk[-1], 0.0, atol=1e-10)
    assert (np.abs(v_jerk) <= vmax[None, :] * (1.0 + 1e-5)).all()
    assert (np.abs(a_jerk) <= amax[None, :] * (1.0 + 1e-5)).all()
    assert (np.abs(j_jerk) <= jmax[None, :] * (1.0 + 1e-5)).all()

    # Verify C^2 continuity across interior knots & finite left/right jerk within limits.
    eps = 1e-7
    for tk in traj_jerk.times[1:-1]:
        pl, vl, al, jl = traj_jerk.sample_with_jerk(float(tk - eps))
        pr, vr, ar, jr = traj_jerk.sample_with_jerk(float(tk + eps))
        assert np.allclose(pl, pr, atol=1e-5)
        assert np.allclose(vl, vr, atol=1e-5)
        assert np.allclose(al, ar, atol=1e-3)
        assert (np.abs(jl) <= jmax * (1.0 + 1e-4)).all()
        assert (np.abs(jr) <= jmax * (1.0 + 1e-4)).all()

    ok, err, bad = check_trajectory_bounds(traj_jerk, vmax, amax, jerk_limits=jmax)
    assert ok and err is None and bad == []

    # 2. Counterexample A: knots strictly inside joint limits, mid-segment quintic exceeds upper limit!
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    world.set_jerk_limits([25.0, 25.0, 30.0])
    x_hi = float(world.upper_limits[0])
    bad_pos_traj = Trajectory(
        times=np.array([0.0, 1.0]),
        positions=np.array([[x_hi - 0.05, 0.2, 0.0], [x_hi - 0.05, 0.2, 0.0]]),
        velocities=np.array([[0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]]),  # mid-segment overshoots x_hi by +0.20 m!
        accelerations=np.zeros((2, 3)),
        spline_order=5,
    )
    ok_pos, err_pos = world.validate_trajectory(bad_pos_traj, collision_substeps=1)
    assert not ok_pos
    assert err_pos is not None and "position" in err_pos and "segment 0" in err_pos and "joint 0" in err_pos

    # 3. Counterexample B: fixed dt where v & a are well inside limits, ONLY jerk exceeds limit!
    world_tight_j = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    world_tight_j.set_jerk_limits([2.0, 2.0, 2.0])
    wp_2 = np.array([[0.0, 0.2, 0.0], [0.1, 0.2, 0.0]])
    traj_fixed = retime_trajectory(
        wp_2, world_tight_j.velocity_limits, world_tight_j.acceleration_limits,
        stop_at_endpoints=True, dt=0.5, jerk_limits=world_tight_j.jerk_limits, spline_order=5,
    )
    assert np.isclose(traj_fixed.duration, 0.5)
    ok_j, err_j = world_tight_j.validate_trajectory(traj_fixed, collision_substeps=4)
    assert not ok_j
    assert err_j is not None and "jerk" in err_j and "segment 0" in err_j and "joint 0" in err_j

    # 4. End-to-end MotionPlanner.plan with C^2 quintic spline + jerk limits + strict_ccd=True!
    world.add_box("mid_post", (0.12, 0.12, 0.25), (0.0, 0.35, 0.50))
    planner = MotionPlanner(world, backend="coal")
    req = MotionPlanRequest(
        start=np.array([-0.45, 0.35, 0.0]),
        goal=np.array([0.45, 0.35, 0.0]),
        config=SolverConfig(
            waypoints=12,
            margin=0.005,
            world_margin=0.015,
            collision_substeps=6,
            stop_at_endpoints=True,
            spline_order=5,
            strict_ccd=True,
        ),
    )
    res = planner.plan(req)
    assert res.success, res.reason
    assert res.trajectory is not None and res.trajectory.spline_order == 5
    assert res.diagnostics.retime_ms >= 0.0
    ok_val, err_val = world.validate_trajectory(
        res.trajectory, margin=0.005, world_margin=0.015, strict_ccd=True,
    )
    assert ok_val, err_val


def test_review_regressions_quintic_peaks_concurrent_sync_and_attached_rename(simple_snapshot, tmp_path):
    """Regression tests for all P1/P2 findings from 2.1 + 2.2 review."""
    import copy
    import threading
    from gobot.planning.retime import check_trajectory_bounds

    # 1. P1: Quintic near-zero c5 interior acceleration peak (a(0)=0, a(1)=4e-15, a(0.5)=-3.0 > 2.0).
    traj_acc = Trajectory(
        times=np.array([0.0, 1.0]),
        positions=np.array([[0.0], [0.0]]),
        velocities=np.array([[1.0], [-1.0]]),
        accelerations=np.array([[0.0], [4e-15]]),
        spline_order=5,
    )
    ok_acc, err_acc, _ = check_trajectory_bounds(traj_acc, np.array([10.0]), np.array([2.0]))
    assert not ok_acc and err_acc is not None and "acceleration" in err_acc

    # 2. P1/P2: Wrong-length limit vectors & zero-duration moving segment & invalid SolverConfig(spline_order=4).
    with pytest.raises(ValueError, match="length"):
        check_trajectory_bounds(traj_acc, np.array([10.0, 10.0]), np.array([2.0]))
    with pytest.raises(ValueError, match="length"):
        check_trajectory_bounds(traj_acc, np.array([10.0]), np.array([2.0]), jerk_limits=np.array([5.0, 5.0]))
    traj_zdt = Trajectory(
        times=np.array([0.0, 0.0]),
        positions=np.array([[0.0], [0.5]]),
        velocities=np.zeros((2, 1)),
        accelerations=np.zeros((2, 1)),
        spline_order=5,
    )
    ok_zdt, err_zdt, _ = check_trajectory_bounds(traj_zdt, np.array([10.0]), np.array([10.0]))
    assert not ok_zdt and err_zdt is not None and "Zero-duration" in err_zdt
    with pytest.raises(ValueError, match="spline_order"):
        SolverConfig(spline_order=4)

    # 3. P2: Snapshot joint jerk_limit change must be detected by sync_from_snapshot.
    snap_j0 = copy.deepcopy(simple_snapshot)
    snap_j0["robots"][0]["joints"][0]["jerk_limit"] = 20.0
    world_j = PlanningWorld.from_snapshot(snap_j0, {"planning": "base"})
    snap_j1 = copy.deepcopy(snap_j0)
    snap_j1["robots"][0]["joints"][0]["jerk_limit"] = 5.0
    with pytest.raises(ValueError, match="jerk"):
        world_j.sync_from_snapshot(snap_j1)

    # 4. P1: Two concurrent sync_from_snapshot writers must never publish a mixed/hybrid scene!
    #    State A: both boxes at y = +0.55. State B: both boxes at y = -0.55.
    #    At q = [0, 0, 0] (tip at (0, 0, 0.5)), every published snapshot MUST have equal distance to both boxes
    #    (never one box at +0.55 and the other at -0.55, or one at +0.55 and one at +1.55!).
    snap_a = copy.deepcopy(simple_snapshot)
    snap_a["loose_shapes"] = [
        {"name": "b1", "path": "World/b1", "stable_id": "s1", "type": 1, "size": [0.1, 0.1, 0.1],
         "transform": _make_tf([0.3, 0.40, 0.50]), "collision_layer": 1, "collision_mask": 1, "disabled": False},
        {"name": "b2", "path": "World/b2", "stable_id": "s2", "type": 1, "size": [0.1, 0.1, 0.1],
         "transform": _make_tf([-0.3, 0.40, 0.50]), "collision_layer": 1, "collision_mask": 1, "disabled": False},
    ]
    snap_b = copy.deepcopy(snap_a)
    snap_b["loose_shapes"][0]["transform"] = _make_tf([0.3, 1.40, 0.50])
    snap_b["loose_shapes"][1]["transform"] = _make_tf([-0.3, 1.40, 0.50])

    world_c = PlanningWorld.from_snapshot(snap_a, {"planning": "base"})
    q0 = np.array([0.0, 0.0, 0.0])
    d_a = np.sort(world_c.collision_distances(q0))
    d_b = np.sort(PlanningWorld.from_snapshot(snap_b, {"planning": "base"}).collision_distances(q0))

    stop = threading.Event()
    errs = []

    def writer(snap_target):
        try:
            for _ in range(80):
                if stop.is_set():
                    break
                world_c.sync_from_snapshot(snap_target)
        except Exception as exc:
            errs.append(exc)

    ta = threading.Thread(target=writer, args=(snap_a,))
    tb = threading.Thread(target=writer, args=(snap_b,))
    ta.start()
    tb.start()
    while ta.is_alive() or tb.is_alive():
        d_now = np.sort(world_c.collision_distances(q0))
        ok_a = np.allclose(d_now, d_a, atol=1e-9)
        ok_b = np.allclose(d_now, d_b, atol=1e-9)
        assert ok_a or ok_b, f"Observed mixed/hybrid scene from concurrent sync writers: {d_now} vs A={d_a}, B={d_b}"
    ta.join()
    tb.join()
    assert not errs, f"Concurrent sync writer raised: {errs}"

    # 5. P1: Renaming a real attached obstacle node in a live Gobot Scene must NOT duplicate its collision shape!
    xml = """<mujoco model="rename_attached_test">
      <worldbody>
        <body name="base" pos="0 0 0">
          <body name="slider" pos="0 0 0.2">
            <joint name="jx" type="slide" axis="1 0 0" range="-1 1"/>
            <geom name="tip" type="sphere" size="0.05"/>
          </body>
        </body>
        <geom name="payload" type="box" pos="0.5 0 0.2" size="0.05 0.05 0.05"/>
      </worldbody>
    </mujoco>"""
    xml_path = tmp_path / "rename_attached_test.xml"
    xml_path.write_text(xml, encoding="utf-8")

    ctx = gobot.app.context()
    prev = ctx.project_path
    try:
        gobot.set_project_path(str(tmp_path))
        gobot.import_mjcf_scene(str(xml_path), "res://rename_attached_test.jscn", name="rename_attached_test")
        ctx.set_project_path(str(tmp_path))
        ctx.load_scene("res://rename_attached_test.jscn")

        w_live = PlanningWorld.from_scene(ctx, {"rename_attached_test": "base"})
        payload_shape_name = [n for n in w_live._native.shape_names() if "payload" in n.lower()]
        assert len(payload_shape_name) == 1
        w_live.attach_object(payload_shape_name[0], ("rename_attached_test", "slider"), local_position=(0.0, 0.0, 0.12))
        shapes_before = list(w_live._native.shape_names())

        # Rename the real Gobot scene node!
        payload_node = None
        stack = [ctx.root]
        while stack:
            cur = stack.pop()
            if "payload" in cur.name.lower():
                payload_node = cur
                break
            stack.extend(cur.children)
        assert payload_node is not None
        payload_node.name = "payload_renamed"

        sync_res = w_live.sync_from_scene(ctx)
        assert not sync_res.changed, f"Renaming attached real node unexpectedly mutated world: {sync_res}"
        assert sync_res.shapes_added == 0
        assert list(w_live._native.shape_names()) == shapes_before
        assert payload_shape_name[0] in w_live._native.attached_shape_names()

        # 6. P1: Mixing from_scene + sync_from_snapshot (without node_id) must NOT duplicate the attached shape!
        raw_snap = ctx.compile_physics_scene_snapshot()
        mix_res = w_live.sync_from_snapshot(raw_snap)
        assert not mix_res.changed, f"Mixing from_scene and sync_from_snapshot duplicated/mutated world: {mix_res}"
        assert mix_res.shapes_added == 0
        assert list(w_live._native.shape_names()) == shapes_before
    finally:
        ctx.clear_scene()
        if prev:
            ctx.set_project_path(prev)

    # 7. P1: Reparenting a shape to a different link must update link_key and collision filtering!
    snap_lk0 = copy.deepcopy(simple_snapshot)
    snap_lk0["loose_shapes"] = [
        {"name": "panel", "path": "World/link_a", "stable_id": "sid_panel", "type": 1,
         "size": [0.10, 0.10, 0.20], "transform": _make_tf([0.0, 0.35, 0.50]),
         "collision_layer": 1, "collision_mask": 1, "disabled": False},
    ]
    w_lk = PlanningWorld.from_snapshot(snap_lk0, {"planning": "base"})
    w_lk.set_collision_enabled(("planning", "left_tip"), "World/link_a", False)
    pairs_disabled = w_lk.num_pairs
    snap_lk1 = copy.deepcopy(snap_lk0)
    snap_lk1["loose_shapes"][0]["path"] = "World/link_b"  # moved to link_b, where collision is NOT disabled!
    r_lk = w_lk.sync_from_snapshot(snap_lk1)
    assert r_lk.changed and r_lk.shapes_replaced == 1
    assert w_lk.num_pairs > pairs_disabled, "Reparented obstacle kept stale link_key ACM suppression!"

    # 8. P1: validate_trajectory must pin ONE snapshot across jerk_limits and collision checks!
    #    State A: tight jerk limit (traj fails jerk), obstacle far away.
    #    State B: loose jerk limit (traj passes jerk), obstacle dead-on blocking path.
    #    Every valid world version rejects this trajectory; cross-version tearing would falsely pass!
    w_pin = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    w_pin.add_box("gate", (0.14, 0.14, 0.20), (0.0, 0.35, 0.50))
    traj_pin = retime_trajectory(
        np.array([[-0.35, 0.35, 0.0], [0.35, 0.35, 0.0]]),
        w_pin.velocity_limits, w_pin.acceleration_limits, stop_at_endpoints=True, spline_order=5,
    )
    w_pin.set_jerk_limits(1000.0)
    # Verify State B (loose jerk + blocking box) fails collision, and State A (tight jerk + far box) fails jerk.
    ok_b, _ = w_pin.validate_trajectory(traj_pin, margin=0.0, world_margin=0.0)
    assert not ok_b
    w_pin.update_obstacle_pose("gate", (3.0, 3.0, 0.50))
    w_pin.set_jerk_limits(0.1)
    ok_a, _ = w_pin.validate_trajectory(traj_pin, margin=0.0, world_margin=0.0)
    assert not ok_a
    # Prove validate_trajectory reads limits from the pinned C++ WorldSnapshot (not stale/mutated Python attrs)
    # and runs bounds + collision against that exact same snapshot even if another thread mutates world mid-call:
    w_pin.jerk_limits = np.full(3, np.inf)  # corrupt Python attr; pinned C++ snapshot still has 0.1!
    ok_a2, err_a2 = w_pin.validate_trajectory(traj_pin, margin=0.0, world_margin=0.0)
    assert not ok_a2 and err_a2 is not None and "jerk" in err_a2
    w_pin.set_jerk_limits(1000.0)

    # 9. P1/P2: Native quintic CCD missing spline params, empty samples, invalid jerk_scale & polyline dt,
    #    tiny dq jerk retime, tiny positive dt playback & rejection, and one-sided lower_limits enforcement.
    with pytest.raises(ValueError, match="accelerations"):
        w_pin._native.validate_trajectory_collision(
            traj_pin.positions, traj_pin.positions, 1, 0.0, 0.0, False, 5e-4,
            traj_pin.times, traj_pin.velocities, False, None, 5,
        )
    with pytest.raises(ValueError, match="accelerations"):
        w_pin._native.validate_trajectory_collision(
            traj_pin.positions, traj_pin.positions, 1, 0.0, 0.0, False, 5e-4,
            None, None, False, None, 5,
        )
    with pytest.raises(ValueError, match="samples must have shape"):
        w_pin._native.validate_trajectory_collision(
            traj_pin.positions, np.empty((0, 3), dtype=np.float64), 1, 0.0, 0.0, False, 5e-4,
        )
    with pytest.raises(ValueError, match="jerk_scale"):
        SolverConfig(jerk_scale=0.0)
    with pytest.raises(ValueError, match="jerk_scale"):
        w_pin.validate_trajectory(traj_pin, jerk_scale=-1.0)
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        w_pin.validate_trajectory(traj_pin.positions, dt=-0.1)
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        w_pin.validate_trajectory(traj_pin.positions, dt=float("nan"))

    # Tiny positive duration segment playback must reach exact endpoint.
    t_short_end = Trajectory(
        times=np.array([0.0, 5e-13]),
        positions=np.array([[0.0], [1.0]]),
        velocities=np.zeros((2, 1)),
        accelerations=np.zeros((2, 1)),
        spline_order=5,
    )
    p_end_sample, _, _ = t_short_end.sample(5e-13)
    assert np.allclose(p_end_sample, [1.0], atol=1e-12)

    # Tiny non-zero displacement (5e-13) with finite jerk limit must be retimed to satisfy jerk bounds.
    t_tiny = retime_trajectory(
        np.array([[0.0], [5e-13]]), np.array([2.0]), np.array([5.0]),
        stop_at_endpoints=True, jerk_limits=np.array([5.0]), spline_order=5,
    )
    ok_tiny, err_tiny, _ = check_trajectory_bounds(
        t_tiny, np.array([2.0]), np.array([5.0]), jerk_limits=np.array([5.0]),
    )
    assert ok_tiny, err_tiny

    # Tiny positive duration segment (5e-13 s) with non-zero displacement (5e-13 m) must NOT bypass bounds!
    t_short = Trajectory(
        times=np.array([0.0, 5e-13]),
        positions=np.array([[0.0], [5e-13]]),
        velocities=np.zeros((2, 1)),
        accelerations=np.zeros((2, 1)),
        spline_order=5,
    )
    ok_short, _, _ = check_trajectory_bounds(
        t_short, np.array([2.0]), np.array([5.0]), jerk_limits=np.array([5.0]),
    )
    assert not ok_short

    # One-sided lower_limits only must be enforced (never silently ignored).
    t_lo = Trajectory(
        times=np.array([0.0, 1.0]),
        positions=np.array([[-0.5], [-0.5]]),
        velocities=np.zeros((2, 1)),
        accelerations=np.zeros((2, 1)),
        spline_order=5,
    )
    ok_lo, err_lo, _ = check_trajectory_bounds(
        t_lo, np.array([2.0]), np.array([5.0]), lower_limits=np.array([0.0]),
    )
    assert not ok_lo and err_lo is not None and "position" in err_lo


def test_tiny_positive_duration_ccd_matches_playback_for_cubic_and_quintic(simple_snapshot):
    """Time-scaling a colliding spline from h=1.0 s down to h=5e-13 s (scaling v->v/h) preserves
    the exact geometric curve; strict_ccd=True must catch the mid-segment penetration for both
    cubic (order=3) and quintic (order=5) trajectories."""
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    # Endpoints at q = (0.0, 0.35, 0.0) are completely collision-free (~0.215 m clearance).
    # Midpoint dips to y = 0.078 (penetration ~57 mm into obstacle at y in [-0.21, 0.09] with sphere r=0.045).
    q_end = np.array([0.0, 0.35, 0.0], dtype=np.float64)
    y_mid_target = 0.078
    dip = float(q_end[1] - y_mid_target)
    assert float(world.collision_distances(q_end).min()) > 0.15

    for order in (3, 5):
        # cubic:   p(0.5) - p0 = -(h/4)*(v_base/h) = -dip   -> v_base = 4 * dip
        # quintic: p(0.5) - p0 = -(5*h/16)*(v_base/h) = -dip -> v_base = (16/5) * dip
        v_base = (3.2 if order == 5 else 4.0) * dip
        mid_ref = None
        for h_scale in (1.0, 5e-13):
            traj = Trajectory(
                times=np.array([0.0, h_scale], dtype=np.float64),
                positions=np.vstack([q_end, q_end]),
                velocities=np.array([[0.0, -v_base / h_scale, 0.0], [0.0, v_base / h_scale, 0.0]], dtype=np.float64),
                accelerations=np.zeros((2, 3), dtype=np.float64),
                spline_order=order,
            )
            mid_pos, _, _ = traj.sample(0.5 * h_scale)
            assert math.isclose(float(mid_pos[1]), y_mid_target, abs_tol=1e-12)
            if mid_ref is None:
                mid_ref = mid_pos.copy()
            else:
                assert np.allclose(mid_pos, mid_ref, atol=1e-12)

            # Confirm midpoint penetrates ~57 mm into the obstacle.
            assert float(world.collision_distances(mid_pos).min()) < -0.05

            # Relax motion limits so validation isolates collision/CCD with substeps=1 (only safe endpoints sampled).
            ok, reason = world.validate_trajectory(
                traj,
                margin=0.0,
                world_margin=0.0,
                collision_substeps=1,
                strict_ccd=True,
                velocity_scale=1e30,
                acceleration_scale=1e30,
                jerk_scale=1e60,
            )
            assert not ok, f"CCD missed mid-segment penetration for order={order}, h={h_scale}"
            assert reason is not None and "clearance" in reason
