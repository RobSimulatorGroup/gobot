"""Comprehensive tests for Gobot's unified C++-backed motion planning architecture."""

from __future__ import annotations

import math
from pathlib import Path
import runpy
from time import perf_counter

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import gobot
from gobot.planning import (
    CartesianPlanConfig,
    CartesianWaypoint,
    MotionPlanner,
    MotionPlanRequest,
    MuJoCoWarpBackend,
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


def test_cartesian_linear_arc_and_tcp_offset_panda(panda_snapshot):
    """6D Cartesian MoveL + directed MoveC with TCP offset and orientation tracking on 7-DOF Panda (cubic & quintic)."""
    snap, (start_xyz, _, _) = panda_snapshot
    mounts = (("panda_link0", "panda_link2"), ("panda_link4", "panda_link6"), ("panda_link7", "panda_hand"))
    world = PlanningWorld.from_snapshot(
        snap, {"panda": "panda_link0"},
        ignored_link_pairs=[(("panda", a), ("panda", b)) for a, b in mounts],
    )
    planner = MotionPlanner(world, backend="coal")

    ee = ("panda", "panda_hand")
    down = np.diag([-1.0, 1.0, -1.0])
    seed = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, -2.3], dtype=np.float64)
    ik0 = planner.solve_ik(
        [PoseConstraint(ee, start_xyz + np.array([0.0, 0.0, 0.18]), down)],
        seed=seed, margin=0.002, world_margin=0.02,
    )
    assert ik0.success, ik0.reason
    q_start = ik0.positions

    tcp_pos_local = np.array([0.0, 0.0, 0.08], dtype=np.float64)
    tcp_rot_local = Rotation.from_euler("z", 0.3).as_matrix()

    def eval_tcp(q_arr):
        lp, lr = world.link_pose(q_arr, ee)
        if lp.ndim == 1:
            return lp + lr @ tcp_pos_local, lr @ tcp_rot_local
        return lp + np.einsum("nij,j->ni", lr, tcp_pos_local), lr @ tcp_rot_local

    p0, r0 = eval_tcp(q_start)
    p1 = p0 + np.array([-0.08, -0.06, 0.04], dtype=np.float64)
    r1 = r0 @ Rotation.from_euler("x", 0.12).as_matrix()

    # Directed circular arc from p1 through p_via to p2 in XZ plane (radius 0.045 m), staying clear on -Y side.
    arc_c = p1 + np.array([0.0, 0.0, 0.045], dtype=np.float64)
    p_via = arc_c + np.array([-0.045, 0.0, 0.0], dtype=np.float64)
    p2 = arc_c + np.array([0.0, 0.0, 0.045], dtype=np.float64)
    r2 = r1 @ Rotation.from_euler("y", -0.10).as_matrix()

    waypoints = [
        CartesianWaypoint(position=p1, rotation=r1, motion_type="linear", blend_radius=0.0),
        CartesianWaypoint(position=p2, rotation=r2, motion_type="circular", via_position=p_via, blend_radius=0.0),
    ]

    for order in (3, 5):
        cfg = CartesianPlanConfig(
            max_step=0.015,
            max_rot_step=0.05,
            max_pos_error=1e-3,
            max_rot_error=1e-2,
            spline_order=order,
            strict_ccd=True,
            margin=0.002,
            world_margin=0.002,
        )
        res = planner.plan_cartesian_path(
            q_start,
            waypoints,
            link=ee,
            tcp_offset_position=tcp_pos_local,
            tcp_offset_rotation=tcp_rot_local,
            config=cfg,
        )
        assert res.success, f"order={order}: {res.reason}"
        assert res.trajectory is not None
        assert res.fraction == pytest.approx(1.0)
        assert res.completed_waypoints == 2
        assert res.world_version == world.version
        assert res.diagnostics.total_ms > 0.0
        assert res.diagnostics.seed_strategy == "cartesian"
        assert res.max_pos_error <= cfg.max_pos_error + 1e-6
        assert res.max_rot_error <= cfg.max_rot_error + 1e-5

        # Independent dense verification of continuous spline curve + strict CCD validation.
        ts = np.linspace(res.trajectory.times[0], res.trajectory.times[-1], 160)
        q_dense, _, _ = res.trajectory.sample(ts)
        pts, _ = eval_tcp(q_dense)
        assert np.linalg.norm(pts[0] - p0) < 1e-6
        assert np.linalg.norm(pts[-1] - p2) < cfg.max_pos_error
        passed_via = False
        for pt in pts:
            d_lin = p1 - p0
            u_lin = float(np.clip(np.dot(pt - p0, d_lin) / np.dot(d_lin, d_lin), 0.0, 1.0))
            err_lin = float(np.linalg.norm(pt - (p0 + u_lin * d_lin)))
            rel = pt - arc_c
            err_arc = abs(math.hypot(float(rel[0]), float(rel[2])) - 0.045) + abs(float(rel[1]))
            if rel[0] > 2e-3:
                err_arc += 1.0  # Must follow directed arc through -X via point
            if float(np.linalg.norm(pt - p_via)) < 0.015:
                passed_via = True
            assert min(err_lin, err_arc) <= cfg.max_pos_error + 2e-4
        assert passed_via

        ok_val, err_val = world.validate_trajectory(
            res.trajectory,
            margin=cfg.margin,
            world_margin=cfg.effective_world_margin,
            strict_ccd=True,
        )
        assert ok_val, err_val


def test_cartesian_sharp_corner_vs_blend_and_coarse_spline_refinement(simple_snapshot, panda_snapshot):
    """Verify sharp-corner stop vs bounded corner blend, and continuous spline refinement when coarse knots bow off line."""
    # 1. Sharp vs blended 90-degree corner on simple_snapshot prismatic XY tip.
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    planner = MotionPlanner(world, backend="coal")
    tip = ("planning", "left_tip")
    q_start = np.array([-0.55, 0.30, 0.0], dtype=np.float64)
    p0, _ = world.link_pose(q_start, tip)
    p_corner = p0 + np.array([0.25, 0.0, 0.0])
    p_end = p_corner + np.array([0.0, 0.25, 0.0])

    res_sharp = planner.plan_cartesian_path(
        q_start,
        [CartesianWaypoint(p_corner, blend_radius=0.0), CartesianWaypoint(p_end, blend_radius=0.0)],
        link=tip,
        config=CartesianPlanConfig(max_step=0.05, max_pos_error=1e-3, spline_order=5),
    )
    assert res_sharp.success, res_sharp.reason
    assert res_sharp.trajectory is not None
    knot_pts, _ = world.link_pose(res_sharp.trajectory.positions, tip)
    corner_idx = int(np.argmin(np.linalg.norm(knot_pts - p_corner[None, :], axis=-1)))
    assert 0 < corner_idx < res_sharp.trajectory.num_waypoints - 1
    assert np.linalg.norm(res_sharp.trajectory.velocities[corner_idx]) < 1e-12
    assert np.linalg.norm(res_sharp.trajectory.accelerations[corner_idx]) < 1e-12

    res_blend = planner.plan_cartesian_path(
        q_start,
        [CartesianWaypoint(p_corner, blend_radius=0.06), CartesianWaypoint(p_end, blend_radius=0.0)],
        link=tip,
        config=CartesianPlanConfig(max_step=0.03, max_pos_error=1e-3, spline_order=5),
    )
    assert res_blend.success, res_blend.reason
    assert res_blend.trajectory is not None
    ts = np.linspace(res_blend.trajectory.times[0], res_blend.trajectory.times[-1], 120)
    qs, vs, _ = res_blend.trajectory.sample(ts)
    pts, _ = world.link_pose(qs, tip)
    dists_to_corner = np.linalg.norm(pts - p_corner[None, :], axis=-1)
    min_i = int(np.argmin(dists_to_corner))
    assert 0.005 < float(dists_to_corner[min_i]) <= 0.06
    assert float(np.linalg.norm(vs[min_i])) > 1e-3

    # 2. Coarse max_step on 7-DOF Panda: raw 2-knot joint chord bows off straight line,
    # planner must catch and subdivide so actual spline stays within max_pos_error=1e-3 everywhere.
    psnap, (pstart_xyz, _, _) = panda_snapshot
    mounts = (("panda_link0", "panda_link2"), ("panda_link4", "panda_link6"), ("panda_link7", "panda_hand"))
    pworld = PlanningWorld.from_snapshot(
        psnap, {"panda": "panda_link0"},
        ignored_link_pairs=[(("panda", a), ("panda", b)) for a, b in mounts],
    )
    pplanner = MotionPlanner(pworld, backend="coal")
    ee = ("panda", "panda_hand")
    down = np.diag([-1.0, 1.0, -1.0])
    pseed = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, -2.3], dtype=np.float64)
    pik = pplanner.solve_ik(
        [PoseConstraint(ee, pstart_xyz + np.array([0.0, 0.0, 0.18]), down)],
        seed=pseed, margin=0.002, world_margin=0.02,
    )
    assert pik.success, pik.reason
    pq0 = pik.positions
    pp0, pr0 = pworld.link_pose(pq0, ee)
    pp1 = pp0 + np.array([-0.14, -0.06, 0.08], dtype=np.float64)
    pres = pplanner.plan_cartesian_path(
        pq0,
        [CartesianWaypoint(pp1, rotation=pr0)],
        link=ee,
        config=CartesianPlanConfig(max_step=0.5, max_pos_error=1e-3, max_rot_error=1e-2, max_subdivision_depth=6),
    )
    assert pres.success, pres.reason
    assert pres.trajectory is not None
    assert pres.trajectory.num_waypoints > 2
    assert pres.max_pos_error <= 1e-3 + 1e-6

    # 3. hold_orientation=True locks start TCP orientation across position-only waypoints on Panda.
    pres_hold = pplanner.plan_cartesian_path(
        pq0,
        [CartesianWaypoint(pp1)],
        link=ee,
        config=CartesianPlanConfig(max_step=0.03, max_pos_error=1e-3, max_rot_error=5e-3, hold_orientation=True),
    )
    assert pres_hold.success, pres_hold.reason
    assert pres_hold.trajectory is not None
    ts_h = np.linspace(pres_hold.trajectory.times[0], pres_hold.trajectory.times[-1], 60)
    qh, _, _ = pres_hold.trajectory.sample(ts_h)
    _, rots_h = pworld.link_pose(qh, ee)
    for r_i in rots_h:
        ang = float(np.linalg.norm(Rotation.from_matrix(pr0.T @ r_i).as_rotvec()))
        assert ang <= 5e-3 + 1e-5


def test_cartesian_obstacle_partial_fraction_jump_and_invalid_arcs(simple_snapshot):
    """Obstacle blocking mid-path yields monotone verified partial prefix; strict jump & degenerate arcs enforced."""
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    planner = MotionPlanner(world, backend="coal")
    tip = ("planning", "left_tip")

    # Path straight into the static obstacle at (0, -0.06, 0.5) with half-extents (0.16, 0.15, 0.15).
    q_start = np.array([0.0, 0.45, 0.0], dtype=np.float64)
    p0, _ = world.link_pose(q_start, tip)
    p_blocked = p0 + np.array([0.0, -0.55, 0.0])

    cfg = CartesianPlanConfig(
        max_step=0.02,
        max_pos_error=1e-3,
        avoid_collisions=True,
        margin=0.002,
        world_margin=0.005,
        strict_ccd=True,
    )
    res = planner.plan_cartesian_path(q_start, [CartesianWaypoint(p_blocked)], link=tip, config=cfg)
    assert not res.success
    assert res.trajectory is not None
    assert 0.1 < res.fraction < 0.85
    assert res.completed_waypoints == 0
    assert res.reason is not None and ("collision" in res.reason.lower() or "clearance" in res.reason.lower())

    # Partial prefix must itself be 100% executable under strict_ccd!
    ok_val, err_val = world.validate_trajectory(
        res.trajectory,
        margin=cfg.margin,
        world_margin=cfg.effective_world_margin,
        strict_ccd=True,
    )
    assert ok_val, err_val

    # Prismatic jump threshold rejected when subdivision disabled, succeeds when subdivision enabled.
    p_clear = p0 + np.array([0.40, 0.0, 0.0])
    j_fail = planner.plan_cartesian_path(
        q_start,
        [CartesianWaypoint(p_clear)],
        link=tip,
        config=CartesianPlanConfig(max_step=0.40, jump_threshold_prismatic=0.10, max_subdivision_depth=0),
    )
    assert not j_fail.success
    assert j_fail.fraction == pytest.approx(0.0)
    assert j_fail.reason is not None and "jump" in j_fail.reason.lower()

    j_ok = planner.plan_cartesian_path(
        q_start,
        [CartesianWaypoint(p_clear)],
        link=tip,
        config=CartesianPlanConfig(max_step=0.40, jump_threshold_prismatic=0.10, max_subdivision_depth=4),
    )
    assert j_ok.success, j_ok.reason
    assert j_ok.fraction == pytest.approx(1.0)

    # Degenerate collinear and duplicate circular arcs raise ValueError.
    with pytest.raises(ValueError, match="collinear"):
        planner.plan_cartesian_path(
            q_start,
            [CartesianWaypoint(p0 + np.array([0.2, 0.0, 0.0]), motion_type="circular",
                               via_position=p0 + np.array([0.1, 0.0, 0.0]))],
            link=tip,
        )
    with pytest.raises(ValueError, match="distinct"):
        planner.plan_cartesian_path(
            q_start,
            [CartesianWaypoint(p0 + np.array([0.2, 0.1, 0.0]), motion_type="circular", via_position=p0)],
            link=tip,
        )


def test_cartesian_review_regressions_movec_via_blend_rot_pure_rot_and_first_seg_prefix(simple_snapshot, tmp_path):
    """Regression tests for 4 Cartesian review counterexamples:
    1. Small near-full YZ circular arc on XY-only tip cannot skip via and report success.
    2. Blended corner transitioning from constrained to free orientation still achieves specified rotation.
    3. Failed trailing pure-rotation waypoint after a free-orientation segment cannot report fraction=1 / completed=2.
    4. Collision inside the first coarse IK segment bisects along path parameter to return verified partial prefix.
    """
    # 1. Near-full 1.5 mm YZ circle on XY-only left_tip must fail (cannot reach +3 mm Z via_position).
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    planner = MotionPlanner(world, backend="coal")
    tip = ("planning", "left_tip")
    q0 = np.array([-0.55, 0.45, 0.0], dtype=np.float64)
    p0, _ = world.link_pose(q0, tip)
    R = 0.0015
    center = p0 + np.array([0.0, 0.0, R])
    p_via = center + np.array([0.0, 0.0, R])
    ang_end = -0.5 * math.pi + 2.0 * math.pi - 0.15
    p_end = center + np.array([0.0, R * math.cos(ang_end), R * math.sin(ang_end)])
    res_arc = planner.plan_cartesian_path(
        q0,
        [CartesianWaypoint(p_end, motion_type="circular", via_position=p_via)],
        link=tip,
        config=CartesianPlanConfig(max_step=0.01, max_pos_error=1e-3, avoid_collisions=False),
    )
    assert not res_arc.success
    assert res_arc.fraction < 1.0
    assert res_arc.completed_waypoints == 0

    # 2. X-Y-Yaw robot: blended corner from constrained yaw=1.0 rad to free orientation must reach yaw=1.0 rad.
    xyyaw_xml = tmp_path / "xyyaw.xml"
    xyyaw_xml.write_text("""
    <mujoco model="xyyaw">
      <compiler angle="radian"/>
      <worldbody>
        <body name="base" pos="0 0 0.5">
          <body name="bx">
            <inertial pos="0 0 0" mass="0.1" diaginertia="0.01 0.01 0.01"/>
            <joint name="jx" type="slide" axis="1 0 0" range="-2 2"/>
            <body name="by">
              <inertial pos="0 0 0" mass="0.1" diaginertia="0.01 0.01 0.01"/>
              <joint name="jy" type="slide" axis="0 1 0" range="-2 2"/>
              <body name="tip">
                <joint name="jyaw" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                <geom type="sphere" size="0.03"/>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """)
    gobot.set_project_path(str(tmp_path))
    gobot.import_mjcf_scene(str(xyyaw_xml), "res://xyyaw.jscn", name="xyyaw")
    ctx = gobot.app.context()
    ctx.set_project_path(str(tmp_path))
    ctx.load_scene("res://xyyaw.jscn")
    try:
        xyyaw_snap = ctx.compile_physics_scene_snapshot()
    finally:
        ctx.clear_scene()
    w_rot = PlanningWorld.from_snapshot(xyyaw_snap, {"xyyaw": "base"})
    p_rot = MotionPlanner(w_rot, backend="coal")
    rot_tip = ("xyyaw", "tip")
    q_rot0 = np.zeros(3, dtype=np.float64)
    rp0, _ = w_rot.link_pose(q_rot0, rot_tip)
    r_target = Rotation.from_euler("z", 1.0).as_matrix()
    res_blend_rot = p_rot.plan_cartesian_path(
        q_rot0,
        [
            CartesianWaypoint(rp0 + np.array([0.4, 0.0, 0.0]), rotation=r_target, blend_radius=0.15),
            CartesianWaypoint(rp0 + np.array([0.4, 0.4, 0.0]), rotation=None, blend_radius=0.0),
        ],
        link=rot_tip,
        config=CartesianPlanConfig(max_step=0.02, max_pos_error=1e-3, max_rot_error=1e-2, avoid_collisions=False),
    )
    assert res_blend_rot.success, res_blend_rot.reason
    assert res_blend_rot.trajectory is not None
    ts = np.linspace(res_blend_rot.trajectory.times[0], res_blend_rot.trajectory.times[-1], 200)
    qs, _, _ = res_blend_rot.trajectory.sample(ts)
    assert float(np.min(np.abs(qs[:, 2] - 1.0))) <= 1e-2

    # 3. Trailing pure-rotation waypoint failing IK after a free-orientation arc must report fraction < 1 and completed=1.
    # On simple_snapshot left_tip (XY slide only, no rotation DOF), start rot is I; request free arc then target rot=Rz(0.5).
    p_arc_c = p0 + np.array([0.10, 0.0, 0.0])
    p_arc_via = p_arc_c + np.array([0.0, 0.10, 0.0])
    p_arc_end = p_arc_c + np.array([0.10, 0.0, 0.0])
    res_pure_rot = planner.plan_cartesian_path(
        q0,
        [
            CartesianWaypoint(p_arc_end, motion_type="circular", via_position=p_arc_via, rotation=None),
            CartesianWaypoint(p_arc_end, motion_type="linear", rotation=Rotation.from_euler("z", 0.5).as_matrix()),
        ],
        link=tip,
        config=CartesianPlanConfig(max_step=0.02, max_pos_error=1e-3, max_rot_error=1e-2, avoid_collisions=False),
    )
    assert not res_pure_rot.success
    assert res_pure_rot.trajectory is not None
    assert res_pure_rot.completed_waypoints == 1
    assert 0.2 < res_pure_rot.fraction < 1.0

    # 4. Single coarse 1.4 m linear segment hitting obstacle inside seg 0 (start, mid, end all collision-free)
    # must bisect inside seg 0 and return the verified partial prefix under strict_ccd=True.
    world_seg0 = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    planner_seg0 = MotionPlanner(world_seg0, backend="coal")
    q_s0 = np.array([-0.70, 0.45, 0.0], dtype=np.float64)
    sp0, _ = world_seg0.link_pose(q_s0, tip)
    # Thin wall at x = -0.34 (width 0.04 -> face at -0.36, sphere r=0.045 -> collides around x=-0.405, ~0.295 m from start).
    # Start x=-0.70, midpoint x=0.0, end x=+0.70 are all completely clear of the wall!
    world_seg0.add_box("thin_wall", (0.04, 0.20, 0.20), np.array([-0.34, 0.45, 0.50]))
    cfg_seg0 = CartesianPlanConfig(
        max_step=2.0,
        jump_threshold_prismatic=5.0,
        max_subdivision_depth=0,
        avoid_collisions=True,
        margin=0.002,
        world_margin=0.005,
        strict_ccd=True,
    )
    res_seg0 = planner_seg0.plan_cartesian_path(
        q_s0,
        [CartesianWaypoint(sp0 + np.array([1.40, 0.0, 0.0]))],
        link=tip,
        config=cfg_seg0,
    )
    assert not res_seg0.success
    assert res_seg0.trajectory is not None, res_seg0.reason
    assert 0.15 < res_seg0.fraction < 0.25
    assert res_seg0.completed_waypoints == 0
    ok_s0, err_s0 = world_seg0.validate_trajectory(
        res_seg0.trajectory,
        margin=cfg_seg0.margin,
        world_margin=cfg_seg0.effective_world_margin,
        strict_ccd=True,
    )
    assert ok_s0, err_s0

    # 5. P1: 180-deg hairpin reversal blend (forward 30 mm, back 30 mm, blend=6 mm) must reach ~27 mm apex for order 3 & 5.
    for order in (3, 5):
        res_hairpin = planner.plan_cartesian_path(
            q0,
            [
                CartesianWaypoint(p0 + np.array([0.030, 0.0, 0.0]), blend_radius=0.006),
                CartesianWaypoint(p0, blend_radius=0.0),
            ],
            link=tip,
            config=CartesianPlanConfig(max_step=0.01, max_pos_error=1e-3, spline_order=order, avoid_collisions=False),
        )
        assert res_hairpin.success, f"order={order}: {res_hairpin.reason}"
        assert res_hairpin.trajectory is not None
        ts_h = np.linspace(res_hairpin.trajectory.times[0], res_hairpin.trajectory.times[-1], 200)
        qh, _, _ = res_hairpin.trajectory.sample(ts_h)
        pts_h, _ = world.link_pose(qh, tip)
        max_dx = float(np.max(pts_h[:, 0] - p0[0]))
        assert abs(max_dx - 0.027) <= 1e-3, f"order={order}: hairpin apex reached {max_dx:.6f} m, expected ~0.027 m"

    # 5b. P2: 200 mm hairpin reversal with 38 mm blend and max_step=10 mm: adjacent IK knots must satisfy <= max_step.
    res_big_blend = planner.plan_cartesian_path(
        q0,
        [
            CartesianWaypoint(p0 + np.array([0.200, 0.0, 0.0]), blend_radius=0.038),
            CartesianWaypoint(p0, blend_radius=0.0),
        ],
        link=tip,
        config=CartesianPlanConfig(max_step=0.01, max_pos_error=1e-3, avoid_collisions=False),
    )
    assert res_big_blend.success, res_big_blend.reason
    assert res_big_blend.trajectory is not None
    knot_pts_bb, _ = world.link_pose(res_big_blend.trajectory.positions, tip)
    adj_dists = np.linalg.norm(np.diff(knot_pts_bb, axis=0), axis=-1)
    assert float(np.max(adj_dists)) <= 0.01 + 1e-6, f"max adjacent knot TCP spacing {float(np.max(adj_dists)):.6f} m > 0.01 m"

    # 5c. P2: Same 200 mm hairpin with tiny max_step=0.20 mm (< default 0.25 mm IK tol): actual adjacent IK TCP
    #     spacing MUST still stay <= 0.20 mm (cannot skip steps via IK fast-return and jump 0.40 mm!).
    res_fine_blend = planner.plan_cartesian_path(
        q0,
        [
            CartesianWaypoint(p0 + np.array([0.200, 0.0, 0.0]), blend_radius=0.038),
            CartesianWaypoint(p0, blend_radius=0.0),
        ],
        link=tip,
        config=CartesianPlanConfig(max_step=0.0002, max_pos_error=1e-3, avoid_collisions=False),
    )
    assert res_fine_blend.success, res_fine_blend.reason
    assert res_fine_blend.trajectory is not None
    knot_pts_fb, _ = world.link_pose(res_fine_blend.trajectory.positions, tip)
    adj_dists_fb = np.linalg.norm(np.diff(knot_pts_fb, axis=0), axis=-1)
    assert float(np.max(adj_dists_fb)) <= 0.0002 + 1e-6, (
        f"max adjacent knot TCP spacing {float(np.max(adj_dists_fb)):.6f} m > 0.0002 m"
    )

    # 6. P2: IK-stage collision exhausting max_subdivision_depth on first interval still recovers verified prefix.
    world_ikcol = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    planner_ikcol = MotionPlanner(world_ikcol, backend="coal")
    # Start x=-0.70, depth-4 node at x=-0.6125; box at x=-0.59, width=0.08 (face at -0.63) + sphere r=0.045 + margin 0.005
    # leaves first ~20 mm clear while covering x=-0.6125!
    world_ikcol.add_box("near_wall", (0.08, 0.20, 0.20), np.array([-0.59, 0.45, 0.50]))
    cfg_ikcol = CartesianPlanConfig(
        max_step=2.0,
        jump_threshold_prismatic=5.0,
        max_subdivision_depth=4,
        avoid_collisions=True,
        margin=0.002,
        world_margin=0.005,
        strict_ccd=True,
    )
    res_ikcol = planner_ikcol.plan_cartesian_path(
        q_s0,
        [CartesianWaypoint(sp0 + np.array([1.40, 0.0, 0.0]))],
        link=tip,
        config=cfg_ikcol,
    )
    assert not res_ikcol.success
    assert res_ikcol.trajectory is not None, res_ikcol.reason
    assert 0.005 < res_ikcol.fraction < 0.04
    ok_ikc, err_ikc = world_ikcol.validate_trajectory(
        res_ikcol.trajectory,
        margin=cfg_ikcol.margin,
        world_margin=cfg_ikcol.effective_world_margin,
        strict_ccd=True,
    )
    assert ok_ikc, err_ikc

    # 7. P2: Later unreachable waypoint IK error must NOT overwrite earlier segment-0 collision truncation reason.
    res_reason = planner_seg0.plan_cartesian_path(
        q_s0,
        [
            CartesianWaypoint(sp0 + np.array([1.40, 0.0, 0.0])),
            CartesianWaypoint(sp0 + np.array([1.40, 0.0, 0.20])),  # unreachable Z
        ],
        link=tip,
        config=cfg_seg0,
    )
    assert not res_reason.success
    assert res_reason.trajectory is not None
    assert res_reason.reason is not None and "clearance" in res_reason.reason and "thin_wall" in res_reason.reason

    # 8. P3: First waypoint equal to start pose preserves completed_waypoints=1 when second waypoint fails.
    res_wp0 = planner.plan_cartesian_path(
        q0,
        [
            CartesianWaypoint(p0),
            CartesianWaypoint(p0 + np.array([0.0, 0.0, 0.20])),  # unreachable Z
        ],
        link=tip,
        config=CartesianPlanConfig(avoid_collisions=False),
    )
    assert not res_wp0.success
    assert res_wp0.trajectory is None
    assert res_wp0.fraction == pytest.approx(0.0)
    assert res_wp0.completed_waypoints == 1


def test_constrained_global_planning_upright_cup_around_pillar_panda(panda_snapshot):
    """2.4 Acceptance test: carry an upright cup (local Z aligned with world -Z within 0.05 rad, free yaw)
    around the central pillar on 7-DOF Panda with continuous manifold + strict_ccd verification."""
    snap, (start_xyz, goal_xyz, _) = panda_snapshot
    mounts = (("panda_link0", "panda_link2"), ("panda_link4", "panda_link6"), ("panda_link7", "panda_hand"))
    world = PlanningWorld.from_snapshot(
        snap, {"panda": "panda_link0"},
        ignored_link_pairs=[(("panda", a), ("panda", b)) for a, b in mounts],
    )
    planner = MotionPlanner(world, backend="coal")
    hand = ("panda", "panda_hand")
    down = np.diag([-1.0, 1.0, -1.0])
    seed = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, -2.3], dtype=np.float64)

    # Global upright-cup manifold constraint: free yaw around Z, tilt <= 0.05 rad everywhere!
    upright_cup = PoseConstraint(
        hand,
        rotation=down,
        rotation_tolerance=(0.05, 0.05, math.inf),
    )

    ik_start = planner.solve_ik(
        [PoseConstraint(hand, start_xyz, down, position_tolerance=1e-4, rotation_tolerance=(0.01, 0.01, math.inf))],
        seed=seed, margin=0.002, world_margin=0.02,
    )
    ik_goal = planner.solve_ik(
        [PoseConstraint(hand, goal_xyz, down, position_tolerance=1e-4, rotation_tolerance=(0.01, 0.01, math.inf))],
        seed=seed, margin=0.002, world_margin=0.02,
    )
    assert ik_start.success and ik_goal.success

    # 1. Verify existing interior-only PathConstraint semantics are untouched (does NOT reject out-of-manifold endpoints
    # when not passed in manifold_constraints), while manifold_constraints rejects an out-of-manifold start!
    q_tilted = ik_start.positions.copy()
    q_tilted[5] = float(np.clip(q_tilted[5] + 0.5, world.lower_limits[5], world.upper_limits[5]))
    bad_m = planner.plan(MotionPlanRequest(
        start=q_tilted,
        goal=ik_goal.positions,
        manifold_constraints=[upright_cup],
        config=SolverConfig(waypoints=12, margin=0.002, world_margin=0.015),
    ))
    assert not bad_m.success
    assert bad_m.reason is not None and "manifold" in bad_m.reason.lower()

    # 2. Plan globally around the central pillar keeping the cup continuously upright (quintic C^2 + strict_ccd)!
    res = planner.plan(MotionPlanRequest(
        start=ik_start.positions,
        goal=ik_goal.positions,
        manifold_constraints=[upright_cup],
        config=SolverConfig(
            waypoints=16,
            margin=0.002,
            world_margin=0.015,
            spline_order=5,
            strict_ccd=True,
            stop_at_endpoints=True,
            rrt_timeout_s=2.5,
        ),
    ))
    assert res.success, res.reason
    assert res.trajectory is not None
    assert res.world_version == world.version

    # Dense independent check: cup orientation tilt stays <= 0.05 rad at every instant along the retimed spline!
    ts = np.linspace(res.trajectory.times[0], res.trajectory.times[-1], 200)
    q_dense, _, _ = res.trajectory.sample(ts)
    _, rots = world.link_pose(q_dense, hand)
    cos_vals = np.clip(np.einsum("ni,i->n", rots[:, :, 2], down[:, 2]), -1.0, 1.0)
    tilts = np.arccos(cos_vals)
    assert float(np.max(tilts)) <= 0.05 + 1e-4, f"max cup tilt {float(np.max(tilts)):.5f} rad > 0.05 rad"

    # Independent atomic validate_trajectory (bounds + strict_ccd + continuous manifold)!
    ok_val, err_val = world.validate_trajectory(
        res.trajectory,
        margin=0.002,
        world_margin=0.015,
        strict_ccd=True,
        manifold_constraints=[upright_cup],
    )
    assert ok_val, err_val

    # 3. Counterexample: spline whose endpoints are upright (tilt == 0) but mid-segment tilts > 0.05 rad MUST be rejected!
    v_bad = np.zeros((2, world.num_joints), dtype=np.float64)
    v_bad[0, 5] = 1.6
    v_bad[1, 5] = -1.6
    bad_spline = Trajectory(
        times=np.array([0.0, 1.0], dtype=np.float64),
        positions=np.vstack([ik_start.positions, ik_start.positions]),
        velocities=v_bad,
        accelerations=np.zeros((2, world.num_joints), dtype=np.float64),
        spline_order=5,
    )
    ok_bad, err_bad = world.validate_trajectory(
        bad_spline,
        margin=0.0,
        world_margin=0.0,
        velocity_scale=1e3,
        acceleration_scale=1e3,
        jerk_scale=1e4,
        manifold_constraints=[upright_cup],
    )
    assert not ok_bad
    assert err_bad is not None and "manifold" in err_bad.lower()


def test_manifold_review_regressions_positional_args_batch_warp_snapshot_free_pi_and_tiny_tol(simple_snapshot, panda_snapshot):
    """Regression tests for all 9 review findings on 2.4 constrained global planning."""
    # 1. P2 #8: MotionPlanRequest positional arguments keep exact pre-2.4 meaning (seed is 5th arg!).
    seed_arr = np.zeros((8, 3), dtype=np.float64)
    req_pos = MotionPlanRequest(np.zeros(3), np.zeros(3), None, (), seed_arr)
    assert req_pos.seed is seed_arr
    assert req_pos.manifold_constraints == ()

    # 2. P1 #1 & #3 & P2 #9: Unreachable interior waypoint_target cannot be bypassed by raw CBiRRT;
    #    identical start==goal succeeds; tiny tolerance 1e-10 is never silently relaxed to 1e-9.
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    planner = MotionPlanner(world, backend="coal")
    tip = ("planning", "left_tip")
    q_clear = np.array([-0.55, 0.45, 0.0], dtype=np.float64)
    p_clear, _ = world.link_pose(q_clear, tip)
    plane_z = PoseConstraint(tip, position=p_clear, position_tolerance=(math.inf, math.inf, 1e-3))

    res_same = planner.plan(MotionPlanRequest(
        start=q_clear,
        goal=q_clear,
        manifold_constraints=[plane_z],
        config=SolverConfig(waypoints=8, margin=0.002, world_margin=0.005),
    ))
    assert res_same.success, res_same.reason
    assert res_same.trajectory is not None and res_same.trajectory.num_waypoints >= 2

    # Tiny tolerance 1e-10 violated by 5e-10 must be rejected by validate_trajectory!
    tiny_c = PoseConstraint(tip, position=p_clear + np.array([5e-10, 0.0, 0.0]), position_tolerance=1e-10)
    ok_tiny, _ = world.validate_trajectory(
        np.vstack([q_clear, q_clear]),
        margin=0.0,
        world_margin=0.0,
        manifold_constraints=[tiny_c],
    )
    assert not ok_tiny

    # 3. P2 #7: Rotating right_tip freely across pi around Z with free_count==2 (X constrained, Y/Z free) passes!
    rtip = ("planning", "right_tip")
    free_yz = PoseConstraint(rtip, rotation=np.eye(3), rotation_tolerance=(0.05, math.inf, math.inf))
    q_a = np.array([-0.55, 0.45, 2.9], dtype=np.float64)
    q_b = np.array([-0.55, 0.45, 3.3], dtype=np.float64)
    t_pi = Trajectory(
        times=np.array([0.0, 1.0]),
        positions=np.vstack([q_a, q_b]),
        velocities=np.zeros((2, 3)),
        accelerations=np.zeros((2, 3)),
        spline_order=3,
    )
    ok_pi, err_pi = world.validate_trajectory(
        t_pi, margin=0.0, world_margin=0.0, manifold_constraints=[free_yz],
    )
    assert ok_pi, err_pi

    # 4. P2 #4 & #6: Batch plan on Panda with upright manifold_constraints produces finite positions rows
    #    even if mid-segment refinement increases knot count, and plan_with_snapshot pins world_version!
    psnap, (start_xyz, goal_xyz, _) = panda_snapshot
    mounts = (("panda_link0", "panda_link2"), ("panda_link4", "panda_link6"), ("panda_link7", "panda_hand"))
    pworld = PlanningWorld.from_snapshot(
        psnap, {"panda": "panda_link0"},
        ignored_link_pairs=[(("panda", a), ("panda", b)) for a, b in mounts],
    )
    pplanner = MotionPlanner(pworld, backend="coal")
    hand = ("panda", "panda_hand")
    down = np.diag([-1.0, 1.0, -1.0])
    seed = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, -2.3], dtype=np.float64)
    upright = PoseConstraint(hand, rotation=down, rotation_tolerance=(0.05, 0.05, math.inf))
    ik_s = pplanner.solve_ik(
        [PoseConstraint(hand, start_xyz, down, position_tolerance=1e-4, rotation_tolerance=(0.01, 0.01, math.inf))],
        seed=seed, margin=0.002, world_margin=0.02,
    )
    ik_g = pplanner.solve_ik(
        [PoseConstraint(hand, goal_xyz, down, position_tolerance=1e-4, rotation_tolerance=(0.01, 0.01, math.inf))],
        seed=seed, margin=0.002, world_margin=0.02,
    )
    assert ik_s.success and ik_g.success

    req_b = MotionPlanRequest(
        start=ik_s.positions,
        goal=ik_g.positions,
        manifold_constraints=[upright],
        config=SolverConfig(waypoints=12, margin=0.002, world_margin=0.015, rrt_timeout_s=2.0),
    )
    batch = pplanner.plan_batch([req_b])
    assert batch.success[0], batch.reasons[0]
    assert batch.results[0].positions is not None
    assert batch.positions.shape == (1, batch.results[0].positions.shape[0], pworld.num_joints)
    assert np.isfinite(batch.positions[0]).all()
    assert batch.results[0].waypoint_indices is not None

    # 5. Review Round 2 P1: plan_batch with waypoints=2 around a blocking box MUST preserve ALL verified
    #    obstacle-avoidance knots in batch.positions[0] so validating batch.positions[0] as a polyline passes!
    w_box = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    w_box.add_box("mid_blocker", (0.20, 0.20, 0.40), np.array([0.0, 0.45, 0.50]))
    pl_box = MotionPlanner(w_box, backend="coal")
    q_left = np.array([-0.55, 0.45, 0.0], dtype=np.float64)
    q_right = np.array([0.55, 0.45, 0.0], dtype=np.float64)
    p_left, _ = w_box.link_pose(q_left, tip)
    box_plane_z = PoseConstraint(tip, position=p_left, position_tolerance=(math.inf, math.inf, 1e-3))

    req_coarse = MotionPlanRequest(
        start=q_left,
        goal=q_right,
        manifold_constraints=[box_plane_z],
        config=SolverConfig(
            waypoints=2, margin=0.002, world_margin=0.005,
            rrt_step_size=0.15, rrt_timeout_s=2.0, strict_ccd=True,
        ),
    )
    req_same_b = MotionPlanRequest(
        start=q_left,
        goal=q_left,
        manifold_constraints=[box_plane_z],
        config=SolverConfig(waypoints=2, margin=0.002, world_margin=0.005),
    )
    batch_detour = pl_box.plan_batch([req_coarse, req_same_b])
    assert batch_detour.success[0] and batch_detour.success[1], batch_detour.reasons
    assert batch_detour.results[0].trajectory is not None
    n_detour = batch_detour.results[0].trajectory.num_waypoints
    assert n_detour > 2
    assert batch_detour.positions.shape == (2, n_detour, w_box.num_joints)
    assert np.allclose(batch_detour.positions[0], batch_detour.results[0].positions)
    ok_bp, err_bp = w_box.validate_trajectory(
        batch_detour.positions[0],
        margin=0.002,
        world_margin=0.005,
        manifold_constraints=[box_plane_z],
    )
    assert ok_bp, err_bp
    ok_bp1, err_bp1 = w_box.validate_trajectory(
        batch_detour.positions[1],
        margin=0.002,
        world_margin=0.005,
        manifold_constraints=[box_plane_z],
    )
    assert ok_bp1, err_bp1

    # Also verify multi-seed (num_seeds=4) with waypoints=2 retains dense CBiRRT fallback and succeeds!
    res_ms = pl_box.plan(MotionPlanRequest(
        start=q_left,
        goal=q_right,
        manifold_constraints=[box_plane_z],
        config=SolverConfig(
            waypoints=2, num_seeds=4, margin=0.002, world_margin=0.005,
            rrt_step_size=0.15, rrt_timeout_s=2.0, strict_ccd=True,
        ),
    ))
    assert res_ms.success, res_ms.reason
    assert res_ms.trajectory is not None and res_ms.trajectory.num_waypoints > 2
    np.testing.assert_array_equal(res_ms.waypoint_indices, [0, res_ms.trajectory.num_waypoints - 1])
    np.testing.assert_allclose(res_ms.positions[res_ms.waypoint_indices], [q_left, q_right])

    # 6. Review Round 2 P2: Continuous manifold check MUST reject a spline slightly over tolerance (1.0003 * tol)!
    peak_over = 1e-3 * 1.0003
    v_over = np.zeros((2, 3), dtype=np.float64)
    v_over[0, 0] = 4.0 * peak_over
    v_over[1, 0] = -4.0 * peak_over
    t_over = Trajectory(
        times=np.array([0.0, 1.0]),
        positions=np.vstack([q_clear, q_clear]),
        velocities=v_over,
        accelerations=np.zeros((2, 3)),
        spline_order=3,
    )
    plane_x = PoseConstraint(tip, position=p_clear, position_tolerance=(1e-3, math.inf, math.inf))
    ok_over, _ = world.validate_trajectory(
        t_over, margin=0.0, world_margin=0.0, manifold_constraints=[plane_x],
    )
    assert not ok_over

    # 7. Review Round 2 P3: Zero tolerance in manifold_constraints is explicitly rejected at Python entry points!
    zero_pos_c = PoseConstraint(tip, position=p_clear, position_tolerance=0.0)
    zero_rot_c = PoseConstraint(rtip, rotation=np.eye(3), rotation_tolerance=(0.05, 0.0, math.inf))
    with pytest.raises(ValueError, match="zero tolerance"):
        MotionPlanRequest(start=q_clear, goal=q_clear, manifold_constraints=[zero_pos_c])
    with pytest.raises(ValueError, match="zero tolerance"):
        MotionPlanRequest(start=q_clear, goal=q_clear, manifold_constraints=[zero_rot_c])
    with pytest.raises(ValueError, match="zero tolerance"):
        world.validate_trajectory(np.vstack([q_clear, q_clear]), manifold_constraints=[zero_pos_c])

    # Verify pinned snapshot isolation via _plan_warp_single(req, snap=pinned) after world mutates!
    pinned = pworld._native.snapshot()
    pinned_ver = int(pinned.version)
    pworld.add_box("later_blocker", (0.1, 0.1, 0.1), np.array([0.0, 0.0, 5.0]))
    assert pworld.version > pinned_ver
    res_pinned = pplanner._plan_warp_single(req_b, snap=pinned)
    assert res_pinned.success, res_pinned.reason
    assert res_pinned.world_version == pinned_ver


def test_mixed_batch_keeps_spline_knot_indices(simple_snapshot):
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    planner = MotionPlanner(world)
    start = np.array([-.7, .4, 0.])
    goal = np.array([.7, .4, 0.])
    via = np.array([0., .55, 0.])
    batch = planner.plan_batch([
        MotionPlanRequest(start, goal, waypoint_targets={1: via}, config=SolverConfig(waypoints=3)),
        MotionPlanRequest(start, start, config=SolverConfig(waypoints=7)),
    ])
    assert batch.success.all(), batch.reasons
    row = batch.results[0]
    np.testing.assert_array_equal(batch.positions[0, :3], row.trajectory.positions)
    np.testing.assert_array_equal(batch.positions[0, 3:], np.tile(goal, (4, 1)))
    np.testing.assert_allclose(batch.positions[0, row.waypoint_indices[1]], via)
    assert world.validate_trajectory(row.trajectory, strict_ccd=True)[0]


def test_manifold_free_slide_and_allowed_near_antipodal_pose(simple_snapshot):
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    planner = MotionPlanner(world)
    start = np.array([-.7, .4, 0.])
    goal = np.array([.7, .4, 0.])
    tip = ("planning", "left_tip")
    p, _ = world.link_pose(start, tip)
    plane = PoseConstraint(tip, position=p, position_tolerance=(math.inf, math.inf, .0001))
    for count in (2, 32):
        valid, reason = world.validate_trajectory(np.linspace(start, goal, count), manifold_constraints=[plane])
        assert valid, reason

    arm = ("planning", "right_arm")
    tol = math.radians(179.5)
    rotation = PoseConstraint(arm, rotation=np.eye(3), rotation_tolerance=(math.inf, tol, tol))
    for angle, expected in [(178., True), (179., True), (179.9, False), (180., False)]:
        q = start.copy()
        q[2] = math.radians(angle)
        valid, reason = world.validate_trajectory(np.stack([q, q]), manifold_constraints=[rotation])
        assert valid == expected, reason
        result = planner.plan(MotionPlanRequest(q, q, manifold_constraints=[rotation], config=SolverConfig(waypoints=2)))
        assert result.success == expected, result.reason


def test_voxel_esdf_occupancy_point_cloud_and_custom_esdf_api(simple_snapshot):
    """Phase 3.1: CPU 3D ESDF APIs (occupancy grid, point cloud, custom ESDF), automatic & custom link spheres,
    ESDF-only SQP/IK/Cartesian/CCD collision avoidance, ACM override, and scene-sync preservation."""
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    # Remove any static Coal obstacles so collision avoidance is driven purely by the ESDF.
    for sname in list(world._native.shape_names()):
        if not sname.startswith("planning/"):
            world.remove_obstacle(sname)
    assert not world.has_esdf
    assert world.esdf_info is None
    auto_spheres = world.link_spheres()
    assert len(auto_spheres) >= 2

    v0 = world.version
    world.clear_esdf()
    assert world.version == v0  # no-op clear_esdf does not bump version

    # 1. Occupancy grid ESDF: exact distance & analytic trilinear gradient check.
    occ = np.zeros((25, 25, 15), dtype=np.uint8)
    origin = np.array([-0.625, -0.05, 0.15], dtype=np.float64)
    res = 0.05
    # Place a blocking voxel pillar around x in [-0.125, +0.125], y in [0.35, 0.55], z in [0.15, 0.90].
    occ[10:15, 8:12, :] = 1
    world.set_occupancy_grid(occ, origin, res)
    assert world.has_esdf
    assert world.version == v0 + 1
    info = world.esdf_info
    assert info is not None and info["shape"] == (25, 25, 15) and not info["is_empty"]

    probe = np.array([0.23, 0.44, 0.52], dtype=np.float64)
    d_val, g_val = world.query_esdf(probe, return_gradient=True)
    eps = 1e-5
    for a in range(3):
        dp = np.zeros(3)
        dp[a] = eps
        fd = (world.query_esdf(probe + dp) - world.query_esdf(probe - dp)) / (2.0 * eps)
        assert abs(g_val[a] - fd) < 1e-5

    # 2. Straight chord across the ESDF pillar is rejected by validate_trajectory (even with substeps=1 via CCD!).
    planner = MotionPlanner(world, backend="coal")
    q_start = np.array([-0.55, 0.45, 0.0], dtype=np.float64)
    q_goal = np.array([0.55, 0.45, 0.0], dtype=np.float64)
    ok_bad, err_bad = world.validate_trajectory(
        np.vstack([q_start, q_goal]),
        margin=0.002,
        world_margin=0.01,
        collision_substeps=1,
        strict_ccd=True,
    )
    assert not ok_bad and err_bad is not None and "esdf" in err_bad.lower()

    # 3. MotionPlanner plans a smooth detour around the ESDF pillar (quintic + strict_ccd)!
    plan_res = planner.plan(MotionPlanRequest(
        start=q_start,
        goal=q_goal,
        config=SolverConfig(waypoints=14, margin=0.002, world_margin=0.015, spline_order=5, strict_ccd=True),
    ))
    assert plan_res.success, plan_res.reason
    assert plan_res.trajectory is not None
    assert plan_res.world_version == world.version
    ok_v, err_v = world.validate_trajectory(plan_res.trajectory, margin=0.002, world_margin=0.015, strict_ccd=True)
    assert ok_v, err_v
    ts = np.linspace(plan_res.trajectory.times[0], plan_res.trajectory.times[-1], 80)
    q_dense, _, _ = plan_res.trajectory.sample(ts)
    assert float(world.esdf_distances(q_dense).min()) >= 0.015 - 1e-4

    # 4. Cartesian path planner stops cleanly before the ESDF pillar and recovers a verified partial prefix!
    tip = ("planning", "left_tip")
    p_start, _ = world.link_pose(q_start, tip)
    p_goal, _ = world.link_pose(q_goal, tip)
    cart_res = planner.plan_cartesian_path(
        q_start,
        [CartesianWaypoint(p_goal)],
        link=tip,
        config=CartesianPlanConfig(max_step=0.05, avoid_collisions=True, world_margin=0.01, strict_ccd=True),
    )
    assert not cart_res.success
    assert cart_res.trajectory is not None
    assert 0.1 < cart_res.fraction < 0.6
    ok_c, err_c = world.validate_trajectory(cart_res.trajectory, margin=0.002, world_margin=0.01, strict_ccd=True)
    assert ok_c, err_c

    # 5. Custom ESDF & custom link spheres override + ACM disable + sync_from_snapshot preserves ESDF!
    custom_sdf = np.full((5, 5, 5), 0.5, dtype=np.float64)
    world.set_esdf(custom_sdf, origin=(-1.0, -1.0, 0.0), resolution=0.5)
    assert world.has_esdf
    assert world.query_esdf(np.array([0.0, 0.0, 0.5])) == pytest.approx(0.5, abs=1e-5)

    world.set_link_spheres(tip, [([0.0, 0.0, 0.0], 0.08)])
    tip_spheres = world.link_spheres(tip)
    assert len(tip_spheres) == 1 and tip_spheres[0].radius == pytest.approx(0.08)
    world.clear_link_spheres(tip)
    assert len(world.link_spheres(tip)) >= 1

    # ACM disables ESDF checks on left_tip.
    n_before = len(world.link_spheres(esdf_only=True))
    world.set_collision_enabled(tip, "esdf", False)
    assert len(world.link_spheres(esdf_only=True)) < n_before
    world.set_collision_enabled(tip, "esdf", True)
    assert len(world.link_spheres(esdf_only=True)) == n_before

    # Point cloud with empty array clears obstacles without crashing.
    world.set_point_cloud(np.empty((0, 3), dtype=np.float64), resolution=0.05)
    assert not world.has_esdf
    assert np.isinf(world.query_esdf(np.array([0.0, 0.0, 0.5])))
    world.clear_esdf()
    assert world.esdf_info is None


def test_panda_point_cloud_esdf_obstacle_avoidance_and_speed(panda_snapshot):
    """Phase 3.1 Acceptance: 7-DOF Panda avoids a 3D point-cloud pillar via C++ ESDF + auto link sphere-cover."""
    snap, (start_xyz, goal_xyz, _) = panda_snapshot
    mounts = (("panda_link0", "panda_link2"), ("panda_link4", "panda_link6"), ("panda_link7", "panda_hand"))
    world = PlanningWorld.from_snapshot(
        snap, {"panda": "panda_link0"},
        ignored_link_pairs=[(("panda", a), ("panda", b)) for a, b in mounts],
    )
    # Remove the static Coal pillar obstacle and replace it with a dense 3D point cloud of the pillar!
    pillar_names = [n for n in world._native.shape_names() if "pillar" in n.lower() or "obstacle" in n.lower()]
    for pname in pillar_names:
        world.remove_obstacle(pname)
    # Disable ESDF on panda_link0 (bolted to the table) just like a real perception pipeline.
    world.set_collision_enabled(("panda", "panda_link0"), "esdf", False)

    # Sample a dense 3D point cloud on the central pillar between start_xyz and goal_xyz.
    mid_xy = 0.5 * (start_xyz[:2] + goal_xyz[:2])
    xs = np.linspace(mid_xy[0] - 0.04, mid_xy[0] + 0.04, 6)
    ys = np.linspace(mid_xy[1] - 0.04, mid_xy[1] + 0.04, 6)
    zs = np.linspace(0.15, 0.70, 22)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    cloud = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

    t0 = perf_counter()
    world.set_point_cloud(cloud, resolution=0.025, padding=0.25, point_radius=0.005)
    esdf_build_ms = (perf_counter() - t0) * 1000.0
    assert world.has_esdf
    assert esdf_build_ms < 250.0, f"point cloud ESDF build took {esdf_build_ms:.1f} ms"
    assert len(world.link_spheres(esdf_only=True)) >= 8

    planner = MotionPlanner(world, backend="coal")
    hand = ("panda", "panda_hand")
    down = np.diag([-1.0, 1.0, -1.0])
    seed = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, -2.3], dtype=np.float64)

    ik_s = planner.solve_ik(
        [PoseConstraint(hand, start_xyz, down, position_tolerance=1e-4, rotation_tolerance=1e-2)],
        seed=seed, margin=0.002, world_margin=0.015,
    )
    ik_g = planner.solve_ik(
        [PoseConstraint(hand, goal_xyz, down, position_tolerance=1e-4, rotation_tolerance=1e-2)],
        seed=seed, margin=0.002, world_margin=0.015,
    )
    assert ik_s.success and ik_g.success, (ik_s.reason, ik_g.reason)

    # Plan around the point-cloud pillar with quintic C^2 spline + strict_ccd!
    res = planner.plan(MotionPlanRequest(
        start=ik_s.positions,
        goal=ik_g.positions,
        config=SolverConfig(
            waypoints=16,
            margin=0.002,
            world_margin=0.015,
            spline_order=5,
            strict_ccd=True,
            rrt_timeout_s=2.0,
        ),
    ))
    assert res.success, res.reason
    assert res.trajectory is not None
    ok_v, err_v = world.validate_trajectory(res.trajectory, margin=0.002, world_margin=0.015, strict_ccd=True)
    assert ok_v, err_v

    ts = np.linspace(res.trajectory.times[0], res.trajectory.times[-1], 100)
    q_dense, _, _ = res.trajectory.sample(ts)
    min_esdf_clear = float(world.esdf_distances(q_dense).min())
    assert min_esdf_clear >= 0.015 - 1e-4, f"min sphere ESDF clearance {min_esdf_clear:.5f} m < 0.015 m"

    # Benchmark 64^3 CPU 3D EDT speed (should be well under 50 ms on multi-core CPU).
    occ64 = np.zeros((64, 64, 64), dtype=np.uint8)
    occ64[28:36, 28:36, 10:54] = 1
    t1 = perf_counter()
    world.set_occupancy_grid(occ64, origin=(-1.0, -1.0, 0.0), resolution=0.03)
    edt64_ms = (perf_counter() - t1) * 1000.0
    assert edt64_ms < 200.0, f"64^3 3D EDT took {edt64_ms:.1f} ms"


def test_esdf_corner_clearance_and_partial_collision_group(simple_snapshot):
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    q_safe = np.array([-.4, .4, 0.])
    q_corner = np.array([.299, .299, 0.])
    tip = ("planning", "left_tip")
    p, _ = world.link_pose(q_corner, tip)
    occ = np.zeros((5, 5, 5), dtype=np.uint8)
    occ[2, 2, 2] = 1
    world.set_occupancy_grid(occ, p - .299, .1)
    # Each coordinate is 1 mm inside the occupied voxel. The tip sphere has radius 45 mm.
    assert world.query_esdf(p) > .045
    assert world.esdf_distances(q_corner).min() < 0
    ok, reason = world.validate_trajectory(np.stack([q_safe, q_corner]), margin=0, strict_ccd=True)
    assert not ok and "ESDF" in reason

    snap = world._native.snapshot()
    q = np.stack([q_safe, q_safe, q_corner])
    margins = np.zeros(snap.num_pairs)
    rows, ids, _, _, frames, groups_ok, _ = snap.eval_collision_rows(q, margins, margins, 0., True, 2, True)
    np.testing.assert_array_equal(groups_ok, [True, False])
    # The CPU ESDF fallback must also work through the public Warp evaluator.
    warp = object.__new__(MuJoCoWarpBackend)
    warp.world = world
    warp_rows = warp.eval_rows(q, margins, margins, 0., True, 2, True, snap=snap)
    np.testing.assert_array_equal(warp_rows.seg_ok, groups_ok)
    assert (warp_rows.pair_ids < 0).any()
    hit = snap.first_violation_continuous(q, 1, frames, margins, verify=True, strict=True)
    assert hit is not None and hit[2] < 0
    esdf_hits = ids < 0
    assert esdf_hits.any()
    with pytest.raises(ValueError, match="Coal pair"):
        snap.query_distances_indexed(frames, rows[esdf_hits], ids[esdf_hits], False)

    assert world.esdf_distances(np.empty((0, 2, 3))).shape == (0, 2, len(world.link_spheres(esdf_only=True)))
    with pytest.raises(ValueError):
        world.esdf_distances(np.zeros(6))
    with pytest.raises(ValueError):
        world.query_esdf([np.nan, 0, 0])
    v = world.version
    with pytest.raises(ValueError, match="safety limit"):
        world.set_point_cloud([], .1, shape=(1 << 22,) * 3)
    with pytest.raises(ValueError, match="float"):
        world.set_esdf(np.full((2, 2, 2), 1e100), (0, 0, 0), .1)
    assert world.version == v

    # A supplied field with slope four crosses zero 2 cm from the tip center.
    # The 4.5 cm sphere intersects that plane even though the raw value is 8 cm.
    h = .01
    values = np.broadcast_to((4 * (np.arange(11) - 5) * h)[:, None, None], (11, 3, 3))
    world.set_esdf(values, p - [7.5 * h, 1.5 * h, 1.5 * h], h)
    assert world.query_esdf(p) == pytest.approx(.08)
    assert world.esdf_distances(q_corner).min() < 0
    ok, reason = world.validate_trajectory(np.stack([q_corner, q_corner]), margin=0, strict_ccd=True)
    assert not ok and "ESDF" in reason


def test_esdf_attached_sphere_uses_payload_frame(simple_snapshot):
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    tip = ("planning", "left_tip")
    q = np.array([-.4, .4, 0.])
    pos, rot = world.link_pose(q, tip)
    offset = np.array([.2, .15, .1])
    local_rot = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    world.add_box("payload", (.02, .02, .02), pos)
    world.attach_object("payload", tip, local_position=offset, local_rotation=local_rot)
    center = np.array([.05, 0, 0])
    world.set_link_spheres("payload", [(center, .005)])
    np.testing.assert_allclose(world.link_spheres("payload")[0].center, center)
    p = pos + rot @ (offset + local_rot @ center)
    world.set_point_cloud([p], .01)
    ok, reason = world.validate_trajectory(np.stack([q, q]), margin=0, strict_ccd=True)
    assert not ok and "payload" in reason and "ESDF" in reason
    old = world._native.snapshot()
    world.detach_object("payload", q)
    assert not world.link_spheres("payload")
    assert "payload" in [s["link_key"].removeprefix("::") for s in old.esdf_spheres()]


def test_start_boundary_retime_and_counterexamples():
    """Phase 3.2: start-velocity/acceleration retiming, exact slicing, and boundary counterexamples."""
    from gobot.planning import retime_trajectory, slice_trajectory_suffix
    from gobot.planning.retime import check_trajectory_bounds

    vmax = np.array([2.0, 2.0], dtype=np.float64)
    amax = np.array([6.0, 6.0], dtype=np.float64)
    jmax = np.array([40.0, 40.0], dtype=np.float64)
    lo = np.array([-3.0, -3.0], dtype=np.float64)
    hi = np.array([3.0, 3.0], dtype=np.float64)
    pos = np.array([[0.0, 0.0], [0.4, 0.2], [0.8, 0.5], [1.2, 0.8]], dtype=np.float64)

    # Cubic & quintic with fixed start boundary.
    v0 = np.array([0.9, -0.5], dtype=np.float64)
    a0 = np.array([1.2, -0.8], dtype=np.float64)
    traj3 = retime_trajectory(pos, vmax, amax, spline_order=3, start_velocity=v0, lower_limits=lo, upper_limits=hi)
    p0, v_eval, _ = traj3.sample(0.0)
    np.testing.assert_allclose(p0, pos[0], atol=1e-12)
    np.testing.assert_allclose(v_eval, v0, atol=1e-12)
    ok3, reason3, _ = check_trajectory_bounds(traj3, vmax, amax, lower_limits=lo, upper_limits=hi)
    assert ok3, reason3

    traj5 = retime_trajectory(
        pos, vmax, amax, jerk_limits=jmax, spline_order=5,
        start_velocity=v0, start_acceleration=a0, lower_limits=lo, upper_limits=hi,
    )
    p0_5, v0_5, a0_5 = traj5.sample(0.0)
    np.testing.assert_allclose(p0_5, pos[0], atol=1e-12)
    np.testing.assert_allclose(v0_5, v0, atol=1e-12)
    np.testing.assert_allclose(a0_5, a0, atol=1e-12)
    ok5, reason5, _ = check_trajectory_bounds(
        traj5, vmax, amax, lower_limits=lo, upper_limits=hi, jerk_limits=jmax, check_jerk=True,
    )
    assert ok5, reason5

    # Exact Hermite suffix slicing equivalence for both cubic and quintic.
    for traj in (traj3, traj5):
        t_cut = 0.42 * traj.duration
        suf = slice_trajectory_suffix(traj, t_cut)
        assert suf.times[0] == 0.0
        assert suf.duration == pytest.approx(traj.duration - t_cut, abs=1e-12)
        dts = np.linspace(0.0, suf.duration, 20)
        po, vo, ao = traj.sample(t_cut + dts)
        ps, vs, a_s = suf.sample(dts)
        np.testing.assert_allclose(ps, po, atol=1e-11)
        np.testing.assert_allclose(vs, vo, atol=1e-10)
        if traj.spline_order == 5:
            np.testing.assert_allclose(a_s, ao, atol=1e-9)

    # Counterexample 1: duplicate first waypoint (q1 == q0) with non-zero start_velocity must preserve v0.
    dup = np.array([[0.0, 0.0], [0.0, 0.0], [0.5, 0.3]], dtype=np.float64)
    traj_dup = retime_trajectory(dup, vmax, amax, spline_order=3, start_velocity=v0, lower_limits=lo, upper_limits=hi)
    np.testing.assert_allclose(traj_dup.sample(0.0)[1], v0, atol=1e-12)
    ok_dup, reason_dup, _ = check_trajectory_bounds(traj_dup, vmax, amax, lower_limits=lo, upper_limits=hi)
    assert ok_dup, reason_dup

    # Counterexample 2: start_acceleration on spline_order == 3 must raise ValueError.
    with pytest.raises(ValueError, match="spline_order == 5"):
        retime_trajectory(pos, vmax, amax, spline_order=3, start_velocity=v0, start_acceleration=a0)
    with pytest.raises(ValueError, match="spline_order == 5"):
        MotionPlanRequest(
            start=pos[0], goal=pos[-1], start_velocity=v0, start_acceleration=a0,
            config=SolverConfig(spline_order=3),
        )

    # Counterexample 3: deterministic infeasible quintic boundary (v0 == +vmax, a0 > 0) must fail bounds check.
    v_lim = np.array([2.0, 0.0], dtype=np.float64)
    a_pos = np.array([1.0, 0.0], dtype=np.float64)
    with pytest.raises(ValueError, match="velocity limit"):
        retime_trajectory(pos, vmax, amax, jerk_limits=jmax, spline_order=5,
                          start_velocity=v_lim, start_acceleration=a_pos, lower_limits=lo, upper_limits=hi)


def test_fixed_dt_start_acceleration_and_locked_joint_boundary(simple_snapshot):
    """Phase 3.2: fixed-dt 1/2 a h^2 first-segment bound and locked-joint start_velocity rejection."""
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    world.define_group("xy_only", [("planning", "x"), ("planning", "y")])
    planner = MotionPlanner(world, backend="coal")

    q0 = np.array([-0.6, 0.35, 0.0], dtype=np.float64)
    qg = np.array([0.4, 0.35, 0.0], dtype=np.float64)
    h = 0.25
    v0 = np.array([0.8, 0.0, 0.0], dtype=np.float64)

    res = planner.plan(MotionPlanRequest(
        start=q0, goal=qg, start_velocity=v0,
        config=SolverConfig(waypoints=10, dt=h, stop_at_endpoints=True, margin=0.01, world_margin=0.01),
    ))
    assert res.success, res.reason
    assert res.trajectory is not None
    p_s, v_s, _ = res.trajectory.sample(0.0)
    np.testing.assert_allclose(p_s, q0, atol=1e-9)
    np.testing.assert_allclose(v_s, v0, atol=1e-9)
    # First segment must satisfy |q1 - q0 - h * v0| <= 0.5 * a_max * h^2.
    amax = world.acceleration_limits
    diff = np.abs(res.trajectory.positions[1] - res.trajectory.positions[0] - h * v0)
    assert np.all(diff <= 0.5 * amax * h * h * (1.0 + 1e-5))

    # Non-zero start_velocity on locked inactive joint (index 2) must fail with boundary conflict.
    res_lock = planner.plan(MotionPlanRequest(
        start=q0, goal=qg, group="xy_only",
        start_velocity=np.array([0.2, 0.0, 0.5], dtype=np.float64),
        config=SolverConfig(waypoints=10, margin=0.01, world_margin=0.01),
    ))
    assert not res_lock.success
    assert res_lock.reason is not None and "locked joint" in res_lock.reason.lower()


def test_reactive_replan_reuse_drift_ambiguity_fallback_and_stale_world(simple_snapshot):
    """Phase 3.2: replan reuse/drift/ambiguity/BiRRT-fallback/timeout and stale world_version detection."""
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    planner = MotionPlanner(world, backend="coal")

    q_start = np.array([-0.6, 0.40, 0.0], dtype=np.float64)
    q_goal = np.array([0.6, 0.40, 0.0], dtype=np.float64)
    cfg = SolverConfig(waypoints=12, margin=0.005, world_margin=0.01, stop_at_endpoints=True, strict_ccd=True)

    init = planner.plan(MotionPlanRequest(start=q_start, goal=q_goal, config=cfg))
    assert init.success and init.trajectory is not None
    t_cut = 0.35 * init.trajectory.duration
    q_now, dq_now, _ = init.trajectory.sample(t_cut)

    # 1. Exact state + goal=None (keeps prev_traj final config) -> reused_prev.
    r_reuse = planner.replan(q_now, dq_now, prev_traj=init.trajectory, t_on_prev=t_cut, config=cfg)
    assert r_reuse.success and r_reuse.trajectory is not None
    assert r_reuse.diagnostics.seed_strategy == "reused_prev"
    assert r_reuse.world_version == world.version
    np.testing.assert_allclose(r_reuse.trajectory.positions[-1], q_goal, atol=1e-9)

    # 2. Small drift beyond rounding tolerance -> warm_start SQP matching exact (q_now, dq_now).
    q_drift = q_now + np.array([0.0, 0.012, 0.0])
    r_drift = planner.replan(q_drift, dq_now, prev_traj=init.trajectory, t_on_prev=t_cut, config=cfg)
    assert r_drift.success and r_drift.trajectory is not None
    assert r_drift.diagnostics.seed_strategy == "warm_start"
    np.testing.assert_allclose(r_drift.trajectory.sample(0.0)[0], q_drift, atol=1e-9)
    np.testing.assert_allclose(r_drift.trajectory.sample(0.0)[1], dq_now, atol=1e-9)

    # 3. Reversal trajectory without t_on_prev -> ambiguity disables direct reuse, still succeeds via warm_start.
    from gobot.planning import retime_trajectory
    rev_knots = np.array([
        [-0.6, 0.40, 0.0],
        [0.0, 0.40, 0.0],
        [0.6, 0.40, 0.0],
        [0.0, 0.40, 0.0],
        [-0.6, 0.40, 0.0],
    ], dtype=np.float64)
    rev_traj = retime_trajectory(rev_knots, world.velocity_limits, world.acceleration_limits, stop_at_endpoints=True)
    r_ambig = planner.replan(
        np.array([0.0, 0.40, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        prev_traj=rev_traj,
        t_on_prev=None,
        config=cfg,
    )
    assert r_ambig.success and r_ambig.trajectory is not None
    assert r_ambig.diagnostics.seed_strategy != "reused_prev"

    # 4. New waypoint_targets or spline_order mismatch disables direct reuse.
    cfg5 = SolverConfig(waypoints=12, margin=0.005, world_margin=0.01, stop_at_endpoints=True, spline_order=5, strict_ccd=True)
    r_ord = planner.replan(q_now, dq_now, prev_traj=init.trajectory, t_on_prev=t_cut, config=cfg5)
    assert r_ord.success and r_ord.trajectory is not None
    assert r_ord.diagnostics.seed_strategy != "reused_prev"
    assert r_ord.trajectory.spline_order == 5

    # 5. Dynamic obstacle blocks previous path + stale world_version detection before handoff.
    tip_mid, _ = world.link_pose(np.array([0.22, 0.42, 0.0]), ("planning", "left_tip"))
    world.add_box("blocker", (0.14, 0.14, 0.25), tip_mid)
    # Previously planned r_reuse is now stale (world_version mismatch) and invalid on the updated world!
    assert r_reuse.world_version != world.version
    ok_stale, _ = world.validate_trajectory(r_reuse.trajectory, margin=0.005, world_margin=0.01, strict_ccd=True)
    assert not ok_stale

    # Replan on updated world detours around blocker while preserving exact (q_now, dq_now).
    r_detour = planner.replan(q_now, dq_now, prev_traj=init.trajectory, t_on_prev=t_cut, time_budget_s=0.5, config=cfg)
    assert r_detour.success and r_detour.trajectory is not None, r_detour.reason
    assert r_detour.world_version == world.version
    assert r_detour.diagnostics.seed_strategy != "reused_prev"
    np.testing.assert_allclose(r_detour.trajectory.sample(0.0)[0], q_now, atol=1e-9)
    np.testing.assert_allclose(r_detour.trajectory.sample(0.0)[1], dq_now, atol=1e-9)
    ok_det, err_det = world.validate_trajectory(r_detour.trajectory, margin=0.005, world_margin=0.01, strict_ccd=True)
    assert ok_det, err_det

    # 6. Global BiRRT fallback preserving exact start velocity when warm-start SQP is trapped by a wall.
    world.update_obstacle_pose("blocker", tip_mid + np.array([0.0, 0.0, 0.0]))
    world.add_box("wall_extra", (0.12, 0.42, 0.25), tip_mid + np.array([0.0, -0.14, 0.0]))
    r_fb = planner.replan(
        q_now, dq_now, prev_traj=init.trajectory, t_on_prev=t_cut,
        time_budget_s=1.0, config=SolverConfig(
            waypoints=14, margin=0.005, world_margin=0.01, stop_at_endpoints=True,
            iterations=15, use_global_seeding=True, strict_ccd=True, rrt_timeout_s=0.6,
        ),
    )
    assert r_fb.success and r_fb.trajectory is not None, r_fb.reason
    assert r_fb.diagnostics.seed_strategy == "birrt"
    np.testing.assert_allclose(r_fb.trajectory.sample(0.0)[0], q_now, atol=1e-9)
    np.testing.assert_allclose(r_fb.trajectory.sample(0.0)[1], dq_now, atol=1e-9)
    ok_fb, err_fb = world.validate_trajectory(r_fb.trajectory, margin=0.005, world_margin=0.01, strict_ccd=True)
    assert ok_fb, err_fb

    # 7. Exhausted time_budget_s reports timed_out=True cleanly.
    r_to = planner.replan(q_now, dq_now, prev_traj=init.trajectory, t_on_prev=t_cut, time_budget_s=1e-6, config=cfg)
    assert not r_to.success
    assert r_to.diagnostics.timed_out


def test_panda_closed_loop_reactive_replanning_coal_and_esdf(panda_snapshot):
    """Phase 3.2 Acceptance: 7-DOF Panda 20 Hz closed-loop reactive replanning (Coal, ESDF, Quintic C^2)."""
    import importlib.util

    demo_path = Path(__file__).resolve().parents[2] / "examples" / "reactive_replanning" / "demo.py"
    spec = importlib.util.spec_from_file_location("reactive_replanning_demo", demo_path)
    assert spec is not None and spec.loader is not None
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)

    for mode, order in (("coal", 3), ("esdf", 3), ("coal", 5)):
        summary = demo.run_demo(mode=mode, hz=20.0, spline_order=order, strict_ccd=True, max_ticks=90, verbose=False)
        assert summary["reached_goal"], f"{mode} (order={order}) demo did not reach goal: {summary}"
        assert summary["success_rate"] >= 0.90, f"{mode} (order={order}) low success rate: {summary}"
        assert summary["max_pos_jump"] <= 1e-6, f"{mode} pos jump {summary['max_pos_jump']}"
        assert summary["max_vel_jump"] <= 1e-5, f"{mode} vel jump {summary['max_vel_jump']}"
        if order == 5:
            assert summary["max_acc_jump"] <= 1e-4, f"{mode} acc jump {summary['max_acc_jump']}"
        assert "reused_prev" in summary["paths"]
        assert ("warm_start" in summary["paths"]) or ("global_fallback" in summary["paths"])


def test_reactive_retime_rejects_infeasible_boundary_and_preserves_short_knots():
    from gobot.planning import slice_trajectory_suffix

    for order in (3, 5):
        with pytest.raises(ValueError, match="velocity"):
            retime_trajectory([[0.], [.5]], [1.], [2.], dt=.1, start_velocity=[2.], spline_order=order)
        times = np.array([0., 5e-13, 1.])
        q = np.array([[0.], [1.], [0.]])
        prev = Trajectory(times, q, np.zeros_like(q), np.zeros_like(q), spline_order=order)
        suffix = slice_trajectory_suffix(prev, 0.)
        np.testing.assert_array_equal(suffix.times, times)
        np.testing.assert_array_equal(suffix.sample(times)[0], prev.sample(times)[0])
        bad = Trajectory(times[::-1], q, np.zeros_like(q), np.zeros_like(q), spline_order=order)
        with pytest.raises(ValueError, match="non-decreasing"):
            slice_trajectory_suffix(bad, 0.)


def test_short_quintic_acceleration_check_is_translation_invariant():
    from gobot.planning.retime import check_trajectory_bounds

    for offset in (0., .5):
        positions = np.full((2, 1), offset)
        prev = Trajectory(np.array([0., 1e-16]), positions, np.full_like(positions, .1),
                          np.zeros_like(positions), spline_order=5)
        assert abs(prev.sample(2.5e-17)[2][0]) > 1e15
        ok, reason, bad_segments = check_trajectory_bounds(prev, np.ones(1), np.ones(1))
        assert not ok and "acceleration" in reason and bad_segments == [0]


def test_replan_reuse_respects_deadline_state_and_locked_quintic_joint(simple_snapshot):
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    planner = MotionPlanner(world)
    q = np.array([-.5, .4, 0.])
    goal = q + [.15, 0., 0.]
    cfg = SolverConfig(waypoints=10, strict_ccd=True, stop_at_endpoints=True, spline_order=5)
    pos = np.stack([q, goal, goal])
    prev = Trajectory(np.arange(3.), pos, np.zeros_like(pos), np.zeros_like(pos), spline_order=5)
    res = planner.replan(q, np.zeros(3), prev, config=cfg, t_on_prev=0.)
    assert res.success and res.diagnostics.seed_strategy == "reused_prev"
    assert len(res.waypoint_indices) == cfg.waypoints
    assert res.waypoint_indices[-1] == 2

    late = planner.replan(q, np.zeros(3), prev, config=cfg, t_on_prev=0., time_budget_s=1e-9)
    assert not late.success and late.diagnostics.timed_out
    drift = q + [0., 5e-7, 0.]
    corrected = planner.replan(drift, np.zeros(3), prev, config=cfg, t_on_prev=0.)
    assert corrected.success and corrected.diagnostics.seed_strategy != "reused_prev"
    np.testing.assert_allclose(corrected.trajectory.positions[0], drift, atol=1e-14, rtol=0.)
    # Equal positions and zero velocities at every knot still allow a quintic joint excursion.
    prev.accelerations[1, 1] = .4
    assert abs(prev.sample(.5)[0][1] - q[1]) > .001
    locked = planner.replan(q, np.zeros(3), prev, config=cfg, group=[("planning", "x")], t_on_prev=0.)
    assert locked.success and locked.diagnostics.seed_strategy != "reused_prev"
    np.testing.assert_allclose(locked.trajectory.sample(np.linspace(0., locked.trajectory.duration, 31))[0][:, 1], q[1])


@pytest.mark.parametrize("order", [3, 5])
def test_replan_anchor_on_nonuniform_loop(simple_snapshot, order):
    world = PlanningWorld.from_snapshot(simple_snapshot, {"planning": "base"})
    planner = MotionPlanner(world)
    cfg = SolverConfig(waypoints=10, spline_order=order, strict_ccd=True)
    times = np.array([0., 1., 3., 4., 5., 80.])
    positions = np.zeros((len(times), 3))
    positions[:, 0] = [-.375, -.25, -.421875, -.296875, -.20, -.20]
    positions[:, 1] = .4
    velocities = np.zeros_like(positions)
    velocities[:4, 0] = .125
    prev = Trajectory(times, positions, velocities, np.zeros_like(positions), spline_order=order)

    # The initial state occurs a second time inside the short constant-velocity segment.
    q, v, _ = prev.sample(0.)
    t_repeat = 3.375
    np.testing.assert_allclose(prev.sample(t_repeat)[:2], (q, v), atol=1e-14, rtol=0.)
    explicit = planner.replan(q, v, prev, t_on_prev=t_repeat, config=cfg)
    assert explicit.success and explicit.diagnostics.seed_strategy == "reused_prev"
    assert explicit.trajectory.duration == times[-1] - t_repeat
    automatic = planner.replan(q, v, prev, config=cfg)
    assert automatic.success, automatic.reason
    assert automatic.diagnostics.seed_strategy != "reused_prev"
