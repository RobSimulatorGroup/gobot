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
