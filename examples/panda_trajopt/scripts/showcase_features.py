"""Headless showcase of Gobot's C++-backed motion planning architecture on the Panda workcell.

Demonstrates:
1. Declarative Pose-to-Pose + Waypoint planning with TOPP time-optimal retiming
2. Singularity-free free-axis PoseConstraint (free tool Z rotation)
3. Multi-candidate GoalSet with C++ parallel batch IK & automatic best-goal selection
4. Full-trajectory PathConstraint (keep hand pointing down along the entire motion)
5. Mutable PlanningWorld: dynamic obstacle add / O(1) pose update / remove, attach/detach payload, ACM
6. Hybrid global + local planning: C++ BiRRT-Connect + Batch Racing around a blocking baffle
7. PlanningGroup partial-DOF optimization
"""

from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

import gobot
from gobot.planning import (
    MotionPlanner,
    MotionPlanRequest,
    PathConstraint,
    PlanningWorld,
    PoseConstraint,
    SolverConfig,
)
from plan import DOWN, HAND, ROBOT, START_SEED, build_world


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    ctx = gobot.app.context()
    ctx.set_project_path(str(project_dir))
    ctx.load_scene("res://world.jscn")
    try:
        snapshot = ctx.compile_physics_scene_snapshot()
        start_pos = np.asarray(ctx.root.find("Start").position, dtype=np.float64)
        goal_pos = np.asarray(ctx.root.find("Goal").position, dtype=np.float64)
        insp_pos = np.asarray(ctx.root.find("Inspection").position, dtype=np.float64)
    finally:
        ctx.clear_scene()

    world = build_world(snapshot)
    planner = MotionPlanner(world, backend="coal")

    # 1. Free-axis PoseConstraint (allow arbitrary yaw around world Z).
    _section("1. Free-Axis PoseConstraint (free Z rotation)")
    free_z_ik = planner.solve_ik(
        [PoseConstraint(HAND, start_pos, DOWN, rotation_tolerance=(1e-3, 1e-3, math.inf))],
        seed=START_SEED,
        margin=0.002,
        world_margin=0.020,
    )
    assert free_z_ik.success, free_z_ik.reason
    pos_fk, rot_fk = world.link_pose(free_z_ik.positions, HAND)
    z_align = float(np.dot(rot_fk[:, 2], DOWN[:, 2]))
    print(f"  pos_err={1000.0 * np.linalg.norm(pos_fk - start_pos):.4f} mm, tool-Z dot target-Z={z_align:.6f}")

    # 2. Multi-candidate GoalSet (4 yaw orientations, C++ parallel batch IK + shortest-path selection).
    _section("2. Multi-Candidate GoalSet")
    candidates = [
        [PoseConstraint(HAND, goal_pos, Rotation.from_euler("z", deg, degrees=True).as_matrix() @ DOWN)]
        for deg in (-60.0, -20.0, 20.0, 60.0)
    ]
    res_gs = planner.plan(MotionPlanRequest(
        start=free_z_ik.positions,
        goal=candidates,
        config=SolverConfig(
            waypoints=12, dt=None, retime=True, margin=0.002, world_margin=0.020,
            collision_substeps=4, first_feasible=True, use_global_seeding=True, num_seeds=4,
        ),
    ))
    assert res_gs.success, res_gs.reason
    print(
        f"  selected candidate #{res_gs.goal_index} (yaw={(-60, -20, 20, 60)[res_gs.goal_index]} deg), "
        f"strategy={res_gs.diagnostics.seed_strategy}, "
        f"{res_gs.diagnostics.total_ms:.1f} ms (ik={res_gs.diagnostics.ik_ms:.1f} ms), "
        f"duration={res_gs.trajectory.duration:.2f} s"
    )

    # 3. Full-trajectory PathConstraint (keep tool pointing down throughout the transfer).
    _section("3. Full-Trajectory PathConstraint (keep hand upright / pointing down)")
    res_pc = planner.plan(MotionPlanRequest(
        start=free_z_ik.positions,
        goal=[PoseConstraint(HAND, insp_pos, DOWN, rotation_tolerance=(1e-3, 1e-3, math.inf))],
        path_constraints=[
            PathConstraint(
                PoseConstraint(HAND, rotation=DOWN, rotation_tolerance=(0.05, 0.05, math.inf)),
            ),
        ],
        config=SolverConfig(
            waypoints=12, dt=None, retime=True, margin=0.002, world_margin=0.015,
            collision_substeps=4, stop_at_endpoints=True, first_feasible=True,
        ),
    ))
    assert res_pc.success, res_pc.reason
    _, rots = world.link_pose(res_pc.positions, HAND)
    tilts = np.degrees(np.arccos(np.clip(np.einsum("ij,j->i", rots[:, :, 2], DOWN[:, 2]), -1.0, 1.0)))
    print(
        f"  solved in {res_pc.diagnostics.total_ms:.1f} ms ({res_pc.diagnostics.qp_calls} QPs), "
        f"max tool tilt={tilts.max():.2f} deg (limit ~2.86 deg)"
    )

    # 4. Dynamic PlanningWorld: add_box -> attach_object -> plan -> detach_object -> O(1) update -> remove.
    _section("4. Mutable PlanningWorld (attach/detach payload + O(1) obstacle update)")
    q_pick = free_z_ik.positions
    p_hand, r_hand = world.link_pose(q_pick, HAND)
    payload_offset = np.array([0.0, 0.0, 0.14])
    payload_world_pos = p_hand + r_hand @ payload_offset
    world.add_box("grasped_brick", (0.05, 0.05, 0.06), payload_world_pos, r_hand)
    world.attach_object("grasped_brick", HAND, local_position=payload_offset, touch_links=[HAND])
    res_carried = planner.plan(MotionPlanRequest(
        start=q_pick,
        goal=res_gs.positions[-1],
        config=SolverConfig(
            waypoints=12, dt=None, retime=True, margin=0.002, world_margin=0.015,
            collision_substeps=4, first_feasible=True, use_global_seeding=True, num_seeds=4,
        ),
    ))
    assert res_carried.success, res_carried.reason
    world.detach_object("grasped_brick", res_carried.positions[-1])
    t0 = perf_counter()
    world.update_obstacle_pose("grasped_brick", payload_world_pos + np.array([0.0, 0.0, 0.05]))
    update_us = (perf_counter() - t0) * 1e6
    world.remove_obstacle("grasped_brick")
    print(
        f"  carried brick transfer: {res_carried.diagnostics.total_ms:.1f} ms "
        f"(strategy={res_carried.diagnostics.seed_strategy}), "
        f"O(1) obstacle pose update: {update_us:.1f} us"
    )

    # 5. Hybrid global + local planning: C++ BiRRT + Batch Racing around a dynamic baffle.
    _section("5. Hybrid C++ BiRRT + Batch Racing")
    baffle_center = np.array([float(insp_pos[0]), float(insp_pos[1]), float(insp_pos[2]) + 0.16])
    world.add_box("temporary_baffle", (0.05, 0.12, 0.24), baffle_center)
    res_rrt = planner.plan(MotionPlanRequest(
        start=q_pick,
        goal=res_gs.positions[-1],
        config=SolverConfig(
            waypoints=14, dt=None, retime=True, margin=0.002, world_margin=0.015,
            collision_substeps=4, first_feasible=True, use_global_seeding=True, num_seeds=4,
        ),
    ))
    world.remove_obstacle("temporary_baffle")
    assert res_rrt.success, res_rrt.reason
    print(
        f"  strategy={res_rrt.diagnostics.seed_strategy}, "
        f"total={res_rrt.diagnostics.total_ms:.1f} ms "
        f"(seed={res_rrt.diagnostics.seed_ms:.1f}, sqp={res_rrt.diagnostics.sqp_ms:.1f} ms), "
        f"duration={res_rrt.trajectory.duration:.2f} s"
    )

    # 6. PlanningGroup (freeze base joint panda_joint1, optimize joints 2..7 only).
    _section("6. PlanningGroup (lock panda_joint1, optimize upper 6 joints)")
    upper_arm = world.define_group("upper_arm", world.joint_names[1:])
    q_group_goal = q_pick.copy()
    q_group_goal[3] = np.clip(q_group_goal[3] + 0.35, world.lower_limits[3], world.upper_limits[3])
    q_group_goal[5] = np.clip(q_group_goal[5] - 0.25, world.lower_limits[5], world.upper_limits[5])
    res_grp = planner.plan(MotionPlanRequest(
        start=q_pick,
        goal=q_group_goal,
        group=upper_arm,
        config=SolverConfig(waypoints=10, dt=None, retime=True, margin=0.002, world_margin=0.015),
    ))
    assert res_grp.success, res_grp.reason
    drift_j1 = float(np.max(np.abs(res_grp.positions[:, 0] - q_pick[0])))
    q_mid, v_mid, _ = res_grp.trajectory.sample(0.5 * res_grp.trajectory.duration)
    print(
        f"  locked joint1 drift={drift_j1:.2e} rad, "
        f"mid-trajectory |v|_max={float(np.max(np.abs(v_mid))):.2f} rad/s, "
        f"total={res_grp.diagnostics.total_ms:.1f} ms"
    )


if __name__ == "__main__":
    main()
