"""Plan the Panda inspection motion with the C++-backed gobot.planning pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from gobot.planning import (
    MotionPlanner,
    MotionPlanRequest,
    PlanningWorld,
    PoseConstraint,
    SolverConfig,
)

ROBOT = "panda"
HAND = (ROBOT, "panda_hand")
LINK_NAMES = [f"panda_link{i}" for i in range(9)] + [HAND[1]]
IGNORED_MOUNTS = (
    ("panda_link0", "panda_link2"),
    ("panda_link4", "panda_link6"),
    ("panda_link7", "panda_hand"),
)
DOWN = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
START_SEED = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, -2.3], dtype=np.float64)
INSPECTION_INDEX = 8


def build_world(snapshot: dict) -> PlanningWorld:
    return PlanningWorld.from_snapshot(
        snapshot,
        {ROBOT: "panda_link0"},
        ignored_link_pairs=[((ROBOT, a), (ROBOT, b)) for a, b in IGNORED_MOUNTS],
    )


def solve(
    snapshot: dict,
    start,
    goal,
    inspection,
    *,
    backend: str = "coal",
    first_feasible: bool = False,
) -> dict:
    """Solve Panda start -> inspection -> goal declaratively with TOPP retiming."""
    world = build_world(snapshot)
    planner = MotionPlanner(world, backend=backend)

    inspection_rotation = Rotation.from_euler("z", 25.0, degrees=True).as_matrix() @ DOWN
    request = MotionPlanRequest(
        start=[PoseConstraint(HAND, start, DOWN)],
        goal=[PoseConstraint(HAND, goal, DOWN)],
        start_seed=START_SEED,
        waypoint_targets={
            INSPECTION_INDEX: PoseConstraint(
                HAND,
                inspection,
                inspection_rotation,
                position_tolerance=1e-4,
                rotation_tolerance=1e-4,
            ),
        },
        config=SolverConfig(
            waypoints=16,
            dt=None,
            retime=True,
            velocity_scale=0.6,
            acceleration_scale=0.5,
            margin=0.002,
            world_margin=0.025,
            collision_substeps=3,
            max_segment_length=0.075,
            stop_at_endpoints=True,
            iterations=40,
            first_feasible=first_feasible,
        ),
    )

    result = planner.plan(request)
    if not result.success:
        raise ValueError(result.reason)

    valid, reason = world.validate_trajectory(
        result.trajectory,
        collision_substeps=24,
        margin=0.002,
        world_margin=0.020,
        velocity_scale=0.6,
        acceleration_scale=0.5,
        check_acceleration=True,
    )
    if not valid:
        raise ValueError(f"Original geometry failed 24-substep curve check: {reason}")

    position, rotation = world.link_pose(result.positions[INSPECTION_INDEX], HAND)
    position_error = float(np.linalg.norm(position - np.asarray(inspection, dtype=np.float64)))
    angle_error = float(Rotation.from_matrix(inspection_rotation.T @ rotation).magnitude())

    # Evaluate densely sampled clearance on seed polyline vs. actual retimed Hermite curve.
    steps = 24
    frac = (np.arange(steps, dtype=np.float64) / steps)[:, None]
    q_init = result.initial_positions
    seed_samples = np.concatenate(
        (((1.0 - frac)[None, :, :] * q_init[:-1, None, :] + frac[None, :, :] * q_init[1:, None, :]).reshape(-1, world.num_joints),
         q_init[-1:]),
        axis=0,
    )
    t_grid = np.concatenate(
        ((result.trajectory.times[:-1, None] + (np.arange(steps) / steps)[None, :] * np.diff(result.trajectory.times)[:, None]).reshape(-1),
         result.trajectory.times[-1:])
    )
    opt_samples, _, _ = result.trajectory.sample(t_grid)
    samples = np.stack((seed_samples, opt_samples), axis=0)
    world_pairs = [i for i, (_, other) in enumerate(world.collision_pairs) if not other.startswith(f"{ROBOT}/")]
    clearance = world.collision_distances(samples)[:, :, world_pairs].min(axis=1)
    hits = sorted({world.collision_pairs[world_pairs[i]][1].split("/")[-2]
                   for i in np.flatnonzero(clearance[0] < 0.0)})

    return {
        "world": world,
        "planner": planner,
        "result": result,
        "links": LINK_NAMES,
        "inspection_error": position_error,
        "inspection_angle": angle_error,
        "seed_hits": ", ".join(hits),
        "world_clearance": float(clearance[1].min()),
    }


def _main() -> None:
    import gobot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("coal", "mujoco_warp"), default="coal")
    parser.add_argument("--fast", action="store_true", help="Return first feasible trajectory")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    ctx = gobot.app.context()
    ctx.set_project_path(str(project_dir))
    ctx.load_scene("res://world.jscn")
    try:
        snapshot = ctx.compile_physics_scene_snapshot()
        start = ctx.root.find("Start").position
        goal = ctx.root.find("Goal").position
        inspection = ctx.root.find("Inspection").position
    finally:
        ctx.clear_scene()

    out = solve(snapshot, start, goal, inspection, backend=args.backend, first_feasible=args.fast)
    res = out["result"]
    diag = res.diagnostics
    print(
        f"Panda TrajOpt ({args.backend}, strategy={diag.seed_strategy}): "
        f"{diag.total_ms:.1f} ms total "
        f"(ik={diag.ik_ms:.1f}, seed={diag.seed_ms:.1f}, sqp={diag.sqp_ms:.1f}, "
        f"val={diag.validate_ms:.1f}, retime={diag.retime_ms:.1f} ms, qp={diag.qp_calls})"
    )
    print(
        f"  duration={res.trajectory.duration:.2f} s, "
        f"inspection_err={1000.0 * out['inspection_error']:.3f} mm / "
        f"{np.degrees(out['inspection_angle']):.3f} deg, "
        f"world_clearance={1000.0 * out['world_clearance']:.2f} mm, "
        f"seed_hits=[{out['seed_hits']}]"
    )


if __name__ == "__main__":
    _main()
