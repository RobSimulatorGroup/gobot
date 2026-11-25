"""Plan the loaded container-unloading motion with gobot.planning."""

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

ROBOT = "M10iA"
PAYLOAD = (ROBOT, "Parcel")
LINK_NAMES = ["base_link", *(f"link_{i}" for i in range(1, 7)), PAYLOAD[1]]
DT = 0.23
SAMPLES_PER_SEGMENT = 24


def conveyor_motion(snapshot: dict, drop_pose: np.ndarray) -> np.ndarray:
    """World translations to seat the carton and carry it to the last supported position."""
    deck = next(s for s in snapshot["loose_shapes"] if s["name"] == "RollerConveyor_RollerEnvelopeCollider")
    robot = next(r for r in snapshot["robots"] if r["name"] == ROBOT)
    parcel = next(link for link in robot["links"] if link["name"] == PAYLOAD[1])
    box = next(s for s in parcel["shapes"] if s["name"] == "Parcel_CartonCollider")
    deck_pose = np.asarray(deck["transform"]["matrix"], dtype=np.float64)
    axes = deck_pose[:3, :3]
    center = axes.T @ (drop_pose[:3] - deck_pose[:3, 3])
    carton_axes = axes.T @ Rotation.from_quat(drop_pose[3:]).as_matrix()
    half_carton = np.abs(carton_axes) @ (np.asarray(box["size"], dtype=np.float64) / 2.0)
    half_deck = np.asarray(deck["size"], dtype=np.float64) / 2.0
    gap = center[2] - half_carton[2] - half_deck[2]
    if (
        np.any(np.abs(center[:2]) + half_carton[:2] > half_deck[:2])
        or abs(carton_axes[2, 2] - 1.0) > 1e-4
        or not 0.0 <= gap <= 0.05
    ):
        return np.empty((0, 3), dtype=np.float64)
    travel = center[1] + half_deck[1] - half_carton[1]
    return np.stack((-gap * axes[:, 2], -travel * axes[:, 1]))


def _arm_seed(position) -> list[float]:
    return [float(np.arctan2(position[1], position[0])), 0.12, -0.50, 0.0, 0.65, 0.0]


def solve(
    snapshot: dict,
    pick,
    drop,
    *,
    backend: str = "coal",
    first_feasible: bool = True,
    retime: bool = True,
) -> dict:
    """Plan a loaded pick -> drop transfer declaratively with C++ IK, SQP, and TOPP retiming."""
    world = PlanningWorld.from_snapshot(snapshot, {ROBOT: "base_link"})
    planner = MotionPlanner(world, backend=backend)

    pick_pos, pick_deg = pick
    drop_pos, drop_deg = drop
    pick_rot = Rotation.from_euler("xyz", pick_deg, degrees=True).as_matrix()
    drop_rot = Rotation.from_euler("xyz", drop_deg, degrees=True).as_matrix()

    cfg = SolverConfig(
        waypoints=14,
        dt=None if retime else DT,
        retime=retime,
        velocity_scale=0.65,
        acceleration_scale=0.6,
        margin=0.001,
        world_margin=0.014,
        collision_substeps=SAMPLES_PER_SEGMENT // 2,
        iterations=50,
        stop_at_endpoints=retime,
        first_feasible=first_feasible,
    )
    result = planner.plan(MotionPlanRequest(
        start=[PoseConstraint(PAYLOAD, pick_pos, pick_rot)],
        goal=[PoseConstraint(PAYLOAD, drop_pos, drop_rot)],
        start_seed=_arm_seed(pick_pos),
        goal_seed=_arm_seed(drop_pos),
        config=cfg,
    ))
    if not result.success:
        raise ValueError(result.reason)

    # Evaluate carton clearance on dense segment samples: seed polyline vs. actual retimed Hermite curve.
    u = (np.arange(SAMPLES_PER_SEGMENT, dtype=np.float64) / SAMPLES_PER_SEGMENT)[:, None]
    q_init = result.initial_positions
    seed_samples = np.concatenate(
        (((1.0 - u)[None, :, :] * q_init[:-1, None, :] + u[None, :, :] * q_init[1:, None, :]).reshape(-1, world.num_joints),
         q_init[-1:]),
        axis=0,
    )
    t_grid = np.concatenate(
        ((result.trajectory.times[:-1, None]
          + (np.arange(SAMPLES_PER_SEGMENT) / SAMPLES_PER_SEGMENT)[None, :]
          * np.diff(result.trajectory.times)[:, None]).reshape(-1),
         result.trajectory.times[-1:])
    )
    opt_samples, _, _ = result.trajectory.sample(t_grid)
    samples = np.stack((seed_samples, opt_samples), axis=0)
    payload_pairs = [
        i for i, (a, b) in enumerate(world.collision_pairs)
        if a.startswith(f"{ROBOT}/{PAYLOAD[1]}/") and not b.startswith(f"{ROBOT}/")
    ]
    distances = world.collision_distances(samples)[..., payload_pairs]
    min_by_pair = distances.min(axis=1)
    nearest = [world.collision_pairs[payload_pairs[int(i)]][1] for i in min_by_pair.argmin(axis=1)]

    drop_p, drop_r = world.link_pose(result.positions[-1], PAYLOAD)
    drop_pose = np.concatenate((drop_p, Rotation.from_matrix(drop_r).as_quat()))

    return {
        "world": world,
        "planner": planner,
        "result": result,
        "names": LINK_NAMES,
        "handoff": conveyor_motion(snapshot, drop_pose),
        "carton_clearance": min_by_pair.min(axis=1),
        "nearest": nearest,
    }


def _main() -> None:
    import gobot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("coal", "mujoco_warp"), default="coal")
    parser.add_argument("--full-sqp", action="store_true", help="Continue optimizing after first feasible path")
    parser.add_argument("--fixed-dt", action="store_true", help="Use fixed dt=0.23s instead of TOPP retiming")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    ctx = gobot.app.context()
    ctx.set_project_path(str(project_dir))
    ctx.load_scene("res://world.jscn")
    try:
        snapshot = ctx.compile_physics_scene_snapshot()
        pick = (ctx.root.find("Pick").position, ctx.root.find("Pick").rotation_degrees)
        drop = (ctx.root.find("Drop").position, ctx.root.find("Drop").rotation_degrees)
    finally:
        ctx.clear_scene()

    out = solve(
        snapshot, pick, drop,
        backend=args.backend,
        first_feasible=not args.full_sqp,
        retime=not args.fixed_dt,
    )
    res = out["result"]
    diag = res.diagnostics
    seed_mm, opt_mm = 1000.0 * out["carton_clearance"]
    seed_pair, opt_pair = (str(p).split("/")[-2] for p in out["nearest"])
    print(
        f"Container Unloading ({args.backend}, strategy={diag.seed_strategy}): "
        f"{diag.total_ms:.1f} ms total "
        f"(ik={diag.ik_ms:.1f}, seed={diag.seed_ms:.1f}, sqp={diag.sqp_ms:.1f}, "
        f"val={diag.validate_ms:.1f}, retime={diag.retime_ms:.1f} ms, qp={diag.qp_calls})"
    )
    print(
        f"  duration={res.trajectory.duration:.2f} s, "
        f"seed carton clearance={seed_mm:+.1f} mm ({seed_pair}), "
        f"optimized={opt_mm:+.1f} mm ({opt_pair}), "
        f"handoff_steps={len(out['handoff'])}"
    )


if __name__ == "__main__":
    _main()
