# Panda inspection with C++-backed TrajOpt & Hybrid Planning

Open the workcell from the repository root and click **Play**.

```sh
.venv/bin/gobot_editor --path examples/panda_trajopt
```

The translucent pink arm follows the straight joint seed. The solid arm plays
the time-parameterized (`retime_trajectory`) optimized trajectory on the same
clock. Pink and cyan curves trace the hand, and red arrows mark active collision
witnesses via `MotionPlanner.debug_draw(render, result, link=HAND)`. **Pause**
holds both poses at the same instant. **Stop** restores the edited scene. Move
**Start**, **Goal**, the yellow **Inspection** marker, or any fixture and press
**Play** to replan in a background thread without launching an external
subprocess.

At waypoint 8 (zero-based) the hand must reach the marker inside the blue
inspection gauge, pointing down and turned 25 degrees about world Z from its
endpoint orientation. The Console reports the timing breakdown (`ik`, `seed`,
`sqp`, `validate`, `retime`), QP count, trajectory duration, pose error, seed
obstacle hits, and minimum sampled environment clearance.

The arm has 11 convex collision pieces for its base, joint housing, arm shells,
palm, and separate fingertips. The nine movable fixture nodes use four faceted
hull meshes with distinct sizes, orientations, and placements. The worktop,
gauge column, and both jaws also participate in collision checks. Three
mechanically connected pairs are excluded from self-collision tests via the
world's allowed collision matrix (ACM).

## Headless scripts

Run the Panda inspection planner headlessly:

```sh
.venv/bin/python examples/panda_trajopt/scripts/plan.py
```

Run the feature showcase covering free-axis `PoseConstraint`, multi-candidate
`GoalSet`, full-trajectory `PathConstraint`, mutable `PlanningWorld`
(`add_box` / `attach_object` / `detach_object` / O(1) `update_obstacle_pose` /
`remove_obstacle`), C++ BiRRT + Batch Racing, and `PlanningGroup`:

```sh
.venv/bin/python examples/panda_trajopt/scripts/showcase_features.py
```

## Planning architecture

Both scripts use Gobot's C++-backed `gobot.planning` stack (`PlanningWorld`,
`MotionPlanner`, `MotionPlanRequest`, `PoseConstraint`, `SolverConfig`):

```python
from gobot.planning import MotionPlanner, MotionPlanRequest, PlanningWorld, PoseConstraint, SolverConfig

world = PlanningWorld.from_snapshot(snapshot, {"panda": "panda_link0"}, ignored_link_pairs=...)
planner = MotionPlanner(world, backend="coal")

result = planner.plan(MotionPlanRequest(
    start=[PoseConstraint(("panda", "panda_hand"), start, DOWN)],
    goal=[PoseConstraint(("panda", "panda_hand"), goal, DOWN)],
    start_seed=START_SEED,
    waypoint_targets={8: PoseConstraint(("panda", "panda_hand"), inspection, inspection_rotation)},
    config=SolverConfig(
        waypoints=16, dt=None, retime=True, velocity_scale=0.6, acceleration_scale=0.5,
        margin=0.002, world_margin=0.025, collision_substeps=3, max_segment_length=0.075,
        stop_at_endpoints=True, iterations=40, first_feasible=False,
    ),
))
q, qd, qdd = result.trajectory.sample(0.5 * result.trajectory.duration)
```

Key pipeline stages:

- **C++ `PlanningWorld` (`_coal_native`)** owns multi-branch forward kinematics,
  exact analytic pose/collision Jacobians, Coal convex/mesh distance queries,
  early-exit AABB screening, Conservative Advancement continuous collision
  checking, parallel C++ IK, C++ BiRRT-Connect global seeding, and the native
  C++ trust-region SQP + ProxAL-SSN QP solver. No PyTorch or OSQP dependency is
  needed on the Coal planning hot path.
- **TOPP Retiming (`retime_trajectory`)** assigns exact cubic-Hermite velocity-
  and acceleration-bounded timestamps and exposes continuous evaluation via
  `result.trajectory.sample(t)`.

On a Ryzen 7 3700X (TBB enabled), the headless Panda inspection task solves in
**~31 ms** (0.5 ms IK, 27.6 ms C++ SQP over 12 QPs, 1.8 ms validation, 1.0 ms
C++ TOPP retiming + continuous Hermite CCD), achieving `0.001 mm` / `0.000 deg`
inspection error and `40.1 mm` minimum sampled world clearance.

## Build

The default Python build includes Coal and initializes its pinned sources:

```sh
uv sync --reinstall-package gobot
```

Coal uses CMake 3.22+, Eigen, Boost serialization/filesystem, and Assimp
development packages. Select `"coal"` (CPU) or `"mujoco_warp"` (GPU) via
`MotionPlanner(world, backend=...)`.

The Panda joint frames and limits are adapted from the Apache-2.0 Genesis
`panda_bullet/panda.urdf` at revision `ffd119c30118ebb9745fc009349efb2112a8cb76`.
The scene's hulls are authored for this example. The trajectory task builds on
[PyRoki's TrajOpt example](https://github.com/chungmin99/pyroki/blob/388e43e1fc0d0ee382968d3dd72970fd62a0450c/examples/07_trajopt.py).
