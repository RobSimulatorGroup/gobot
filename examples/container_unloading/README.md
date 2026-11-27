# Container unloading with a FANUC M-10iA

Open the project from the repository root and press **Play**. Each run plans a
loaded transfer from the pickup slot to the roller conveyor against the live
edited scene in a background thread. **Pause** holds both robots at the same
instant. **Stop** restores the scene.

```sh
.venv/bin/gobot_editor --path examples/container_unloading
```

Or run the planner headlessly from the command line:

```sh
.venv/bin/python examples/container_unloading/plan.py --backend coal
```

The 400 mm wide carton starts in a 470 mm opening, with 35 mm on each side.
Four-high stacks flank the carton and a long shipping box spans the top of the
opening. The workcell uses the approximate inside dimensions of a 20-foot dry
container, 5.90 m long, 2.35 m wide, and 2.39 m high. The default camera looks
through the open double doors. Both folded-steel side walls, the roof, the
closed end, the door frame, and the open leaves have collision envelopes. Five
packed tiers fill the back of the container. A fixed joint attaches the carton
to the side-mounted vacuum tool during transport. The carton starts 30 mm above
its support.

The translucent pink robot and carton follow the straight joint interpolation.
The solid yellow robot and carton follow the retimed optimized motion on the
same clock, with pink/cyan payload paths and red collision witnesses drawn via
`MotionPlanner.debug_draw(...)`. Watch the third carton in the left-hand stack
as the pink wrist swings: the solid wrist first draws the carton out of the
stack, then swings it toward the conveyor.

| Loaded trajectory | Minimum sampled carton-to-environment signed distance | Closest obstacle |
| --- | ---: | --- |
| Straight joint interpolation | −345.3 mm | Third carton in the port stack (`Port3`) |
| Optimized | +14.1 mm | Third carton in the starboard stack (`Starboard3`) |

## Planning and playback

`plan.py` submits a single declarative `MotionPlanRequest` with world-space
`PoseConstraint` start/goal targets to `MotionPlanner`:

- C++ `_coal_native` solves endpoint IK (`~0.14–1.2 ms`), evaluates multi-branch
  forward kinematics and exact analytic collision/pose Jacobians, and runs
  AABB-screened Coal queries + Conservative Advancement CCD on all 350 shape pairs.
- Native C++ trust-region SQP with a custom ProxAL-SSN L1-hinge box QP solver
  (`libgobot_planning.so`: `planning_sqp.hpp` / `planning_qp.hpp` /
  `planning_planner.hpp`, parallelized with TBB) finds a verified 14-waypoint
  collision-free path with 14 mm environment clearance and 1 mm self-collision
  clearance (`~122 ms` cold full unloading without via, `~36 ms` with via).
- `retime_trajectory` computes an exact cubic-Hermite velocity- and
  acceleration-bounded TOPP timing profile (`~2.86 s`), and `play.py` samples
  `result.trajectory.sample(t)` in real time with live C++ FK.

At the end of the planned motion, the carton is 17 mm above the rollers. The
robot holds its release pose while the carton settles and moves 2.46 m along the
rollers and out through the doorway in a kinematic animation. Moving the drop
marker outside the handoff area ends the replay at the loaded pose.

## Batch planning and backend comparison

`MotionPlanner.plan_batch` accepts a list of heterogeneous `MotionPlanRequest`
objects, merges compatible collision queries across active trajectories, solves
per-trajectory OSQP subproblems in parallel, and verifies both convex hulls and
authored triangle meshes.

```python
from gobot.planning import (
    MotionPlanner,
    MotionPlanRequest,
    PlanningWorld,
    PoseConstraint,
    SolverConfig,
)

world = PlanningWorld.from_snapshot(snapshot, {"M10iA": "base_link"})
planner = MotionPlanner(world, backend="mujoco_warp")  # or "coal"

cfg = SolverConfig(
    waypoints=14, dt=0.23, margin=0.001, world_margin=0.014,
    collision_substeps=12, iterations=50, first_feasible=True,
)
result = planner.plan_batch([
    MotionPlanRequest(start=starts[0], goal=goals[0], waypoint_targets={6: joint_via_0}, config=cfg),
    MotionPlanRequest(start=starts[1], goal=goals[1], waypoint_targets={8: PoseConstraint(("M10iA", "Parcel"), xyz_1)}, config=cfg),
])
```

Select `"coal"` for parallel CPU convex/mesh queries or `"mujoco_warp"` for
GPU GJK/EPA with cached Warp scratch buffers (automatically routing tiny
queries `< 128` pairs to CPU Coal and synchronizing on dynamic world updates).
Both backends share the same C++ `PlanningWorld`, analytic Jacobians, and
authored mesh verification.

Run the headless comparison from the repository root:

```sh
.venv/bin/python examples/container_unloading/benchmark_batch.py --sizes 1 4 8 32 --repeats 3 --solver sqp_fast
```

Measured on a Ryzen 7 3700X (8C/16T, oneTBB enabled) and RTX 4070 (`cuda:0`)
across 350 shape pairs, 12 collision substeps, and `dt=None` TOPP retiming
(median of 3 warm runs; every returned trajectory passes kinematic-chain
continuous Hermite CCD and is independently verified at 313 configurations on
both convex hulls and authored triangle meshes, plus exact velocity and
acceleration limits):

- **C++ Parallel Batch IK**: `0.13 ms` (`B=1`), `0.52 ms` (`B=4`), `0.80 ms` (`B=8`), `1.48 ms` (`B=16`), `2.85 ms` (`B=32`, `0.09 ms/solve`)
- **Short transfer (zero-QP fast path)**: `1.28 ms` median (11 warm runs)

| Trajectories (`B`) | Backend | `sqp_fast` median | Per trajectory | Solved & Verified | Min carton clearance |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `coal` | **0.036 s** | 35.7 ms | 1/1 | 14.24 mm |
| 1 | `mujoco_warp` (exp.) | **0.136 s** | 136.1 ms | 1/1 | 14.24 mm |
| 4 | `coal` | **0.050 s** | 12.5 ms | 4/4 | 14.24 mm |
| 4 | `mujoco_warp` (exp.) | **0.259 s** | 64.8 ms | 4/4 | 14.24 mm |
| 8 | `coal` | **0.149 s** | 18.6 ms | 8/8 | 14.00 mm |
| 8 | `mujoco_warp` (exp.) | **0.747 s** | 93.3 ms | 8/8 | 14.00 mm |
| 32 | `coal` | **0.402 s** | 12.6 ms | 32/32 | 14.00 mm |
| 32 | `mujoco_warp` (exp.) | **2.189 s** | 68.4 ms | 32/32 | 14.00 mm |
| 64 | `coal` | **0.901 s** | 14.1 ms | 64/64 | — |
| 128 | `coal` | **1.666 s** | 13.0 ms | 128/128 | — |

## Requirements and assets

The default Gobot build includes the native Coal planner used by this example.
See the [Panda build instructions](../panda_trajopt/README.md#build).

The workcell dimensions and tooling are authored for this example. The FANUC
M-10iA represents the six-axis industrial arm in the unloading footage. The
seven STL meshes and joint frames come from ROS-Industrial's FANUC support
package at revision `d8f42bd73584b255df87098395512538882caea1`.
[Model source](https://github.com/ros-industrial/fanuc/tree/d8f42bd73584b255df87098395512538882caea1/fanuc_m10ia_support) ·
[BSD license](meshes/LICENSE)
