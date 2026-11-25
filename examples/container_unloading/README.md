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
  (`planning_sqp.hpp` / `planning_qp.hpp`) finds a verified 14-waypoint
  collision-free path with 14 mm environment clearance and 1 mm self-collision
  clearance (`~180 ms` cold full unloading without via, `~63 ms` with via).
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
.venv/bin/python examples/container_unloading/benchmark_batch.py --sizes 1 4 8 32 64 128 --repeats 3 --solver sqp_fast
```

Measured on a Ryzen 7 3700X (8C/16T) and RTX 4070 (`cuda:0`) across 350 shape
pairs, 12 collision substeps, and `dt=None` TOPP retiming (median of 3 warm
runs; every returned trajectory is independently verified along its actual
continuous cubic Hermite curve at 313 configurations on both convex hulls and
authored triangle meshes, plus exact velocity and acceleration limits):

- **C++ Parallel Batch IK**: `0.14 ms` (`B=1`), `0.46 ms` (`B=4`), `0.84 ms` (`B=8`), `1.56 ms` (`B=16`), `3.05 ms` (`B=32`), `6.34 ms` (`B=64`), `12.35 ms` (`B=128`, `0.10 ms/solve`)
- **Short transfer (zero-QP fast path)**: `1.74 ms` median (11 warm runs)

| Trajectories (`B`) | Backend | `sqp_fast` median | Per trajectory | Solved & Verified | Min carton clearance |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `coal` | **0.063 s** | 63.1 ms | 1/1 | 14.28 mm |
| 1 | `mujoco_warp` | **0.175 s** | 174.6 ms | 1/1 | 14.28 mm |
| 4 | `coal` | **0.205 s** | 51.2 ms | 4/4 | 14.00 mm |
| 4 | `mujoco_warp` | **0.376 s** | 94.1 ms | 4/4 | 14.28 mm |
| 8 | `coal` | **0.486 s** | 60.7 ms | 8/8 | 14.00 mm |
| 8 | `mujoco_warp` | **0.981 s** | 122.6 ms | 8/8 | 14.00 mm |
| 32 | `coal` | **1.162 s** | 36.3 ms | 32/32 | 14.00 mm |
| 32 | `mujoco_warp` | **2.895 s** | 90.5 ms | 32/32 | 14.00 mm |
| 64 | `coal` | **2.305 s** | 36.0 ms | 64/64 | 14.00 mm |
| 64 | `mujoco_warp` | **5.265 s** | 82.3 ms | 64/64 | 14.00 mm |
| 128 | `coal` | **4.736 s** | 37.0 ms | 128/128 | 14.00 mm |
| 128 | `mujoco_warp` | **11.028 s** | 86.2 ms | 128/128 | 14.00 mm |

## Requirements and assets

Use a Coal-enabled Gobot build (`-DGOB_BUILD_COAL=ON`) and the `planning-sqp`
Python extra for OSQP. See the [Panda build instructions](../panda_trajopt/README.md#build).

The workcell dimensions and tooling are authored for this example. The FANUC
M-10iA represents the six-axis industrial arm in the unloading footage. The
seven STL meshes and joint frames come from ROS-Industrial's FANUC support
package at revision `d8f42bd73584b255df87098395512538882caea1`.
[Model source](https://github.com/ros-industrial/fanuc/tree/d8f42bd73584b255df87098395512538882caea1/fanuc_m10ia_support) ·
[BSD license](meshes/LICENSE)
