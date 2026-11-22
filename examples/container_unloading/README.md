# Container unloading with a FANUC M-10iA

Open the project from the repository root and press **Play**. Each run plans a loaded transfer from the pickup slot to the roller conveyor using the edited scene. **Pause** holds the two robots at the same instant. **Stop** restores the scene.

```sh
.venv/bin/gobot_editor --path examples/container_unloading
```

The 400 mm wide carton starts in a 470 mm opening, with 35 mm on each side. Four-high stacks flank the carton and a long shipping box spans the top of the opening. The workcell uses the approximate inside dimensions of a 20-foot dry container, 5.90 m long, 2.35 m wide and 2.39 m high. The default camera looks through the open double doors. Both folded-steel side walls, the roof, the closed end, the door frame and the open leaves have collision envelopes. Five packed tiers fill the back of the container. A fixed joint attaches the carton to the side-mounted vacuum tool during transport. The carton starts 30 mm above its support.

The translucent pink robot and carton follow the straight joint interpolation. The solid yellow robot and carton follow the optimized motion at the same playback speed. Watch the third carton in the left-hand stack as the pink wrist swings. The solid wrist first draws the carton out of the stack, then swings it toward the conveyor. Both trajectories use the same pickup and drop poses.

| Loaded trajectory | Minimum sampled carton-to-environment signed distance | Closest obstacle |
| --- | ---: | --- |
| Straight joint interpolation | −345.3 mm | Third carton in the port stack |
| Optimized | +14.0 mm | Long carton above the slot |

## Planning and playback

The worker computes both endpoint configurations with IK, then calls Gobot's CPU trust-region SQP solver with 14 joint waypoints spaced 0.23 s apart. Only the endpoints are specified. Coal queries signed distances between the robot hulls, the attached carton and the environment. Scene poses and the joint trajectory are recomputed on each Play.

Before playback the planner checks joint and velocity limits and validates collision geometry at 313 joint configurations, including segment interiors. The acceptance thresholds are 8 mm for the environment and 1 mm for self-collision. The table reports the minimum for the *carton* over those samples. Continuous-time clearance between samples is unverified.

At the end of the planned motion, the carton is 17 mm above the rollers. The robot holds its release pose while the carton settles and moves 2.46 m along the rollers and out through the doorway in a kinematic animation. The animation uses the roller envelope and carton dimensions to stop while the carton is still fully supported. Moving the drop outside the handoff area ends the replay at the loaded pose.

## Batch planning and backend comparison

`PlannerBatch(planner).solve_trajopt` accepts start and goal tensors shaped
`[B, J]` and returns `positions[B, T, J]`, a success mask, and a reason for
each failed row. Each task can constrain different waypoint indices with joint
vectors or world-space `PoseTarget` objects. The default seed interpolates
through the specified interior joint configurations. Supply `seed[B, T, J]`
to initialize from other trajectories.

```python
from gobot.planning import Planner, PlannerBatch, PoseTarget

planner = Planner(snapshot, {"M10iA": "base_link"}, collision_backend="mujoco_warp")
batch = PlannerBatch(planner)
result = batch.solve_trajopt(
    starts, goals, dt=0.23, waypoints=14,
    waypoint_targets=[{6: joint_via_0}, {8: PoseTarget(("M10iA", "Parcel"), xyz_1)}],
    margin=0.001, world_margin=0.014, collision_substeps=12,
)
```

Both backends use Torch FK, CPU SQP and OSQP. Compatible segments share a
batched distance or Jacobian evaluation. Each trajectory has its own QP,
trust radius and feasibility checks. Set `chunk_size` to bound simultaneous
trajectories (default 8).

Select `"coal"` for CPU convex queries or `"mujoco_warp"` for MuJoCo
Warp GJK/EPA on the current Warp device. Both require a Coal-enabled build.
Warp computes float32 distances and witnesses and checks the signed gap between
their support planes. Indeterminate queries use double-precision Coal. Both
planners verify the authored mesh geometry with Coal.

Run the headless comparison from the repository root.

```sh
.venv/bin/python examples/container_unloading/benchmark_batch.py --sizes 1 4 --repeats 3
```

The benchmark derives slightly different pickups, midpoints and drops from the
same FANUC transfer. Both planners receive the same tasks and seed construction.
The first timing covers FK and distances at identical configurations. The second
covers the full planner call including transfers, SQP and internal validation.
Model construction, JIT, endpoint IK and independent verification are outside that
timer. The benchmark also checks each solution on 313 configurations with Coal
at 1 mm self clearance and 14 mm environment clearance.

The planner caches static world AABBs and updates each moving shape's bounds
at each sampled configuration. Pairs whose AABB gap exceeds their clearance
pass validation immediately. Halfspaces and overlapping bounds go to narrow
phase. SQP linearization screens at each pair's clearance plus 50 mm, then
forms derivatives only for exact distances inside that window. Trial steps
recompute candidates. `collision_distances()` queries every enabled pair.

On a Ryzen 7 3700X and RTX 4070 with one Torch CPU thread, the medians of
two warm planner calls for 350 shape pairs were as follows.

| Trajectories | Coal before | Coal with screening | Warp before | Warp with screening | Verified per backend |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32.28 s | 12.64 s | 31.26 s | 12.43 s | 1/1 |
| 4 | 131.63 s | 51.47 s | 124.17 s | 49.76 s | 4/4 |

The one-trajectory Coal solve linearized 15 times at 12 samples per segment
and 10 times on the retry at 24 samples per segment. Across both grids,
41,433 of 1,911,000 sample-pairs needed exact distances at linearization.
25,461 rows needed collision gradients. In the four-trajectory benchmark
the largest public distance discrepancy across 1,252 configurations was
0.058 mm. Both backends achieved 14.00 mm minimum sampled carton clearance.

The benchmark also times a straight, 11.7 cm move from the drop configuration.
This path passes its first validation and requires zero QP steps. In an
11-run warm Coal measurement the median planner call took 8.65 ms, down
from 120 ms before AABB screening. The long container transfers in the table
require iterative optimization.

## Requirements and assets

Use a Coal-enabled Gobot build and the `planning-sqp` Python extra for OSQP. The Panda example documents the build command and system dependencies. [Build instructions](../panda_trajopt/README.md#build)

The workcell dimensions and tooling are authored for this example. The FANUC M-10iA represents the six-axis industrial arm in the unloading footage. The seven STL meshes and joint frames come from ROS-Industrial's FANUC support package at revision `d8f42bd73584b255df87098395512538882caea1`. [Model source](https://github.com/ros-industrial/fanuc/tree/d8f42bd73584b255df87098395512538882caea1/fanuc_m10ia_support) · [BSD license](meshes/LICENSE)
