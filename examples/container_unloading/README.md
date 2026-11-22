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
| Optimized | +15.3 mm | Third carton in the starboard stack |

## Planning and playback

The worker computes both endpoint configurations with IK, then calls Gobot's CPU trust-region SQP solver with 14 joint waypoints spaced 0.23 s apart. The example uses `solver="sqp_fast"` to return the first verified path. Only the endpoints are specified. Coal queries signed distances between the robot hulls, the attached carton and the environment. Scene poses and the joint trajectory are recomputed on each Play.

Before playback the planner checks joint and velocity limits and validates collision geometry at 313 joint configurations, including segment interiors. The acceptance thresholds are 14 mm for the environment and 1 mm for self-collision. The table reports the minimum for the *carton* over those samples. Continuous-time clearance between samples is unverified.

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
    margin=0.001, world_margin=0.014, collision_substeps=12, solver="sqp_fast",
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
.venv/bin/python examples/container_unloading/benchmark_batch.py --sizes 1 4 --repeats 3 --solver sqp_fast
```

The benchmark derives slightly different pickups, midpoints and drops from the
same FANUC transfer. Both planners receive the same tasks and seed construction.
The first timing covers FK and distances at identical configurations. The second
covers the full planner call including transfers, SQP and internal validation.
Model construction, JIT, endpoint IK and independent verification are outside that
timer. The benchmark also checks each solution on 313 configurations with Coal
at 1 mm self clearance and 14 mm environment clearance.

The planner caches static world AABBs and updates each moving shape's bounds
at each sampled configuration. For each trajectory segment it takes the union
of the sampled bounds, including the final waypoint on the last segment during
validation. A pair with sufficient clearance between these bounds passes for
every sample in the segment. Remaining pairs get per-sample AABB checks.
Halfspaces always reach narrow phase. SQP linearization uses a 50 mm buffer
above each clearance to select exact distances for derivatives. Trial steps
recompute candidates. `collision_distances()` queries every enabled pair.

Validation checks the convex model before the authored geometry. Each trajectory
remembers the last failing sample and pair for each model. A fresh distance at
that sample can reject the next trial immediately. If the pair clears or the
sampling grid changes, the planner screens the full grid again.

With `sqp_fast`, each trajectory can finish as soon as its seed or an accepted
step passes the waypoint, motion-limit and collision checks. The first Coal or
Warp attempt starts on the verification grid. For each segment and shape pair
it linearizes the closest sampled configuration within the check distance.
Trial steps recompute the closest sample. A 0.1 mm extra optimization margin
helps clear the requested threshold. If needed, the planner retries with all
nearby sample rows and the requested margins. The standard `sqp` solver
continues minimizing the velocity and acceleration costs after finding a path.

On a Ryzen 7 3700X and RTX 4070 with one Torch CPU thread, median planner times
for 350 shape pairs were as follows. The fast mode uses three warm repetitions,
and the earlier comparisons use two.

| Trajectories | Backend | Before AABB screening | Per-sample screening | Segment screening (`sqp`) | `sqp_fast` | Verified |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Coal | 32.28 s | 12.64 s | 5.21 s | 0.95 s | 1/1 |
| 1 | MuJoCo Warp | 31.26 s | 12.43 s | 5.47 s | 1.43 s | 1/1 |
| 4 | Coal | 131.63 s | 51.47 s | 20.65 s | 2.32 s | 4/4 |
| 4 | MuJoCo Warp | 124.17 s | 49.76 s | 20.70 s | 2.86 s | 4/4 |

The one-trajectory standard Coal solve linearized 15 times at 12 samples per segment
and 10 times on the retry at 24 samples per segment. Across both grids,
41,433 of 1,911,000 sample-pairs needed exact distances at linearization.
25,461 rows needed collision gradients. In the four-trajectory benchmark
the largest public distance discrepancy across 1,252 configurations was
0.058 mm. Both backends achieved 14.00 mm minimum sampled carton clearance.

On the identical single task, `sqp_fast` linearized 854 collision rows and
solved 25 QPs, compared with 56 QPs for `sqp`. The joint smoothness objective
was 0.36435 for the first feasible path and 0.36322 after standard optimization.

In a separate single-trajectory Coal profile, 5 of 27 validation calls reached
the authored model. Authored AABB comparisons fell from 2,957,850 to 30,344.
Convex validation used another 74,066 comparisons. These counts cover the
internal trajectory validation calls only.

The benchmark also times a straight, 11.7 cm move from the drop configuration.
This path passes its first validation and requires zero QP steps. In an
11-run warm Coal measurement the median planner call took 5.09 ms, compared
with 8.65 ms with per-sample screening and 120 ms before screening. The long
container transfers in the table require iterative optimization.

## Requirements and assets

Use a Coal-enabled Gobot build and the `planning-sqp` Python extra for OSQP. The Panda example documents the build command and system dependencies. [Build instructions](../panda_trajopt/README.md#build)

The workcell dimensions and tooling are authored for this example. The FANUC M-10iA represents the six-axis industrial arm in the unloading footage. The seven STL meshes and joint frames come from ROS-Industrial's FANUC support package at revision `d8f42bd73584b255df87098395512538882caea1`. [Model source](https://github.com/ros-industrial/fanuc/tree/d8f42bd73584b255df87098395512538882caea1/fanuc_m10ia_support) · [BSD license](meshes/LICENSE)
