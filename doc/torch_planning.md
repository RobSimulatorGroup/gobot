# Torch motion planning

`gobot.planning` implements PyRoki's core IK and trajectory optimization
workflow with PyTorch. It reads the same physics scene snapshot as the
simulation providers and uses Gobot's differentiable forward kinematics.

```python
import gobot
from gobot.planning import Planner, PoseTarget

context = gobot.app.context()
context.set_project_path("/path/to/project")
context.load_scene("res://world.jscn")
planner = Planner(context.compile_physics_scene_snapshot(), {"Robot": "base"})

start = planner.initial_positions  # Supply measured joint positions when available.
target = PoseTarget(("Robot", "tool"), position=(0.45, 0.10, 0.35))
ik = planner.solve_ik([target], seed=start)
if not ik.success:
    raise RuntimeError(ik.reason)
plan = planner.solve_trajopt(start, ik.positions, dt=0.10, waypoints=16)
if not plan.success:
    raise RuntimeError(plan.reason)
print(planner.joint_names, plan.positions)
```

`PoseTarget` accepts a world position in metres, a world rotation as a
proper 3×3 matrix, or both. Pass multiple targets to `solve_ik` for
simultaneous IK. `plan_poses` runs IK for both endpoint target sets and then
optimizes the joint trajectory.

Select the base of each robot to plan jointly across robots. A robot with
independently rooted arms uses a sequence of bases, for example
`{"Robot": ("left_base", "right_base")}`. Bases are fixed at their snapshot
poses, and unselected links contribute static obstacles in the snapshot.
Use `planner.joint_names` to order positions and velocities. Revolute joint
coordinates are in radians, prismatic coordinates are in metres, and `dt`
is in seconds.

## Batched IK

`PlannerBatch(planner)` solves independent IK problems on one shared model.
Give a target position shape `(B, 3)` or a rotation shape `(B, 3, 3)` to run
`B` solves. A single position, rotation, link name or seed is shared across
the batch. You can also supply a sequence of link names or a seed of shape
`(B, joints)`. Multiple `PoseTarget` entries constrain multiple links in
each solve.

```python
from gobot.planning import PlannerBatch, PoseTarget

targets = PoseTarget(("Robot", "tool"), position=[[0.45, 0.10, 0.35], [0.40, 0.05, 0.35]])
result = PlannerBatch(planner).solve_ik([targets], seed=start)
solutions = result.positions[result.success]
```

Use `PlannerBatch([planner_a, planner_b])` for row-specific model frames,
joint limits or collision geometry. The planners must share link and joint
topology, joint order, active shape pairs, dtype and device. The boolean
`success` mask identifies feasible rows. Failed rows contain NaN.

## Collision geometry

- Sphere and capsule distances use the authored primitives. World boxes and
  planes use their authored frames. Cylinders and robot boxes use enclosing
  capsules. Meshes use their bounding boxes in the shape frame with enclosing
  capsules on robot links.
- `link_capsules={("Robot", "tool"): [Capsule(a, b, radius)]}` replaces a
  selected link's geometry with explicit proxies in link-local coordinates.
  Use equal endpoints for spheres. An empty list omits that link's shapes.
- Collision filtering uses shape layers and masks. Shapes on the same link
  or on directly joined links are skipped. Use `ignored_link_pairs` for
  additional link pairs that are allowed to touch.
- `planner.collision_distances(q)` returns differentiable clearances in metres.
  `planner.collision_pairs` labels those distances, including repeated link
  pairs when they have multiple shapes. Negative distances indicate proxy
  overlap. Deformable bodies and terrain heightfields are unsupported.

## Solving and verification

For PyRoki-style pose-to-pose planning, pass the two sets of world-space targets
to the planner. Each set may include both arms.

```python
plan = planner.plan_poses(start_targets, goal_targets, waypoints=50, dt=0.02,
                          start_seed=current_joint_positions)
if not plan.success:
    raise RuntimeError(plan.reason)
q = plan.positions  # (50, number of joints), in planner.joint_names order
```

`plan.initial_positions` contains the actual joint waypoints at the start of
trajectory optimization, including the fixed endpoints. It is available once
the seed has been constructed. The [Panda editor example](../examples/panda_trajopt/README.md)
plays this seed and the optimized poses on the same clock.

The endpoint IK objective favors nearby joint configurations for the two
poses. The trajectory solver biases joints toward their authored rest pose
and penalizes adjacent joint differences, self-collision proximity and
five-point central accelerations scaled by `1 / dt²`. The acceleration term
applies to paths of at least five waypoints. Projected Levenberg–Marquardt
and an augmented Lagrangian enforce joint bounds, velocity limits and
clearance. Each segment contributes local Jacobian blocks.

With four or more waypoints, `plan_poses` holds the first two and last two
configurations equal for zero endpoint segment velocity. Joint-to-joint
`solve_trajopt` enables this behavior with `stop_at_endpoints=True`.
Pose planning defaults to 0.02 m self clearance and 0.05 m world clearance.
An explicit `margin` applies to both, and `world_margin` overrides it for
obstacles.

For world clearance, the planner connects each proxy sphere's positions at
adjacent waypoints with a swept capsule. A robot capsule supplies five sphere
centers along its axis. These chords approximate the motion of rotating
links. The planner also checks the original proxies at interpolated joint
configurations, including self and inter-arm pairs. Final verification uses
twice the requested `collision_substeps` per segment. `validate_trajectory`
accepts the same margins and a sample density for an independent check. Clearance
checks allow up to one micrometre of numerical tolerance while requiring
nonnegative distance. Velocity checks use a relative tolerance of 1e-6.

The optimizer is local. `solve_trajopt` accepts a waypoint seed, and
`solve_ik` accepts a seed configuration. An unsuccessful result carries
a reason and has `positions=None`. The default device and type are CPU and
float64. The planner accepts an explicit Torch device and float32.

Based on the [PyRoki trajectory optimization example](https://github.com/chungmin99/pyroki/blob/388e43e1fc0d0ee382968d3dd72970fd62a0450c/examples/pyroki_snippets/_trajopt.py).
The upstream MIT notice is included with the [planning package](../python/gobot/planning/LICENSE-PyRoki.txt).
