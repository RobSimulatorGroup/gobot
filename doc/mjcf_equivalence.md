# MJCF Equivalence Notes

This page explains the current Gobot goal for MJCF import and MuJoCo runtime
execution. The short version is:

- `.jscn` is the editable source of truth after import.
- `Robot3D.source_path` is provenance or fallback data, not the normal Go1
  simulation path.
- MuJoCo body state is the runtime source of truth after stepping.
- The editor viewport must display the runtime body state, not a second pose
  reconstructed by editor-only transforms.

## What "Equivalent" Means Here

Equivalent does not mean Gobot stores every private MuJoCo XML field. It means
common robot MJCF files keep the physical semantics that decide whether the
robot stands, contacts the floor, and responds to commands:

- link hierarchy and joint axes
- joint limits and initial/reset positions
- capsule, box, sphere, cylinder, and mesh collision shapes
- MJCF `fromto` placement for capsule/cylinder geoms
- geom friction, contact type, contact affinity, contact dimension, `solref`,
  `solimp`, `margin`, and `gap`
- position, velocity, and motor actuator intent, including gains, control
  ranges, force ranges, and gear values
- inertial properties needed by the backend

That is different from the earlier shortcut where the MuJoCo backend could use
`Robot3D.source_path` and load the original XML directly. Loading the original
XML can hide importer bugs because the simulation never actually uses the
imported Gobot scene data.

## Import Path

The intended path is:

```text
MJCF XML -> Gobot nodes/resources in .jscn -> MuJoCo runtime spec/model
```

After import, MJCF data should be represented as normal Gobot scene objects:

- `Robot3D` owns the robot node.
- `Link3D` owns link inertial data and visual/collision children.
- `Joint3D` owns joint axis, limits, initial position, and drive config.
- `CollisionShape3D` owns contact and friction parameters.
- shape resources, including `CapsuleShape3D`, preserve geometry.

For Go1, `examples/go1/go1.jscn` intentionally has an empty `source_path`.
If Go1 still stands, it proves the authored Gobot data is good enough to build
the MuJoCo model.

## Runtime Path

The runtime path is:

```text
.jscn scene data -> PhysicsSceneSnapshot -> MuJoCo model -> PhysicsSceneState
```

After `mj_step`, MuJoCo owns the authoritative body poses and joint state.
Gobot copies that state back into the runtime scene so the editor viewport,
debug collision lines, Python state queries, and physics panel agree.

This matters because a robot can be physically correct while the viewport still
looks wrong. That was the Go1 foot/calf issue: MuJoCo contacts and base height
matched the reference model, but some editor debug geometry was still following
Gobot-side scene transforms. The fix is to sync non-base link transforms from
the backend body poses during `SimulationServer::ApplyWorldStateToScene()`.

## Go1 Debug Checklist

When Go1 appears to sink through the floor, check these in order:

1. In the Physics panel, confirm `MuJoCo CPU` is selected and the world is
   built.
2. Check the joint table: position control targets should be near the stand
   targets, for example thigh around `0.9` and calf around `-1.8`.
3. Check base height after reset and a short run. A healthy default stand smoke
   test is around `z=0.26`; values near `0.05` mean the robot has collapsed.
4. Confirm `examples/go1/go1.jscn` has `source_path: ""`.
5. For stand-only debugging, run with `GOBOT_GO1_POLICY=""`. For interactive
   walking, the example tries `res://policies/go1.pt` by default and
   `GOBOT_GO1_POLICY` can override that path.
6. If physics height/contact are correct but debug shapes look below the floor,
   suspect scene-to-viewport synchronization or debug shape drawing, not MuJoCo
   contact generation.

The commonly used Go1 standing qpos with base `z=0.27` starts with the foot
spheres about `1.8cm` inside the ground in MuJoCo itself. MuJoCo resolves that
penetration after stepping, but it looks wrong in an editor viewport. The Gobot
Go1 example therefore resets the base to `z=0.288` for the same joint pose, so
the scene starts without visible foot penetration while still settling to the
same standing configuration.

## Go1 Keyboard Control

Play Mode keyboard input is routed through Gobot's engine `Input` service
instead of direct SDL or ImGui callbacks in `examples/go1/scripts/go1.py`.
The engine API only exposes key state and viewport control focus; Go1's WASD
mapping lives in the example script because it is controller configuration, not
a global engine action map. Click the 3D Viewer while Play Mode is running to
give the runtime scene keyboard control; press `Esc` to release it. Text
fields, popups, and other editor UI should not drive the robot.

The Go1 script polls `context.input` during `_physics_process()`. `W/S` control
forward speed, `Q/E` strafe, `A/D` yaw, `Space` commands stop, and `R` resets
the runtime pose. These keys drive the policy command velocity when the default
`res://policies/go1.pt` policy, or a `GOBOT_GO1_POLICY` override, is loaded.
Without a policy the example keeps the standing PD target and does not fake
walking.

## Regression Tests

The main guards are:

- `TestResourceFormatMJCF.imports_capsule_contact_and_position_actuator_semantics`
  checks MJCF capsule/contact/actuator/keyframe data is imported into Gobot
  nodes and resources.
- `TestSimulationServer.mujoco_authored_position_actuator_respects_imported_limits`
  checks authored `.jscn` actuator and joint limit data generate a useful
  MuJoCo model.
- `TestSimulationServer.syncs_backend_link_pose_to_non_base_link_scene_transform`
  checks stepped backend link poses are copied back into the scene for viewport
  and collision debug use.
- `tests/python/smoke_real_scene.py --expect-go1-stand --expect-empty-robot-source-path`
  checks the Go1 example builds from `.jscn`, does not auto-load a local policy,
  and remains above a minimum standing height.
