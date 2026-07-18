# MJCF Equivalence Notes

This page explains the current Gobot goal for MJCF import and MuJoCo runtime
execution. The short version is:

- `.jscn` is the editable source of truth after import.
- `Robot3D.source_path` is import provenance and is never a simulation input.
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

Go1 must stand even if its `source_path` is changed to a missing file. The
physics snapshot intentionally contains no import source path, proving the
authored Gobot data is sufficient to build the MuJoCo model.

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
4. Confirm changing `Robot3D.source_path` does not change the compiled model.
5. For interactive walking, export the current manifest-backed policy to
   `res://policies/go1_velocity.onnx`; `GOBOT_GO1_POLICY` can override that
   path. The optional forward-running actor uses
   `res://policies/go1_velocity_run.onnx` or `GOBOT_GO1_RUN_POLICY`.
6. If physics height/contact are correct but debug shapes look below the floor,
   suspect scene-to-viewport synchronization or debug shape drawing, not MuJoCo
   contact generation.

The Go1 task defines a `0.278m` base clearance relative to the selected terrain
spawn origin. The world-space reset `z` can therefore be negative on pits or
positive on raised patches; it must not be interpreted as a flat-ground
constant.

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
the runtime pose. `Shift+W` selects the optional run actor and a `3.0 m/s`
command; `Shift+S` keeps the balanced actor and raises reverse command magnitude
to `1.5 m/s`. The base policy is loaded from
`res://policies/go1_velocity.onnx`, its `.pt` fallback, or
`GOBOT_GO1_POLICY`. The run actor uses the matching `_run` paths or
`GOBOT_GO1_RUN_POLICY`. Missing, legacy, or contract-mismatched base policies
fail at startup; an absent run actor falls back to the base actor.

The run actor is trained separately from the reference-aligned balanced actor.
Its run-only rewards reduce diagonal trot support and pair the front/rear leg
motions, but the current rough-terrain policy is not a strict phase-locked
bound. Use the evaluator's contact, action, foot-speed, and foot-height pair
metrics rather than inferring gait only from commanded speed.

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
