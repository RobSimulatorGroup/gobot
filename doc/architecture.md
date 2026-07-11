# Gobot Architecture

This document is the authoritative description of Gobot's engine boundaries.
Roadmap documents may describe future work, but they must not define a second
runtime path that conflicts with this one.

## Ownership

- Scene nodes and resources own authored data.
- `PhysicsSceneCompiler` reads a scene and produces `CompiledPhysicsScene`.
- `PhysicsWorld` owns backend runtime state and consumes only
  `PhysicsSceneSnapshot`; it does not traverse or retain scene nodes.
- `SimulationServer` owns the scene bindings needed to synchronize runtime
  state into a Play Mode scene. Backends never write scene nodes directly.
- Renderers consume scene/render items. They do not own scene data or editor
  policy.
- The editor composes scene, simulation, rendering, commands, and tools.
- Python bindings expose these services without exposing backend pointers,
  native ids, OpenGL handles, or editor widget internals.

## Physics Pipeline

The only supported runtime path is:

```text
.jscn / live authored scene
  -> PhysicsSceneCompiler
  -> PhysicsSceneSnapshot + SceneBindings
  -> PhysicsWorld::Build(snapshot)
  -> backend runtime model and state
```

`Robot3D.source_path` records import provenance. It is not copied into the
physics snapshot and is never opened by a physics backend. Importing URDF or
MJCF must first produce normal Gobot nodes and resources.

Physics snapshots and states contain values and stable Gobot names, not scene
node pointers. `SceneBindings` stays above `PhysicsWorld` and is used only when
a runtime scene needs visual synchronization.

The compiler reports structured diagnostics and rejects duplicate runtime robot,
link, joint, and sensor names. Different Scene paths do not create different
backend namespaces. `test_architecture_dependencies` prevents physics runtime
code from including Scene headers, reopening `source_path`, or restoring a
`BuildFromScene` backend entry point.

## Batch Simulation

Batch APIs are backend-neutral and always include an environment dimension.
Names are resolved when a batch view is created; step/reset paths operate on
typed contiguous buffers. MuJoCo CPU is the semantic reference backend.

`PhysicsWorld::StepRobotBatch()` accepts `PhysicsRobotBatchStepRequest` and
returns `PhysicsRobotBatchStepResult`. The request describes controls, named
link dynamics overrides, external wrenches, sensors, and contact-history needs
without backend ids. MuJoCo owns its worker pool and all raw model/data access.

Python task code may consume NumPy views for state, actions, resets, commands,
and task buffers. It must not read `mjModel`, `mjData`, MuJoCo ids, or per-node
runtime dictionaries in the hot path.

## Policy Contract

Training checkpoints, exported ONNX models, and Play Mode use one
`gobot.rl.PolicyManifest`. The manifest records the task version, full
observation and action specs, joint order, physics step, decimation, control
parameters, model description, and recursive simulation-scene resource digest.
The digest canonicalizes `.jscn` JSON and excludes script, import provenance,
visibility, and debug-visualization properties that cannot affect simulation.

New policy loaders must reject missing or mismatched manifests. Observation
dimension guessing is not a compatibility contract. ONNX stores the manifest
under `gobot.policy_manifest` and also gets a `.manifest.json` sidecar for
inspection without a model runtime.

## Runtime Scenes

An edited scene and a runtime scene are different owners. Play Mode packs the
edited scene, instantiates a runtime clone, compiles physics from that clone,
and destroys the clone on Stop. Headless training compiles authored scene data
without creating editor nodes for each environment.

Scene lifecycle, physics stepping, and script notifications are composed by an
application/session layer. `SceneTree` must not select physics backends or own a
global simulation world.

## Dependency Direction

The intended build direction is:

```text
core
  <- scene
  <- scene_io
  <- physics_api <- physics_backends
  <- simulation
  <- rendering
  <- app
  <- python / editor
```

Temporary violations still being removed are Scene resources allocating GPU
RIDs, `SceneTree` driving the simulation singleton, rendering code including
editor code, and locomotion task-buffer ownership living in a pybind
translation unit instead of a simulation service.
New code must not add more dependencies in those directions.

## Current Migration Order

1. Move locomotion command/history/buffer ownership into a bindable simulation
   service on top of the completed typed physics batch contract.
2. Represent procedural terrain configuration as a small versioned scene
   resource; do not check generated height arrays into projects.
3. Split CMake targets so invalid dependency directions fail at link time.
4. Remove global active-context/singleton requirements from headless workflows.
