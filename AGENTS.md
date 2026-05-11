# Gobot Agent Guide

This file is the root instruction document for future coding agents working in this repository.

Read this before making changes. More specific `AGENTS.md` files in subdirectories may add local rules.

## Product Direction

Gobot is intended to become an open-source robotics editing, simulation, and rendering engine in the same broad category as Isaac Sim.

The long-term engine should support:

- robot asset editing and scene composition
- real-time viewport rendering and editor debug visualization
- physics simulation, collision detection, sensors, and robot articulation
- reinforcement learning and motion-planning workflows
- import/export of robotics scene formats, especially OpenUSD
- Python bindings as the top-level user scripting API

Python should relate to Gobot in a similar way that GDScript relates to Godot: users should be able to create scenes, inspect and modify nodes, run simulation steps, invoke robotics algorithms, and drive high-level engine/editor workflows without touching backend-specific rendering or ImGui internals.

## Architecture Boundaries

Keep the dependency direction clean:

1. `Scene` owns user data and runtime node state.
2. `RenderBackend` consumes scene data and renders it.
3. `Editor` coordinates tools, selection, commands, debug visualization, and viewport presentation.
4. `PythonBinding` sits above stable engine APIs and must not depend on OpenGL, ImGui internals, or editor-only tree layout.

Do not let backend implementation details leak upward into scene or Python APIs. Do not make the editor tree represent the full internal engine tree unless a debugging tool explicitly needs that.

Current rendering/editor boundary shape:

- `MeshStorage` should stay a low-level mesh allocation/upload/free boundary.
  It must not parse asset files, traverse scene nodes, or know about editor selection.
  Current transitional APIs include `MeshAllocate`, `MeshInitialize`, `MeshSetBox`, `MeshSetSurface`, primitive setters, `OwnsMesh`, and `MeshFree`.
- `RendererSceneRender` owns the scene render pass:
  `RenderScene(render_target, scene_root, camera)`.
- `RendererDebugDraw` stays independent for editor and simulation debug rendering.
  Future operations should include `DrawAABB`, `DrawLine`, `DrawFrustum`, contact points, trajectories, joint frames, and sensor frustums.
- `EditorViewportRenderer` is the editor-layer coordinator.
  It composes scene pass, debug pass, and overlay tools.
- The editor owns an edited-scene boundary.
  `SceneTree` should show the user scene root, not editor UI nodes, dock nodes, render targets, or engine service objects.

## Python Binding Direction

Add Python bindings only on top of stable C++ services. Avoid binding temporary editor/backend implementation details.

Recommended module shape:

- `gobot.Engine` or `gobot.App`: lifecycle, project load, simulation step.
- `gobot.Scene`: create/load/save scene, access root, query nodes.
- `gobot.Node` / `gobot.Node3D`: transforms, hierarchy, reflected properties.
- `gobot.render`: high-level viewport/render commands, no backend-specific OpenGL objects.
- `gobot.sim`: physics stepping, collision queries, articulations, controllers, sensors.
- `gobot.rl`: reset/step/observe/reward adapters or integration hooks.

Python APIs should call the same engine services used by C++ and editor code. They should not duplicate editor traversal logic.

## Current Priority Order

Work in this order unless the user explicitly redirects:

1. Tighten `Scene / Editor / RenderBackend` boundaries.
2. Make scene serialization reliable with `PackedScene`, `SceneState`, and `.jscn`.
3. Add OpenUSD as an optional dependency and import path.
4. Stabilize mesh/material data models for PBR rendering.
5. Add physics/simulation abstractions before binding a specific SDK such as NVIDIA Newton.
6. Add Python bindings after the C++ service boundaries are stable enough to bind cleanly.

Prefer changes that compile, test, and leave a clean boundary. Avoid broad refactors that cross these layers without a concrete reason.

## Scene Serialization Plan

Goal: make `PackedScene`, `SceneState`, and `.jscn` a reliable base for editor save/load, OpenUSD import/export, and simulation scene exchange.

Known issues:

- `PackedScene` / `SceneState` are still thin; `NodeData` and `__NODES__` load/save need real implementation.
- `ResourceFormatSaverSceneInstance::Save()` and `LoadResource()` mix resource serialization and scene serialization concerns.
- `VariantSerializer` currently handles too many responsibilities: type conversion, resource references, and property traversal.
- `Node` lacks a clear serialization interface for hierarchy and reflected storage properties.
- Check `FindResources()` compound-type handling before relying on it; there has been a suspected inverted condition.

Target `SceneState` data shape:

```cpp
SceneState::PropertyData {
    std::string name;
    Variant value;
};

SceneState::NodeData {
    std::string type;       // reflected class name, for example "Node3D"
    std::string name;
    int parent;             // index into nodes_ array, -1 means root
    Ref<PackedScene> instance;
    std::vector<PropertyData> properties;
};
```

Required operations:

- `PackedScene::Pack(Node* root)`: capture node type, name, parent, instance reference, and storage properties.
- `PackedScene::Instantiate()`: recreate nodes by reflected type, restore properties, and rebuild parent-child relationships.

Implementation phases:

1. Define `SceneState` structures and accessors.
2. Verify reflected storage properties for `Node` subclasses.
3. Implement `Pack` and `Instantiate`.
4. Refactor `.jscn` resource loading/saving around the real `SceneState`.
5. Connect editor save/load and add round-trip tests.

Suggested `.jscn` version 2 shape:

```json
{
  "__VERSION__": 2,
  "__META_TYPE__": "SCENE",
  "__TYPE__": "PackedScene",
  "__EXT_RESOURCES__": [],
  "__SUB_RESOURCES__": [],
  "__NODES__": [
    {
      "type": "Node3D",
      "name": "RobotRoot",
      "parent": -1,
      "properties": {
        "position": [0, 0, 0]
      }
    }
  ]
}
```

## OpenUSD Plan

OpenUSD support should be optional. The project must still build, test, and run when OpenUSD is not installed.

Goals:

- Import `.usd`, `.usda`, `.usdc` into `PackedScene`.
- Export `PackedScene` to USD after the core scene representation is stable.
- Support hierarchy, transforms, mesh geometry, materials, and later USD Physics schemas.
- Do not leak USD-specific concepts into the general scene API.

Phases:

1. Add a `GOB_BUILD_OPENUSD` CMake option, `GOBOT_HAS_OPENUSD` define, and `ResourceFormatLoaderUSD` skeleton.
2. Import USD hierarchy: `UsdGeomXformable` to `Node3D`, transform ops to `Node3D` transform properties.
3. Add a real storable mesh resource, such as `ArrayMesh` or `SurfaceMesh`.
4. Map USD mesh data: points, normals, UVs, face indices, and triangulation.
5. Map `UsdShadePreviewSurface` into `PBRMaterial3D`.
6. Add export only after import and serialization are solid.

## Rendering And Debug Visualization

Keep normal scene rendering and editor/simulation debug visualization separate.

Scene render pass:

- Draws user scene content.
- Consumes stable scene state and mesh/material resources.
- Should not know about selection, gizmo state, editor panels, or ImGui.

Debug draw pass:

- Draws grid, axes, AABBs, frustums, contact points, trajectories, and robot debug data.
- Can be used by editor and simulation tools.
- Should expose explicit draw commands over time instead of hardcoded editor-only content.

Editor viewport:

- Coordinates scene pass, debug pass, selection wireframes, gizmos, and overlays.
- Owns editor interaction policy such as orbit, pan, focus, and view cube behavior.

## Simulation Direction

Before integrating a heavy physics SDK, define stable engine-facing simulation boundaries.

Likely services:

- `SimulationServer`: world stepping, time control, reset, deterministic mode.
- `PhysicsServer`: bodies, collision shapes, constraints, queries.
- `RobotModel` / articulation support: joints, limits, drives, kinematics.
- Sensor APIs: camera, depth, lidar, contact, force/torque, joint state.
- Debug hooks: draw contact points, collision shapes, joint frames, planned paths.

Robotics algorithms can later live as plugins, Python packages, C++ services with Python wrappers, or a mix of those. Keep engine data ownership clear before choosing.

## Reinforcement Learning Locomotion Plan

Goal: make Gobot able to train and run policies that control robots walking or running inside Gobot scenes. The first target should be a single articulated robot, such as a humanoid or quadruped, running on a flat ground plane with deterministic reset and repeatable stepping. Do not start with a large distributed RL stack; first make one environment reliable.

Current RL direction is documented in `doc/mujoco_rl_plan.md`. Treat that file as the detailed roadmap for MuJoCo CPU and MuJoCo Warp work.

Key direction:

- Focus future RL simulation work on `MuJoCoCpu` as the semantic baseline and `MuJoCoWarp` as the CUDA graph vectorization path.
- Do not reintroduce PhysX as public backend types unless the project direction changes explicitly.
- Gobot `SceneTree` / `.jscn` remains the authoring source of truth. MJCF, `mjModel`, MuJoCo data, and Warp arrays are runtime artifacts compiled below the scene/physics boundary.
- RL training should use `gobot.rl.ManagerBasedEnv` / `VectorEnv` style APIs and normal Python scripts. Gymnasium and rsl_rl belong in compatibility wrappers, not core engine APIs.
- `NodeScript` / `ScenePlaySession` are only for editor Play Mode, single-scene debugging, and policy playback. RL training must not run node scripts or depend on the editor viewport.
- Python Panel `Run Once` is a tool-script action against the active editor context. It must not install per-frame callbacks, enter Play Mode, or participate in vectorized environments.
- Do not add C++-specific environment variable hooks for Python package discovery. Use normal Python packaging, `PYTHONPATH` in tests, or explicit project paths.

Python is the intended top-level RL workflow. Users should train and evaluate policies through the `gobot` Python package and ordinary Python training scripts, while Gobot's C++ layer owns deterministic simulation, robot state extraction, joint controllers, reset, reward, and termination. Do not make a separate `gobot_headless` executable the primary training entry point; keep training orchestration in Python and expose stable C++ environment services through bindings. Do not implement Python bindings first; design the C++ API in a shape that can be bound cleanly, then expose it to Python after the C++ environment API is covered by tests.

Target Python workflow:

```python
import gobot

env = gobot.rl.LocomotionEnv("res://world.jscn", robot="H2")
obs, info = env.reset(seed=1)

for _ in range(1000):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

Core engine requirements:

1. Simulation must expose a deterministic fixed-step API:
   `Reset(seed)`, `Step(dt or fixed_ticks)`, `GetTime()`, `SetPaused()`, and `SetRealtime(false)`.
   RL training should not depend on editor frame rate or viewport rendering.
2. `Robot3D` and the simulation layer need a stable articulation interface:
   enumerate joints by stable name, read joint position/velocity/effort, read base pose and velocity, read contact state, and apply actions.
3. Action APIs should be backend-neutral:
   support target position, target velocity, torque/effort, PD gains, and actuator limits without exposing MuJoCo-specific fields upward.
4. Observation APIs should be explicit and versioned:
   base orientation, base angular/linear velocity, projected gravity, command velocity, joint position/velocity, previous action, foot contact, and optional terrain or sensor data.
5. Reset must restore the scene and physics world consistently:
   robot root transform, joint state, velocities, actuator state, random seed, contact cache, and any scene objects changed during the episode.
6. Termination and reward should be engine-facing services, not ImGui code:
   height/orientation failure checks, timeout, fallen state, forward velocity tracking, energy penalty, action smoothness, foot slip, and collision penalties.
7. Simulation snapshots should separate authoring state from runtime state:
   `.jscn` / `PackedScene` describes the environment; runtime episode state is reset from that scene plus an explicit randomized initial state.
8. Debug visualization should be optional:
   draw contacts, center of mass, support polygon, target velocity, trajectories, and reward components through debug draw commands, not scene nodes.

Control direction:

- Reinforcement learning should not be the only control layer. For locomotion, the normal path should be:
  `policy action -> target joint command -> PID/PD joint controller -> backend actuator command`.
- Provide a backend-neutral joint controller API before exposing control to Python:
  per-joint mode, target position, target velocity, feed-forward torque, `kp`, `ki`, `kd`, effort limit, velocity limit, integral clamp, and action smoothing.
- Start with PD position control for locomotion:
  policy outputs normalized target joint offsets or target joint positions; the controller converts them to actuator commands each physics tick.
- Add PID only where useful:
  integral terms should be optional and clamped because they can destabilize reset-heavy RL episodes.
- Controller gains must be part of robot/environment configuration, not hardcoded in the MuJoCo backend or policy script.
- The controller must expose both commanded and measured state:
  previous action, target position/velocity, tracking error, applied effort, and saturation flags can be used in observations and reward terms.
- Keep torque control available for advanced policies, but make PD/PID joint control the first stable path for robot walking/running.
- Add tests for controller behavior:
  zero error produces bounded output, command saturation works, integral clamp works, reset clears controller state, and repeated fixed-step runs are deterministic.

Recommended C++ service shape:

- `SimulationServer`: owns stepping, reset, deterministic mode, and scene/world synchronization.
- `RobotController`: backend-neutral action input for a robot articulation.
- `JointController`: per-joint PID/PD control, limits, state reset, and command saturation.
- `RobotState`: compact read-only state for base, joints, contacts, and actuators.
- `ManagerBasedEnv` / `VectorizedEnvironment`: `reset(seed) -> observation`, `step(action) -> step result`.
- `ObservationSpec` and `ActionSpec`: names, shapes, units, bounds, and version strings.
- `RewardTerm` / `TerminationCondition`: composable C++ terms with optional Python configuration later.
- `VectorizedEnvironment`: later, owns many independent environments for batched training.

Minimal locomotion environment phases:

1. Build one Python-driven deterministic environment:
   load scene through the Python API, build physics world, reset robot pose, step without rendering, and verify repeated seeds produce identical observations.
2. Add robot action/state plumbing:
   map policy action vectors to named joints, clamp to limits, apply PD or torque control, and read back joint/base/contact state.
3. Add the first PD joint controller:
   configure per-joint gains and limits, convert normalized policy actions into target joint positions, and reset controller state at episode reset.
4. Define a first observation and action spec:
   keep it stable and documented so Python/RL code can depend on it.
5. Implement simple reward and termination:
   alive bonus, forward velocity tracking, upright penalty, energy penalty, action rate penalty, timeout, and fallen termination.
6. Add a C++ smoke test:
   reset an environment, apply zero action and random action, step several ticks, verify finite observations and deterministic replay.
7. Add Python bindings only after the C++ environment API is stable:
   expose `env.reset(seed)`, `env.step(action)`, `env.observation_space`, `env.action_space`, and scene loading.
8. Add compatibility with Gymnasium-style wrappers:
   keep this in Python or a thin adapter layer; do not make core engine APIs depend on Gymnasium.
9. Add vectorized Python training support:
   multiple worlds from Python, no editor windows, optional render capture for evaluation, and deterministic seeding per environment.

Important boundaries:

- RL code must not depend on ImGui panels, editor selection, viewport picking, OpenGL RIDs, or MuJoCo raw pointers.
- Backend-specific details may live below the physics backend interface, but observations/actions must use Gobot names and engine units.
- Do not write policy logic into `Robot3D`; keep robot assets separate from controllers and training tasks.
- Do not make imported URDF/MJCF files the only source of truth for runtime control. Gobot scene data should own the editable robot/environment composition, while physics backends compile the runtime model from that state.
- Prefer Python-level headless tests before editor features. The editor can later inspect, debug, and launch RL environments, but training must work as normal Python code without the editor or a separate headless executable.

## Development Rules

- Prefer existing project patterns and local helper APIs.
- Keep edits scoped to the requested boundary or feature.
- Add tests for behavior that affects serialization, scene traversal, rendering contracts, or editor commands.
- After meaningful C++ changes, build with CMake and run the relevant tests.
- Do not expose OpenGL RIDs, ImGui IDs, or editor panel internals through scene or Python-facing APIs.
- Do not let `###` ImGui internal IDs appear as user-visible names in editor panels.
- Preserve user scene data as the editor-facing source of truth; editor UI nodes are implementation details.
