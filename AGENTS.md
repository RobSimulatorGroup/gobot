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

## Development Rules

- Prefer existing project patterns and local helper APIs.
- Keep edits scoped to the requested boundary or feature.
- Add tests for behavior that affects serialization, scene traversal, rendering contracts, or editor commands.
- After meaningful C++ changes, build with CMake and run the relevant tests.
- Do not expose OpenGL RIDs, ImGui IDs, or editor panel internals through scene or Python-facing APIs.
- Do not let `###` ImGui internal IDs appear as user-visible names in editor panels.
- Preserve user scene data as the editor-facing source of truth; editor UI nodes are implementation details.
