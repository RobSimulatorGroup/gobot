# Gobot Architecture Notes

This document records the current design direction from the editor/rendering cleanup work.

## Long-Term Goal

Gobot should grow into a robot editing, simulation, and rendering engine in the same broad category as Isaac Sim. The engine should support robotics workflows such as reinforcement learning, motion planning, collision detection, robot asset editing, and simulation/debug visualization.

The top-level user interaction layer should include Python bindings. Python should relate to Gobot in a similar way that GDScript relates to Godot: users can create scenes, inspect and modify nodes, run simulation steps, invoke robotics algorithms, and drive the editor/engine without reaching into backend-specific rendering code.

## Boundary Rules

The core dependency direction should be:

1. `Scene` owns user data and runtime node state.
2. `RenderBackend` consumes scene data and renders it.
3. `Editor` coordinates tools, selection, debug visualization, and viewport presentation.
4. `PythonBinding` should sit above the engine API and should not depend on OpenGL, ImGui internals, or editor-only tree layout.

Python-facing APIs should target stable engine concepts such as `Scene`, `Node`, `Node3D`, `Mesh`, simulation systems, and commands. They should not expose OpenGL RIDs or editor panel implementation details except through intentional high-level wrappers.

## Current Refactor Priorities

1. `MeshStorage` should stay a low-level mesh allocation/upload/free boundary.
   It should not parse asset files, traverse scenes, or know about editor selection.
   The current transitional API includes `MeshAllocate`, `MeshInitialize`, `MeshSetBox`, `MeshSetSurface`, primitive setters, `OwnsMesh`, and `MeshFree`.
2. `RendererSceneRender` owns the scene render pass:
   `RenderScene(root, camera, target)`.
3. `RendererDebugDraw` stays independent for editor and simulation debug rendering. Future operations should include `DrawAABB`, `DrawLine`, `DrawFrustum`, contact points, trajectories, and robot joint/debug overlays.
4. `EditorViewportRenderer` is the editor-layer coordinator. It composes scene pass, debug pass, and later overlay/pass tools without moving editor decisions into `RenderBackend`.
5. The editor keeps a fixed edited-scene boundary. SceneTree should display the user scene root, not the full engine tree with editor UI nodes.

## URDF Import Direction

URDF import should create normal scene nodes, not store robot structure as opaque strings on one node.

Current scene shape:

- `Robot3D` is the imported robot root and stores the source URDF path.
- `Link3D` is a robot link container and owns inertial metadata such as mass, center of mass, and inertia.
- Visual geometry is represented by `MeshInstance3D` children under the owning `Link3D`.
- Collision geometry is represented by `CollisionShape3D` children under the owning `Link3D`.
- `Joint3D` keeps joint metadata and connects the parent link to the child link subtree.

This matches the intended robotics model: `Link3D` carries robot semantics, while rendering and collision are authored through explicit child nodes/resources that the editor, renderer, physics, and later Python APIs can all inspect.

Mesh asset import is optional through Assimp. Gobot should prefer the `3rdparty/assimp` git submodule when `GOB_BUILD_ASSIMP` is enabled, and may fall back to a system `assimp` package if the submodule is unavailable. The default submodule build is import-only and enables only common robot asset formats: `COLLADA`, `OBJ`, `STL`, `PLY`, and `GLTF`. Add more formats through `GOB_ASSIMP_IMPORTERS` when there is a concrete need.

When Assimp and a `RenderServer` are available, URDF visual mesh references can load into `ArrayMesh`. Without Assimp or render initialization, the importer should keep a `Mesh` resource path placeholder so scene parsing and tests remain usable without a rendering backend.

## Python Binding Direction

The Python layer should be added after the C++ API boundaries are stable enough to bind cleanly.

Recommended shape:

- `gobot.Engine` or `gobot.App`: lifecycle, project load, simulation step.
- `gobot.Scene`: create/load/save scene, access root, query nodes.
- `gobot.Node` / `gobot.Node3D`: transforms, hierarchy, components/properties.
- `gobot.render`: high-level viewport/render commands, no backend-specific objects.
- `gobot.sim`: physics, collision queries, robot articulation, controllers.
- `gobot.rl` or integration hooks: reset/step/observe/reward adapters.

The binding should call stable C++ services. It should not duplicate editor logic or traverse the editor's internal node tree.

## Open Design Questions

- Whether the simulation world should be a scene subsystem, a separate `SimulationServer`, or both.
- Whether robotics algorithms live as engine plugins, Python packages, or C++ services with Python wrappers.
- How to represent robot articulations, sensors, and collision geometry in the scene model.
- How command/undo should work across C++, editor UI, and Python.
