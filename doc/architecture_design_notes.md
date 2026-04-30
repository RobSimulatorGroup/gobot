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

## Rendering Pipeline Rewrite Plan

The renderer should treat authored mesh visuals and collision/debug visualization as separate products of the scene, not as one shared rendering path with different colors. This keeps robot asset rendering, physics debugging, editor tools, and later Python APIs from depending on backend-specific details.

The intended responsibility split is:

1. `Scene` owns semantic nodes and resources:
   - `MeshInstance3D` represents authored visual geometry.
   - `Material3D` and derived resources describe stable material properties.
   - `CollisionObject3D` / `CollisionShape3D` represent collision semantics.
   - Robot links, joints, limits, and transforms remain normal scene data.
2. `RenderBackend` owns GPU resources and draw execution:
   - `MeshStorage` uploads, updates, and frees mesh buffers.
   - `MaterialStorage` maps stable material resources to shader parameters.
   - `RendererSceneRender` draws normal visual geometry.
   - `RendererDebugDraw` draws grid, axes, collision shapes, joint frames, AABBs, contacts, trajectories, and sensor frustums.
3. `Editor` owns visualization policy:
   - whether visuals, collisions, joints, or debug overlays are visible.
   - whether collision is shown as wireframe, high-transparency solid, or selected-only.
   - selection outlines, gizmos, viewport navigation, and tool overlays.

The target pass structure is:

1. `SceneOpaquePass`: opaque authored mesh visuals.
2. `SceneTransparentPass`: transparent authored mesh visuals, with sorting added when needed.
3. `DebugGridPass`: viewport grid and world axes.
4. `CollisionDebugPass`: collision wireframes and optional transparent simplified shapes.
5. `RobotDebugPass`: joint axes, limit arcs, link frames, and articulation diagnostics.
6. `EditorOverlayPass`: selection highlight, gizmos, and editor-only handles.

Mesh rendering should move toward a material model suitable for robot assets and imported scenes:

- First implement a simple `BlinnPhongMaterial3D` or unlit/lit preview path so existing assets render predictably.
- Add `StandardMaterial3D` as the Disney/Principled PBR-facing material with base color, metallic, roughness, normal, emissive, and opacity properties.
- Keep the first Disney PBR implementation to the common real-time subset: GGX distribution, Smith visibility, Schlick Fresnel, Burley or Lambert diffuse, and glTF/USD-friendly metallic-roughness inputs.
- Keep material resources stable and backend-agnostic. Do not expose OpenGL programs, texture handles, or renderer IDs through scene or Python-facing APIs.
- Add lighting and environment support in small steps: directional light, point/spot lights, image-based lighting, then shadows.

Collision rendering should remain a debug tool, even when it is visually solid:

- Generate collision debug primitives from collision shape resources instead of treating collision shapes as normal mesh visuals.
- Default to line rendering with clear per-shape colors.
- Support optional transparent solid overlays for simplified inspection.
- Keep collision opacity, selected-only display, and visibility toggles as editor/debug draw policy.
- Rebuild cached debug geometry only when a collision shape changes; transform changes should update instance transforms only.

The implementation order should be:

1. Define the intermediate render item lists produced from scene traversal:
   - visual mesh items.
   - collision debug items.
   - robot/editor debug items.
2. Normalize visual rendering around `MeshInstance3D`, mesh resources, transforms, and material resources.
3. Add `Material3D` storage and a simple lit material path before expanding to PBR.
4. Extend `RendererDebugDraw` with explicit collision and robotics helpers such as `DrawBoxWireframe`, `DrawSphereWireframe`, `DrawCapsuleWireframe`, `DrawAABB`, `DrawFrame`, and `DrawArc`.
5. Add editor viewport toggles for visuals, collisions, joints, and selected-only debug display.
6. Split rendering into explicit pass functions, even if the first OpenGL implementation still shares some internal state.
7. Add tests or focused sample scenes for URDF visual/collision separation, transform correctness, and material fallback behavior.

The first milestone should be conservative: keep the current scene visible, render visual meshes through the normal scene pass, render collisions through debug draw as wireframes, and expose editor toggles for collision and joint overlays. PBR, transparent sorting, shadows, and advanced robot diagnostics can build on that separation.

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
