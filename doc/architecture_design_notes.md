# Gobot Architecture Notes

This document records the current design direction from the editor/rendering cleanup work.

For the current MuJoCo CPU, MuJoCo Warp, and `gobot.rl` roadmap, see
`doc/mujoco_rl_plan.md`.

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

## Script System

Script support is Python-only for v1. `NodeScript` is the scene-node script
mechanism and runs only inside `ScenePlaySession` for editor Play Mode,
single-scene debugging, and policy playback. Starting Play Mode packs the
edited scene and instantiates a runtime clone; scripts attach to that clone, the
viewport renders that clone, and physics worlds are built from that clone. Stop
destroys the runtime clone after `_exit_tree`, so script changes do not dirty or
mutate the edited scene.

The editor exposes scene playback through the top menu bar controls. While a
script session is running, physics backend and world-build controls are locked
to avoid replacing the world under script-driven runtime state. Node script
stdout/stderr is captured and routed through the editor logging path so Play
Mode prints appear in the Console.

Python Panel `Run Once` remains a tool-script entry point against the edited
scene and may modify it only through scene commands and undo/redo. The panel
normalizes Python indentation to spaces on save/run, supports `Ctrl+S` for the
script file while focused, and resource `.py` files can be dragged onto
SceneTree nodes to attach them as `NodeScript` resources.

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

## Physics Engine Integration Plan

Gobot should support more than one physics implementation. The engine-facing API must stay stable while CPU and GPU physics backends can be selected, tested, and replaced independently. Robotics workflows should drive the design: articulated robots, joint limits, collision queries, sensors, deterministic stepping, and later reinforcement-learning loops matter more than generic rigid-body demos.

The intended dependency boundary is:

1. `Scene` owns authored robot data:
   - `Robot3D`, `Link3D`, `Joint3D`, `CollisionShape3D`, transforms, limits, inertial data, and material/mesh resources.
   - Scene nodes do not expose MuJoCo, Newton, CUDA, or backend object handles.
2. `PhysicsServer` owns backend selection and backend capability reporting:
   - available backends, CPU/GPU flags, robotics focus, and human-readable status.
   - creation of backend-specific `PhysicsWorld` instances.
3. `PhysicsWorld` owns one runtime simulation world:
   - build from a scene snapshot.
   - reset, step, query, and eventually sync results back to scene nodes.
   - keep backend implementation details below this interface.
4. `SimulationServer` should become the high-level simulation entry point:
   - fixed-step accumulation, pause/play/single-step, reset, time scale, deterministic mode.
   - owns or references the active `PhysicsWorld`.
   - is the API that editor tools and future Python bindings should call.
5. `Editor` only coordinates simulation controls and debug visualization:
   - play/pause, single-step, backend selection UI, collision/joint/contact overlays.
   - it should not call MuJoCo/Newton APIs directly.

Backend direction:

- `NullPhysicsWorld` remains the always-available no-op backend for editor startup, tests, and systems without optional SDKs.
- `MuJoCoCpu` should be the first real backend because it is robotics-focused and strong at articulated bodies, URDF-style robots, contacts, and deterministic CPU simulation.
- `MuJoCoWarp` should be the high-throughput RL backend direction after the CPU semantics are tested. Its design should use persistent buffers, reset masks, and CUDA graph replay.
- `NewtonGpu` should stay reserved for GPU robotics experiments, but it should not shape the public Gobot API before a small prototype proves the needed concepts.
- `RigidIpcCpu` should be reserved as a research/validation backend based on intersection-free rigid body dynamics. Its role is robust contact, tight-fit geometry, and offline verification, not first-pass real-time robot control or RL throughput.
- The backend list should be explicit rather than implicit: users should be able to see which backends are compiled in, available at runtime, CPU/GPU capable, and suitable for robotics.

The scene-to-physics data model should be backend-neutral:

- `PhysicsSceneSnapshot`: robots, loose collision shapes, total object counts.
- `PhysicsRobotSnapshot`: robot source path, links, joints.
- `PhysicsLinkSnapshot`: transform, mass, center of mass, inertia, collision shapes.
- `PhysicsJointSnapshot`: parent/child links, joint type, axis, limits, velocity/effort limits, current joint position.
- `PhysicsShapeSnapshot`: shape type, transform, box/sphere/cylinder/mesh dimensions, disabled state.

Implementation phases:

1. Stabilize the abstraction layer:
   - keep `PhysicsBackendType`, `PhysicsBackendInfo`, `PhysicsWorldSettings`, `PhysicsWorld`, and `PhysicsServer` backend-neutral.
   - add focused tests for backend capability reporting and scene snapshot capture.
2. Add `SimulationServer`:
   - fixed time step, accumulated stepping, pause/play, single-step, reset.
   - build the active physics world from a scene root.
   - expose current simulation time, frame count, active backend, and last error.
3. Add result synchronization:
   - define how simulated body poses and joint states flow back into `Robot3D`/`Joint3D`.
   - avoid overwriting authored assembly transforms while in editor assembly mode.
   - keep robot motion mode and physics simulation mode behavior explicit.
   - preserve compatible simulation state when a physics world is rebuilt after structural edits.
4. Implement the first MuJoCo backend slice:
   - optional `GOB_BUILD_MUJOCO`.
   - prefer a local MuJoCo SDK/package through `GOB_MUJOCO_ROOT` or `CMAKE_PREFIX_PATH`.
   - allow opt-in source fetching through `GOB_FETCH_MUJOCO` and a pinned `GOB_MUJOCO_GIT_TAG`.
   - do not make MuJoCo a default submodule unless Gobot later needs a fully vendored offline dependency set.
   - create a MuJoCo model from the neutral robot snapshot or from the source URDF when that is the most reliable path.
   - map joint limits, axes, inertial data, and primitive collision shapes first.
   - support reset and fixed-step simulation before adding advanced controls.
5. Add physics debug visualization:
   - contacts, collision shapes, link frames, joint frames, center of mass, and broadphase AABBs.
   - use `RendererDebugDraw`; do not route debug drawing through normal mesh rendering.
6. Add queries and controls:
   - ray cast, shape cast, overlap tests, closest points.
   - joint position/velocity/torque state.
   - position/velocity/torque drives for robot control.
7. Add GPU backend prototypes only after the high-level contract is tested:
   - `MuJoCoWarp` for CUDA graph vector simulation.
   - `NewtonGpu` for robotics/GPU articulation experiments if it proves useful.
   - keep CPU/GPU backend differences behind capability flags and shared query/step APIs.
8. Add a Rigid IPC prototype only after MuJoCo stepping and the shared synchronization path are usable:
   - keep it optional and CPU-first.
   - expose it as a robust-contact/offline validation backend.
   - do not require it to support high-throughput RL or full articulated robot controls in the first slice.
   - start from the published implementation or a small isolated solver prototype instead of rewriting the full method inside core engine code immediately.
9. Bind Python only after the C++ simulation service is stable:
   - `gobot.sim.create_world`, `gobot.sim.step`, `gobot.sim.reset`, joint state/control, collision queries.
   - no backend handles in Python unless intentionally exposed as advanced plugin APIs.

Testing requirements:

- Unit tests for backend capability reporting and unavailable optional dependencies.
- Snapshot tests for robot/link/joint/collision data captured from normal scene nodes.
- Round-trip tests that save/load robot scenes without changing initial joint positions.
- Simulation service tests for pause, fixed-step accumulation, reset, and backend switching.
- Backend-specific tests should be skipped cleanly when the optional SDK is not compiled in.

The first useful milestone is not full dynamics. It is a reliable simulation shell: load or import a robot, build a backend-neutral physics snapshot, select a backend, step a world deterministically through `SimulationServer`, and expose enough state for editor visualization and future Python control.

### Scene To MuJoCo Runtime Pipeline

The authoritative data flow should remain scene-first:

1. `Robot3D`, `Link3D`, `Joint3D`, and `CollisionShape3D` live in the Gobot scene tree.
2. `PhysicsWorld::BuildFromScene(scene_root)` captures a backend-neutral `PhysicsSceneSnapshot`.
3. `MuJoCoPhysicsWorld` reads `Robot3D::source_path` and builds a MuJoCo model from URDF/MJCF XML or an equivalent generated `mjSpec`.
4. Link, joint, and actuator bindings map Gobot names to MuJoCo body, joint, and actuator ids.
5. Runtime stepping applies controls, calls `mj_step`, reads MuJoCo state, and updates `PhysicsSceneState`.
6. `SimulationServer::ApplyWorldStateToScene()` writes simulation-owned state back to Gobot scene nodes.

URDF imports should mark a root link with no visual, no collision, and no inertial data as `LinkRole::VirtualRoot`. The scene snapshot carries this as `PhysicsLinkRole::VirtualRoot`, and physics backends should ignore it for body synchronization instead of warning about a missing backend body. This keeps virtual wrapper links semantic and avoids name-based filters such as `world` / `<robot>_world`.

Scene edits must use two different paths:

- Runtime state edits, such as joint position targets, velocities, efforts, reset poses, floating base pose, and control mode, should update `PhysicsSceneState` / `mjData` and call `mj_forward` without rebuilding `mjModel`.
- Model/topology edits, such as changing `Robot3D::source_path`, adding/removing links or joints, changing joint type/axis/limits, changing collision shapes, or changing actuators, require rebuilding the physics world.

The rebuild path is:

1. Pause stepping.
2. Save the previous `PhysicsSceneState`.
3. Capture a fresh scene snapshot.
4. Destroy old backend runtime data (`mjData`) and model data (`mjModel`).
5. Parse/load/generate MuJoCo XML or `mjSpec`, compile with `mj_compile`, and allocate `mjData`.
6. Rebuild link, joint, and actuator bindings by name.
7. Restore compatible previous state by robot/link/joint name, preserving only matching joint types.
8. Sync the restored scene state into `mjData`.
9. Call `mj_forward`.
10. Sync MuJoCo state back into `PhysicsSceneState` and then into scene nodes where motion mode permits it.

`SimulationServer::RebuildWorldFromScene(scene_root, preserve_state)` owns this high-level operation. Backend implementations should keep details such as `mjModel`, `mjData`, MuJoCo ids, and `mjSpec` below `PhysicsWorld`; editor and future Python APIs should call the simulation service rather than backend-specific APIs.

Exporting a temporary URDF or MJCF file is acceptable as a transition step, but it should not become the long-term source of truth. The preferred direction is `Gobot Scene -> PhysicsSceneSnapshot -> mjSpec -> mj_compile`, so backend-specific concepts do not leak upward into scene or Python-facing APIs.

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

- Whether the simulation world should live only behind `SimulationServer`, or also have a scene subsystem object for multi-world editing.
- Whether robotics algorithms live as engine plugins, Python packages, or C++ services with Python wrappers.
- How to represent sensors and advanced robot articulations once the basic `Robot3D`/`Link3D`/`Joint3D` model is stable.
- How command/undo should work across C++, editor UI, and Python.
