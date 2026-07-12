# Gobot MuJoCo RL Plan

This document records the intended Gobot reinforcement-learning direction for
MuJoCo CPU and MuJoCo Warp. Gobot scene data remains the authoring source of
truth. MJCF, `mjModel`, MuJoCo data, and Warp arrays are runtime artifacts
compiled from Gobot scenes.

## Runtime Shape

The target pipeline is:

```text
Gobot SceneTree / .jscn
  -> scene-to-physics compile layer
  -> Python task envs using Gobot batch action/state APIs
  -> MuJoCo CPU semantic baseline
  -> MuJoCo Warp CUDA graph fast path
  -> editor debug / policy playback
```

Training is Python-driven through `gobot.rl`. Core engine APIs must not depend
on Gymnasium, rsl_rl, ImGui, viewport selection, or raw MuJoCo/Warp pointers.
Compatibility wrappers can live above the core API.

## Script And Editor Boundaries

- `NodeScript` and `ScenePlaySession` are for editor Play Mode, single-scene
  debugging, and policy playback.
- Packaged editor examples live under `examples/` in the source tree and
  `gobot/examples/` in wheels. See `doc/examples.md`.
- Play Mode runs scripts on a runtime clone of the edited scene. The editor
  viewport and physics world use that clone while playback is active, so script
  motion is visible without mutating the edited scene.
- Physics controls that switch backend or rebuild the world are disabled while
  scripts are running.
- RL training does not run node scripts and does not enter editor Play Mode.
- Python Panel `Run Once` executes a tool script against the active editor
  context. It does not install tick callbacks, enter Play Mode, or participate
  in vectorized RL stepping.
- Editor playback should use the same Gobot action/control API as training, but
  it remains a visualization/debug surface rather than the training runtime.

Long-term, Gobot should provide an explicit runtime scene owner/root for
headless simulation, offscreen rendering, editor Play Mode, and policy
playback. Loading a scene for runtime use should attach instantiated nodes to a
runtime tree with normal lifecycle, path lookup, world membership, and
visibility semantics, instead of returning a permanently detached root and
spreading `IsInsideTree()` exceptions through engine code. This runtime root is
not the edited SceneTree: it must not appear in the editor SceneTree dock, must
not participate in undo/redo or dirty-state tracking, and must be destroyed as a
runtime artifact when the session/evaluation ends.

## Scene To Runtime Compile Layer

The compile layer should produce a stable runtime name map from `.jscn` or a
live `SceneTree`:

- robot names
- body/link names
- joint names
- actuator names
- sensor names

Gobot joint controllers are the public control API. They should cover passive,
effort, position, velocity, gains, limits, action scaling, and clipping. The
MuJoCo backend translates Gobot joint commands into MuJoCo actuator controls
below this boundary.

The Python-facing runtime map and state APIs should expose Gobot names and
engine units. They should not expose backend array indices as the primary user
contract.

## Python Task Environment Layer

The intended package shape is:

```text
gobot.sim
  MuJoCo CPU stepping/state/control

gobot.rl
  locomotion helpers
  training config helpers
  compatibility wrappers above task envs

gobot.app
  editor/runtime context
```

Default step order:

```text
process_action(action)

for substep in decimation:
    apply_action()
    physics_step()
    update_runtime_state()

compute_termination()
compute_reward()
reset_done_envs()
compute_observation()
```

Default semantics:

- `physics_dt` and `env_dt` are separate.
- `env_dt = physics_dt * decimation`.
- Reward terms use `env_dt`.
- Done environments auto-reset by default.
- Generic batch environments may opt into `final_observation`; the Go1 task
  follows immediate reset semantics and does not build a terminal observation
  in its hot path.
- Core APIs are batched even when `num_envs == 1`.

Current implementation status:

- The legacy string-dispatched `gobot.rl.ManagerBasedEnv` prototype has been
  removed.
- Vectorized training environments currently live as normal Python task modules
  under `examples/`.
- Task code defines observations, rewards, resets, commands, and metrics in
  Python while C++ provides generic batch stepping, action application, and
  runtime state extraction.
- The batch API should stay backend-neutral: MuJoCo CPU may implement it as one
  shared `mjModel` with one `mjData` per environment, while MuJoCo Warp should
  expose the same reset/action/state surface over `wp_model` / `wp_data`.
- Fixed task functions should be expressed as NumPy batch functions over
  structured arrays, not as arbitrary per-step Python scene traversal. That is
  the array-first contract: action, physics state, reward, done, reset masks,
  and observations are all `(num_envs, *)` arrays.
- The previous runtime LLVM/JIT task-kernel path has been removed from the Go1
  training surface. Future code generation can be reconsidered only after the
  CPU batch contract, backend arrays, reset masks, and async rollout path are
  stable.
- Python task code may read batch state arrays, previous actions, commands,
  randomization buffers, and done masks. It must not call `node.find()`, pull
  Python dictionaries from `get_runtime_state()`, or mutate scene nodes inside
  the hot reward/observation path.

## MuJoCo CPU VectorEnv Baseline

The first real vector backend should prioritize correctness over throughput.

Required behavior:

- API is always batched:
  - action: `(num_envs, action_dim)`
  - observation: `(num_envs, obs_dim)`
  - reward: `(num_envs,)`
  - terminated/truncated: `(num_envs,)`
- `num_envs > 1` may initially own multiple independent CPU MuJoCo worlds/data.
- One VectorEnv only supports one scene topology.
- Each environment owns independent seed, episode length, reset state, command,
  target, and runtime metrics.
- CPU VectorEnv is the behavior reference for MuJoCo Warp.
- Reset and terminal handling should be expressed as explicit environment masks
  so CPU and Warp can share task code.

Current native CPU implementation status:

- The generic C++ `NativeVectorEnv` / task-json path has been removed.
- Go1 training loads the Gobot `.jscn` scene and uses the engine's typed CPU
  batch API. Python never loads MJCF or reads `mjModel` / `mjData`.
- A future engine-backed vector env should expose a clean task API instead of
  serializing reward/observation definitions through opaque JSON.
- Go1 exposes task runtime metadata in `env.cfg["task_runtime"]`. The current
  runtime is `numpy`: C++ owns persistent action/task buffers, command update,
  typed physics stepping, and state/contact/height-scan extraction; Python
  computes observation, reward, and termination with vectorized NumPy.

Current Gobot Go1 CPU batch env shape:

```text
Python Go1VelocityEnv.step(action)
  -> NativeLocomotionBatchBackend.step_task_inputs(...)
  -> AppContext native batch view
       prepares clipped/scaled joint targets
       submits PhysicsRobotBatchStepRequest with Gobot names and typed arrays
  -> PhysicsWorld::StepRobotBatch(...)
       applies per-environment model randomization and pushes
       steps the backend's persistent CPU worker pool
       returns PhysicsRobotBatchStepResult state/contact/sensor arrays
  -> vectorized NumPy reward, termination, observation, and reset
  -> stable BatchEnvState returned to the training wrapper
```

The scene remains the source of truth. `PhysicsSceneCompiler` compiles the
loaded `.jscn` hierarchy once, and the MuJoCo backend builds each CPU
environment from that snapshot. Training must not load an MJCF file as a
second runtime source.

The backend-neutral kernel split is complete: Python sources include no physics
backend headers, `MuJoCoPhysicsWorld` has no Python friend, and architecture
tests reject raw MuJoCo pointers or binding storage in Python code. Batched
velocity-command state, sampling, timers, and frame conversion now live in the
Python-independent `LocomotionCommandRuntime` service with an independent RNG
per environment. `LocomotionBatchRuntime` now composes that service and owns
the resolved robot-state layout, float32 base/joint/link/foot/sensor buffers,
batched foot contact events, air/contact timing, landing forces, peak heights,
and collision counters. Every physics result is checked against the resolved
name and dimension contract before extraction. Task-specific reward and
observation scratch arrays are allocated by the Python task backend; the
pybind class only adapts NumPy arrays and lifecycle calls to the simulation
service.

Performance work must preserve deterministic reset, stable joint ordering,
substep contact history, and the policy manifest contract. Use persistent
workers and preallocated buffers; do not reintroduce direct-XML benchmark
paths as training APIs.

## MuJoCo Warp Fast Path

The Warp backend must be designed around persistent buffers, reset masks, and
CUDA graph replay.

Warp's LLVM/codegen path is useful as an implementation reference, not as an
immediate runtime dependency for Gobot's CPU MuJoCo training path. Gobot should
first stabilize flat Gobot/MuJoCo batch buffers:

```text
.jscn + robot task config
  -> mjModel + offset tables + TaskRuntimeMetadata
  -> native batch step + NumPy task update
  -> optional generated native/Warp kernel later
```

That keeps `.jscn` / SceneTree compile-only for training hot paths and avoids
turning arbitrary Python functions into runtime scene traversal. If Gobot later
adds more code generation, it must preserve the same batch buffer contract and
remain optional behind the NumPy reference path.

Initialization:

```text
compile model
put_model()
put_data(nworld=num_envs, nconmax, njmax)
allocate persistent action/obs/reward/done buffers
allocate persistent reset_mask
create_graph()
```

Graphs to capture and manage:

- `step_graph`
- `forward_graph`
- `reset_graph`
- `sense_graph`

Runtime rule: never replace arrays captured by a graph during the training loop.
Only mutate array contents.

Per environment step:

```text
process_action(actions)

for substep in decimation:
    write persistent ctrl/action buffer
    launch step_graph

compute termination/reward
write reset_mask for done envs
launch reset_graph if needed
launch forward_graph if needed
launch sense_graph if sensors enabled
compute observation
```

`reset_graph` reads the persistent GPU reset mask:

```text
reset_mask[num_envs]
mjwarp.reset_data(model, data, reset=reset_mask)
```

These operations require graph recapture:

- model or data arrays are replaced
- `expand_model_fields()` changes backing arrays
- sensor context changes
- render or raycast pipeline changes
- per-world variant fields are reallocated
- contact or constraint capacity is rebuilt
- `nconmax` or `njmax` changes

Training loops must report an error instead of silently falling back to CPU when
the requested Warp path is unavailable.

Training loops must also avoid:

- per-step action/observation/reward buffer allocation
- per-step model/data array replacement
- per-env Python loops around physics stepping
- implicit CPU fallback

### Implemented Provider Boundary

The first provider infrastructure follows this split:

```text
SceneTree / .jscn
  -> PhysicsSceneCompiler
  -> PhysicsServer backend artifact compiler
  -> versioned PhysicsSceneArtifact (canonical MJCF + digest + name map)
  -> gobot.rl.MuJoCoWarpProvider
  -> MuJoCo Warp model/data and Torch CUDA views
```

`AppContext.compile_scene_artifact()` performs compilation without installing a
runtime `PhysicsWorld`. C++ and MuJoCo Warp communicate through the artifact
value, not through `mjModel*`, Warp arrays, CUDA pointers, or editor state. The
provider is an optional Python package layer and therefore is not a native
`PhysicsBackendType` exposed to scene nodes or editor serialization.

The provider currently owns persistent model/data arrays, a reset mask,
zero-copy Torch views, and captured step/forward/reset/sense graphs. It rejects
incompatible artifacts, changed captured storage, unavailable CUDA runtimes,
fixed-capacity overflow, and non-finite state explicitly. The RSL-RL adapter
preserves device-native action, observation, reward, and timeout tensors when a
task environment supplies them.

This is provider infrastructure, not yet a complete CUDA Go1 task. The current
Go1 reward, observation, terrain scan, contact history, command, and domain
randomization implementation remains the NumPy/MuJoCo CPU semantic baseline.
Moving that task to CUDA requires persistent task buffers and Warp/Torch kernels
for those terms; it must not wrap the CPU task in device transfers.

### Newton Admission Boundary

Newton should use the same provider lifecycle before it becomes a public Gobot
backend. A prototype may live in an optional Python provider/plugin and consume
either the existing compiled artifact or a Newton-specific artifact registered
through `PhysicsServer`. It must not add Newton handles or Warp arrays to
`Scene`, `Robot3D`, `SimulationServer`, or editor APIs.

The initial Newton prototype must demonstrate all of the following:

- stable Gobot robot/link/joint/sensor name resolution
- fixed-capacity batched allocation with actionable overflow diagnostics
- persistent device buffers and explicit graph recapture rules
- masked reset and deterministic seed replay
- joint control, state, contacts, and required sensor parity with MuJoCo CPU
- short-horizon CPU parity tests and a standalone throughput benchmark
- isolated optional dependency versions so Newton cannot silently replace the
  Warp/MuJoCo versions used by the MuJoCo Warp provider

Until those checks pass, do not add `NewtonGpu` to the public backend enum and
do not make Newton a core build or wheel dependency. The local upstream
checkouts are references only; normal Gobot builds and wheels must not discover
or import them implicitly.

## Randomization And Variants

Data randomization is cheap and should happen through persistent data buffers:

- `qpos` / `qvel`
- initial pose
- target command
- episode state

Model randomization is more expensive:

- mass
- friction
- damping
- actuator gains
- geom/material/mesh variants

Model randomization should prefer expanded per-world model fields when possible.
Replacing captured arrays requires graph recapture. Each environment needs an
independent RNG, and seed replay should be deterministic.

## Sensors, Rendering, And Playback

- Training does not depend on the editor viewport.
- Sensor work belongs in `sense_graph` for the Warp path.
- The editor should show one environment or a selected environment.
- Policy playback uses the same Gobot action/control API as training.
- Physics/debug panels may display backend, simulation time, active policy,
  controlled joints, current action, reward/debug terms, and selected env id.

## Test Checklist

Scene script boundary:

- Play Mode runs `_ready`, `_process`, `_physics_process`, and `_exit_tree`.
- Python Panel Run Once does not start Play Mode.

Scene to runtime compile:

- cartpole `.jscn` compiles to a stable joint/action map.
- slider effort or position control is effective.
- passive hinge state is preserved.

Manager layer:

- action clipping
- reward dt scaling
- timeout produces `truncated`
- failure produces `terminated`
- terminal observation is saved before reset

CPU VectorEnv:

- `num_envs=1` and `num_envs=N` env0 match under the same seed and actions.
- reset masks only reset done environments.
- fixed seed replay is deterministic.
- observation/reward/done shapes are correct and finite.

MuJoCo Warp:

- step/forward/reset/sense graphs can be captured and replayed.
- changing reset mask contents affects graph replay.
- replacing captured model/data arrays requires recapture.
- captured buffers are not reallocated inside the training loop.

Parity and diagnostics:

- CPU and Warp match over short horizons within tolerance.
- contact/constraint capacity overflow reports a clear diagnostic.
- NaN guards identify the environment id.
- editor policy playback does not mutate authored scene data.
