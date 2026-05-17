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
  -> Python task envs / gobot.rl ManagerBasedEnv
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

## `gobot.rl` Manager Layer

The intended package shape is:

```text
gobot.sim
  MuJoCo CPU stepping/state/control

gobot.rl
  ManagerBasedEnv
  managers
  specs
  wrappers

gobot.app
  editor/runtime context
```

`ManagerBasedEnv` is organized around these pieces:

- `ActionManager`
- `ObservationManager`
- `RewardManager`
- `TerminationManager`
- `EventManager`
- `CommandManager`
- `Recorder` / `Metrics`

Default step order:

```text
process_action(action)

for substep in decimation:
    apply_action()
    physics_step()
    update_runtime_state()

compute_termination()
compute_reward()
store_terminal_observation()
reset_done_envs()
compute_observation()
```

Default semantics:

- `physics_dt` and `env_dt` are separate.
- `env_dt = physics_dt * decimation`.
- Reward terms use `env_dt`.
- Done environments auto-reset by default.
- `info["terminal_observation"]` stores the observation before reset.
- Core APIs are batched even when `num_envs == 1`.

Current implementation status:

- `gobot.rl.ManagerBasedEnv` exists as the single-runtime Python manager
  reference path.
- Example vectorized training environments currently live as normal Python
  task modules under `examples/` instead of a generic native C++ task-json
  backend.
- `GymWrapper` and `RslRlVecEnvWrapper` live above the core API.

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

Current native CPU implementation status:

- The generic C++ `NativeVectorEnv` / task-json path has been removed.
- CartPole and Go1 training use explicit Python MuJoCo `VecEnv` classes in the
  example projects.
- A future engine-backed vector env should expose a clean task API instead of
  serializing reward/observation definitions through opaque JSON.

EnvPool reference notes:

- EnvPool uses an `ActionBufferQueue -> ThreadPool -> StateBufferQueue` model.
  Workers pull single-env action slices and push finished state slices into
  preallocated buffers.
- Its fastest path avoids most locks: action dispatch uses atomic ring-buffer
  allocation plus lightweight semaphores, and state writes use packed atomic
  offsets to reserve non-overlapping output slices.
- EnvPool also keeps a stock of preallocated state buffers so the hot path
  usually avoids allocation, and async mode can return whichever environments
  finish first instead of waiting for the slowest environment.
- Gobot's first CPU VecEnv intentionally does not copy the full EnvPool queue
  machinery. It keeps the engine-facing boundary simple while validating
  deterministic reset, named joint control, observation/reward contracts, and
  rsl_rl training.
- Follow-up performance work can replace the row-parallel worker dispatch with
  EnvPool-style fixed queues, preallocated output buffers, per-env ready queues,
  optional CPU affinity, and true async "first completed batch" semantics once
  the Gobot simulation contracts are stable.

## MuJoCo Warp Fast Path

The Warp backend must be designed around persistent buffers, reset masks, and
CUDA graph replay.

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
