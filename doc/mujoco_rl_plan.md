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
  the UniLab-style contract: action, physics state, reward, done, reset masks,
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
- CartPole and Go1 training use explicit Python MuJoCo `VecEnv` classes in the
  example projects.
- A future engine-backed vector env should expose a clean task API instead of
  serializing reward/observation definitions through opaque JSON.
- Go1 exposes task runtime metadata in `env.cfg["task_runtime"]`. The current
  runtime is `numpy`: native C++ performs batched action preparation, MuJoCo
  stepping, command update, contact/height-scan extraction, and persistent array
  updates; Python computes observation, reward, and termination with vectorized
  NumPy over those arrays.

Current Gobot Go1 CPU batch env shape:

```text
Python Go1VelocityEnv.step(action)
  -> torch action clipping / scaling
  -> GobotSceneBatchBackend.step(target_joint_positions, decimation)
       -> gobot.app.AppContext.set_batch_joint_position_targets(...)
       -> step_batch(decimation, workers)
  -> GobotSceneBatchBackend.refresh()
       -> cached numpy arrays for base, joints, links, sensors, contacts
  -> Python task update_state:
       command sampling, contact history, reward, termination, reset, obs tensors
```

The Go1 task borrows UniLab's useful structure without adopting UniLab's XML
entrypoint:

```text
apply_action(action) -> backend.step(ctrl) -> backend cached arrays -> update_state()
```

This path is closest to the intended engine contract. It preserves Gobot scene
authorship, node names, terrain/sensor nodes, debug visibility, and controller
semantics. Its main throughput cost is that hot-loop observation/reward/reset
work still crosses through Gobot runtime state extraction and Python/Torch task
code. The `GobotSceneBatchBackend` facade is the public training-side boundary;
the direct XML `_MujocoBatchPool` remains a benchmark/internal probe.

State-array MuJoCo batch stepping shape, inspired by UniLab `mujoco_uni`
`batch_env`:

```text
Python benchmark / future task hot path
  -> state0: ndarray[num_envs, nstate]
  -> control: ndarray[num_envs, nstep, ncontrol]
  -> gobot._MujocoBatchPool
       owns one mjModel
       owns one mjData per worker
       uses persistent worker threads
       dynamically assigns env chunks
       releases the GIL
       calls mj_setState / mj_step / mj_getState in C++
  -> next_state: ndarray[num_envs, nstate]
  -> optional sensordata: ndarray[num_envs, nsensordata]
```

This path is deliberately internal for now. It bypasses SceneTree traversal and
named Gobot runtime state extraction, so it is a performance probe and a
candidate hot physics kernel for future RL tasks, not a replacement for Gobot's
authoring/runtime boundary. The useful lesson from UniLab is the data-oriented
contract: batch state/control arrays, persistent C++ worker resources, and no
per-environment Python calls inside the step loop.

The benchmark script is:

```bash
uv run --extra train python benchmark/mujoco_uni_batch_benchmark.py \
  --backend gobot --num-envs 64 --steps 100 --warmup-steps 10 \
  --nstep 10 --threads 16 --actions random
```

For comparison, `--backend rollout` runs DeepMind MuJoCo's official Python
`mujoco.rollout.Rollout` extension. That is also direct C++ MuJoCo stepping; it
is not Gobot and it is not UniLab. It remains a useful performance reference for
raw state/control rollout when the API is specialized around MuJoCo arrays.

`mujoco.rollout.Rollout` is generally not the right training runtime by itself.
It can accept a new `state0` and a control trajectory each call, but its contract
is batch trajectory rollout, not an interactive vector environment. It does not
own per-environment episode state, reset masks, command sampling, domain
randomization, reward/termination terms, observation fusion, named Gobot
joint/link/sensor extraction, or debug/runtime state. Those pieces can be built
around it in Python, but then the training loop tends to reintroduce the Python
bookkeeping and state-copy costs Gobot is trying to move below the hot path.
Gobot should therefore treat rollout as a performance reference and borrow its
data-oriented C++ loop shape, not make it the public RL environment API.

Measured on 2026-06-20 with Go1 flat MJCF, random controls, `nstep=10`,
`threads=16`, `steps=100`, `warmup_steps=10`:

| Path | Envs | Mean call/step ms | Env steps/s | Physics ticks/s | Relative to current Go1 env |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current Gobot Go1 env | 16 | 14.581 | 1,097 | 10,973 | 1.00x |
| Gobot `_MujocoBatchPool` | 16 | 0.939 | 16,648 | 166,483 | 15.17x |
| MuJoCo `rollout.Rollout` | 16 | 0.810 | 18,975 | 189,749 | 17.29x |
| Current Gobot Go1 env | 64 | 51.094 | 1,253 | 12,526 | 1.00x |
| Gobot `_MujocoBatchPool` | 64 | 2.452 | 25,544 | 255,443 | 20.39x |
| MuJoCo `rollout.Rollout` | 64 | 2.168 | 28,307 | 283,075 | 22.60x |

Interpretation:

- The current Gobot Go1 env is end-to-end task stepping: Gobot runtime state,
  named joints/links/sensors, Torch observations, rewards, done/reset handling,
  and action bookkeeping are included.
- `_MujocoBatchPool` and MuJoCo `rollout.Rollout` measure raw MuJoCo
  state/control stepping only. They do not compute Go1 observations, rewards,
  resets, command randomization, terrain curriculum, or Gobot debug state.
- Gobot's internal state-array pool is now within roughly 88-90% of MuJoCo
  `rollout.Rollout` on this benchmark. The remaining gap is expected: rollout is
  a mature MuJoCo-specific extension with a very narrow API and tuned thread
  scheduling, while Gobot's pool is a first internal probe that still returns
  Python-facing arrays through Gobot's binding layer.
- The next useful optimization is not to expose `_MujocoBatchPool` as a public
  API. Instead, move Go1 task hot-path physics, selected sensor extraction, and
  observation buffers toward persistent state/control arrays while preserving
  Gobot scene compilation and named configuration at initialization time.

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
