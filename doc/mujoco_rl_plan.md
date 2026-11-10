# MuJoCo RL: contracts and roadmap

MuJoCo CPU is the semantic baseline; MuJoCo Warp provides CUDA batch execution.
Both consume Gobot-authored scenes. Start with the [Go1 guide](../examples/go1/README.md)
for runnable training commands and [architecture](architecture.md) for engine
ownership and policy/checkpoint contracts.

The headless training entry point is
`python -m gobot.rl.train`. See [the training framework](gobot_training.md).
Velocity PPO supports synchronized multi-GPU training. Motion tracking and AMP
use their own task configurations and learner capabilities.

## Runtime

```text
.jscn / authored scene
  -> PhysicsSceneCompiler
  -> native CPU batch or portable PhysicsSceneArtifact
  -> Python task environment
  -> rsl_rl / other training adapter
```

CPU batches use the [MuJoCo native batch runtime](mujoco_cpu_batch.md). Warp and
Newton are Python providers consuming a validated scene artifact from
`AppContext.compile_scene_artifact()`. Compilation does not install an active
physics world. Providers validate content digests, producer metadata,
dimensions, topology, and the explicit control map.

Robot/task presets, model selection and example evaluators live under `examples/`,
with shipped velocity recipes in `examples/locomotion/presets.py`. Reusable task
implementations live in `gobot.rl.locomotion`. The trainer accepts `RunConfig` and
a typed `VelocityTaskCfg`, or an importable task configuration factory; it does
not select a robot by name. `LocomotionRobotCfg`, `VelocityTaskCfg`, and
`make_velocity_env` share CPU/Warp observations, reward configuration, termination,
commands and curriculum. See [G1](../examples/g1/README.md) and the
[shared tools](../examples/locomotion/README.md). C++ supplies generic controls,
stepping, sensors, and state extraction. Task APIs use the typed batch boundary.

## Task contract

| Value | Shape or rule |
| --- | --- |
| Actions | `(num_envs, action_dim)` |
| Observations | `(num_envs, obs_dim)` |
| Reward, terminated, truncated | `(num_envs,)` |
| Environment time step | `env_dt = physics_dt * decimation` |
| Reward scaling | Use `env_dt` |
| Reset | Explicit masks and per-environment episode state; Warp velocity events share Torch RNG with policy sampling |
| Topology | One scene topology per batch; also batched when `num_envs == 1` |

The step order is:

```text
process action
  -> apply controls, step physics, update state for each substep
  -> termination and reward (last substep state, previous ray sample)
  -> interval events and reset done environments
  -> forward once, advance commands, sense and return observations
```

Standard velocity training uses unmodified RSL-RL PPO, including its timeout
bootstrap from the value saved when the action was sampled. Failure and timeout
masks are independent. Reset clears actuator targets; the next policy action
sets the usual default-pose offsets. Transition diagnostics are optional and
must not consume RNG or add forward/sense calls. AMP remains an explicit
extension with its own final-state value handling. The reference behavior and
executable comparison protocol are documented in `doc/mjlab_comparison.md`.

Task hot paths operate on contiguous NumPy arrays for CPU or device tensors
for CUDA. Resolve Gobot joint/link/sensor names once. Keep scene traversal,
per-node dictionaries, raw backend pointers, and per-environment Python physics
loops out of step/reward/observation code. Gymnasium and rsl_rl adapters stay
above the engine API.

`LocomotionBatchRuntime` owns typed robot/sensor/contact buffers and composes
`LocomotionCommandRuntime` for command sampling. Python owns task-specific
observation/reward/termination scratch arrays. Controllers use Gobot joint
commands, gains, limits, scaling, and clipping; backend actuator translation
stays inside the physics implementation.

## CPU and CUDA execution

CPU stepping follows:

```text
CpuVelocityEnv.step
  -> NativeLocomotionBatchBackend.step_task_inputs
  -> PhysicsWorld::StepRobotBatch
  -> MuJoCo native batch workers and typed state/contact/sensor results
  -> vectorized NumPy task update
```

Per-environment randomization, external forces, deterministic reset, stable
joint ordering, and substep contact history are part of this contract. The
scene compiles once; training does not reload source MJCF as a second model.

Warp owns persistent model/data, controls, reset masks, sensor buffers, and
Torch CUDA views. It captures step, forward, reset, and sense graphs. Mutate
captured arrays in place. Replacing storage, expanding model fields, changing
sensor/raycast contexts, or rebuilding contact/constraint capacities requires
graph recapture. Overflow, non-finite state, incompatible artifacts, and
unavailable CUDA must produce explicit errors. Backend selection never silently
falls back to CPU.

`WarpVelocityEnv` keeps physics, sensing, task state, observations, rewards,
resets, and actions on CUDA. CPU remains the short-horizon parity oracle.
Model randomization uses expanded per-world fields; data randomization and reset
reuse persistent buffers. Warp velocity events and PPO share the Torch device RNG.
Substep contact timers advance with physics ticks, not forward/sensing passes.
Weighted, time-scaled non-finite reward terms are zeroed; observations are not
sanitized. RSL-RL rejects non-finite training outputs. Rollout exceptions are
reported across workers before optimization.

Provider cache keys include the artifact digest, provider version, and normalized
configuration. Prepared assets use `$XDG_CACHE_HOME/gobot/physics` or
`~/.cache/gobot/physics`; `GOBOT_PHYSICS_CACHE_DIR` overrides the root. Entries
use checksums, process locks, corruption invalidation, and atomic publication.

## Policies, checkpoints, and playback

Training, ONNX export, and playback share `gobot.rl.PolicyManifest`. Loaders
reject missing or mismatched observation/action specs, joint order, control
settings, or simulation-scene digest. See the [policy contract](architecture.md#policy-contract).

`RslRlVecEnvWrapper` passes task training state through without interpreting
curriculum fields. Velocity checkpoints save scheduler progress, terrain
assignments, profile-local progress and RNG state. Trainer resume requires the
same world size and environments per rank, restores those assignments, then
starts fresh physics episodes. Checkpoint restoration preserves saved terrain assignments;
explicit reset applies the reference curriculum. Exact runtime checkpoints are a separate engine
feature.

AMP datasets carry a `MotionFeatureSpec` with named joints and links. The loader
derives feature widths and velocity indices from that layout. The trainer checks
the dataset layout, control period and robot physics identity against the task.
Dataset-specific file names and train/validation/holdout splits belong in example converters.

Go1 policy admission requires survival and commanded planar/yaw progress across
all authored terrain cells. Run evaluation also records paired-foot gait
metrics; reward or speed alone does not establish a successful running gait.

Play Mode uses a runtime clone and never modifies the edited scene. Training
runs without node scripts or editor dependencies. Python-backed playback uses
`ProviderPlaySession`; `SimulationServer` owns its fixed-step clock and
pause/reset/sync/stop lifecycle. Native and external sessions are exclusive.
Callback or sync failures pause the session until reset. Python Panel **Run
Once** remains a tool action and does not start a simulation session.

## Remaining work and validation

- Consolidate runtime scene ownership across headless simulation, rendering,
  and Play Mode, keeping it outside editor undo/redo and dirty-state tracking.
- Keep the [Newton provider](newton_openusd.md) behind the artifact/session
  boundary. Direct neutral-snapshot compilation and graph capture remain
  follow-up work. Do not add backend handles to scene or editor APIs.
- Optimize buffers and rollout only while preserving reset, contact, and policy
  contracts. Reconsider generated task kernels after the array contract is stable.

Required regression coverage:

- Stable compiled names/control maps; effective effort/position control.
- Action clipping, reward time scaling, finite shapes, and correct done flags.
- Same-seed replay, masked reset, and single/batched state parity with randomization disabled.
- Reference RNG consumption and curriculum-preserving checkpoint restoration.
- CPU/Warp short-horizon parity for state, contacts, sensors, and task terms.
- Graph replay with changing reset masks; recapture after storage changes.
- Clear capacity/non-finite diagnostics and deterministic provider shutdown.
- Play lifecycle callbacks, native/external exclusion, failure pausing, and
  preservation of authored scene data.

Velocity tasks calculate rewards from the last physics substep, then refresh
kinematics and sensors after events and resets for the returned observation. Native command
RNG streams are included in CPU training checkpoints. G1 and Go1 exercise the
same trainer, and custom robot factories must not require edits to the runner.
