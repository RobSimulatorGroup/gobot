# MuJoCo RL: contracts and roadmap

MuJoCo CPU is the semantic baseline; MuJoCo Warp provides CUDA batch execution.
Both consume Gobot-authored scenes. Start with the [Go1 guide](../examples/go1/README.md)
for runnable training commands and [architecture](architecture.md) for engine
ownership and policy/checkpoint contracts.

## Runtime

```text
.jscn / authored scene
  -> PhysicsSceneCompiler
  -> native CPU batch or portable PhysicsSceneArtifact
  -> Python task environment
  -> rsl_rl / other training adapter
```

CPU batches use the pinned [mjbatch native core](mjbatch_cpu.md). Warp and
Newton are Python providers consuming a validated schema-v3 artifact from
`AppContext.compile_scene_artifact()`. Compilation does not install an active
physics world. Providers validate content digests, producer metadata,
dimensions, topology, and the explicit control map.

Training tasks live under `examples/`; they define observations, rewards,
termination, commands, and curriculum in Python. C++ supplies generic controls,
stepping, sensors, and state extraction. The old string-dispatched
`ManagerBasedEnv`, task-JSON `NativeVectorEnv`, and Go1 LLVM/JIT task path have
been removed. Future task APIs should retain the typed batch boundary.

## Task contract

| Value | Shape or rule |
| --- | --- |
| Actions | `(num_envs, action_dim)` |
| Observations | `(num_envs, obs_dim)` |
| Reward, terminated, truncated | `(num_envs,)` |
| Environment time step | `env_dt = physics_dt * decimation` |
| Reward scaling | Use `env_dt` |
| Reset | Explicit masks; each environment has independent RNG and episode state |
| Topology | One scene topology per batch; also batched when `num_envs == 1` |

The step order is:

```text
process action
  -> apply controls, step physics, update state for each substep
  -> termination and reward
  -> reset done environments
  -> observation
```

Go1 auto-resets without constructing terminal observations in its hot path.
Generic environments that offer `final_observation` must capture it before
reset. Timeouts set `truncated`; task failures set `terminated`.

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
Go1VelocityEnv.step
  -> NativeLocomotionBatchBackend.step_task_inputs
  -> PhysicsWorld::StepRobotBatch
  -> mjbatch workers and typed state/contact/sensor results
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

`Go1WarpVelocityEnv` keeps physics, sensing, task state, observations, rewards,
resets, and actions on CUDA. CPU remains the short-horizon parity oracle.
Model randomization should use expanded per-world fields; data randomization
and reset should reuse persistent buffers. Each environment retains its own
random stream.

Provider cache keys include the artifact digest, provider version, and normalized
configuration. Prepared assets use `$XDG_CACHE_HOME/gobot/physics` or
`~/.cache/gobot/physics`; `GOBOT_PHYSICS_CACHE_DIR` overrides the root. Entries
use checksums, process locks, corruption invalidation, and atomic publication.

## Policies, checkpoints, and playback

Training, ONNX export, and playback share `gobot.rl.PolicyManifest`. Loaders
reject missing or mismatched observation/action specs, joint order, control
settings, or simulation-scene digest. See the [policy contract](architecture.md#policy-contract).

`RslRlVecEnvWrapper` passes task training state through without interpreting
curriculum fields. Go1 saves scheduler progress, terrain assignments,
profile-local progress, and RNG state. Equal batch sizes restore assignments;
changed sizes preserve each terrain type's level histogram. Switching from
`balanced` to `run` restarts only the profile-local curriculum. Legacy fallback
must be explicit. Training resume starts fresh physics episodes; exact runtime
checkpoints are a separate engine feature.

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
- Same-seed replay, env-0 parity between single/batched execution, and masked reset.
- CPU/Warp short-horizon parity for state, contacts, sensors, and task terms.
- Graph replay with changing reset masks; recapture after storage changes.
- Clear capacity/non-finite diagnostics and deterministic provider shutdown.
- Play lifecycle callbacks, native/external exclusion, failure pausing, and
  preservation of authored scene data.
