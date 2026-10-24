# Native MuJoCo CPU batches

Gobot uses the `mjBatch` C runtime shipped in MuJoCo **3.17.0**. This includes
upstream commit `c00ed815386eb2690c8be28855c45ea4d683e6c4` (October 21, 2026).
The archived third-party mjbatch submodule and its adaptation patch are removed.

Existing Gobot batch, reset, checkpoint, controller and training interfaces remain
the entry points. MuJoCo Warp remains the CUDA provider. The single-pole
[cartpole example](../examples/cartpole/README.md) also uses the official
`mujoco.batch.Batch` Python wrapper for trajectory optimization.

## Ownership and stepping

- MuJoCo owns worker threads, shared model assets, per-thread workspaces,
  integration rows and expanded per-environment model parameters. Gobot retains
  controller state, selected derived outputs and variable-length contact wrenches.
- The private `MuJoCoBatch` adapter uses `mjb_apply` to step and capture results
  before the native workspace is reused. Controls and spring forces are evaluated
  every physics tick. Engine callbacks return before raw MuJoCo stepping, so native
  error handling does not jump over C++ destructors.
- Observations retain normal `mj_step` timing. No extra forward pass changes the
  meaning of contact forces, accelerometers or actuator observations.
- Individual engine operations use one scratch model/data pair. These expose
  cached results without copying shared mesh and terrain assets per environment.
- `sim_workers=0` uses logical CPU count, capped by environment count. Because
  native batch thread counts are fixed, a worker-count change recreates the batch
  and restores its state. It does not reset the simulation.
- Mass, center-of-mass and friction randomization use expanded model fields;
  `mjb_setConst` refreshes derived constants without changing episode state.

## Checkpoints and failures

Checkpoints store integration state, warning counters, model inputs, derived
outputs, contacts and engine controller state. Their backend payload is private
and versioned, and checks the MuJoCo version. Older mjbatch payloads are not
accepted; checkpoints are runtime objects, not an authoring file format.
Selection resets and restores leave other environments unchanged.

Native and engine callback failures identify the environment. Other environments
finish their call; invalid simulation state requires reset. The adapter does not
install a global MuJoCo log handler. Sleep and native MuJoCo IPC remain unsupported
by this CPU batching path; Gobot's separate IPC provider is unaffected.

## Build and validation

Install the pinned Python dependencies or point CMake at a MuJoCo 3.17 SDK.
Configuration checks native batch headers and symbols. `GOB_BUILD_MUJOCO=OFF`
requires neither MuJoCo nor a patch utility.

Run `test_mujoco_batch`, physics/simulation/controller tests, cartpole playback and
Go1 CPU tests. Core coverage includes independent MuJoCo parity, sensors and
contacts, changing worker counts, randomization, checkpoint replay, selective
reset, malformed checkpoints and error recovery. Use the Go1 benchmark with the
same environment/worker counts, warmup, seed and actions when comparing speed or
memory with an earlier revision.
