# Physics runtime migration

This work implements the agreed M0–M5 runtime plan. The acceptance target is
runtime and physics correctness, not a successful conveyor bag/carton flip.

## Frozen comparison

The reference is Gobot `2379635b0fa1388d1faf2ca994bac68b94519650` and libuipc
`a87612b19d27595a1bf1fb1e119d570b75f78e52`. SceneTree/.jscn remains authoritative.
Do not change authored geometry, materials, timestep or controller gains to
obtain a faster runtime comparison. See `ipc_manipulation_benchmarks.md` for
the previous RTX 4070 measurements and their limitations.

Use Release builds and record artifact digests, solver configuration, output
selection, completed microsteps, failures, synchronized stage timings and
sampled process memory. Run GPU benchmarks sequentially in three fresh processes.
Conveyor measurements retain 800 warmup and 100 contact-acquisition steps.

## Delivery sequence

1. M1: SDK-independent session/progress contract, native/provider adapters.
2. M2: worker-owned provider lifecycle, bounded completed snapshots and scene sync.
3. M3: common compile snapshot and explicit contact ownership.
4. M4: restricted scalar-joint articulation coupling and physical admission.
5. M5: deformable views, per-shard reset, isolated-environment benchmarks.

Internal API callers are migrated together; retired entry points are not kept
as permanent compatibility layers. Independent reset first uses one environment
per IPC world. Shared-world partial reset and persistent composite checkpoints
are outside this delivery. An internal single-slot rollback checkpoint does not
qualify as a persistent user checkpoint.

## Implemented runtime boundary

`SimulationSession` and `SimulationTaskWorker` are SDK-independent C++ services.
Native physics and Python data executors use the same bounded worker mailbox;
the session preserves committed microsteps when a later microstep fails.
Provider imports now live under `gobot.sim.providers`, outside the RL package.

For asynchronous Play, compile the scene on its owner thread and create a
`SimulationRuntimeSpec("module:function", arguments)` from configuration and
compiled artifact data. Binary artifact blobs are encoded explicitly; live
contexts, nodes, providers and Python callbacks are rejected as configuration.
The factory creates the provider, optional controller and output translator on
the worker and returns `RuntimeComponents`. Build, control, step, reset, close
and SDK destruction remain on that thread. A factory must not access scene
services through globals either.

`AsyncSimulationSession` defaults to environment 0 and at most 60 snapshot
publications per second. Three leased CPU frames retain read-only NumPy buffers;
holding all frames skips presentation without stopping physics. Subscriptions
changed during a solve are coalesced and applied after its completion, preserving
the completed step. Editor Stop retires the worker without joining; explicit
`shutdown()` and interpreter shutdown join after releasing the Python GIL.
The Python adapter keeps one CPython thread state for the worker's lifetime,
while releasing the GIL between calls. This also covers factory failure and
retirement/reinstallation: Torch's separate pybind11 internals may cache the
state used on first import, so per-callback state creation/destruction is unsafe.
The generic C++ worker remains independent of Python.

`ProviderPlaySession` connects completed clocks to `SimulationServer`. It rejects
a Play/runtime timestep mismatch before advancing physics. Physics failure and
snapshot failure are separate: a failed snapshot pauses presentation/playback
while retaining the already completed tick and time. Runtime metadata is copied
to the editor; the editor never reads a worker-owned provider for diagnostics.
External asynchronous sessions are polled independently of the render cadence.

`SceneSnapshotSync` applies selected robot poses and authored-local deformable
vertices on the scene owner thread. The C++ `SimulationSceneSync` validates the
whole deformable batch before applying vertices. Display-only environment offsets
do not modify physical coordinates or move soft vertices twice.

The `examples/mujoco_libuipc` four-environment press and `examples/libuipc` native
FR3 Play scripts use this path. Their controller targets, authored assets, solver
settings and global backend defaults are retained. Native contact-force fields
are requested for presentation only when the debug display is enabled.

## Validation and remaining work

C++ tests cover session progress, bounded workers, epoch discard, frame leases,
invalid output, atomic deformable batches and the embedded Python/Play bridge.
The bridge test runs the worker between editor Python calls, exercises reset,
and verifies a snapshot error retains a completed tick. CPU solver doubles also
exercise both real example factories with compiled binary scene artifacts;
these tests establish the threading/data contract, not contact correctness.

On 2026-09-12 the RTX 4070 host driver was working at 580.178.04. Cold Torch
imports on the worker reproduced `_PyThreadState_Attach: non-NULL old thread
state` on CPU and CUDA before the lifetime fix above. Fresh-process regressions
now pass on both devices, including CPU factory failure and interpreter exit.
The real four-environment MuJoCo Warp + libuipc factory completed its 304-tick
press/hold/release cycle with finite robot/soft-body snapshots, full-batch reset,
another step, and shutdown. The installed editor also completed an automated
Play smoke run (210 rendered frames, 145 completed physics ticks) and exited
normally. These are runtime regression checks, not throughput or grasp-success
acceptance; the repeated M0 GPU comparisons remain pending.

M2 still includes migration of rope and Newton policy Play plus their GPU validation.
M3 common compilation/contact ownership, M4 production articulation coupling,
and M5 independent shard reset and batch admission remain outstanding. The M0
repeated GPU comparisons must use the frozen assets/settings above.

Run the CPU contract and example-factory checks without opening the editor:

```bash
ctest --test-dir build -R 'SimulationSessionContract|SimulationDataWorkerContract|SimulationSceneSync|PythonSimulationRuntime' --output-on-failure
uv run --no-sync --with pytest python -m pytest tests/python/test_simulation_runtime.py tests/python/test_simulation_runtime_torch.py tests/python/test_simulation_session.py tests/python/test_simulation_scene_snapshot.py tests/python/test_libuipc_async_play.py -q
```

Run the cold-import CUDA and full-cycle press regressions on a GPU host:

```bash
GOBOT_RUN_WARP_IPC_GPU_TEST=1 uv run --no-sync --with pytest python -m pytest tests/python/test_simulation_runtime_torch.py -q
```

These tests require the rebuilt Python extension. For a local editable install,
refresh the Python install component from the configured Python ABI build after
building `gobot_python` and `gobot_editor`. The two example editor launch commands
remain `uv run gobot_editor --path examples/mujoco_libuipc` and
`uv run gobot_editor --path examples/libuipc`; they require working CUDA.
