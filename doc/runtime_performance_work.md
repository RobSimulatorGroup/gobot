# Runtime performance implementation

Target: the full editor on an RTX 4070 (12 GB), Release, 1920x1080.
Physics configurations and example strategies must remain unchanged.

## Progress

- [ ] Capture before/after editor and process GPU-memory baselines.
- [x] Isolate project resolution and resource cache identity by context.
- [x] Make event subscriptions lifetime-owned and step failures explicit.
- [x] Move physics debug preparation out of the graphics backend.
- [x] Bound OpenGL/Luisa mesh/image caches and share immutable image storage.
- [x] Add retained native state views and indexed C++ command batches.
- [x] Introduce bounded native worker scheduling without background scene access.
- [ ] Move external Python providers and their scene synchronization into this scheduling contract.
- [ ] Complete CPU-only material/shader ownership and public command batch bindings.
- [ ] Enforce independently tested module dependencies.
- [ ] Complete contact-phase, input-latency, and long-running GPU-memory admission.
- [x] Add opt-in SuperDex stage timings and a real-contact-triggered tick capture.

Verified: event connection lifetime; independent project/resource identity;
sensor-preview invalidation (camera-only updates reuse the preview); 112 physics,
worker, and context tests; 14 ScenePlaySession tests; retained NumPy state views
after reset/close. The OpenGL cache test on the RTX 4070 performs 100 vertex
updates without uploading unchanged indices, and removes inactive allocations.
Four Luisa CUDA rendering/cache regressions pass on the same GPU. A 30-minute
residency plateau is not yet verified.

Native worker requests carry compiled value data, not a previously built SDK
instance. Backend construction, Build, Step, Reset, and destruction all occur on
the worker. This is required by Mochi/Marl thread binding; transferring a world
built on the owner thread caused an actual SuperDex shutdown deadlock.
Slow-build/solve (200 ms), nonblocking Stop/Reset, epoch rejection, repeated Play,
valid partial versus invalid state, and native MuJoCo/SuperDex checkpoint replay
have regression coverage. Checkpoint capture includes queued commands accepted
before the next tick. Headless stepping remains synchronous by default.

MuJoCo single, environment batch, and robot batch stepping share native reset
and finite-state checks. Invalid timesteps are rejected before dispatch; batch
worker errors are returned on the owner thread. Scene synchronization does not
erase an already latched physics failure.

Before captures (100 warmup + 500 measured frames, 1920x1080, existing physics):

| Capture | Frame p50 / p95 / p99 (ms) | Physics dispatch p95 (ms) | RTF |
| --- | --- | --- | --- |
| Go1 Play | 16.67 / 17.31 / 17.68 | 4.51 | 0.995 |
| Conveyor Play, initial approach | 286.21 / 427.68 / 603.73 | 419.18 | 0.00637 |

Artifacts: `build/benchmarks/before-go1-play-corrected` and
`build/benchmarks/before-conveyor-play`. The initial Go1 idle capture was
3700x2032 and is not an admission baseline. Conveyor contact-phase, rope,
Luisa switching, and 30-minute residency tests remain outstanding.

Valid worker capture: `build/benchmarks/after-conveyor-worker-owned-lifecycle-v2`.
Conveyor initial approach: frame p50/p95/p99 = 16.67/16.88/19.68 ms,
main-thread physics dispatch p95 = 0.074 ms, RTF = 0.00537, process peak VRAM =
42 MiB. World ready at 523 ms after Play request; editor exits normally. This
demonstrates UI decoupling, not faster physical solving or contact acceptance.
The older `after-conveyor-worker-play` and `conveyor-worker-exit-diagnostic`
captures timed out and are not valid baselines. The first owned-lifecycle capture
was rejected because it read world status after SceneTree finalization; reports
now snapshot status before cleanup and reject failed/timed-out process exits.

## Capture protocol

Use a disposable copy of the project configuration when running automated editor
captures: editor camera-state persistence must not change the authored baseline.
Record the git and SDK commits, CMake build configuration, driver, resolution,
scene, renderer, physics settings, scripted actions, and machine load with results.

The editor accepts `--benchmark-output PATH`, `--benchmark-warmup 100`,
`--benchmark-frames 500`, and optionally `--benchmark-play`. It starts sampling
after the configured main scene has loaded and exits after the bounded capture.
Warmup samples are retained but excluded by `benchmark/editor_runtime_report.py`.
The first scene load, Play call, and actual backend-ready latency are reported
separately. Physics settings use the engine's reflected serializer. Native worker
physics timings are recorded once per completed tick, not once per displayed
frame. Synchronous multi-tick dispatches explicitly report last-tick timing
coverage. Dispatch/process/draw are CPU wall times, not GPU kernel times. Frame
intervals include event/presentation cadence. RTF must accompany FPS.

Capture Go1 idle and Play, conveyor and rope contact phases, and Luisa scene
switching. Record per-process GPU memory using NVML (including graphics processes)
alongside device-wide usage, never substitute Torch's allocator statistics for
total residency. Missing backends or telemetry are unavailable, not a pass.

Admission targets: Go1 60 FPS, complex-scene UI p95 <= 33.3 ms/p99 <= 50 ms,
input acknowledgement <= 100 ms, standard-scene peak VRAM <= 10 GiB. Repeated
scene/resource churn must plateau at live/in-flight resource residency over 30
minutes. A 200 ms injected physics tick must not stall the editor; fixed dt and
solver quality must not be reduced to satisfy that test.

## Remaining boundaries

External provider callbacks still run synchronously. Moving them requires an
explicit solver-only callback, GIL release and device-context handling, plus
main-thread policy and scene-sync callbacks. Existing rope strategy, rewards,
grip metrics and debug policy stay in the example. Do not move NodeScript
callbacks or scene handles onto the worker to claim asynchronous support.

SceneTree no longer reaches a global SimulationServer, and EngineContext no
longer includes PythonScriptRunner. Include-boundary regressions cover those
changes, but OBJECT targets are not independent library boundaries. Splitting
core/scene_io/scene/simulation/render/app/python/editor into independently linked
targets remains work, including importer render-server gates and shader/material
GPU ownership. The full implementation plan is not yet complete.

## Final checkpoint, 2026-09-06

Final Release captures on NVIDIA 580.173.02, RTX 4070, 1920x1080, 100 warmup
render frames plus 500 measured render frames:

| Scene | Frame p50 / p95 / p99 (ms) | Physics step p50 / p95 (ms) | RTF | Process VRAM peak |
| --- | --- | --- | --- | --- |
| Go1 idle | 16.66 / 16.89 / 16.94 | not running | n/a | 51 MiB |
| Go1 Play | 16.66 / 17.11 / 17.75 | 0.558 / 0.720 | 0.9955 | 45 MiB |
| Conveyor Play, initial approach | 16.68 / 16.89 / 16.98 | 246.47 / 864.45 | 0.00518 | 42 MiB |

Artifacts: `build/benchmarks/after-go1-idle-final`,
`build/benchmarks/after-go1-worker-final`,
`build/benchmarks/after-conveyor-worker-final`. Each contains `frames.json`,
`metadata.json`, `summary.json`, and `editor.log`. The source is an uncommitted
worktree based on `f8c9503ebe861518e9b0bca8908489e0ff45864c`; metadata records
worktree status, SDK commits and CMake options, not a clean release commit.

Go1 records 1667 measured physics ticks; the conveyor records only 22 because
the CPU solve is slow. Conveyor solve p50/p95 = 245.39/863.48 ms. These short
startup samples are not the planned 100-physics-tick warmup + 500-tick contact
benchmark. GPU memory covers the process, while render-cache counters exclude
debug uploads and backend acceleration structures. None of these captures is
a long-running peak-memory or contact-quality acceptance result.

The final native worker/server regression rerun passes 63 tests. The broader
physics/context suite passes 112 tests and ScenePlaySession passes 14 tests.
The Python state-view, conveyor and benchmark-report regressions pass 20 tests;
event-connection and debug-preview regressions pass 7 tests. On the RTX 4070,
all 3 OpenGL resource-lifetime tests and all 6 Luisa module/GPU tests pass,
including deforming-geometry reuse and retired-scene resource reclamation.
These are focused lifetime checks, not the outstanding 30-minute churn test.
GCC 12.3 with SuperDex, MuJoCo, Luisa, libuipc, Assimp, OpenUSD and EGL disabled
builds successfully in `build/runtime-minimal`; 18 generic tests pass and the
2 real-SDK tests are explicitly skipped. Existing editor build options and
example solver settings were not reduced. All benchmark editor processes exit
normally; no editor is intentionally left running.

## SuperDex diagnostics follow-up

See `doc/superdex_performance_diagnostics.md` for the stage API, build commands,
timing nesting/parallelism rules and contact-capture results. The unchanged
interactive conveyor reached bilateral contact at tick 847 but diverged on tick
942, before finishing its 100-tick contact warmup. Its 95 completed contact ticks
show step p50/p95 1106.50/1659.66 ms; assembly (including collision) accounts for
about 70% of SDK elapsed time and linear solving about 30%. This is diagnostic
evidence, not a completed 500-tick or grasp-quality admission pass. Python contact
dictionary conversion also remains expensive despite the retained vertex views.

## Static force rejection and editor recheck

The conveyor's `warehouse_frame::frame` reproduces the reported static-link
failure: a force command was accepted and only rejected by the next SuperDex
step. Runtime viewport clicks can issue the same invalid spring command.
Force-target validation now rejects static or missing links before changing
pending forces, including before an asynchronous batch is queued. Fixed children
of movable links remain eligible. The viewport only activates dragging after a
successful command. Backend fallback errors now include the target name.

GCC builds and 96 C++ / 29 Python and architecture tests pass. The installed
`uv run` package was rebuilt. A real conveyor headless check rejects the static
command immediately and still completes the next tick. Tests also cover dynamic
rigid forces on both native workers and preservation of rigid-body properties
through scene packing.

Automatic editor Play captures, 100 warmup render frames plus 500 measured render
frames, both exit normally without simulation errors:

| Mode | Physics samples | Step p50 / p95 (ms) | Frame p50 / p95 (ms) |
| --- | ---: | ---: | ---: |
| Full validation | 23 | 252.27 / 835.15 | 16.67 / 16.90 |
| Explicit preview | 81 | 67.30 / 146.70 | 16.68 / 27.98 |

Artifacts are `build/benchmarks/conveyor-force-fix-validation-repeat-20260906`
and `build/benchmarks/conveyor-force-fix-preview-20260906`. No build or regression
job ran concurrently with those two captures. The earlier 180-frame validation
smoke overlapped Python tests and is not used for timing conclusions.

Full mode retains 6,292 nodes, self-contact, Newton 16 / line search 8 and contact
force output. Preview has 780 nodes, no shell self-contact, Newton 16 / line
search 6, no nodal contact-force output, and skips the initial settling commands.
Both have stage profiling disabled. These are different physical configurations
and short startup windows, not matched contact-phase speedup measurements.
The full-mode timing is close to the previous 246.47 ms startup sample; it remains
far from real-time. Preview is a reduced-quality inspection option, not a fix for
full-resolution SDK solve cost or a validated grasp trajectory.
