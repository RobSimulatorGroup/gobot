# Physics runtime follow-up

This iteration implements benchmark failure reporting, opt-in IPC stage
profiling, and asynchronous dual-FR3 rope Play. It does not admit a successful
bag/carton flip, independent composite reset, or production joint-space coupling.

## Editor benchmark correctness

The native executable and Python-launched editor now return exit code 2 when a
capture fails. JSON retains `status`, `failure_stage`, `error`, `exit_code`
and completed physics ticks in the measurement window. Starting Play can fail
before a physics world exists; this is distinct from `SimulationServer.faulted`.
Play script errors after startup, physics faults, incomplete frame windows,
missing worlds, and zero completed physics ticks cannot produce successful
captures. The report summarizer also rejects a stationary simulation clock.

`--benchmark-timeout SECONDS` defaults to 600 seconds. The deadline is checked
at editor iteration boundaries; it is not a preemptive cancellation mechanism
for a blocked synchronous SDK call or process shutdown. External capture
supervision should retain its process timeout.

## IPC stage profiling

`LibuipcBatchConfig(enable_stage_profiling=True)` enables synchronized SDK
timers only around native batch steps. Default execution leaves profiling off.
The SDK timer stack, enable state and synchronization callback are thread-local,
so one worker's diagnostics cannot consume another worker's scopes.
Batch module ABI is now 7; rebuild the native solver and Python extension
together. Old modules fail the version check.

Native diagnostics expose `last_stage_profile_json`. The composite provider
retains each coupling attempt under `ipc_stage_profiles`, including failed
attempts before rollback. Timings are hierarchical and inclusive, in seconds.
The root is an untimed grouping node. Named scopes cover contact candidate
search/filtering, system assembly, preconditioner assembly, PCG, and line search.
Do not sum children and parents. CUDA graph regions may appear as aggregate
scopes rather than per-kernel detail.

The contact benchmark converts timed scopes to milliseconds, sums repeated
coupling attempts at the same path, retains raw profiles and labels the run
`synchronized_stage_profile`. Such runs intentionally omit throughput rates:
the instrumentation synchronizes CUDA and changes execution overhead.
Missing/invalid profiling records fail the diagnostic capture.

Run normal latency/VRAM measurement first, in fresh sequential processes:

```bash
uv run --no-sync python benchmark/conveyor_ipc_batch.py \
  --counts 1 --repeats 3 --warmup-steps 800 --steps 100 \
  --report build/benchmarks/ipc-followup/throughput.json
uv run --no-sync python benchmark/conveyor_ipc_batch.py \
  --counts 1 --profile-stages --warmup-steps 800 --steps 100 \
  --report build/benchmarks/ipc-followup/stages.json
```

The same CLI supports `--counts 1 2 4 8`. Reports record dependency versions,
driver, scene/artifact identity, source revisions/diff hashes, synchronized
step latency, contact coverage and sampled per-process GPU memory. Physical
parameters, convergence limits, trajectory and asset geometry remain unchanged.
These windows measure contact acquisition, not successful flips.

## Rope worker ownership

`rope_twist_play.py` owns scene bindings, input and debug display.
`rope_twist_config.py` holds immutable configuration and sensor specifications.
`rope_twist_runtime.py` creates the SDK provider, controller and output exporter
on the physics worker. Only compiled data and settings cross the factory boundary.

The existing controller, gravity compensation, torque limits, timestep and
quality-specific solver settings are retained. Display snapshots are limited
to 60 Hz and environment 0. Robot poses and authored-local soft vertices use
the common scene snapshot service. Winding/slip/reaction metrics are compact
numeric fields. Contact output is subscribed only while contact arrows are
enabled; disabling it clears contact arrows immediately. The owner thread
never synchronizes CUDA or reads a provider tensor.

Reset restores the provider and controller on the worker. Automatic cycle
reset uses the controller's completed flag, including early stall completion.
Stop retires the worker without waiting for a solve on the UI thread.
The default installed native solver is used unless explicitly overridden by
`GOBOT_LIBUIPC_SOLVER_MODULE`; unrelated build directories are not searched.

## Validation

- Rebuilt the Release editor, Python extension, libuipc SDK and solver module;
  installed the Python component into the active editable environment.
- 68 Python tests passed across rope assets/control, coupling, snapshot/runtime
  contracts and benchmark reporting. The legacy scene fixture comparison now
  recognizes omitted volumetric defaults and float32 transform decomposition
  roundoff; geometry and materials remain exact and the authored scene is unchanged.
- 14 C++ IPC/session/worker contract tests passed.
- `gobot_ipc_timer_test` passed overlapping worker timer scopes, disabled
  instrumentation and exception cleanup without a CUDA device.
- Three real editor tests passed startup failure, runtime script failure and
  timeout, each with exit code 2 and the expected structured error.
- The real rope GPU worker test passed stepping, reset-to-initial-state,
  finite selected snapshots, contact subscription changes and shutdown.
- RTX 4070 editor smoke: 90 measured frames after 10 warmup frames, 63 physical
  ticks in the measurement window, `scheduling=worker`, ready and no error.
  Play returned in 419 ms; initial worker readiness took 124.5 s on that run.
  This is a startup/short-run check, not a full rope-twist physical acceptance.

Reproduce the optional GPU and editor checks:

```bash
GOBOT_RUN_WARP_IPC_GPU_TEST=1 uv run --no-sync --with pytest python -m pytest tests/python/test_rope_async_runtime.py -q
GOBOT_RUN_RENDER_GPU_TEST=1 uv run --no-sync --with pytest python -m pytest tests/python/test_editor_benchmark_failure.py -q
cmake --build build/cp313-cp313-linux_x86_64 --target gobot_ipc_timer_test
build/cp313-cp313-linux_x86_64/modules/libuipc_solver/gobot_ipc_timer_test
```

## Remaining architecture work

Independent composite environment reset/fault isolation, common compilation
snapshots, expanded revolute-joint coupling, geometric hand/bag intersection
checks and full task-success throughput remain separate subsequent work.
The composite provider still advertises full-batch reset and no persistent
user checkpoint or full-step CUDA Graph.

## RTX 4070 contact capture, 2026-09-22

Artifacts: `build/benchmarks/physics-refactor-20260922/contact-throughput.json`
and `contact-stages.json`, with individual process logs and JSON beside them.
Driver 580.178.04, MuJoCo/Warp 3.12.0, Warp 1.17.0, Torch 2.11.0+cu128.
No other compute processes were observed by the memory sampler.

All three uninstrumented one-environment trials failed strict Newton convergence
before completing the 800-step warmup:

| Trial | Completed steps | Failed step | Newton limit | Sampled process peak |
| --- | ---: | ---: | ---: | ---: |
| 1 | 731 | 732 | 48 | 1.234 GiB |
| 2 | 742 | 743 | 48 | 1.236 GiB |
| 3 | 737 | 738 | 48 | 1.236 GiB |

There are zero measured contact-window samples and therefore no accepted
median/p95 throughput baseline. The failures are not out-of-memory errors.
The scene hash and numerical solver configuration match the older benchmark
(excluding temporary workspace paths and the new disabled profiling flag),
but the compiled artifact digest differs. No speedup/regression ratio against
the historical 3.10 measurements is justified by these incomplete runs.

The separate synchronized profile completed 734 steps and failed at step 735.
Its last 16 successful warmup profiles (ticks 719–734) and the failed coupling
attempt are retained. The failed attempt had 48 Newton iterations and 5,170
PCG iterations; the last relative linear residual was 0.000993, with linear
convergence reported true and nonlinear convergence false.

| Failed-attempt scope | Inclusive time |
| --- | ---: |
| SDK pipeline | 1,101.59 ms |
| DCD candidate search, initial plus Newton iterations | 270.19 ms |
| Line search | 286.89 ms |
| Dynamic contact effect calculation | 180.55 ms |
| Linear system assembly | 136.15 ms |
| Linear solve | 225.04 ms |

Trajectory candidate search (228.50 ms) is already inside line search.
Preconditioner assembly (12.51 ms) is already inside linear system assembly.
These are one failed, instrumented attempt, not phase-average performance
claims. They prioritize nonlinear/contact convergence and repeated contact
work; linear acceleration alone would leave most of this attempt's cost.
Do not increase iteration limits or relax strict convergence solely to convert
this capture into a passing benchmark. Next investigate contact transitions,
nonlinear residuals/step lengths and repeated collision/assembly work under
the same authored scene before scaling environment count.
