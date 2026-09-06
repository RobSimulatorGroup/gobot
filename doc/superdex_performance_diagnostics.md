# SuperDex stage profiling

This diagnostic path measures the unchanged native conveyor scene, not the
reduced editor preview. It does not enable CUDA, change solver tolerances,
disable self-contact, or alter trajectories. Headless stepping uses the normal
`SimulationServer` and the example's existing joint targets and external forces.

## Build and API

The SDK fork supplies the optional C++ utility `mochi::SetStepProfilingEnabled`
and `mochi::GetStepProfile` in `mochi_physics/utils/step_profiling.h`. It is
independent of generated scene APIs and checkpoint serialization. Counters belong
to a scene, not a process or thread-local collector; parallel task contributions
are atomic, and they are reset before each SDK step, including `Step(0)`.

Build/install the modified SDK before building Gobot. For this workspace's
existing standalone SDK installation:

```bash
.venv/bin/cmake --build build/superdex/build --parallel 4
.venv/bin/cmake --install build/superdex/build
uv sync --inexact --reinstall-package gobot --no-build-isolation-package gobot \
  -C cmake.define.GOB_SUPERDEX_ROOT=/home/wqq/gobot/build/superdex/install
```

A clean build can use the normal isolated SDK ExternalProject instead. Older
prebuilt SDKs still build/run without profiling. Explicitly requesting timings
with an SDK that lacks the utility fails with a rebuild instruction; missing
timings are not fabricated as zeros.

Profiling is off by default. On a context used for native SuperDex simulation:

```python
settings = context.get_superdex_solver_settings()
settings["record_solver_timings"] = True
context.set_superdex_solver_settings(settings)
# Build the world if it has not been built yet, then advance one physics tick.
context.step_once()
diagnostics = context.get_solver_diagnostics()
assert diagnostics["timings_available"]
for stage in diagnostics["stage_timings"]:
    print(stage["name"], stage["time_seconds"], stage["calls"], stage["parallel_sum"])
```

Gobot diagnostics aggregate every completed SDK substep. Reset/checkpoint restore
clear timing samples instead of replaying old measurements. Immutable worker
frames retain their own diagnostics after a later tick or world destruction.
`linear_iterations` counts iterative linear-solver iterations summed across
islands/substeps; a direct solver can legitimately report zero iterations.
Existing Newton/line-search counters retain their SDK maximum-over-islands
semantics, summed across Gobot substeps.
The line-search counter accumulates trials over Newton iterations; it can exceed
the configured per-search limit of eight without violating that limit.

## Timing interpretation

All times are inclusive elapsed seconds, not CPU utilization or CUDA kernel
times. There are two different aggregation domains:

| Domain | Stages | Interpretation |
| --- | --- | --- |
| Wall time | `apply_forces`, `state_sync`, `sdk_pre_step`, `sdk_islands`, `sdk_post_step` | Sequential portions of the backend step |
| Accumulated scopes | `island_prepare`, `island_newton`, `island_queries` | Sum across potentially concurrent islands |
| Accumulated nested scopes | `assembly`, `collision_detection`, `contact_jacobians`, `linear_setup`, `linear_solve`, `line_search` | Sum across potentially concurrent islands/tasks, with nesting |

- `solve_time_seconds` is the entire SDK ECS physics pipeline, not linear solve.
- `assembly` includes collision/contact updates and residual/Hessian assembly.
  Body assembly may run concurrently with collision detection.
- `line_search` includes trial-state updates, residual/energy assembly, and
  rollback after unsuccessful trials. It overlaps the reported assembly time.
- `collision_detection` measures the assembly-time detection pipeline, including
  its waits. Conservative pre-step broad-phase work belongs to `sdk_pre_step`.
- `contact_jacobians` includes waiting for articulated Jacobian tasks.
- `linear_solve` includes preconditioner setup/factorization and solving.
- Accumulated scopes can exceed wall time. Never add the entire timing table or
  subtract overlapping scopes to claim exclusive CPU percentages.
- Python command construction, state-view conversion, and observed tick elapsed
  time are recorded separately. Trace writing is outside step latency.

The current Gobot adapter calls `CreateContext(0)`, meaning single-threaded SDK
execution. The editor's dedicated physics worker is a separate concern; it
prevents UI blocking but does not parallelize this solver.

## Contact capture

```bash
uv run python benchmark/superdex_contact_benchmark.py \
  --output build/benchmarks/conveyor-contact-cpu \
  --warmup-ticks 100 --measure-ticks 500 \
  --max-ticks 1800 --max-wall-seconds 1800
```

Run without an editor, builds, training jobs, or other captures competing for
CPU time. Use a new output directory for each run. The default target is
`blue_mailer`, requiring contacts with both hands. Other parcel groups and a
single-hand trigger can be selected explicitly. Do not compare different targets,
quality settings, warmup counts, or trigger requirements as equivalent baselines.

The runner advances normally from authored tick zero. The first observed
hand-to-target contact satisfying the trigger starts the warmup. After exactly
100 warmup physics ticks, it measures 500 consecutive ticks. Losing contact does
not restart the window or discard expensive frames. Reports include contact
coverage and separate contact-present/contact-lost distributions. They also
retain per-phase startup/approach data. This is not a benchmark of 500 guaranteed
grasped frames or proof of successful manipulation.

Artifacts:

- `ticks.jsonl`: metadata followed by one flushed record per completed physics
  tick, including stage durations/calls, convergence, contact state and penetration.
  A failed SDK step is recorded separately before world cleanup; non-finite
  diagnostics are preserved as strings, not coerced to valid numeric samples.
- `summary.json`: p50, nearest-rank p95, maxima and totals for the entire run,
  precontact, warmup, measurement, contact-only/lost subsets and authored phases.
  Includes example metrics, reset error, source hashes, revisions, CPU/GPU and
  requested solver configuration.

Missing stage telemetry, invalid state/time progression, or divergence is an
error. Not reaching contact or completing the window returns a nonzero exit code
and preserves partial data. A window with contact only during warmup also fails.
The wall-time guard is checked between SDK steps;
it cannot preempt a hung SDK call. A completed performance capture is not a
contact-quality acceptance result. Penetration is SDK-reported contact distance,
not an independent geometric intersection measurement.

## Optimization order

Use actual contact samples to choose the next change. Large linear time calls for
separating preconditioner setup from iterations before attempting CUDA. Large
assembly/collision time calls for examining repeated line-search assemblies,
contact search/update work and safe reuse of structural data. Change one mechanism
at a time and repeat both correctness and fixed-configuration timing tests.

Provider async remains a separate follow-up: isolate solver-only work, transfer
commands/state with explicit ownership, keep Python scene access on the owner
thread, and preserve error/Stop/Reset semantics. It improves responsiveness of
provider-backed examples; it is not an explanation or fix for native SuperDex's
SDK solve cost.

## Initial contact result, 2026-09-06

Machine: Ryzen 7 3700X, RTX 4070 12 GB, driver 580.173.02. Release SDK/Gobot,
CPU single-threaded, full 6,292 deformable vertices, 51 links and 44 joints.
No editor, CUDA, or concurrent compilation was used during the capture.

Artifact: `build/benchmarks/superdex-contact-profile-20260906`.
This is a **failed admission capture**, not a completed 100+500 benchmark:
both hands first contacted the blue bag at tick 847; 95 consecutive contact
ticks were recorded through tick 941, then tick 942 diverged. The original
interactive solver settings (Newton 16, line search 8, 2 ms, one substep)
were unchanged. All 95 completed contact ticks reported `stopped`, not
`converged`. There are zero post-warmup measurement samples.

Exploratory timings from those 95 contact-warmup ticks only:

| Scope | p50 (ms) | p95 (ms) |
| --- | ---: | ---: |
| Normal `step_once` | 1106.50 | 1659.66 |
| SDK ECS physics | 1098.18 | 1650.32 |
| Assembly, including collision | 784.66 | 1196.15 |
| Collision detection, nested in assembly | 506.49 | 809.33 |
| Linear solve, including preconditioner | 311.98 | 477.92 |
| Line search, overlapping assembly | 643.17 | 1044.44 |
| Native state synchronization | 7.30 | 9.53 |
| Two Python state-view reads per tick | 144.04 | 164.58 |

Ratios of summed durations to summed SDK wall time in this single-thread run:
assembly 69.57%, collision 44.18% (part of assembly), linear solve 29.86%.
Line-search trial work overlaps assembly and must not be added to those shares.
Even hypothetically eliminating all linear solve cost with everything else
unchanged gives only about a 1.43x SDK speedup, not a measured CUDA result.

There is also a binding cost outside the SDK. Around tick 719 there were 13,804
directed contact records; two state-view reads took 161.55 ms. The binding still
builds a Python dictionary for every contact, even on the NumPy state-view path.
Candidates are compact contact arrays/queries and reuse of an already retained
completed state, not removing physical contacts from the solver.

The SDK-reported maximum penetration over the 95 contact ticks was 0.380 mm.
That does not qualify this run as physically accepted: the run subsequently
diverged, did not complete the window, and has no independent penetration check.
The failure is retained as evidence; increasing iteration limits is a separate
stability comparison, not a performance improvement to this baseline.

### Accurate stability comparison

Artifact: `build/benchmarks/superdex-contact-profile-accurate-20260906`.
Only the existing quality option changed: Newton's limit was 48 instead of 16;
all other physics/scene settings stayed the same. Bilateral contact again began
at tick 847. It passed the original failure location, completed 100 warmup ticks,
and recorded 244 measured ticks (947-1190) before the 1800-second guard stopped
the capture at a tick boundary, after 1803.80 seconds including cleanup.

It did not diverge during that interval, but all 244 measured ticks still reported
`stopped`, not `converged`. Only 195/244 ticks (79.92%) had bilateral hand contact;
all 49 contact-loss ticks were retained. Reset error was zero. SDK-reported peak
penetration was 0.382 mm, without an independent geometry check.

| Scope | p50 (ms) | p95 (ms) |
| --- | ---: | ---: |
| Normal `step_once` | 3658.19 | 5011.45 |
| SDK ECS physics | 3653.64 | 5004.73 |
| Assembly, including collision | 2719.68 | 3858.37 |
| Collision detection, nested in assembly | 1988.07 | 2833.78 |
| Linear solve, including preconditioner | 834.82 | 1422.32 |
| Two Python state-view reads | 62.35 | 119.74 |

The smaller state-view cost is from a different physical/contact interval, not
a binding optimization. The capture remains incomplete (244/500 ticks), so the
48-iteration configuration is not promoted to a default or a stability pass.
Both traces and partial summaries are retained, with distinct failure versus
timeout status. These results prioritize nonlinear/contact stability, then
assembly/collision work and contact-state transport, ahead of a CUDA-only bet.

## Implementation verification

- Rebuilt and installed the GCC 12.3 CPU SDK and the editable Gobot package.
- SuperDex-enabled C++ regression: 38 tests passed across the backend, step
  failure contract, and simulation worker, including retained diagnostics.
- Python regression: 52 tests passed across the capture protocol, state frames,
  conveyor example, build backend, and editor runtime report.
- SDK-disabled minimal build: 15 generic simulation tests passed; the MuJoCo and
  SuperDex worker integration tests were explicitly skipped.
- A one-tick wall-time smoke capture retained its partial trace, reported an
  incomplete window with exit code 2, and reset with zero state error.

These checks validate the diagnostics and failure-reporting path. They do not
replace the still-incomplete 100+500 contact capture or certify grasp stability.
