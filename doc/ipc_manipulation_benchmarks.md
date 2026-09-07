# Manipulation coupling and batch experiments

These headless tests leave the editor's SuperDex playback and authored
`conveyor_packages.jscn` unchanged. A finite solver state or a completed timing
window is not a successful bag/carton flip. Full-batch reset remains the only
reset mode for the MuJoCo Warp + libuipc composite provider.

## Continuous grip feedback

```bash
uv run --no-sync python examples/conveyor_packages/conveyor_mujoco_ipc.py \
  --grip-feedback continuous --grasp-wait-steps 500 \
  --report build/benchmarks/conveyor-ipc-acceptance/continuous.json
uv run --no-sync python examples/conveyor_packages/conveyor_mujoco_ipc.py \
  --grip-feedback fixed --grasp-wait-steps 500 \
  --report build/benchmarks/conveyor-ipc-acceptance/fixed.json
```

Both modes use the same scene, fitting, dt, material, coupling iterations and
acceptance gates. Independent feedback starts after grasp confirmation; it
never writes bag vertices, body poses or package forces. Compare repeated
runs, not just first-contact times from two nondeterministic GPU trajectories.
The [acceptance document](conveyor_ipc_acceptance.md) describes limits and the
current failed lift result.

## Joint-space coupling A/B

Build the explicitly requested, non-installed experiment target in a configured
CUDA/libuipc build (adjust the build directory for another Python ABI):

```bash
uv run --no-sync cmake --build build/cp313-cp313-linux_x86_64 \
  --target gobot_ipc_articulation_probe -j2
uv run --no-sync python benchmark/ipc_articulation_ab.py --mode joint --no-contact \
  --report build/benchmarks/ipc-articulation/joint-free.json
uv run --no-sync python benchmark/ipc_articulation_ab.py --mode proxy --no-contact \
  --report build/benchmarks/ipc-articulation/proxy-free.json
uv run --no-sync python benchmark/ipc_articulation_ab.py --mode joint \
  --report build/benchmarks/ipc-articulation/joint-contact.json
uv run --no-sync python benchmark/ipc_articulation_ab.py --mode proxy \
  --report build/benchmarks/ipc-articulation/proxy-contact.json
```

The fixture is a temporary Gobot SceneTree compiled through the ordinary
artifact services, not a MuJoCo or IPC asset maintained separately. Two serial
prismatic joints have constant non-diagonal inertia `[[3,2],[2,2]]` kg. The
same authored box meshes, masses and tetrahedral block feed both experiments.
The contact-free case moves the block sideways on its support, away from the
press; ground contact remains active.

- `proxy`: the existing two-iteration `SolverCoupledProxy`, exact native
  link wrenches and ordinary MuJoCo Warp stepping.
- `joint`: MuJoCo Warp predicts the free increment. The constant inertia is
  expanded from the same compiled MuJoCo model. IPC receives that matrix and
  increment through `ExternalArticulationConstraint`. Both joint position and
  velocity are corrected from the solved increment, followed by FK, without
  resetting either world per tick. Joint/DOF mappings are resolved by name.

This is a host-staged, single-environment scalar-joint prototype, not a new
engine backend or public provider. It deliberately excludes floating roots,
revolute joints, constraints/limits active during contact, and full LEAP assets.
It cannot certify their dynamics or batch speed. The current joint limits are
checked after correction and fail closed, not clamped silently.

Reports include state, soft height, interface translation error, generalized
reactions and timing. Joint reaction is inferred from increment correction and
therefore includes solver residual; proxy reaction is `J^T` times the exact
contact wrench. Timing includes host transfer/JSON/FK and is diagnostic only.
No-hand-contact increments must agree within 10 micrometers. A contact test
alone cannot admit a route that fails this check.

The initial test exposed an external-articulation Hessian assembly error in
the SDK: local triangular writes omitted contributions when two joints had
different body sets, including a shared link. Assembly now retains the global
upper triangle from all four body blocks for every ordered mass-matrix entry.
The full symmetric mass matrix already supplies the opposite joint pair, so
lower-triangle blocks must not be mirrored and counted again. This only changes
the external-articulation constraint, not the ordinary proxy constraint.

### Measured joint comparison (2026-09-07)

All four 400-step cases passed their fixture checks on the RTX 4070. At the
same `1e-3` linear tolerance, the joint route's no-hand-contact maximum increment
error fell from `1.12409e-4 m` before the fix to `6.10151e-8 m` afterward. The
GPU regression also exercises a non-diagonal SPD matrix, opposite-signed
increments and rejection of indefinite, asymmetric and non-finite matrices.

| Contact case | Joint constraint | Two-iteration proxy |
| --- | ---: | ---: |
| Final carrier displacement (mm) | -26.0806 | -26.0873 |
| Final pad-joint displacement (mm) | -24.1211 | -24.1282 |
| Final soft-block height (mm) | 149.5883 | 149.5805 |
| Maximum interface translation error (mm) | 0.001526 | 0.001507 |
| Final generalized reaction norm (N) | 104.6395 | 104.6422 |

The final joint coordinates agree within `7.1e-6 m`; block heights agree within
`7.9e-6 m`. These are first-run diagnostic comparisons, not a full articulation
accuracy or speed admission. The joint prototype still uses host staging and
has not been connected to the LEAP controller. Neither full bag nor carton
flipping is admitted by these two-slider tests.

Local reports under `build/benchmarks/ipc-articulation/` are
`joint-free-corrected.json`, `proxy-free.json`, `joint-contact-corrected.json`
and `proxy-contact-corrected.json`. Earlier reports without `corrected` in the
joint filename predate the SDK fix and must not be used as passing evidence.

## Contact-stage batch baseline

```bash
uv run --no-sync python benchmark/conveyor_ipc_batch.py --counts 1 2 4 8 \
  --warmup-steps 800 --steps 100 \
  --report build/benchmarks/conveyor-ipc-batch/contact.json
```

Each size runs in a fresh process, sequentially, with both LEAP hands, the
blue shell and tetrahedral fill: 44 joints, 2600 soft vertices and 36 coupled
links per environment. Each environment has its own controller state. No
mesh/material simplification, dt increase, reduced convergence limit, or
cross-environment contact is introduced. The one-shard batch uses 2 ms dt,
two coupling iterations, Newton 48, line search 16 and strict convergence.

The first 800 steps are real settling/approach/acquisition, excluded from the
timing window. The following 100 contact-acquisition steps are measured, with
per-environment contact coverage and proxy error gates. They are not a lift or
flip workload, and may precede the continuous-feedback arming point. No masked
reset or successful batched grasp is implied.
Contact coverage requires at least one hand link's reaction to exceed 0.01 N
in each measured step; bilateral opposed pinch is reported separately and is
not the contact-stage benchmark's pass criterion.

`provider_step_ms` measures synchronized coupled stepping. End-to-end latency
also includes controller updates, command transfer, full state readback and
finite-state checks. `environment_steps_per_second` uses the sum of those
measured durations, not the reciprocal of median latency. This Python/readback
path is a diagnostic baseline, not the final device-resident training loop.

Memory uses sampled NVML compute-process accounting via `nvidia-smi`, matched
by PID and normalized GPU UUID. It includes the CUDA context, Warp, IPC and
Torch allocations. A 1 s sampler spans build, warmup and measurement; it may
miss intra-step allocation peaks. Other compute PIDs are recorded. Missing or
unsupported accounting is an error, not zero bytes. Do not estimate batch
capacity by dividing total card memory by one environment's allocation.

### Measured batch baseline (2026-09-07)

RTX 4070, NVIDIA driver 580.173.02, Release build, 800 warmup steps followed by
100 measured contact-acquisition steps. The sampler recorded no other compute
PIDs. These are individual runs, not repeated statistical estimates; GPU
trajectories and iteration counts are not bitwise deterministic. The editor
was not part of the measured workload.

| Environments | Provider median / p95 (ms) | End-to-end environment steps/s | Sampled process peak (GiB) | Result |
| ---: | ---: | ---: | ---: | --- |
| 1 | 322.4 / 578.7 | 3.127 | 2.611 | 100 contact steps |
| 2 | 535.0 / 914.2 | 4.006 | 2.721 | 100 contact steps |
| 4 | 989.5 / 1742.0 | 4.393 | 2.934 | 100 contact steps |
| 8 | unavailable | unavailable | 3.359 | Newton failure at frame 812 |

All successful windows had 100% per-environment contact coverage, zero
confirmed bilateral-pinch coverage and maximum proxy surface error below
0.212 mm. They measure contact acquisition, not grasp success. Local reports
are `build/benchmarks/conveyor-ipc-batch/contact-memory-fixed-{1,2}.json` and
`build/benchmarks/conveyor-ipc-batch/contact-1-2-4-8-{4,8}.json`; the 8-environment
`.log` records the failed Newton frame. The historical failed report lacks
structured final diagnostics; future failures now retain those diagnostics
and completed-step counts where available.

The 8-environment trial failed strict Newton convergence at frame 812 after
reaching the 48-iteration limit. Its sampled process peak was 3.359 GiB and
there was no out-of-memory error. An incomplete window does not produce a
valid throughput result; increasing the iteration limit to make only this
batch pass would change the comparison.

Initial 1/2-environment reports named `contact-1-2-4-8-1.json` and
`contact-1-2-4-8-2.json` contain no matching process-memory samples because
Torch and NVML use different UUID prefixes. They are superseded by the
`contact-memory-fixed` runs. The sampler now normalizes the prefixes and
rejects missing memory measurements. The 4/8 reports already used the fix.

The last measured 4-environment step spent about 1292 ms in the two IPC
advances and 0.92 ms in rigid advance. This is one diagnostic step, not a
phase-average profile, but it points to contact solving as the immediate
performance problem. The available memory headroom does not establish a
usable training batch size: convergence and completed-task throughput must
also pass. Full LEAP joint coupling, sustained airborne grip, full bag/carton
flipping and independent environment resets remain unvalidated.

CPU tests:

```bash
uv run --no-sync python -m pytest tests/python/test_conveyor_mujoco_ipc.py \
  tests/python/test_ipc_articulation_ab.py \
  tests/python/test_conveyor_ipc_batch_benchmark.py \
  tests/python/test_mujoco_ipc_provider.py -q
GOBOT_RUN_IPC_ARTICULATION_GPU_TEST=1 uv run --no-sync python -m pytest \
  tests/python/test_ipc_articulation_ab.py -q
```
