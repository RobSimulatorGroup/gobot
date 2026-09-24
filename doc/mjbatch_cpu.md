# MuJoCo CPU batches with mjbatch

Gobot uses the native core of [mjbatch](https://github.com/kevinzakka/mjbatch),
pinned at `2f169146b643adda65e195d8ad13d997abdd2a2e` in `3rdparty/mjbatch`,
with MuJoCo 3.12.0. Existing Gobot batch, reset, checkpoint, and training APIs
remain the entry points. The old Gobot CPU worker implementation is removed;
MuJoCo Warp remains the CUDA provider.

## Build and ownership

Python builds initialize the dependency automatically. For standalone CMake:

```sh
git submodule update --init 3rdparty/mjbatch
```

`GOB_BUILD_MUJOCO=ON` copies upstream headers into the build tree and applies
[`mjbatch-native.patch`](../cmake/patches/mjbatch-native.patch) with the system
`patch` utility. The submodule stays unchanged. No mjbatch Python/nanobind
package is required; MuJoCo-disabled builds need neither mjbatch nor `patch`.

- Each worker owns a model/data workspace; each environment stores integration
  state, warnings, model overrides, derived fields, and contact results.
  Solver arenas and mesh assets are shared across environments using a worker.
- Worker-count changes preserve state. `sim_workers=0` uses logical CPU count,
  capped by environment count.
- Controllers and spring forces run each physics tick. Derived observations
  preserve `mj_step` timing without an extra `mj_forward`. Contact forces are
  captured before workspace reuse; mass/COM updates preserve integration state
  through `set_const` field discovery.
- Checkpoints include parameters and derived state in a private versioned
  payload. Worker errors identify the environment; invalid steps require reset.
  MuJoCo sleep is unsupported because `mjSTATE_INTEGRATION` omits its bookkeeping.

## Maintenance and validation

Update the upstream commit and adaptation patch together. Build from a clean
checkout, then run `test_mjbatch_core`, physics/simulation/controller tests, and
Go1 CPU tests. Coverage includes independent MuJoCo parity, sensors and contact
forces, worker-count changes, randomization, checkpoint replay, partial resets,
and error recovery. Also check a `GOB_BUILD_MUJOCO=OFF` build.

Use `benchmark/go1_velocity_benchmark.py --backend mujoco-cpu` and
`/usr/bin/time -v` with matching environment/worker counts, seed, actions, and
warmup. See [Go1 benchmark commands](../examples/go1/README.md#benchmark-and-parity).

## Recorded comparison

These local runs used random actions, seed 42, no observation noise, 5 warmup
and 30 measured steps. `auto` resolved to 16 workers, capped by environment
count. RSS includes Python and Torch; initialization was measured separately
with no warmup and one step.

| Environments | Workers | Old env steps/s | mjbatch env steps/s | Old peak MiB | mjbatch peak MiB | Old init ms | mjbatch init ms |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 384 | 395 | 727 | 719 | 656 | 650 |
| 1 | auto | 360 | 389 | 739 | 719 | 675 | 649 |
| 32 | 1 | 633 | 633 | 780 | 758 | 735 | 737 |
| 32 | auto | 3281 | 3381 | 781 | 733 | 736 | 718 |
| 256 | 1 | 606 | 608 | 1217 | 894 | 1339 | 1133 |
| 256 | auto | 2964 | 2923 | 1215 | 804 | 1365 | 1152 |
| 1024 | 1 | 587 | 584 | 2591 | 1285 | 3358 | 2590 |
| 1024 | auto | 2674 | 2570 | 2591 | 955 | 3357 | 2568 |

These are single-run comparisons: the old installed Release build had OpenUSD
enabled; the new Release build had it disabled. Both used MuJoCo 3.12 and the
same task settings. At 1024 environments with automatic workers, peak memory
fell about 63%, initialization time fell 24%, and throughput fell 4%. The
observed benefit is memory use, with similar throughput. Per-environment state,
observations, and contacts still scale with batch size.
