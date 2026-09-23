# MuJoCo CPU batch runtime

Gobot uses the native core of [mjbatch](https://github.com/kevinzakka/mjbatch)
for CPU simulation batches. The upstream source is pinned to
`2f169146b643adda65e195d8ad13d997abdd2a2e` in `3rdparty/mjbatch`.
Gobot continues to compile and run against MuJoCo **3.12.0**; it does not
install mjbatch's Python wheel or change the Warp/Newton dependency stack.

Initialize the dependency with:

```sh
git submodule update --init 3rdparty/mjbatch
```

`GOB_BUILD_MUJOCO=ON` builds the native adaptation automatically. CMake copies
the upstream headers into the build directory and applies
`cmake/patches/mjbatch-native.patch` using the system `patch` utility. The
submodule stays unmodified. With `GOB_BUILD_MUJOCO=OFF`, neither mjbatch nor
the patch utility is required. The Python build backend bootstraps the pinned
submodule and the wheel includes the upstream Apache-2.0 license.

## Runtime ownership

The scene compiler supplies one template model. Each active worker owns one
model/data workspace; each environment retains its integration state, warnings,
model overrides, required derived fields and contact results. Solver arenas and
mesh assets are not duplicated for every environment. Worker changes preserve
environment state. `sim_workers=0` still selects the number of logical CPUs,
clamped to the environment count.

The adaptation removes Python/nanobind from the batch core and adds scoped
native visits. Gobot keeps control, sensors, named scene bindings and result
conversion in its MuJoCo backend. Software controllers and spring forces run
every physics tick. Derived observations keep MuJoCo's existing `mj_step`
timing; loading another environment does not add an extra `mj_forward`.
Contacts and their forces are captured before a worker's constraint arena is
reused. Mass/center-of-mass changes use mjbatch's `set_const` field-discovery
algorithm while preserving the integration state.

Existing Gobot batch, reset, checkpoint and Python training interfaces remain
the entry points. Checkpoints retain environment parameters and derived state
as a private, versioned backend payload. Native worker errors include the
environment index. Reset is required after an invalid physics step. MuJoCo
sleep is rejected because its bookkeeping is not part of `mjSTATE_INTEGRATION`.

## Updating the dependency

Keep the upstream commit and adaptation patch together. Apply and build the
patch from a clean source checkout, then run `test_mjbatch_core`, the Gobot
physics/simulation/controller tests and the Python Go1 CPU tests. The native
core test compares shared workers with independent MuJoCo models and data,
including contacts, randomization and worker-count changes.

Use `benchmark/go1_velocity_benchmark.py --backend mujoco-cpu` to compare
throughput, and `/usr/bin/time -v` to measure peak resident memory. Keep the
environment count, worker count, seed, actions and warmup/measurement steps
identical between versions. Memory for integration states, observations and
contacts still scales with the number of environments.

## Validation and measurements

The Release build passed 118 CTest entries: five native mjbatch tests, the
physics, simulation, joint-controller and locomotion-batch suites, and the
Python build-backend, robot-batch-view and velocity-environment scripts.
Native tests cover independent MuJoCo parity (including IMU, rotations and
contact forces), changing worker counts, parameter randomization, checkpoint
replay, partial reset and recovery after worker errors. A separate
`GOB_BUILD_MUJOCO=OFF` configuration also compiled the MuJoCo backend object.

The following local Go1 measurements used random actions, seed 42, observation
noise disabled, five warmup steps and 30 measured steps. `auto` resolved to
16 workers, capped by the environment count. Peak RSS includes the whole
Python process, including Torch. Initialization was measured separately with
zero warmup steps and one step, using the benchmark's `initialization_ms` field.

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

These are single-run, indicative comparisons. The baseline was the previously
installed Release package with OpenUSD enabled; the new local Release build
has OpenUSD disabled. Both use MuJoCo 3.12 and the same task settings. At
1024 environments with automatic workers, observed peak memory fell by about
63%, initialization took about 24% less time, and throughput was about 4%
lower. This change primarily reduces batch memory; it does not establish a
general throughput improvement.

Local raw results are in `build/mjbatch-validation/before-*.json` and
`native-*.json`, with `/usr/bin/time -v` output in the corresponding logs.
CTest uses the new module in `build/python`. An existing installed Gobot
package must be rebuilt/reinstalled to use the new backend from ordinary
Python invocations.
