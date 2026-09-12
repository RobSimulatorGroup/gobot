# Physics dependency update — 2026-09-12

Gobot pins released Python packages and integrates upstream SDK changes into
its existing adaptation branches. SceneTree/.jscn remains the authoring source.

| Dependency | Previous | Updated source |
| --- | --- | --- |
| MuJoCo, including the C++ SDK | 3.10.0 | 3.12.0, Newton-compatible release |
| MuJoCo Warp | 3.10.0.2 | 3.12.0, Newton-compatible release |
| Newton | 1.4.0 | 1.6.0 |
| Warp | 1.15.0 | 1.17.0 |
| warp-nn | 0.3.0 | 0.3.1, required by Newton's ONNX extra |
| libuipc | `a87612b1`, Wuqiqi123 fork | RobSimulatorGroup/libuipc `main` at `14928d40e3a6ef72c23777c5ceb16be2cb8f9ac4`, plus Gobot adaptations |
| SuperDex | `efc3ebb` | RobSimulatorGroup/project_superdex `main` at `3cb546806891fa0d12b781e377c7eb8666e25524`, plus Gobot adaptations |

The integration commits are `e4b20ab` (SuperDex, `gobot-superdex-core`) and
`4bf860a2` (libuipc, `gobot-device-interop`), both published on the requested forks.
The gitlinks in Gobot pin the integration commits. The SDK integration branches
remain `gobot-device-interop` and `gobot-superdex-core`; neither SDK is built
from a floating branch. The SuperDex repository has `main`, not `master`.

## Compatibility decisions

- Gobot requests `newton[sim,onnx]` and pins the shared MuJoCo stack to 3.12.0,
  the latest compatible releases at update time. Newton 1.6.0 requires the 3.12
  series. Testing the newer 3.13 stack failed GPU kernel compilation because
  Newton omits the new `contact_adhesion_in` argument to `contact_force_fn`.
  The compatible versions were selected explicitly; no dependency override
  or silent runtime fallback is used.
- MuJoCo's build requirement, runtime requirement and CMake release tag change
  together. Existing native Gobot binaries must be rebuilt; changing only the
  Python package leaves the old MuJoCo SONAME unresolved.
- libuipc's new `cuda_tool` implementation replaces its external muda build
  dependency. Gobot's device force/target bindings, contact gradient views,
  friction history preservation, cross-joint Hessian assembly and recoverable
  solver errors are retained on the new implementation. Embedded METIS/GKlib
  and derived muda license texts remain in the runtime bundle.
- Gobot explicitly keeps libuipc's fully implicit integration and
  `newton/min_iter=1`. The new upstream defaults enable semi-implicit stopping
  and permit termination at iteration zero. With Gobot's interactive tolerance,
  that yielded stale near-zero exported tangential forces in the resting-cube
  friction regression. Restoring the existing settings passes all four
  host/device and checkpoint variants without changing their tolerances.
- SuperDex retains GCC core-only SDK export, failed-step state preservation and
  opt-in stage timings. Its ExternalProject re-enters the incremental SDK build
  so a source/submodule update cannot leave old installed SDK objects in use.
- Historical Go1 and Newton G1 reference traces retain their producer versions
  and tolerances; the dependency upgrade does not regenerate acceptance data.
- Warp 1.17's default mode does not guarantee bit-exact GPU reductions. The G1
  20-step reset/replay test uses absolute tolerances of `1e-5 m` for base
  translation distance and `1e-5 rad` for base rotation and each joint angle,
  with zero relative tolerance. Rotation comparison measures the shortest
  angle between normalized quaternions and accepts equivalent opposite signs;
  non-finite values and non-unit input quaternions still fail. This checks
  numerical reproducibility, not bit-exact GPU determinism. Experiments with
  `RUN_TO_RUN` did not meet the existing G1 height baseline, including when
  policy inference retained its normal mode. Gobot does not enable that
  experimental configuration by default.
- Newton disables the MuJoCo sensor pass when the imported model has no
  sensors. MuJoCo Warp 3.12 otherwise allocates tactile preprocessing buffers
  even for such models, and its sensor module cannot compile in deterministic
  mode because a tactile kernel mixes maximum and addition reductions. Models
  with authored sensors keep the pass enabled; deterministic sensor support is
  still subject to that upstream limitation.
- The MuJoCo/libuipc Play example resolves the installed Gobot solver by default.
  It no longer prioritizes unrelated `build/libuipc-novcpkg` artifacts, which
  can load an old ABI beside the current SDK. Explicit
  `GOBOT_LIBUIPC_SOLVER_MODULE` overrides remain supported.
- MuJoCo Warp and Newton initialization validate published Warp disk-cache
  metadata and reject empty kernel/LTO binaries before loading models. Damaged
  entries are moved into `.gobot-quarantine` inside the active Warp cache,
  preserving them for inspection; Warp recompiles the missing entries on
  demand. Valid entries, in-progress compiler directories and symlink targets
  are left alone. Physics steps are not retried and numerical settings do not
  change as part of cache recovery.

## Reproduction

Install/rebuild the current checkout with `uv sync`, then run normal commands
such as `uv run gobot_editor --path examples/mujoco_libuipc`.
Warp's first run compiles kernels for the new version; that cold-start time is
separate from steady-state simulation performance.

Targeted CPU and provider checks:

```sh
uv run --no-sync --with pytest python -m pytest \
  tests/python/test_build_backend.py \
  tests/python/test_newton_provider.py \
  tests/python/test_mujoco_ipc_provider.py \
  tests/python/test_simulation_runtime_torch.py
```

GPU checks require a working NVIDIA driver:

```sh
uv run --no-sync python tests/python/test_mujoco_warp_provider.py
GOBOT_RUN_NEWTON_GPU_TEST=1 uv run --no-sync --with pytest python -m pytest tests/python/test_newton_provider.py
GOBOT_RUN_NEWTON_G1_GPU_TEST=1 uv run --no-sync python tests/python/test_newton_g1_gpu.py
GOBOT_RUN_LIBUIPC_BATCH_GPU_TEST=1 uv run --no-sync --with pytest python -m pytest tests/python/test_libuipc_batch_gpu.py
GOBOT_RUN_WARP_IPC_GPU_TEST=1 uv run --no-sync --with pytest python -m pytest tests/python/test_simulation_runtime_torch.py
```

## Verification

- GCC 12 CPU SuperDex core SDK build/install and Gobot integration succeeded.
  74 C++ tests passed (SuperDex, PhysicsServer, IPC solver ABI and Python runtime).
- The full editor/Python build with libuipc's CUDA backend succeeded and the
  editable Gobot package was reinstalled.
- MuJoCo Warp 3.12 passed the real four-environment cartpole test: CPU parity,
  graph stepping, selective reset, checkpoint and stable allocated memory.
  The short smoke measured about 29,000 env-steps/s on RTX 4070; it is not a
  conveyor/soft-body throughput benchmark.
- Newton provider: 21 tests passed, including real GPU stepping, robot import,
  controls, selective reset and both contact modes.
- The separate G1 four-environment GPU test passed in normal mode, including
  the original standing-height reference, 20-step reset replay within the
  documented tolerances, selective reset and forward walking. Maximum replay
  errors were `1.258e-7 m` in base translation, `7.723e-7 rad` in base rotation
  and `7.898e-7 rad` in joint angles. The historical standing and walking
  acceptance data and tolerances remain unchanged. This short replay check
  does not establish bit-exact determinism or bound long-horizon divergence.
- libuipc: 11 batch GPU tests and three single-scene GPU tests passed, including
  friction feedback, strict failure rewind, device buffers and coupled stepping.
- The five cold-import/runtime regressions passed, including the real
  four-environment 304-tick press cycle, reset and re-step.
- The editor opened `examples/mujoco_libuipc`, entered Play and recorded 180
  render samples / 63 completed physics ticks with `world_ready=true`,
  `faulted=false` and no error. This is a startup/Play smoke, not a performance
  baseline (other validation processes ran concurrently).
- 71 Python build, example, provider and runtime checks passed; the two GPU
  cases skipped in that run were subsequently exercised by the opt-in suite.
- libuipc was compiled with CUDA 12.2. Its conditional WHILE graph path requires
  a newer toolkit and is not covered by this build; block graph replay is built.

### Rope Play cache recovery

The reported `JSONDecodeError` in `warp._src.context.Module.load` came from
seven empty cached module metadata files, with empty PTX files alongside them.
Two empty shared LTO/fatbin files also caused `nvJitLink` invalid-input errors
once the module cache had been repaired. The cause of the original empty
writes has not been established; available disk space was 155 GB.

After quarantining those entries and recompiling, the editor successfully
started `examples/dual_arm_rope_twist` in Play and captured 60 render frames
with `world_ready=true`, `faulted=false` and no error. The recorded smoke is
`/tmp/gobot-rope-cache-recovery.json`; this verifies startup and short stepping,
not a full rope-twisting cycle. Eight cache recovery regressions and 21 Newton
provider checks passed (29 tests; the optional GPU case was not enabled in
that unit-test run).
