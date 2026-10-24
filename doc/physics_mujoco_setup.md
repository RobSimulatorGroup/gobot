# MuJoCo setup

MuJoCo CPU is enabled by default. Gobot pins **MuJoCo 3.17.0** for both the C++
SDK and Python package, and uses its native `mjBatch` runtime. The standalone
mjbatch dependency and patch utility are no longer required.

For the complete Python environment, run `uv sync` from the repository root.
This rebuilds Gobot against the same MuJoCo library as the Python package.
Updating only the Python wheel leaves the older Gobot binary incompatible.

For standalone CMake, provide the installed SDK or allow source fetching:

```sh
cmake -S . -B build \
  -DGOB_BUILD_MUJOCO=ON \
  -DGOB_MUJOCO_ROOT="$PWD/.venv/lib/python3.13/site-packages/mujoco"
cmake --build build -j
```

| Setting | Effect |
| --- | --- |
| `GOB_BUILD_MUJOCO=OFF` | Build without MuJoCo support |
| `GOB_MUJOCO_ROOT` | Prefer this SDK or Python wheel directory |
| `GOB_FETCH_MUJOCO=ON` | Fetch the pinned release if no SDK is found |
| `GOB_FORCE_FETCH_MUJOCO=ON` | Build the pinned source instead of a discovered SDK |
| `GOB_MUJOCO_GIT_TAG` | Fetched release, default `3.17.0` |

Configuration rejects SDKs without the native batch API. Keep source and Python
versions aligned. See [CPU batching](mujoco_cpu_batch.md) for ownership, checkpoint
and thread-count behavior.

Gobot pins MuJoCo Warp 3.17.0 and Newton 1.7.0. It requests Newton's ONNX extra
and owns the simulation dependency pins itself, because Newton's optional `sim`
extra still selects an older MuJoCo release. A provider-local solver subclass
adapts Newton's mean-inertia statistics and reset-mask handling to the new stack;
installed Newton modules are not modified. This adaptation is version checked.
