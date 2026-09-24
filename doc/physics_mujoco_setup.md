# MuJoCo setup

MuJoCo CPU is enabled by default. Gobot currently pins MuJoCo **3.12.0** in
`CMakeLists.txt` and `pyproject.toml`. See [build and test](building.md) for a
complete CPU configuration.

## Python builds

Python builds use the pinned `mujoco` wheel's headers and shared library from
the build environment. They disable CMake's source-fetch fallback. CPU batches
also use the pinned `3rdparty/mjbatch` native headers; the build backend
initializes that submodule automatically. No mjbatch Python package is needed.

## Standalone CMake

Initialize mjbatch, then provide an installed MuJoCo SDK or let CMake discover
the selected Python environment's wheel:

```bash
git submodule update --init 3rdparty/mjbatch
cmake -S . -B build/cpu \
  -DGOB_BUILD_MUJOCO=ON \
  -DGOB_BUILD_LIBUIPC=OFF \
  -DGOB_MUJOCO_ROOT=/path/to/mujoco
```

`GOB_MUJOCO_ROOT` accepts a CMake install prefix or an SDK with `include/` and
`lib/` or `bin/`. A package prefix can also be supplied through
`CMAKE_PREFIX_PATH`. Dynamic SDK libraries must be visible at runtime, for
example through `LD_LIBRARY_PATH=/path/to/mujoco/lib`.

| Option | Effect |
| --- | --- |
| `GOB_BUILD_MUJOCO=OFF` | Disable MuJoCo and mjbatch |
| `GOB_FETCH_MUJOCO=ON` | Fetch the pinned source when no local package is found; standalone default |
| `GOB_FORCE_FETCH_MUJOCO=ON` | Force a native source build even if an SDK is available |
| `GOB_FETCH_MUJOCO=OFF` | Require a local SDK/package; Python build default |
| `GOB_MUJOCO_GIT_TAG` | Select the fetched release; keep it aligned with the Python dependency |

A fetched build may download MuJoCo's own dependencies. The
[mjbatch adapter](mjbatch_cpu.md) requires the system `patch` utility and leaves
the upstream submodule unchanged.

## Verify simulation

Load a robot scene, select **MuJoCo CPU** in the Physics panel, confirm
**Available**, then **Build World** and **Step**. Check the frame counter and
joint state. Falling requires a floating base; contact requires a floor.

The backend compiles authored Gobot nodes and resources. `Robot3D.source_path`
is import provenance and is never reopened during simulation. See
[MJCF equivalence](mjcf_equivalence.md) for import checks.
