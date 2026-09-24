# Build and test

For a published package, use `pip install gobot`. The workflows below require a
complete Git checkout. Dependency versions live in `pyproject.toml`,
`CMakeLists.txt`, and the pinned submodules.

## Editable Python development

Requirements: Linux, Python 3.10+, uv, Git, CMake 3.27+, Ninja, `patch`, a C++20
compiler, and system development libraries for Eigen, OpenGL/EGL, and X11.
The default SuperDex build needs GCC 12+ or a supported Clang. Default Python
builds also need a CUDA Toolkit with `nvcc`; libuipc needs TBB and urdfdom
development packages. See the [CI workflow](../.github/workflows/ci.yml) for
the Ubuntu CPU dependency list and the [libuipc guide](libuipc_ipc.md) for CUDA
build requirements.

```bash
git clone https://github.com/RobSimulatorGroup/gobot.git
cd gobot
uv sync
uv run gobot_editor --path examples/go1
```

The build backend initializes pinned submodules and caches the LuisaCompute,
gsplat, and OpenUSD SDKs. The first build needs network access for uncached
dependencies. Later builds reuse them.

Python edits are live. The editable editor launcher runs an incremental native
build before starting. For other Python entry points, rebuild and reinstall
Gobot after C++ or CMake changes:

```bash
uv sync --reinstall-package gobot --no-build-isolation-package gobot
```

After `uv sync`, `uv run gobot_editor` and `.venv/bin/gobot_editor` use the same
installation. Use that environment's launcher to keep Python, `_core`,
`libgobot`, and the editor consistent. Avoid mixing it with a standalone
build through `PYTHONPATH`.

## CMake builds and tests

A standalone build is separate from the editable installation. This CPU
configuration needs no CUDA Toolkit:

```bash
git submodule update --init --depth=1 \
  3rdparty/assimp 3rdparty/gli 3rdparty/googletest 3rdparty/imgui \
  3rdparty/meshoptimizer 3rdparty/mjbatch 3rdparty/project_superdex \
  3rdparty/pybind11 3rdparty/rttr 3rdparty/stb

python3 -m venv build/cpu-venv
source build/cpu-venv/bin/activate
python -m pip install 'cmake>=3.27' ninja \
  -r tests/python/requirements-cpu.txt

cmake -S . -B build/cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
  -DGOB_BUILD_TESTS=ON \
  -DGOB_BUILD_LIBUIPC=OFF \
  -DGOB_BUILD_LUISA_RENDERER=OFF \
  -DGOB_BUILD_OPENUSD=OFF
cmake --build build/cpu --parallel 4
ctest --test-dir build/cpu --output-on-failure
```

Python CTest cases load `build/cpu/python` in an isolated interpreter. They do
not use an older installed `_core`. Optional GPU tests skip when unavailable.
Rebuilding this tree does not update `.venv`; use the reinstall command above
when the installed editor or Python package needs the changes.
Run `deactivate` when finished with the CPU test environment.

## Optional features

| CMake option | Standalone default | Python build default | Details |
| --- | --- | --- | --- |
| `GOB_BUILD_MUJOCO` | ON | ON | [MuJoCo SDK and mjbatch](physics_mujoco_setup.md) |
| `GOB_BUILD_SUPERDEX` | ON | ON | Experimental CPU backend; CUDA is separately disabled |
| `GOB_BUILD_LIBUIPC` | ON | ON | [CUDA IPC solver](libuipc_ipc.md) |
| `GOB_BUILD_LUISA_RENDERER` | OFF | ON | [CUDA renderer](luisa_rendering_plan.md) |
| `GOB_BUILD_GSPLAT_INFERENCE` | ON | ON | [Gaussian rendering](gaussian_splatting.md), requires Luisa |
| `GOB_BUILD_OPENUSD` | OFF | ON | [USD importer](newton_openusd.md) |

Set CMake options with `-D<name>=OFF`. For an editable Python build, pass the
same option through uv, for example:

```bash
uv sync --reinstall-package gobot --no-build-isolation-package gobot \
  -C cmake.define.GOB_BUILD_SUPERDEX=OFF
```

An explicit option remains in an existing CMake cache until reconfigured.
Python package dependencies still install as declared; disabling a native
module does not create a separate CPU-only wheel dependency set.

## Build a wheel

```bash
uv run --with build python -m build --wheel
uv pip install --force-reinstall dist/gobot-*.whl
```

The build backend prepares the native SDKs and bundles their required runtime
files. Wheel users need system EGL/OpenGL dispatch libraries and a compatible
NVIDIA driver for GPU features, but no SDK checkouts or CUDA compiler.
