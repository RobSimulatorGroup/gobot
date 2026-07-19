# gobot

Gobot is a Linux robotics simulation package with a Python-first workflow.
Use it from Python to create scenes, step simulation, inspect robot state, and
drive reinforcement-learning experiments.

![overview](https://raw.githubusercontent.com/RobSimulatorGroup/gobot/master/doc/overview.png)

[![PyPI](https://img.shields.io/pypi/v/gobot)](https://pypi.org/project/gobot/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/RobSimulatorGroup/gobot/blob/master/LICENSE)
[![CI](https://github.com/RobSimulatorGroup/gobot/actions/workflows/ci.yml/badge.svg?branch=master&event=push)](https://github.com/RobSimulatorGroup/gobot/actions)

## Install

Gobot currently publishes full-featured Linux x86_64 wheels for NVIDIA GPU
systems. The wheel includes CPU simulation, MuJoCo Warp training dependencies,
and the LuisaCompute CUDA viewport renderer.

```bash
pip install gobot -i https://pypi.org/simple
```

Check the install:

```python
import gobot

print(gobot.__file__)
print(gobot.__version__)
print(gobot.backend_infos())
```

## Python Usage

Set a project root before using `res://` paths:

```python
import gobot

gobot.set_project_path("/path/to/project")
scene = gobot.load_scene("res://world.jscn")

print(scene.root.name)
```

## Editor

Start the editor from the Python environment you want Gobot to use:

```bash
gobot_editor
```

For source checkout development:

```bash
cd /path/to/gobot
scripts/build_luisa_compute.sh
uv sync
uv run gobot_editor --path examples/go1
```

`uv sync` installs Gobot editable: Python files import directly from the
checkout, while `_core`, `libgobot`, and `gobot_editor` come from the same
build installed in `.venv`. After changing C++ or CMake files, rebuild and
reinstall only Gobot:

```bash
uv sync --reinstall-package gobot --no-build-isolation-package gobot
```

The initial `uv sync` installs the build backend in `.venv`; disabling build
isolation on later native rebuilds prevents CMake from caching a temporary
build environment's Python executable.

Python-only edits need no rebuild. Do not add `PYTHONPATH` or launch an editor
from `build/python`; that can combine source files, native extensions, and
executables from different builds. The package launcher selects the native
artifacts installed in its own environment.

`uv run` is environment activation, not a separate Gobot runtime. These are
equivalent after `uv sync`:

```bash
uv run gobot_editor --path examples/go1
.venv/bin/gobot_editor --path examples/go1
```

Alternatively, run `source .venv/bin/activate` once and then use
`gobot_editor` or `python` directly. At runtime,
`GOBOT_PYTHON_LIBRARY=/other/libpython.so` still overrides automatic
libpython discovery.

For a standalone CMake build used by C++ tests, use a separate build directory
and the Python selected by `uv`:

```bash
cmake -S . -B build/dev -DPython3_EXECUTABLE="$(uv python find)"
cmake --build build/dev -j
ctest --test-dir build/dev --output-on-failure
```

Python CTest cases run against `build/dev/python` in an isolated interpreter,
so an older editable `_core` in `.venv` cannot shadow the artifact under test.
That standalone build is not the artifact used by the `.venv` console script;
run the following command when the installed editor must be updated:

```bash
uv sync --reinstall-package gobot --no-build-isolation-package gobot
```

The default environment includes ONNX Runtime, PyTorch, rsl_rl, MuJoCo Warp,
training logs, video capture, and ONNX export support. There are no separate
CPU, CUDA, or training extras. Selecting `mujoco-cpu` or `mujoco-warp` remains
an explicit runtime choice; requesting Warp never silently falls back to CPU.

Run Go1 rough-terrain training on MuJoCo Warp through `uv`:

```bash
uv run \
  python -m examples.go1.train.go1_velocity_train \
  --backend mujoco-warp \
  --device cuda:0 \
  --num-envs 256 \
  --iterations 10000 \
  --no-step-extras
```

Select the CPU semantic baseline explicitly when CUDA is not desired:

```bash
uv run \
  python -m examples.go1.train.go1_velocity_train \
  --backend mujoco-cpu \
  --device cpu \
  --num-envs 64 \
  --iterations 10000
```

Packaged examples are available from the editor start screen under `Examples`.
See `doc/examples.md` for packaging details.

## Examples

| Example | Preview |
| --- | --- |
| CartPole | <img src="doc/video/cartpole.gif" alt="CartPole example" width="420"> |
| Go1 policy playback | <img src="doc/video/go1.gif" alt="Go1 policy playback" width="420"> |

## Local Wheel Build

From a source checkout:

```bash
git clone https://github.com/RobSimulatorGroup/gobot.git
cd gobot
git submodule update --init --recursive
scripts/build_luisa_compute.sh
uv run --with build python -m build --wheel
uv pip install --force-reinstall dist/gobot-*.whl
```

## Notes

- Supported platform: Linux.
- Python package name: `gobot`.
- MuJoCo support is included in release wheels when available in the build.
- Release wheels install the MuJoCo Warp Python provider and LuisaCompute CUDA
  renderer by default; neither becomes a C++ scene API.
- A system CUDA Toolkit is needed to build from source, but not to use a wheel.
  Wheel users need a compatible NVIDIA driver providing `libcuda.so.1` and
  the system libglvnd EGL/OpenGL dispatch libraries. Driver libraries are not
  copied into the wheel because they must match the target machine's driver.
- Packaged examples: `gobot/examples/` in wheels and `examples/` in source.
- MuJoCo RL roadmap: `doc/mujoco_rl_plan.md`.
- Luisa CUDA renderer architecture and build guide: `doc/luisa_rendering_plan.md`.
