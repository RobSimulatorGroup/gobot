# gobot

Gobot is a Linux robotics simulation package with a Python-first workflow.
Use it from Python to create scenes, step simulation, inspect robot state, and
drive reinforcement-learning experiments.

![overview](https://raw.githubusercontent.com/RobSimulatorGroup/gobot/master/doc/overview.png)

[![PyPI](https://img.shields.io/pypi/v/gobot)](https://pypi.org/project/gobot/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/RobSimulatorGroup/gobot/blob/master/LICENSE)
[![CI](https://github.com/RobSimulatorGroup/gobot/actions/workflows/ci.yml/badge.svg?branch=master&event=push)](https://github.com/RobSimulatorGroup/gobot/actions)

## Install

Gobot currently publishes Linux wheels.

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
uv sync
uv run gobot_editor
```

For a direct CMake build, use the Python selected by `uv`:

```bash
cmake -S . -B build -DPython3_EXECUTABLE="$(uv python find)"
cmake --build build -j
./build/python/gobot/gobot_editor
```

At runtime, `GOBOT_PYTHON_LIBRARY=/other/libpython.so` still overrides the
compiled default.

The default install includes the lightweight ONNX Runtime path used for example
policy playback. The heavier training stack stays optional:

```bash
uv sync --extra train
```

`train` installs CUDA PyTorch, `rsl-rl-lib`, `tensordict`, and `tensorboard`
for training or directly loading `.pt` checkpoints.

Run example training through `uv` rather than a conda Python path:

```bash
uv run --extra train python examples/go1/train/go1_velocity_train.py --task go1_rough --num-envs 256 --iterations 1500
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
uv run --with build python -m build --wheel
uv pip install --force-reinstall dist/gobot-*.whl
```

## Notes

- Supported platform: Linux.
- Python package name: `gobot`.
- MuJoCo support is included in release wheels when available in the build.
- Packaged examples: `gobot/examples/` in wheels and `examples/` in source.
- MuJoCo RL roadmap: `doc/mujoco_rl_plan.md`.
