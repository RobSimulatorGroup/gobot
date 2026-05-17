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
python -m pip install -e . -Cbuild-dir=build/editable
gobot_editor
```

The default install does not install PyTorch or RSL-RL, so it will not replace
the `torch` already in your environment. If you want pip to install optional RL
dependencies:

```bash
python -m pip install -e ".[rl]" -Cbuild-dir=build/editable
```

The `.[rl]` extra may upgrade `torch` if your current version does not satisfy
`rsl-rl-lib`.

For verbose rebuild output:

```bash
python -m pip install -v -e . -Cbuild-dir=build/editable -Cbuild.verbose=true
```

For repeated C++ edit/build cycles, keep `build-dir` fixed so CMake and Ninja
can reuse the previous build tree. If the build requirements are already
installed in the active Python environment, `--no-build-isolation` avoids
creating a temporary build environment on every reinstall:

```bash
python -m pip install -e . --no-build-isolation -Cbuild-dir=build/editable
```

Packaged examples are available from the editor start screen under `Examples`.
See `doc/examples.md` for packaging details.

## Local Wheel Build

From a source checkout:

```bash
git clone https://github.com/RobSimulatorGroup/gobot.git
cd gobot
git submodule update --init --recursive
python -m pip install -U build scikit-build-core
python -m build --wheel
python -m pip install --force-reinstall dist/gobot-*.whl
```

For a faster local build without MuJoCo:

```bash
python -m build --wheel -Ccmake.define.GOB_BUILD_MUJOCO=OFF
```

## Notes

- Supported platform: Linux.
- Python package name: `gobot`.
- MuJoCo support is included in release wheels when available in the build.
- Packaged examples: `gobot/examples/` in wheels and `examples/` in source.
- MuJoCo RL roadmap: `doc/mujoco_rl_plan.md`.
