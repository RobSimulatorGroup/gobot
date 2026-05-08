# gobot

Gobot is a Linux robotics simulation package with a Python-first workflow.
Use it from Python to create scenes, step simulation, inspect robot state, and
drive reinforcement-learning experiments.

![overview](./doc/overview.png)

[![license](https://img.shields.io/github/license/RobSimulatorGroup/gobot.svg)](https://github.com/RobSimulatorGroup/gobot/blob/master/LICENSE)
[![CI](https://github.com/RobSimulatorGroup/gobot/actions/workflows/ci.yml/badge.svg?branch=master&event=push)](https://github.com/RobSimulatorGroup/gobot/actions)

## Install

Gobot currently publishes Linux wheels.

```bash
python -m pip install -U gobot
```

Check the install:

```python
import gobot

print(gobot.__file__)
print(gobot.backend_infos())
```

## Python Usage

Create and inspect a small test scene:

```python
import gobot

scene = gobot.create_test_scene()
root = scene.root

print(root.name)
print(root.type)
print([child.name for child in root.children])
```

Run a minimal reinforcement-learning environment:

```python
import gobot

env = gobot.RLEnvironment()
obs, info = env.reset(seed=1)

for _ in range(10):
    obs, reward, terminated, truncated, info = env.step([0.0])
    if terminated or truncated:
        obs, info = env.reset()
```

Use the Gymnasium-style adapter:

```python
from gobot.gym_adapter import GobotGymEnv

env = GobotGymEnv()
obs, info = env.reset(seed=1)
obs, reward, terminated, truncated, info = env.step([0.0])
```

Set a project root when working with `res://` paths:

```python
import gobot

gobot.set_project_path("/path/to/project")
scene = gobot.load_scene("res://world.jscn")
```

## Editor

The wheel also installs the editor command:

```bash
gobot-editor
```

The executable and its local shared libraries are installed inside the Python
package, so a normal `pip install gobot` is enough for the packaged runtime.

## Local Wheel Build

From a source checkout:

```bash
git clone https://github.com/RobSimulatorGroup/gobot.git
cd gobot
git submodule update --init --recursive
python -m pip install -U build scikit-build-core
python -m build --wheel
python -m pip install --force-reinstall dist/gobot-0.1.0-*.whl
```

For a faster local build without MuJoCo:

```bash
python -m build --wheel -Ccmake.define.GOB_BUILD_MUJOCO=OFF
```

## Notes

- Supported platform: Linux.
- Python package name: `gobot`.
- Main RL entry point: `gobot.RLEnvironment`.
- Gym-style helpers: `gobot.gym_adapter`.
- MuJoCo support is included in release wheels when available in the build.
