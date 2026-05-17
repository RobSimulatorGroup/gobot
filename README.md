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

gobot.app.context().set_project_path("/tmp/gobot-demo")
gobot.scene.save_cartpole_scene("res://cartpole.jscn")

env = gobot.rl.ManagerBasedEnv(
    {"scene": "res://cartpole.jscn", "backend": "null", "controlled_joints": ["slider"]}
)
obs, info = env.reset(seed=1)

for _ in range(10):
    obs, reward, terminated, truncated, info = env.step([[0.0]])
    if terminated[0] or truncated[0]:
        obs, info = env.reset()
```

Use the Gymnasium-style adapter:

```python
import gobot

env = gobot.rl.GymWrapper(
    gobot.rl.ManagerBasedEnv(
        {"scene": "res://cartpole.jscn", "backend": "null", "controlled_joints": ["slider"]}
    )
)
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
gobot_editor
```

The executable and its local shared libraries are installed inside the Python
package, so a normal `pip install gobot` is enough for the packaged runtime.
Check the packaged editor version with `gobot_editor --version`.

For source checkout development, use an editable install in the Python
environment you want Gobot to run with:

```bash
cd /path/to/gobot
python -m pip install -e . --no-deps -Cbuild-dir=build/editable
gobot_editor
```

For verbose rebuild output:

```bash
python -m pip install -v -e . --no-deps -Cbuild-dir=build/editable -Cbuild.verbose=true
```

For repeated C++ edit/build cycles, keep `build-dir` fixed so CMake and Ninja
can reuse the previous build tree. If the build requirements are already
installed in the active Python environment, `--no-build-isolation` avoids
creating a temporary build environment on every reinstall:

```bash
python -m pip install -e . --no-deps --no-build-isolation -Cbuild-dir=build/editable
```

The `gobot_editor` command is the recommended development entry point. It
selects the current Python environment, finds that environment's `libpython`,
sets the runtime environment, and then launches the packaged editor executable.
Do not run `build/.../gobot_editor` directly unless you also set
`GOBOT_PYTHON_LIBRARY` to the exact `libpython` path for the Python environment
you want to embed.

Packaged examples are available from the editor start screen under `Examples`.
The same files are installed inside the Python package at `gobot/examples/`.
The first bundled example is `cartpole`, a MuJoCo-backed inverted pendulum scene
with a Python `NodeScript` controller.
The Python launcher records packaged example roots in `~/.gobot/projects.json`
under `example_roots`, and the GUI reads that list instead of guessing from the
editor executable path. If both a source checkout and a pip install are present,
same-name examples are de-duplicated and the pip-installed root is preferred
when launched through the Python package.
See `doc/examples.md` for the example project layout and packaging rules.

To locate examples from Python:

```python
from importlib import resources

print(resources.files("gobot").joinpath("examples"))
```

## Local Wheel Build

From a source checkout:

```bash
git clone https://github.com/RobSimulatorGroup/gobot.git
cd gobot
git submodule update --init --recursive
python -m pip install -U build scikit-build-core
python -m build --wheel
python -m pip install --force-reinstall dist/gobot-0.1.5-*.whl
```

For a faster local build without MuJoCo:

```bash
python -m build --wheel -Ccmake.define.GOB_BUILD_MUJOCO=OFF
```

## Notes

- Supported platform: Linux.
- Python package name: `gobot`.
- Main RL entry point: `gobot.rl.ManagerBasedEnv`.
- Gym-style helpers: `gobot.rl.GymWrapper`.
- MuJoCo support is included in release wheels when available in the build.
- Packaged examples: `gobot/examples/` in wheels and `examples/` in source.
- MuJoCo RL roadmap: `doc/mujoco_rl_plan.md`.
