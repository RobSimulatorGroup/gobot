# Gobot

Gobot is a Linux robotics engine for editing scenes, simulating robots, rendering,
and reinforcement learning, with Python as the scripting API.

[![PyPI](https://img.shields.io/pypi/v/gobot)](https://pypi.org/project/gobot/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/RobSimulatorGroup/gobot/blob/master/LICENSE)
[![CI](https://github.com/RobSimulatorGroup/gobot/actions/workflows/ci.yml/badge.svg?branch=master&event=push)](https://github.com/RobSimulatorGroup/gobot/actions)

![Gobot editor](https://raw.githubusercontent.com/RobSimulatorGroup/gobot/master/doc/overview.png)

## Install

```bash
pip install gobot
gobot_editor
```

Linux x86_64 wheels include MuJoCo CPU, MuJoCo Warp, Newton, PyTorch,
LuisaCompute rendering, Gaussian Splatting inference, and OpenUSD. There are no
feature extras. GPU features need a compatible NVIDIA driver and system
EGL/OpenGL libraries; wheels do not require a CUDA Toolkit.

Open a packaged project from **Examples** on the editor start screen.

## Python

```python
import gobot

gobot.set_project_path("/path/to/project")
scene = gobot.load_scene("res://world.jscn")
print(scene.root.name)
print(gobot.backend_infos())
```

## Develop from source

Source builds need a C++20 toolchain and, with the default features, a CUDA
Toolkit. See [building](doc/building.md) for prerequisites and CPU-only builds.

```bash
git clone https://github.com/RobSimulatorGroup/gobot.git
cd gobot
uv run gobot_editor --path examples/go1
```

The first run initializes pinned dependencies and builds the editable package.
Later editor launches rebuild native changes incrementally.

## Examples and training

| CartPole | Go1 policy playback |
| --- | --- |
| <img src="doc/video/cartpole.gif" alt="CartPole" width="420"> | <img src="doc/video/go1.gif" alt="Go1" width="420"> |

Run a small Go1 CPU training job from the checkout:

```bash
uv run python -m examples.go1.train.go1_velocity_train --cpu-batch --iterations 10
```

See the [example catalog](doc/examples.md) and [Go1 guide](examples/go1/README.md)
for CUDA training, evaluation, and policy playback.

## Documentation

- [Documentation index](doc/README.md)
- [Build and test](doc/building.md)
- [Engine architecture](doc/architecture.md)
- [MuJoCo and RL](doc/mujoco_rl_plan.md)
- [Rendering](doc/luisa_rendering_plan.md)
