# Gobot Examples

Gobot ships small, self-contained example projects in the source tree under
`examples/`. Python wheels install the same projects into `gobot/examples/`
inside the Python package.

For a pip install, this means the examples live under the installed package
directory, for example:

```text
<python-env>/lib/pythonX.Y/site-packages/gobot/examples
```

You can query the exact path from Python:

```python
from importlib import resources

print(resources.files("gobot").joinpath("examples"))
```

## Editor Discovery

When the editor opens without a current project, the Resources panel shows an
`Examples` section. The editor discovers example projects by reading
`example_roots` from `~/.gobot/projects.json`:

```json
{
    "example_roots": [
        "/path/to/site-packages/gobot/examples"
    ],
    "projects": []
}
```

The packaged Python launcher updates this list before running `gobot_editor`.
It moves the installed package's examples root to the front of the list.
For a source checkout, the editor falls back to the build-time source
`examples/` directory and writes it into `projects.json`.

Each immediate child directory is treated as an example project when it contains
at least one `.jscn` scene file. Opening an example sets that directory as the
current project, so `res://` paths resolve within the example project.
If multiple roots contain the same example directory name, the editor shows the
first one from `example_roots` and hides the later duplicates.

## Cartpole

`examples/cartpole` is an inverted-pendulum project with two workflows:

- `cartpole.jscn`: the editable Gobot scene.
- `scripts/cartpole.py`: a Python `NodeScript` controller attached to the root
- `env.py` and `train.py`: a direct Python MuJoCo + rsl_rl training environment.
- `inverted_pendulum.xml`: the MJCF model used by the training script.

The training environment is intentionally local to the example. It does not use
Gobot task JSON or project-specific code inside the `gobot` Python package. The
editor script loads the trained policy for playback in the Gobot scene.

## Go1

`examples/go1` is a direct Python MuJoCo + rsl_rl velocity-tracking example for
the Unitree Go1. It contains:

- `go1_env.py`: the vectorized MuJoCo environment.
- `train.py`: the rsl_rl PPO training entry point.
- `go1_scene.xml`: an MJCF reference scene with ground and position actuators.
- `go1.xml` and `assets/`: the robot MJCF and meshes used by training.

The default asset path can be overridden with `GOBOT_GO1_XML`.

## Packaging Rules

The Python package install step includes:

- `.jscn` scene files.
- `.py` scripts.
- `.xml` MuJoCo scene files.
- policy checkpoints and other example assets checked into `examples/`.

Generated Python cache files and directories are excluded.
