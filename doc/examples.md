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
current project, so `res://` paths resolve within the example project. If the
project has a `project.gobot` file with a `main_scene` entry, the editor opens
that scene automatically.
If multiple roots contain the same example directory name, the editor shows the
first one from `example_roots` and hides the later duplicates.

## Cartpole

`examples/cartpole` is an inverted-pendulum project with two workflows:

- `cartpole.jscn`: the editable Gobot scene.
- `scripts/cartpole.py`: a Python `NodeScript` controller attached to the root
- `env.py` and `train.py`: a direct Python MuJoCo + rsl_rl training environment.
- `inverted_pendulum.xml`: the MJCF model used by the training script.
- `project.gobot`: sets `cartpole.jscn` as the project main scene.

The training environment is intentionally local to the example. It does not use
Gobot task JSON or project-specific code inside the `gobot` Python package. The
editor script loads `policies/cartpole.onnx` for lightweight playback in the
Gobot scene. Install `gobot[train]` only to load `.pt` checkpoints directly.

## Go1

`examples/go1` is a Unitree Go1 policy-playback and training example. The
editor playback path uses ONNX Runtime when `policies/go1_velocity.onnx` is
present, so it does not need the rsl_rl training stack. Training uses Gobot's
scene-authored MuJoCo CPU batch world through the normal Python API. It contains:

- `train/go1_velocity_train.py`: the rsl_rl PPO training entry point.
- `train/go1_velocity_env.py`: the Go1-owned vectorized Gobot velocity environment.
- `go1_profile.py`: the example-local articulation, default pose, drives, and actuator limits.
- `go1_velocity_contract.py`: the policy task name and version shared by training and playback.
- `train/go1_velocity_cfg.py`: rewards, PPO, command, solver, and terrain spawn-curriculum settings.
- `go1_scene.jscn`: the authored scene with an editor-visible `Terrain3D` node.
- `terrain/rough_terrain.jres`: the compact versioned procedural terrain recipe shared by editor, Play, and training.
- `go1.jscn` and `assets/`: the imported robot scene and source meshes.
- `policies/go1_velocity.onnx`: generated lightweight policy playback graph.

The default Gobot install can play ONNX policies. Install `gobot[train]` only
to train or load `.pt` checkpoints directly. Install `imageio imageio-ffmpeg`
for MP4 training captures and `onnx` for ONNX export.

## Packaging Rules

The Python package install step includes:

- `.jscn` scene files.
- `.py` scripts.
- `.xml` MuJoCo scene files.
- source assets checked into `examples/`; generated Go1 policies stay local.

Generated Python cache files and directories are excluded.
