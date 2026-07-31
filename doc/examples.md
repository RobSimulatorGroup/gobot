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

## Project Load Hooks

A project may declare a trusted Python setup hook in `project.gobot`:

```json
{
    "main_scene": "res://main.jscn",
    "project_load_hook": "res://download_assets.py"
}
```

The hook must be a `.py` file inside the project. The editor runs it in a
separate process using the Python environment that launched `gobot_editor`, and
opens the main scene only after the hook exits successfully. A nonzero exit
keeps the project setup dialog open with Retry and Close actions. Project hooks
are executable code, so only open projects from sources you trust.

Hooks can publish modal progress by writing one compact JSON object per line to
stdout with the `GOBOT_PROGRESS ` prefix:

```python
print('GOBOT_PROGRESS {"current":1048576,"total":8388608,"message":"Downloading assets"}', flush=True)
```

Other stdout and stderr lines are forwarded to the editor Console. Gobot sets
`GOBOT_PROJECT_HOOK=1`, `GOBOT_PROJECT_DIR`, and `PYTHONUNBUFFERED=1` in the hook
process. Hooks should be idempotent so reopening a prepared project completes
quickly without downloading the same assets again.

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
Gobot scene. The default installation can also load `.pt` checkpoints directly.

## Go1

`examples/go1` is a Unitree Go1 policy-playback and training example. The
editor playback path uses ONNX Runtime when `policies/go1_velocity.onnx` is
present, so it does not need the rsl_rl training stack. Training consumes the
same scene-authored robot and terrain through an explicit MuJoCo CPU semantic
baseline or MuJoCo Warp CUDA backend. It contains:

- `train/go1_velocity_train.py`: the rsl_rl PPO training entry point.
- `train/go1_velocity_env.py`: the Go1-owned MuJoCo CPU vectorized environment.
- `train/go1_warp_velocity_env.py`: the device-native MuJoCo Warp environment.
- `go1_profile.py`: the example-local articulation, default pose, drives, and actuator limits.
- `go1_velocity_contract.py`: the policy task name and version shared by training and playback.
- `train/go1_velocity_cfg.py`: rewards, PPO, command, solver, and terrain spawn-curriculum settings.
- `train/go1_gait.py`: run-only paired-leg gait scoring shared by CPU parity tests and Warp training.
- `go1_scene.jscn`: the authored scene with an editor-visible `Terrain3D` node.
- `terrain/rough_terrain.jres`: the compact versioned procedural terrain recipe shared by editor, Play, and training.
- `go1.jscn` and `assets/`: the imported robot scene and source meshes.
- `policies/go1_velocity.onnx`: released balanced policy playback graph with an embedded manifest.

The default Gobot install can play ONNX policies, train or load `.pt`
checkpoints, capture MP4 video, and export ONNX policies.

Playback supports `W/S` forward/reverse, `Q/E` strafe, `A/D` yaw, `Space`
stop, and `R` reset. `GOBOT_GO1_POLICY` overrides the policy path. Policy admission
evaluates every authored terrain cell, requires both survival and commanded
planar/yaw progress, and reports paired-leg gait metrics. The current
validation snapshot, checkpoint comparison, measured rates, and gait limits are
recorded in `examples/go1/README.md`.

## Newton G1

`examples/newton_g1` reproduces Newton 1.4's
`robot_policy --robot g1_29dof` playback contract inside Gobot. It uses the
official `g1_29dof_with_hand_rev_1_0.usda` as the visible scene asset, compiles
that USD once into a Gobot `.jscn`, and steps the artifact compiled from the
same `Robot3D`/`Link3D`/`Joint3D` tree with `NewtonProvider` on CUDA. Gobot
remains responsible for the viewport; the Newton viewer is not loaded.

The project load hook downloads only the files needed by this example at a
pinned revision of the official
[`newton-physics/newton-assets`](https://github.com/newton-physics/newton-assets)
repository. Each file is checked by size and Git blob SHA-1, partial downloads
resume, and verified files are reused without network access. Only seven USD
composition files and the policy, YAML, and license are downloaded; no
companion MJCF or duplicate STL set is needed. The generated scene has a
versioned cache stamp, and its large render/collision meshes are kept in a
sibling binary `.meshes/` cache instead of expanding into JSON. The hook
validates both the stamp and every external mesh before reusing the scene.
The downloaded `assets/` directory is excluded from source and wheel packages.

Press Play, then use `I`/`K` for forward/backward, `J`/`L` for lateral motion,
`U`/`O` for yaw, and `P` to reset. The policy runs at 50 Hz over a 200 Hz Newton
simulation and controls all 43 G1 joints, including the hands. The compiled
artifact also retains the 43 affine actuators authored in the USD, for 86
actuators total, instead of silently replacing the source model's dynamics.

## Packaging Rules

The Python package install step includes:

- `.jscn` scene files.
- `.py` scripts.
- `.xml` MuJoCo scene files.
- source assets checked into `examples/`, including the released Go1 playback policy.
- pinned manifests and project hooks for large optional example assets.

Generated Python cache files and directories are excluded.
