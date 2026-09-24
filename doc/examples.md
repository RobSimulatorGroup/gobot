# Examples

Run `gobot_editor` and choose a project under **Examples**. From a source
checkout, open a project directly:

```bash
uv run gobot_editor --path examples/go1
```

| Project | Purpose | Guide |
| --- | --- | --- |
| `cartpole` | Inverted-pendulum policy playback and local MuJoCo training | [Project files](../examples/cartpole) |
| `go1` | Rough-terrain locomotion, CPU/CUDA training, ONNX playback | [Go1](../examples/go1/README.md) |
| `newton_g1` | Humanoid policy playback with Newton and USD assets | [Newton G1](../examples/newton_g1/README.md) |
| `libuipc` | FEM/contact and articulated grasping demos | [libuipc](../examples/libuipc/README.md) |
| `mujoco_libuipc` | Coupled rigid/deformable batch simulation | [MuJoCo + libuipc](../examples/mujoco_libuipc/README.md) |
| `conveyor_packages` | Conveyor and two-hand manipulation experiments | [Conveyor](../examples/conveyor_packages/README.md) |
| `dual_arm_rope_twist` | Two-arm rope manipulation | [Rope twist](../examples/dual_arm_rope_twist/README.md) |
| `gaussian_splatting` | Pretrained 3D Gaussian environment | [Gaussian Splatting](../examples/gaussian_splatting/README.md) |

Each project guide contains its controls, dependencies, and commands. Go1
supports MuJoCo CPU and CUDA training; Newton, libuipc, and Gaussian rendering
require a compatible GPU runtime.

## Discovery and packaging

Source projects live in `examples/`; wheels install them under `gobot/examples/`.
Find the installed directory with:

```python
from importlib import resources

print(resources.files("gobot").joinpath("examples"))
```

The launcher registers that directory in `example_roots` in
`~/.gobot/projects.json`. The editor also supports the build-time source
directory. When names overlap, the first root wins.

A project's `project.gobot` selects an existing `.jscn` or `.gsplat` main scene.
Opening it sets the project root for `res://` paths. A top-level scene file is
also accepted for compatibility. Wheels include checked-in scenes, scripts,
assets, policies, and download manifests; generated caches and large downloaded
assets are excluded.

## Project load hooks

Projects that need external assets can declare a Python setup hook:

```json
{
  "main_scene": "res://main.jscn",
  "project_load_hook": "res://download_assets.py"
}
```

The hook must be a `.py` file inside the project. It runs in a separate process
using the launcher's Python, before the scene opens. Failure shows Retry and
Close actions. Hooks execute code, so open projects from trusted sources and
make hooks idempotent.

Report progress by printing one compact JSON object per line:

```python
print('GOBOT_PROGRESS {"current":1,"total":8,"message":"Downloading assets"}', flush=True)
```

Other output goes to the Console. The hook receives `GOBOT_PROJECT_HOOK=1`,
`GOBOT_PROJECT_DIR`, and `PYTHONUNBUFFERED=1`.
