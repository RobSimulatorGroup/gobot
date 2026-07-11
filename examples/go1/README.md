# Go1 Example

This example keeps the robot project self-contained:

- `assets/xml/go1.xml` is the original MJCF robot asset retained for import/equivalence checks.
- `go1_profile.py` owns the example-local Go1 articulation and controller profile.
- `go1_velocity_contract.py` owns the versioned policy identity shared by training and playback.
- `go1.jscn` is the Gobot robot scene imported from `go1.xml`.
- `go1_scene.jscn` references `go1.jscn` and owns the `terrain_world/terrain` authoring nodes.
- `terrain/rough_terrain.jres` is the versioned procedural terrain recipe used by the editor, playback, and training.
- `train/go1_velocity_train.py` trains the Go1-owned velocity task into `policies/`.
- `train/go1_velocity_env.py` contains the Go1 rsl_rl vector environment. It trains from the Gobot `.jscn` scene through a scene-authored CPU batch backend facade, not by importing XML directly.
- `train/go1_velocity_cfg.py` contains rewards, PPO, command, solver, and terrain spawn-curriculum settings.
- `scripts/go1.py` is attached to the `go1_scene.jscn` root and plays a trained policy in `gobot_editor`.
- `policies/go1_velocity.pt` is the default training output.
- `policies/go1_velocity.onnx` is generated from that checkpoint for lightweight playback.
- `tools/export_policy_onnx.py` validates the checkpoint manifest and embeds it in ONNX.
- `project.gobot` sets `go1_scene.jscn` as the project main scene.

Regenerate the Gobot robot asset after editing the MJCF. The current editable
install supplies `gobot`; no source/build `PYTHONPATH` is needed:

```bash
cd /home/wqq/gobot
uv run python -m examples.go1.tools.refresh_go1_robot_scene
```

This refreshes `go1.jscn` from `assets/xml/go1.xml` and restores the
Gobot-authored runtime sensors used by training. The scene keeps a compact
reference to `terrain/rough_terrain.jres`; native `Terrain3D` generation lazily
builds render and collision geometry without serializing generated height
arrays. The terrain is visible as soon as the scene is opened, before entering
Play Mode, and training consumes the same authored recipe.

Train from the repository root:

```bash
cd /home/wqq/gobot
uv run --extra train python -m examples.go1.train.go1_velocity_train \
  --num-envs 256 \
  --iterations 10000 \
  --device cuda \
  --sim-workers 0 \
  --log-dir logs/go1_rough_velocity \
  --policy-out policies/go1_velocity.pt
```

For a small CPU smoke run, use the built-in preset instead of a separate entry
script:

```bash
cd /home/wqq/gobot
uv run --extra train python -m examples.go1.train.go1_velocity_train --cpu-batch --iterations 10
```

Benchmark the Gobot Go1 vector env hot path without the PPO learner:

```bash
cd /home/wqq/gobot
uv run --extra train python benchmark/go1_velocity_benchmark.py \
  --num-envs 64 \
  --steps 100 \
  --warmup-steps 10 \
  --device cpu \
  --sim-workers 0 \
  --no-obs-noise \
  --profile-step
```

`--profile-step` reports the major Go1 CPU batch env phases: action
preparation/application, physics, state refresh, command update, contact
updates, reward, termination/reset, observation build, tensor conversion, and
extras/logging.

The `go1_rough_velocity` task follows the Go1 rough-terrain observation,
reward, command, event, robot-dynamics, collision, and mixed-terrain settings
while loading the Gobot `.jscn` scene as the source of truth. Simulation always
uses Gobot's CPU MuJoCo batch runtime. `--device` controls the PyTorch learner;
it does not select the physics backend.

Resume from the latest checkpoint in the log directory:

```bash
cd /home/wqq/gobot
uv run --extra train python -m examples.go1.train.go1_velocity_train \
  --num-envs 256 \
  --iterations 10000 \
  --device cuda \
  --sim-workers 0 \
  --render-video-interval 100 \
  --log-dir logs/go1_rough_velocity \
  --policy-out policies/go1_velocity.pt \
  --resume
```

Export a trained checkpoint for lightweight editor playback:

```bash
cd /home/wqq/gobot
uv run --extra train python -m examples.go1.tools.export_policy_onnx \
  --checkpoint examples/go1/policies/go1_velocity.pt \
  --output examples/go1/policies/go1_velocity.onnx
```

The exporter embeds the same `gobot.rl.PolicyManifest` written by training and
writes `go1_velocity.onnx.manifest.json`. Playback rejects policies without this
manifest, mismatched observation/action specs, a different joint order, or a
different scene resource digest. This makes stale 45-dimensional policies fail
at load time instead of accepting keyboard commands that the policy cannot use.

To play the Torch checkpoint directly:

```bash
cd /home/wqq/gobot
GOBOT_GO1_POLICY=res://policies/go1_velocity.pt \
  uv run --extra train gobot_editor --path examples/go1
```

After exporting `policies/go1_velocity.onnx`, play it in the editor with
keyboard velocity commands:

```bash
cd /home/wqq/gobot
uv run gobot_editor --path examples/go1
```

Playback prefers `policies/go1_velocity.onnx` and then tries the training
checkpoint `policies/go1_velocity.pt`. Both paths require the current policy
manifest; older checkpoints are rejected instead of being guessed from tensor
dimensions. A missing or rejected policy aborts Play with the exact validation
error. Rough terrain remains available in the editor without entering Play.

Click the 3D viewer, then use `W/S` for forward/backward, `Q/E` for strafe,
`A/D` for yaw rate, `Space` to stop, and `R` to reset. The command limits are
`vx=1.0`, `vy=1.0`, and `yaw_rate=0.5`. Physics rate, policy decimation, PD
gains, action scaling, reset height, and solver settings come from the policy
manifest. Terrain geometry comes from the versioned scene resource and is
covered by the manifest's scene bundle digest.

Run the editor from an editable install and open the `examples/go1` project.
