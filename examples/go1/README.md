# Go1 Example

This example keeps the robot project self-contained:

- `assets/xml/go1.xml` is the UniLab-aligned MJCF robot asset used for import.
- `assets/xml/locomotion_task.xml` is the UniLab Go1 locomotion fragment used as the task reference.
- `assets/xml/go1_scene.xml` is a flat MJCF reference scene that includes `go1.xml` and `locomotion_task.xml`.
- `go1.jscn` is the Gobot robot scene imported from `go1.xml`.
- `go1_scene.jscn` is the Gobot scene that references `go1.jscn`.
- `train/go1_velocity_train.py` trains the Go1-owned velocity task into `policies/`.
- `train/go1_velocity_env.py` contains the Go1 rsl_rl vector environment. It trains from the Gobot `.jscn` scene through a scene-authored CPU batch backend facade, not by importing XML directly.
- `train/go1_velocity_cfg.py` contains Go1 joints, rewards, PPO, command, and terrain curriculum settings.
- `scripts/go1.py` is attached to the `go1_scene.jscn` root and plays a trained policy in `gobot_editor`.
- `policies/go1.onnx` is the default lightweight rough-terrain playback policy.
- `policies/go1.pt` is the Torch checkpoint used by training, export, and fallback playback.
- `tools/export_policy_onnx.py` converts `policies/go1.pt` to `policies/go1.onnx`.
- `project.gobot` sets `go1_scene.jscn` as the project main scene.

Regenerate the Gobot robot asset after editing the MJCF. The current editable
install supplies `gobot`; no source/build `PYTHONPATH` is needed:

```bash
cd /home/wqq/gobot
uv run python examples/go1/tools/refresh_go1_robot_scene.py
```

This refreshes `go1.jscn` from `assets/xml/go1.xml` and restores the
Gobot-authored runtime sensors used by training. Do not regenerate
`go1_scene.jscn` from `go1_scene.xml` for training; the training world keeps
the Gobot `.jscn` terrain instance as the source of truth.

Train from the Go1 project root:

```bash
cd /home/wqq/gobot
uv pip install imageio imageio-ffmpeg
uv run --extra train python examples/go1/train/go1_velocity_train.py \
  --task go1_rough \
  --num-envs 256 \
  --iterations 1500 \
  --device cuda \
  --sim-workers 0 \
  --render-video-interval 100 \
  --render-video-num-envs 1 \
  --render-video-env-id 0 \
  --render-video-steps 240 \
  --render-video-fps 30 \
  --render-video-width 1280 \
  --render-video-height 720 \
  --log-dir logs/go1_velocity \
  --policy-out policies/go1_velocity.pt
```

For a small CPU smoke run, use the built-in preset instead of a separate entry
script:

```bash
cd /home/wqq/gobot
uv run --extra train python examples/go1/train/go1_velocity_train.py --cpu-batch --iterations 10
```

Benchmark the Gobot Go1 vector env hot path without the PPO learner:

```bash
cd /home/wqq/gobot
uv run --extra train python benchmark/go1_velocity_benchmark.py \
  --task go1_flat \
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

Benchmark a UniLab-style raw MuJoCo state-array batch step path:

```bash
cd /home/wqq/gobot
uv run --extra train python benchmark/mujoco_uni_batch_benchmark.py \
  --num-envs 64 \
  --steps 100 \
  --warmup-steps 10 \
  --nstep 10 \
  --threads 16
```

`mujoco_uni_batch_benchmark.py` uses `mujoco._batch_env`/`BatchEnvPool` when
that extension is installed. Otherwise it falls back to official
`mujoco.rollout.Rollout` and prints `Backend: rollout`, which is still useful
as a raw MuJoCo state-array stepping baseline but is not the persistent
`BatchEnvPool` implementation.

The default `go1_rough` task follows the Go1 rough-terrain observation,
reward, command, event, robot-dynamics, collision, and mixed-terrain settings
while loading the Gobot `.jscn` scene as the source of truth. Simulation always
uses Gobot's CPU MuJoCo batch runtime. `--device` controls the PyTorch learner;
it does not select the physics backend.

Resume from the latest checkpoint in the log directory:

```bash
cd /home/wqq/gobot
uv pip install imageio imageio-ffmpeg
uv run --extra train python examples/go1/train/go1_velocity_train.py \
  --task go1_rough \
  --num-envs 256 \
  --iterations 1500 \
  --device cuda \
  --sim-workers 0 \
  --render-video-interval 100 \
  --log-dir logs/go1_velocity \
  --policy-out policies/go1_velocity.pt \
  --resume
```

Export a trained checkpoint for lightweight editor playback:

```bash
cd /home/wqq/gobot
uv pip install onnx
uv run --extra train python examples/go1/tools/export_policy_onnx.py \
  --checkpoint examples/go1/policies/go1_velocity.pt \
  --output examples/go1/policies/go1.onnx
```

The default Go1 playback loads `policies/go1.onnx`, exported from the shipped
Torch checkpoint with the current observation schema. Set
`GOBOT_GO1_POLICY=res://policies/go1.pt` and install/use `gobot[train]` if you
want to play the Torch checkpoint directly.

Play the current Go1 ONNX policy in the editor with keyboard velocity
commands:

```bash
cd /home/wqq/gobot
uv run gobot_editor --path examples/go1
```

Click the 3D viewer, then use `W/S` for forward/backward, `Q/E` for strafe,
`A/D` for heading, `Space` to stop, and `R` to reset. The keyboard command
limits default to the UniLab Go1 rough task ranges: `vx=1.0`, `vy=1.0`, and
heading-rate input `0.5`; the policy yaw command is produced by UniLab-style
heading feedback. Playback defaults to `200Hz` physics with `4` max substeps
and `50Hz` policy updates (`decimation=4`), matching the Go1 rough
configuration's `0.005s` physics step. Go1-specific hip/thigh/calf PD gains and
`trunk` reset height `0.32m` are built in. Sensor debug visualization is
enabled by the sensor nodes and velocity command debug is driven by the runtime
`VelocityCommandDebug3D` node.

Run the editor from an editable install and open the `examples/go1` project.
