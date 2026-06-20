# Go1 Example

This example keeps the robot project self-contained:

- `assets/xml/go1.xml` is the MJCF robot asset used for import.
- `assets/xml/go1_scene.xml` is the MJCF scene that includes `go1.xml`.
- `go1.jscn` is the Gobot robot scene imported from `go1.xml`.
- `go1_scene.jscn` is the Gobot scene that references `go1.jscn`.
- `train/go1_velocity_train.py` trains the Go1-owned velocity task into `policies/`.
- `train/go1_velocity_env.py` contains the Go1 rsl_rl vector environment.
- `train/go1_velocity_cfg.py` contains Go1 joints, rewards, PPO, command, and terrain curriculum settings.
- `scripts/go1.py` is attached to the `go1_scene.jscn` root and plays a trained policy in `gobot_editor`.
- `policies/go1.onnx` is the default lightweight rough-terrain playback policy.
- `policies/go1.pt` is the Torch checkpoint used by training, export, and fallback playback.
- `tools/export_policy_onnx.py` converts `policies/go1.pt` to `policies/go1.onnx`.
- `project.gobot` sets `go1_scene.jscn` as the project main scene.

Regenerate the Gobot scene after editing the MJCF:

```python
import gobot

gobot.set_project_path("/home/wqq/gobot/examples/go1")
gobot.import_mjcf_scene(
    "res://assets/xml/go1_scene.xml",
    "res://go1_scene.jscn",
    name="go1_scene",
    script="res://scripts/go1.py",
)
```

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

Benchmark the Gobot Go1 vector env hot path without the PPO learner:

```bash
cd /home/wqq/gobot
uv run --extra train python examples/go1/train/go1_velocity_benchmark.py \
  --task go1_flat \
  --num-envs 64 \
  --steps 100 \
  --warmup-steps 10 \
  --device cpu \
  --sim-workers 0 \
  --no-obs-noise
```

Benchmark a UniLab-style raw MuJoCo state-array batch step path:

```bash
cd /home/wqq/gobot
uv run --extra train python examples/go1/train/mujoco_uni_batch_benchmark.py \
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

The default Go1 rough task now includes terrain-normal upright reward, foot
contact history, rough-terrain illegal-contact penalties, per-env terrain
curriculum, encoder-bias/reset randomization, and scheduled base velocity
pushes. CUDA is used by default when PyTorch reports it as available; pass
`--device cpu` only when you explicitly want CPU PPO.

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

Play the mjlab-derived Go1 ONNX policy in the editor with keyboard velocity
commands:

```bash
cd /home/wqq/gobot
uv run gobot_editor --path examples/go1
```

Click the 3D viewer, then use `W/S` for forward/backward, `Q/E` for strafe,
`A/D` for yaw, `Space` to stop, and `R` to reset. The keyboard command limits
default to the mjlab Go1 rough task ranges: `vx=1.0`, `vy=1.0`, `yaw=0.5`.
Playback defaults to `500Hz` physics with `10` max substeps and `50Hz`
policy updates (`decimation=10`). At a `60Hz` editor render rate this averages
about `8.33` physics ticks per rendered frame, matching the Go1 training
configuration's `0.002s` physics step. Go1-specific hip/thigh/calf PD gains and
`trunk` reset height `0.278m` are built in. Sensor debug visualization is
enabled by the sensor nodes and velocity command debug is driven by the runtime
`VelocityCommandDebug3D` node.

Run the editor from an editable install and open the `examples/go1` project.
