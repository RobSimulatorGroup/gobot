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
- `policies/go1.onnx` is the default lightweight playback policy.
- `policies/go1.pt` is the training checkpoint fallback for environments with `gobot[train]`.
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

The default Go1 rough task now includes terrain-normal upright reward, foot
contact history, rough-terrain illegal-contact penalties, per-env terrain
curriculum, encoder-bias/reset randomization, and scheduled base velocity
pushes. CUDA is used by default when PyTorch reports it as available; pass
`--device cpu` only when you explicitly want CPU PPO.

Resume from the latest checkpoint in the log directory:

```bash
cd /home/wqq/gobot
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
uv run --extra train python examples/go1/tools/export_policy_onnx.py \
  --checkpoint examples/go1/policies/go1_velocity.pt \
  --output examples/go1/policies/go1.onnx
```

The default Gobot install can play ONNX policies. Install `gobot[train]` only
when training or directly loading `.pt` checkpoints.

Run the editor from an editable install and open the `examples/go1` project.
