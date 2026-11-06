# G1 training in Gobot

G1 supports velocity PPO, motion-prior AMP, and SuperMimic motion tracking through
[one training entry](../../doc/gobot_training.md). The robot and terrain are Gobot
`.jscn` / `.jres` resources. MuJoCo CPU is the control/state reference; MuJoCo Warp
runs batched CUDA training.

The [100k BONES result](../../doc/g1_bones_100k_results.md) records the completed
SuperMimic eight-GPU run and complete-clip evaluation. See the
[reference/data tools](reproduction/README.md) for preparing and evaluating that task.

## Prepare velocity and AMP assets

```bash
uv run python -m examples.g1.prepare
uv run python -m examples.g1.prepare_motion
uv run python -m examples.g1.reference_gaits
```

Robot preparation reads Unitree assets from `examples/newton_g1/assets`, retaining
original inertias and visual meshes and adding the reference collision proxies and
link exclusions. `--asset-root` selects another asset copy. Motion preparation
records pinned LAFAN1 data provenance, validates joint order/quaternions and converts
30 Hz motion to 50 Hz using forward kinematics from the Gobot robot scene.

The velocity policy controls 29 body joints; 14 hand joints retain fixed targets.
Standard velocity PPO uses one observation frame; its flat task omits the terrain scan.
AMP adds ten proprioceptive frames and the terrain scan. The critic has privileged state. Physics runs at 200 Hz and control at 50 Hz.

## Train

```bash
uv run python -m gobot.rl.train g1-velocity \
  --backend mujoco-cpu --num-envs 4 --iterations 10 --no-eval --output logs/g1_cpu

uv run python -m gobot.rl.train g1-amp \
  --backend mujoco-warp --num-envs 1024 --dataset /data/g1/locomotion.npz \
  --iterations 2000 --hours 1 --output logs/g1_amp

uv run python -m gobot.rl.train g1-amp \
  --backend mujoco-warp --num-envs 1024 --dataset /data/g1/locomotion.npz \
  --resume logs/g1_amp/latest.pt --iterations 1000 --output logs/g1_amp_resume
```

Use `--terrain flat|rough` for standard velocity PPO. AMP also supports `mixed`.
Training can start from scratch. Resume restores compatible optimizer, discriminator,
normalization, curriculum and RNG state, then starts new physics episodes.

Use `--set env.rewards.NAME=VALUE` and `--set agent.algorithm.NAME=VALUE` for task and
learner settings; `--dry-run` validates configuration before simulation. A custom
environment requires `--no-eval` and a matching independent evaluator. Details,
including Python factory configuration, are in the training guide.

## Evaluate, export and play

```bash
uv run python -m examples.locomotion.assessment \
  --checkpoint logs/g1_amp/best.pt --output logs/g1_amp/validation.json
uv run python -m examples.locomotion.record \
  --checkpoint logs/g1_amp/best.pt --continuous --seconds 20 \
  --output logs/g1_amp/walk.mp4
uv run python -m examples.locomotion.verify_export \
  --checkpoint logs/g1_amp/best.pt --output logs/g1_amp/candidate.onnx
uv run gobot_editor --path examples/g1
```

Evaluation measures physical velocity/yaw errors, traversal, clearance, contact and
footfall metrics. A failure ends the recorded rollout; automatic reset is never
presented as recovery. Export checks Torch/ONNX inference with the observation
normalizer and embeds the policy contract.

Set `GOBOT_G1_POLICY` to the exported ONNX for editor playback. W/S controls forward
velocity, Q/E strafe, A/D yaw, T running command, and R reset. Play uses a runtime
clone and does not modify the edited scene. Running is intended for flat and gentle
terrain. See [shared locomotion](../locomotion/README.md) for full admission criteria.

## Validation

```bash
uv run python tests/python/test_g1_amp.py
uv run python tests/python/test_g1_amp.py --backend mujoco-cpu
uv run python tests/python/test_shared_velocity_gpu.py
uv run python tests/python/test_locomotion_training.py --backend mujoco-cpu
```

The tests cover task contracts and short optimization/resume runs; policy admission
is measured separately. Reference repositories and licensing are described in `NOTICE`.
