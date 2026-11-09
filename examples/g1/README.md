# G1 training in Gobot

G1 supports velocity PPO, motion-prior AMP, and SuperMimic motion tracking through
[one training entry](../../doc/gobot_training.md). The robot and terrain are Gobot
`.jscn` / `.jres` resources. MuJoCo CPU is the control/state reference; MuJoCo Warp
runs batched CUDA training.

See the [reference/data tools](reproduction/README.md) for preparing and evaluating
SuperMimic motion tracking.

## Locomotion model and AMP data

The checked-in `g1.jscn` and `g1.meshes/` contain the 29-joint G1 with rigid palms
from mjlab `dcdf6ce5`. Training, evaluation and playback use this same asset.
Masses, inertias, collision shapes and drives are checked against the reference;
there are no finger actuators. No separate mesh download is needed.

To deliberately regenerate the robot from that reference checkout:

```bash
uv run python -m examples.g1.prepare --source mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml
```

AMP motion data must be prepared for the current robot model:

```bash
uv run python -m examples.g1.prepare_motion
uv run python -m examples.g1.reference_gaits
```

Old articulated-hand checkpoints describe different dynamics. Normal resume and
playback reject them. Use the evaluator's explicit `--allow-model-change` option
for a measured policy-transfer experiment; this does not migrate optimizer state.

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
normalizer and embeds the policy metadata.

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
