# G1 velocity training and playback

G1 supports velocity PPO, motion-prior AMP, BeyondMimic tracking, and SuperMimic through
[one training entry](../../doc/gobot_training.md). The robot and terrain are Gobot
`.jscn` / `.jres` resources. MuJoCo CPU is the control/state reference; MuJoCo Warp
runs batched CUDA training. G1 assets, sensor placement and presets live in this
example and `python/gobot/rl/locomotion/presets.py`. The shared trainer consumes a typed
task configuration and does not select robot models.

See the [reference/data tools](supermimic/README.md) for preparing and evaluating
SuperMimic motion tracking.

The single-motion player is [G1 tracking](../g1_track/README.md).
BeyondMimic preparation, synchronized training and editor inference: [Tracking guide](../../doc/gobot_tracking.md).

## Locomotion model and AMP data

The checked-in `g1.jscn` and `g1.meshes/` contain the 29-joint G1 with rigid palms
from mjlab `dcdf6ce5`. Training, evaluation and playback use this same asset.
Masses, inertias, collision shapes, drives and the pelvis IMU placement are checked against the reference;
there are no finger actuators. No separate mesh download is needed.

To deliberately regenerate the robot from that reference checkout:

```bash
uv run python -m examples.g1_velocity.prepare --source mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml
```

AMP motion data records named joints and tracked links. The trainer validates
their layout, control period and robot physics identity. Standard velocity PPO
does not need a motion dataset:

```bash
uv run python -m examples.g1_velocity.prepare_motion
uv run python -m examples.g1_velocity.reference_gaits
```

Standard velocity PPO uses one observation frame; its flat task omits the terrain scan.
AMP adds ten proprioceptive frames and the terrain scan. The critic has privileged
state. Physics runs at 200 Hz and control at 50 Hz. The standard velocity task uses
the mjlab event order and RSL-RL PPO; AMP uses terminal observations for timeout
bootstrapping. See the [training guide](../../doc/gobot_training.md) for state timing,
randomness, numerical handling and checkpoint contracts.

## Train

```bash
uv run python -m gobot.rl.train g1-velocity \
  --backend mujoco-cpu --num-envs 4 --iterations 10 --no-eval --output logs/g1_velocity_cpu

uv run python -m gobot.rl.train g1-amp \
  --backend mujoco-warp --num-envs 1024 --dataset /data/g1/locomotion.npz \
  --iterations 2000 --hours 1 --output logs/g1_amp

uv run python -m gobot.rl.train g1-amp \
  --backend mujoco-warp --num-envs 1024 --dataset /data/g1/locomotion.npz \
  --resume logs/g1_amp/latest.pt --iterations 1000 --output logs/g1_amp_resume
```

Use `--terrain flat|rough` for standard velocity PPO. AMP also supports `mixed`.
Resume restores optimizer, discriminator, normalization, curriculum and RNG state, then
starts new physics episodes without changing saved terrain assignments. Resume
requires the same world size and environments per rank.

Use `--set env.rewards.NAME=VALUE` and `--set agent.algorithm.NAME=VALUE` for task and
learner settings; `--dry-run` validates configuration before simulation. A custom
environment requires `--no-eval` and a matching independent evaluator. Details,
including custom robot factories returning `VelocityTaskCfg`, are in the training
guide. A new robot supplies its own scene, joint/control profile and named sensors;
the training loop does not need a new robot branch.

## Evaluate, export and play

```bash
uv run python -m gobot.rl.locomotion.assessment \
  --checkpoint logs/g1_velocity/model_final.pt --output logs/g1_velocity/validation.json
uv run python -m gobot.rl.locomotion.record \
  --checkpoint logs/g1_velocity/model_final.pt --continuous --seconds 20 \
  --output logs/g1_velocity/walk.mp4
uv run python -m gobot.rl.locomotion.verify_export \
  --checkpoint logs/g1_velocity/model_final.pt --output logs/g1_velocity/candidate.onnx
GOBOT_G1_POLICY="$PWD/logs/g1_velocity/candidate.onnx" \
  uv run gobot_editor --path examples/g1_velocity
```

Evaluation measures physical velocity/yaw errors, traversal, clearance, contact and
footfall metrics. A failure ends the recorded rollout; automatic reset is never
presented as recovery. Export checks Torch/ONNX inference with the observation
normalizer and embeds the policy metadata.

No pretrained velocity weights are bundled with this project. Export a trained `g1-velocity`
checkpoint and set `GOBOT_G1_POLICY` to its ONNX path before starting the editor.
AMP and tracking policies use different observation contracts
and cannot be used by this velocity playback script. W/S controls forward
velocity, Q/E strafe, A/D yaw, T running command, and R reset. Play uses a runtime
clone and does not modify the edited scene. Running is intended for flat and gentle
terrain. See [shared locomotion](../../doc/gobot_training.md) for full admission criteria.

## Validation

```bash
uv run python tests/python/test_g1_amp.py
uv run python tests/python/test_g1_amp.py --backend mujoco-cpu
uv run python tests/python/test_shared_velocity_gpu.py
uv run python -m pytest -q tests/python/test_motion_schema.py tests/python/test_velocity_reliability.py
uv run python tests/python/test_locomotion_training.py --backend mujoco-cpu
```

The tests cover task contracts and short optimization/resume runs; policy admission
is measured separately. Reference repositories and licensing are described in `NOTICE`.

SuperMimic 世界模型动作跟踪的独立入口见 [SuperMimic 指南](supermimic/README.md)。
