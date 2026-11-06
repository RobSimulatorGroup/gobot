# Go1 and G1 locomotion

Go1/G1 velocity PPO, G1 AMP and G1 SuperMimic use the
[shared Gobot training entry](../../doc/gobot_training.md). Reusable velocity
configuration and training live in `gobot.rl.locomotion`; this directory contains
robot recipes, evaluation, export and cluster tools.

## Train and resume

```bash
python -m gobot.rl.train go1-velocity \
  --backend mujoco-cpu --num-envs 4 --iterations 10 --no-eval --output logs/go1_cpu

python -m gobot.rl.train g1-velocity \
  --backend mujoco-warp --config examples/locomotion/configs/g1_velocity.json \
  --output logs/g1_control

python -m gobot.rl.train g1-amp \
  --backend mujoco-warp --dataset /data/g1/locomotion.npz \
  --config examples/locomotion/configs/g1_amp.json \
  --output logs/g1_amp
```

`--list` shows the available task names. Python factories can supply custom workflows
without registration. `--backend mujoco-cpu|mujoco-warp` selects physics explicitly;
unsupported combinations fail before training. CPU physics may use a CUDA learner
through `--device cuda:0`.

`--iterations` counts PPO updates. `--resume CHECKPOINT` restores the same algorithm,
optimizer, AMP discriminator/replay, observation statistics, curriculum and RNG,
then starts fresh physical episodes. `--initialize CHECKPOINT` retains the actor and
its normalizer and starts a fresh critic, optimizer and exploration. Changing backend
or task semantics is a new experiment, not an optimizer resume.

Use `--terrain flat|rough` for standard velocity PPO; AMP and phase-gait tasks may
also select `mixed`. Training starts directly, without an admission gate or mandatory
initial checkpoint. `--hours` bounds a run and `--budget-hours` bounds its ledger.
SIGTERM/SIGINT request a checkpoint and orderly exit.

## Rewards and training settings

Recipes contain an `experiment` object; optional `env` and `agent` objects override
task and learner defaults. CLI overrides apply last:

```bash
python -m gobot.rl.train go1-velocity \
  --backend mujoco-cpu --num-envs 4 --iterations 10 --no-eval \
  --set env.rewards.foot_slip=-0.1 \
  --set agent.algorithm.learning_rate=0.0002 --output logs/custom --dry-run
```

Custom environment settings require `--no-eval` and an independent evaluator matching
that task. The built-in evaluator rebuilds the selected preset and must not silently
judge a different MDP. Prepare robot/motion assets first; generated evaluation terrain
requires a writable project.

Standard velocity presets follow the local mjlab reference with RSL-RL 5.5.2 PPO:
learning rate 1e-3, adaptive scheduling, entropy coefficient 0.01, scalar action
standard deviation initialized to 1.0, and 24 rollout steps per update. Experiment-specific
update guards and normalization warmups have been removed. Reward is integrated once
over `physics_dt × decimation`; physical termination and time limits remain separate.
AMP and SuperMimic keep their own additional learning objectives.

## Robots and terrain

| Robot | Policy task | Terrain |
| --- | --- | --- |
| G1 | Velocity PPO with optional human AMP prior | Flat, slopes, stairs, rough noise and waves |
| Go1 | Velocity PPO with optional walk/trot/gallop phase rewards | Flat, slopes, stairs, rough noise and waves |

Go1 blends walking into diagonal trot at 0.8 m/s and gallop at 2 m/s. Its phase clock
is observed during training and playback. It does not use animal motion capture.
G1 AMP matches supported forward speed and gait; standing, reverse and lateral
commands do not receive style reward. Subject 4 is validation and subject 5 is held
out. G1 uses world-upright rewards; Go1 may align to the terrain normal.

The practical terrain presets include ascending/descending stairs and slopes,
noise ±3 cm and waves up to 8 cm. Running on rough/mixed stages is limited to flat
terrain and gentle slopes. The standard curriculum compares traveled distance with integrated commanded displacement and the time remaining in the episode.

## Evaluate and export

```bash
python -m examples.locomotion.status logs/g1_amp
python -m examples.locomotion.assessment \
  --checkpoint logs/g1_amp/best.pt --output logs/g1_amp/validation.json
python -m examples.locomotion.admit \
  --checkpoint logs/g1_amp/best.pt --output logs/g1_amp/admission \
  --budget-file logs/admission-budget.json --budget-hours 1 --hours 1
python -m examples.locomotion.record \
  --checkpoint logs/g1_amp/best.pt --output logs/g1_amp/walk.mp4 --continuous --seconds 12
python -m examples.locomotion.verify_export \
  --checkpoint logs/g1_amp/best.pt --output logs/g1_amp/candidate.onnx
```

Validation reports velocity/yaw errors, cross-track deviation, slip, clearance,
per-cell success and measured footfall metrics. Admission uses held-out terrain
seeds, nominal and perturbed scenarios, command changes and sustained running.
Thresholds include ≥95% overall traversal, ≥90% per terrain/difficulty, velocity
error ≤max(0.15 m/s, 20% of command), yaw-rate error ≤0.2 rad/s and cross-track error
≤0.5 m. Resets and crossing a different terrain cell cannot count as success.

Gait validation measures contacts and stride cycles, not policy-clock or
discriminator scores. Export checks inference equivalence and embeds the policy
manifest. See the [Go1](../go1/README.md) and [G1](../g1/README.md) guides for playback.

## Cluster and validation

`python -m examples.locomotion.package --config CONFIG --output BUNDLE.tar.gz`
packages selected recipes, source and assets with hashes. Install/build Gobot on
the destination host and use the same training entry. PPO/AMP uses independent
runs per GPU; synchronized SuperMimic uses [the Slurm launcher](cluster/train_tracking.sbatch).

```bash
python tests/python/test_natural_locomotion.py
python tests/python/test_natural_locomotion.py --backend mujoco-cpu
python tests/python/test_locomotion_training.py --backend mujoco-cpu
python tests/python/test_shared_velocity_gpu.py
```

Short optimization tests validate the training pipeline, not learned-policy quality.
Formal G1 SuperMimic results are recorded in the
[100k BONES report](../../doc/g1_bones_100k_results.md).
