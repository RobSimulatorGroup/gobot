# Locomotion examples

The reusable trainer and environment live in `gobot.rl.locomotion`.
This directory contains robot presets, evaluation, export and cluster tools.

- [Training, configuration, resume and custom robots](../../doc/gobot_training.md)
- [Paired mjlab training](../../doc/mjlab_comparison.md)
- [Go1 assets and playback](../go1/README.md)
- [G1 assets, AMP and playback](../g1/README.md)
- [G1 motion tracking](../g1/reproduction/README.md)

## Commands

```bash
python -m gobot.rl.train go1-velocity \
  --backend mujoco-cpu --num-envs 4 --iterations 10 --no-eval --output logs/go1_cpu
python -m examples.locomotion.assessment \
  --checkpoint logs/go1_cpu/model_final.pt --terrain flat --output logs/go1_cpu/evaluation.json
python -m examples.locomotion.record \
  --checkpoint logs/go1_cpu/model_final.pt --output logs/go1_cpu/walk.mp4 --continuous
python -m examples.locomotion.verify_export \
  --checkpoint logs/go1_cpu/model_final.pt --output logs/go1_cpu/policy.onnx
```

Use `go1-gait` for walk/trot/gallop phase rewards and `g1-amp` for the human motion prior.
These are explicit task extensions; standard velocity PPO uses the reference task rewards.

`status` reads a run directory. `admit` evaluates multiple held-out scenarios.
`package` bundles selected configs, sources and assets with hashes.
Slurm launchers are in `cluster/`; velocity PPO and motion tracking support `torchrun`,
while AMP uses one training process.
