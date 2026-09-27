# Cartpole

Balance the pole and track a target cart position. Training and editor playback
share `cartpole.jscn`, the seven-input observation contract, force limits, and a
240 Hz physics step in `cartpole_task.py`.

Run from the repository root with an editable Gobot installation:

```bash
uv run gobot_editor --path examples/cartpole
uv run python -m examples.cartpole.train.train --device cpu --num-envs 64 --iterations 10
```

Physics uses Gobot's MuJoCo CPU batch (mjbatch). `--sim-workers` selects the CPU
worker count; `--device` selects the learner device. Training does not run node
scripts. Episode resets randomize cart/pole state and the target position.

Relative log and policy paths are resolved under `examples/cartpole`.
The default training output is `policies/cartpole.pt`; export it for playback:

```bash
uv run python -m examples.cartpole.tools.export_policy_onnx
```

The NumPy task in `train/env.py` returns `BatchEnvState`; training uses the
shared `gobot.rl.rsl_rl.RslRlVecEnvWrapper` for TensorDict/PyTorch conversion.
