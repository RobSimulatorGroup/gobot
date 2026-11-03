# Shared Go1/G1 velocity validation

Recorded on 2026-10-25 using the local MuJoCo 3.17.0 / MuJoCo Warp 3.17.0
stack and an NVIDIA GeForce RTX 4070 (12 GiB). These are engineering smoke
results, not policy-admission results.

## Build and correctness

- CMake built `gobot_python`, `test_locomotion_batch_runtime`, and
  `test_locomotion_command_runtime`; all 13 locomotion C++ cases passed.
- Existing Go1 CPU environment, released ONNX loading, policy manifest and
  provider Play lifecycle tests passed.
- Existing Go1 Warp environment and generic MuJoCo Warp provider tests passed.
- Shared task tests passed: grouped feet, profile validation, command-progress
  curriculum, solver-limit/capacity classification, deterministic reset, masked
  reset, held hand joints, measured angular momentum, checkpoint mode validation,
  and authored ray-pattern translation.
- G1 CPU/Warp reset observations and four control steps of state and all reward
  terms passed the configured numerical tolerances. Maximum measured base-position
  difference was 7.63e-6 m and joint-position difference 5.99e-5 rad.
- The editable Python package was rebuilt/reinstalled successfully. A normal
  Python entry point created the G1 environment with 29 actions, 286 actor
  observations and 298 critic observations.

The parity test identified and fixed a CPU state-timing issue: `mj_step` had
integrated joint positions while derived body positions still described the
previous physical tick. New tasks refresh end-of-step sensing and retain
substep contact history. Existing Go1 presets explicitly retain legacy timing.

## Training and artifacts

| Robot | Backend | Environments | PPO updates | Resume updates |
| --- | --- | ---: | ---: | ---: |
| Go1 | MuJoCo CPU | 64 | 10 | 2 |
| Go1 | MuJoCo Warp | 256 | 10 | 2 |
| G1 | MuJoCo CPU | 64 | 10 | 2 |
| G1 | MuJoCo Warp | 256 | 10 | 2 |

Each run saved a manifest-backed checkpoint and restored curriculum/RNG state
before continuing. Local smoke artifacts are under `/tmp/gobot-shared-*` and
are not bundled as released policies.

G1 ONNX export passed. ONNX Runtime and the checkpoint's Torch actor differed by
at most 1.79e-7 on a fixed random input batch. The real editor completed a bounded
Play run with the exported actor, 60 measured frames, an active external world,
and exit status zero. The Play script targets the runtime clone and leaves
scene ownership to the editor.

The evaluation tool traversed all 70 authored cells for a 20-step smoke rollout.
It reported survival 1.0 and forward admission 0.0: the short-trained actor had
not learned the requested speed. Full 1000-step, multi-seed policy admission and
long training remain separate experiments.

## Go1 throughput check

Compared the previous environment implementations from the initial Git HEAD
with the shared implementations on the same native runtime. Observation noise
was disabled; other preset settings were retained. Each result is the median
of three synchronized 100-step windows after 20 warmup steps.

| Execution | Batch | Before, env-steps/s | Shared, env-steps/s | Change |
| --- | ---: | ---: | ---: | ---: |
| CPU, 4 workers | 64 | 1,970 | 1,841 | -6.5% |
| Warp | 256 | 14,752 | 15,296 | +3.7% |

These short measurements include runtime variation and are not portable speed
claims. Singleton-foot GPU tasks retain direct sensor-buffer views; multi-shape
feet aggregate into persistent device buffers. GPU tests verify reset does not
replace captured state storage.

## Repeat

Use the shared commands in [the G1 guide](../examples/g1/README.md).
For focused tests after preparing the assets:

```bash
.venv/bin/python tests/python/test_shared_velocity.py
.venv/bin/python tests/python/test_shared_velocity.py --integration
.venv/bin/python tests/python/test_shared_velocity_gpu.py
.venv/bin/python tests/python/test_velocity_env.py
.venv/bin/python tests/python/test_go1_mujoco_warp_env.py
```

The GPU tests require CUDA access. G1 integration tests explicitly skip when
its prepared mesh cache is absent; configuration and curriculum unit tests run
independently of those assets.
