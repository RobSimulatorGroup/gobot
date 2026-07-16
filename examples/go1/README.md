# Go1 Example

This example keeps the robot project self-contained:

- `assets/xml/go1.xml` is the original MJCF robot asset retained for import/equivalence checks.
- `go1_profile.py` owns the example-local Go1 articulation and controller profile.
- `go1_velocity_contract.py` owns the versioned policy identity shared by training and playback.
- `go1.jscn` is the Gobot robot scene imported from `go1.xml`.
- `go1_scene.jscn` references `go1.jscn` and owns the `terrain_world/terrain` authoring nodes.
- `terrain/rough_terrain.jres` is the versioned procedural terrain recipe used by the editor, playback, and training.
- `train/go1_velocity_train.py` trains the Go1-owned velocity task into `policies/`.
- `train/go1_velocity_env.py` is the MuJoCo CPU semantic baseline.
- `train/go1_warp_velocity_env.py` is the CUDA-native MuJoCo Warp task path.
- `train/go1_scene_runtime.py` applies the shared scene, solver, controller, and terrain contract before either backend compiles its runtime.
- `train/go1_training_state.py` preserves and resizes terrain-curriculum assignments across checkpoints.
- `train/go1_velocity_cfg.py` contains rewards, PPO, command, solver, and terrain spawn-curriculum settings.
- `train/go1_velocity_video.py` owns optional checkpoint-triggered evaluation video capture.
- `scripts/go1.py` is attached to the `go1_scene.jscn` root and plays a trained policy in `gobot_editor`.
- `policies/go1_velocity.pt` is the default training output.
- `policies/go1_velocity.onnx` is generated from that checkpoint for lightweight playback.
- `tools/checkpoint_policy.py` is the shared deterministic checkpoint actor used by evaluation and ONNX export.
- `tools/evaluate_velocity_policy.py` evaluates policy admission over every authored terrain cell.
- `tools/export_policy_onnx.py` validates the checkpoint manifest and embeds it in ONNX.
- `tools/export_gobot_trace.py`, `tools/export_reference_trace.py`, `tools/compare_go1_traces.py`, and `tools/go1_parity.py` form the pinned reference-parity workflow.
- `tools/refresh_go1_robot_scene.py` regenerates the editable robot scene from the source MJCF.
- `project.gobot` sets `go1_scene.jscn` as the project main scene.

## Validated Behavior (Gobot 0.1.12, 2026-07-16)

The current local playback policy was selected from `model_5100.pt` through
`model_6000.pt` by fixed-seed terrain-cell admission, not by training reward or
checkpoint age. `model_5600.pt` is the selected balance: later checkpoints
tracked yaw more aggressively but regressed forward rough-terrain admission.
Generated checkpoints and exported policies remain local artifacts and are not
committed to the repository.

The comparison below used all 70 authored terrain cells, seed `123`, and the
same policy-admission threshold for each checkpoint. Forward evaluation used
two 1000-step episodes per cell; yaw and 2 m/s run evaluation used one 500-step
episode per cell.

| Metric | Previous `model_5100` | Selected `model_5600` |
| --- | ---: | ---: |
| Forward 1 m/s admission | 0.829 | 0.836 |
| Forward command-direction speed | 0.831 m/s | 0.837 m/s |
| Yaw 0.5 rad/s admission | 0.043 | 0.914 |
| Yaw command ratio | 0.348 | 0.514 |
| Run 2 m/s admission | 0.814 | 0.829 |
| Run command-direction speed | 1.587 m/s | 1.637 m/s |

Directional checks for the selected policy produced:

| Command | Admission | Survival | Measured command-direction rate |
| --- | ---: | ---: | ---: |
| Forward 1 m/s | 0.836 | 0.936 | 0.837 m/s |
| Left strafe 1 m/s | 0.857 | 0.929 | 0.821 m/s |
| Reverse 1 m/s | 0.857 | 0.943 | 0.890 m/s |
| Yaw 0.5 rad/s | 0.914 | 1.000 | 0.257 rad/s |
| Forward run 2 m/s | 0.829 | 0.829 | 1.637 m/s |

Editor Play was also checked through the exported ONNX policy. Runtime logs
showed forward body velocity around `0.75-1.02 m/s`, reverse velocity around
`-0.80 m/s`, and yaw response around `+/-0.4 rad/s`; the robot traversed from a
regular pyramid slope L4 cell into an inverted pyramid-stairs L5 cell without a
false world-height reset.

What changed to produce and verify this improvement:

- Playback now uses terrain-relative base clearance for fall detection. A
  below-zero terrain cell no longer causes a false reset solely from world Z.
- Diagonal planar keyboard commands are normalized, and `Shift+W/S` exposes
  the trained `[-1.5, 2.0] m/s` run range.
- Runtime logs identify the nearest authored terrain type and exact level.
- Checkpoints preserve command progress, terrain level/type assignments, and
  RNG state. Resume no longer silently resets a learned terrain curriculum.
- The evaluator requires survival plus planar and yaw command progress. A
  stationary policy can no longer pass merely by avoiding a fall.
- Checkpoint selection considers forward, reverse, strafe, yaw, and run
  commands instead of selecting the final training iteration automatically.

Known limit: inverted pyramid slopes remain the hardest terrain. At the fixed
1 m/s evaluation, L4 averaged only about `0.074 m/s`; at a 2 m/s command the
inverted-slope terrain type admitted 40% of cells. This is still an open policy
capability issue, not a terrain-loading or keyboard-input issue.

Regenerate the Gobot robot asset after editing the MJCF. The current editable
install supplies `gobot`; no source/build `PYTHONPATH` is needed:

```bash
cd /home/wqq/gobot
uv run python -m examples.go1.tools.refresh_go1_robot_scene
```

This refreshes `go1.jscn` from `assets/xml/go1.xml` and restores the
Gobot-authored runtime sensors used by training. The scene keeps a compact
reference to `terrain/rough_terrain.jres`; native `Terrain3D` generation lazily
builds render and collision geometry without serializing generated height
arrays. The terrain is visible as soon as the scene is opened, before entering
Play Mode, and training consumes the same authored recipe.

Train from the repository root:

```bash
cd /home/wqq/gobot
uv run --extra train --extra mujoco-warp \
  python -m examples.go1.train.go1_velocity_train \
  --backend mujoco-warp \
  --num-envs 2048 \
  --iterations 10000 \
  --device cuda:0 \
  --no-step-extras \
  --log-dir logs/go1_rough_velocity \
  --policy-out policies/go1_velocity.pt
```

For a small CPU smoke run, use the built-in preset instead of a separate entry
script:

```bash
cd /home/wqq/gobot
uv run --extra train python -m examples.go1.train.go1_velocity_train --cpu-batch --iterations 10
```

The equivalent fully explicit CPU selection is:

```bash
uv run --extra train python -m examples.go1.train.go1_velocity_train \
  --backend mujoco-cpu \
  --device cpu \
  --num-envs 64 \
  --iterations 10
```

Benchmark the Gobot Go1 vector env hot path without the PPO learner:

```bash
cd /home/wqq/gobot
uv run --extra train python benchmark/go1_velocity_benchmark.py \
  --backend mujoco-cpu \
  --num-envs 64 \
  --steps 100 \
  --warmup-steps 10 \
  --device cpu \
  --sim-workers 0 \
  --no-obs-noise \
  --profile-step
```

Benchmark the same task on MuJoCo Warp. The measured interval is synchronized
before and after all steps so CUDA enqueue time is not reported as simulation
throughput:

```bash
uv run --extra train --extra mujoco-warp \
  python benchmark/go1_velocity_benchmark.py \
  --backend mujoco-warp \
  --num-envs 2048 \
  --steps 100 \
  --warmup-steps 20 \
  --device cuda:0 \
  --no-step-extras \
  --json-out /tmp/go1_warp_benchmark.json
```

`--profile-step` reports the major Go1 CPU batch env phases: action
preparation/application, physics, state refresh, command update, contact
updates, reward, termination/reset, observation build, tensor conversion, and
extras/logging.

The checked reference trace pins the upstream rough-terrain task revision and
compares joint order, observation dimensions, all 70 terrain assignments and
reset scans, all 40 physical heightfield fingerprints, stable first-step
dynamics, all 16 reward terms, termination flags, and an eight-step flat state
trajectory. The first impact on wave terrain is shape/finite checked because
the pinned reference itself is not bitwise repeatable there. The reference and
Gobot traces must use identical MuJoCo, MuJoCo Warp, Warp, and Torch versions;
a dynamics-stack version mismatch is a parity failure.

Regenerate and compare the traces with the two independent environments:

```bash
# Maintainer-only: install the pinned reference checkout at the revision and
# with the compatibility patch recorded in examples/go1/tools/go1_parity.py.
MPLCONFIGDIR=/tmp/gobot-matplotlib .venv/bin/python \
  -m examples.go1.tools.export_reference_trace \
  --output /tmp/go1_reference_trace.json \
  --device cuda:0

uv run --extra train --extra mujoco-warp \
  python -m examples.go1.tools.export_gobot_trace \
  --output /tmp/go1_gobot_trace.json

uv run --no-sync python -m examples.go1.tools.compare_go1_traces \
  /tmp/go1_reference_trace.json /tmp/go1_gobot_trace.json
```

The reference exporter and Gobot exporter execute independent environment
implementations with an identical dynamics stack. The reference checkout and
its compatibility patch are fixture-generation dependencies only; they are not
Gobot runtime dependencies. The fixture records each MuJoCo, MuJoCo Warp,
Torch, Warp, and task revision.

Evaluate one or more checkpoints over every authored terrain cell. The report
includes planar and yaw command progress, tracking error, survival, combined
admission, and illegal-contact rates by terrain level, type, and individual
level/type cell:

```bash
uv run --extra train --extra mujoco-warp \
  python -m examples.go1.tools.evaluate_velocity_policy \
  examples/go1/logs/go1_rough_velocity/model_900.pt \
  --device cuda:0 \
  --command-x 1.0 \
  --command-y 0.0 \
  --command-yaw 0.0 \
  --episodes-per-cell 2 \
  --max-steps 1000 \
  --min-progress-ratio 0.5 \
  --json-out /tmp/go1_velocity_evaluation.json
```

The defaults intentionally use the editor's full forward command and a complete
20-second training episode. Shorter `0.5 m/s` or 10-second rollouts are useful
for smoke tests but are not a rough-terrain stability admission check. An
episode is admitted only when it survives and averages at least half of each
requested planar-speed and yaw-rate component, so a policy that stands still in
a terrain depression or ignores a turn command does not pass.

The `go1_rough_velocity` task follows the Go1 rough-terrain observation,
reward, command, event, robot-dynamics, collision, and mixed-terrain settings
while loading the Gobot `.jscn` scene as the source of truth. `--backend`
selects `mujoco-warp` or `mujoco-cpu`; there is no implicit fallback. Warp keeps
simulation, sensing, task state, observations, rewards, resets, and policy
actions on CUDA. For the CPU backend, `--device` only selects the PyTorch
learner device and `--sim-workers` controls native CPU stepping.

Resume from the latest checkpoint in the log directory:

```bash
cd /home/wqq/gobot
uv run --extra train --extra mujoco-warp \
  python -m examples.go1.train.go1_velocity_train \
  --backend mujoco-warp \
  --num-envs 2048 \
  --iterations 10000 \
  --device cuda:0 \
  --no-step-extras \
  --render-video-interval 100 \
  --log-dir logs/go1_rough_velocity \
  --policy-out policies/go1_velocity.pt \
  --resume
```

Current checkpoints preserve command-curriculum progress, terrain level/type
assignments, and the environment random generator. Resuming with a different
`--num-envs` keeps each terrain type's level distribution. Legacy checkpoints
that only contain `common_step_counter` report that terrain levels restart from
the authored initial range instead of silently claiming an exact resume. Active
MuJoCo episode state is intentionally not checkpointed; resume resets fresh
episodes with the restored scheduler, curriculum, and random generator.

Export a trained checkpoint for lightweight editor playback:

```bash
cd /home/wqq/gobot
uv run --extra train python -m examples.go1.tools.export_policy_onnx \
  --checkpoint examples/go1/policies/go1_velocity.pt \
  --output examples/go1/policies/go1_velocity.onnx
```

The exporter embeds the same `gobot.rl.PolicyManifest` written by training and
writes `go1_velocity.onnx.manifest.json`. Playback rejects policies without this
manifest, mismatched observation/action specs, a different joint order, or a
different scene resource digest. This makes stale 45-dimensional policies fail
at load time instead of accepting keyboard commands that the policy cannot use.

To play the Torch checkpoint directly:

```bash
cd /home/wqq/gobot
GOBOT_GO1_POLICY=res://policies/go1_velocity.pt \
  uv run --extra train gobot_editor --path examples/go1
```

After exporting `policies/go1_velocity.onnx`, play it in the editor with
keyboard velocity commands:

```bash
cd /home/wqq/gobot
uv run gobot_editor --path examples/go1
```

Playback prefers `policies/go1_velocity.onnx` and then tries the training
checkpoint `policies/go1_velocity.pt`. Both paths require the current policy
manifest; older checkpoints are rejected instead of being guessed from tensor
dimensions. A missing or rejected policy aborts Play with the exact validation
error. Rough terrain remains available in the editor without entering Play.

Click the 3D viewer, then use `W/S` for forward/backward, `Q/E` for strafe,
`A/D` for yaw rate, `Shift+W/S` for the trained run range, `Space` to stop, and
`R` to reset. The normal command limits are `vx=1.0`, `vy=1.0`, and
`yaw_rate=0.5`; run mode uses `vx=[-1.5, 2.0]`. Physics rate, policy decimation,
PD gains, action scaling, reset height, and solver settings come from the
policy manifest. Terrain geometry comes from the versioned scene resource and
is covered by the manifest's scene bundle digest.

Run the editor from an editable install and open the `examples/go1` project.
