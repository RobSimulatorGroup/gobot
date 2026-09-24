# Go1: training and playback

The editable `go1_scene.jscn` and `terrain/rough_terrain.jres` are shared by
training and playback. MuJoCo CPU is the reference backend; MuJoCo Warp keeps
the full task on CUDA. Run commands from the repository root using an
[editable installation](../../doc/building.md).

## Play

```bash
uv run gobot_editor --path examples/go1
```

Press Play and focus the 3D view. Use `W/S` for forward/backward, `Q/E` for
strafe, `A/D` for yaw, `Space` to stop, and `R` to reset. Planar speed is limited
to 1 m/s and yaw to 0.5 rad/s.

Playback uses the bundled balanced policy, `policies/go1_velocity.onnx`.
Override it with a manifest-backed `.onnx` or `.pt` policy:

```bash
GOBOT_GO1_POLICY=res://policies/go1_velocity.pt \
  uv run gobot_editor --path examples/go1
```

Missing policies or mismatched manifests abort Play with a diagnostic. The
manifest supplies joint order, control/solver settings, physics rate, and
observation/action specs; its scene digest also covers terrain resources.

## Train

Small CPU smoke run (64 environments):

```bash
uv run python -m examples.go1.train.go1_velocity_train --cpu-batch --iterations 10
```

CUDA training:

```bash
uv run python -m examples.go1.train.go1_velocity_train \
  --backend mujoco-warp --device cuda:0 --num-envs 256 \
  --iterations 10000 --no-step-extras \
  --log-dir logs/go1_rough_velocity --policy-out policies/go1_velocity.pt
```

For explicit CPU selection, use `--backend mujoco-cpu --device cpu --num-envs 64`.
`--sim-workers` controls CPU physics workers (`0` selects hardware concurrency,
capped by environment count). On CPU, `--device` selects the PyTorch learner's
device. Requested backends never silently fall back.

Add `--resume` with the same `--log-dir` to load its latest checkpoint, or use
`--checkpoint <path>`. Resume restores curriculum, terrain distribution, and
RNG state, then starts fresh physics episodes. Changing batch size preserves
per-terrain level distributions. `--render-video-interval 100` records an
evaluation video every 100 iterations.

Fine-tune running from an admitted balanced checkpoint:

```bash
uv run python -m examples.go1.train.go1_velocity_train \
  --backend mujoco-warp --device cuda:0 --num-envs 256 \
  --training-profile run --checkpoint examples/go1/policies/go1_velocity.pt \
  --iterations 1000 --save-interval 25 --no-step-extras \
  --log-dir logs/go1_warp_run --policy-out policies/go1_velocity_run_candidate.pt
```

The `run` profile adds paired-leg gait shaping and speed curriculum. Switching
from `balanced` restarts only the profile-local curriculum; resuming `run`
restores it. Checkpoints and logs are local outputs. Evaluate candidates before
promoting an exported actor to the bundled ONNX path.

## Evaluate and export

Evaluate survival and commanded progress over all authored terrain cells:

```bash
uv run python -m examples.go1.tools.evaluate_velocity_policy \
  examples/go1/policies/go1_velocity.pt --device cuda:0 \
  --command-x 1.0 --command-y 0.0 --command-yaw 0.0 \
  --episodes-per-cell 2 --max-steps 1000 --min-progress-ratio 0.5 \
  --json-out /tmp/go1_velocity_evaluation.json
```

Admission requires survival and at least half of each requested planar/yaw
command. Check forward, reverse, strafe, yaw, and run commands. Reports include
terrain-specific results, illegal contacts, and gait metrics; survival or
training reward alone is insufficient. Short smoke rollouts do not replace
full terrain admission.

```bash
uv run python -m examples.go1.tools.export_policy_onnx \
  --checkpoint examples/go1/policies/go1_velocity.pt \
  --output examples/go1/policies/go1_velocity.onnx
```

Export validates and embeds `gobot.rl.PolicyManifest` and writes a
`.manifest.json` sidecar. Playback rejects incompatible contracts.

## Benchmark and parity

Measure CPU task throughput without PPO:

```bash
uv run python benchmark/go1_velocity_benchmark.py \
  --backend mujoco-cpu --device cpu --num-envs 64 --sim-workers 0 \
  --steps 100 --warmup-steps 10 --no-obs-noise --profile-step
```

Measure CUDA throughput (the benchmark synchronizes the measured interval):

```bash
uv run python benchmark/go1_velocity_benchmark.py \
  --backend mujoco-warp --device cuda:0 --num-envs 256 \
  --steps 100 --warmup-steps 20 --no-step-extras \
  --json-out /tmp/go1_warp_benchmark.json
```

CPU profiling reports action, physics, state, command, contact, reward, reset,
observation, tensor-conversion, and logging costs. See the
[mjbatch measurements](../../doc/mjbatch_cpu.md) for CPU memory and throughput.

Reference parity uses independent implementations with identical MuJoCo,
MuJoCo Warp, Warp, and Torch versions. The pinned reference revision and
compatibility patch are recorded in [go1_parity.py](tools/go1_parity.py).
After preparing that reference environment, regenerate and compare traces:

```bash
MPLCONFIGDIR=/tmp/gobot-matplotlib .venv/bin/python \
  -m examples.go1.tools.export_reference_trace \
  --output /tmp/go1_reference_trace.json --device cuda:0
uv run python -m examples.go1.tools.export_gobot_trace \
  --output /tmp/go1_gobot_trace.json
uv run --no-sync python -m examples.go1.tools.compare_go1_traces \
  /tmp/go1_reference_trace.json /tmp/go1_gobot_trace.json
```

Parity checks joint order, observation dimensions, 70 terrain assignments,
40 heightfields, reset scans, 16 reward terms, termination, and short state
trajectories. Wave-impact checks cover shape/finiteness because the pinned
reference is not bitwise repeatable there. GPU CI isolates the fixture's pinned
dynamics stack from the normal runtime dependencies.

## Project files

| Files | Responsibility |
| --- | --- |
| `go1.jscn`, `go1_scene.jscn`, `terrain/` | Authored robot and terrain |
| `go1_profile.py`, `go1_velocity_contract.py` | Articulation/controller profile and policy identity |
| `train/go1_scene_runtime.py`, `train/go1_velocity_cfg.py` | Shared scene/solver setup and task configuration |
| `train/go1_velocity_env.py`, `train/go1_warp_velocity_env.py` | CPU and CUDA task implementations |
| `train/go1_training_state.py`, `train/go1_gait.py` | Curriculum checkpoints and run gait scoring |
| `scripts/go1.py`, `tools/` | Editor playback, evaluation, export, and parity tools |
| `assets/xml/go1.xml` | Import/equivalence source asset |

After changing the source MJCF, regenerate the editable robot and runtime sensors:

```bash
uv run python -m examples.go1.tools.refresh_go1_robot_scene
```

## Recorded policy results (2026-07-17)

The bundled balanced actor was selected from `model_5100`–`model_6000` by
fixed-seed terrain admission. Results for `model_5600` used all 70 cells and
seed 123; forward used two 1000-step episodes per cell, yaw/run one 500-step
episode per cell.

| Command | Admission | Survival | Command-direction rate |
| --- | ---: | ---: | ---: |
| Forward 1 m/s | 0.836 | 0.936 | 0.837 m/s |
| Left strafe 1 m/s | 0.857 | 0.929 | 0.821 m/s |
| Reverse 1 m/s | 0.857 | 0.943 | 0.890 m/s |
| Yaw 0.5 rad/s | 0.914 | 1.000 | 0.257 rad/s |
| Forward run 2 m/s | 0.829 | 0.829 | 1.637 m/s |

The experimental `model_6575` run actor reached 2.686 m/s with 0.529 admission
under a 3 m/s command, using two 1000-step episodes per cell. Its diagonal trot
support was 0.162 versus 0.351 for the speed-only `model_5900`. It improved
paired motion but did not establish a strict bound: exact front/rear-pair
support occupied less than 1% of samples. Run actors remain local experiments;
normal editor playback uses the balanced actor. Out-of-bounds termination is
reported separately from illegal contacts on the finite terrain.
