# Go1: training and playback

Velocity PPO and the walk/trot/gallop task use the
[shared locomotion training framework](../locomotion/README.md).
The gait task adds phase observations and uses practical terrain resources.

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

Playback reads `policies/go1_velocity.onnx` by default. Export a policy for the
current scene and set its manifest-backed `.onnx` or `.pt` path:

```bash
GOBOT_GO1_POLICY=res://policies/go1_velocity.pt \
  uv run gobot_editor --path examples/go1
```

Missing policies or mismatched manifests abort Play with a diagnostic. The
manifest supplies joint order, control/solver settings, physics rate, and
observation/action specs; its scene digest also covers terrain resources.

## Train

Go1 uses the [shared training framework](../../doc/gobot_training.md). The PPO task
learns commanded velocity; the gait task adds observable walk/trot/gallop phase
rewards. Both use Gobot-authored scenes and can select MuJoCo CPU or MuJoCo Warp.

```bash
uv run python -m gobot.rl.train go1-velocity \
  --backend mujoco-cpu --num-envs 4 --iterations 10 --no-eval --output logs/go1_cpu

uv run python -m gobot.rl.train go1-gait \
  --backend mujoco-warp --device cuda:0 --num-envs 1024 \
  --config examples/locomotion/configs/go1_gait.json --output logs/go1_gait
```

`--terrain flat|rough` selects the standard velocity task terrain. Training can start
from scratch. Resume restores the optimizer, curriculum and RNG state and starts fresh
physics episodes. Switching backend starts a new experiment.

## Evaluate, export and play a trained policy

The shared evaluator checks commanded progress, velocity/yaw error, contacts,
terrain coverage and measured gait. Training reward alone does not establish
successful walking or running.

```bash
uv run python -m examples.locomotion.assessment \
  --checkpoint logs/go1_gait/best.pt --output logs/go1_gait/validation.json
uv run python -m examples.locomotion.record \
  --checkpoint logs/go1_gait/best.pt --output logs/go1_gait/walk.mp4 --continuous
uv run python -m examples.locomotion.verify_export \
  --checkpoint logs/go1_gait/best.pt --output logs/go1_gait/candidate.onnx
```

For a shared-workflow candidate, open `go1_practical_scene.jscn` and set
`GOBOT_GO1_POLICY` to its exported ONNX. Its playback script carries the same phase
clock and observation layout. `T` cycles walk/trot/run commands. The bundled
`go1_scene.jscn` baseline has its own policy contract; incompatible policies are
rejected rather than guessed from their tensor dimensions.

See [shared locomotion](../locomotion/README.md) for full admission, custom rewards,
resume semantics and training on a cluster.

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
[CPU batch guide](../../doc/mujoco_cpu_batch.md) for CPU memory and throughput.

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
| `go1_profile.py`, `go1_policy_info.py` | Articulation/controller profile and policy identity |
| `train/go1_velocity_cfg.py` | Robot preset using the shared scene and solver APIs |
| `gobot.rl.locomotion.cpu_env`, `warp_env` | Shared CPU and CUDA task implementations |
| `train/go1_gait.py` | Robot foot and joint mapping for gait evaluation |
| `scripts/go1.py`, `tools/` | Editor playback, evaluation, export, and parity tools |
| `assets/xml/go1.xml` | Import/equivalence source asset |

After changing the source MJCF, regenerate the editable robot and runtime sensors:

```bash
uv run python -m examples.go1.tools.refresh_go1_robot_scene
```
