# G1 multi-terrain velocity training

G1 uses the shared `gobot.rl.locomotion` velocity task: MuJoCo CPU for reference
execution and MuJoCo Warp for GPU training. The policy controls 29 body joints;
the 14 hand joints remain in the model with fixed position targets. This is a
new policy contract, separate from the existing Newton 43-action demonstration.

## Prepare

After native changes, rebuild the editable installation as described in
[building.md](../../doc/building.md):

```bash
uv sync --reinstall-package gobot --no-build-isolation-package gobot
uv run python -m examples.g1.prepare
```

The preparation tool reads
`examples/newton_g1/assets/unitree_g1/mjcf/g1_29dof_with_hand_rev_1_0.xml`
and its `meshes/` directory. An alternate source can be selected with
`--asset-root /path/to/unitree_g1`. Missing source assets produce an explicit
error. The tool copies the referenced meshes into the ignored local
`assets/meshes/` cache and regenerates `g1.jscn`; it does not change the source
asset. Prepare assets before running the G1 integration tests.

`g1_scene.jscn`, `g1.jscn`, and `terrain/rough_terrain.jres` own the authored
robot and terrain. The training preset uses simple collision primitives,
seven capsules per foot, grouped PD gains and motor limits, pelvis IMU,
torso attitude penalties, and measured whole-body angular momentum.
Training loads the Gobot scene, rather than reloading source MJCF.

## Train, resume, evaluate, export

Run these commands from the repository root:

```bash
uv run python -m examples.locomotion.train --robot g1 \
  --backend mujoco-cpu --device cpu --num-envs 64 --iterations 10 \
  --log-dir /tmp/g1_cpu_smoke

uv run python -m examples.locomotion.train --robot g1 \
  --backend mujoco-warp --device cuda:0 --num-envs 256 --iterations 10000 \
  --no-step-extras --log-dir examples/g1/logs/rough

uv run python -m examples.locomotion.train --robot g1 \
  --backend mujoco-warp --num-envs 256 --iterations 2 --resume \
  --log-dir examples/g1/logs/rough

uv run python -m examples.locomotion.evaluate --robot g1 \
  --checkpoint examples/g1/logs/rough/model_final.pt \
  --command 1 0 0 --steps 1000 --episodes-per-cell 2 \
  --json-out /tmp/g1_forward.json

uv run python -m examples.locomotion.export \
  --checkpoint examples/g1/logs/rough/model_final.pt \
  --output examples/g1/policies/g1_velocity.onnx
```

Use `--robot go1` for the existing Go1 preset through the same tools. Its old
training entry points, policy identity and published ONNX remain supported.

Checkpoints carry task/policy identity, control parameters, held hand targets,
terrain assignments, curriculum mode and RNG state. Resume begins fresh
physics episodes. Same-sized batches restore terrain assignments; resizing
preserves each terrain type's level distribution. The G1 command-progress
curriculum includes the command distance remaining after an early failure and
never promotes standing drift. Initial spawns use levels 0–1 of ten levels.

Evaluation covers every generated cell in the seven terrain categories and
reports survival, commanded progress, velocity error, slip, and termination
causes. At least half of the requested planar/yaw speed is required for
admission. Evaluate forward, reverse, lateral and yaw commands separately.
A ten-iteration run validates the training pipeline, not a usable walking
policy; no smoke-trained weights are bundled.

## Play

```bash
uv run gobot_editor --path examples/g1
```

After exporting a policy, press Play. `W/S` controls forward/backward velocity,
`Q/E` lateral velocity, `A/D` yaw, and `R` resets. `GOBOT_G1_POLICY` may select
another ONNX file. Playback uses the active Play scene clone and synchronizes
its links through `ProviderPlaySession`; training does not run node scripts.
Playback resets to a fixed central terrain cell near the authored origin.
The policy and scene/control contracts must match. GPU access is required for
this playback path.

## Reference and runtime semantics

Robot motor parameters, collision primitives and G1 posture rewards were
adapted from local mjlab commit
`8bb2bc9e187797689ec82c78c35019afd2fd0376` (Apache-2.0); see `NOTICE`.
Gobot does not import mjlab at runtime.

G1 refreshes kinematics and sensors after the final physics substep, while
preserving contact history from all substeps. Existing Go1 presets retain their
legacy sensor timing for policy compatibility. This timing is part of the new
policy manifest. Physics runs at 200 Hz and policy control at 50 Hz.

MuJoCo Warp 3.17 reports solver/line-search iteration limits separately from
capacity loss. `provider.solver_limit_counts()` exposes those diagnostics;
contact/constraint capacity overflow still raises an error. Inspect these
counts when tuning solver budgets for a new robot or terrain.

Recorded engineering checks and throughput measurements are in
[the validation report](../../doc/locomotion_shared_validation.md).
