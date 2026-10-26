# G1 motion-prior locomotion

This example trains a 29-action G1 using a human motion prior (AMP) and PPO.
MuJoCo Warp runs the training batch; MuJoCo CPU is the state/control reference.
The 14 hand joints remain present with fixed targets. The robot and terrain
are authored in Gobot `.jscn` / `.jres` resources.

The goal is natural walking on flat ground, slopes, stairs, rough and wave
terrain, with running on flat ground and gentle slopes. A saved checkpoint is
not an admitted walking policy. Use the evaluation tools below to measure its
actual capabilities. Experiment outcomes are recorded in the run directories;
no smoke-trained policy is published as a usable policy.

## Prepare

After native changes:

```bash
uv sync --reinstall-package gobot --no-build-isolation-package gobot
uv run python -m examples.g1.prepare
uv run python -m examples.g1.prepare_motion
uv run python -m examples.g1.reference_gaits
```

Robot preparation reads the shared Unitree source in `examples/newton_g1/assets`.
Use `--asset-root` to choose another copy. It retains original inertias and
visual meshes, adds mjlab's primitive collision proxies **and their four link
collision exclusions**, and saves the result in `g1.jscn`. Exclusions are
storable scene data and survive MJCF import, `.jscn`, and physics compilation.

Motion preparation downloads 18 G1-retargeted LAFAN1 walk/run/sprint clips at a
pinned dataset revision. Source checksums and provenance are saved alongside
ignored local assets. Quaternions and joint order are validated, samples are
resampled from 30 to 50 Hz, and forward kinematics uses the robot compiled from
the Gobot scene. The subject5 clips are held out. Use `prepare_motion --offline`
to regenerate features after robot changes without downloading again.

## Train and resume

All commands run from the repository root. Start with a bounded experiment:

```bash
uv run python -m examples.g1.train_amp --stage flat --num-envs 1024 \
  --iterations 2000 --hours 1 --log-dir examples/g1/logs/flat

uv run python -m examples.g1.train_amp --stage flat --num-envs 1024 \
  --resume examples/g1/logs/flat/latest.pt --iterations 5000 --hours 2 \
  --log-dir examples/g1/logs/flat_continue
```

Stages are `flat`, `rough`, and `mixed`. Changing stage loads the actor, critic,
optimizer and motion prior while starting a fresh stage curriculum. Same-stage
resume also restores terrain assignments and RNG state, then starts new physical
episodes. A small `--style-weight 0 --reference-reset-probability 0` run is the
PPO control experiment. CPU smoke runs use `--backend mujoco-cpu --device cpu`.

The shared `logs/gpu_budget.json` ledger and process lock bound training to at
most 24 GPU-hours. `--hours` bounds one job. SIGTERM/SIGINT request a checkpoint
and orderly shutdown. Each run records configuration, source snapshot, metrics,
TensorBoard events, recent weights, and up to three ranked candidates.

```bash
uv run python -m examples.g1.run_training --num-envs 1024 \
  --output examples/g1/logs/campaign --budget-hours 24
```

The campaign runs stage gates before advancing and reserves time for admission.
It exits with status 2 and an explicit report when a stage fails or the budget
expires. This is an experiment budget, not a guarantee of convergence.

## Evaluate and export

```bash
uv run python -m examples.g1.evaluate_amp \
  --checkpoint examples/g1/logs/rough/best.pt --seed 101 \
  --command 1 0 0 --episodes-per-cell 10 --json-out /tmp/g1_walk.json

uv run python -m examples.g1.evaluate_amp \
  --checkpoint examples/g1/logs/mixed/best.pt --continuous \
  --slope-degrees 5 --command 2 0 0 --json-out /tmp/g1_run.json

uv run python -m examples.g1.verify_export \
  --checkpoint examples/g1/logs/mixed/best.pt \
  --output examples/g1/policies/g1_velocity.onnx
```

Per-cell evaluation travels from -3 m to +3 m within an 8 m assigned cell.
Leaving through another side fails; a neighboring terrain cannot inflate the
birth cell's score. Continuous tests use a separate 64 m course. Terrain seeds
101, 202 and 303 produce held-out evaluation scenes saved for inspection under
`assets/evaluation`. All metrics use physical state before automatic resets.

The admission target is 95% overall and at least 90% per terrain/level, with
planar speed error below `max(0.15 m/s, 20% of commanded speed)`. Running also
requires alternating support, at least 5% flight time and sustained progress.
Sitting, low pelvis clearance, and prohibited body-ground support are failures.
Cadence and stride length are checked against held-out motion windows, not the
AMP discriminator's score. Videos remain part of the final review.

`verify_export` checks Torch/ONNX parity on real rollout observations, including
the observation normalizer and its epsilon. Deployment validates the robot,
control and observation contract independently of terrain identity. Checkpoints
still record their training scene digest; changing robot mass, collision shape,
collision exclusions, drives or observation layout invalidates that contract.

## Record and play

```bash
uv run python -m examples.g1.record_amp \
  --checkpoint examples/g1/logs/mixed/best.pt --continuous --transition \
  --seconds 20 --output /tmp/g1_walk_run.mp4

uv run gobot_editor --path examples/g1
```

Videos use Gobot's renderer and the same scene-compiled physics. A failure ends
the recording; an automatic reset is never presented as recovery. A JSON
trajectory accompanies the MP4.

In editor Play: W/S controls forward/backward velocity, Q/E lateral velocity,
A/D yaw, T toggles the forward running command, and R resets. `GOBOT_G1_POLICY`
selects an alternate ONNX. Playback runs on the Play clone without modifying
the edited scene. Running commands are intended for flat and gentle terrain.

## Contracts and checks

The actor receives ten frames of proprioception, the current command, and a
187-ray terrain scan (1,150 inputs). The critic receives 298 current privileged
inputs. Physics runs at 200 Hz and control at 50 Hz. AMP uses adjacent physical
states, separate walk/run heads, bounded replay, and mixed expert/policy feature
normalization. Its model is not required for inference. Flat training can use
reference-state initialization; evaluation always starts independently of clips.

```bash
uv run python tests/python/test_g1_amp.py
uv run python tests/python/test_g1_amp.py --backend mujoco-cpu
uv run python tests/python/test_g1_amp.py --backend mujoco-warp
uv run python tests/python/test_shared_velocity_gpu.py
```

The reference repositories and licensing are described in `NOTICE`. This is an
AMP/PPO locomotion implementation; it does not implement SuperMimic's learned
world-model/BPTT algorithm.

## Recover after reboot

`latest.pt` is published atomically, and the cumulative budget ledger is updated
at every optimizer iteration. Resume uses the saved iteration and optimizer
state; progress after the most recent checkpoint is repeated. Offline time is
not charged. Inspect runs with:

```bash
uv run python -m examples.g1.training_status
```

To keep a bounded run independent of a terminal session on Linux:

```bash
systemd-run --user --unit=gobot-g1-training \
  --property=WorkingDirectory="$PWD" --property=RuntimeMaxSec=11000 \
  --property=TimeoutStopSec=90 --collect \
  "$PWD/.venv/bin/python" -B -m examples.g1.train_amp \
  --num-envs 1024 --hours 3 --iterations 6000 \
  --resume examples/g1/logs/flat/latest.pt \
  --log-dir examples/g1/logs/flat_recovered \
  --log-stdout examples/g1/logs/flat_recovered.stdout.log
```

The transient service does not restart automatically after a machine reboot.
Check the saved run and resume explicitly. `systemctl --user stop` requests an
orderly checkpoint through the training process's SIGTERM handler. For a
controlled tuning experiment, `--learning-rate`, `--schedule`, and
`--reset-action-std` explicitly override restored settings and are recorded in
that run's configuration.
