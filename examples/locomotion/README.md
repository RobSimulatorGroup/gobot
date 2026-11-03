# G1 and Go1 natural locomotion

Train headlessly from Gobot-authored robot and terrain scenes. G1 uses PPO with an
optional human AMP prior conditioned on gait and commanded body-frame velocity. Go1 uses PPO with a soft walk/trot/gallop clock;
it does **not** use animal motion capture or AMP. Both use MuJoCo CPU as the
reference implementation and MuJoCo Warp for CUDA training.

No new policy is admitted by the implementation or smoke tests alone. Existing
bundled playback policies remain unchanged until candidates pass admission and
visual review.

## Full SuperMimic reference (2026-10-28)

The separate `supermimic_reference.py` runner tests Uni-SuperMimic's complete
world-model/BPTT pipeline with pinned physics and all 40 motion clips. It is not
AMP. ModelScope asset preparation is provided by `reference_assets.py`; all assets
are checksummed, including the existing 18 clips after newline normalization.
See [reference configuration, validation, live status and limits](../../doc/supermimic_reference_status.md).
The local full-network benchmark and checkpoint resume passed; the bounded formal
reference run uses 512 environments and automatically evaluates full-clip coverage.
A tracking checkpoint still requires separate velocity/terrain admission.
`supermimic_video.py --watch --run-dir <run>` records fixed walking/running/sprinting
windows every 15 minutes and on completion. The current video and next recording
are listed in `<run>/videos/latest.json` and `watch_status.json`.

## Frozen v1 experiment workflow

The checked-in campaign recipes below retain schema 1 for comparison with the
October 26 runs. Schema 2 is experimental and is not launched by this frozen campaign.


From the repository root, after installing the editable Gobot package:

```bash
uv run --no-sync python -m examples.locomotion.campaign \
  --output examples/locomotion/logs/natural_v1 \
  --budget-file examples/g1/logs/gpu_budget.json \
  --additional-hours 6 --total-hours 24 --num-envs 1024
```

This runs independent seeds 42/43, starting with G1 control and Go1 baseline/natural
comparisons. Each job is capped at 30 minutes; the campaign shares a locked GPU-hour
ledger. G1's admitted control checkpoint seeds matched PPO/AMP gait experiments.
Natural gait must pass before rough terrain, and rough terrain before mixed commands.
A failed stage does not silently advance. The same campaign command resumes saved
jobs; it never charges time while the machine was off.

Individual runs use checked-in JSON recipes:

```bash
uv run --no-sync python -m examples.locomotion.train \
  --config examples/locomotion/configs/g1_control.json \
  --log-dir examples/locomotion/logs/g1_control

uv run --no-sync python -m examples.locomotion.train \
  --config examples/locomotion/configs/g1_amp.json \
  --initialize examples/locomotion/logs/g1_control/best.pt \
  --log-dir examples/locomotion/logs/g1_amp
```

`--resume` restores the same algorithm, optimizer, discriminator/replay, observation
statistics, curriculum and RNG states, then starts fresh physical episodes.
`--initialize` (also `--fork`) retains the actor and its normalizer, starting a fresh
critic, optimizer and exploration. In schema 1 G1 AMP starts at `flat_gait`; schema 2
also permits early AMP during `flat_control`. Resume checks semantic configuration
signatures; old signatures require an explicit fork or the frozen v1 runner. Training and
validation use separate simulation processes and RNG streams.

`--num-envs`, `--iterations`, `--hours`, `--budget-hours`, `--seed`, and evaluation
interval/count flags override the recipe. A CPU smoke run can use
`--backend mujoco-cpu --num-envs 4 --iterations 3 --no-eval` with a control recipe.

Training uses normalized observations, log standard deviation bounded to [0.1, 1.0],
initial standard deviation 0.5, fixed 3e-4 learning rate and 0.003 entropy coefficient.
Non-finite values stop the run. AMP style collapse for 200 updates after warmup or
three screenings without improvement also stop it. Two consecutive passed screenings
finish a stage early. Checkpoints are atomic and all screened candidates are retained.

## What is being compared

| Robot | Motion and policy observations | Target terrain |
|---|---|---|
| G1 | 10 proprioceptive frames, command, terrain scan; AMP walk/run transitions matched by speed | Seven types; slopes 20°, stairs 15 cm |
| Go1 | Proprioception, command, terrain scan, phase sin/cos and gait weights | Seven types; slopes 20°, stairs 10 cm |

The practical terrains include flat, ascending/descending stairs and slopes, rough
noise ±3 cm, and waves up to 8 cm amplitude. Go1's original steeper terrain remains
in `terrain/rough_terrain.jres` as a separate challenge set. Running on rough/mixed
stages is limited to flat terrain and the first gentle slope levels (≤5°).

Go1 softly blends four-beat walking into diagonal trot at 0.8 m/s and gallop at
2.0 m/s, with 0.2 m/s transition bands. The clock is observable during training,
exported inference, reset and editor playback. Turning/reversing and difficult
terrain reduce the timing reward. No reward penalizes trot merely for being trot.
The baseline uses identical observations with the clock reward set to zero.

G1 reference processing retains dataset checksums and robot-physics identity.
Subject 4 is validation, subject 5 is held out; neither is sampled for training.
AMP only applies to supported forward-motion speeds, with no style reward for
standing, backward or lateral commands. Reference-state resets use the commanded
speed, including running. The discriminator starts with one update per PPO update;
its joint expert/policy normalization is frozen during those discriminator minibatches.
The motion reward ramps from zero to weight 2 over 500 PPO updates.

G1 stays upright relative to world gravity; Go1 may align to a reliable terrain
normal. Both new recipes use post-step sensing and pre-reset terminal observations.
Curriculum progress follows the commanded direction rather than radial distance.

## Status and evaluation

```bash
uv run --no-sync python -m examples.locomotion.status examples/locomotion/logs/natural_v1
uv run --no-sync python -m examples.locomotion.assessment \
  --checkpoint RUN/best.pt --output RUN/validation.json
```

Status distinguishes starting, running, evaluating, completed, stopped, failed,
lost processes and unavailable status. PID start time and PID namespace prevent
mistaking a reused PID or sandbox visibility restriction for a live/dead training job.
`completed` means the job ended normally; read `admitted` for policy quality.

Validation reports velocity/yaw errors, cross-track deviation, slip, clearance,
per-cell success and independent footfall metrics. Ranking includes the worst
tracking violation when all candidates fail admission. Gait analysis measures actual
contacts/stride cycles; it never uses the policy clock or discriminator score as
proof of a natural gait.

Final admission uses three new terrain seeds (1001/2002/3003), at least ten trials
per terrain/difficulty/seed, nominal and perturbed continuous scenarios, command
changes, 20-second gentle-slope running and 30-second flat running/stop-start tests:

```bash
uv run --no-sync python -m examples.locomotion.admit \
  --checkpoint RUN/best.pt --output RUN/admission \
  --budget-file examples/g1/logs/gpu_budget.json --budget-hours 24 --hours 1
```

Required mobility: ≥95% overall and ≥90% per terrain/difficulty; velocity error
≤max(0.15 m/s, 20% commanded speed), yaw-rate error ≤0.2 rad/s, and straight-route
cross-track deviation ≤0.5 m. Walking routes must traverse the assigned terrain;
resets or crossing another boundary cannot count as success. Stops must settle
below 0.1 m/s after two seconds.

G1 additionally needs alternating support and held-out speed-matched cadence/stride
ranges; running needs ≥5% flight. Go1 requires measured walk/trot/gallop stride phases,
accepts mixed patterns in transition bands, rejects simultaneous hopping, and checks
flight during fast gallop. Perturbation tests enable configured friction/COM/encoder
randomization and pushes. Final export occurs only after all numerical gates pass;
visual review is still required before promoting a policy.

Record a candidate or inspect export equivalence without promoting it:

```bash
uv run --no-sync python -m examples.locomotion.record \
  --checkpoint RUN/best.pt --output RUN/walk.mp4 --continuous --seconds 12
uv run --no-sync python -m examples.locomotion.verify_export \
  --checkpoint RUN/best.pt --output RUN/candidate.onnx
```

For candidate editor playback, open `go1_practical_scene.jscn` in the Go1 project
and set `GOBOT_GO1_POLICY` to the verified ONNX file. Its dedicated script uses the
same phase wrapper as training. `T` cycles walk/trot/run speeds; W/S, Q/E and A/D
control translation and yaw, Space stops and R resets. G1 uses its existing shared
playback script with `GOBOT_G1_POLICY`. These explicit overrides do not replace the
bundled policies. Final editor/visual admission remains pending until a trained
candidate passes the numerical gates.

## Transfer to a cluster

```bash
uv run --no-sync python -m examples.locomotion.package \
  --config examples/locomotion/configs/g1_control.json \
  --config examples/locomotion/configs/g1_amp.json \
  --config examples/locomotion/configs/go1_baseline.json \
  --config examples/locomotion/configs/go1_natural.json \
  --output examples/locomotion/logs/natural-training-bundle.tar.gz
```

The archive contains an exact source overlay, robot/terrain assets, processed G1
motions with provenance, locked dependencies, checksums and a base Git revision.
Build/install native Gobot for the destination Python/platform, verify `bundle.json`,
then run the same Python entry point. No local native binary or absolute home-directory
path is required. Use larger `num_envs` after a GPU smoke test and set the cluster
GPU-hour budget explicitly.

The first cluster workflow is independent jobs per GPU (different seeds/recipes),
not synchronized multi-GPU AMP. Give concurrent jobs separate output directories
and ledgers; sum their GPU-hours. `WORLD_SIZE>1` is rejected to prevent unsynchronized
discriminators from silently producing invalid experiments.

## Regression checks

```bash
uv run --no-sync python tests/python/test_natural_locomotion.py
uv run --no-sync python tests/python/test_natural_locomotion.py --backend mujoco-cpu
uv run --no-sync python tests/python/test_natural_locomotion.py --backend mujoco-warp
uv run --no-sync python tests/python/test_locomotion_training.py --backend mujoco-warp
uv run --no-sync python tests/python/test_g1_amp.py
uv run --no-sync python tests/python/test_locomotion_budget.py
```

The optimizer test executes PPO and AMP, resumes their checkpoints and invokes the
independent evaluator on both backends. These small runs check correctness only.

## Gobot SuperMimic tracking

The scene-authored port, randomized training, checkpoint recovery, frozen-actor
forks and independent validation are available through `examples.g1.train_tracking`.
Full-network smoke training has run in Gobot; full-library migration admission
is still pending. See the [G1 tracking workflow](../g1/README.md),
[measured port status](../../doc/supermimic_port_status.md), and
[cluster job instructions](cluster/README.md).
