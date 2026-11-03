# Gobot tracking jobs on a GPU cluster

These jobs use Gobot-authored scenes and the standalone `gobot.rl.motion_tracking`
and `gobot.rl.supermimic` packages. Each job owns one GPU. Seed arrays are independent
experiments, not distributed synchronized optimization.

The current frozen-checkpoint migration has outstanding full-clip failures. The
training command requires a passed full-library paired report before accepting a
formal recipe. The pipeline-smoke recipe checks execution without admitting a
walking policy. An archive labelled `diagnostic_not_admitted` must not be treated
as a passed model.

## Install and check the bundle

Start from `bundle.json`'s exact base commit with submodules, extract the overlay,
and verify checksums before changing source files. Build native Gobot for the
cluster's Python/platform. The local `.so` files are intentionally not copied.
Use `uv sync --frozen` and the repository build instructions; the core dependency
versions are MuJoCo/MuJoCo Warp 3.17, Warp 1.17, and Torch 2.11.0+cu128 in the
recorded reference environment. A cluster image must supply compatible NVIDIA
GPU drivers and a compiler for native Gobot/optional Warp kernels.

```bash
uv sync --frozen
uv pip install --python .venv/bin/python pytest==9.2.0
uv run --no-sync python -m examples.locomotion.package --verify bundle.json
uv run --no-sync python -m pytest -q \
  tests/python/test_articulation_batch.py \
  tests/python/test_tracking_randomization.py \
  tests/python/test_tracking_training.py
```

Use the `pipeline_smoke.sbatch` job first. It runs 64 environments, full reference
networks, 32-step BPTT, real physical randomization, and six updates. A successful
process means that forward/backward/checkpoint execution works; it does not imply
motion quality or admission.

```bash
sbatch examples/locomotion/cluster/pipeline_smoke.sbatch
```

After the execution smoke, `benchmark_tracking.sbatch` scans 512/1024/2048 environments
and matched model/policy batch sizes. It records startup time, per-iteration timing
and peak Torch memory (not total driver memory). Select the GPU type and environment
count from those measurements; the largest batch is not assumed fastest.

Partition, account, GPU type and container/module commands are site-specific and
must be supplied with the submission arguments or the local launcher. The scripts
do not guess an SSH endpoint, account or queue. For non-Slurm systems, use the same
Python commands with one independent process per assigned GPU.

## Complete the migration comparison

A bundle created with `--reference` also carries the pinned independent reference
source/assets and the checkpoint's `run_config.json`. It does not add UniLab to
Gobot's runtime requirements. Build the separate reference environment with:

```bash
bash examples/locomotion/cluster/bootstrap_reference.sh
```

The script installs reference dependencies without the extras that force MuJoCo
3.11, then applies the recorded 3.17 compatibility patch to that isolated copy.
It refuses to modify the original baseline environment or an unexpected dependency
source. The package versions and patch hashes are recorded in the run identity.

Use the same GPU model for the two array jobs and freeze the checkpoint, data and
numeric mode. Both jobs request complete deterministic execution and have six-hour
scheduler limits; they write separate ledgers whose usage can be summed.

```bash
export GOBOT_CHECKPOINT="$PWD/examples/locomotion/logs/supermimic_reference/g1_reference_s42_512/ckpt_012000.pt"
export GOBOT_PAIR_OUTPUT="$PWD/examples/locomotion/logs/cluster/paired_317"
sbatch examples/locomotion/cluster/evaluate_tracking_pair.sbatch
```

After **both** jobs finish, compute the gate:

```bash
uv run --no-sync python -m examples.locomotion.tracking_compare \
  --reference "$GOBOT_PAIR_OUTPUT/reference/evaluation.json" \
  --candidate "$GOBOT_PAIR_OUTPUT/gobot/evaluation.json" \
  --reference-identity "$GOBOT_PAIR_OUTPUT/reference/identity.json" \
  --candidate-identity "$GOBOT_PAIR_OUTPUT/gobot/identity.json" \
  --output "$GOBOT_PAIR_OUTPUT/admission.json"
```

The gate rejects changed checkpoints, numeric modes, clip denominators, incomplete
coverage, mean return loss above 0.02, and new failed clips. Formal training also
checks the reported robot physics signature. A higher average return does not
cancel a newly failed motion.

## Fresh training, fork, and resume

The default cluster recipe uses 2,048 environments and subject1–3 for training.
Subject4 is validation; subject5 remains final holdout. The coverage40 recipe is a
separate engineering experiment. Weights with all-library or unknown ancestry
cannot initialize strict subject-split training.

```bash
export GOBOT_ADMISSION="$GOBOT_PAIR_OUTPUT/admission.json"
export GOBOT_CONFIG="$PWD/examples/g1/train/g1_tracking_cluster.json"
sbatch examples/locomotion/cluster/train_tracking.sbatch
```

The array runs seeds 42 and 43. Override `GOBOT_RUN_ROOT`, `GOBOT_PYTHON`, and
`GOBOT_PROJECT_ROOT` if needed. Additional arguments are passed to the Python
runner. The same runner works directly:

```bash
uv run --no-sync python -m examples.g1.train_tracking \
  --config examples/g1/train/g1_tracking_cluster.json \
  --output RUN --ledger RUN/gpu_budget.json --admission ADMISSION.json
```

Resume each seed from its own checkpoint; do not pass one seed’s checkpoint to both array tasks.

`--resume RUN/last.pt` restores actor, world model, critic, all optimizers,
normalizers, task sampler, startup friction/CoM context and RNG state. It starts
fresh physical episodes and refills replay; this is not bitwise continuation of
an interrupted physical trajectory. The iteration target and reporting cadence
may change, while semantic configuration, data, software and batch size must
match. Periodic checkpoints are atomic; SIGTERM/SIGINT/SIGUSR1 request a save at
an iteration boundary. Slurm sends USR1 three minutes before the job deadline.

`--fork CHECKPOINT` starts fresh optimizer, critic and replay state while loading
actor/normalizers and world-model weights. It enters `adaptation`, with the actor
and its normalization statistics frozen. The actor does not automatically unfreeze
after a fixed number of updates. Legacy reference checkpoints additionally require
`--source-manifest` and a matching `--source-identity` to bind data provenance to
those exact weights. Use the coverage40 recipe for the existing all-library teacher.

For an adaptation checkpoint, collect independent real-rollout measurements:

```bash
uv run --no-sync python -m examples.g1.validate_tracking_adaptation \
  --config RECIPE.json --checkpoint ADAPTATION/last.pt --baseline PARENT.pt \
  --output ADAPTATION/validation_metrics.json
```

Without a threshold file this reports measurements only and cannot authorize
promotion. Declare thresholds **before** the acceptance evaluation, using the
measured source/target baseline. The file contains:

```json
{
  "world_model_rmse": {"1": 0.1, "8": 0.5, "32": 1.0},
  "critic_relative_rmse": 0.05,
  "minimum_survival_fraction": 0.9,
  "minimum_step_return": 0.8
}
```

These numbers illustrate the schema; they are not validated G1 acceptance values.
World-model normalization uses the baseline checkpoint's fixed scales. The report
includes 1/8/32-step model errors, 32-step real-return critic residuals, survival,
per-step return, and bitwise equality of the frozen actor/normalizers to its source.
Only a matching passed report permits `--resume ADAPTATION/last.pt --promote REPORT`.
This warmup gate is separate from full motion, velocity and terrain admission.

## Videos and usage

The default recipe queues a fixed preview request every 15 minutes and at the end,
with an immutable actor checkpoint and checksum. Missed ticks during a long compile
are recorded instead of fabricating repeated previews. The four windows are walk,
run, sprint and dance, each eight seconds. Render them on a graphics-capable worker:

```bash
uv run --no-sync python -m examples.g1.tracking_video_queue \
  --run-dir RUN --budget-file VIDEO_LEDGER.json --hours 1
```

`--video-mode render` launches that worker from the trainer on the same GPU; use it
only when the training image has EGL/OpenGL and enough memory. Its overlapping wall
time belongs to the parent allocation. `queue` keeps compute-only training independent
of a renderer and leaves explicit pending requests for a graphics worker. Failed
video requests are recorded and are not silently marked complete.

The cluster recipes use a finite per-job wall-time limit and usage accounting.
`total_gpu_hours: null` means no additional cross-job cap for a new cluster ledger;
it does **not** remove an authorization already attached to a ledger. Retain the
scheduler's allocated GPU-hours alongside each process ledger when comparing
end-to-end training cost. A failed/short screening does not satisfy admission.
