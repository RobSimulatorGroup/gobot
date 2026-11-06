# SuperMimic reference and BONES workflow

These tools prepare data and inspect the single training implementation at
`python -m gobot.rl.train`. Reference code runs only for independent checks;
training uses Gobot scenes and services.

- [Training/install guide](../../../doc/gobot_training.md)
- [Measured 100k BONES results](../../../doc/g1_bones_100k_results.md)
- [Detailed progress and historical diagnostics](../../../doc/supermimic_reproduced_progress.md)

Run commands from the repository root with the installed Gobot environment.
Large data, checkpoints and logs are ignored by Git. The compact measured-result
record in `results/` contains summary values, provenance and report hashes.

## Tool map

| Command module | Purpose |
|---|---|
| `reference` | Verify pinned sources/releases and export baseline evaluation lists |
| `export_model` / `examples.g1.prepare_tracking` | Export the actual reference model; author/audit the Gobot scene |
| `export_bones_reference` | Export the historical filter, reference converter and G1 assets from Git objects |
| `prepare_bones` | Convert, independently filter, hash and pack the BONES data |
| `evaluate_bones` | Prepare a fixed plan, evaluate GPU shards and merge complete results |
| `record_tracking` | Replay one successful or failed clip and render actual/reference videos |
| `compare_benchmarks` | Compare already completed reports with identical clip names/frame counts |

`record_failure` remains a small compatibility entry point to `record_tracking`.

## 1. Pin reference inputs

`reference.lock.json` fixes SuperMimic commit `ba610534` and mjlab `8bb2bc9`,
as well as published release SHA256/size. The original published stack used
MuJoCo 3.16; Gobot needs 3.17's native batch API. Record original-stack validation
and same-stack comparisons separately.

Set `REFERENCE_CHECKOUT`, `MJLAB_CHECKOUT`, `RELEASE_DIRECTORY`, `BASELINE_DIRECTORY`,
`REFERENCE_EXPORT`, `AUTHORED_PROJECT` and `BONES_REFERENCE` to your paths. The release
directory must contain the three files named in the lock. A shallow reference
checkout may need the historical training-filter commit fetched before export.

```bash
python -m examples.g1.reproduction.reference \
  --reference "$REFERENCE_CHECKOUT" --mjlab "$MJLAB_CHECKOUT" \
  --release "$RELEASE_DIRECTORY" --output "$BASELINE_DIRECTORY"

python -m examples.g1.reproduction.export_bones_reference \
  --reference "$REFERENCE_CHECKOUT" --mjlab "$MJLAB_CHECKOUT" \
  --baseline "$BASELINE_DIRECTORY" --output "$BONES_REFERENCE"
```

The exporter reads fixed Git objects, not edited working-tree files. It creates
`filter_train.py`, `filter_eval.py`, `motion_convert.py`, the reference `LICENSE`,
`source_manifest.json`, pinned model assets under `g1/`, and verified published
full/filtered evaluation-name lists. Changed existing inputs are rejected.

The training filter is from `d9be385b3eeb13a638fb33fb200d3bfd9be9839f`.
The later acrobatics keyword additions in `ba610534` affect the evaluation subset,
not the original e8 training set. Published lists are compared **after** independent
filtering; they do not select the reconstructed training data. Original training
library bytes remain unavailable even when clip membership matches.

## 2. Prepare the Gobot scene and data

`REFERENCE_PYTHON` is the independent interpreter with pinned reference/mjlab
sources installed; `python` below is the rebuilt Gobot interpreter. Use the same
MuJoCo 3.17 version for model export and Gobot import.

```bash
"$REFERENCE_PYTHON" -m examples.g1.reproduction.export_model --output "$REFERENCE_EXPORT"
python -m examples.g1.prepare_tracking \
  --reference-export "$REFERENCE_EXPORT" --project-path "$AUTHORED_PROJECT"
```

For BONES, install `pandas` and `tyro` in the preparation environment for the
reference scripts. Set `BONES_CSV` to the extracted `g1/csv` directory and
`BONES_PREPARED` to a new output directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m examples.g1.reproduction.prepare_bones \
  --project "$AUTHORED_PROJECT" --input "$BONES_CSV" \
  --output "$BONES_PREPARED" --reference "$BONES_REFERENCE" \
  --workers 32 --shard-size 256
```

A pilot uses `--limit 256` and a **separate** output directory. Repeating the same
command checks and reuses completed shards. Changed input files, code, scene or
selection rules invalidate resume. `status.json` reports progress; `manifest.json`
reports completion and `ready_for_paired_training` separately.

| Output | Contents |
|---|---|
| `library.pt` | Sorted training clips, 71D fp16 states, names/starts/lengths |
| `npz/` | Dense training clips, hard-linked to completed shards |
| `train_manifest.csv`, `eval_manifest.csv` | Independent keep/reject decisions and reasons |
| `manifest_all.jsonl` | Every raw clip's source/NPZ hashes and frame count |
| `preparation.json`, `manifest.json` | Frozen versions/rules and final identity/parity checks |
| `parts/`, `status.json` | Resumable conversion output and progress |

The measured full conversion retained 129943/142220 training clips (47475680 frames),
with 129491 in the filtered evaluation subset. The packed file is about 6.75 GB.
The library and its metadata are immutable inputs once training starts.

LAFAN1 is separate: convert its 40 G1 CSV clips at 30 Hz using
`python -m gobot.rl.motion_tracking.convert --input ... --input-fps 30 --compact`.
Neither BONES training nor `--library` automatically adds LAFAN1 data.

## 3. Train

See the [training and resume guide](../../../doc/gobot_training.md). Preserve the original
`reproduced_bones.json` recipe (35/250 capacity) for reference; the measured 8×4090
run used `reproduced_bones_nj500.json` (35/500). A capacity change is a new experiment
configuration and is rejected during checkpoint resume. Eight training ranks alter
global sample count and learning-rate scaling relative to four ranks.

## 4. Evaluate complete clips

Set `CHECKPOINT` to the rank-zero final checkpoint and `EVAL_DIRECTORY` to a new
output directory. Evaluation needs the **dense NPZ directory and manifests**, not
just `library.pt`. No optimizer is created.

```bash
python -m examples.g1.reproduction.evaluate_bones prepare \
  --checkpoint "$CHECKPOINT" --project "$AUTHORED_PROJECT" \
  --data "$BONES_PREPARED" --output "$EVAL_DIRECTORY" \
  --num-envs 1024 --max-clips 1024 --max-frames 1000000 \
  --nconmax 128 --njmax 500
```

On an allocated node with eight visible, idle GPUs, each process evaluates disjoint
shards; this is not torchrun data-parallel training. The loop preserves worker
exit failures and merges only after every worker succeeds:

```bash
(
  set -euo pipefail
  pids=()
  for i in {0..7}; do
    CUDA_VISIBLE_DEVICES="$i" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
    python -u -m examples.g1.reproduction.evaluate_bones worker \
      --output "$EVAL_DIRECTORY" --device cuda:0 --index "$i" --workers 8 \
      > "$EVAL_DIRECTORY/worker$i.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if (( failed )); then exit 1; fi
  python -m examples.g1.reproduction.evaluate_bones merge --output "$EVAL_DIRECTORY"
)
```

Rerun an interrupted worker with the same arguments to validate/reuse completed
shards. Missing, duplicate, nonfinite and prematurely truncated results prevent
successful merge. Each physics step checks overflow before reset can clear flags.

The actor is deterministic, but additional Warp deterministic reductions are **off**,
matching the pinned reference implementation. `--deterministic` enables them only
for a separate diagnostic; the measured large-batch deterministic attempt exceeded
Warp's workspace shape limit. Do not confuse policy determinism with bitwise physics.

Final outputs are `full.json`, `filtered.json`, `failures.json` and `summary.json`.
Return is normalized by actual executed steps; coverage and expected-length return
are reported separately. This is training-motion evaluation, not held-out generalization.
Record capacity 128/500 when comparing with the published 35/250 results.

## 5. Record success or failure

```bash
CUDA_VISIBLE_DEVICES=0 python -m examples.g1.reproduction.record_tracking \
  --evaluation "$EVAL_DIRECTORY" --clip walk_forward_loop_001__A040 \
  --outcome success --output "$EVAL_DIRECTORY/videos/walk_forward.mp4"

CUDA_VISIBLE_DEVICES=0 python -m examples.g1.reproduction.record_tracking \
  --evaluation "$EVAL_DIRECTORY" --clip flip_360_hero_landing_001__A304 \
  --outcome failure --output "$EVAL_DIRECTORY/videos/flip.mp4"
```

The recorder replays the original shard, captures post-physics link poses **before
reset**, and then renders actual/reference poses side by side using Gobot CameraSensor.
Green success requires all expected frames without termination; red failure requires
an actual termination. Both stop before showing autoreset and hold the labeled final
frame for one second. `imageio-ffmpeg` supplies the H.264 encoder; EGL is required.
The selected clip currently must be in its shard's first queue wave (the 1024-lane
plan above covers every shard this way). JSON, pose NPZ and contact-sheet PNG sidecars
retain both original and replay outcomes. GPU replay can shift failure time slightly.

## Integrity and historical results

Preparation and evaluation plans bind their own script hashes. Keep the exact
`prepare_bones.py` / `evaluate_bones.py` that produced a run if you need to resume
or record it later. Cleanup does not bypass those identity checks. Neither the
training math nor the frozen preparer/evaluator source was changed in the 2026-11-06
workflow cleanup; `record_failure` delegates to the generalized recorder.

The optional diagnostic modules `check_kinematics`, `check_controls`, `math_fixture`,
`training_fixture`, `check_release_policy`, `check_task`, `check_context`, `check_update`
and `reference_benchmark` remain available (`--help`). They isolate model, algorithm,
randomization and fixed-policy parity; their results do not replace from-scratch
training and complete-clip evaluation. See [progress](../../../doc/supermimic_reproduced_progress.md)
for measured tolerances and historical limitations.

Run local data/report/recording boundary tests without GPU training:

```bash
python -m pytest -q tests/python/test_bones_preparation.py \
  tests/python/test_bones_evaluation.py tests/python/test_bones_reference_export.py \
  tests/python/test_tracking_recording.py tests/python/test_cluster_launchers.py
```
