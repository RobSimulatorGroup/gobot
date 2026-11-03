#!/usr/bin/env bash
#SBATCH --job-name=gobot-tracking-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --output=gobot-tracking-bench-%A_%a.log
set -euo pipefail
GOBOT_ROOT="${GOBOT_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
GOBOT_PYTHON="${GOBOT_PYTHON:-$GOBOT_ROOT/.venv/bin/python}"
GOBOT_COUNTS=(512 1024 2048)
GOBOT_COUNT="${GOBOT_COUNTS[${SLURM_ARRAY_TASK_ID:-0}]}"
GOBOT_OUTPUT="${GOBOT_RUN_ROOT:-$GOBOT_ROOT/examples/locomotion/logs/cluster}/bench_${SLURM_ARRAY_JOB_ID:-manual}_$GOBOT_COUNT"
cd "$GOBOT_ROOT"
mkdir -p "$GOBOT_OUTPUT"
exec "$GOBOT_PYTHON" -u -m examples.g1.train_tracking \
  --config examples/g1/train/g1_tracking_full_network_smoke.json \
  --num-envs "$GOBOT_COUNT" --world-model-batch "$GOBOT_COUNT" --policy-batch "$GOBOT_COUNT" \
  --iterations 10 --output "$GOBOT_OUTPUT" --ledger "$GOBOT_OUTPUT/gpu_budget.json"
