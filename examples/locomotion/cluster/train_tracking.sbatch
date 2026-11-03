#!/usr/bin/env bash
#SBATCH --job-name=gobot-tracking
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --signal=B:USR1@180
#SBATCH --output=gobot-tracking-%A_%a.log
set -euo pipefail
GOBOT_ROOT="${GOBOT_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
GOBOT_PYTHON="${GOBOT_PYTHON:-$GOBOT_ROOT/.venv/bin/python}"
GOBOT_CONFIG="${GOBOT_CONFIG:-$GOBOT_ROOT/examples/g1/train/g1_tracking_cluster.json}"
: "${GOBOT_ADMISSION:?Set GOBOT_ADMISSION to the passed complete paired fidelity report}"
GOBOT_RUN_ROOT="${GOBOT_RUN_ROOT:-$GOBOT_ROOT/examples/locomotion/logs/cluster}"
GOBOT_SEEDS=(42 43)
GOBOT_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
GOBOT_SEED="${GOBOT_SEEDS[$GOBOT_INDEX]}"
GOBOT_OUTPUT="$GOBOT_RUN_ROOT/${SLURM_ARRAY_JOB_ID:-manual}_s$GOBOT_SEED"
cd "$GOBOT_ROOT"
mkdir -p "$GOBOT_OUTPUT"
# Each array task owns one GPU. No torchrun/DDP or shared optimizer is implied.
# Forward additional arguments such as --fork/--source-manifest/--source-identity.
exec "$GOBOT_PYTHON" -u -m examples.g1.train_tracking --config "$GOBOT_CONFIG" \
  --output "$GOBOT_OUTPUT" --seed "$GOBOT_SEED" --ledger "$GOBOT_OUTPUT/gpu_budget.json" \
  --admission "$GOBOT_ADMISSION" "$@"
