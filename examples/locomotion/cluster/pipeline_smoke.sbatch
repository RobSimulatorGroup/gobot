#!/usr/bin/env bash
#SBATCH --job-name=gobot-tracking-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=gobot-tracking-smoke-%j.log
set -euo pipefail
GOBOT_ROOT="${GOBOT_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
GOBOT_PYTHON="${GOBOT_PYTHON:-$GOBOT_ROOT/.venv/bin/python}"
GOBOT_OUTPUT="${GOBOT_RUN_ROOT:-$GOBOT_ROOT/examples/locomotion/logs/cluster}/smoke_${SLURM_JOB_ID:-manual}"
cd "$GOBOT_ROOT"
mkdir -p "$GOBOT_OUTPUT"
exec "$GOBOT_PYTHON" -u -m examples.g1.train_tracking \
  --config examples/g1/train/g1_tracking_full_network_smoke.json \
  --output "$GOBOT_OUTPUT" --ledger "$GOBOT_OUTPUT/gpu_budget.json"
