#!/usr/bin/env bash
#SBATCH --job-name=gobot-tracking-parity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=gobot-tracking-parity-%A_%a.log
set -euo pipefail
GOBOT_ROOT="${GOBOT_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
GOBOT_PYTHON="${GOBOT_PYTHON:-$GOBOT_ROOT/.venv/bin/python}"
GOBOT_REFERENCE_PYTHON="${GOBOT_REFERENCE_PYTHON:-$GOBOT_ROOT/examples/locomotion/logs/supermimic_reference/venv_317/bin/python}"
: "${GOBOT_CHECKPOINT:?Set GOBOT_CHECKPOINT to the frozen reference checkpoint}"
: "${GOBOT_PAIR_OUTPUT:?Set GOBOT_PAIR_OUTPUT to a fresh shared output directory}"
cd "$GOBOT_ROOT"
mkdir -p "$GOBOT_PAIR_OUTPUT"
if [[ "${SLURM_ARRAY_TASK_ID:-0}" == 0 ]]; then
  exec "$GOBOT_REFERENCE_PYTHON" -u -m examples.locomotion.tracking_reference_probe \
    --checkpoint "$GOBOT_CHECKPOINT" --output "$GOBOT_PAIR_OUTPUT/reference" \
    --ledger "$GOBOT_PAIR_OUTPUT/reference_gpu_budget.json" --hours 5.9 --deterministic
else
  exec "$GOBOT_PYTHON" -u -m examples.g1.tracking evaluate --checkpoint "$GOBOT_CHECKPOINT" \
    --backend mujoco-warp --output "$GOBOT_PAIR_OUTPUT/gobot" \
    --ledger "$GOBOT_PAIR_OUTPUT/gobot_gpu_budget.json" --hours 5.9 --deterministic
fi
