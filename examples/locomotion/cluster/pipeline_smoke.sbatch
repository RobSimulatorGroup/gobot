#!/usr/bin/env bash
#SBATCH --job-name=gobot-supermimic-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=gobot-supermimic-smoke-%j.log
set -euo pipefail
GOBOT_ROOT="${GOBOT_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
GOBOT_PYTHON="${GOBOT_PYTHON:-$GOBOT_ROOT/.venv/bin/python}"
# One torchrun process per allocated GPU; an explicit override also works outside Slurm.
GOBOT_NPROC_PER_NODE="${GOBOT_NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
if [[ ! "$GOBOT_NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "GOBOT_NPROC_PER_NODE must be a positive integer" >&2
  exit 2
fi
if [[ "${SLURM_JOB_NUM_NODES:-1}" != 1 ]]; then
  echo "This standalone launcher requires a single-node allocation" >&2
  exit 2
fi
if [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[1-9][0-9]*$ ]] &&
   (( GOBOT_NPROC_PER_NODE > SLURM_GPUS_ON_NODE )); then
  echo "Requested ranks exceed the allocated GPU count" >&2
  exit 2
fi
: "${GOBOT_SCENE_PROJECT:?Set the prepared Gobot G1 scene project}"
: "${GOBOT_MOTION_DATA:?Set the directory of converted 50 Hz NPZ clips}"
GOBOT_OUTPUT="${GOBOT_RUN_ROOT:-$GOBOT_ROOT/examples/g1/logs/cluster}/smoke_${SLURM_JOB_ID:-manual}"
cd "$GOBOT_ROOT"
exec "$GOBOT_PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$GOBOT_NPROC_PER_NODE" \
  -m gobot.rl.train g1-tracking --project "$GOBOT_SCENE_PROJECT" --data "$GOBOT_MOTION_DATA" \
  --config examples/g1/train/reproduced_smoke.json --output "$GOBOT_OUTPUT" "$@"
