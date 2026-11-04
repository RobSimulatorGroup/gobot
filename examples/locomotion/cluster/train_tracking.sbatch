#!/usr/bin/env bash
#SBATCH --job-name=gobot-supermimic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=gobot-supermimic-%j.log
set -euo pipefail
GOBOT_ROOT="${GOBOT_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
GOBOT_PYTHON="${GOBOT_PYTHON:-$GOBOT_ROOT/.venv/bin/python}"
: "${GOBOT_SCENE_PROJECT:?Set the prepared Gobot G1 scene project}"
: "${GOBOT_MOTION_LIBRARY:?Set the packed BONES training library}"
GOBOT_OUTPUT="${GOBOT_RUN_ROOT:-$GOBOT_ROOT/examples/g1/logs/cluster}/g1_${SLURM_JOB_ID:-manual}_s${GOBOT_SEED:-0}"
cd "$GOBOT_ROOT"
exec "$GOBOT_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m gobot.rl.train --project "$GOBOT_SCENE_PROJECT" --library "$GOBOT_MOTION_LIBRARY" \
  --config examples/g1/train/reproduced_bones.json --output "$GOBOT_OUTPUT" --seed "${GOBOT_SEED:-0}" "$@"
