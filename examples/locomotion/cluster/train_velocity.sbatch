#!/usr/bin/env bash
#SBATCH --job-name=gobot-velocity
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
set -euo pipefail

: "${GOBOT_REPO:?Set GOBOT_REPO to the installed Gobot checkout}"
: "${GOBOT_OUTPUT:?Set GOBOT_OUTPUT to a writable run directory}"
cd "$GOBOT_REPO"
seeds=(7 42 123)
index="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ! "$index" =~ ^[0-2]$ ]]; then
  echo "SLURM_ARRAY_TASK_ID must be 0, 1 or 2" >&2
  exit 2
fi
seed="${seeds[$index]}"
output="$GOBOT_OUTPUT/seed-$seed"
python_bin="${GOBOT_PYTHON:-$GOBOT_REPO/.venv/bin/python}"

# Independent seeds, one GPU per run. The standard trainer owns physics and PPO.
exec "$python_bin" -B -m gobot.rl.train "${GOBOT_TASK:-go1-velocity}" \
  --terrain "${GOBOT_TERRAIN:-rough}" --backend mujoco-warp --device cuda:0 \
  --seed "$seed" --num-envs "${GOBOT_NUM_ENVS:-1024}" \
  --iterations "${GOBOT_ITERATIONS:-10000}" --no-eval \
  --hours "${GOBOT_HOURS:-3.8}" --budget-hours "${GOBOT_BUDGET_HOURS:-12}" \
  --budget-file "$output/budget.json" --output "$output" "$@"
