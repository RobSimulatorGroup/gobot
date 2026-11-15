#!/usr/bin/env bash
#SBATCH --job-name=gobot-velocity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
set -euo pipefail

: "${GOBOT_REPO:?Set GOBOT_REPO to the installed Gobot checkout}"
: "${GOBOT_OUTPUT:?Set GOBOT_OUTPUT to a writable run directory}"
cd "$GOBOT_REPO"
python_bin="${GOBOT_PYTHON:-$GOBOT_REPO/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# One shared PPO policy; num-envs is the batch size on each GPU.
exec "$python_bin" -B -m torch.distributed.run --standalone \
  --nproc-per-node="${GOBOT_NPROC_PER_NODE:-8}" \
  -m gobot.rl.train "${GOBOT_TASK:-go1-velocity}" \
  --terrain "${GOBOT_TERRAIN:-rough}" --backend mujoco-warp \
  --seed "${GOBOT_SEED:-42}" --num-envs "${GOBOT_NUM_ENVS:-1024}" \
  --iterations "${GOBOT_ITERATIONS:-10000}" --no-eval \
  --hours "${GOBOT_HOURS:-3.8}" --budget-hours "${GOBOT_BUDGET_HOURS:-32}" \
  --budget-file "$GOBOT_OUTPUT/budget.json" --output "$GOBOT_OUTPUT" "$@"
