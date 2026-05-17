"""Train Go1 velocity tracking with PPO via rsl_rl."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from go1_env import Go1VecEnv


TRAIN_CFG = {
    "num_steps_per_env": 24,
    "save_interval": 100,
    "obs_groups": {
        "actor": ["policy"],
        "critic": ["policy"],
    },
    "actor": {
        "class_name": "rsl_rl.models.MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "rsl_rl.modules.GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    },
    "critic": {
        "class_name": "rsl_rl.models.MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
    },
    "algorithm": {
        "class_name": "rsl_rl.algorithms.PPO",
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "clip_param": 0.2,
        "gamma": 0.99,
        "lam": 0.95,
        "value_loss_coef": 1.0,
        "entropy_coef": 0.01,
        "learning_rate": 1e-3,
        "max_grad_norm": 1.0,
        "schedule": "adaptive",
        "desired_kl": 0.01,
        "rnd_cfg": None,
        "symmetry_cfg": None,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--log-dir", type=str, default="logs/go1")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    example_dir = Path(__file__).resolve().parent
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = example_dir / log_dir
    os.makedirs(log_dir, exist_ok=True)

    print(f"Device: {args.device} | Envs: {args.num_envs} | Iters: {args.iterations}")
    print(f"Log dir: {log_dir}")

    env = Go1VecEnv(num_envs=args.num_envs, device=args.device, seed=args.seed)
    runner = OnPolicyRunner(env, copy.deepcopy(TRAIN_CFG), log_dir=str(log_dir), device=args.device)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)

    final_path = log_dir / "model_final.pt"
    runner.save(str(final_path), infos={"gobot_go1": env.cfg})
    print(f"Saved: {final_path}")
    env.close()


if __name__ == "__main__":
    main()
