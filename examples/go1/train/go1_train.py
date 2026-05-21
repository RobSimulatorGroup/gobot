"""Train the Gobot Go1 example with PPO via rsl_rl."""

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
    parser.add_argument("--num-envs", "--num_envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--log-dir", "--log_dir", type=str, default="logs/go1")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-out", type=str, default="policies/go1.pt")
    args = parser.parse_args()

    project_path = Path(__file__).resolve().parents[1]
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = project_path / log_dir
    os.makedirs(log_dir, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Envs: {args.num_envs}")
    print(f"Log dir: {log_dir}")

    env = Go1VecEnv(
        num_envs=args.num_envs,
        max_episode_length=1000,
        device=args.device,
        project_path=project_path,
        seed=args.seed,
    )
    runner = OnPolicyRunner(env, copy.deepcopy(TRAIN_CFG), log_dir=str(log_dir), device=args.device)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)

    final_path = log_dir / "model_final.pt"
    runner.save(str(final_path), infos={"gobot_go1": env.cfg})

    policy_path = Path(args.policy_out)
    if not policy_path.is_absolute():
        policy_path = project_path / policy_path
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    runner.save(str(policy_path), infos={"gobot_go1": env.cfg})

    print(f"Saved final model to {final_path}")
    print(f"Saved editor policy to {policy_path}")
    env.close()


if __name__ == "__main__":
    main()
