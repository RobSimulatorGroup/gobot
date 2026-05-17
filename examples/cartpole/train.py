"""Train the Gobot CartPole example with PPO via rsl_rl."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from env import CartPoleVecEnv


TRAIN_CFG = {
    "num_steps_per_env": 64,
    "save_interval": 100,
    "obs_groups": {
        "actor": ["policy"],
        "critic": ["policy"],
    },
    "actor": {
        "class_name": "rsl_rl.models.MLPModel",
        "hidden_dims": [128, 128, 64],
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
        "hidden_dims": [128, 128, 64],
        "activation": "elu",
        "obs_normalization": True,
    },
    "algorithm": {
        "class_name": "rsl_rl.algorithms.PPO",
        "num_learning_epochs": 8,
        "num_mini_batches": 4,
        "clip_param": 0.2,
        "gamma": 0.99,
        "lam": 0.95,
        "value_loss_coef": 1.0,
        "entropy_coef": 0.01,
        "learning_rate": 3e-4,
        "max_grad_norm": 1.0,
        "schedule": "adaptive",
        "desired_kl": 0.01,
        "rnd_cfg": None,
        "symmetry_cfg": None,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--log-dir", type=str, default="logs/position_tracking")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-range", type=float, default=0.8)
    parser.add_argument("--disturbance-interval", type=int, default=480)
    parser.add_argument("--disturbance-duration", type=int, default=60)
    parser.add_argument("--disturbance-std", type=float, default=0.05)
    parser.add_argument("--disturbance-clip", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-out", type=str, default="policies/cartpole.pt")
    args = parser.parse_args()

    project_path = Path(__file__).resolve().parent
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = project_path / log_dir
    os.makedirs(log_dir, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Log dir: {log_dir}")

    env = CartPoleVecEnv(
        num_envs=args.num_envs,
        max_episode_length=1000,
        device=args.device,
        project_path=project_path,
        target_range=args.target_range,
        disturbance_interval=args.disturbance_interval,
        disturbance_duration=args.disturbance_duration,
        disturbance_std=args.disturbance_std,
        disturbance_clip=args.disturbance_clip,
        seed=args.seed,
    )

    runner = OnPolicyRunner(env, copy.deepcopy(TRAIN_CFG), log_dir=str(log_dir), device=args.device)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)

    final_path = log_dir / "model_final.pt"
    runner.save(str(final_path), infos={"gobot_cartpole": env.cfg})

    policy_path = Path(args.policy_out)
    if not policy_path.is_absolute():
        policy_path = project_path / policy_path
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    runner.save(str(policy_path), infos={"gobot_cartpole": env.cfg})

    print(f"Saved final model to {final_path}")
    print(f"Saved editor policy to {policy_path}")
    env.close()


if __name__ == "__main__":
    main()
