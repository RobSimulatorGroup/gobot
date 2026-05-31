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
        "critic": ["critic"],
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
    parser.add_argument("--terrain-curriculum", dest="terrain_curriculum", action="store_true", default=True)
    parser.add_argument("--no-terrain-curriculum", dest="terrain_curriculum", action="store_false")
    parser.add_argument("--terrain-curriculum-steps", type=int, default=None)
    parser.add_argument("--spawn-jitter", type=float, default=0.35)
    parser.add_argument("--base-clearance", type=float, default=0.32)
    parser.add_argument("--height-scan", dest="height_scan", action="store_true", default=True)
    parser.add_argument("--no-height-scan", dest="height_scan", action="store_false")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in log-dir.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume from.")
    args = parser.parse_args()

    project_path = Path(__file__).resolve().parents[1]
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = project_path / log_dir
    os.makedirs(log_dir, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Envs: {args.num_envs}")
    print(f"Log dir: {log_dir}")
    terrain_curriculum_steps = args.terrain_curriculum_steps
    if terrain_curriculum_steps is None:
        terrain_curriculum_steps = max(1, int(args.iterations * TRAIN_CFG["num_steps_per_env"] * 0.6))

    env = Go1VecEnv(
        num_envs=args.num_envs,
        max_episode_length=1000,
        device=args.device,
        project_path=project_path,
        seed=args.seed,
        terrain_curriculum=args.terrain_curriculum,
        terrain_curriculum_steps=terrain_curriculum_steps,
        spawn_jitter=args.spawn_jitter,
        base_clearance=args.base_clearance,
        height_scan=args.height_scan,
    )
    print(f"Obs: {env.num_obs}")
    print(f"Terrain curriculum: {args.terrain_curriculum} ({terrain_curriculum_steps} policy steps)")
    runner = OnPolicyRunner(env, copy.deepcopy(TRAIN_CFG), log_dir=str(log_dir), device=args.device)
    checkpoint = _resolve_checkpoint(args.checkpoint, log_dir) if args.checkpoint or args.resume else None
    if checkpoint is not None:
        infos = runner.load(str(checkpoint), map_location=args.device)
        env.set_training_progress(runner.current_learning_iteration * TRAIN_CFG["num_steps_per_env"])
        print(f"Resumed checkpoint: {checkpoint}")
        print(f"Resume iteration: {runner.current_learning_iteration}")
        if infos:
            print(f"Checkpoint infos: {sorted(infos.keys())}")
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=checkpoint is None)

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


def _resolve_checkpoint(checkpoint: str | None, log_dir: Path) -> Path:
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = log_dir / path
        if not path.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        return path

    checkpoints = sorted(
        log_dir.glob("model_*.pt"),
        key=lambda path: _checkpoint_iteration(path),
    )
    if not checkpoints:
        raise FileNotFoundError(f"--resume requested but no model_*.pt checkpoints found in {log_dir}")
    return checkpoints[-1]


def _checkpoint_iteration(path: Path) -> int:
    stem = path.stem
    if stem.startswith("model_"):
        suffix = stem.removeprefix("model_")
        if suffix.isdigit():
            return int(suffix)
    return -1


if __name__ == "__main__":
    main()
