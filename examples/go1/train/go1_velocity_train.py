"""Train Go1 with the Gobot-native MJLab-style velocity task."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

import gobot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="go1_rough", help="Velocity task name, e.g. go1_rough or go1_flat.")
    parser.add_argument("--num-envs", "--num_envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--max-episode-length", type=int, default=None)
    parser.add_argument("--log-dir", "--log_dir", type=str, default="logs/go1_velocity")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-out", type=str, default="policies/go1_velocity.pt")
    parser.add_argument("--no-terrain-curriculum", dest="terrain_curriculum", action="store_false", default=True)
    parser.add_argument("--no-obs-noise", dest="obs_noise", action="store_false", default=True)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest model_*.pt in log-dir.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume from.")
    args = parser.parse_args()

    project_path = Path(__file__).resolve().parents[1]
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = project_path / log_dir
    os.makedirs(log_dir, exist_ok=True)

    cfg = gobot.rl.velocity_task_cfg(args.task, project_path=project_path)
    cfg.terrain_curriculum = bool(args.terrain_curriculum)
    cfg.observations.actor_noise = bool(args.obs_noise)
    cfg.terrain_curriculum_steps = max(1, int(args.iterations * 24 * 0.6))

    print(f"Task: {cfg.name}")
    print(f"Device: {args.device}")
    print(f"Envs: {args.num_envs}")
    print(f"Log dir: {log_dir}")

    env = gobot.rl.GobotVelocityEnv(
        cfg,
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        max_episode_length=args.max_episode_length,
    )
    print(f"Obs actor/critic/actions: {env.num_obs}/{env.num_privileged_obs}/{env.num_actions}")

    train_cfg = gobot.rl.rsl_rl_train_cfg(
        experiment_name=cfg.name,
        max_iterations=args.iterations,
        save_interval=50,
        obs_normalization=False,
    )
    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir=str(log_dir), device=args.device)
    checkpoint = _resolve_checkpoint(args.checkpoint, log_dir) if args.checkpoint or args.resume else None
    if checkpoint is not None:
        infos = runner.load(str(checkpoint), map_location=args.device)
        env.set_training_progress(runner.current_learning_iteration * train_cfg["num_steps_per_env"])
        print(f"Resumed checkpoint: {checkpoint}")
        print(f"Resume iteration: {runner.current_learning_iteration}")
        if infos:
            print(f"Checkpoint infos: {sorted(infos.keys())}")

    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=checkpoint is None)

    final_path = log_dir / "model_final.pt"
    runner.save(str(final_path), infos={"gobot_velocity": env.cfg})

    policy_path = Path(args.policy_out)
    if not policy_path.is_absolute():
        policy_path = project_path / policy_path
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    runner.save(str(policy_path), infos={"gobot_velocity": env.cfg})

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
    checkpoints = sorted(log_dir.glob("model_*.pt"), key=_checkpoint_iteration)
    if not checkpoints:
        raise FileNotFoundError(f"--resume requested but no model_*.pt checkpoints found in {log_dir}")
    return checkpoints[-1]


def _checkpoint_iteration(path: Path) -> int:
    suffix = path.stem.removeprefix("model_")
    return int(suffix) if suffix.isdigit() else -1


if __name__ == "__main__":
    main()
