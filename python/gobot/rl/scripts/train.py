"""Train Gobot RL tasks with RSL-RL."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .. import VectorEnv
from ..rsl_rl import GobotOnPolicyRunner, RslRlVecEnvWrapper, rsl_rl_cfg_to_dict
from ..tasks.cartpole import CartPoleEnvCfg, make_agent_cfg, make_env_cfg


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.task != "cartpole":
        raise SystemExit(f"unsupported Gobot RL task: {args.task!r}")
    run_cartpole(args)


def run_cartpole(args: argparse.Namespace) -> Path:
    env_cfg = CartPoleEnvCfg(
        project_path=args.env_project_path,
        scene=args.env_scene,
        backend=args.env_backend,
        num_envs=args.env_num_envs,
        batch_size=args.env_batch_size or args.env_num_envs,
        num_workers=args.env_num_workers,
        seed=args.seed,
        max_episode_steps=args.env_max_episode_steps,
        randomize_target_position=args.env_randomize_target,
        target_cart_position_min=args.env_target_min,
        target_cart_position_max=args.env_target_max,
        randomize_initial_angle=args.env_randomize_initial_angle,
    )
    agent_cfg = make_agent_cfg()
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.agent_max_iterations
    agent_cfg.num_steps_per_env = args.agent_num_steps_per_env
    agent_cfg.save_interval = args.agent_save_interval
    agent_cfg.experiment_name = args.agent_experiment_name
    agent_cfg.run_name = args.agent_run_name
    agent_cfg.logger = args.agent_logger
    agent_cfg.clip_actions = args.agent_clip_actions
    agent_cfg.algorithm.learning_rate = args.agent_learning_rate
    agent_cfg.algorithm.num_learning_epochs = args.agent_num_learning_epochs
    agent_cfg.algorithm.num_mini_batches = args.agent_num_mini_batches

    if args.device != "cpu":
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("non-CPU training requires torch.") from error
        if str(args.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested {args.device}, but CUDA is not available")

    np.random.seed(args.seed)
    task_cfg = make_env_cfg(env_cfg)
    task_cfg.metadata = {**dict(task_cfg.metadata), "seed": args.seed}
    log_dir = _make_log_dir(Path(args.log_root), agent_cfg.experiment_name, agent_cfg.run_name)
    params_dir = log_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    _write_json(params_dir / "env.json", task_cfg.to_dict())
    _write_json(params_dir / "agent.json", asdict(agent_cfg))

    print(
        "[INFO] Gobot PPO training: task=cartpole "
        f"device={args.device} num_envs={env_cfg.num_envs} "
        f"workers={env_cfg.num_workers or 'auto'} iterations={agent_cfg.max_iterations}"
    )
    print(f"[INFO] Logging to: {log_dir}")

    env = VectorEnv(task_cfg)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions, device=args.device)
    runner = GobotOnPolicyRunner(
        wrapped_env,
        rsl_rl_cfg_to_dict(agent_cfg),
        log_dir=str(log_dir),
        device=args.device,
    )
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()
    return log_dir


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="gobot_train")
    parser.add_argument("task", choices=["cartpole"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-root", default="logs/rsl_rl")

    parser.add_argument("--env.project-path", dest="env_project_path", default="/home/wqq/gobot/examples/cartpole")
    parser.add_argument("--env.scene", dest="env_scene", default="res://cartpole.jscn")
    parser.add_argument("--env.backend", dest="env_backend", default="mujoco")
    parser.add_argument("--env.num-envs", dest="env_num_envs", type=int, default=64)
    parser.add_argument("--env.batch-size", dest="env_batch_size", type=int, default=0)
    parser.add_argument("--env.num-workers", dest="env_num_workers", type=int, default=0)
    parser.add_argument("--env.max-episode-steps", dest="env_max_episode_steps", type=int, default=500)
    parser.add_argument("--env.randomize-target", dest="env_randomize_target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--env.target-min", dest="env_target_min", type=float, default=-1.0)
    parser.add_argument("--env.target-max", dest="env_target_max", type=float, default=1.0)
    parser.add_argument(
        "--env.randomize-initial-angle",
        dest="env_randomize_initial_angle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--agent.max-iterations", dest="agent_max_iterations", type=int, default=300)
    parser.add_argument("--agent.num-steps-per-env", dest="agent_num_steps_per_env", type=int, default=24)
    parser.add_argument("--agent.save-interval", dest="agent_save_interval", type=int, default=50)
    parser.add_argument("--agent.experiment-name", dest="agent_experiment_name", default="cartpole")
    parser.add_argument("--agent.run-name", dest="agent_run_name", default="")
    parser.add_argument("--agent.logger", dest="agent_logger", choices=["tensorboard", "wandb"], default="tensorboard")
    parser.add_argument("--agent.clip-actions", dest="agent_clip_actions", type=float, default=1.0)
    parser.add_argument("--agent.learning-rate", dest="agent_learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--agent.num-learning-epochs", dest="agent_num_learning_epochs", type=int, default=5)
    parser.add_argument("--agent.num-mini-batches", dest="agent_num_mini_batches", type=int, default=4)
    return parser.parse_args(argv)


def _make_log_dir(log_root: Path, experiment_name: str, run_name: str) -> Path:
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name:
        name = f"{name}_{run_name}"
    return (log_root / experiment_name / name).resolve()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
