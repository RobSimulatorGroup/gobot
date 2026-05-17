"""Train registered Gobot RL tasks with RSL-RL."""

from __future__ import annotations

from dataclasses import asdict, dataclass, make_dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Sequence, Type

import numpy as np
import tyro

from .. import VectorEnv
from ..rsl_rl import GobotOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper, rsl_rl_cfg_to_dict
from ..tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls


TYRO_FLAGS = (
    tyro.conf.AvoidSubcommands,
    tyro.conf.FlagConversionOff,
    tyro.conf.UsePythonSyntaxForLiteralCollections,
)


@dataclass
class TrainConfig:
    env: Any
    agent: RslRlBaseRunnerCfg
    device: str = "cpu"
    log_root: str = "logs/rsl_rl"

    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        return TrainConfig(env=load_env_cfg(task_id), agent=load_rl_cfg(task_id))


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
    if cfg.device != "cpu":
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("non-CPU training requires torch.") from error
        if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested {cfg.device}, but CUDA is not available")

    cfg.agent.seed = int(cfg.agent.seed)
    if hasattr(cfg.env, "seed"):
        setattr(cfg.env, "seed", int(cfg.agent.seed))
    np.random.seed(cfg.agent.seed)

    task_cfg = cfg.env
    task_cfg.metadata = {**dict(task_cfg.metadata), "seed": int(cfg.agent.seed), "task_id": task_id}

    params_dir = log_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    _write_json(params_dir / "env.json", cfg.env.to_dict())
    _write_json(params_dir / "agent.json", asdict(cfg.agent))

    print(
        "[INFO] Gobot PPO training: "
        f"task={task_id} device={cfg.device} num_envs={task_cfg.num_envs} "
        f"iterations={cfg.agent.max_iterations}"
    )
    print(f"[INFO] Logging to: {log_dir}")

    env = VectorEnv(task_cfg)
    try:
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions, device=cfg.device)
        runner_cls = load_runner_cls(task_id) or GobotOnPolicyRunner
        runner = runner_cls(
            wrapped_env,
            rsl_rl_cfg_to_dict(cfg.agent),
            log_dir=str(log_dir),
            device=cfg.device,
        )
        runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
    finally:
        env.close()


def launch_training(task_id: str, args: TrainConfig | None = None) -> Path:
    args = args or TrainConfig.from_task(task_id)
    log_root_path = (Path(args.log_root) / args.agent.experiment_name).resolve()
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.agent.run_name:
        log_dir_name += f"_{args.agent.run_name}"
    log_dir = log_root_path / log_dir_name
    run_train(task_id, args, log_dir)
    return log_dir


def main(argv: Sequence[str] | None = None) -> None:
    # Import built-in tasks to populate the registry.
    import gobot.rl.tasks  # noqa: F401

    prog = Path(sys.argv[0]).name if argv is None else "gobot_train"
    all_tasks = list_tasks()
    if not all_tasks:
        raise SystemExit("no Gobot RL tasks are registered")
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_top_level_help(all_tasks, prog)
        return
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        args=args,
        add_help=False,
        return_unknown_args=True,
        config=TYRO_FLAGS,
    )
    default_cfg = TrainConfig.from_task(chosen_task)
    cfg_type = _train_config_type(type(default_cfg.env), type(default_cfg.agent))
    cfg = tyro.cli(
        cfg_type,
        args=remaining_args,
        default=default_cfg,
        prog=f"{prog} {chosen_task}",
        config=TYRO_FLAGS,
    )
    launch_training(chosen_task, cfg)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _train_config_type(env_type: Type[Any], agent_type: Type[Any]) -> Type[TrainConfig]:
    return make_dataclass(
        "ResolvedTrainConfig",
        [
            ("env", env_type),
            ("agent", agent_type),
            ("device", str),
            ("log_root", str),
        ],
        bases=(TrainConfig,),
    )


def _print_top_level_help(tasks: Sequence[str], prog: str) -> None:
    print(f"usage: {prog} TASK [OPTIONS]\n")
    print("Train a registered Gobot RL task with RSL-RL.\n")
    print("tasks:")
    for task_id in tasks:
        print(f"  {task_id}")
    print("\nexamples:")
    example_task = tasks[0] if tasks else "TASK"
    print(
        f"  {prog} {example_task} "
        "--env.project-path /path/to/gobot/project "
        "--env.num-envs 64 --agent.max-iterations 200 --device cpu"
    )
    print(f"\nRun `{prog} TASK --help` for task-specific env and agent options.")


if __name__ == "__main__":
    main()
