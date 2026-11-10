"""Configurable velocity PPO/AMP workflows shared by the CLI and Python."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from gobot.rl.locomotion.experiment import RunConfig
from .workflow import TrainingSpec, update_config


@dataclass
class LocomotionTrainConfig:
    experiment: RunConfig = field(default_factory=RunConfig)
    task_factory: str | None = None
    evaluation_command: tuple[str, ...] = ()
    backend: str = "mujoco-warp"
    device: str | None = None
    output: Path = Path("logs/locomotion")
    resume: Path | None = None
    initialize: Path | None = None
    budget_file: Path | None = None
    no_eval: bool = True
    terrain_reference: Path | None = None
    # Partial overrides of the selected task/agent defaults. Python callers may
    # also supply a complete VelocityTaskCfg as env.
    env: Any = field(default_factory=dict)
    agent: dict = field(default_factory=dict)

    def resolve_env(self):
        from gobot.rl.locomotion.config import VelocityTaskCfg
        if isinstance(self.env, VelocityTaskCfg):
            from copy import deepcopy
            return deepcopy(self.env)
        if not self.task_factory:
            raise ValueError("provide env=VelocityTaskCfg(...) or task_factory='module:function'")
        import importlib
        module, name = self.task_factory.split(":", 1)
        defaults = getattr(importlib.import_module(module), name)(self.experiment)
        if not isinstance(defaults, VelocityTaskCfg):
            raise TypeError("task_factory must return VelocityTaskCfg")
        profile_fields = {"robot_name", "base_link", "joint_names", "default_joint_pos", "kp", "kd", "foot_names", "foot_link_names"}
        if defaults.robot is not None and isinstance(self.env, dict) and profile_fields.intersection(self.env):
            raise ValueError("configure actuator/joint fields through env.robot; derived env fields are owned by the robot profile")
        return update_config(defaults, self.env)

    def resolve_agent(self, env=None):
        from gobot.rl.locomotion.training import agent_config, validate_agent_config
        env = self.resolve_env() if env is None else env
        defaults = agent_config(self.experiment, experiment_name=env.name)
        allowed = {"actor", "critic", "algorithm", "num_steps_per_env", "obs_groups"}
        if set(self.agent) - allowed:
            raise ValueError(f"unsupported agent setting {sorted(set(self.agent) - allowed)[0]!r}; use experiment for run length, checkpoints and logging, env for action limits")
        _check_keys(defaults, self.agent, "agent")
        config = update_config(defaults, self.agent)
        validate_agent_config(self.experiment, config)
        return config

    def validate(self):
        self.experiment.__post_init__()
        if self.terrain_reference is not None:
            if self.experiment.stage != 'rough' or self.experiment.recipe != 'baseline':
                raise ValueError('--terrain-reference requires rough baseline training')
            from gobot.rl.locomotion.terrain_reference import load
            load(self.terrain_reference)
        if self.resume and self.initialize:
            raise ValueError("choose --resume or --initialize, not both")
        if not self.no_eval and not self.evaluation_command:
            raise ValueError("training requires --no-eval or an explicit evaluation_command")
        if isinstance(self.env, dict) and self.env and not self.no_eval:
            raise ValueError("custom env/reward settings require --no-eval and an independent evaluator matching that task")
        if self.experiment.reference_reset_probability and (
            self.backend != "mujoco-warp" or not self.experiment.stage.startswith("flat")
        ):
            raise ValueError("reference-state reset requires flat MuJoCo Warp training and a matching motion dataset")
        from gobot.rl.locomotion.training import validate_task_config
        env = self.resolve_env()
        agent = self.resolve_agent(env)
        validate_task_config(self.experiment, env, agent)


def _check_keys(defaults, overrides, path):
    if not isinstance(overrides, dict):
        raise ValueError(f"{path} must be an object")
    for key, value in overrides.items():
        if key not in defaults:
            raise ValueError(f"unknown configuration field: {path}.{key}")
        if isinstance(defaults[key], dict):
            _check_keys(defaults[key], value, f"{path}.{key}")


def _arguments(parser):
    parser.add_argument("--terrain", dest="stage", choices=("flat", "rough", "mixed"))
    for name, kind in (("iterations", int), ("hours", float), ("budget-hours", float),
                       ("eval-interval", int), ("eval-steps", int), ("eval-envs", int),
                       ("save-interval", int)):
        parser.add_argument("--" + name, type=kind)
    parser.add_argument("--initialize", "--fork", dest="initialize", type=Path)
    parser.add_argument("--budget-file", type=Path)
    parser.add_argument("--terrain-reference", type=Path, help="Verify and sample the shared authored training terrain")
    parser.add_argument("--no-eval", action="store_true", default=None)
    parser.add_argument("--dataset", type=Path, help="AMP motion feature dataset")


def _configure(cfg, args):
    changes = {key: getattr(args, key) for key in (
        "stage", "iterations", "hours", "budget_hours", "eval_interval", "eval_steps", "eval_envs", "save_interval"
    ) if getattr(args, key) is not None}
    if args.dataset is not None:
        changes["motion_dataset"] = str(args.dataset)
    cfg = replace(cfg, experiment=replace(cfg.experiment, **changes))
    return replace(cfg, **{key: getattr(args, key) for key in (
        "initialize", "budget_file", "no_eval", "terrain_reference"
    ) if getattr(args, key) is not None})


def _run(cfg):
    from gobot.rl.locomotion.training import run
    source_files = ()
    if cfg.task_factory:
        import importlib
        module = importlib.import_module(cfg.task_factory.split(":", 1)[0])
        if getattr(module, "__file__", None):
            source_files = (module.__file__,)
    return run(
        cfg.experiment, cfg.output, backend=cfg.backend, device=cfg.device,
        budget_file=cfg.budget_file, resume=cfg.resume, initialize=cfg.initialize,
        no_eval=cfg.no_eval, terrain_reference=cfg.terrain_reference,
        task_cfg=cfg.resolve_env(),
        agent_cfg=cfg.resolve_agent(), evaluation_command=cfg.evaluation_command, source_files=source_files,
    )


def training_spec(*, task_factory: str | None = None, experiment: RunConfig | None = None,
                  env=None, name="velocity", evaluation_command=()) -> TrainingSpec:
    """Build a trainer from a typed task or an importable task configuration factory.

    The task supplies robot assets, controls, rewards and sensors. The trainer
    owns rollout collection, PPO, checkpointing and distributed execution.
    """
    from copy import deepcopy
    settings = experiment if experiment is not None else RunConfig()

    def factory():
        return LocomotionTrainConfig(
            experiment=deepcopy(settings), task_factory=task_factory,
            env=deepcopy(env) if env is not None else {},
            output=Path("logs") / name, evaluation_command=tuple(evaluation_command),
            no_eval=not bool(evaluation_command),
        )

    return TrainingSpec(
        config_factory=factory, run=_run, description=f"{name} velocity training",
        step_unit="ppo_updates", supports_distributed=settings.recipe != "amp", common_paths={
            "backend": "backend", "device": "device", "num_envs": "experiment.num_envs",
            "seed": "experiment.seed", "output": "output", "resume": "resume",
        }, add_arguments=_arguments, configure=_configure,
    )
