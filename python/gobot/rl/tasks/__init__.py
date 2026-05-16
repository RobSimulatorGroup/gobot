"""Built-in Gobot RL task configs."""

from .registry import (
    list_tasks,
    load_env_builder,
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
    register_gobot_task,
)

# Import built-in task modules so they register their task IDs.
from .cartpole import (  # noqa: E402
    CartPoleEnvCfg,
    CartPoleTrainCfg,
    make_agent_cfg,
    make_balance_agent_cfg,
    make_balance_env_cfg,
    make_env_cfg,
    make_target_agent_cfg,
    make_target_env_cfg,
    make_train_cfg,
)

__all__ = [
    "CartPoleEnvCfg",
    "CartPoleTrainCfg",
    "list_tasks",
    "load_env_builder",
    "load_env_cfg",
    "load_rl_cfg",
    "load_runner_cls",
    "make_agent_cfg",
    "make_balance_agent_cfg",
    "make_balance_env_cfg",
    "make_env_cfg",
    "make_target_agent_cfg",
    "make_target_env_cfg",
    "make_train_cfg",
    "register_gobot_task",
]
