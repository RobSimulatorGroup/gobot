"""Built-in Gobot RL task configs."""

from .registry import (
    list_tasks,
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
    register_gobot_task,
)

# Import built-in task modules so they register their task IDs.
from .cartpole import (  # noqa: E402
    CartPoleBalanceEnvCfg,
    CartPoleBalancePlayEnvCfg,
    CartPoleBalancePPOCfg,
    CartPoleTargetEnvCfg,
    CartPoleTargetPlayEnvCfg,
    CartPoleTargetPPOCfg,
)

__all__ = [
    "CartPoleBalanceEnvCfg",
    "CartPoleBalancePlayEnvCfg",
    "CartPoleBalancePPOCfg",
    "CartPoleTargetEnvCfg",
    "CartPoleTargetPlayEnvCfg",
    "CartPoleTargetPPOCfg",
    "list_tasks",
    "load_env_cfg",
    "load_rl_cfg",
    "load_runner_cls",
    "register_gobot_task",
]
