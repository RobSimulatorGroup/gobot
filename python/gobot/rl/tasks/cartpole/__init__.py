"""CartPole task registration."""

from __future__ import annotations

from ..registry import register_gobot_task
from .builder import cartpole_env_cfg
from .env_cfg import (
    CartPoleBalanceEnvCfg,
    CartPoleBalancePlayEnvCfg,
    CartPoleTargetEnvCfg,
    CartPoleTargetPlayEnvCfg,
)
from .rl_cfg import CartPoleBalancePPOCfg, CartPoleTargetPPOCfg


register_gobot_task(
    "Gobot-CartPole-Balance",
    env_cfg=CartPoleBalanceEnvCfg,
    env_builder=cartpole_env_cfg,
    play_env_cfg=CartPoleBalancePlayEnvCfg,
    rl_cfg=CartPoleBalancePPOCfg,
)

register_gobot_task(
    "Gobot-CartPole-Target",
    env_cfg=CartPoleTargetEnvCfg,
    env_builder=cartpole_env_cfg,
    play_env_cfg=CartPoleTargetPlayEnvCfg,
    rl_cfg=CartPoleTargetPPOCfg,
)


__all__ = [
    "CartPoleBalanceEnvCfg",
    "CartPoleBalancePlayEnvCfg",
    "CartPoleBalancePPOCfg",
    "CartPoleTargetEnvCfg",
    "CartPoleTargetPlayEnvCfg",
    "CartPoleTargetPPOCfg",
    "cartpole_env_cfg",
]
