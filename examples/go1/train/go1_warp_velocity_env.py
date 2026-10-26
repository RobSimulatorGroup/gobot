"""Backward-compatible Go1 preset for the shared velocity environment."""
from gobot.rl.locomotion import warp_env as _implementation
from gobot.rl.locomotion.warp_env import WarpVelocityEnv
from .go1_velocity_cfg import go1_velocity_cfg

class Go1WarpVelocityEnv(WarpVelocityEnv):
    def __init__(self, cfg=None, **kwargs):
        super().__init__(cfg if cfg is not None else go1_velocity_cfg(), **kwargs)

def __getattr__(name):
    return getattr(_implementation, name)
