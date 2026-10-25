"""Backward-compatible Go1 preset for the shared velocity environment."""
from gobot.rl.locomotion import cpu_env as _implementation
from gobot.rl.locomotion.cpu_env import CpuVelocityEnv
from .go1_velocity_cfg import go1_velocity_cfg

class Go1VelocityEnv(CpuVelocityEnv):
    def __init__(self, cfg=None, **kwargs):
        super().__init__(cfg if cfg is not None else go1_velocity_cfg(), **kwargs)

    def _run_go1_rough_task_numpy(self):
        return self._run_velocity_task_numpy()

    def _go1_pose_std(self, command_speed):
        return self._posture_std(command_speed)

def __getattr__(name):
    return getattr(_implementation, name)
