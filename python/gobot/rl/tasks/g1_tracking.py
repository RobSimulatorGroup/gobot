"""The built-in G1 task uses the pinned SuperMimic-Reproduced algorithm."""
from copy import deepcopy
from gobot.rl.motion_tracking.env import MotionTrackingEnv
from gobot.rl.motion_tracking.training import Trainer
from gobot.rl.supermimic.config import TrainConfig
from .registry import TaskSpec, register_task

TASK_ID = 'Gobot-Tracking-Flat-G1-SuperMimic-v0'


def env_cfg(**kwargs):
    return TrainConfig(**kwargs)


def play_env_cfg(**kwargs):
    cfg = env_cfg(**kwargs)
    cfg.env = deepcopy(cfg.env)
    cfg.env.domain_randomization = cfg.env.obs_noise = cfg.env.reset_noise = False
    cfg.env.action_noise_std = 0.
    cfg.env.sampling_mode = 'start'
    return cfg


register_task(TASK_ID, TaskSpec(env_cfg_factory=env_cfg,play_env_cfg_factory=play_env_cfg,
    runner_cfg_factory=TrainConfig,env_factory=MotionTrackingEnv,runner_factory=Trainer,
    description='G1 SuperMimic-Reproduced: 93D proprioception, 680D goals, explicit dynamics and BPTT'))
