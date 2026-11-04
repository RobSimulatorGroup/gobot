"""Configure rank-local Gobot simulation and reference training."""
from copy import deepcopy
import random
import numpy as np
import torch
from gobot.rl.supermimic.distributed import init_distributed, finalize


def run(cfg, task_id='Gobot-Tracking-Flat-G1-SuperMimic-v0'):
    from gobot.rl import tasks
    cfg = deepcopy(cfg)
    cfg.validate()
    rank, local_rank, world = init_distributed(cfg.device)
    if world>1 and cfg.device.startswith('cuda'):
        cfg.device = f'cuda:{local_rank}'
    seed = cfg.seed+1000*rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    cfg.policy.goal_dim = cfg.env.future_frames*68+(9 if cfg.env.anchor_obs else 0)
    cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
    cfg.policy_trainer.future_frames = cfg.env.future_frames
    cfg.policy_trainer.history = cfg.policy.history
    env = trainer = None
    try:
        env = tasks.make_env(task_id,cfg)
        trainer = tasks.make_runner(task_id,env,cfg)
        return trainer.run()
    finally:
        if trainer:
            trainer.close()
        if env:
            env.close()
        finalize()
