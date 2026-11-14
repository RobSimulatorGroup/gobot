"""Tracking coordination; RSL-RL owns synchronous PPO gradients/normalizers."""

from dataclasses import replace
import hashlib
import torch
from gobot.rl.locomotion.distributed import DistributedTraining, cpu_state


def resolve_config(cfg, group):
    count = cfg.num_envs
    if cfg.batch_scope == "global":
        if count % group.world_size:
            raise ValueError("global num_envs must be divisible by world size")
        count //= group.world_size
    local = replace(
        cfg, num_envs=count, device=group.device, seed=cfg.seed + group.rank
    )
    local.validate()
    return local


def state_digest(state):
    digest = hashlib.sha256()

    def visit(value):
        if torch.is_tensor(value):
            v = value.detach().cpu().contiguous()
            digest.update(str((v.dtype, tuple(v.shape))).encode())
            digest.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(repr(key).encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode())

    visit(state)
    return digest.hexdigest()


__all__ = ["DistributedTraining", "cpu_state", "resolve_config", "state_digest"]
