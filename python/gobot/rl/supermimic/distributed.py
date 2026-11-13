"""Synchronous data parallelism for the official WM/BPTT algorithm.

No learning-rate scaling. Global batches are split evenly by default. Collectives
stay outside compiled functions, and gradient averaging precedes both clipping
and Muon/Adam. CUDA graph capture records the same NCCL order on every rank.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import os

import torch
import torch.distributed as dist


def resolve_config(cfg, world_size, local_rank=0):
    """Return per-rank sizes without mutating the user's global recipe."""
    changes = {}
    if cfg.batch_scope == "global":
        for name, size in (
            ("num_envs", cfg.num_envs),
            ("wm.batch_size", cfg.wm.batch_size),
            ("policy.batch_size", cfg.policy.batch_size),
        ):
            if size % world_size:
                raise ValueError(
                    f"global {name}={size} must be divisible by world_size={world_size}"
                )
        changes.update(
            num_envs=cfg.num_envs // world_size,
            wm=replace(cfg.wm, batch_size=cfg.wm.batch_size // world_size),
            policy=replace(cfg.policy, batch_size=cfg.policy.batch_size // world_size),
        )
    if cfg.device and cfg.device.startswith("cuda") and world_size > 1:
        changes["device"] = f"cuda:{local_rank}"
    resolved = replace(cfg, **changes)
    resolved.validate()
    # Four equal critic minibatches on every rank; never silently drop samples.
    if world_size > 1 and resolved.policy.batch_size * resolved.policy.horizon % 4:
        raise ValueError("per-rank policy.batch_size * horizon must be divisible by 4")
    return resolved


class Distributed:
    def __init__(self, device="cpu"):
        self.device = device
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.primary = self.rank == 0
        self.owns_group = False

    @classmethod
    def initialize(cls, device):
        world = int(os.environ.get("WORLD_SIZE", "1"))
        owns = world > 1 and not dist.is_initialized()
        if owns:
            dist.init_process_group(
                "nccl" if device.startswith("cuda") else "gloo",
                timeout=timedelta(minutes=30),
            )
        result = cls(device)
        result.owns_group = owns
        return result

    def close(self):
        if self.owns_group:
            dist.destroy_process_group()

    def all_ready(self, ready):
        return not self.any(not ready)

    def any(self, value):
        if self.world_size == 1:
            return bool(value)
        flag = torch.tensor(int(value), device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def sum(self, values):
        result = torch.as_tensor(values, dtype=torch.float64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(result)
        return result.cpu().tolist()

    def maximum(self, value):
        if self.world_size == 1:
            return value
        result = torch.tensor(value, dtype=torch.float64, device=self.device)
        dist.all_reduce(result, op=dist.ReduceOp.MAX)
        return result.item()

    def primary_call(self, fn):
        """Propagate rank-zero I/O errors instead of stranding peers at a barrier."""
        outcome = [None]
        if self.primary:
            try:
                outcome[0] = (True, fn())
            except Exception as error:
                outcome[0] = (False, f"{type(error).__name__}: {error}")
        if self.world_size > 1:
            dist.broadcast_object_list(outcome, src=0, device=torch.device(self.device))
        success, value = outcome[0]
        if not success:
            raise RuntimeError(value)
        return value

    @torch.no_grad()
    def broadcast(self, module):
        if self.world_size > 1:
            for value in (*module.parameters(), *module.buffers()):
                dist.broadcast(value, src=0)

    def gather(self, value):
        if self.world_size == 1:
            return [value]
        result = [None] * self.world_size if self.primary else None
        dist.gather_object(value, result, dst=0)
        return result


class GradientReducer:
    """One fused collective per optimizer step, with a fixed active parameter set.

    The first eager warmup verifies the gradient layout on all ranks. An inactive
    encoder retains grad=None (and no optimizer state), as in the single-GPU run.
    Rank-dependent/changed layouts fail rather than silently pairing wrong grads.
    """

    def __init__(self):
        self.layout = None

    @torch.no_grad()
    def __call__(self, parameters):
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return
        layout = tuple(
            (tuple(p.shape), p.dtype, p.grad is not None) for p in parameters
        )
        if self.layout is None:
            layouts = [None] * dist.get_world_size()
            dist.all_gather_object(layouts, layout)
            if any(item != layout for item in layouts):
                raise RuntimeError(
                    "gradient layout differs across official learner ranks"
                )
            self.layout = layout
        elif layout != self.layout:
            raise RuntimeError("official learner gradient layout changed after warmup")
        grads = [p.grad for p in parameters if p.grad is not None]
        # Current official models are FP32; bucket by dtype to keep this generic.
        for dtype in dict.fromkeys(g.dtype for g in grads):
            group = [g for g in grads if g.dtype == dtype]
            flat = torch.cat([g.reshape(-1) for g in group])
            dist.all_reduce(flat)
            flat.div_(dist.get_world_size())
            offset = 0
            for grad in group:
                grad.copy_(flat[offset : offset + grad.numel()].view_as(grad))
                offset += grad.numel()
