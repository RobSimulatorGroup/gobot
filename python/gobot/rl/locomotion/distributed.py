"""Process coordination for PPO; RSL-RL owns gradient and normalization updates."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import os

import torch
import torch.distributed as dist


@dataclass
class DistributedTraining:
    device: str
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    _owns_group: bool = False

    @classmethod
    def from_environment(cls, device: str):
        world = int(os.environ.get('WORLD_SIZE', '1'))
        rank = int(os.environ.get('RANK', '0')) if world > 1 else 0
        local = int(os.environ.get('LOCAL_RANK', '0')) if world > 1 else 0
        if world < 1 or not 0 <= rank < world or local < 0:
            raise ValueError('invalid torchrun rank or world size')
        if device.startswith('cuda') and world > 1:
            device = f'cuda:{local}'
        return cls(device, rank, local, world)

    @property
    def main(self):
        return self.rank == 0

    @property
    def algorithm_config(self):
        if self.world_size == 1:
            return None
        return dict(global_rank=self.rank, local_rank=self.local_rank, world_size=self.world_size)

    def initialize(self):
        if self.device.startswith('cuda'):
            torch.cuda.set_device(self.device)
            os.environ['MUJOCO_EGL_DEVICE_ID'] = str(torch.cuda.current_device())
        if self.world_size > 1:
            if dist.is_initialized():
                raise RuntimeError('PPO training must own its process group; launch with torchrun')
            dist.init_process_group(
                backend='nccl' if self.device.startswith('cuda') else 'gloo',
                rank=self.rank, world_size=self.world_size, timeout=timedelta(minutes=10))
            self._owns_group = True

    def close(self):
        if self._owns_group:
            dist.destroy_process_group()
            self._owns_group = False

    def on_main(self, function):
        """Run shared I/O once and propagate its result or error to every worker."""
        if self.world_size == 1:
            return function()
        result = [None, None]
        error = None
        if self.main:
            try:
                result[0] = function()
            except Exception as caught:
                error = caught
                result[1] = f'{type(caught).__name__}: {caught}'
        dist.broadcast_object_list(result, src=0, device=torch.device(self.device))
        if result[1] is not None:
            if error is not None:
                raise error
            raise RuntimeError(f'rank 0 failed: {result[1]}')
        return result[0]

    def check_error(self, error, *, phase):
        """Finish a local rollout before any worker enters PPO collectives.

        Rollout collection has no collectives. A worker that fails must still
        meet its peers here, or they block in the next PPO all-reduce.
        """
        if self.world_size == 1:
            if error is not None:
                raise error
            return
        failed = self.reduce([error is not None], maximum=True)[0]
        if not failed:
            return
        errors = [None] * self.world_size
        message = None if error is None else f'rank {self.rank}: {type(error).__name__}: {error}'
        dist.all_gather_object(errors, message)
        detail = '; '.join(message for message in errors if message is not None)
        raise RuntimeError(f'{phase} failed: {detail}') from error

    def gather(self, value):
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size if self.main else None
        dist.gather_object(value, values, dst=0)
        return values

    def reduce(self, values, *, maximum=False):
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX if maximum else dist.ReduceOp.SUM)
        return tensor.tolist()

    def stop_reason(self, requested, budget):
        # The budget and its file lock belong only to rank 0. A signal on any
        # worker stops all workers at the same rollout boundary.
        stopped, exhausted = self.reduce(
            [bool(requested), self.main and budget is not None and budget.remaining <= 0], maximum=True)
        return 'signal' if stopped else 'budget' if exhausted else None


def cpu_state(value):
    """Keep gathered checkpoints off peer GPUs and independent of live buffers."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_state(item) for item in value)
    return value
