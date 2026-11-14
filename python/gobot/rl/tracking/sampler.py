# Adapted from mjlab dcdf6ce5, Apache-2.0. See LICENSE-mjlab.
# Modified: explicit RNG, validation, and accumulation across multiple reset calls.
"""Failure-count EMA sampler. Timeouts and manual resets are not failures."""

import math
import torch
import torch.nn.functional as F


class AdaptiveSampler:
    def __init__(
        self,
        frames,
        dt,
        *,
        device="cpu",
        alpha=0.001,
        uniform=0.1,
        kernel_size=1,
        decay=0.8,
    ):
        if frames < 2 or dt <= 0 or not math.isfinite(dt):
            raise ValueError("invalid motion clock")
        if (
            not 0 < alpha <= 1
            or uniform <= 0
            or not math.isfinite(uniform)
            or kernel_size < 1
            or not 0 <= decay <= 1
        ):
            raise ValueError("invalid sampler parameters")
        self.frames = frames
        self.bins = int(frames // (1 / dt)) + 1
        self.alpha = alpha
        self.uniform = uniform
        self.failed = torch.zeros(self.bins, device=device)
        self.pending = torch.zeros_like(self.failed)
        self.kernel = decay ** torch.arange(
            kernel_size, device=device, dtype=torch.float32
        )
        self.kernel /= self.kernel.sum()

    def record(self, frames, failures):
        bins = (frames[failures] * self.bins // self.frames).clamp(0, self.bins - 1)
        self.pending.add_(torch.bincount(bins, minlength=self.bins))

    def update(self):
        self.failed.mul_(1 - self.alpha).add_(self.pending, alpha=self.alpha)
        self.pending.zero_()

    def probabilities(self):
        w = self.failed + self.uniform / self.bins
        w = F.pad(w[None, None], (0, len(self.kernel) - 1), mode="replicate")
        w = F.conv1d(w, self.kernel[None, None]).flatten()
        return w / w.sum()

    def sample(self, count, generator, mode="adaptive"):
        if count == 0:
            return torch.empty(0, device=self.failed.device, dtype=torch.long)
        kw = dict(device=self.failed.device, generator=generator)
        if mode == "start":
            return torch.zeros(count, device=self.failed.device, dtype=torch.long)
        if mode == "uniform":
            return torch.randint(self.frames, (count,), **kw)
        if mode != "adaptive":
            raise ValueError("unknown sampling mode")
        bins = torch.multinomial(
            self.probabilities(), count, replacement=True, generator=generator
        )
        return ((bins + torch.rand(count, **kw)) / self.bins * (self.frames - 1)).long()
