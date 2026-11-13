# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# Copyright (c) 2020 Preferred Networks, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from rsl-rl-lib 5.5.2. See LICENSE-rsl-rl.
"""Running statistics shared across ranks, with stable CUDA graph buffers."""

from __future__ import annotations
import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(
        self,
        shape: int | tuple[int, ...] | list[int],
        eps: float = 1e-2,
        until: int | None = None,
    ) -> None:
        """Initialize EmpiricalNormalization module.

        .. note:: The normalization parameters are computed over the whole batch, not for each environment separately.

        Args:
            shape: Shape of input values except batch axis.
            eps: Small value for stability.
            until: If this arg is specified, the module learns input values until the sum of batch sizes exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self) -> torch.Tensor:
        """Return the current running mean."""
        return self._mean.squeeze(0).clone()  # type: ignore

    @property
    def std(self) -> torch.Tensor:
        """Return the current running standard deviation."""
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize mean and variance of values based on empirical values."""
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        """Learn input values without computing the output values of them."""
        if not self.training:
            return
        if self.until is not None and self.count >= self.until:
            return

        count_x = torch.tensor(
            x.shape[0], dtype=self.count.dtype, device=self.count.device
        )  # type: ignore
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)

        if torch.distributed.is_initialized():
            # Compute the global mean first, then combine the local variances around that mean
            local_mean_x = mean_x
            torch.distributed.all_reduce(count_x)
            mean_sum_x = mean_x * x.shape[0]
            torch.distributed.all_reduce(mean_sum_x)
            mean_x = mean_sum_x / count_x
            var_sum_x = x.shape[0] * (var_x + (local_mean_x - mean_x).square())
            torch.distributed.all_reduce(var_sum_x)
            var_x = var_sum_x / count_x

        self.count += count_x
        rate = count_x / self.count
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std.copy_(torch.sqrt(self._var))

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """De-normalize values based on empirical values."""
        return y * (self._std + self.eps) + self._mean
