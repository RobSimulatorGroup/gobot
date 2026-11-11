"""Small RSL-RL model adapters for the reference AMP training recipe."""
from __future__ import annotations

import torch
from torch import nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.modules.distribution import GaussianDistribution


class ClippedNormalization(EmpiricalNormalization):
    """Reuse RSL running moments with the reference standard-deviation floor."""
    def __init__(self, shape, *, until=100000000):
        super().__init__(shape, eps=0., until=until)

    @torch.no_grad()
    def update(self, x):
        super().update(x)
        self._std.clamp_min_(1.e-4)

    def forward(self, x):
        return super().forward(x).clamp(-10., 10.)


class AffineOutput(nn.Module):
    def __init__(self, scale, offset):
        super().__init__()
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32).clone())
        self.register_buffer('offset', torch.as_tensor(offset, dtype=torch.float32).clone())

    def forward(self, x):
        return x * self.scale + self.offset


class JointRangeGaussian(GaussianDistribution):
    """Optimize normalized means; expose physical joint-angle samples to PPO/env."""
    def __init__(self, output_dim, *, action_scale, action_offset, init_std=.05, std_type='scalar'):
        if std_type != 'scalar':
            raise ValueError('JointRangeGaussian uses fixed scalar-space standard deviations')
        super().__init__(output_dim, init_std=init_std, learn_std=False)
        scale = torch.as_tensor(action_scale, dtype=torch.float32)
        offset = torch.as_tensor(action_offset, dtype=torch.float32)
        if (scale.shape != (output_dim,) or offset.shape != (output_dim,)
                or not bool(torch.isfinite(scale).all() and torch.isfinite(offset).all())
                or not bool((scale > 0).all()) or not 0 < init_std < float('inf')):
            raise ValueError('invalid joint action scale, offset or standard deviation')
        self.output_transform = AffineOutput(scale, offset)
        with torch.no_grad():
            self.std_param.mul_(scale)

    def update(self, mlp_output):
        super().update(self.output_transform(mlp_output))

    def deterministic_output(self, mlp_output):
        return self.output_transform(mlp_output)

    def as_deterministic_output_module(self):
        return AffineOutput(self.output_transform.scale, self.output_transform.offset)


class ReferenceMLP(MLPModel):
    """RSL MLP with reference initialization and clipped observation normalization."""
    def __init__(self, *args, normalization_samples=100000000, **kwargs):
        super().__init__(*args, **kwargs)
        if self.obs_normalization:
            self.obs_normalizer = ClippedNormalization(self.obs_dim, until=normalization_samples)
        layers = [module for module in self.mlp.modules() if isinstance(module, nn.Linear)]
        for layer in layers:
            nn.init.zeros_(layer.bias)
        if self.distribution is not None:
            nn.init.uniform_(layers[-1].weight, -.01, .01)
