"""Synthetic tensor-only probe for learner updates; this is not a physics environment.

Both processes use identical dynamics/rewards to isolate the official learner's
objective, survival, bootstrap, GAE and optimizer order. No Gobot import here.
"""

from __future__ import annotations

import torch
from torch import nn
from tensordict import TensorDict


class DifferentiableProbeEnv:
    def __init__(self, starts):
        self.starts = starts
        self.representation = nn.Module()
        self.representation.dt = 0.02
        self.dynamics = nn.Linear(29, 68, bias=False)
        self.max_step_reward = 0.1

    def sample_initial_states(self):
        return self.starts.clone()

    def reset(self, batch):
        self.state = batch["state"].detach().clone()
        self.motion_goal = batch["goal_motion"]
        self.velocity_goal = batch["goal_velocity"]
        self.context = batch["context"]
        return self.observation()

    def observation(self):
        return TensorDict(
            {
                "history": self.state.unsqueeze(1),
                "goal_motion": self.motion_goal + 0.03 * self.state[:, :1],
                "goal_velocity": self.velocity_goal + 0.02 * self.state[:, 1:4],
                "critic": torch.cat([self.state, self.motion_goal, self.context], -1),
            },
            batch_size=[self.state.shape[0]],
        )

    def step(self, action):
        self.state = self.state + 0.02 * self.dynamics(action)
        reward = 0.1 * torch.exp(
            -self.state.square().mean(-1)
        ) - 0.001 * action.square().sum(-1)
        done = torch.sigmoid((self.state[:, 0] - 0.5) * 3)
        return self.observation(), reward, done, {}
