"""Adapt the learning state layout to scene-owned differentiable kinematics."""
from __future__ import annotations

import torch
from torch import nn

from gobot.rl.kinematics import SceneKinematics
from gobot.rl.supermimic.geometry import mat_to_quat
from gobot.rl.supermimic.rotation_primitives import matrix_from_quat


class SceneStateKinematics(nn.Module):
    """Named FK for [root p, wxyz, joints] and [v_world, w_body, joint rates]."""

    def __init__(self, scene: SceneKinematics):
        super().__init__()
        self.scene = scene
        self.body_names = scene.body_names
        self.joint_names = scene.joint_names

    def body_index(self, name):
        return self.body_names.index(name)

    def body_indices(self, names):
        return [self.body_index(name) for name in names]

    def forward(self, positions, velocities):
        count = len(self.joint_names)
        if positions.shape[-1] != 7+count or velocities.shape[-1] != 6+count:
            raise ValueError('learning state dimensions differ from the scene articulation')
        rotation = matrix_from_quat(positions[..., 3:7])
        angular_world = (rotation @ velocities[..., 3:6, None]).squeeze(-1)
        p, r, v, w = self.scene(positions[..., :3], rotation, velocities[..., :3], angular_world,
                                positions[..., 7:], velocities[..., 6:], all_bodies=True)
        return {'pos': p, 'quat': mat_to_quat(r), 'lin_vel': v, 'ang_vel': w}
