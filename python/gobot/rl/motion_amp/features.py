"""Versioned global-frame AMP features (all quaternions are xyzw)."""
from __future__ import annotations
import torch
from gobot.rl.spec import ActionSpec, ObservationSpec, SpecField
from . import rotations as rotation


class Features:
    def __init__(self, profile, device, *, base_center_of_mass=(0., 0., 0.), velocity_reference='base_body_com'):
        if velocity_reference!='base_body_com':
            raise ValueError('the aligned AMP task requires base_body_com velocity')
        self.velocity_reference = velocity_reference
        self.base_center_of_mass = torch.as_tensor(base_center_of_mass, dtype=torch.float32, device=device)
        if self.base_center_of_mass.shape != (3,) or not bool(torch.isfinite(self.base_center_of_mass).all()):
            raise ValueError('base center of mass must be a finite local xyz vector')
        indices = profile['rotation_dof_indices']
        self.indices = torch.tensor(indices, device=device)
        self.axes = torch.tensor(profile['rotation_axes'], dtype=torch.float32, device=device)
        self.moving = self.indices >= 0
        self.rotation_count = len(indices)
        if ([i for i in indices if i >= 0] != list(range(len(profile['joint_names']))) or
                any(i < -1 for i in indices) or self.axes.shape != (len(indices),3) or
                not bool(torch.isfinite(self.axes).all()) or
                not torch.allclose(self.axes.norm(dim=-1),torch.ones(len(indices),device=device),atol=1.e-5)):
            raise ValueError('rotation map must enumerate each controlled hinge once with a unit axis')
        self.key_indices = [profile['body_names'].index(n) for n in profile['key_bodies']]
        self.obs_spec = ObservationSpec('motion_amp_global_com_velocity_v2', (
            SpecField('root_height', 1, 'm'), SpecField('root_rotation_xz', 6),
            SpecField('root_com_linear_velocity', 3, 'm/s'), SpecField('root_angular_velocity', 3, 'rad/s'),
            SpecField('joint_rotation_xz', len(indices)*6),
            SpecField('joint_velocity', len(profile['joint_names']), 'rad/s'),
            SpecField('key_position_relative_root', len(self.key_indices)*3, 'm')))
        self.action_spec = ActionSpec('motion_amp_absolute_joint_targets_v1',
            tuple(SpecField(n, 1, 'rad') for n in profile['joint_names']),
            lower=profile['action_low'], upper=profile['action_high'])
        self.frame_dim = self.obs_spec.dim+2

    def _velocity_offset(self, quaternion, angular_velocity):
        offset = (rotation.quat_to_matrix(quaternion) @ self.base_center_of_mass[:, None]).squeeze(-1)
        return torch.linalg.cross(angular_velocity, offset)

    def root_com_velocity(self, quaternion, origin_velocity, angular_velocity):
        """Adapt Gobot's link-origin velocity to the reference policy's COM velocity."""
        return origin_velocity + self._velocity_offset(quaternion, angular_velocity)

    def root_origin_velocity(self, quaternion, com_velocity, angular_velocity):
        """Convert reference reset velocities before passing them to ArticulationBatch."""
        return com_velocity - self._velocity_offset(quaternion, angular_velocity)

    def joint_rotations(self, q):
        angles = q[..., self.indices.clamp_min(0)] * self.moving
        return rotation.axis_angle_to_quat(self.axes, angles)

    def joint_coordinates(self, quaternions):
        return rotation.quat_twist_angle(quaternions[..., self.moving, :], self.axes[self.moving])

    def actor(self, position, quaternion, velocity, angular, joint_rot, qd, keys):
        return torch.cat((position[..., 2:3], rotation.quat_to_tan_norm(quaternion), velocity, angular,
            rotation.quat_to_tan_norm(joint_rot).flatten(-2), qd,
            (keys-position.unsqueeze(-2)).flatten(-2)), -1)

    def frame(self, position, quaternion, velocity, angular, joint_rot, qd, keys):
        # Position remains absolute in history storage. Relative XY is computed
        # against the newest frame when presenting a discriminator observation.
        return torch.cat((position, rotation.quat_to_tan_norm(quaternion),
            rotation.quat_to_tan_norm(joint_rot).flatten(-2),
            (keys-position.unsqueeze(-2)).flatten(-2), velocity, angular, qd), -1)

    def discriminator(self, history):
        relative = history.clone()
        relative[..., :2] -= history[:, -1:, :2]
        return relative.flatten(1)

    def live(self, state):
        quaternion = state.base_pose[..., 3:]
        angular = state.base_velocity[..., 3:]
        velocity = state.base_velocity[..., :3]
        velocity = self.root_com_velocity(quaternion, velocity, angular)
        args = (state.base_pose[..., :3], state.base_pose[..., 3:],
                velocity, angular,
                self.joint_rotations(state.joint_position), state.joint_velocity,
                state.link_pose[..., self.key_indices, :3])
        return self.actor(*args), self.frame(*args)
