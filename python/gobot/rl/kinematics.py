"""Differentiable tree kinematics from Gobot's owned physics scene snapshot.

Transforms describe the authored rest geometry, as used by PhysicsSceneCompiler.
Joint initial_position is a reset coordinate, not a second rest-frame rotation.
Positions and velocities refer to link origins in world coordinates.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np
import torch
from torch import nn


def axis_rotation(axis, angle):
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(axis.shape[:-1] + (3, 3))
    eye = torch.eye(3, dtype=angle.dtype, device=angle.device)
    return eye + torch.sin(angle)[..., None, None] * skew + (1-torch.cos(angle))[..., None, None] * (skew @ skew)


class SceneKinematics(nn.Module):
    """Fixed, revolute/continuous and prismatic joints with one floating root."""

    def __init__(self, snapshot, *, robot_name, base_link, joint_names, tracked_links=None,
                 device='cpu', dtype=torch.float32):
        super().__init__()
        if snapshot.get('schema_version') != 1:
            raise ValueError('unsupported physics snapshot schema')
        matches = [robot for robot in snapshot['robots'] if robot['name'] == robot_name]
        if len(matches) != 1:
            raise ValueError(f'expected one robot {robot_name!r}')
        robot = matches[0]
        links = {link['name']: link for link in robot['links'] if link.get('role') != 'virtual_root'}
        self.joint_names = tuple(joint_names)
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError('joint names must be unique')
        if base_link not in links:
            raise ValueError('base link is not a physical link')
        parent_joints = {}
        for joint in robot['joints']:
            kind = int(joint['type'])
            if kind not in (0, 1, 2, 3, 4):
                raise ValueError(f"unsupported joint type for {joint['name']}")
            child = joint['child_link']
            if child in parent_joints:
                raise ValueError('multiple joints on one child or closed constraints are unsupported')
            parent_joints[child] = joint
            if kind == 4 and child != base_link:
                raise ValueError('only the root may float')
        root_joint = parent_joints.get(base_link)
        if root_joint and root_joint['parent_link']:
            raise ValueError('base link must be the articulation root')
        names = [base_link]
        while len(names) < len(links):
            added = [name for name in links if name not in names and name in parent_joints
                     and parent_joints[name]['parent_link'] in names]
            if not added:
                raise ValueError('disconnected links or cyclic physics topology; author fixed joints explicitly')
            names.extend(added)
        self.body_names = tuple(names)
        self.num_bodies = len(names)
        self.parent = [-1]
        self.joint_idx = [-1]
        self.kinds = [4]
        a, b, axes = [np.eye(4)], [np.eye(4)], [np.array([1., 0, 0])]
        lower = np.full(len(self.joint_names), -np.inf)
        upper = np.full(len(self.joint_names), np.inf)
        seen = set()

        def transform(value):
            result = np.asarray(value['global_transform']['matrix'], dtype=np.float64)
            if (result.shape != (4, 4) or not np.isfinite(result).all()
                    or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1.e-5)
                    or not np.isclose(np.linalg.det(result[:3, :3]), 1., atol=1.e-5)):
                raise ValueError('FK requires finite rigid transforms without scale/shear')
            return result

        for name in names[1:]:
            joint = parent_joints[name]
            kind = int(joint['type'])
            parent = names.index(joint['parent_link'])
            self.parent.append(parent); self.kinds.append(kind)
            if kind == 0:
                index = -1
            else:
                if joint['name'] not in self.joint_names:
                    raise ValueError(f"missing movable joint {joint['name']!r} in control order")
                index = self.joint_names.index(joint['name']); seen.add(joint['name'])
                if joint['upper_limit'] > joint['lower_limit']:
                    lower[index], upper[index] = joint['lower_limit'], joint['upper_limit']
            self.joint_idx.append(index)
            anchor = transform(joint)
            a.append(np.linalg.inv(transform(links[names[parent]])) @ anchor)
            b.append(np.linalg.inv(anchor) @ transform(links[name]))
            axis = np.asarray(joint['axis'], dtype=np.float64)
            if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1.e-10:
                raise ValueError('joint axis must be finite and nonzero')
            axes.append(axis / np.linalg.norm(axis))
        if seen != set(self.joint_names):
            raise ValueError('control order contains unknown or fixed joints')
        a, b, axes = map(np.asarray, (a, b, axes))
        for name, values in (('parent_joint', a), ('joint_child', b), ('axes', axes), ('q_lo', lower), ('q_hi', upper)):
            self.register_buffer(name, torch.as_tensor(values, dtype=dtype, device=device))
        self.tracked_idx = tuple(names.index(name) for name in (tracked_links or names))
        payload = {'names': names, 'joints': self.joint_names, 'parent': self.parent,
                   'kinds': self.kinds, 'a': a.tolist(), 'b': b.tolist(), 'axes': axes.tolist()}
        self.digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

    @property
    def device(self):
        return self.axes.device

    def forward(self, p, rotation, velocity, angular_velocity, q, qd, all_bodies=False):
        if q.shape != qd.shape or q.shape[-1] != len(self.joint_names):
            raise ValueError('joint state does not match the kinematics control order')
        positions, rotations = [p], [rotation]
        velocities, angular = [velocity], [angular_velocity]
        for i in range(1, self.num_bodies):
            parent, index = self.parent[i], self.joint_idx[i]
            parent_r = rotations[parent]
            a, b = self.parent_joint[i], self.joint_child[i]
            anchor_r = parent_r @ a[:3, :3]
            anchor_offset = (parent_r @ a[:3, 3, None]).squeeze(-1)
            anchor_p = positions[parent] + anchor_offset
            anchor_v = velocities[parent] + torch.linalg.cross(angular[parent], anchor_offset)
            axis = (anchor_r @ self.axes[i, :, None]).squeeze(-1)
            if self.kinds[i] in (1, 2):
                moving_r = anchor_r @ axis_rotation(self.axes[i], q[..., index])
                w = angular[parent] + axis * qd[..., index, None]
            else:
                moving_r, w = anchor_r, angular[parent]
            if self.kinds[i] == 3:
                shift = axis * q[..., index, None]
                anchor_p = anchor_p + shift
                anchor_v = anchor_v + torch.linalg.cross(angular[parent], shift) + axis * qd[..., index, None]
            offset = (moving_r @ b[:3, 3, None]).squeeze(-1)
            positions.append(anchor_p + offset)
            rotations.append(moving_r @ b[:3, :3])
            velocities.append(anchor_v + torch.linalg.cross(w, offset))
            angular.append(w)
        ids = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (torch.stack([positions[i] for i in ids], -2),
                torch.stack([rotations[i] for i in ids], -3),
                torch.stack([velocities[i] for i in ids], -2),
                torch.stack([angular[i] for i in ids], -2))
