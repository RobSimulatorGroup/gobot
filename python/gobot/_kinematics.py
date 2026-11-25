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

try:
    from gobot import _coal_native as _native_fk
except ImportError:
    _native_fk = None


def axis_rotation(axis, angle):
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(axis.shape[:-1] + (3, 3))
    eye = torch.eye(3, dtype=angle.dtype, device=angle.device)
    return eye + torch.sin(angle)[..., None, None] * skew + (1-torch.cos(angle))[..., None, None] * (skew @ skew)


class SceneKinematics(nn.Module):
    """Fixed, revolute/continuous and prismatic joints with one floating root."""

    def __init__(self, snapshot, *, robot_name, base_link, joint_names, tracked_links=None,
                 device='cpu', dtype=torch.float32, include_virtual_base=False):
        super().__init__()
        if snapshot.get('schema_version') != 1:
            raise ValueError('unsupported physics snapshot schema')
        matches = [robot for robot in snapshot['robots'] if robot['name'] == robot_name]
        if len(matches) != 1:
            raise ValueError(f'expected one robot {robot_name!r}')
        robot = matches[0]
        links = {link['name']: link for link in robot['links']
                 if link.get('role') != 'virtual_root' or (include_virtual_base and link['name'] == base_link)}
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
        # An imported world can contain fixed world geometry alongside the
        # articulation (for example mjlab's terrain body). FK describes the
        # selected base tree, not unrelated static siblings. A disconnected
        # movable joint remains an error rather than being silently discarded.
        static = {name for name in links if name != base_link and
                  (name not in parent_joints or
                   (int(parent_joints[name]['type']) == 0 and not parent_joints[name]['parent_link']))}
        while True:
            added = {name for name, joint in parent_joints.items() if name in links and name not in static
                     and int(joint['type']) == 0 and joint['parent_link'] in static}
            if not added:
                break
            static.update(added)
        if tracked_links and static.intersection(tracked_links):
            raise ValueError('tracked links must belong to the selected articulation, not static world geometry')
        links = {name: link for name, link in links.items() if name not in static}
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
        skew = np.zeros((len(axes), 3, 3), dtype=np.float64)
        skew[:, 0, 1], skew[:, 0, 2] = -axes[:, 2], axes[:, 1]
        skew[:, 1, 0], skew[:, 1, 2] = axes[:, 2], -axes[:, 0]
        skew[:, 2, 0], skew[:, 2, 1] = -axes[:, 1], axes[:, 0]
        skew2 = skew @ skew
        for name, values in (
            ('parent_joint', a), ('joint_child', b), ('axes', axes),
            ('q_lo', lower), ('q_hi', upper),
            ('_eye3', np.eye(3, dtype=np.float64)),
            ('_axis_skew', skew), ('_axis_skew2', skew2),
        ):
            self.register_buffer(name, torch.as_tensor(values, dtype=dtype, device=device))
        self.tracked_idx = tuple(names.index(name) for name in (tracked_links or names))
        self._np_parent = np.ascontiguousarray(self.parent, dtype=np.int64)
        self._np_kinds = np.ascontiguousarray(self.kinds, dtype=np.int64)
        self._np_joint_idx = np.ascontiguousarray(self.joint_idx, dtype=np.int64)
        self._np_parent_joint = np.ascontiguousarray(a, dtype=np.float64)
        self._np_joint_child = np.ascontiguousarray(b, dtype=np.float64)
        self._np_axes = np.ascontiguousarray(axes, dtype=np.float64)
        self._np_tracked_idx = np.ascontiguousarray(self.tracked_idx, dtype=np.int64)
        self._np_all_idx = np.ascontiguousarray(np.arange(self.num_bodies, dtype=np.int64))
        ancestor = np.zeros((self.num_bodies, len(self.joint_names)), dtype=bool)
        is_prismatic = np.zeros((len(self.joint_names),), dtype=bool)
        for body in range(1, self.num_bodies):
            j = self.joint_idx[body]
            if j >= 0 and self.kinds[body] == 3:
                is_prismatic[j] = True
            cur = body
            while cur > 0:
                cj = self.joint_idx[cur]
                if cj >= 0:
                    ancestor[body, cj] = True
                cur = self.parent[cur]
        self.register_buffer('body_ancestor', torch.as_tensor(ancestor[list(self.tracked_idx)], dtype=torch.bool, device=device))
        self.register_buffer('all_body_ancestor', torch.as_tensor(ancestor, dtype=torch.bool, device=device))
        self.register_buffer('joint_is_prismatic', torch.as_tensor(is_prismatic, dtype=torch.bool, device=device))
        payload = {'names': names, 'joints': self.joint_names, 'parent': self.parent,
                   'kinds': self.kinds, 'a': a.tolist(), 'b': b.tolist(), 'axes': axes.tolist()}
        self.digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

    @property
    def device(self):
        return self.axes.device

    def poses(self, position, rotation, q, all_bodies=False):
        """Evaluate poses over any leading batch dimensions of q."""
        return self._evaluate(position, rotation, q, all_bodies)

    def poses_and_joints(self, position, rotation, q, all_bodies=False):
        """Evaluate link poses alongside world-space joint origins and axes."""
        return self._evaluate(position, rotation, q, all_bodies, return_joints=True)

    def forward(self, p, rotation, velocity, angular_velocity, q, qd, all_bodies=False):
        if q.shape != qd.shape:
            raise ValueError('joint state does not match the kinematics control order')
        return self._evaluate(p, rotation, q, all_bodies, velocity, angular_velocity, qd)

    def _evaluate(self, p, rotation, q, all_bodies, velocity=None, angular_velocity=None, qd=None,
                  return_joints=False):
        if q.shape[-1] != len(self.joint_names):
            raise ValueError('joint state does not match the kinematics control order')
        if (_native_fk is not None and qd is None and self.axes.ndim == 2
                and q.device.type == 'cpu' and p.device.type == 'cpu'
                and not torch.is_grad_enabled()
                and not q.requires_grad and not p.requires_grad and not rotation.requires_grad
                and not torch._C._functorch.is_functorch_wrapped_tensor(q)
                and not torch._C._functorch.is_functorch_wrapped_tensor(p)
                and not torch._C._functorch.is_functorch_wrapped_tensor(rotation)):
            batch_shape = q.shape[:-1]
            q_flat = np.ascontiguousarray(q.reshape(-1, len(self.joint_names)).numpy(), dtype=np.float64)
            n_rows = q_flat.shape[0]
            if n_rows > 0:
                if p.ndim == 1:
                    rp = np.ascontiguousarray(p.numpy(), dtype=np.float64)
                    rr = np.ascontiguousarray(rotation.numpy(), dtype=np.float64)
                else:
                    # Detect expanded constant root to avoid materializing N copies.
                    if all(s == 0 for s in p.stride()[:-1]) and all(s == 0 for s in rotation.stride()[:-2]):
                        rp = np.ascontiguousarray(p.reshape(-1, 3)[0].numpy(), dtype=np.float64)
                        rr = np.ascontiguousarray(rotation.reshape(-1, 3, 3)[0].numpy(), dtype=np.float64)
                    else:
                        rp = np.ascontiguousarray(p.reshape(n_rows, 3).numpy(), dtype=np.float64)
                        rr = np.ascontiguousarray(rotation.reshape(n_rows, 3, 3).numpy(), dtype=np.float64)
                idx = self._np_all_idx if all_bodies else self._np_tracked_idx
                out = _native_fk.forward_kinematics(
                    self._np_parent, self._np_kinds, self._np_joint_idx,
                    self._np_parent_joint, self._np_joint_child, self._np_axes,
                    idx, rp, rr, q_flat, return_joints,
                )
                n_t = len(idx)
                pos = torch.from_numpy(out[0]).to(q.dtype).reshape(*batch_shape, n_t, 3)
                rot = torch.from_numpy(out[1]).to(q.dtype).reshape(*batch_shape, n_t, 3, 3)
                if return_joints:
                    jpos = torch.from_numpy(out[2]).to(q.dtype).reshape(*batch_shape, len(self.joint_names), 3)
                    jax = torch.from_numpy(out[3]).to(q.dtype).reshape(*batch_shape, len(self.joint_names), 3)
                    return pos, rot, jpos, jax
                return pos, rot
        positions, rotations = [p], [rotation]
        velocities, angular = [velocity], [angular_velocity]
        joint_origins = [None] * len(self.joint_names) if return_joints else None
        joint_axes = [None] * len(self.joint_names) if return_joints else None
        use_cached_skew = self.axes.ndim == 2
        for i in range(1, self.num_bodies):
            parent, index = self.parent[i], self.joint_idx[i]
            parent_r = rotations[parent]
            a, b = self.parent_joint[i], self.joint_child[i]
            kind = self.kinds[i]
            anchor_r = parent_r @ a[..., :3, :3]
            anchor_offset = (parent_r @ a[..., :3, 3, None]).squeeze(-1)
            anchor_p = positions[parent] + anchor_offset
            moving_r = anchor_r
            if kind in (1, 2):
                angle = q[..., index]
                if use_cached_skew:
                    rot = (self._eye3 + torch.sin(angle)[..., None, None] * self._axis_skew[i]
                           + (1 - torch.cos(angle))[..., None, None] * self._axis_skew2[i])
                else:
                    rot = axis_rotation(self.axes[i], angle)
                moving_r = anchor_r @ rot
            need_axis = kind == 3 or (qd is not None and kind in (1, 2)) or (return_joints and index >= 0)
            if need_axis:
                axis = (anchor_r @ self.axes[i][..., None]).squeeze(-1)
            if return_joints and index >= 0:
                joint_origins[index] = anchor_p
                joint_axes[index] = axis
            if kind == 3:
                shift = axis * q[..., index, None]
                anchor_p = anchor_p + shift
            offset = (moving_r @ b[..., :3, 3, None]).squeeze(-1)
            positions.append(anchor_p + offset)
            rotations.append(moving_r @ b[..., :3, :3])
            if qd is not None:
                w = angular[parent]
                anchor_v = velocities[parent] + torch.linalg.cross(w, anchor_offset)
                if kind in (1, 2):
                    w = w + axis * qd[..., index, None]
                elif kind == 3:
                    anchor_v = anchor_v + torch.linalg.cross(w, shift) + axis * qd[..., index, None]
                velocities.append(anchor_v + torch.linalg.cross(w, offset))
                angular.append(w)
        ids = range(self.num_bodies) if all_bodies else self.tracked_idx
        poses = (torch.stack([positions[i] for i in ids], -2),
                 torch.stack([rotations[i] for i in ids], -3))
        if return_joints:
            return (*poses, torch.stack(joint_origins, -2), torch.stack(joint_axes, -2))
        if qd is None:
            return poses
        return (*poses,
                torch.stack([velocities[i] for i in ids], -2),
                torch.stack([angular[i] for i in ids], -2))
