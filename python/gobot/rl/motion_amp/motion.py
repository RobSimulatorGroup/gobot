"""Numeric motion archives and scene-owned kinematics; no pickle or MJCF runtime."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from . import rotations as rotation


class MotionLibrary:
    def __init__(self, path, profile, name, features, kinematics, device):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != profile['files']['motions.npz']:
            raise ValueError('motion archive digest mismatch')
        matches = [m for m in profile['motions'] if m['name'] == name]
        if len(matches) != 1:
            raise ValueError(f'unknown motion {name!r}')
        self.length = float(matches[0]['length'])
        if not math.isfinite(self.length) or self.length <= 0:
            raise ValueError('motion length must be positive')
        self.wrap = torch.tensor(matches[0]['wrap_delta'], device=device)
        self.features, self.kinematics, self.device = features, kinematics, device
        with np.load(path, allow_pickle=False) as archive:
            self.frames = {field:torch.tensor(archive[f'{name}_{field}'], device=device)
                for field in ('root_pos','root_rot','root_vel','root_ang_vel','joint_rot','dof_vel')}
        self.count = len(self.frames['root_pos'])
        expected = {'root_pos':(3,), 'root_rot':(4,), 'root_vel':(3,), 'root_ang_vel':(3,),
            'joint_rot':(features.rotation_count, 4), 'dof_vel':(features.action_spec.dim,)}
        for field, shape in expected.items():
            value = self.frames[field]
            if value.shape != (self.count, *shape) or not bool(torch.isfinite(value).all()) or self.count < 2:
                raise ValueError(f'invalid motion field {field}')
        self.key_indices = [kinematics.body_names.index(n) for n in profile['key_bodies']]

    def sample_times(self, count, generator):
        return torch.rand(count, device=self.device, generator=generator)*self.length

    def sample(self, times):
        """Interpolate cyclic poses; retain the reference's lower-frame velocities."""
        phase = times/self.length
        cycles = torch.floor(phase)
        frame = (phase-cycles)*(self.count-1)
        lo = frame.long(); hi = (lo+1).clamp_max(self.count-1)
        weight = frame-lo
        a = self.frames
        p = (1-weight[..., None])*a['root_pos'][lo] + weight[..., None]*a['root_pos'][hi]
        p = p + cycles[..., None]*self.wrap
        r = rotation.slerp(a['root_rot'][lo], a['root_rot'][hi], weight)
        j = rotation.slerp(a['joint_rot'][lo], a['joint_rot'][hi], weight[..., None])
        return p, r, a['root_vel'][lo], a['root_ang_vel'][lo], j, a['dof_vel'][lo]

    def history(self, times, steps, dt):
        offsets = torch.arange(1-steps, 1, device=self.device)*dt
        p, r, v, w, j, qd = self.sample(times[:, None]+offsets)
        q = self.features.joint_coordinates(j)
        bodies, *_ = self.kinematics(p, rotation.quat_to_matrix(r), v, w, q, qd, all_bodies=True)
        return self.features.frame(p, r, v, w, j, qd, bodies[..., self.key_indices, :])
