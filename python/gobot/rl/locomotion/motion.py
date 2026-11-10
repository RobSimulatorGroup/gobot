"""Named robot motion features and validated, clip-bounded reference sampling."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch


@dataclass(frozen=True)
class MotionDatasetCfg:
    path: str
    history_dt: float | None = None


@dataclass(frozen=True)
class MotionPriorCfg:
    dataset: str
    weight: float = 2.0
    fall_penalty: float = 2.0
    learning_rate: float = 1e-4
    gradient_penalty: float = 10.0
    replay_capacity: int = 200_000
    batch_size: int = 1024
    updates_per_iteration: int = 1
    warmup_updates: int = 500
    speed_tolerance: float = .25
    lateral_tolerance: float = .15
    yaw_tolerance: float = .35
    seed: int = 42
    run_blend_start: float = 1.4
    run_blend_end: float = 1.8

    def __post_init__(self):
        if self.weight < 0 or self.fall_penalty < 0 or self.learning_rate <= 0:
            raise ValueError("motion weights must be nonnegative and learning rate positive")
        if min(self.replay_capacity, self.batch_size, self.updates_per_iteration) < 1:
            raise ValueError("motion replay, batch and update counts must be positive")
        if self.run_blend_end <= self.run_blend_start:
            raise ValueError("running transition must have a positive speed interval")


def quat_inverse_apply(q, vector):
    xyz, scalar = q[..., 1:], q[..., :1]
    t = 2 * torch.cross(xyz.expand_as(vector), vector, dim=-1)
    return vector - scalar * t + torch.cross(xyz.expand_as(vector), t, dim=-1)


@dataclass(frozen=True)
class MotionFeatureSpec:
    """Named layout shared by reference conversion and runtime feature extraction."""

    joint_names: tuple[str, ...]
    link_names: tuple[str, ...]

    def __post_init__(self):
        for field in ("joint_names", "link_names"):
            value = getattr(self, field)
            if isinstance(value, (str, bytes)):
                raise ValueError(f"motion {field} must be a sequence of names")
            names = tuple(value)
            if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
                raise ValueError(f"motion {field} must contain unique nonempty names")
            object.__setattr__(self, field, names)
        if not self.joint_names:
            raise ValueError("motion feature layout requires named joints")

    @property
    def fields(self):
        return (("joint_position", len(self.joint_names)),
                ("joint_velocity", len(self.joint_names)),
                ("base_linear_velocity_body", 3),
                ("base_angular_velocity_body", 3),
                ("projected_gravity", 3),
                ("link_position_body", 3 * len(self.link_names)))

    @property
    def dim(self):
        return sum(size for _, size in self.fields)

    def section(self, name):
        start = 0
        for key, size in self.fields:
            if key == name:
                return slice(start, start + size)
            start += size
        raise KeyError(name)

    @property
    def velocity_indices(self):
        linear = self.section("base_linear_velocity_body").start
        yaw = self.section("base_angular_velocity_body").start + 2
        return (linear, linear + 1, yaw)

    def metadata(self):
        return {"schema": "gobot_motion_features_v1", "joint_names": list(self.joint_names),
                "link_names": list(self.link_names)}

    @classmethod
    def from_metadata(cls, value):
        if not isinstance(value, dict) or value.get("schema") != "gobot_motion_features_v1":
            raise ValueError("motion dataset requires a named feature_spec")
        return cls(value["joint_names"], value["link_names"])


def motion_features(state, *, spec: MotionFeatureSpec, device="cpu"):
    """Joint state, base twist/gravity and named link positions in the base frame."""
    def t(key):
        value = state[key]
        if isinstance(value, np.ndarray) and not value.flags.writeable:
            value = value.copy()
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    position, q = t("base_position"), t("base_quaternion")
    joints, velocities = t("joint_position"), t("joint_velocity")
    links = t("link_position")
    linear, angular = t("base_linear_velocity_body"), t("base_angular_velocity_body")
    count = len(position)
    expected = {"base_position": (position, (count, 3)), "base_quaternion": (q, (count, 4)),
                "joint_position": (joints, (count, len(spec.joint_names))),
                "joint_velocity": (velocities, (count, len(spec.joint_names))),
                "link_position": (links, (count, len(spec.link_names), 3)),
                "base_linear_velocity_body": (linear, (count, 3)),
                "base_angular_velocity_body": (angular, (count, 3))}
    for name, (value, shape) in expected.items():
        if tuple(value.shape) != shape:
            raise ValueError(f"motion {name} shape {tuple(value.shape)} differs from feature layout {shape}")
    gravity = torch.zeros_like(position)
    gravity[..., 2] = -1
    relative = quat_inverse_apply(q[:, None], links - position[:, None]).flatten(1)
    return torch.cat((joints, velocities, linear, angular,
                      quat_inverse_apply(q, gravity), relative), dim=-1)


class VelocityReferenceIndex:
    """Precomputed, ragged velocity buckets; GPU sampling has no per-environment search."""
    origin = np.array([-3.2, -1., -1.2], np.float32)
    spacing = np.array([.2, .1, .2], np.float32)
    shape = np.array([33, 21, 13], np.int64)

    def __init__(self, pairs, labels, *, feature_spec: MotionFeatureSpec, device, tolerances=(.25,.15,.35)):
        from itertools import product
        indices = np.asarray(feature_spec.velocity_indices)
        if pairs.ndim != 2 or pairs.shape[1] != 2 * feature_spec.dim:
            raise ValueError("motion transition width differs from feature layout")
        first = pairs[:, indices]
        last = pairs[:, indices + feature_spec.dim]
        center=np.rint(((first+last)*.5-self.origin)/self.spacing).astype(np.int64)
        moving=(np.linalg.norm(first[:,:2],axis=1)>=.2)&(np.linalg.norm(last[:,:2],axis=1)>=.2)
        tolerance=np.asarray(tolerances,np.float32)
        offsets=np.ceil(tolerance/self.spacing).astype(int)
        keys=[];indices=[];cells=int(np.prod(self.shape))
        for shift in product(*(range(-int(n),int(n)+1) for n in offsets)):
            bins=center+shift
            target=self.origin+bins*self.spacing
            valid=moving & ((bins>=0)&(bins<self.shape)).all(1)
            valid &= (np.abs(first-target)<=tolerance+1e-6).all(1)&(np.abs(last-target)<=tolerance+1e-6).all(1)
            valid &= (np.abs(target[:,0])<.05)|((first[:,0]*target[:,0]>0)&(last[:,0]*target[:,0]>0))
            ids=np.flatnonzero(valid)
            key=labels[ids]*cells+(bins[ids,0]*self.shape[1]+bins[ids,1])*self.shape[2]+bins[ids,2]
            keys.append(key);indices.append(ids)
        keys=np.concatenate(keys);indices=np.concatenate(indices)
        counts=np.bincount(keys,minlength=2*cells)
        order=np.argsort(keys,kind='stable')
        self.indices=torch.as_tensor(indices[order],device=device)
        self.counts=torch.as_tensor(counts,device=device)
        self.offsets=torch.as_tensor(np.r_[0,np.cumsum(counts[:-1])],device=device)
        self.first=torch.as_tensor(first,device=device);self.last=torch.as_tensor(last,device=device)
        self.tolerance=torch.tensor(tolerances,device=device)
        self._origin=torch.as_tensor(self.origin,device=device);self._spacing=torch.as_tensor(self.spacing,device=device)
        self._shape=torch.as_tensor(self.shape,device=device);self.cells=cells
        self.signature={'schema':'directional_motion_v3','tolerances':list(tolerances),
                        'feature_spec':feature_spec.metadata()}

    def commands(self, value):
        if value.ndim==1:return torch.stack((value,torch.zeros_like(value),torch.zeros_like(value)),1)
        if value.shape[-1]!=3:raise ValueError('reference commands require vx, vy, yaw rate')
        return value

    def lookup(self, labels, commands):
        commands=self.commands(commands)
        bins=((commands-self._origin)/self._spacing).round().long()
        valid=((bins>=0)&(bins<self._shape)).all(1)&(commands[:,:2].norm(dim=1)>=.2)
        bins=torch.minimum(bins.clamp_min(0),self._shape-1)
        keys=labels*self.cells+(bins[:,0]*self.shape[1]+bins[:,1])*self.shape[2]+bins[:,2]
        return keys,valid&(self.counts[keys]>0)

    def sample(self, labels, commands, generator):
        commands=self.commands(commands)
        keys,available=self.lookup(labels,commands)
        chosen=torch.zeros_like(labels);supported=torch.zeros_like(available)
        if not len(self.indices):return chosen,supported
        for _ in range(4):
            draw=(torch.rand(len(labels),device=labels.device,generator=generator)*self.counts[keys]).long()
            ids=self.indices[(self.offsets[keys]+draw).clamp_max(len(self.indices)-1)]
            valid=available & ((self.first[ids]-commands).abs()<=self.tolerance+1e-6).all(1)
            valid &= ((self.last[ids]-commands).abs()<=self.tolerance+1e-6).all(1)
            valid &= (commands[:,0].abs()<.05)|((self.first[ids,0]*commands[:,0]>0)&(self.last[ids,0]*commands[:,0]>0))
            use=valid&~supported;chosen[use]=ids[use];supported|=valid
        return chosen,supported


class MotionDataset:
    def __init__(self,path,*,device='cpu',seed=42,split='train',tolerances=(.25,.15,.35)):
        cfg = path if isinstance(path, MotionDatasetCfg) else MotionDatasetCfg(str(path))
        path = Path(cfg.path)
        self.digest = hashlib.sha256(path.read_bytes()).hexdigest()
        pairs, labels, roots, joints = [], [], [], []
        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data['metadata']))
            if self.metadata.get('schema') != 'gobot_motion_v2':
                raise ValueError('unsupported motion schema; regenerate data with a named feature layout')
            self.feature_spec = MotionFeatureSpec.from_metadata(self.metadata.get('feature_spec'))
            self.dt = float(self.metadata['dt'])
            if not math.isfinite(self.dt) or self.dt <= 0:
                raise ValueError('motion control period must be finite and positive')
            if cfg.history_dt is not None and not math.isclose(cfg.history_dt, self.dt):
                raise ValueError('motion dataset control period differs from the requested history_dt')
            for i, clip in enumerate(self.metadata['clips']):
                # Splits are data provenance. Dataset-specific subject names belong in converters.
                if clip['split'] != split:
                    continue
                features = data[f'features_{i}']
                valid = data[f'valid_{i}'].astype(bool)
                if (features.ndim != 2 or features.shape[1] != self.feature_spec.dim
                        or valid.shape != (len(features),) or not np.isfinite(features[valid]).all()):
                    raise ValueError(f'invalid reference clip {clip["name"]}')
                gait = clip['gait']
                if isinstance(gait, bool) or not isinstance(gait, int) or gait not in (0, 1):
                    raise ValueError('velocity motion prior requires gait labels 0 (walking) or 1 (running)')
                adjacent = valid[:-1] & valid[1:]
                if not adjacent.any():
                    continue
                pairs.append(np.concatenate((features[:-1], features[1:]), axis=1)[adjacent])
                labels.append(np.full(int(adjacent.sum()), gait, np.int64))
                has_root, has_joints = f'root_{i}' in data, f'joint_{i}' in data
                if has_root != has_joints:
                    raise ValueError(f'incomplete reset state in reference clip {clip["name"]}')
                if has_root:
                    root, joint = data[f'root_{i}'], data[f'joint_{i}']
                    if (root.shape != (len(features), 13)
                            or joint.shape != (len(features), 2 * len(self.feature_spec.joint_names))
                            or not np.isfinite(root[valid]).all() or not np.isfinite(joint[valid]).all()):
                        raise ValueError(f'invalid reset state in reference clip {clip["name"]}')
                    roots.append(root[:-1][adjacent])
                    joints.append(joint[:-1][adjacent])
        if not pairs:
            raise ValueError('no usable reference transitions')
        if roots and len(roots) != len(pairs):
            raise ValueError('reset states must be supplied for all selected reference clips')
        raw, gaits = np.asarray(np.concatenate(pairs), dtype=np.float32), np.concatenate(labels)
        self.pairs = torch.as_tensor(raw, device=device)
        self.labels = torch.as_tensor(gaits, device=device)
        linear = self.feature_spec.section('base_linear_velocity_body').start
        self.speeds = self.pairs[:, linear:linear + 2].norm(dim=1)
        self.groups = tuple(torch.where(self.labels == i)[0] for i in range(2))
        if any(len(group) == 0 for group in self.groups):
            raise ValueError('both walking and running references are required')
        self.root = torch.as_tensor(np.concatenate(roots), device=device) if roots else None
        self.joints = torch.as_tensor(np.concatenate(joints), device=device) if joints else None
        self.mean, self.std = self.pairs.mean(0), self.pairs.std(0).clamp_min(.05)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.index = VelocityReferenceIndex(raw, gaits, feature_spec=self.feature_spec,
                                            device=device, tolerances=tolerances)
        self.sampler_signature = {**self.index.signature, 'dataset': self.digest, 'split': split, 'dt': self.dt}

    def sample(self,count):
        labels=torch.randint(2,(count,),generator=self.generator,device=self.pairs.device)
        indices=torch.zeros_like(labels)
        for gait,group in enumerate(self.groups):
            mask=labels==gait
            indices[mask]=group[torch.randint(len(group),(int(mask.sum()),),generator=self.generator,device=self.pairs.device)]
        return self.pairs[indices],labels

    def sample_matching(self,labels,commands,tolerance=None):
        if tolerance is not None and abs(tolerance-float(self.index.tolerance[0]))>1e-6:
            raise ValueError('set matching tolerances when constructing the dataset')
        ids,supported=self.index.sample(labels,commands,self.generator)
        return self.pairs[ids],supported

    def supported_heads(self,commands):
        commands=self.index.commands(commands)
        return torch.stack([self.index.lookup(torch.full((len(commands),),i,device=commands.device,dtype=torch.long),commands)[1] for i in range(2)],1)


class ReferenceStateInitializer:
    """Reference resets and AMP share identical gait/direction sampling and split rules."""
    def __init__(self,path,*,device,probability=.5,max_speed=None):
        if not 0<=probability<=1:raise ValueError('reference reset probability must be in [0,1]')
        self.probability=probability;self.max_speed=max_speed
        self.dataset=path if isinstance(path,MotionDataset) else MotionDataset(path,device=device)
        if self.dataset.root is None:raise ValueError('reference reset requires physical root and joint states')
        self.root=self.dataset.root;self.joints=self.dataset.joints;self.speed=self.dataset.speeds

    def sample(self,count,generator,commands=None):
        selected=torch.rand(count,device=self.root.device,generator=generator)<self.probability
        if commands is None:
            indices=torch.randint(len(self.root),(count,),device=self.root.device,generator=generator)
        else:
            labels=(commands[:,:2].norm(dim=1)>=1.6).long()
            indices,supported=self.dataset.index.sample(labels,commands,generator)
            selected &= supported
        if self.max_speed is not None:selected &= self.speed[indices]<=self.max_speed
        return selected,self.root[indices],self.joints[indices]
