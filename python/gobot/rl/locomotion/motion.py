"""Named robot motion features and validated, clip-bounded reference sampling."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import torch


@dataclass(frozen=True)
class MotionDatasetCfg:
    path: str
    history_dt: float = .02
    holdout_subject: str = "subject5"


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


def motion_features(state, *, device="cpu"):
    """79 physical features for 29 joints and four links; no terrain/world position."""
    def t(key):
        value = state[key]
        if isinstance(value, np.ndarray) and not value.flags.writeable:
            value = value.copy()
        return torch.as_tensor(value, dtype=torch.float32, device=device)
    q = t("base_quaternion")
    gravity = torch.zeros_like(t("base_position")); gravity[..., 2] = -1
    relative = t("link_position") - t("base_position")[:, None]
    relative = quat_inverse_apply(q[:, None], relative).flatten(1)
    return torch.cat((t("joint_position"), t("joint_velocity"), t("base_linear_velocity_body"),
                      t("base_angular_velocity_body"), quat_inverse_apply(q, gravity), relative), dim=-1)


class VelocityReferenceIndex:
    """Precomputed, ragged velocity buckets; GPU sampling has no per-environment search."""
    origin = np.array([-3.2, -1., -1.2], np.float32)
    spacing = np.array([.2, .1, .2], np.float32)
    shape = np.array([33, 21, 13], np.int64)

    def __init__(self, pairs, labels, *, device, tolerances=(.25,.15,.35)):
        from itertools import product
        first=pairs[:,[58,59,63]];last=pairs[:,[137,138,142]]
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
        self.signature={'schema':'directional_motion_v2','tolerances':list(tolerances)}

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
        cfg=path if isinstance(path,MotionDatasetCfg) else MotionDatasetCfg(str(path))
        if cfg.history_dt!=.02:raise ValueError('motion transitions require the 50 Hz control period')
        path=Path(cfg.path);self.digest=hashlib.sha256(path.read_bytes()).hexdigest()
        pairs=[];labels=[];roots=[];joints=[]
        with np.load(path,allow_pickle=False) as data:
            self.metadata=json.loads(str(data['metadata']))
            if self.metadata['schema']!='gobot_motion_v1' or self.metadata['dt']!=.02:
                raise ValueError('unsupported motion schema or control period')
            for i,clip in enumerate(self.metadata['clips']):
                clip_split='validation' if 'subject4' in clip['name'] else clip['split']
                if clip_split!=split:continue
                f=data[f'features_{i}'];valid=data[f'valid_{i}'].astype(bool)
                if f.ndim!=2 or f.shape[1]!=79 or valid.shape!=(len(f),) or not np.isfinite(f[valid]).all():
                    raise ValueError(f'invalid reference clip {clip["name"]}')
                valid=valid[:-1]&valid[1:]
                if not valid.any():continue
                pairs.append(np.concatenate((f[:-1],f[1:]),axis=1)[valid])
                labels.append(np.full(int(valid.sum()),clip['gait'],np.int64))
                if f'root_{i}' in data:
                    roots.append(data[f'root_{i}'][:-1][valid]);joints.append(data[f'joint_{i}'][:-1][valid])
        if not pairs:raise ValueError('no usable reference transitions')
        raw=np.concatenate(pairs);gaits=np.concatenate(labels)
        self.pairs=torch.as_tensor(raw,device=device);self.labels=torch.as_tensor(gaits,device=device)
        self.speeds=self.pairs[:,58:60].norm(dim=1)
        self.groups=tuple(torch.where(self.labels==i)[0] for i in range(2))
        if any(len(group)==0 for group in self.groups):raise ValueError('both walking and running references are required')
        self.root=torch.as_tensor(np.concatenate(roots),device=device) if len(roots)==len(pairs) else None
        self.joints=torch.as_tensor(np.concatenate(joints),device=device) if len(joints)==len(pairs) else None
        self.mean=self.pairs.mean(0);self.std=self.pairs.std(0).clamp_min(.05)
        self.generator=torch.Generator(device=device).manual_seed(seed)
        self.index=VelocityReferenceIndex(raw,gaits,device=device,tolerances=tolerances)
        self.sampler_signature={**self.index.signature,'dataset':self.digest,'split':split,'validation_subject':'subject4','holdout_subject':'subject5'}

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
