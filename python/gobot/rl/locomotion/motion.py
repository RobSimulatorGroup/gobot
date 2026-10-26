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
    updates_per_iteration: int = 8
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


class MotionDataset:
    def __init__(self, path, *, device="cpu", seed=42, split="train"):
        cfg = path if isinstance(path, MotionDatasetCfg) else MotionDatasetCfg(str(path))
        if cfg.history_dt != .02:
            raise ValueError('motion transitions must use the 50 Hz control period')
        path = Path(cfg.path)
        self.digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata"]))
            if self.metadata["schema"] != "gobot_motion_v1" or self.metadata["dt"] != .02:
                raise ValueError("unsupported motion feature schema or control period")
            pairs, labels = [], []
            for i, clip in enumerate(self.metadata["clips"]):
                if clip["split"] != split:
                    continue
                values = data[f"features_{i}"]
                valid = data[f"valid_{i}"].astype(bool)
                if values.ndim != 2 or values.shape[1] != 79 or valid.shape != (len(values),):
                    raise ValueError(f"invalid feature shape in motion clip {clip['name']!r}")
                if not np.isfinite(values[valid]).all():
                    raise ValueError(f"non-finite valid features in motion clip {clip['name']!r}")
                valid = valid[:-1] & valid[1:]
                pair = np.concatenate((values[:-1], values[1:]), axis=1)[valid]
                if not len(pair):
                    continue
                pairs.append(pair)
                labels.append(np.full(len(pair), clip["gait"], np.int64))
        if not pairs:
            raise ValueError(f"motion dataset contains no usable {split} transitions")
        self.pairs = torch.as_tensor(np.concatenate(pairs), device=device)
        self.labels = torch.as_tensor(np.concatenate(labels), device=device)
        self.groups = tuple(torch.where(self.labels == i)[0] for i in range(2))
        if any(len(group) == 0 for group in self.groups):
            raise ValueError("both walking and running reference transitions are required")
        self.mean = self.pairs.mean(0)
        self.std = self.pairs.std(0).clamp_min(.05)
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def sample(self, count):
        labels = torch.randint(2, (count,), generator=self.generator, device=self.pairs.device)
        indices = torch.empty_like(labels)
        for gait, group in enumerate(self.groups):
            mask = labels == gait
            indices[mask] = group[torch.randint(len(group), (int(mask.sum()),), generator=self.generator, device=self.pairs.device)]
        return self.pairs[indices], labels


class ReferenceStateInitializer:
    """Physical reference states for flat-ground reset; uses the environment RNG."""
    def __init__(self, path, *, device, probability=.5, max_speed=1.2):
        if not 0 <= probability <= 1:
            raise ValueError('reference reset probability must be in [0, 1]')
        self.probability = probability
        roots, joints = [], []
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            for i, clip in enumerate(metadata['clips']):
                if clip['split'] == 'train':
                    valid = data[f'valid_{i}'].astype(bool)
                    valid &= np.linalg.norm(data[f'root_{i}'][:, 7:9], axis=1) <= max_speed
                    roots.append(data[f'root_{i}'][valid])
                    joints.append(data[f'joint_{i}'][valid])
        if not roots or not sum(len(root) for root in roots):
            raise ValueError("no valid reference initial states at the requested speed")
        self.root = torch.as_tensor(np.concatenate(roots), device=device)
        self.joints = torch.as_tensor(np.concatenate(joints), device=device)

    def sample(self, count, generator):
        selected = torch.rand(count, device=self.root.device, generator=generator) < self.probability
        indices = torch.randint(len(self.root), (count,), device=self.root.device, generator=generator)
        return selected, self.root[indices], self.joints[indices]
