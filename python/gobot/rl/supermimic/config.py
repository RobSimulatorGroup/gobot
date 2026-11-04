# Adapted from CuboiLeo/SuperMimic-Reproduced, ba6105349069bf040f1fdad60873d13138d505a2.
# Copyright (c) 2026 Yuxi Liu. SPDX-License-Identifier: MIT
# See LICENSE-SuperMimic-Reproduced.
"""All training configuration in one dataclass tree (tyro-friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math

from gobot.rl.supermimic.policy_trainer import PolicyTrainerCfg
from gobot.rl.supermimic.policy import CriticCfg, PolicyCfg
from gobot.rl.supermimic.wm_trainer import WMTrainerCfg
from gobot.rl.supermimic.world_model import WorldModelCfg


@dataclass
class DataCfg:
    npz_dir: Path = Path("data/npz/lafan1")
    clips: list[str] | None = None  # None = all npz in npz_dir
    limit: int | None = None
    manifest: Path | None = None  # csv with name,keep (filter_bones_seed.py output)
    library: Path | None = None  # packed compact library (.pt from build_library.py); overrides npz_dir


@dataclass
class EnvCfg:
    num_envs: int = 4096
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    sampling_mode: str = "return"
    replay_capacity: int = 1000
    future_frames: int = 10
    compact_library: bool = False
    anchor_obs: bool = False


@dataclass
class ScheduleCfg:
    total_iterations: int = 320_000  # env steps (plan D12 reading A)
    updates_per_env_step: float = 0.4  # UTD
    policy_updates_per_wm_update: int = 1
    warmup_iterations: int = 40  # env steps before the first update (fills k-step windows)
    lr_final_ratio: float = 0.03
    lr_decay_fraction: float = 250_000 / 320_000
    log_interval: int = 50  # iterations
    checkpoint_interval: int = 4000  # iterations
    max_hours: float | None = None


@dataclass
class TrainConfig:
    run_name: str = "debug"
    log_root: Path = Path("logs")
    seed: int = 0
    device: str = "cuda:0"
    data: DataCfg = field(default_factory=DataCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    resume: Path | None = None
    project: Path = Path("examples/g1")
    scene: str = "res://g1_29dof_flat_scene.jscn"
    backend: str = "mujoco-warp"
    capture_graphs: bool = True
    deterministic: bool = False
    nconmax: int = 35
    njmax: int = 250
    stop_iteration: int | None = None

    def validate(self):
        counts = {
            'num_envs':self.env.num_envs, 'replay_capacity':self.env.replay_capacity,
            'future_frames':self.env.future_frames, 'history':self.policy.history,
            'horizon':self.policy_trainer.horizon, 'wm_batch':self.wm_trainer.batch_size,
            'wm_batches':self.wm_trainer.batches_per_update, 'wm_window':self.wm_trainer.window,
            'policy_batch':self.policy_trainer.batch_size, 'critic_epochs':self.policy_trainer.critic_epochs,
            'critic_minibatches':self.policy_trainer.critic_minibatches,
            'total_iterations':self.schedule.total_iterations, 'warmup_iterations':self.schedule.warmup_iterations,
            'log_interval':self.schedule.log_interval, 'checkpoint_interval':self.schedule.checkpoint_interval,
            'policy_updates_per_wm_update':self.schedule.policy_updates_per_wm_update,
            'nconmax':self.nconmax, 'njmax':self.njmax,
        }
        if self.stop_iteration is not None:
            counts['stop_iteration']=self.stop_iteration
        for name,value in counts.items():
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(f'{name} must be a positive integer')
        for name,value in (('episode_length_s',self.env.episode_length_s),
                           ('updates_per_env_step',self.schedule.updates_per_env_step),
                           ('wm_lr',self.wm_trainer.lr),('policy_lr',self.policy_trainer.lr),
                           ('critic_lr',self.policy_trainer.critic_lr)):
            if not math.isfinite(value) or value<=0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.env.action_noise_std) or self.env.action_noise_std<0:
            raise ValueError('action_noise_std must be finite and nonnegative')
        if self.env.sampling_mode not in ('return','uniform','start'):
            raise ValueError('unknown motion sampling mode')
        if self.env.replay_capacity<=max(self.policy.history+2,self.wm_trainer.window+1):
            raise ValueError('replay capacity must exceed history and WM window')
        if self.schedule.warmup_iterations<=self.wm_trainer.window:
            raise ValueError('warmup must fill a complete WM window')
        if self.policy_trainer.horizon*self.policy_trainer.batch_size<self.policy_trainer.critic_minibatches:
            raise ValueError('critic minibatches exceed rollout transitions')
        if not math.isclose(self.world_model.state_spec.dt,.02):
            raise ValueError('G1 tracking requires 50 Hz world-model integration')

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key, typ in (("data", DataCfg), ("env", EnvCfg), ("world_model", WorldModelCfg),
                         ("wm_trainer", WMTrainerCfg), ("policy", PolicyCfg), ("critic", CriticCfg),
                         ("policy_trainer", PolicyTrainerCfg), ("schedule", ScheduleCfg)):
            if key in value:
                data = dict(value[key])
                if key == "world_model" and isinstance(data.get("state_spec"), dict):
                    from .state_spec import StateSpecCfg
                    data["state_spec"] = StateSpecCfg(**data["state_spec"])
                value[key] = typ(**data)
        return cls(**value)
