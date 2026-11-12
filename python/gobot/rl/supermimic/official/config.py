"""Official GenCoord configuration on Gobot scenes; single-process training."""
from dataclasses import dataclass, field
import math
from pathlib import Path

from .policy_learner import ActorCriticPolicyLearnerCfg
from .sampler import ReturnSamplerCfg
from .wm_learner import WorldModelLearnerCfg


@dataclass
class Config:
    project: Path = Path('examples/g1/assets/official')
    scene: str = 'res://g1_29dof_flat_scene.jscn'
    library: Path = Path('data/bones_official_50hz')
    backend: str = 'mujoco-warp'
    device: str | None = None
    seed: int = 0
    num_envs: int = 4096
    steps: int = 100000
    max_seconds: float | None = None
    output: Path = Path('logs/g1-supermimic-official/seed0')
    resume: Path | None = None
    buffer_size: int = 1000
    replay_ratio: float = 16.
    history_length: int = 1
    actor_obs: str = 'rep_state'
    sampling_mode: str = 'return'
    termination_softness: float = .05
    dynamics_hidden_dims: tuple[int,...] = (2048,2048,2048)
    actor_hidden_dims: tuple[int,...] = (2048,1024,512,256)
    critic_hidden_dims: tuple[int,...] = (256,128)
    latent_dim: int = 32
    latent_bc_weight: float = .1
    wm: WorldModelLearnerCfg = field(default_factory=WorldModelLearnerCfg)
    policy: ActorCriticPolicyLearnerCfg = field(default_factory=ActorCriticPolicyLearnerCfg)
    sampler: ReturnSamplerCfg = field(default_factory=ReturnSamplerCfg)
    log_interval: int = 10
    save_interval: int = 1000
    compile: bool = True
    capture_graphs: bool = True
    capture_updates: bool = True
    verify_data: bool = True
    domain_randomization: bool = True
    reset_noise: bool = True
    episode_length_s: float = 10.
    nconmax: int = 35
    njmax: int = 500
    torch_threads: int = 4

    def validate(self):
        if isinstance(self.seed,bool) or not isinstance(self.seed,int) or self.seed<0:
            raise ValueError('seed must be a nonnegative integer')
        if not 0<self.policy.gamma<1 or not 0<=self.policy.lam<=1:
            raise ValueError('actor-critic gamma must be in (0,1), lambda in [0,1]')
        if not math.isfinite(self.policy.noise) or self.policy.noise<0:
            raise ValueError('policy noise must be finite and nonnegative')
        for name in ('nominal_error_root_pose_pos','nominal_error_root_pose_ori','nominal_error_joint_pos','nominal_error_joint_vel'):
            if not math.isfinite(getattr(self.wm,name)) or getattr(self.wm,name)<=0:
                raise ValueError('world-model loss scales must be finite and positive')
        for name in ('num_envs','steps','buffer_size','history_length','latent_dim','log_interval','save_interval','nconmax','njmax','torch_threads'):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,int) or v<1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('replay_ratio','episode_length_s','termination_softness'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(f'{name} must be finite and positive')
        if self.max_seconds is not None and (not math.isfinite(self.max_seconds) or self.max_seconds<=0):
            raise ValueError('max_seconds must be finite and positive')
        if self.actor_obs not in ('rep_state','proprio') or (self.history_length>1 and self.actor_obs!='proprio'):
            raise ValueError('history_length > 1 requires proprio actor observations')
        if self.sampling_mode not in ('return','uniform','start'):
            raise ValueError('sampling_mode must be return, uniform or start')
        if self.wm.horizon<2 or self.buffer_size<max(self.wm.horizon,self.history_length):
            raise ValueError('replay must fit the WM window/history, and WM horizon must be >=2')
        for cfg in (self.wm,self.policy):
            for name in ('num_batches','batch_size','horizon'):
                v=getattr(cfg,name)
                if isinstance(v,bool) or not isinstance(v,int) or v<1:
                    raise ValueError(f'learner {name} must be a positive integer')
            if not math.isfinite(cfg.lr) or cfg.lr<=0:
                raise ValueError('learner lr must be finite and positive')
        if self.policy.batch_size*self.policy.horizon<4:
            raise ValueError('critic needs at least four rollout transitions')
        if self.backend=='mujoco-cpu' and self.device not in (None,'cpu'):
            raise ValueError('mujoco-cpu requires cpu')
        for dims in (self.dynamics_hidden_dims,self.actor_hidden_dims,self.critic_hidden_dims):
            if not dims or any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in dims):
                raise ValueError('network dimensions must be positive integers')
