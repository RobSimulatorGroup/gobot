"""Portable recipes and admission checks for independent tracking training jobs."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

from gobot.rl.supermimic import SuperMimicConfig
from gobot.rl.motion_tracking import MotionRandomizationCfg


@dataclass(frozen=True)
class TrackingExperiment:
    schema_version: int = 1
    name: str = 'g1_tracking_subject_split'
    purpose: str = 'training'
    split: str = 'train'
    num_envs: int = 2048
    seed: int = 42
    hours: float = 12.
    total_gpu_hours: float | None = None
    deterministic: bool = False
    checkpoint_minutes: float = 10.
    stop_reserve_seconds: float = 90.
    video_mode: str = 'queue'
    video_interval_minutes: float = 15.
    algorithm: dict = field(default_factory=lambda: {
        'iterations':12000,'wm_batch':2048,'pi_batch':2048,'horizon':32,
        'wm_only_iters':500,'ckpt_every':250,'log_every':10,
        'amp':'bf16','compile':True})
    randomization: dict = field(default_factory=lambda: {'enabled':True})

    def __post_init__(self):
        if self.schema_version!=1 or self.purpose not in ('training','pipeline_smoke'):
            raise ValueError('unknown tracking experiment version/purpose')
        if self.video_mode not in ('queue','render','off') or not math.isfinite(self.video_interval_minutes) or self.video_interval_minutes<=0:
            raise ValueError('invalid video schedule')
        if self.split not in ('coverage40','train'):
            raise ValueError('training cannot consume validation or holdout subjects')
        if isinstance(self.num_envs,bool) or not isinstance(self.num_envs,int) or self.num_envs<1:
            raise ValueError('num_envs must be a positive integer')
        if not all(math.isfinite(v) and v>0 for v in (self.hours,self.checkpoint_minutes,self.stop_reserve_seconds)):
            raise ValueError('job limits must be finite and positive')
        if self.total_gpu_hours is not None and (not math.isfinite(self.total_gpu_hours) or self.total_gpu_hours<=0):
            raise ValueError('total_gpu_hours must be positive or null for per-job accounting')
        if self.stop_reserve_seconds>=self.hours*3600:
            raise ValueError('checkpoint reserve must leave time for the job')
        cfg=self.learner_config()
        for name in ('iterations','buffer_steps','horizon','goal_frames','hist','wm_window','wm_batches',
                     'wm_batch','pi_batch','critic_epochs','critic_minibatches','warmup_steps','ckpt_every','log_every','lr_decay_iters'):
            value=getattr(cfg,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(f'{name} must be a positive integer')
        if min(cfg.utd,cfg.lr)<=0 or not all(math.isfinite(x) for x in (cfg.utd,cfg.lr,cfg.action_noise)) or cfg.action_noise<0:
            raise ValueError('invalid optimizer or exploration configuration')
        if (not 0<cfg.gamma<1 or not 0<=cfg.lam<=1 or isinstance(cfg.wm_only_iters,bool)
                or not isinstance(cfg.wm_only_iters,int) or cfg.wm_only_iters<0):
            raise ValueError('invalid return or warmup configuration')
        if max(cfg.hist,cfg.wm_window)>cfg.buffer_steps or cfg.critic_minibatches>cfg.horizon*cfg.pi_batch:
            raise ValueError('replay or critic minibatch is too small for its configured windows')
        for widths in (cfg.wm_hidden,cfg.policy_hidden,cfg.critic_hidden,cfg.test_hidden_dims):
            if widths is not None and (not widths or any(isinstance(x,bool) or not isinstance(x,int) or x<1 for x in widths)):
                raise ValueError('network widths must be positive integers')
        reference_architecture=(tuple(cfg.wm_hidden)==(2048,2048,2048) and
                                tuple(cfg.policy_hidden)==(2048,1024,512,256) and tuple(cfg.critic_hidden)==(256,128))
        if self.purpose=='training' and (cfg.test_hidden_dims is not None or
                (cfg.horizon,cfg.hist,cfg.goal_frames,cfg.wm_window,cfg.z_dim)!=(32,10,10,8,32) or not reference_architecture):
            raise ValueError('formal SuperMimic recipes retain full networks and the 32-step horizon')
        if self.purpose=='pipeline_smoke' and cfg.iterations>10:
            raise ValueError('pipeline smoke is limited to 10 updates')
        MotionRandomizationCfg(**self.randomization)

    def learner_config(self):
        return SuperMimicConfig(**(self.algorithm|{'seed':self.seed}))

    @classmethod
    def load(cls,path):
        return cls(**json.loads(Path(path).read_text()))

    def metadata(self):
        return asdict(self)


def require_tracking_admission(report, *, robot_signature, clip_names):
    if report.get('protocol')!='paired_tracking_fidelity_v1' or report.get('passed') is not True:
        raise ValueError('formal training requires a passed paired tracking fidelity report')
    rows=report.get('per_clip',[])
    if {row.get('clip') for row in rows}!=set(clip_names) or len(rows)!=len(clip_names):
        raise ValueError('admission report does not cover the full pinned motion library')
    if any(not isinstance(row.get(key),bool) for row in rows for key in ('reference_complete','candidate_complete')):
        raise ValueError('admission completion flags must be boolean')
    if any(row.get('reference_complete') and not row.get('candidate_complete') for row in rows):
        raise ValueError('admission contains new failed clips')
    delta=sum(float(row['return_delta']) for row in rows)/len(rows)
    if not math.isfinite(delta) or delta < -.02 or report.get('new_failed_clips'):
        raise ValueError('admission fidelity thresholds were not met')
    if report.get('candidate_physics',{}).get('robot_signature')!=robot_signature:
        raise ValueError('admission belongs to a different/unidentified robot physics contract')
