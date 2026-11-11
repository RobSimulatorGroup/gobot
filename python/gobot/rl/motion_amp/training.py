"""Gobot's scene environment, RSL-RL PPO/runner/logger, and an AMP extension."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from gobot.rl.policy import POLICY_MANIFEST_KEY, PolicyManifest, scene_bundle_digest, policy_manifest_from_checkpoint
from gobot.rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlModelCfg, rsl_rl_cfg_to_dict
from gobot.rl.training.workflow import TrainingSpec, config_dict
from gobot.rl.training.status import write_json
from gobot.rl.locomotion.logger import _install_cpu_logger_bookkeeping
from gobot.rl.locomotion.training import snapshot_sources
from .algorithm import AMPConfig
from .env import MotionAMPEnv


def default_agent():
    cfg=RslRlOnPolicyRunnerCfg(num_steps_per_env=32,save_interval=100,experiment_name='g1-motion-amp',
        obs_groups={'actor':('actor',),'critic':('actor',)},
        actor=RslRlModelCfg(hidden_dims=(1024,512),activation='relu',obs_normalization=True,
            class_name='gobot.rl.motion_amp.models:ReferenceMLP',
            distribution_cfg={'class_name':'gobot.rl.motion_amp.models:JointRangeGaussian','init_std':.05,'std_type':'scalar'}),
        critic=RslRlModelCfg(hidden_dims=(1024,512),activation='relu',obs_normalization=True,
            class_name='gobot.rl.motion_amp.models:ReferenceMLP'))
    cfg.algorithm.learning_rate=1.e-4
    cfg.algorithm.optimizer='sgd'
    cfg.algorithm.schedule='fixed'
    cfg.algorithm.num_mini_batches=8
    cfg.algorithm.use_clipped_value_loss=False
    cfg.algorithm.max_grad_norm=1.e9
    cfg.algorithm.entropy_coef=0.
    cfg.algorithm.class_name='gobot.rl.motion_amp.algorithm:MotionAMP'
    agent=rsl_rl_cfg_to_dict(cfg)
    agent['actor']['normalization_samples']=100000000
    agent['critic']['normalization_samples']=100000000
    agent['algorithm'].update(optimizer_momentum=.9,advantage_clip=4.)
    return agent


@dataclass
class Config:
    project_path: Path = Path('examples/g1/assets/mimickit/gobot')
    scene_path: str = 'res://world.jscn'
    motion: str = 'walk'
    velocity_reference: str = 'base_body_com'
    backend: str = 'mujoco-warp'
    device: str | None = None
    num_envs: int = 4096
    seed: int = 1
    updates: int = 2000
    output: Path = Path('logs/g1-motion-amp/walk_s1')
    resume: Path | None = None
    agent: dict = field(default_factory=default_agent)
    amp: AMPConfig = field(default_factory=AMPConfig)
    capture_graphs: bool = True
    torch_threads: int = 4

    def validate(self):
        for name in ('updates','torch_threads','num_envs'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<=0:
                raise ValueError(f'{name} must be a positive integer')
        self.amp.validate()
        if self.motion not in ('walk','run'):raise ValueError('motion must be walk or run')
        if self.velocity_reference!='base_body_com':
            raise ValueError('the aligned AMP task requires base_body_com velocity')
        if self.backend=='mujoco-cpu' and self.device not in (None,'cpu'):
            raise ValueError('mujoco-cpu requires --device cpu')
        if self.agent['algorithm']['class_name']!='gobot.rl.motion_amp.algorithm:MotionAMP':
            raise ValueError('this task requires the RSL-RL MotionAMP extension')
        if any(self.agent[key]['class_name']!='gobot.rl.motion_amp.models:ReferenceMLP' for key in ('actor','critic')):
            raise ValueError('aligned AMP requires ReferenceMLP actor and critic')
        if self.agent['actor']['distribution_cfg']['class_name']!='gobot.rl.motion_amp.models:JointRangeGaussian':
            raise ValueError('aligned AMP requires JointRangeGaussian actions')
        if self.agent['algorithm']['optimizer']!='sgd':
            raise ValueError('aligned AMP uses SGD')
        if self.agent.get('logger')!='tensorboard':
            raise ValueError('this task uses local TensorBoard logging')
        for name in ('num_steps_per_env','save_interval'):
            if not isinstance(self.agent[name],int) or self.agent[name]<=0:
                raise ValueError(f'{name} must be a positive integer')
        if self.num_envs*self.agent['num_steps_per_env']<self.agent['algorithm']['num_mini_batches']:
            raise ValueError('rollout is smaller than the PPO minibatch count')
        if not (Path(self.project_path)/'motion_amp/profile.json').is_file():
            raise ValueError('missing prepared AMP assets: copy examples/g1/assets/mimickit/gobot into the project')


def actor_config(env, settings):
    actor=deepcopy(settings)
    low=np.asarray(env.profile['action_low']);high=np.asarray(env.profile['action_high'])
    actor['distribution_cfg'].update(action_scale=((high-low)/2).tolist(),action_offset=((high+low)/2).tolist())
    return actor


def manifest_for(env, model=None):
    p=env.profile
    robot=next(r for r in env.batch.snapshot['robots'] if r['name']==p['robot_name'])
    joints={j['name']:j for j in robot['joints']}
    ordered=[joints[name] for name in p['joint_names']]
    return PolicyManifest(task_name='g1-motion-amp',observation_spec=env.observation_spec.metadata(),
        action_spec=env.action_spec.metadata(),joint_names=p['joint_names'],physics_dt=p['physics_dt'],
        decimation=p['decimation'],control={'mode':'absolute_position',
            'kp':[j['drive_stiffness'] for j in ordered],'kd':[j['drive_damping'] for j in ordered],
            'effort_limit':[j['effort_limit'] for j in ordered],'clip_to_action_bounds':True},
        model=model or {**actor_config(env,default_agent()['actor']),'obs_normalizer_eps':0.,'obs_normalizer_clip':10.},
        scene_path=env.scene_path,scene_digest=scene_bundle_digest(env.project_path,env.scene_path),
        extras={'motion':env.motion_name,'motion_sha256':p['files']['motions.npz'],
            'history_steps':env.history_steps,'discriminator_spec':'motion_amp_global_com_history_v2',
            'root_linear_velocity_reference':env.velocity_reference,
            'base_center_of_mass':env.features.base_center_of_mass.cpu().tolist(),
            'source_commit':p['source_commit'],'solver_settings':p['solver_settings'],
            'termination':'non_allowed_link_static_world_contact_0.1N','episode_length_s':p['episode_length_s']})


def validate_manifest(payload,env):
    saved=policy_manifest_from_checkpoint(payload)
    if saved is None:raise ValueError('AMP checkpoint is missing its policy manifest')
    expected=manifest_for(env,model=saved.model).metadata();actual=saved.metadata()
    expected['scene'].pop('path');actual['scene'].pop('path')
    if expected!=actual:raise ValueError('AMP policy manifest does not match the environment')
    return saved


class Runner(OnPolicyRunner):
    """Add Gobot provenance/checkpoint validation to RSL-RL's existing runner."""
    def __init__(self,env,config,output,device,manifest,signature):
        self.manifest,self.signature=manifest,signature
        super().__init__(env,config,str(output),device)
        self.alg.demo_sampler=env.sample_demonstrations
        _install_cpu_logger_bookkeeping(self.logger)

    def save(self,path,infos=None):
        info=dict(infos or {})
        info.update({POLICY_MANIFEST_KEY:self.manifest.metadata(), 'training_signature':self.signature,
            'resume_semantics':'optimizer/discriminator/replay restored; fresh physics episodes',
            'torch_rng':torch.get_rng_state(), 'motion_rng':self.env.generator.get_state(),
            'cuda_rng':torch.cuda.get_rng_state(self.device) if self.device.startswith('cuda') else None})
        target=Path(path);temporary=target.with_suffix('.tmp')
        super().save(str(temporary),info)
        temporary.replace(target)

    def load(self,path,**kwargs):
        payload=torch.load(path,map_location=self.device,weights_only=True)
        validate_manifest(payload,self.env.env)
        if payload['infos'].get('training_signature')!=self.signature:
            raise ValueError('resume requires the same task, PPO, AMP, backend and batch configuration')
        result=super().load(path,map_location=self.device,**kwargs)
        self.current_learning_iteration+=1
        torch.set_rng_state(result['torch_rng'].cpu())
        self.env.generator.set_state(result['motion_rng'].cpu())
        if result['cuda_rng'] is not None:torch.cuda.set_rng_state(result['cuda_rng'].cpu(),self.device)
        self.env.reset()
        return result


def run(config):
    config.validate()
    torch.set_num_threads(config.torch_threads)
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    output=Path(config.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',config_dict(config))
    env=None;started=time.time()
    try:
        env=MotionAMPEnv(config.project_path,scene_path=config.scene_path,motion=config.motion,
            num_envs=config.num_envs,backend=config.backend,device=config.device,seed=config.seed,
            capture_graphs=config.capture_graphs,autoreset=True,velocity_reference=config.velocity_reference,
            diagnostics_path=output/'numerical_failures')
        if config.amp.observation_dim!=env.features.frame_dim*env.history_steps:
            raise ValueError('AMP observation size does not match the task')
        agent=deepcopy(config.agent);agent['seed']=config.seed
        agent['actor']=actor_config(env,agent['actor'])
        agent['algorithm']['amp_cfg']=asdict(config.amp)
        model={**agent['actor'],'obs_normalizer_eps':0.,'obs_normalizer_clip':10.}
        manifest=manifest_for(env,model)
        semantic=config_dict(config)
        for name in ('updates','output','resume','project_path'):semantic.pop(name)
        # COM-v2's saved signature predates this explicit (now fixed) field.
        semantic.pop('velocity_reference')
        signature=hashlib.sha256(json.dumps({'config':semantic,'scene':manifest.scene_digest},sort_keys=True).encode()).hexdigest()
        source_digest=snapshot_sources(output,env.project_path)
        write_json(output/'run.json',{'algorithm':'rsl_rl.PPO + Gobot AMP','source_digest':source_digest,
            'training_signature':signature,'manifest':manifest.metadata(),'agent':agent})
        runner=Runner(RslRlVecEnvWrapper(env,device=env.device),agent,output,env.device,manifest,signature)
        if config.resume is not None:runner.load(config.resume)
        initial=next(runner.alg.actor.mlp.parameters()).detach().clone()
        write_json(output/'status.json',{'state':'training','algorithm':'rsl_rl.PPO + Gobot AMP'})
        runner.learn(config.updates,init_at_random_ep_len=False)
        runner.save(str(output/'model.pt'))
        difference=float((next(runner.alg.actor.mlp.parameters()).detach()-initial).abs().max())
        result={'state':'trained','updates':config.updates,'last_iteration':runner.current_learning_iteration,
            'samples':config.updates*config.num_envs*agent['num_steps_per_env'],
            'elapsed_seconds':time.time()-started,'actor_max_weight_change':difference,
            'checkpoint':str(output/'model.pt'),
            'checkpoint_sha256':hashlib.sha256((output/'model.pt').read_bytes()).hexdigest(),
            'algorithm':'rsl_rl.PPO + Gobot AMP','exact_physics_resume':False}
        write_json(output/'status.json',result)
        return result
    except BaseException as error:
        write_json(output/'status.json',{'state':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
            'error':str(error),'elapsed_seconds':time.time()-started})
        raise
    finally:
        if env is not None:env.close()


def training_spec():
    return TrainingSpec(Config,run,description='G1 motion imitation with RSL-RL PPO and Gobot AMP')
