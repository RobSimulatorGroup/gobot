"""Official motion-only GenCoord learner driven by Gobot scene physics.

The official Ratio clock is preserved. Checkpoints add optimizer/RNG/context
state; resume starts fresh episodes and refills replay, not a bitwise trajectory.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from tensordict import TensorDict

from gobot.rl.training.workflow import TrainingSpec, config_dict
from .config import Config
from .distributed import Distributed, resolve_config
from .logging import MetricsWriter, Progress, write_json
from .env import TrackingEnv
from .imagination import WorldModelEnv
from .models import Dynamics, MultiEncoderAE, build_mlp
from .motion import digest
from .policy_learner import ActorCriticPolicyLearner
from .replay import ReplayBuffer, FieldSpec
from .state import GenCoordRepresentation
from .tasks import EncoderSpec
from .wm_learner import WorldModelLearner

FORMAT='gobot_supermimic_official_v1'


class Ratio:
    """Pinned runner's fractional update accumulator, indexed by control step."""
    def __init__(self,ratio):
        self.ratio=ratio;self.previous=None

    def __call__(self,step):
        if self.previous is None:
            self.previous=int(step)
            return 1
        repeats=int((int(step)-self.previous)*self.ratio)
        self.previous+=repeats/self.ratio
        return repeats


def source_identity():
    root=Path(__file__).parent
    files={p.name:digest(p) for p in sorted(root.glob('*.py'))}
    # Scene compilation and kinematics also define this training experiment.
    from gobot.rl import articulation,kinematics
    for module in (articulation,kinematics):files[module.__name__]=digest(module.__file__)
    return files


class Trainer:
    def __init__(self,cfg,env,distributed=None,progress=None):
        self.cfg,self.env=cfg,env
        self.distributed=distributed or Distributed(env.device)
        self.progress=progress
        self.writer=None
        self.output=Path(cfg.output)
        if progress is None:
            self.distributed.primary_call(lambda: self.output.mkdir(parents=True,exist_ok=False))
        self.sources=source_identity()
        self.distributed.primary_call(lambda: write_json(self.output/'config.json',config_dict(cfg)))
        self.representation=GenCoordRepresentation(29,env.default_q,dt=env.step_dt,
            decimation=env.batch.decimation,actor_obs_mode=cfg.actor_obs).to(env.device)
        self.representation.bind_kinematics(env.fk)
        self.dynamics=Dynamics(self.representation,29,33,env.action_scale,env.default_q,cfg.dynamics_hidden_dims).to(env.device)
        self.wm=WorldModelLearner(self.dynamics,self.representation,cfg.wm,env.device,compile=cfg.compile)
        obs_dim=68 if cfg.actor_obs=='rep_state' else 93
        # Match official construction, including the initially inactive velocity encoder.
        specs=[EncoderSpec('motion',640),EncoderSpec('velocity',3)]
        self.actor=MultiEncoderAE(obs_dim,29,specs,{'motion':1.,'velocity':0.},cfg.latent_dim,
                                 cfg.actor_hidden_dims,cfg.latent_bc_weight,cfg.history_length).to(env.device)
        self.critic=build_mlp(68+670+33,cfg.critic_hidden_dims,1).to(env.device)
        for module in (self.dynamics,self.actor,self.critic):
            self.distributed.broadcast(module)
        schema={'state':FieldSpec(71),'action':FieldSpec(29),'prev_action':FieldSpec(29),
                'motion_time':FieldSpec(dtype=torch.long),'terminated':FieldSpec(dtype=torch.bool),
                'truncated':FieldSpec(dtype=torch.bool),'sim_ctx':FieldSpec(33)}
        if cfg.actor_obs=='proprio' and cfg.history_length>1:schema['actor_obs']=FieldSpec(93)
        self.buffer=ReplayBuffer(schema,cfg.buffer_size,cfg.num_envs,env.device)
        self.imagined=WorldModelEnv(self.dynamics,self.representation,env.motion,env.soft_limits,
            cfg.policy.batch_size,self.buffer,cfg.history_length,cfg.termination_softness,env.device,
            env.profile['height_threshold'],env.profile['tilt_threshold'])
        if cfg.compile and env.device.startswith('cuda'):
            self.imagined.step=torch.compile(self.imagined.step)
            self.imagined.reset=torch.compile(self.imagined.reset)
        self.policy=ActorCriticPolicyLearner(self.actor,self.critic,self.imagined,cfg.policy,env.device,
                                            compile=cfg.compile,capture=cfg.capture_updates)
        self.wm._captured_update.enabled=cfg.capture_updates
        self.ratio=Ratio(cfg.num_envs/cfg.wm.num_batches/cfg.wm.batch_size/cfg.wm.horizon*cfg.replay_ratio)
        self.control_steps=0;self.learner_updates=0;self.skipped_updates=0
        self.elapsed_seconds=0.;self.stop_requested=False
        self.history=None
        semantic=config_dict(cfg)
        for name in ('output','resume','initialize_from','steps','max_seconds','log_interval','save_interval','verify_data','project','library','torch_threads','tensorboard','heartbeat_seconds'):
            semantic.pop(name,None)
        semantic['device']=torch.device(env.device).type
        identity={'config':semantic,'world_size':self.distributed.world_size,'source':self.sources,'motion':env.motion.identity,
                  'scene':env.batch.artifact.content_digest,'profile':env.profile_digest}
        self.signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        self.identity=identity
        self.initialization=None
        if cfg.resume:self.load(cfg.resume)
        if cfg.initialize_from:self.initialize(cfg.initialize_from)
        metadata={'format':FORMAT,'identity':identity,'signature':self.signature,
            'world_size':self.distributed.world_size,'rank_seed_rule':'seed + rank',
            'batch_scope':cfg.batch_scope,'local_num_envs':cfg.num_envs,
            'global_num_envs':cfg.num_envs*self.distributed.world_size,
            'global_wm_batch_size':cfg.wm.batch_size*self.distributed.world_size,
            'global_policy_batch_size':cfg.policy.batch_size*self.distributed.world_size,
            'learning_rate_scaling':'none','initialization':self.initialization,
            'resume_semantics':'same world size and code; fresh episodes, replay refill; per-rank RNG/context, shared optimizer/sampler restored',
            'wall_clock':'slowest rank training loop including compilation, logging and periodic saves; excludes environment/data construction and final save',
            'software':{'torch':str(torch.__version__)},'device':env.device,
            'update_ratio':self.ratio.ratio}
        self.distributed.primary_call(lambda: write_json(self.output/'run.json',metadata))
        def create_writer():
            self.writer=MetricsWriter(self.output,cfg,config_dict(cfg),self.distributed.world_size)
        self.distributed.primary_call(create_writer)

    def counters(self):
        world=self.distributed.world_size
        return {'control_steps':self.control_steps,
                'env_transitions':self.control_steps*self.cfg.num_envs*world,
                'local_env_transitions':self.control_steps*self.cfg.num_envs,
                'world_size':world,'local_num_envs':self.cfg.num_envs,
                'learner_updates':self.learner_updates,'skipped_updates':self.skipped_updates,
                'world_model_optimizer_steps':self.learner_updates*self.cfg.wm.num_batches,
                'actor_optimizer_steps':self.learner_updates*self.cfg.policy.num_batches,
                'critic_optimizer_steps':self.learner_updates*self.cfg.policy.num_batches*20}

    def phase(self, name):
        if self.progress is not None:
            self.progress.update(state='training',phase=name,**self.counters(),elapsed_seconds=self.elapsed_seconds)

    def save(self,name='latest.pt'):
        self.phase('checkpoint')
        rng_np=np.random.get_state()
        local={'rank':self.distributed.rank,'context':self.env.context.detach().cpu(),
            'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
            'numpy_rng':[rng_np[0],rng_np[1].tolist(),rng_np[2],rng_np[3],rng_np[4]],
            'cuda_rng':torch.cuda.get_rng_state(self.env.device) if self.env.device.startswith('cuda') else None}
        ranks=self.distributed.gather(local)
        def publish():
            sampler=self.env.sampler
            sampler_state=None if sampler is None else {
                key:getattr(sampler,key).detach().cpu().clone() for key in ('value','count','effort')}
            payload={'format':FORMAT,'checkpoint_version':2,'world_size':self.distributed.world_size,
                'config':config_dict(self.cfg),'signature':self.signature,'identity':self.identity,
                'actor':self.actor.state_dict(),'critic':self.critic.state_dict(),'dynamics':self.dynamics.state_dict(),
                'wm_optimizers':[o.state_dict() for o in self.wm.optimizers],
                'actor_optimizers':[o.state_dict() for o in self.policy.actor_optimizers],
                'critic_optimizers':[o.state_dict() for o in self.policy.critic_optimizers],
                'counters':self.counters(),'elapsed_seconds':self.elapsed_seconds,'ratio_previous':self.ratio.previous,
                'rank_states':ranks,'sampler':sampler_state,'initialization':self.initialization,
                'distribution':self.env.distribution.detach().cpu()}
            # Retain the single-rank context field for existing analysis clients.
            if self.distributed.world_size==1:payload['context']=ranks[0]['context']
            destination=self.output/name;temporary=destination.with_suffix('.tmp')
            torch.save(payload,temporary);temporary.replace(destination)
            return str(destination)
        return Path(self.distributed.primary_call(publish))

    def initialize(self,path):
        """Explicit weight transfer, including normalizers; never pretend to resume."""
        ck=torch.load(path,map_location='cpu',weights_only=True)
        if ck.get('format')!=FORMAT:
            raise ValueError('initialize_from requires an official Gobot checkpoint, not Reproduced')
        for key in ('scene','profile','motion'):
            if ck['identity'][key]!=self.identity[key]:
                raise ValueError(f'initialize_from {key} differs from this run')
        for key in ('actor_obs','history_length','latent_dim','latent_bc_weight',
                    'dynamics_hidden_dims','actor_hidden_dims','critic_hidden_dims'):
            if ck['config'][key]!=config_dict(getattr(self.cfg,key)):
                raise ValueError(f'initialize_from architecture differs: {key}')
        for key,module in (('actor',self.actor),('critic',self.critic),('dynamics',self.dynamics)):
            module.load_state_dict(ck[key])
        self.initialization={'path':str(path),'sha256':digest(path),'source_signature':ck['signature'],
            'semantics':'weights and normalizers only; new optimizers, counters, replay, RNG and sampler'}

    def load(self,path):
        ck=torch.load(path,map_location='cpu',weights_only=True)
        if ck.get('format')!=FORMAT or ck.get('signature')!=self.signature:
            raise ValueError('resume requires matching official code, world size, scene, data, device and training configuration; use initialize_from explicitly for weight transfer')
        if ck.get('world_size')!=self.distributed.world_size or len(ck.get('rank_states',[]))!=self.distributed.world_size:
            raise ValueError('resume requires all checkpoint rank states and the same world size')
        for key,module in (('actor',self.actor),('critic',self.critic),('dynamics',self.dynamics)):
            module.load_state_dict(ck[key])
        for key,optimizers in (('wm_optimizers',self.wm.optimizers),('actor_optimizers',self.policy.actor_optimizers),('critic_optimizers',self.policy.critic_optimizers)):
            for optimizer,state in zip(optimizers,ck[key],strict=True):
                optimizer.load_state_dict(state)
                # Optimizer loading moves state tensors, but leaves group LR tensors on CPU.
                for group in optimizer.param_groups:
                    if torch.is_tensor(group['lr']):group['lr']=group['lr'].to(self.env.device)
        self.control_steps=ck['counters']['control_steps'];self.learner_updates=ck['counters']['learner_updates']
        self.skipped_updates=ck['counters'].get('skipped_updates',0)
        self.elapsed_seconds=ck['elapsed_seconds'];self.ratio.previous=ck['ratio_previous']
        self.initialization=ck.get('initialization')
        local=ck['rank_states'][self.distributed.rank]
        if local['rank']!=self.distributed.rank:raise ValueError('checkpoint rank ordering is invalid')
        self.env.context.copy_(local['context']);self.env.apply_context()
        self.env.distribution.copy_(ck['distribution'])
        if self.env.sampler is not None:
            for key in ('value','count','effort'):
                getattr(self.env.sampler,key).copy_(ck['sampler'][key])
            self.env.sampler._step=0
        random.setstate(local['python_rng'])
        nr=local['numpy_rng'];np.random.set_state((nr[0],np.array(nr[1],dtype=np.uint32),nr[2],nr[3],nr[4]))
        torch.set_rng_state(local['torch_rng'])
        if local['cuda_rng'] is not None:torch.cuda.set_rng_state(local['cuda_rng'],self.env.device)

    @torch.no_grad()
    def collect(self,state):
        env,cfg=self.env,self.cfg
        rep=self.representation.represent(state)
        context=env.normalized_context().clone();frame=env.frame.clone()
        sensor=env.sensor(state) if cfg.actor_obs=='proprio' else None
        current=self.representation.compute_actor_sensor(rep,context,sim_actor_obs=sensor,noisy=cfg.actor_obs=='proprio')
        history=current[:,None] if cfg.history_length==1 else self.history
        _,goal=self.representation.compute_wm_goal(env.motion,frame,state[:,:7])
        actor_obs=TensorDict({'history':history,'goal_motion':goal},batch_size=[cfg.num_envs],device=env.device)
        actions=self.actor(actor_obs)+torch.randn_like(env.action)*cfg.policy.noise
        previous=env.action.clone()
        nxt,reward,terminated,truncated,info=env.step(actions)
        fields={'state':state,'action':actions,'prev_action':previous,'motion_time':frame,
                'terminated':terminated&~truncated,'truncated':truncated,'sim_ctx':context}
        if sensor is not None and cfg.history_length>1:fields['actor_obs']=sensor
        self.buffer.insert(TensorDict(fields,batch_size=[cfg.num_envs]))
        if cfg.history_length>1:
            new=self.representation.add_proprio_noise(env.sensor(nxt))
            shifted=torch.cat((self.history[:,1:],new[:,None]),1)
            self.history=torch.where((terminated|truncated)[:,None,None],new[:,None],shifted)
        return nxt,reward,terminated|truncated,actor_obs,actions

    def run(self):
        cfg,env,distributed=self.cfg,self.env,self.distributed
        self.phase('reset')
        state=env.reset()
        if cfg.history_length>1:
            self.history=self.representation.add_proprio_noise(env.sensor(state))[:,None].repeat(1,cfg.history_length,1)
        episode_reward=torch.zeros(cfg.num_envs,device=env.device)
        episode_length=torch.zeros_like(episode_reward)
        base_elapsed=self.elapsed_seconds;started=time.monotonic()
        interval_started=started;interval_step=self.control_steps;interval_updates=self.learner_updates
        metrics={};completed_return=[];completed_length=[]
        collect_seconds=0.;learn_seconds=0.;stop_reason='steps'
        try:
            while self.control_steps<cfg.steps:
                tick=time.monotonic()
                # Publish phases only around initial compilation and log intervals;
                # the heartbeat keeps status alive during long kernels/compilation.
                report_phase=self.learner_updates<5 or self.control_steps%cfg.log_interval==0
                if report_phase:self.phase('collect')
                current=state
                state,reward,done,actor_obs,actions=self.collect(current)
                self.wm.update_normalization(current,actions)
                self.actor.train();self.actor.update_normalization(actor_obs)
                episode_reward+=reward;episode_length+=1
                if done.any():
                    completed_return.extend((episode_reward[done]/(.1*episode_length[done])).tolist())
                    completed_length.extend(episode_length[done].tolist())
                    episode_reward[done]=0.;episode_length[done]=0.
                collect_seconds+=time.monotonic()-tick
                repeats=self.ratio(self.control_steps)
                self.control_steps+=1
                tick=time.monotonic()
                for _ in range(repeats):
                    data=None
                    if self.buffer.num_transitions>=cfg.wm.horizon:
                        try:
                            data=[self.buffer.sample(cfg.wm.horizon,cfg.wm.batch_size) for _ in range(cfg.wm.num_batches)]
                        except RuntimeError as error:
                            if 'no valid sequences' not in str(error):raise
                    # All ranks skip together: no peer may enter a gradient collective alone.
                    if not distributed.all_ready(data is not None):
                        self.skipped_updates+=1
                        continue
                    if report_phase:self.phase('world_model_compile_or_update')
                    wm_stats=self.wm.update(data)
                    if report_phase:self.phase('policy_compile_or_update')
                    policy_stats=self.policy.update()
                    metrics={key:float(value.detach()) for key,value in (wm_stats|policy_stats).items()}
                    if distributed.any(not all(np.isfinite(value) for value in metrics.values())):
                        raise FloatingPointError('nonfinite official learning metrics on at least one rank')
                    self.learner_updates+=1
                learn_seconds+=time.monotonic()-tick
                self.elapsed_seconds=base_elapsed+time.monotonic()-started
                if self.control_steps%cfg.log_interval==0 or self.control_steps==cfg.steps:
                    elapsed=distributed.maximum(self.elapsed_seconds)
                    interval=distributed.maximum(time.monotonic()-interval_started)
                    keys=sorted(metrics)
                    means=distributed.sum([metrics[key] for key in keys])
                    totals=distributed.sum([sum(completed_return),sum(completed_length),len(completed_return)])
                    row={**self.counters(),'elapsed_seconds':elapsed,
                        **{key:value/distributed.world_size for key,value in zip(keys,means,strict=True)},
                        'Perf/global_transitions_per_second':(self.control_steps-interval_step)*cfg.num_envs*distributed.world_size/interval,
                        'Perf/updates_per_second':(self.learner_updates-interval_updates)/interval,
                        'Perf/interval_seconds':interval,
                        'Perf/collect_seconds':distributed.maximum(collect_seconds),
                        'Perf/learn_seconds':distributed.maximum(learn_seconds)}
                    if env.device.startswith('cuda'):
                        row['Perf/peak_cuda_gib']=distributed.maximum(torch.cuda.max_memory_allocated(env.device)/2**30)
                    if totals[2]:row.update(episode_return=totals[0]/totals[2],episode_length=totals[1]/totals[2],completed_episodes=int(totals[2]))
                    completed_return.clear();completed_length.clear()
                    distributed.primary_call(lambda:self.writer.write(row))
                    if self.progress:self.progress.update(**row,phase='training',state='training')
                    interval_started=time.monotonic();interval_step=self.control_steps;interval_updates=self.learner_updates
                    collect_seconds=0.;learn_seconds=0.
                if self.control_steps%cfg.save_interval==0:
                    self.save(f'checkpoint_{self.control_steps:07d}.pt')
                # Signals and wall budgets must stop every rank at the same step.
                if distributed.any(self.stop_requested):
                    stop_reason='signal';break
                if distributed.any(cfg.max_seconds is not None and time.monotonic()-started>=cfg.max_seconds):
                    stop_reason='time_budget';break
            self.elapsed_seconds=distributed.maximum(base_elapsed+time.monotonic()-started)
            checkpoint=self.save()
            result={'state':'stopped' if stop_reason=='signal' else 'completed',
                    'stop_reason':stop_reason,**self.counters(),
                    'elapsed_seconds':self.elapsed_seconds,'checkpoint':str(checkpoint)}
            if self.progress:self.progress.update(**result,phase='finished')
            else:distributed.primary_call(lambda:write_json(self.output/'status.json',result))
            return result
        finally:
            if self.writer is not None:self.writer.close()


def run(cfg):
    cfg.validate()
    requested=config_dict(cfg)
    world=int(os.environ.get('WORLD_SIZE','1'));local_rank=int(os.environ.get('LOCAL_RANK','0'))
    cfg=resolve_config(cfg,world,local_rank)
    torch.set_num_threads(cfg.torch_threads)
    if cfg.device.startswith('cuda'):
        torch.cuda.set_device(cfg.device)
        torch.set_float32_matmul_precision('high')
    distributed=Distributed.initialize(cfg.device)
    env=None;trainer=None;progress=None;handlers={}
    try:
        distributed.primary_call(lambda:Path(cfg.output).mkdir(parents=True,exist_ok=False))
        distributed.primary_call(lambda:write_json(Path(cfg.output)/'requested_config.json',requested))
        progress=Progress(cfg.output,distributed.rank,cfg.heartbeat_seconds)
        progress.update(phase='loading_scene_and_motions',world_size=world,
                        local_num_envs=cfg.num_envs,global_num_envs=cfg.num_envs*world)
        rank_seed=cfg.seed+distributed.rank
        random.seed(rank_seed);np.random.seed(rank_seed % 2**32);torch.manual_seed(rank_seed)
        print(f'[SuperMimic rank {distributed.rank}/{world}] device={cfg.device} seed={rank_seed} '
              f'envs={cfg.num_envs} WM batch={cfg.wm.batch_size} policy batch={cfg.policy.batch_size}',flush=True)
        env=TrackingEnv(cfg)
        progress.update(phase='constructing_learners')
        trainer=Trainer(cfg,env,distributed,progress)
        def stop(signum,frame):trainer.stop_requested=True
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,stop)
        return trainer.run()
    except BaseException as error:
        if progress is not None:progress.update(state='failed',phase='failed',error=str(error))
        raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        if trainer is not None:
            trainer.wm._captured_update.close()
            trainer.policy._captured_update.close()
        if env is not None:env.close()
        if progress is not None:progress.close()
        distributed.close()


def _arguments(parser):
    parser.add_argument('--project',type=Path)
    parser.add_argument('--batch-scope',choices=('global','per_rank'),help='official sizes default to global and split across ranks')
    parser.add_argument('--initialize-from',type=Path,help='explicit weights-only transfer; fresh optimizer, replay and counters')
    parser.add_argument('--tensorboard',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--library',type=Path,help='official packed BONES-SEED directory')
    parser.add_argument('--steps',type=int,help='absolute control-step target')
    parser.add_argument('--max-seconds',type=float,help='wall-clock budget for this training-loop invocation')
    parser.add_argument('--compile',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--capture-graphs',action=argparse.BooleanOptionalAction,default=None)


def _configure(cfg,args):
    return replace(cfg,**{name:getattr(args,name) for name in
        ('project','library','steps','max_seconds','compile','capture_graphs','batch_scope','initialize_from','tensorboard') if getattr(args,name) is not None})


def training_spec():
    return TrainingSpec(config_factory=Config,run=run,step_unit='control_steps',
        supports_distributed=True,description='Official SuperMimic GenCoord G1 motion tracking',
        add_arguments=_arguments,configure=_configure)
