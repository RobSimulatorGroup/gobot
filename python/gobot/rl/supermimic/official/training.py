"""Official motion-only GenCoord learner driven by Gobot scene physics.

The official Ratio clock is preserved. Checkpoints add optimizer/RNG/context
state; resume starts fresh episodes and refills replay, not a bitwise trajectory.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from tensordict import TensorDict

from gobot.rl.training.workflow import TrainingSpec, config_dict
from .config import Config
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


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def source_identity():
    root=Path(__file__).parent
    files={p.name:digest(p) for p in sorted(root.glob('*.py'))}
    # Scene compilation and kinematics also define this training experiment.
    from gobot.rl import articulation,kinematics
    for module in (articulation,kinematics):files[module.__name__]=digest(module.__file__)
    return files


class Trainer:
    def __init__(self,cfg,env):
        self.cfg,self.env=cfg,env
        self.output=Path(cfg.output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.sources=source_identity()
        write_json(self.output/'config.json',config_dict(cfg))
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
        self.control_steps=0;self.learner_updates=0
        self.elapsed_seconds=0.;self.stop_requested=False
        self.history=None
        semantic=config_dict(cfg)
        for name in ('output','resume','steps','max_seconds','log_interval','save_interval','verify_data','project','library','torch_threads'):
            semantic.pop(name,None)
        identity={'config':semantic,'source':self.sources,'motion':env.motion.identity,
                  'scene':env.batch.artifact.content_digest,'profile':env.profile_digest}
        self.signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        self.identity=identity
        if cfg.resume:self.load(cfg.resume)
        write_json(self.output/'run.json',{'format':FORMAT,'identity':identity,'signature':self.signature,
            'resume_semantics':'fresh episodes, replay refill; optimizer, RNG and startup randomization restored',
            'wall_clock':'training loop including compilation, logging and periodic saves; excludes environment construction/data load and final save',
            'software':{'torch':torch.__version__},'device':env.device,
            'update_ratio':self.ratio.ratio})

    def counters(self):
        return {'control_steps':self.control_steps,'env_transitions':self.control_steps*self.cfg.num_envs,
                'learner_updates':self.learner_updates,
                'world_model_optimizer_steps':self.learner_updates*self.cfg.wm.num_batches,
                'actor_optimizer_steps':self.learner_updates*self.cfg.policy.num_batches,
                'critic_optimizer_steps':self.learner_updates*self.cfg.policy.num_batches*20}

    def save(self,name='latest.pt'):
        sampler=self.env.sampler
        sampler_state=None if sampler is None else {
            key:value.detach().cpu().clone() for key,value in vars(sampler).items() if torch.is_tensor(value)}
        rng_np=np.random.get_state()
        payload={'format':FORMAT,'config':config_dict(self.cfg),'signature':self.signature,'identity':self.identity,
            'actor':self.actor.state_dict(),'critic':self.critic.state_dict(),'dynamics':self.dynamics.state_dict(),
            'wm_optimizers':[o.state_dict() for o in self.wm.optimizers],
            'actor_optimizers':[o.state_dict() for o in self.policy.actor_optimizers],
            'critic_optimizers':[o.state_dict() for o in self.policy.critic_optimizers],
            'counters':self.counters(),'elapsed_seconds':self.elapsed_seconds,'ratio_previous':self.ratio.previous,
            'context':self.env.context.detach().cpu(),'distribution':self.env.distribution.detach().cpu(),
            'sampler':sampler_state,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
            'numpy_rng':[rng_np[0],rng_np[1].tolist(),rng_np[2],rng_np[3],rng_np[4]],
            'cuda_rng':torch.cuda.get_rng_state(self.env.device) if self.env.device.startswith('cuda') else None}
        destination=self.output/name;temporary=destination.with_suffix('.tmp')
        torch.save(payload,temporary);temporary.replace(destination)
        return destination

    def load(self,path):
        ck=torch.load(path,map_location=self.env.device,weights_only=True)
        if ck.get('format')!=FORMAT or ck.get('signature')!=self.signature:
            raise ValueError('resume requires matching official code, scene, data, device and training configuration')
        for key,module in (('actor',self.actor),('critic',self.critic),('dynamics',self.dynamics)):
            module.load_state_dict(ck[key])
        for key,optimizers in (('wm_optimizers',self.wm.optimizers),('actor_optimizers',self.policy.actor_optimizers),('critic_optimizers',self.policy.critic_optimizers)):
            for optimizer,state in zip(optimizers,ck[key],strict=True):optimizer.load_state_dict(state)
        self.control_steps=ck['counters']['control_steps'];self.learner_updates=ck['counters']['learner_updates']
        self.elapsed_seconds=ck['elapsed_seconds'];self.ratio.previous=ck['ratio_previous']
        self.env.context.copy_(ck['context']);self.env.apply_context()
        self.env.distribution.copy_(ck['distribution'])
        if self.env.sampler is not None:
            for key in ('value','count','effort'):
                getattr(self.env.sampler,key).copy_(ck['sampler'][key])
            # Buffered returns refer to old episodes; refill on resume.
            self.env.sampler._step=0
        random.setstate(ck['python_rng'])
        nr=ck['numpy_rng'];np.random.set_state((nr[0],np.array(nr[1],dtype=np.uint32),nr[2],nr[3],nr[4]))
        torch.set_rng_state(ck['torch_rng'].cpu())
        if ck['cuda_rng'] is not None:torch.cuda.set_rng_state(ck['cuda_rng'].cpu(),self.env.device)

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
        cfg,env=self.cfg,self.env
        state=env.reset()
        if cfg.history_length>1:
            self.history=self.representation.add_proprio_noise(env.sensor(state))[:,None].repeat(1,cfg.history_length,1)
        episode_reward=torch.zeros(cfg.num_envs,device=env.device)
        episode_length=torch.zeros_like(episode_reward)
        base_elapsed=self.elapsed_seconds;started=time.monotonic()
        metrics={};completed_return=[];completed_length=[]
        write_json(self.output/'status.json',{'state':'training',**self.counters()})
        try:
            while self.control_steps<cfg.steps and not self.stop_requested:
                current=state
                state,reward,done,actor_obs,actions=self.collect(current)
                # Statistics update outside no_grad collection, as in upstream.
                self.wm.update_normalization(current,actions)
                self.actor.train();self.actor.update_normalization(actor_obs)
                episode_reward+=reward;episode_length+=1
                if done.any():
                    completed_return.extend((episode_reward[done]/(.1*episode_length[done])).tolist())
                    completed_length.extend(episode_length[done].tolist())
                    episode_reward[done]=0.;episode_length[done]=0.
                repeats=self.ratio(self.control_steps)
                self.control_steps+=1
                for _ in range(repeats):
                    if self.buffer.num_transitions<cfg.wm.horizon:continue
                    try:
                        data=[self.buffer.sample(cfg.wm.horizon,cfg.wm.batch_size) for _ in range(cfg.wm.num_batches)]
                    except RuntimeError as error:
                        if 'no valid sequences' not in str(error):raise
                        # Unstable early policies can end every lane before a
                        # full WM window exists; keep collecting, don't fabricate windows.
                        continue
                    wm_stats=self.wm.update(data)
                    policy_stats=self.policy.update()
                    metrics={key:float(value.detach()) for key,value in (wm_stats|policy_stats).items()}
                    if not all(math_isfinite(value) for value in metrics.values()):
                        raise FloatingPointError('nonfinite official learning metrics')
                    self.learner_updates+=1
                self.elapsed_seconds=base_elapsed+time.monotonic()-started
                if self.control_steps%cfg.log_interval==0 or self.control_steps==cfg.steps:
                    row={**self.counters(),'elapsed_seconds':self.elapsed_seconds,**metrics}
                    if completed_return:
                        row.update(episode_return=float(np.mean(completed_return)),episode_length=float(np.mean(completed_length)))
                        completed_return.clear();completed_length.clear()
                    with (self.output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                    print(json.dumps(row,allow_nan=False),flush=True)
                if self.control_steps%cfg.save_interval==0:
                    self.save(f'checkpoint_{self.control_steps:07d}.pt')
                if cfg.max_seconds is not None and time.monotonic()-started>=cfg.max_seconds:break
            self.elapsed_seconds=base_elapsed+time.monotonic()-started
            checkpoint=self.save()
            result={'state':'stopped' if self.stop_requested else 'completed',
                    'stop_reason': 'signal' if self.stop_requested else ('steps' if self.control_steps>=cfg.steps else 'time_budget'),
                    **self.counters(),
                    'elapsed_seconds':self.elapsed_seconds,'checkpoint':str(checkpoint)}
            write_json(self.output/'status.json',result)
            return result
        except BaseException as error:
            write_json(self.output/'status.json',{'state':'failed','error':str(error),**self.counters()})
            raise


def math_isfinite(value):
    return bool(np.isfinite(value))


def run(cfg):
    cfg.validate();torch.set_num_threads(cfg.torch_threads)
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    if cfg.device.startswith('cuda'):
        torch.cuda.set_device(cfg.device)
        torch.set_float32_matmul_precision('high')
    env=TrackingEnv(cfg)
    handlers={}
    try:
        trainer=Trainer(cfg,env)
        def stop(signum,frame):trainer.stop_requested=True
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,stop)
        return trainer.run()
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        env.close()


def _arguments(parser):
    parser.add_argument('--project',type=Path)
    parser.add_argument('--library',type=Path,help='official packed BONES-SEED directory')
    parser.add_argument('--steps',type=int,help='absolute control-step target')
    parser.add_argument('--max-seconds',type=float,help='wall-clock budget for this training-loop invocation')
    parser.add_argument('--compile',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--capture-graphs',action=argparse.BooleanOptionalAction,default=None)


def _configure(cfg,args):
    return replace(cfg,**{name:getattr(args,name) for name in
        ('project','library','steps','max_seconds','compile','capture_graphs') if getattr(args,name) is not None})


def training_spec():
    return TrainingSpec(config_factory=Config,run=run,step_unit='control_steps',
        supports_distributed=False,description='Official SuperMimic GenCoord G1 motion tracking',
        add_arguments=_arguments,configure=_configure)
