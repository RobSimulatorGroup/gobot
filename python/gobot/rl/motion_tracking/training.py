"""Reference collect → world-model fit → actor BPTT → critic fit schedule."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from gobot.rl.supermimic.distributed import allreduce_mean, broadcast_params, rank, world_size
from gobot.rl.supermimic.policy import Policy, Critic
from gobot.rl.supermimic.policy_trainer import PolicyTrainer
from gobot.rl.supermimic.world_model import WorldModel
from gobot.rl.supermimic.wm_trainer import WorldModelTrainer
from gobot.rl.supermimic.schedule import cosine_lr
from .convert import digest
from .identity import source_identity

FORMAT = 'gobot_supermimic_reproduced_v1'


class Trainer:
    def __init__(self, env, cfg):
        self.env, self.cfg = env, deepcopy(cfg)
        self.rank, self.world = rank(), world_size()
        self.log_dir = Path(cfg.log_root)/cfg.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir/'tb')) if self.rank==0 else None
        # Store base rates. Apply sqrt(world) exactly once, including after resume.
        effective = deepcopy(cfg)
        for c, key in ((effective.wm_trainer,'lr'),(effective.policy_trainer,'lr'),(effective.policy_trainer,'critic_lr')):
            setattr(c,key,getattr(c,key)*self.world**.5)
        self.effective = effective
        self.wm = WorldModel(cfg.world_model, env.heads, env.device)
        self.policy = Policy(cfg.policy).to(env.device)
        self.critic = Critic(cfg.critic,pcfg=cfg.policy).to(env.device)
        for m in (self.wm,self.policy,self.critic):
            broadcast_params(m)
        self.wm_trainer = WorldModelTrainer(self.wm,env.library,effective.wm_trainer)
        self.policy_trainer = PolicyTrainer(self.wm,self.policy,self.critic,env.library,effective.policy_trainer)
        self.iteration, self.start_iteration, self.num_updates, self.utd_acc = 0,0,0,0.
        self.identity = {'physics':env.batch.artifact.digest, 'kinematics':env.fk.scene.digest,
                         'data':[(p.name,digest(p)) for p in env.files], 'source':source_identity()}
        if cfg.resume:
            self.load(cfg.resume)
        if self.rank==0:
            import importlib.metadata as metadata
            software = {key:metadata.version(key) for key in ('torch','mujoco','mujoco-warp','warp-lang')}
            info = {'format':FORMAT,'base_config':asdict(self.cfg),'effective_config':asdict(effective),
                    'world_size':self.world, 'identity':self.identity,'software':software,
                    'params':{k:sum(p.numel() for p in m.parameters()) for k,m in (('wm',self.wm),('policy',self.policy),('critic',self.critic))}}
            (self.log_dir/'config.json').write_text(json.dumps(info,indent=2,default=str)+'\n')

    def counters(self):
        def steps(opt):
            return max((int(s['step']) for s in opt.state.values() if 'step' in s), default=0)
        return {'control_steps_per_rank':self.iteration,
                'global_env_transitions':self.iteration*self.env.num_envs*self.world,
                'learner_updates':self.num_updates,
                'world_model_optimizer_steps':steps(self.wm_trainer.opt),
                'policy_optimizer_steps':steps(self.policy_trainer.opt),
                'critic_optimizer_steps':steps(self.policy_trainer.critic_opt)}

    def save(self, name='ckpt_latest.pt'):
        # One shard per rank: sampler and random streams must not be overwritten by rank 0.
        name = name if self.rank==0 else name.replace('.pt',f'.rank{self.rank}.pt')
        payload = {'format':FORMAT,'iteration':self.iteration,'num_updates':self.num_updates,'counters':self.counters(),
            'wm':self.wm.state_dict(),'policy':self.policy.state_dict(),'critic':self.critic.state_dict(),
            'wm_opt':self.wm_trainer.opt.state_dict(),'policy_opt':self.policy_trainer.opt.state_dict(),
            'critic_opt':self.policy_trainer.critic_opt.state_dict(),
            'bin_return':self.env.sampler.bin_return,'bin_memory':self.env.sampler.bin_memory,
            'config':json.loads(json.dumps(asdict(self.cfg),default=str)), 'identity':self.identity,
            'rank':self.rank,'world_size':self.world,
            'startup_com':self.env.xi[:,30:33].detach().cpu().clone(),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(self.env.device) if self.env.device.startswith('cuda') else None,
            'resume_semantics':'fresh_episodes_and_replay_refill; UTD restarts at zero as in reference'}
        target = self.log_dir/name
        temporary = target.with_suffix('.tmp')
        torch.save(payload,temporary)
        temporary.replace(target)

    def load(self, path):
        path = Path(path)
        if self.rank:
            path = path.with_name(path.name.replace('.pt',f'.rank{self.rank}.pt'))
        ck = torch.load(path,map_location=self.env.device,weights_only=True)
        if ck.get('format') != FORMAT:
            raise ValueError('resume requires a Gobot SuperMimic-Reproduced training checkpoint')
        if ck['identity'] != self.identity or ck['world_size'] != self.world or ck['rank'] != self.rank:
            raise ValueError('resume requires identical scene, data, rank count and rank identity')
        old = ck['config']; new = json.loads(json.dumps(asdict(self.cfg),default=str))
        for key in ('env','world_model','policy','critic','wm_trainer','policy_trainer','backend','deterministic','nconmax','njmax','seed'):
            if old[key] != new[key]:
                raise ValueError(f'resume configuration changed: {key}')
        for key in ('total_iterations','updates_per_env_step','policy_updates_per_wm_update','warmup_iterations','lr_final_ratio','lr_decay_fraction'):
            if old['schedule'][key] != new['schedule'][key]:
                raise ValueError(f'resume schedule changed: {key}')
        for key, target in (('wm',self.wm),('policy',self.policy),('critic',self.critic),
                            ('wm_opt',self.wm_trainer.opt),('policy_opt',self.policy_trainer.opt),('critic_opt',self.policy_trainer.critic_opt)):
            target.load_state_dict(ck[key])
        for key in ('bin_return','bin_memory'):
            getattr(self.env.sampler,key).copy_(ck[key])
        startup_com=ck['startup_com'].to(self.env.device)
        if startup_com.shape != self.env.xi[:,30:33].shape or not torch.isfinite(startup_com).all():
            raise ValueError('invalid checkpoint startup COM context')
        self.env.xi[:,30:33].copy_(startup_com)
        self.env._apply_physics_context(include_com=True)
        self.iteration = self.start_iteration = ck['iteration']
        self.num_updates = ck['num_updates']
        if self.world>1:
            import torch.distributed as dist
            counters=[None]*self.world
            dist.all_gather_object(counters,self.counters())
            if any(value!=counters[0] for value in counters[1:]):
                raise ValueError('checkpoint shards contain different training counters')
        torch.set_rng_state(ck['torch_rng'].cpu())
        if ck['cuda_rng'] is not None:
            torch.cuda.set_rng_state(ck['cuda_rng'].cpu(),self.env.device)

    def run(self):
        cfg, sch = self.effective, self.cfg.schedule
        env = self.env
        stop = min(sch.total_iterations, self.cfg.stop_iteration or sch.total_iterations)
        obs, _ = env.reset()
        ep_sum = torch.zeros(env.num_envs,device=env.device)
        ep_len = torch.zeros_like(ep_sum)
        started = time.monotonic()
        metrics = {}
        aggregate, aggregate_n = {}, 0
        interval_return = interval_count = interval_length = 0.
        while self.iteration < stop:
            with torch.no_grad():
                action = self.policy(obs['policy'],obs['goal'])
                obs,reward,term,timeout,_ = env.step(action)
                ep_sum += reward; ep_len += 1
                done = term|timeout
                done_count = int(done.sum())
                return_sum = float((ep_sum[done]/(.1*ep_len[done])).sum())
                interval_return += return_sum; interval_count += done_count
                interval_length += float(ep_len[done].sum())
                ep_sum[done] = 0; ep_len[done] = 0
            self.iteration += 1
            if self.iteration-self.start_iteration >= sch.warmup_iterations:
                self.utd_acc += sch.updates_per_env_step
                while self.utd_acc >= 1.:
                    self.utd_acc -= 1.
                    lr = cosine_lr(self.iteration,sch.total_iterations,cfg.wm_trainer.lr,sch.lr_final_ratio,sch.lr_decay_fraction)
                    self.wm_trainer.set_lr(lr)
                    self.policy_trainer.set_lr(lr*cfg.policy_trainer.lr/cfg.wm_trainer.lr)
                    metrics = {'wm/'+k:v for k,v in self.wm_trainer.update(env.buffer).items()}
                    for _ in range(sch.policy_updates_per_wm_update):
                        metrics.update({'policy/'+k:v for k,v in self.policy_trainer.update(env.buffer).items()})
                    metrics['lr'] = lr
                    self.num_updates += 1
                    for key,value in metrics.items():
                        aggregate[key] = value if key=='lr' else aggregate.get(key,0.)+value
                    aggregate_n += 1
                    if not all(math.isfinite(v) for v in metrics.values()):
                        raise FloatingPointError(f'nonfinite learner metric at control iteration {self.iteration}')
            # Collectives have identical control flow on all ranks, including zero finished episodes.
            if self.iteration % sch.log_interval == 0 or self.iteration == stop:
                total = allreduce_mean(interval_return); count = allreduce_mean(interval_count)
                length = allreduce_mean(interval_length)
                averages={k:v/(1 if k=='lr' else aggregate_n) for k,v in aggregate.items()} if aggregate_n else {}
                row = {'iteration':self.iteration,'num_updates':self.num_updates,
                       'transitions':self.iteration*env.num_envs*self.world,
                       'elapsed_seconds':time.monotonic()-started,**self.counters(),**averages}
                if count:
                    row['train/return'] = total/count
                    row['train/episode_length'] = length/count
                interval_return = interval_count = interval_length = 0.
                aggregate, aggregate_n = {}, 0
                if self.rank==0:
                    # Match the reference's rank-0 diagnostic sampling, including
                    # its RNG consumption on scheduled logging iterations.
                    if self.iteration%sch.log_interval==0 and env.buffer.size>2048:
                        error=self.wm_trainer.online_error(env.buffer,n=1024,k=8)
                        row['wm/online_nse_1step']=float(error[0])
                        row['wm/online_nse_8step']=float(error[-1])
                    row['sampler/bin_return_mean']=float(env.sampler.bin_return.mean())
                    row['sampler/entropy']=float(env.sampler.metrics['sampling_entropy'][0])
                    with open(self.log_dir/'metrics.jsonl','a') as stream:
                        stream.write(json.dumps(row,allow_nan=False)+'\n')
                    for key,value in row.items():
                        self.writer.add_scalar(key,value,self.iteration)
                    print(json.dumps(row,allow_nan=False),flush=True)
                if sch.max_hours is not None and allreduce_mean(float((time.monotonic()-started)/3600 >= sch.max_hours))>0:
                    break
            if self.iteration % sch.checkpoint_interval == 0:
                self.save(); self.save(f'ckpt_{self.iteration:07d}.pt')
        self.save()
        if self.writer:
            self.writer.flush()
        return {'iteration':self.iteration,'num_updates':self.num_updates,'counters':self.counters(),'checkpoint':str(self.log_dir/'ckpt_latest.pt')}

    def close(self):
        if self.writer:
            self.writer.close()
