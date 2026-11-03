"""Local PPO update guards; observation statistics stay fixed within each rollout.

The clipped PPO loss follows rsl_rl (ETH Zurich/NVIDIA, BSD-3-Clause).
This adapter deliberately supports the feed-forward, single-GPU locomotion recipes.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass
import torch
from torch import nn
from rsl_rl.algorithms import PPO


@dataclass(frozen=True)
class PpoGuardCfg:
    enabled: bool = False
    target_kl: float = .01
    stop_kl: float = .03
    rollback_kl: float = .06
    min_learning_rate: float = 3e-5
    max_learning_rate: float = 3e-4
    mean_limit: float = 4.
    mean_penalty: float = .01
    max_rejections: int = 3

    def __post_init__(self):
        if not 0 < self.target_kl < self.stop_kl < self.rollback_kl:
            raise ValueError('KL thresholds must be positive and ordered')
        if not 0 < self.min_learning_rate <= self.max_learning_rate:
            raise ValueError('invalid learning-rate bounds')
        if self.mean_limit <= 0 or self.mean_penalty < 0 or self.max_rejections < 1:
            raise ValueError('invalid PPO guard configuration')


class GuardedPPO(PPO):
    def __init__(self,*args,guard_cfg=None,**kwargs):
        self.guard=PpoGuardCfg(**(guard_cfg or {}))
        super().__init__(*args,**kwargs)
        if self.guard.enabled and (self.rnd or self.symmetry or self.is_multi_gpu or self.actor.is_recurrent):
            raise ValueError('guarded locomotion PPO requires independent feed-forward experiments')
        self.rejected_updates=0
        self.normalization_rejections=0
        self.total_rollbacks=0

    def freeze_normalization(self):
        for model in (self.actor,self.critic):
            normalizer=getattr(model,'obs_normalizer',None)
            if normalizer is not None:normalizer.eval()

    def train_mode(self):
        super().train_mode()
        if self.guard.enabled:self.freeze_normalization()

    def _learning_rate(self,value):
        self.learning_rate=max(self.guard.min_learning_rate,min(self.guard.max_learning_rate,float(value)))
        for group in self.optimizer.param_groups:group['lr']=self.learning_rate

    @torch.no_grad()
    def _statistics_boundary(self,obs):
        """A statistics change is checked separately from the optimizer's policy change."""
        probe=obs[:min(2048,obs.batch_size[0])]
        self.actor(probe,stochastic_output=True)
        before=tuple(value.clone() for value in self.actor.output_distribution_params)
        snapshots=[deepcopy(model.obs_normalizer.state_dict()) for model in (self.actor,self.critic)]
        for model in (self.actor,self.critic):
            model.obs_normalizer.train()
            model.update_normalization(obs)
        self.freeze_normalization()
        self.actor(probe,stochastic_output=True)
        divergence=float(self.actor.get_kl_divergence(before,self.actor.output_distribution_params).mean())
        rejected=not torch.isfinite(torch.tensor(divergence)) or divergence>self.guard.stop_kl
        if rejected:
            for model,state in zip((self.actor,self.critic),snapshots,strict=True):model.obs_normalizer.load_state_dict(state)
            self.normalization_rejections+=1
        else:self.normalization_rejections=0
        return divergence if divergence==divergence and abs(divergence)!=float('inf') else 1.e9,int(rejected)

    def update(self):
        if not self.guard.enabled:return super().update()
        snapshot=deepcopy(PPO.save(self))
        rng=torch.get_rng_state()
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        # A small bank is enough to detect a discontinuous normalization update.
        observations=self.storage.observations.flatten(0,1)
        metrics={k:0. for k in ('value','surrogate','entropy','kl','kl_max','actor_grad_norm','critic_grad_norm','mean_penalty')}
        count=0;rejected=False
        generator=self.storage.mini_batch_generator(self.num_mini_batches,self.num_learning_epochs)
        for batch in generator:
            self.actor(batch.observations,stochastic_output=True)
            params=self.actor.output_distribution_params
            with torch.no_grad():kl=float(self.actor.get_kl_divergence(batch.old_distribution_params,params).mean())
            metrics['kl_max']=max(metrics['kl_max'],kl)
            if not torch.isfinite(torch.tensor(kl)) or kl>self.guard.rollback_kl:
                rejected=True;break
            if kl>self.guard.stop_kl:break
            if kl>2*self.guard.target_kl:self._learning_rate(self.learning_rate/1.5)
            elif 0<kl<self.guard.target_kl/2:self._learning_rate(self.learning_rate*1.5)
            log_prob=self.actor.get_output_log_prob(batch.actions)
            ratio=(log_prob-batch.old_actions_log_prob.squeeze(-1)).exp()
            advantage=batch.advantages.squeeze(-1)
            if self.normalize_advantage_per_mini_batch:advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
            surrogate=torch.maximum(-advantage*ratio,-advantage*ratio.clamp(1-self.clip_param,1+self.clip_param)).mean()
            values=self.critic(batch.observations)
            value_loss=(values-batch.returns).square()
            if self.use_clipped_value_loss:
                clipped=batch.values+(values-batch.values).clamp(-self.clip_param,self.clip_param)
                value_loss=torch.maximum(value_loss,(clipped-batch.returns).square())
            value_loss=value_loss.mean()
            entropy=self.actor.output_entropy.mean()
            mean_penalty=(params[0].abs()-self.guard.mean_limit).clamp_min(0).square().mean()
            loss=surrogate+self.value_loss_coef*value_loss-self.entropy_coef*entropy+self.guard.mean_penalty*mean_penalty
            if not bool(torch.isfinite(loss)):
                rejected=True;break
            self.optimizer.zero_grad();loss.backward()
            actor_grad=nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm)
            critic_grad=nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm)
            if not bool(torch.isfinite(actor_grad)&torch.isfinite(critic_grad)):
                rejected=True;break
            self.optimizer.step()
            with torch.no_grad():
                distribution=self.actor.distribution
                if hasattr(distribution,'log_std_param'):distribution.log_std_param.clamp_(*distribution.log_std_range)
                self.actor(batch.observations,stochastic_output=True)
                after=float(self.actor.get_kl_divergence(batch.old_distribution_params,self.actor.output_distribution_params).mean())
            metrics['kl_max']=max(metrics['kl_max'],after)
            if not torch.isfinite(torch.tensor(after)) or after>self.guard.rollback_kl:
                rejected=True;break
            count+=1
            for key,value in [('value',value_loss),('surrogate',surrogate),('entropy',entropy),('kl',after),
                              ('actor_grad_norm',actor_grad),('critic_grad_norm',critic_grad),('mean_penalty',mean_penalty)]:
                metrics[key]+=float(value.detach()) if isinstance(value,torch.Tensor) else value
            if after>self.guard.stop_kl:break
        for key in metrics:
            if key!='kl_max':metrics[key]/=max(count,1)
        if rejected:
            rate=self.learning_rate
            PPO.load(self,snapshot,None,True)
            torch.set_rng_state(rng)
            if cuda_rng:torch.cuda.set_rng_state_all(cuda_rng)
            self._learning_rate(rate/2)
            self.rejected_updates+=1;self.total_rollbacks+=1
            norm_kl=0.;norm_rejected=0
        else:
            self.rejected_updates=0
            norm_kl,norm_rejected=self._statistics_boundary(observations)
        self.storage.clear()
        self.freeze_normalization()
        metrics.update(update_rejected=int(rejected),normalization_kl=norm_kl,normalization_rejected=norm_rejected,
                       learning_rate=self.learning_rate,optimizer_steps=count,guard_rollbacks=self.total_rollbacks)
        if self.rejected_updates>=self.guard.max_rejections:
            raise FloatingPointError('PPO rejected three consecutive updates; last accepted parameters restored')
        return metrics

    def save(self):
        return {**super().save(),'optimizer_guard':{'config':asdict(self.guard),'learning_rate':self.learning_rate,
            'rejected_updates':self.rejected_updates,'normalization_rejections':self.normalization_rejections,
            'total_rollbacks':self.total_rollbacks}}

    def load(self,loaded_dict,load_cfg=None,strict=True):
        saved=loaded_dict.get('optimizer_guard')
        if self.guard.enabled and (saved is None or saved['config']!=asdict(self.guard)):
            raise ValueError('optimizer guard configuration changed; use an explicit fork')
        result=super().load(loaded_dict,load_cfg,strict)
        if saved:
            self.learning_rate=saved['learning_rate']
            self.rejected_updates=saved['rejected_updates'];self.normalization_rejections=saved['normalization_rejections']
            self.total_rollbacks=saved['total_rollbacks']
        if self.guard.enabled:self.freeze_normalization()
        return result
