"""BCE motion reward and discriminator training around unmodified RSL-RL PPO."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from .models import ClippedNormalization


@dataclass
class AMPConfig:
    observation_dim: int = 2390
    hidden_dims: tuple[int,int] = (1024,512)
    learning_rate: float = 2.5e-4
    weight_decay: float = 1.e-4
    gradient_penalty: float = 10.
    logit_regularization: float = .01
    reward_scale: float = 2.
    replay_capacity: int = 200000
    replay_samples: int = 1000
    batch_size: int = 8192
    updates_per_iteration: int = 32
    normalization_samples: int = 200000000
    optimizer: str = 'sgd'
    momentum: float = .9
    normalization_update: str = 'rollout'
    initialization: str = 'reference'

    def validate(self):
        for key in ('observation_dim','replay_capacity','replay_samples','batch_size','updates_per_iteration','normalization_samples'):
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,int) or value<=0:
                raise ValueError(f'AMP {key} must be a positive integer')
        for key in ('learning_rate','weight_decay','gradient_penalty','logit_regularization','reward_scale'):
            value=getattr(self,key)
            if not math.isfinite(value) or value<0:
                raise ValueError(f'AMP {key} must be finite and nonnegative')
        if self.learning_rate==0 or len(self.hidden_dims)!=2 or min(self.hidden_dims)<=0:
            raise ValueError('invalid discriminator network or learning rate')
        if self.optimizer!='sgd' or not 0 <= self.momentum < 1:
            raise ValueError('invalid discriminator optimizer or momentum')
        if self.normalization_update!='rollout' or self.initialization!='reference':
            raise ValueError('aligned AMP requires rollout normalization and reference initialization')


class MotionAMP(PPO):
    """Only the motion prior is task-specific; PPO's losses/update stay upstream."""
    def __init__(self,*args,amp_cfg=None,optimizer_momentum=0.,advantage_clip=None,**kwargs):
        self.amp_cfg=AMPConfig(**(amp_cfg or {}));self.amp_cfg.validate()
        super().__init__(*args,**kwargs)
        if not 0 <= optimizer_momentum < 1:
            raise ValueError('optimizer_momentum must be in [0,1)')
        if optimizer_momentum:
            if not isinstance(self.optimizer,torch.optim.SGD):
                raise ValueError('optimizer_momentum requires SGD')
            for group in self.optimizer.param_groups:group['momentum']=optimizer_momentum
        if advantage_clip is not None and (not math.isfinite(advantage_clip) or advantage_clip<=0):
            raise ValueError('advantage_clip must be finite and positive')
        self.advantage_clip=advantage_clip
        if self.is_multi_gpu:
            raise ValueError('motion AMP currently supports one learner process')
        c=self.amp_cfg
        self.discriminator=nn.Sequential(nn.Linear(c.observation_dim,c.hidden_dims[0]),nn.ReLU(),
            nn.Linear(c.hidden_dims[0],c.hidden_dims[1]),nn.ReLU(),nn.Linear(c.hidden_dims[1],1)).to(self.device)
        for layer in self.discriminator:
            if isinstance(layer,nn.Linear):nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.discriminator[-1].weight,-1.,1.)
        self.disc_optimizer=torch.optim.SGD(self.discriminator.parameters(),lr=c.learning_rate,
            weight_decay=c.weight_decay,momentum=c.momentum)
        self.disc_normalizer=ClippedNormalization(c.observation_dim,until=c.normalization_samples).to(self.device)
        self.replay=torch.empty(c.replay_capacity,c.observation_dim,device=self.device)
        self.replay_size=self.replay_cursor=0
        self._frames=[]
        self._reward_sum=0.
        self._invalid_count=0
        self.demo_sampler=None

    def _normalize(self,x):
        return self.disc_normalizer(x).clamp(-10,10)

    def process_env_step(self,obs,rewards,dones,extras):
        frame=extras['transition_observation']['disc']
        valid=extras['transition_observation'].get('valid',torch.ones_like(dones,dtype=torch.bool))
        if valid.shape!=dones.shape or valid.dtype!=torch.bool:
            raise ValueError('invalid AMP transition validity mask')
        if bool((~valid & ~extras['terminated'].bool()).any()):
            raise ValueError('invalid AMP transitions must terminate their episodes')
        if frame.shape!=(len(rewards),self.amp_cfg.observation_dim) or not bool(torch.isfinite(frame[valid]).all()):
            raise ValueError('invalid discriminator transition observation')
        with torch.no_grad():
            style=torch.zeros_like(rewards)
            if bool(valid.any()):
                logit=self.discriminator(self._normalize(frame[valid])).squeeze(-1)
                style[valid]=F.softplus(logit).clamp_max(-math.log(.0001))*self.amp_cfg.reward_scale
            # Standard runner logs the actual reward fed into PPO.
            rewards.copy_(style)
            self._reward_sum+=float(style.mean())
            self._frames.append(frame[valid].clone())
            self._invalid_count=getattr(self,'_invalid_count',0)+int((~valid).sum())
            combined=rewards.clone()
            # Bootstrap only time limits using the physical terminal observation.
            # A failure on the last tick is still a failure, with zero bootstrap.
            timeouts=extras.get('time_outs',torch.zeros_like(dones)).bool() & ~extras['terminated'].bool() & valid
            if bool(timeouts.any()):
                final=TensorDict(extras['final_observation'],batch_size=[len(rewards)],device=self.device)[timeouts]
                combined[timeouts]+=self.gamma*self.critic(final).squeeze(-1)
        super().process_env_step(obs,combined,dones,{k:v for k,v in extras.items() if k!='time_outs'})

    @torch.no_grad()
    def _remember(self,frames):
        count=min(len(frames),len(self.replay))
        ids=(torch.arange(count,device=self.device)+self.replay_cursor)%len(self.replay)
        self.replay[ids]=frames[-count:]
        self.replay_cursor=(self.replay_cursor+count)%len(self.replay)
        self.replay_size=min(len(self.replay),self.replay_size+count)

    def update(self):
        if not self._frames or self.demo_sampler is None:
            raise RuntimeError('AMP requires rollout frames and the Gobot motion sampler')
        frames=torch.cat(self._frames)
        if not len(frames):
            raise FloatingPointError('all AMP rollout transitions are invalid; inspect numerical_failures diagnostics')
        reward_mean=self._reward_sum/len(self._frames)
        invalid_count=self._invalid_count
        self._invalid_count=0
        self._frames.clear();self._reward_sum=0.
        transform=getattr(self.actor.distribution,'output_transform',None)
        saturation=None
        if transform is not None:
            normalized_actions=(self.storage.actions-transform.offset)/transform.scale
            saturation=float((normalized_actions.abs()>1).float().mean())
        if self.advantage_clip is not None:
            # RSL computes returns inside inference_mode; do not mutate that tensor.
            self.storage.advantages=self.storage.advantages.clamp(-self.advantage_clip,self.advantage_clip)
        # One expert pool per rollout. Keep the reward/loss coordinate system
        # fixed for the whole iteration, then update moments for the next one.
        c=self.amp_cfg
        demos=self.demo_sampler(len(frames)).detach()
        result=super().update()
        size=min(len(frames),c.replay_samples) if self.replay_size==len(self.replay) else len(frames)
        self._remember(frames[torch.randperm(len(frames),device=self.device)[:size]])
        totals=torch.zeros(4,device=self.device)
        for _ in range(c.updates_per_iteration):
            real=demos[torch.randint(len(demos),(c.batch_size,),device=self.device)]
            current=frames[torch.randint(len(frames),(c.batch_size,),device=self.device)]
            old=self.replay[torch.randint(self.replay_size,(c.batch_size,),device=self.device)]
            real=self._normalize(real).detach().requires_grad_(True)
            fake=self._normalize(torch.cat((current,old))).detach().requires_grad_(True)
            real_logits=self.discriminator(real).squeeze(-1)
            fake_logits=self.discriminator(fake).squeeze(-1)
            classification=.5*(F.binary_cross_entropy_with_logits(real_logits,torch.ones_like(real_logits))+
                               F.binary_cross_entropy_with_logits(fake_logits,torch.zeros_like(fake_logits)))
            real_grad=torch.autograd.grad(real_logits.sum(),real,create_graph=True)[0]
            fake_grad=torch.autograd.grad(fake_logits.sum(),fake,create_graph=True)[0]
            penalty=.5*(real_grad.square().sum(-1).mean()+fake_grad.square().sum(-1).mean())
            loss=classification+c.gradient_penalty*penalty+c.logit_regularization*self.discriminator[-1].weight.square().sum()
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite AMP discriminator loss')
            self.disc_optimizer.zero_grad(set_to_none=True);loss.backward();self.disc_optimizer.step()
            totals+=torch.stack((loss.detach(),penalty.detach(),(real_logits>0).float().mean(),(fake_logits<0).float().mean()))
        with torch.no_grad():self.disc_normalizer.update(torch.cat((demos,frames)))
        metrics=(totals/c.updates_per_iteration).tolist()
        result.update(zip(('amp_loss','amp_gradient_penalty','amp_expert_accuracy','amp_policy_accuracy'),metrics))
        result['amp_reward']=reward_mean
        result['amp_invalid_transitions']=float(invalid_count)
        if saturation is not None:result['action_clip_fraction']=saturation
        return result

    def save(self):
        return {**super().save(),'motion_amp':{'config':asdict(self.amp_cfg),
            'discriminator':self.discriminator.state_dict(),'optimizer':self.disc_optimizer.state_dict(),
            'normalizer':self.disc_normalizer.state_dict(),
            'replay':self.replay[:self.replay_size].clone(),'replay_cursor':self.replay_cursor,
            'learning_rate':self.learning_rate}}

    def load(self,data,load_cfg=None,strict=True):
        saved=data.get('motion_amp')
        if saved is None or asdict(AMPConfig(**saved['config']))!=asdict(self.amp_cfg):
            raise ValueError('AMP checkpoint configuration mismatch')
        result=super().load(data,load_cfg,strict)
        self.discriminator.load_state_dict(saved['discriminator'])
        self.disc_optimizer.load_state_dict(saved['optimizer'])
        self.disc_normalizer.load_state_dict(saved['normalizer'])
        self.replay_size=len(saved['replay']);self.replay_cursor=saved['replay_cursor']
        if self.replay_size>len(self.replay):raise ValueError('AMP replay exceeds capacity')
        self.replay[:self.replay_size]=saved['replay']
        self.learning_rate=saved['learning_rate']
        return result
