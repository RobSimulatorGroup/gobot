"""PPO with a two-head adversarial locomotion prior and terminal-state bootstrapping.

AMP losses follow Peng et al., SIGGRAPH 2021, equations 7-8. Dynamics remain in
Gobot's scene-compiled simulation; the discriminator is a training-only module.
"""
from __future__ import annotations
from dataclasses import asdict
import torch
from torch import nn
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.utils import check_nan
from .motion import MotionDataset, MotionPriorCfg


class MotionPriorPPO(PPO):
    def __init__(self, *args, motion_prior_cfg=None, **kwargs):
        self.prior_cfg = MotionPriorCfg(**motion_prior_cfg)
        super().__init__(*args, **kwargs)
        self.reference = MotionDataset(self.prior_cfg.dataset,device=self.device,seed=self.prior_cfg.seed,
            tolerances=(self.prior_cfg.speed_tolerance,self.prior_cfg.lateral_tolerance,self.prior_cfg.yaw_tolerance))
        self.discriminator = nn.Sequential(nn.Linear(self.reference.pairs.shape[1], 512), nn.ELU(),
                                           nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 2)).to(self.device)
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.prior_cfg.learning_rate)
        self.replay = torch.empty(self.prior_cfg.replay_capacity, self.reference.pairs.shape[1], device=self.device)
        self.replay_gait = torch.empty(self.prior_cfg.replay_capacity, dtype=torch.long, device=self.device)
        self.replay_size = self.replay_cursor = 0
        self.motion_reward_mean = 0.
        self.style_raw_mean = 0.
        self.update_count = 0
        self.replay_command = torch.empty(self.prior_cfg.replay_capacity,3,device=self.device)

    def _normalize(self, pairs):
        return ((pairs - self.reference.mean) / self.reference.std).clamp(-10, 10)

    def _remember(self, pair, gait, command):
        count = min(len(pair), len(self.replay))
        ids = (torch.arange(count, device=self.device) + self.replay_cursor) % len(self.replay)
        self.replay[ids] = pair[-count:]
        self.replay_gait[ids] = gait[-count:]
        self.replay_command[ids] = command[-count:]
        self.replay_cursor = (self.replay_cursor + count) % len(self.replay)
        self.replay_size = min(len(self.replay), self.replay_size + count)

    def process_env_step(self, obs, rewards, dones, extras):
        pair = extras["motion_pair"]
        state = extras["transition_state"]
        command = state["command"]
        cfg = self.prior_cfg
        blend = ((command[:, 0] - cfg.run_blend_start) / (cfg.run_blend_end - cfg.run_blend_start)).clamp(0, 1)
        with torch.no_grad():
            score = self.discriminator(self._normalize(pair))
            style = (1 - .25 * (score - 1).square()).clamp(min=0)
            if hasattr(self.reference, "supported_heads"):
                style = style * self.reference.supported_heads(command)
            style = style[:, 0] * (1 - blend) + style[:, 1] * blend
            active = (command[:, 0] >= .2) & (command[:, 1].abs() <= .15) & (command[:, 2].abs() <= .35)
            level_weight = 1 - .75 * (state["terrain_level"].float() / 9).clamp(0, 1)
            ramp = min(1., getattr(self, 'update_count', 0) / max(cfg.warmup_updates, 1))
            addition = cfg.weight * .02 * ramp * style * active * level_weight
            self.motion_reward_mean = float(addition.mean())
            self.style_raw_mean = float(style[active].mean()) if bool(active.any()) else 0.
            gait = (torch.rand(blend.shape, device=self.device, generator=self.reference.generator) < blend).long()
            if bool(active.any()):
                self._remember(pair[active], gait[active], command[active])
        # Terminal-state bootstrapping belongs to this explicit motion-prior
        # extension. The standard velocity baseline uses unmodified RSL-RL PPO.
        timeouts = extras.get('time_outs', torch.zeros_like(dones)).bool()
        terminated = extras.get('terminated', dones & ~timeouts)
        combined = rewards + addition - cfg.fall_penalty * terminated.bool()
        with torch.no_grad():
            if bool(timeouts.any()):
                final = TensorDict(extras['final_observation'], batch_size=[len(rewards)], device=self.device)[timeouts]
                check_nan(final, combined[timeouts], dones[timeouts])
                combined[timeouts] += self.gamma * self.critic(final).squeeze(-1)
        super().process_env_step(obs, combined, dones, {k: v for k, v in extras.items() if k != 'time_outs'})

    def update(self):
        result = super().update()
        self.update_count += 1
        if self.prior_cfg.weight == 0 or self.replay_size == 0:
            return {**result, "amp": 0., "amp_reward": 0.}
        loss_sum = real_sum = fake_sum = 0.
        cfg = self.prior_cfg
        for _ in range(cfg.updates_per_iteration):
            ids = torch.randint(self.replay_size, (cfg.batch_size,), device=self.device, generator=self.reference.generator)
            labels = self.replay_gait[ids]
            real, supported = self.reference.sample_matching(labels, self.replay_command[ids])
            if not bool(supported.any()):
                continue
            ids, labels, real = ids[supported], labels[supported], real[supported]
            fake_raw = self.replay[ids]
            # Both distributions must set the feature scale. Expert-only scaling
            # lets constant mocap wrists trivially identify every noisy policy.
            with torch.no_grad():
                mixture = torch.cat((real, fake_raw), 0)
                mean = mixture.mean(0)
                variance = mixture.var(0, unbiased=False)
                delta = mean - self.reference.mean
                running_var = .99 * self.reference.std.square() + .01 * variance + .0099 * delta.square()
                if _ == 0:
                    self.reference.mean.lerp_(mean, .01)
                    self.reference.std.copy_(running_var.sqrt().clamp_min(.05))
            real = self._normalize(real).detach().requires_grad_(True)
            fake = self._normalize(fake_raw)
            rows = torch.arange(len(real), device=self.device)
            real_score = self.discriminator(real)[rows, labels]
            fake_score = self.discriminator(fake)[rows, self.replay_gait[ids]]
            gradient = torch.autograd.grad(real_score.sum(), real, create_graph=True)[0]
            loss = .5 * ((real_score - 1).square().mean() + (fake_score + 1).square().mean())
            loss = loss + .5 * cfg.gradient_penalty * gradient.square().sum(1).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite AMP discriminator loss")
            self.disc_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.)
            self.disc_optimizer.step()
            loss_sum += float(loss.detach())
            real_sum += float(real_score.detach().mean())
            fake_sum += float(fake_score.detach().mean())
        result.update(amp=loss_sum / cfg.updates_per_iteration,
                      amp_expert_score=real_sum / cfg.updates_per_iteration,
                      amp_policy_score=fake_sum / cfg.updates_per_iteration,
                      amp_reward=self.motion_reward_mean, amp_style_raw=self.style_raw_mean)
        return result

    def save(self):
        return {**super().save(), "motion_prior": {
            "cfg": asdict(self.prior_cfg), "ppo_learning_rate": self.learning_rate, "dataset_digest": self.reference.digest,
            "discriminator": self.discriminator.state_dict(), "optimizer": self.disc_optimizer.state_dict(),
            "mean": self.reference.mean, "std": self.reference.std,
            "replay": self.replay[:self.replay_size].clone(), "replay_gait": self.replay_gait[:self.replay_size].clone(),
            "replay_size": self.replay_size, "replay_cursor": self.replay_cursor,
            "replay_command": self.replay_command[:self.replay_size].clone(), "update_count": self.update_count,
            "sampler_signature": self.reference.sampler_signature,
            "sampler_rng": self.reference.generator.get_state(),
        }}

    def load(self, loaded_dict, load_cfg=None, strict=True):
        restored = super().load(loaded_dict, load_cfg, strict)
        saved = loaded_dict.get("motion_prior")
        if saved is None:
            raise ValueError("AMP resume requires discriminator and replay state")
        if saved.get("sampler_signature") != self.reference.sampler_signature:
            raise ValueError("AMP sampler configuration changed; use an explicit fork")
        if saved["dataset_digest"] != self.reference.digest:
            raise ValueError("AMP checkpoint motion dataset mismatch")
        if len(saved["replay"]) > len(self.replay):
            raise ValueError("AMP resume replay capacity is smaller than saved buffer")
        self.learning_rate = float(saved.get("ppo_learning_rate", self.learning_rate))
        self.discriminator.load_state_dict(saved["discriminator"])
        self.disc_optimizer.load_state_dict(saved["optimizer"])
        self.reference.mean.copy_(saved["mean"]); self.reference.std.copy_(saved["std"])
        self.replay_size = int(saved["replay_size"])
        self.replay_cursor = int(saved["replay_cursor"])
        self.replay[:self.replay_size] = saved["replay"]
        self.replay_gait[:self.replay_size] = saved["replay_gait"]
        self.replay_command[:self.replay_size] = saved["replay_command"]
        self.update_count = int(saved["update_count"])
        self.reference.generator.set_state(saved["sampler_rng"].cpu())
        return restored
