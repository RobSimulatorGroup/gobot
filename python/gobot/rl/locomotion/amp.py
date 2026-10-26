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
from .motion import MotionDataset, MotionPriorCfg


class MotionPriorPPO(PPO):
    def __init__(self, *args, motion_prior_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_cfg = MotionPriorCfg(**motion_prior_cfg)
        self.reference = MotionDataset(self.prior_cfg.dataset, device=self.device)
        self.discriminator = nn.Sequential(nn.Linear(self.reference.pairs.shape[1], 512), nn.ELU(),
                                           nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 2)).to(self.device)
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.prior_cfg.learning_rate)
        self.replay = torch.empty(self.prior_cfg.replay_capacity, self.reference.pairs.shape[1], device=self.device)
        self.replay_gait = torch.empty(self.prior_cfg.replay_capacity, dtype=torch.long, device=self.device)
        self.replay_size = self.replay_cursor = 0
        self.motion_reward_mean = 0.

    def _normalize(self, pairs):
        return ((pairs - self.reference.mean) / self.reference.std).clamp(-10, 10)

    def _remember(self, pair, gait):
        count = min(len(pair), len(self.replay))
        ids = (torch.arange(count, device=self.device) + self.replay_cursor) % len(self.replay)
        self.replay[ids] = pair[-count:]
        self.replay_gait[ids] = gait[-count:]
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
            style = style[:, 0] * (1 - blend) + style[:, 1] * blend
            active = command[:, :2].norm(dim=1) > .1
            # Relax imitation on difficult terrain and non-forward maneuvers.
            level_weight = 1 - .75 * (state["terrain_level"].float() / 9).clamp(0, 1)
            maneuver_weight = torch.where(command[:, 0] > .1, 1., .25)
            addition = cfg.weight * .02 * style * active * level_weight * maneuver_weight
            failure = dones & ~extras.get("time_outs", torch.zeros_like(dones))
            # A failure is an event cost, not a per-second regularizer. Without it,
            # early termination can avoid future costs during initial exploration.
            combined = rewards + addition - cfg.fall_penalty * failure
            self.motion_reward_mean = float(addition.mean())
            gait = (torch.rand(blend.shape, device=self.device, generator=self.reference.generator) < blend).long()
            self._remember(pair, gait)
            # RSL's default timeout path uses V(s_t). Use V(s_{t+1}) before reset.
            timeouts = extras.get("time_outs")
            if timeouts is not None:
                terminal = TensorDict(extras["final_observation"], batch_size=[len(rewards)], device=self.device)
                value = self.critic(terminal).squeeze(-1)
                combined = combined + self.gamma * value * timeouts
        forwarded = {key: value for key, value in extras.items() if key != "time_outs"}
        super().process_env_step(obs, combined, dones, forwarded)

    def update(self):
        result = super().update()
        if self.prior_cfg.weight == 0:
            return {**result, "amp": 0., "amp_reward": 0.}
        loss_sum = real_sum = fake_sum = 0.
        cfg = self.prior_cfg
        for _ in range(cfg.updates_per_iteration):
            real, labels = self.reference.sample(cfg.batch_size)
            ids = torch.randint(self.replay_size, (cfg.batch_size,), device=self.device, generator=self.reference.generator)
            fake_raw = self.replay[ids]
            # Both distributions must set the feature scale. Expert-only scaling
            # lets constant mocap wrists trivially identify every noisy policy.
            with torch.no_grad():
                mixture = torch.cat((real, fake_raw), 0)
                mean = mixture.mean(0)
                variance = mixture.var(0, unbiased=False)
                delta = mean - self.reference.mean
                running_var = .99 * self.reference.std.square() + .01 * variance + .0099 * delta.square()
                self.reference.mean.lerp_(mean, .01)
                self.reference.std.copy_(running_var.sqrt().clamp_min(.05))
            real = self._normalize(real).detach().requires_grad_(True)
            fake = self._normalize(fake_raw)
            rows = torch.arange(cfg.batch_size, device=self.device)
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
                      amp_reward=self.motion_reward_mean)
        return result

    def save(self):
        return {**super().save(), "motion_prior": {
            "cfg": asdict(self.prior_cfg), "ppo_learning_rate": self.learning_rate, "dataset_digest": self.reference.digest,
            "discriminator": self.discriminator.state_dict(), "optimizer": self.disc_optimizer.state_dict(),
            "mean": self.reference.mean, "std": self.reference.std,
            "replay": self.replay[:self.replay_size].clone(), "replay_gait": self.replay_gait[:self.replay_size].clone(),
            "replay_size": self.replay_size, "replay_cursor": self.replay_cursor,
            "sampler_rng": self.reference.generator.get_state(),
        }}

    def load(self, loaded_dict, load_cfg=None, strict=True):
        restored = super().load(loaded_dict, load_cfg, strict)
        saved = loaded_dict.get("motion_prior")
        if saved is None:
            raise ValueError("AMP resume requires discriminator and replay state")
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
        self.reference.generator.set_state(saved["sampler_rng"].cpu())
        return restored
