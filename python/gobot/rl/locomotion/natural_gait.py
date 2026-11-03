"""Soft quadruped contact timing, with an observable clock and pre-reset rewards."""
from __future__ import annotations
from dataclasses import replace
import math
import torch
from gobot.rl.batch import BatchEnvState
from gobot.rl.spec import ObservationSpec, SpecField

# FR, FL, RR, RL: lateral-sequence walk, diagonal trot, left-lead gallop.
PHASES = ((0., .5, .75, .25), (0., .5, .5, 0.), (.6, .5, .1, 0.))


def gait_weights(speed, cfg):
    walk_to_trot = ((speed - cfg.walk_speed + cfg.blend_width / 2) / cfg.blend_width).clamp(0, 1)
    trot_to_run = ((speed - cfg.run_speed + cfg.blend_width / 2) / cfg.blend_width).clamp(0, 1)
    return torch.stack((1 - walk_to_trot, walk_to_trot - trot_to_run, trot_to_run), -1)


def desired_contact(phase, weights, cfg):
    offsets = phase.new_tensor(PHASES)
    duty = phase.new_tensor(cfg.duty_factors)
    angle = (phase[:, None, None] - offsets[None] - duty[None, :, None] / 2) * math.tau
    # Soft boundaries preserve swing/stance freedom when a foot encounters an obstacle.
    stance = torch.sigmoid(12 * (angle.cos() - torch.cos(math.pi * duty)[None, :, None]))
    return (stance * weights[:, :, None]).sum(1)


class NaturalGaitEnv:
    """Shared CPU/CUDA wrapper. The actor observes everything used by the clock."""
    def __init__(self, env):
        if len(env.cfg_obj.robot.feet) != 4 or not env.cfg_obj.capture_transitions:
            raise ValueError('natural gait requires quadruped pre-reset transitions')
        if tuple(env.cfg_obj.foot_names) != ('FR', 'FL', 'RR', 'RL'):
            raise ValueError('natural gait expects named FR, FL, RR, RL feet')
        self.env = env
        self.device = env.device
        self.phase = torch.zeros(env.num_envs, device=self.device)
        fields = (SpecField('gait_clock', 2), SpecField('gait_weights', 3))
        self.observation_spec = ObservationSpec('gobot_natural_gait_actor_v1', (*env.observation_spec.fields, *fields))
        self.critic_observation_spec = ObservationSpec('gobot_natural_gait_critic_v1', (*env.critic_observation_spec.fields, *fields))
        self.actor_obs_schema = self.observation_spec
        self.num_obs = self.observation_spec.dim
        self.num_privileged_obs = self.critic_observation_spec.dim
        self.task_runtime_metadata = replace(env.task_runtime_metadata, obs_groups_spec=self.obs_groups_spec)
        self.cfg = {**env.cfg, 'num_obs': self.num_obs, 'obs_spec': self.observation_spec.metadata()}
        self._index = 0
        self._buffers = [{"actor": torch.empty(env.num_envs,self.num_obs,device=self.device),
                          "critic": torch.empty(env.num_envs,self.num_privileged_obs,device=self.device)} for _ in range(2)]
        self._state = BatchEnvState(obs={}, reward=None, terminated=None, truncated=None, info={})
        self._render(env.state)

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def state(self):
        return self._state

    @property
    def obs_groups_spec(self):
        return {'actor': self.num_obs, 'critic': self.num_privileged_obs}

    def init_state(self):
        return self._state

    def tensor(self, value, dtype=torch.float32):
        # CPU native state arrays may be read-only views.
        if hasattr(value, 'flags') and not value.flags.writeable:
            value = value.copy()
        return torch.as_tensor(value, device=self.device, dtype=dtype)

    def _clock(self, command):
        weights = gait_weights(self.tensor(command)[:, :2].norm(dim=1), self.cfg_obj.gait)
        return torch.cat((torch.sin(self.phase * math.tau)[:, None], torch.cos(self.phase * math.tau)[:, None], weights), 1)

    def _augment(self, obs, command):
        clock = self._clock(command)
        return {key: torch.cat((self.tensor(value), clock), 1) for key, value in obs.items()}

    def _render(self, core):
        values = self._augment(core.obs, self.env.locomotion_state()['command'])
        for key,value in values.items(): self._buffers[self._index][key].copy_(value)
        self._state.obs = self._buffers[self._index]
        self._state.reward, self._state.terminated, self._state.truncated = core.reward, core.terminated, core.truncated
        self._state.info = dict(core.info)

    def get_observations(self):
        return self._state.obs

    def set_command(self, command):
        self.env.set_command(command)
        self._render(self.env.state)

    def reset(self, env_ids=None, *, seed=None):
        _, info = self.env.reset(env_ids, seed=seed)
        ids = slice(None) if env_ids is None else self.tensor(env_ids, torch.long)
        self.phase[ids] = 0.
        self._render(self.env.state)
        self._state.final_observation = None
        return {k: v[ids].clone() for k, v in self._state.obs.items()}, info

    def reset_all(self, seed=None):
        return self.reset(seed=seed)[0]

    def step(self, action):
        command = self.tensor(self.env.locomotion_state()['command']).clone()
        weights = gait_weights(command[:, :2].norm(dim=1), self.cfg_obj.gait)
        frequency = (weights * command.new_tensor(self.cfg_obj.gait.frequencies)).sum(1)
        moving = command.norm(dim=1) > self.cfg_obj.gait.active_threshold
        self.phase.copy_((self.phase + frequency * self.env.step_dt * moving) % 1.)
        core = self.env.step(action)
        physical = core.info['transition_state']
        contact = self.tensor(physical['foot_force']).norm(dim=-1) > 10.
        target = desired_contact(self.phase, weights, self.cfg_obj.gait)
        timing = 1. - (contact.float() - target).square().mean(1)
        speed = command[:, :2].norm(dim=1)
        actual = self.tensor(physical['base_linear_velocity_body'])[:, :2]
        progress = ((actual * command[:, :2]).sum(1) / speed.square().clamp_min(.01)).clamp(0, 1)
        level = self.tensor(physical['terrain_level']) / max(self.env.terrain_grid_shape[0] - 1, 1)
        # No timing incentive for standing, yaw-only commands, or sliding without progress.
        forward = (command[:,0]>.1)&(command[:,1].abs()<.15)&(command[:,2].abs()<.35)
        timing = timing * torch.where(forward,1.,.2)
        reward = self.cfg_obj.gait.weight * self.env.step_dt * timing * progress * moving * (1 - .8 * level)
        terminal = self._augment(core.info['final_observation'], command)
        done = self.tensor(core.terminated, torch.bool) | self.tensor(core.truncated, torch.bool)
        self.phase[done] = 0.
        self._index = 1-self._index
        self._render(core)
        self._state.reward = self.tensor(core.reward) + reward
        self._state.final_observation = terminal
        self._state.info.update(final_observation=terminal, transition_observation=terminal,
                                gait_reward=reward, _final_observation=done)
        return self._state

    def load_training_state_dict(self, state):
        result = self.env.load_training_state_dict(state)
        self.phase.zero_()  # Restoring the core starts fresh physical episodes.
        self._render(self.env.state)
        self._state.final_observation = None
        return result
