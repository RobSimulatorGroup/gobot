"""History observations and complete transitions shared by CPU, CUDA and playback."""
from __future__ import annotations

from dataclasses import replace
import torch

from gobot.rl.batch import BatchEnvState
from gobot.rl.spec import ObservationSpec, SpecField
from .motion import motion_features


class HistoryVelocityEnv:
    def __init__(self, env, length: int):
        if length < 1 or not env.cfg_obj.capture_transitions:
            raise ValueError("history requires positive length and pre-reset transition capture")
        self.env = env
        self.length = int(length)
        self.proprio_dim = 9 + 3 * env.num_actions
        self.device = env.device
        self._device = torch.device(self.device)
        self.observation_spec = ObservationSpec("gobot_velocity_history_v2", (
            SpecField("proprio_history", self.length * self.proprio_dim),
            SpecField("command", 3, "m/s,rad/s"),
            SpecField("height_scan", env.num_obs - self.proprio_dim - 3, "normalized"),
        ))
        self.actor_obs_schema = self.observation_spec
        self.num_obs = self.observation_spec.dim
        self.critic_observation_spec = env.critic_observation_spec
        self.num_privileged_obs = env.num_privileged_obs
        self.task_runtime_metadata = replace(env.task_runtime_metadata, obs_groups_spec=self.obs_groups_spec)
        self.cfg = {**env.cfg, "num_obs": self.num_obs, "history_length": self.length,
                    "obs_spec": self.observation_spec.metadata()}
        self._history = torch.zeros(env.num_envs, self.length, self.proprio_dim, device=self._device)
        self._actors = [torch.empty(env.num_envs, self.num_obs, device=self._device) for _ in range(2)]
        self._critics = [torch.empty(env.num_envs, self.num_privileged_obs, device=self._device) for _ in range(2)]
        self._index = 0
        self._state = BatchEnvState(obs={}, reward=None, terminated=None, truncated=None, info={})
        self._fill_history(env.state.obs)
        self._render(env.state)

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def state(self):
        return self._state

    @property
    def obs_groups_spec(self):
        return {"actor": self.num_obs, "critic": self.num_privileged_obs}

    @property
    def accepts_device_actions(self):
        return getattr(self.env, "accepts_device_actions", False)

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self._device)

    def _fill_history(self, obs, ids=None):
        current = self._tensor(obs["actor"])[:, :self.proprio_dim]
        if ids is None:
            self._history.copy_(current[:, None, :].expand_as(self._history))
        else:
            self._history[ids] = current[ids, None, :]

    def _render(self, core):
        actor = self._actors[self._index]
        actor[:, :self.length * self.proprio_dim] = self._history.flatten(1)
        actor[:, self.length * self.proprio_dim:] = self._tensor(core.obs["actor"])[:, self.proprio_dim:]
        self._critics[self._index].copy_(self._tensor(core.obs["critic"]))
        self._state.obs = {"actor": actor, "critic": self._critics[self._index]}
        self._state.reward, self._state.terminated, self._state.truncated = core.reward, core.terminated, core.truncated
        self._state.info = dict(core.info)

    def reset(self, env_ids=None, *, seed=None):
        _, info = self.env.reset(env_ids, seed=seed)
        ids = None if env_ids is None else torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        self._fill_history(self.env.state.obs, ids)
        self._render(self.env.state)
        self._state.final_observation = None
        return ({k: v.clone() if ids is None else v[ids].clone() for k, v in self._state.obs.items()}, info)

    def reset_all(self, seed=None):
        return self.reset(seed=seed)[0]

    def get_observations(self):
        return self._state.obs

    def set_command(self, command):
        self.env.set_command(command)
        self._render(self.env.state)

    def step(self, actions):
        before = motion_features(self.env.locomotion_state(), device=self._device) if self.cfg_obj.state_link_names else None
        core = self.env.step(actions)
        terminal = core.info["transition_observation"]
        tail = self._tensor(terminal["actor"])
        final_history = torch.cat((self._history[:, 1:], tail[:, None, :self.proprio_dim]), dim=1)
        final_obs = {"actor": torch.cat((final_history.flatten(1), tail[:, self.proprio_dim:]), dim=1),
                     "critic": self._tensor(terminal["critic"]).clone()}
        self._history[:, :-1] = self._history[:, 1:].clone()
        self._history[:, -1] = self._tensor(core.obs["actor"])[:, :self.proprio_dim]
        done = torch.as_tensor(core.terminated, device=self._device) | torch.as_tensor(core.truncated, device=self._device)
        self._fill_history(core.obs, done)
        self._index = 1 - self._index
        self._render(core)
        self._state.final_observation = final_obs
        self._state.info.update(final_observation=final_obs, transition_observation=final_obs, _final_observation=done)
        if before is not None:
            after = motion_features(core.info["transition_state"], device=self._device)
            self._state.info["motion_pair"] = torch.cat((before, after), dim=1)
        return self._state

    def load_training_state_dict(self, state):
        result = self.env.load_training_state_dict(state)
        # The core already starts fresh episodes; do not consume the RNG twice.
        self._fill_history(self.env.state.obs)
        self._render(self.env.state)
        self._state.final_observation = None
        return result
