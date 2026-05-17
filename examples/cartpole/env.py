"""Vectorized CartPole position-tracking environment.

Task: balance the pole upright while moving the cart to a target position.
The target position is randomized on every episode reset.

Observation (7-dim):
    [cos(theta), sin(theta), cart_pos, cart_vel, theta_vel, target_pos, cart_pos - target_pos]

Action (1-dim):
    horizontal force on the cart, clipped to [-3, 3].

Disturbance:
    every ``disturbance_interval`` steps, the pole hinge receives a short window
    of clipped white-noise torque. This is environment noise, not policy action.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

try:
    import mujoco
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "examples/cartpole requires the Python mujoco package for training. "
        "Install it in your RL environment before running train.py."
    ) from error

from rsl_rl.env import VecEnv


_XML_PATH = Path(__file__).with_name("inverted_pendulum.xml")


class CartPoleVecEnv(VecEnv):
    """rsl_rl VecEnv implemented directly with Python MuJoCo."""

    num_obs: int = 7

    def __init__(
        self,
        num_envs: int = 64,
        max_episode_length: int = 1000,
        device: str = "cpu",
        xml_path: str | Path = _XML_PATH,
        disturbance_interval: int = 480,
        disturbance_duration: int = 60,
        disturbance_start: int = 240,
        disturbance_std: float = 0.05,
        disturbance_clip: float = 0.20,
        target_range: float = 0.8,
        action_limit: float = 3.0,
        seed: int = 42,
    ) -> None:
        self.num_envs = int(num_envs)
        self.max_episode_length = int(max_episode_length)
        self.device = device
        self.xml_path = Path(xml_path)
        self.disturbance_interval = int(disturbance_interval)
        self.disturbance_duration = int(disturbance_duration)
        self.disturbance_start = int(disturbance_start)
        self.disturbance_std = float(disturbance_std)
        self.disturbance_clip = float(disturbance_clip)
        self.target_range = float(target_range)
        self.action_limit = float(action_limit)
        self.num_actions = 1

        self._model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self._datas = [mujoco.MjData(self._model) for _ in range(self.num_envs)]
        self._rng = np.random.default_rng(seed)
        self._targets = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)

        self.cfg = {
            "num_envs": self.num_envs,
            "max_episode_length": self.max_episode_length,
            "xml_path": str(self.xml_path),
            "disturbance_interval": self.disturbance_interval,
            "disturbance_duration": self.disturbance_duration,
            "disturbance_start": self.disturbance_start,
            "disturbance_std": self.disturbance_std,
            "disturbance_clip": self.disturbance_clip,
            "target_range": self.target_range,
            "action_limit": self.action_limit,
        }
        self._obs = self._reset_all()

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {"policy": self._obs.clone()},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def reset(self, seed: int | None = None) -> TensorDict:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.episode_length_buf.zero_()
        self._obs = self._reset_all()
        return self.get_observations()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions_np = actions.detach().cpu().numpy().reshape(self.num_envs, self.num_actions)
        actions_np = np.clip(actions_np, -self.action_limit, self.action_limit)

        obs_list = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        time_outs = np.zeros(self.num_envs, dtype=bool)

        for env_id, data in enumerate(self._datas):
            data.ctrl[0] = actions_np[env_id, 0]
            data.ctrl[1] = self._sample_disturbance(env_id)
            mujoco.mj_step(self._model, data)

            obs = self._get_obs(data, self._targets[env_id])
            rewards[env_id] = self._reward(data, self._targets[env_id])
            terminated = self._terminated(data)

            self.episode_length_buf[env_id] += 1
            timed_out = bool(self.episode_length_buf[env_id] >= self.max_episode_length)
            done = terminated or timed_out
            dones[env_id] = done
            time_outs[env_id] = timed_out and not terminated

            if done:
                self._reset_env(env_id)
                self.episode_length_buf[env_id] = 0
                obs = self._get_obs(data, self._targets[env_id])

            obs_list.append(obs)

        self._obs = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)
        obs_td = TensorDict(
            {"policy": self._obs.clone()},
            batch_size=[self.num_envs],
            device=self.device,
        )
        extras = {
            "time_outs": torch.as_tensor(time_outs, dtype=torch.bool, device=self.device),
            "log": {
                "/cartpole/target_abs_mean": torch.mean(torch.abs(torch.as_tensor(self._targets, device=self.device))),
                "/cartpole/pos_error_abs_mean": torch.mean(torch.abs(self._obs[:, 6])),
            },
        }
        return (
            obs_td,
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(dones, dtype=torch.bool, device=self.device),
            extras,
        )

    def close(self) -> None:
        pass

    def _sample_disturbance(self, env_id: int) -> float:
        if self.disturbance_std <= 0.0:
            return 0.0
        step = int(self.episode_length_buf[env_id].item())
        if step < self.disturbance_start:
            return 0.0
        if self.disturbance_interval > 0:
            offset = (step - self.disturbance_start) % self.disturbance_interval
            if offset >= max(1, self.disturbance_duration):
                return 0.0
        value = float(self._rng.normal(0.0, self.disturbance_std))
        if self.disturbance_clip > 0.0:
            value = float(np.clip(value, -self.disturbance_clip, self.disturbance_clip))
        return value

    def _get_obs(self, data: mujoco.MjData, target: float) -> np.ndarray:
        theta = float(data.qpos[1])
        cart_pos = float(data.qpos[0])
        return np.array(
            [
                np.cos(theta),
                np.sin(theta),
                cart_pos,
                float(data.qvel[0]),
                float(data.qvel[1]),
                target,
                cart_pos - target,
            ],
            dtype=np.float32,
        )

    def _reward(self, data: mujoco.MjData, target: float) -> float:
        theta = float(data.qpos[1])
        cart_pos = float(data.qpos[0])
        cart_vel = float(data.qvel[0])
        theta_vel = float(data.qvel[1])
        pos_error = cart_pos - target
        return float(np.cos(theta) - abs(pos_error) - 0.01 * (cart_vel * cart_vel + theta_vel * theta_vel))

    def _terminated(self, data: mujoco.MjData) -> bool:
        cart_pos = float(data.qpos[0])
        theta = float(np.arctan2(np.sin(data.qpos[1]), np.cos(data.qpos[1])))
        return bool(abs(cart_pos) > 0.95 or abs(theta) > 0.5)

    def _reset_env(self, env_id: int) -> None:
        data = self._datas[env_id]
        mujoco.mj_resetData(self._model, data)
        data.qpos[0] = self._rng.uniform(-0.5, 0.5)
        data.qpos[1] = self._rng.uniform(-0.1, 0.1)
        data.qvel[0] = self._rng.uniform(-0.1, 0.1)
        data.qvel[1] = self._rng.uniform(-0.1, 0.1)
        data.ctrl[:] = 0.0
        self._targets[env_id] = self._rng.uniform(-self.target_range, self.target_range)
        mujoco.mj_forward(self._model, data)

    def _reset_all(self) -> torch.Tensor:
        obs_list = []
        for env_id in range(self.num_envs):
            self._reset_env(env_id)
            obs_list.append(self._get_obs(self._datas[env_id], self._targets[env_id]))
        return torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)
