"""Vectorized Gobot CartPole position-tracking environment.

Task: keep the pole upright while moving the cart to a per-episode target.

Observation (7-dim):
    [cos(theta), sin(theta), cart_pos, cart_vel, theta_vel, target_pos, cart_pos - target_pos]

Action (1-dim):
    horizontal force on the cart, clipped to [-3, 3].

Disturbance:
    every ``disturbance_interval`` steps, the pole hinge receives a short window
    of clipped white-noise torque. This is environment noise, not a policy
    action on the pole hinge.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

import gobot
from rsl_rl.env import VecEnv


_PROJECT_PATH = Path(__file__).resolve().parent


class CartPoleVecEnv(VecEnv):
    """rsl_rl VecEnv backed by Gobot's native MuJoCo vector environment."""

    num_obs: int = 7

    def __init__(
        self,
        num_envs: int = 64,
        max_episode_length: int = 1000,
        device: str = "cpu",
        project_path: str | Path = _PROJECT_PATH,
        disturbance_interval: int = 480,
        disturbance_duration: int = 60,
        disturbance_start: int = 240,
        disturbance_std: float = 0.05,
        disturbance_clip: float = 0.20,
        target_range: float = 0.8,
        action_limit: float = 3.0,
        seed: int = 42,
        num_workers: int = 0,
    ) -> None:
        self.num_envs = int(num_envs)
        self.max_episode_length = int(max_episode_length)
        self.device = device
        self.disturbance_interval = int(disturbance_interval)
        self.disturbance_duration = int(disturbance_duration)
        self.disturbance_start = int(disturbance_start)
        self.disturbance_std = float(disturbance_std)
        self.disturbance_clip = float(disturbance_clip)
        self.target_range = float(target_range)
        self.action_limit = float(action_limit)
        self.num_actions = 1
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self._rng = np.random.default_rng(seed)
        self._targets = np.zeros(self.num_envs, dtype=np.float32)

        self._native = self._make_native_env(Path(project_path), seed, num_workers)
        self.cfg = {
            "num_envs": self.num_envs,
            "max_episode_length": self.max_episode_length,
            "project_path": str(Path(project_path)),
            "disturbance_interval": self.disturbance_interval,
            "disturbance_duration": self.disturbance_duration,
            "disturbance_start": self.disturbance_start,
            "disturbance_std": self.disturbance_std,
            "disturbance_clip": self.disturbance_clip,
            "target_range": self.target_range,
            "action_limit": self.action_limit,
        }
        self._obs = self._reset_all(seed)

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {"policy": self._obs.clone()},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions_np = actions.detach().cpu().numpy().reshape(self.num_envs, self.num_actions)
        actions_np = np.clip(actions_np, -self.action_limit, self.action_limit)

        native_action = np.zeros((self.num_envs, 2), dtype=np.float64)
        native_action[:, 0] = actions_np[:, 0]

        if self.disturbance_std > 0.0:
            steps = self.episode_length_buf.detach().cpu().numpy()
            active = steps >= self.disturbance_start
            if self.disturbance_interval > 0:
                active &= ((steps - self.disturbance_start) % self.disturbance_interval) < max(
                    1, self.disturbance_duration
                )
            if np.any(active):
                noise = self._rng.normal(0.0, self.disturbance_std, active.sum())
                if self.disturbance_clip > 0.0:
                    noise = np.clip(noise, -self.disturbance_clip, self.disturbance_clip)
                native_action[active, 1] = noise

        raw_obs, _, terminated, truncated, _ = self._native.step(native_action)
        raw_obs = np.asarray(raw_obs, dtype=np.float32)

        obs = self._policy_obs(raw_obs)
        rewards = self._reward(obs)
        terminated = self._terminated(obs)

        self.episode_length_buf += 1
        time_outs = self.episode_length_buf >= self.max_episode_length
        dones = terminated | time_outs
        done_np = dones.detach().cpu().numpy()
        if np.any(done_np):
            done_ids = np.flatnonzero(done_np)
            reset_obs = self._reset_envs(done_ids)
            obs[done_ids] = reset_obs
            self.episode_length_buf[done_ids] = 0

        self._obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs_td = TensorDict(
            {"policy": self._obs.clone()},
            batch_size=[self.num_envs],
            device=self.device,
        )
        extras = {
            "time_outs": time_outs.to(dtype=torch.bool),
            "log": {
                "/cartpole/target_abs_mean": torch.mean(torch.abs(torch.as_tensor(self._targets, device=self.device))),
                "/cartpole/pos_error_abs_mean": torch.mean(torch.abs(self._obs[:, 6])),
            },
        }
        return obs_td, rewards, dones.to(dtype=torch.bool), extras

    def close(self) -> None:
        pass

    def _make_native_env(self, project_path: Path, seed: int, num_workers: int):
        gobot.set_project_path(str(project_path))

        cfg = gobot.NativeVectorEnvConfig()
        cfg.scene = "res://cartpole.jscn"
        cfg.robot = "cartpole"
        cfg.backend = gobot.PhysicsBackendType.MuJoCoCpu
        cfg.num_envs = self.num_envs
        cfg.batch_size = self.num_envs
        cfg.num_workers = int(num_workers)
        cfg.physics_dt = 1.0 / 240.0
        cfg.decimation = 1
        cfg.max_episode_steps = self.max_episode_length
        cfg.auto_reset = False
        cfg.seed = int(seed)
        cfg.task_json = json.dumps(
            {
                "observations": {
                    "groups": {
                        "policy": [
                            {"name": "cart_pos", "type": "joint_position", "joint": "slider"},
                            {"name": "cart_vel", "type": "joint_velocity", "joint": "slider"},
                            {"name": "theta", "type": "joint_position", "joint": "hinge", "wrap": True},
                            {"name": "theta_vel", "type": "joint_velocity", "joint": "hinge"},
                        ],
                    }
                },
                "terminations": {"terms": []},
                "rewards": {"terms": []},
                "events": {
                    "terms": [
                        {
                            "name": "reset_joint_state",
                            "type": "reset_joint_state",
                            "joints": {
                                "slider": {
                                    "position": 0.0,
                                    "position_range": [-0.5, 0.5],
                                    "velocity": 0.0,
                                },
                                "hinge": {
                                    "position": 0.0,
                                    "position_range": [-0.1, 0.1],
                                    "velocity": 0.0,
                                },
                            },
                        }
                    ]
                },
            }
        )

        cart_force = gobot.NativeVectorActionConfig()
        cart_force.name = "cart_force"
        cart_force.joint = "slider"
        cart_force.mode = gobot.NativeVectorActionMode.Effort
        cart_force.scale = 1.0
        cart_force.lower = -self.action_limit
        cart_force.upper = self.action_limit
        cart_force.unit = "N"
        cart_force.passive_joints = ["hinge"]

        push = gobot.NativeVectorActionConfig()
        push.name = "pole_white_noise"
        push.joint = "hinge"
        push.mode = gobot.NativeVectorActionMode.Effort
        push.scale = 1.0
        limit = max(self.disturbance_clip, self.disturbance_std * 4.0, 1.0e-6)
        push.lower = -limit
        push.upper = limit
        push.unit = "Nm"

        return gobot.NativeVectorEnv(cfg, [cart_force, push])

    def _reset_all(self, seed: int) -> torch.Tensor:
        raw_obs, _ = self._native.reset(seed)
        self._targets[:] = self._rng.uniform(-self.target_range, self.target_range, self.num_envs)
        obs = self._policy_obs(np.asarray(raw_obs, dtype=np.float32))
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def _reset_envs(self, env_ids: np.ndarray) -> np.ndarray:
        raw_obs, _ = self._native.reset(None, env_ids.tolist())
        self._targets[env_ids] = self._rng.uniform(-self.target_range, self.target_range, len(env_ids))
        return self._policy_obs(np.asarray(raw_obs, dtype=np.float32), env_ids=env_ids)

    def _policy_obs(self, raw_obs: np.ndarray, env_ids: np.ndarray | None = None) -> np.ndarray:
        theta = raw_obs[:, 2]
        cart_pos = raw_obs[:, 0]
        targets = self._targets if env_ids is None else self._targets[env_ids]
        return np.column_stack(
            [
                np.cos(theta),
                np.sin(theta),
                cart_pos,
                raw_obs[:, 1],
                raw_obs[:, 3],
                targets,
                cart_pos - targets,
            ]
        ).astype(np.float32, copy=False)

    def _reward(self, obs: np.ndarray) -> torch.Tensor:
        upright = obs[:, 0]
        pos_error = obs[:, 6]
        cart_vel = obs[:, 3]
        theta_vel = obs[:, 4]
        reward = upright - np.abs(pos_error) - 0.01 * (cart_vel * cart_vel + theta_vel * theta_vel)
        return torch.as_tensor(reward, dtype=torch.float32, device=self.device)

    def _terminated(self, obs: np.ndarray) -> torch.Tensor:
        cart_pos = obs[:, 2]
        theta = np.arctan2(obs[:, 1], obs[:, 0])
        terminated = (np.abs(cart_pos) > 0.95) | (np.abs(theta) > 0.5)
        return torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
