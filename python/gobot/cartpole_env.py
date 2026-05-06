"""CartPole RL adapter built on Gobot's public Python API."""

from __future__ import annotations

import math
from typing import Any, Sequence

from . import _core


class CartPoleEnv:
    def __init__(
        self,
        scene_path: str = "res://cartpole.jscn",
        robot: str = "cartpole",
        backend: str = "mujoco",
        max_episode_steps: int = 500,
        pole_angle_limit: float = 0.35,
        cart_position_limit: float = 2.4,
    ) -> None:
        self.env = _core.RLEnvironment(scene_path, robot=robot, backend=backend)
        config = self.env.get_controller_config()
        config.controlled_joints = ["slider", "hinge"]
        config.default_action = [0.0, 0.0]
        self.env.apply_controller_config(config.to_dict())
        reward_settings = self.env.get_reward_settings()
        reward_settings["terminate_on_fall"] = False
        reward_settings["minimum_base_height"] = -1.0e6
        reward_settings["maximum_base_tilt_radians"] = 1.0e6
        self.env.set_reward_settings(reward_settings)
        self.max_episode_steps = int(max_episode_steps)
        self.pole_angle_limit = float(pole_angle_limit)
        self.cart_position_limit = float(cart_position_limit)
        self._elapsed_steps = 0

    def reset(self, seed: int = 0):
        observation, info = self.env.reset(seed=seed)
        self._elapsed_steps = 0
        return self._cartpole_observation(observation), info

    def step(self, action: Sequence[float]):
        slider_action = float(action[0]) if action else 0.0
        observation, _reward, terminated, truncated, info = self.env.step([slider_action, 0.0])
        self._elapsed_steps += 1
        cartpole_observation = self._cartpole_observation(observation)
        cart_position, _cart_velocity, pole_angle, _pole_velocity = cartpole_observation
        failed = abs(cart_position) > self.cart_position_limit or abs(pole_angle) > self.pole_angle_limit
        timed_out = self._elapsed_steps >= self.max_episode_steps
        reward = self._reward(cartpole_observation, action)
        return cartpole_observation, reward, bool(terminated or failed), bool(truncated or timed_out), info

    def get_action_size(self) -> int:
        return 1

    def get_observation_size(self) -> int:
        return 4

    def get_action_spec(self) -> dict[str, Any]:
        return {
            "version": "cartpole-v1",
            "names": ["slider_target"],
            "lower_bounds": [-1.0],
            "upper_bounds": [1.0],
            "units": ["normalized"],
        }

    def get_observation_spec(self) -> dict[str, Any]:
        return {
            "version": "cartpole-v1",
            "names": ["cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity"],
            "lower_bounds": [-self.cart_position_limit, -math.inf, -math.pi, -math.inf],
            "upper_bounds": [self.cart_position_limit, math.inf, math.pi, math.inf],
            "units": ["m", "m/s", "rad", "rad/s"],
        }

    def get_last_error(self) -> str:
        return self.env.get_last_error()

    def _cartpole_observation(self, observation: Sequence[float]) -> list[float]:
        values = [float(value) for value in observation]
        if len(values) < 17:
            return [0.0, 0.0, 0.0, 0.0]
        cart_position = values[13]
        cart_velocity = values[14]
        pole_angle = values[15]
        pole_velocity = values[16]
        return [cart_position, cart_velocity, pole_angle, pole_velocity]

    def _reward(self, observation: Sequence[float], action: Sequence[float]) -> float:
        cart_position, _cart_velocity, pole_angle, _pole_velocity = observation
        action_value = float(action[0]) if action else 0.0
        upright = max(0.0, math.cos(pole_angle))
        return 1.0 + upright - 0.1 * abs(cart_position) - 0.01 * action_value * action_value


__all__ = ["CartPoleEnv"]
