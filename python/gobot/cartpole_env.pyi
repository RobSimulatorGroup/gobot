from __future__ import annotations

from typing import Any, Sequence


class CartPoleEnv:
    env: Any
    max_episode_steps: int
    pole_angle_limit: float
    cart_position_limit: float

    def __init__(
        self,
        scene_path: str = "res://cartpole.jscn",
        robot: str = "cartpole",
        backend: str = "mujoco",
        max_episode_steps: int = 500,
        pole_angle_limit: float = 0.35,
        cart_position_limit: float = 2.4,
    ) -> None: ...
    def reset(self, seed: int = 0): ...
    def step(self, action: Sequence[float]): ...
    def get_action_size(self) -> int: ...
    def get_observation_size(self) -> int: ...
    def get_action_spec(self) -> dict[str, Any]: ...
    def get_observation_spec(self) -> dict[str, Any]: ...
    def get_last_error(self) -> str: ...


__all__: list[str]
