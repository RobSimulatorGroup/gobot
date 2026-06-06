from typing import Any, Mapping, Sequence

from .cfg import VelocityTaskCfg

class VelocityRuntimeState:
    robot: Mapping[str, Any]
    base: Mapping[str, Any]
    joints: Mapping[str, Mapping[str, Any]]
    links: Mapping[str, Mapping[str, Any]]
    sensors: Mapping[str, Mapping[str, Any]]
    contacts: Sequence[Mapping[str, Any]]

class GobotVelocityEnv:
    is_vector_env: bool
    num_envs: int
    num_actions: int
    num_obs: int
    num_privileged_obs: int
    cfg: dict[str, Any]
    extras: dict[str, Any]
    def __init__(
        self,
        cfg: VelocityTaskCfg | None = ...,
        *,
        num_envs: int = ...,
        device: str = ...,
        seed: int = ...,
        max_episode_length: int | None = ...,
    ) -> None: ...
    def get_observations(self): ...
    def reset(self, seed: int | None = ...): ...
    def step(self, actions): ...
    def close(self) -> None: ...
    def set_training_progress(self, policy_steps: int) -> None: ...

__all__: list[str]
