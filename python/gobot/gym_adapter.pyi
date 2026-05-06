from __future__ import annotations

from typing import Any, Sequence


class GobotBox:
    low: list[float]
    high: list[float]
    shape: tuple[int]
    names: list[str]
    units: list[str]

    def __init__(
        self,
        low: Sequence[float],
        high: Sequence[float],
        names: Sequence[str] | None = None,
        units: Sequence[str] | None = None,
    ) -> None: ...
    def sample(self) -> list[float]: ...


def space_from_spec(spec: dict[str, Any]) -> Any: ...


class GobotGymEnv:
    env: Any
    observation_spec: dict[str, Any]
    action_spec: dict[str, Any]
    observation_space: Any
    action_space: Any

    def __init__(
        self,
        scene_path: str = "",
        robot: str = "robot",
        backend: str = "null",
        env: Any | None = None,
    ) -> None: ...
    def reset(self, seed: int | None = None, options: Any | None = None) -> tuple[list[float], dict[str, Any]]: ...
    def step(self, action: Sequence[float]) -> tuple[list[float], float, bool, bool, dict[str, Any]]: ...
    def close(self) -> None: ...
