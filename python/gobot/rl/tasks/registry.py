"""Task registry for Gobot RL training entrypoints."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from .. import TaskConfig
from ..rsl_rl import RslRlBaseRunnerCfg


EnvBuilder = Callable[[Any], TaskConfig]


@dataclass
class _TaskEntry:
    env_cfg: Any
    rl_cfg: RslRlBaseRunnerCfg
    env_builder: EnvBuilder
    runner_cls: type | None = None


_REGISTRY: dict[str, _TaskEntry] = {}


def register_gobot_task(
    task_id: str,
    *,
    env_cfg: Any,
    rl_cfg: RslRlBaseRunnerCfg,
    env_builder: EnvBuilder,
    runner_cls: type | None = None,
) -> None:
    if task_id in _REGISTRY:
        raise ValueError(f"Gobot RL task {task_id!r} is already registered")
    _REGISTRY[task_id] = _TaskEntry(env_cfg, rl_cfg, env_builder, runner_cls)


def list_tasks() -> list[str]:
    return sorted(_REGISTRY.keys())


def load_env_cfg(task_id: str) -> Any:
    return deepcopy(_REGISTRY[task_id].env_cfg)


def load_rl_cfg(task_id: str) -> RslRlBaseRunnerCfg:
    return deepcopy(_REGISTRY[task_id].rl_cfg)


def load_env_builder(task_id: str) -> EnvBuilder:
    return _REGISTRY[task_id].env_builder


def load_runner_cls(task_id: str) -> type | None:
    return _REGISTRY[task_id].runner_cls


__all__ = [
    "EnvBuilder",
    "list_tasks",
    "load_env_builder",
    "load_env_cfg",
    "load_rl_cfg",
    "load_runner_cls",
    "register_gobot_task",
]
