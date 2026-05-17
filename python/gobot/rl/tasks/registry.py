"""Task registry for Gobot RL training entrypoints."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

from .. import TaskConfig
from ..rsl_rl import RslRlBaseRunnerCfg, rsl_rl_cfg_to_dataclass


@dataclass
class _TaskEntry:
    env_cfg: type | object
    env_builder: Callable[[type | object], TaskConfig] | None
    play_env_cfg: type | object | None
    rl_cfg: RslRlBaseRunnerCfg | type | object
    runner_cls: type | None = None


_REGISTRY: dict[str, _TaskEntry] = {}


def register_gobot_task(
    task_id: str,
    *,
    env_cfg: type | object,
    env_builder: Callable[[type | object], TaskConfig] | None = None,
    play_env_cfg: type | object | None = None,
    rl_cfg: RslRlBaseRunnerCfg | type | object,
    runner_cls: type | None = None,
) -> None:
    if task_id in _REGISTRY:
        raise ValueError(f"Gobot RL task {task_id!r} is already registered")
    _REGISTRY[task_id] = _TaskEntry(env_cfg, env_builder, play_env_cfg, rl_cfg, runner_cls)


def list_tasks() -> list[str]:
    return sorted(_REGISTRY.keys())


def load_env_cfg(task_id: str, play: bool = False) -> TaskConfig:
    entry = _REGISTRY[task_id]
    cfg = entry.play_env_cfg if play and entry.play_env_cfg is not None else entry.env_cfg
    if isinstance(cfg, TaskConfig):
        return deepcopy(cfg)
    if entry.env_builder is None:
        raise TypeError(f"Gobot RL task {task_id!r} has no env_builder for class-style env config")
    return entry.env_builder(cfg)


def load_rl_cfg(task_id: str) -> RslRlBaseRunnerCfg:
    return rsl_rl_cfg_to_dataclass(deepcopy(_REGISTRY[task_id].rl_cfg))


def load_runner_cls(task_id: str) -> type | None:
    return _REGISTRY[task_id].runner_cls


__all__ = [
    "list_tasks",
    "load_env_cfg",
    "load_rl_cfg",
    "load_runner_cls",
    "register_gobot_task",
]
