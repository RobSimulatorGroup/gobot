"""Register task configuration, environment and runner factories independently."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class TaskSpec:
    env_cfg_factory: Callable[..., Any]
    runner_cfg_factory: Callable[..., Any]
    env_factory: Callable[..., Any]
    runner_factory: Callable[..., Any]
    play_env_cfg_factory: Callable[..., Any] | None = None
    algorithm: str = "supermimic"
    description: str = ""


_TASKS: dict[str, TaskSpec] = {}


def register_task(task_id: str, spec: TaskSpec) -> None:
    if not task_id or task_id.strip() != task_id:
        raise ValueError("task ID must be nonempty without surrounding whitespace")
    if task_id in _TASKS:
        raise ValueError(f"task already registered: {task_id}")
    _TASKS[task_id] = spec


def list_tasks() -> tuple[str, ...]:
    return tuple(sorted(_TASKS))


def get_task(task_id: str) -> TaskSpec:
    try:
        return _TASKS[task_id]
    except KeyError:
        raise ValueError(f"unknown task {task_id!r}; available: {', '.join(list_tasks())}") from None


def load_env_cfg(task_id: str, *, play: bool = False, **kwargs):
    spec = get_task(task_id)
    factory = spec.play_env_cfg_factory if play else spec.env_cfg_factory
    if factory is None:
        raise ValueError(f"task {task_id!r} has no play configuration")
    return deepcopy(factory(**kwargs))


def load_runner_cfg(task_id: str, **kwargs):
    return deepcopy(get_task(task_id).runner_cfg_factory(**kwargs))


def make_env(task_id: str, cfg, **kwargs):
    return get_task(task_id).env_factory(cfg, **kwargs)


def make_runner(task_id: str, env, cfg, **kwargs):
    return get_task(task_id).runner_factory(env, cfg, **kwargs)
