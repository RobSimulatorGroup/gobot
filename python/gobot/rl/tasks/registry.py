"""Optional names for training workflows; Python factories work without a registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from gobot.rl.training import TrainingSpec


@dataclass(frozen=True)
class TaskSpec:
    training_factory: Callable[[], TrainingSpec]
    algorithm: str
    description: str
    supported_backends: tuple[str, ...] = ("mujoco-cpu", "mujoco-warp")


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


def load_training_spec(task_id: str) -> TrainingSpec:
    task = get_task(task_id)
    spec = task.training_factory()
    if not isinstance(spec, TrainingSpec):
        raise TypeError(f"training_factory for {task_id!r} must return TrainingSpec")
    if spec.supported_backends != task.supported_backends:
        raise ValueError(f"backend metadata differs between task {task_id!r} and its training workflow")
    return spec
