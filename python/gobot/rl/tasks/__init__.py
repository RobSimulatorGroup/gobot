"""Built-in tasks and registration API for user-defined training tasks."""
from .registry import (
    TaskSpec, get_task, list_tasks, load_env_cfg, load_runner_cfg,
    make_env, make_runner, register_task,
)
from . import g1_tracking as _g1_tracking

__all__ = [
    "TaskSpec", "get_task", "list_tasks", "load_env_cfg", "load_runner_cfg",
    "make_env", "make_runner", "register_task",
]
