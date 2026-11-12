"""Optional built-in task names and registration for training workflows."""
from .registry import TaskSpec, get_task, list_tasks, load_training_spec, register_task
from . import g1_tracking as _g1_tracking
from . import g1_supermimic_official as _g1_supermimic_official
from . import velocity as _velocity

__all__ = ["TaskSpec", "get_task", "list_tasks", "load_training_spec", "register_task"]
