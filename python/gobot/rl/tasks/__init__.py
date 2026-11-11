"""Optional built-in task names and registration for training workflows."""
from .registry import TaskSpec, get_task, list_tasks, load_training_spec, register_task
from . import g1_tracking as _g1_tracking
from . import velocity as _velocity
from . import g1_motion_amp as _g1_motion_amp

__all__ = ["TaskSpec", "get_task", "list_tasks", "load_training_spec", "register_task"]
