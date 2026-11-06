"""G1 tracking shortcut; the algorithm is loaded only when selected."""
from .registry import TaskSpec, register_task


def _training():
    from gobot.rl.training.supermimic import training_spec
    return training_spec()


register_task("g1-tracking", TaskSpec(
    training_factory=_training, algorithm="supermimic",
    description="G1 SuperMimic motion tracking",
))
