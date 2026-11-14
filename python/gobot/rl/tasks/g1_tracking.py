"""Whole-body reference tracking with PPO."""

from .registry import TaskSpec, register_task


def _training():
    from gobot.rl.tracking.training import training_spec

    return training_spec()


register_task(
    "g1-tracking",
    TaskSpec(
        training_factory=_training,
        algorithm="ppo",
        description="mjlab/BeyondMimic G1 whole-body motion tracking",
    ),
)
