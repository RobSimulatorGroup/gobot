"""SuperMimic GenCoord task registration."""

from .registry import TaskSpec, register_task


def _training():
    from gobot.rl.supermimic.training import training_spec

    return training_spec()


register_task(
    "g1-supermimic",
    TaskSpec(
        training_factory=_training,
        algorithm="supermimic",
        description="Official GenCoord G1 motion tracking",
        supported_backends=("mujoco-cpu", "mujoco-warp"),
    ),
)
