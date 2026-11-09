"""Convenience names for Python-configurable velocity PPO/AMP workflows."""
from functools import partial
from .registry import TaskSpec, register_task


def _training(robot, recipe):
    from gobot.rl.training.projects import example_module
    return example_module("locomotion.presets").training_spec(robot=robot, recipe=recipe)


for _name, _robot, _recipe, _algorithm, _description in (
    ("go1-velocity", "go1", "baseline", "ppo", "Go1 standard velocity PPO; flat or rough terrain"),
    ("go1-gait", "go1", "gait", "ppo", "Go1 velocity with walk/trot/gallop phase rewards"),
    ("g1-velocity", "g1", "baseline", "ppo", "G1 standard velocity PPO; flat or rough terrain"),
    ("g1-amp", "g1", "amp", "amp", "G1 velocity with a motion prior"),
):
    register_task(_name, TaskSpec(
        training_factory=partial(_training, _robot, _recipe),
        algorithm=_algorithm, description=_description,
    ))
