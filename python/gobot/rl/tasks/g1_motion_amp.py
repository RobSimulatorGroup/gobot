"""Optional motion AMP recipe, imported lazily when selected."""
from .registry import TaskSpec, register_task


def _training():
    from gobot.rl.motion_amp.training import training_spec
    return training_spec()


register_task('g1-motion-amp', TaskSpec(training_factory=_training, algorithm='motion-amp',
    description='G1 motion imitation with RSL-RL PPO and Gobot AMP'))
