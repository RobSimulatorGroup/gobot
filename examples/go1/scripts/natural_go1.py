"""Play an explicitly selected, manifest-backed natural Go1 candidate."""
import os
from pathlib import Path
import sys
import gobot
from gobot.rl.locomotion.playback import VelocityPolicyPlayback


class Script(gobot.NodeScript):
    def _ready(self):
        self.playback = None
        self.speed_index = 0
        project = Path(self.context.project_path)
        # Example configurations ship next to the project, outside the engine package.
        example_root = str(project.parents[1])
        if example_root not in sys.path:
            sys.path.insert(0, example_root)
        from examples.locomotion.experiment import Experiment, task_config
        value = os.environ.get('GOBOT_GO1_POLICY', 'res://policies/go1_natural.onnx')
        policy = project/value.removeprefix('res://') if value.startswith('res://') else Path(value)
        cfg = task_config(Experiment(robot='go1', recipe='natural', stage='mixed'), play=True)
        cfg.project_path = project
        self.playback = VelocityPolicyPlayback(self.context, self.get_root(), cfg, policy)

    def _physics_process(self, delta):
        inputs = self.context.input
        if inputs.is_key_pressed('R'):
            self.playback.session.reset()
        if inputs.is_key_pressed('T'):
            self.speed_index = (self.speed_index + 1) % 3
        if inputs.has_control_focus and not inputs.is_key_held('Space'):
            speed = (.5, 1.4, 2.8)[self.speed_index]
            self.playback.command = (
                speed*float(inputs.is_key_held('W'))-.3*float(inputs.is_key_held('S')),
                .3*(float(inputs.is_key_held('Q'))-float(inputs.is_key_held('E'))),
                .6*(float(inputs.is_key_held('A'))-float(inputs.is_key_held('D'))))
        else:
            self.playback.command = (0., 0., 0.)

    def _exit_tree(self):
        if self.playback is not None:
            self.playback.close()
            self.playback = None
