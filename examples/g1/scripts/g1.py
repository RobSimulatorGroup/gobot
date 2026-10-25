"""Play a trained 29-action G1 velocity policy on the authored terrain."""
import os
from pathlib import Path
import gobot
from config import g1_velocity_cfg
from gobot.rl.locomotion.playback import VelocityPolicyPlayback

class Script(gobot.NodeScript):
    def _ready(self):
        self.playback = None
        project = Path(self.context.project_path)
        value = os.environ.get('GOBOT_G1_POLICY', 'res://policies/g1_velocity.onnx')
        policy = project / value.removeprefix('res://') if value.startswith('res://') else Path(value)
        self.playback = VelocityPolicyPlayback(self.context, self.get_root(),
            g1_velocity_cfg(project_path=project, play=True), policy)
    def _physics_process(self, delta):
        inputs = self.context.input
        if inputs.is_key_pressed('R'):
            self.playback.session.reset()
        if inputs.has_control_focus:
            self.playback.command = (float(inputs.is_key_held('W')) - float(inputs.is_key_held('S')),
                float(inputs.is_key_held('Q')) - float(inputs.is_key_held('E')),
                .5 * (float(inputs.is_key_held('A')) - float(inputs.is_key_held('D'))))
        else:
            self.playback.command = (0., 0., 0.)
    def _exit_tree(self):
        if self.playback is not None:
            self.playback.close()
            self.playback = None
