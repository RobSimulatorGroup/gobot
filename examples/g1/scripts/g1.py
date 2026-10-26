"""Play a trained 29-action G1 velocity policy on the authored terrain."""
import os
from pathlib import Path
import gobot
from config import g1_amp_cfg
from gobot.rl.locomotion.playback import VelocityPolicyPlayback

class Script(gobot.NodeScript):
    def _ready(self):
        self.playback = None
        self.running = False
        project = Path(self.context.project_path)
        value = os.environ.get('GOBOT_G1_POLICY', 'res://policies/g1_velocity.onnx')
        policy = project / value.removeprefix('res://') if value.startswith('res://') else Path(value)
        self.playback = VelocityPolicyPlayback(self.context, self.get_root(),
            g1_amp_cfg(project_path=project, stage="mixed", play=True), policy)
    def _physics_process(self, delta):
        inputs = self.context.input
        if inputs.is_key_pressed('R'):
            self.playback.session.reset()
        if inputs.is_key_pressed('T'):
            self.running = not self.running
        if inputs.has_control_focus:
            self.playback.command = ((2. if self.running else 1.) * float(inputs.is_key_held('W')) - .3 * float(inputs.is_key_held('S')),
                .3 * (float(inputs.is_key_held('Q')) - float(inputs.is_key_held('E'))),
                .6 * (float(inputs.is_key_held('A')) - float(inputs.is_key_held('D'))))
        else:
            self.playback.command = (0., 0., 0.)
    def _exit_tree(self):
        if self.playback is not None:
            self.playback.close()
            self.playback = None
