"""Gobot Editor playback of recorded mjlab physics and reference link poses.

Uses Scene Play's runtime clone and presentation APIs only. No policy inference
or physics integration is performed by this recording player.
"""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import gobot


def walk(root):
    yield root
    for child in root.children:
        yield from walk(child)


class RecordedMotion:
    num_envs = 1
    capabilities = SimpleNamespace(name="Recorded mjlab rollout", device="CPU", graph_capture=False)

    def __init__(self, poses):
        self.poses = poses
        self.frame = 0.0
        self.speed = 1.0
        self.paused = False

    def step(self, *, nsteps=1):
        if not self.paused:
            self.frame = min(self.frame + nsteps * self.speed, len(self.poses["policy"]) - 1)

    def reset(self):
        self.frame = 0.0
        self.paused = False

    def close(self):
        pass


class Script(gobot.NodeScript):
    def _ready(self):
        self.session = None
        project = Path(self.context.project_path)
        self.report = json.loads((project / "evaluation.json").read_text())
        with np.load(project / "replay.npz") as data:
            self.poses = {name: data[name] for name in data.files}
        self.player = RecordedMotion(self.poses)
        root = self.get_root()
        self.links = {}
        for name in ("Policy", "Reference"):
            robot = root.find(name)
            robot.mode = gobot.RobotMode.Motion
            by_name = {node.name: node for node in walk(robot) if node.type_name == "Link3D"}
            self.links[name] = tuple(by_name[str(n)] for n in self.poses["link_names"])
        self.last_status = None
        self._sync()
        self.session = gobot.sim.ProviderPlaySession(
            self.context, self.player, fixed_dt=float(self.poses["step_dt"]),
            reset=self.player.reset, sync_scene=self._sync,
        ).start()
        print("mjlab recording: BLUE = trained policy; GREEN = reference. "
              "P: pause/resume, R: restart, Left/Right: seek 1s, 1/2/3: 0.25x/1x/2x. "
              "Both views are translated by the same reference pelvis XY for framing.", flush=True)

    def _process(self, delta):
        inputs = self.context.input
        if not inputs.has_control_focus:
            return
        if inputs.is_key_pressed("P"):
            self.player.paused = not self.player.paused
        if inputs.is_key_pressed("R"):
            self.session.reset()
        for key, speed in (("1", 0.25), ("2", 1.0), ("3", 2.0)):
            if inputs.is_key_pressed(key):
                self.player.speed = speed
        for key, offset in (("Left", -1.0), ("Right", 1.0)):
            if inputs.is_key_pressed(key):
                self.player.frame = float(np.clip(
                    self.player.frame + offset / float(self.poses["step_dt"]),
                    0, len(self.poses["policy"]) - 1))
                self._sync()

    def _sync(self):
        index = int(self.player.frame)
        for name, key, offset in (("Policy", "policy", -1.1), ("Reference", "reference", 1.1)):
            poses = self.poses[key][index].copy()
            # Common translation preserves every policy/reference position error.
            center = self.poses["reference"][index, 0, :2]
            poses[:, :2] -= center
            poses[:, 0] += offset
            self.context.apply_link_poses(self.links[name], poses)
        seconds = index * float(self.poses["step_dt"])
        ended = index == len(self.poses["policy"]) - 1
        state = "PAUSED" if self.player.paused else "PLAYING"
        if ended:
            state = "TERMINATED" if self.report["terminated"] else "COMPLETE"
        stamp = (index // 25, state, self.player.speed)
        if self.session is not None and stamp != self.last_status:
            self.last_status = stamp
            error = self.poses["anchor_position_error_m"][index] * 100
            self.session.set_status(f"{state} {seconds:.2f}/{self.report['reference_duration_s']:.2f}s | "
                                    f"{self.player.speed:g}x | torso error {error:.1f}cm | P pause, R restart")

    def _exit_tree(self):
        if self.session is not None:
            self.session.close()
            self.session = None
