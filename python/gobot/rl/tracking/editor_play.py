"""Scene-owned display bindings for live tracking; the worker owns inference."""
import json
from pathlib import Path

import numpy as np
import gobot


def walk(root):
    yield root
    for child in root.children:
        yield from walk(child)


class Script(gobot.NodeScript):
    def _ready(self):
        self.session = self.runtime = None
        self.last_second = -1
        settings = json.loads((Path(self.context.project_path) / "play.json").read_text())
        self.roots = {name: self.get_root().find(name) for name in ("Policy", "Reference")}
        self.links = {}
        for name, root in self.roots.items():
            nodes = {node.name: node for node in walk(root) if node.type_name == "Link3D"}
            self.links[name] = tuple(nodes[key] for key in settings["link_names"])
        runtime = dict(settings["runtime"])
        for key in ("checkpoint", "project_path"):
            if str(runtime.get(key, "")).startswith("res://"):
                runtime[key] = str(Path(self.context.project_path) / runtime[key][6:])
        recipe = gobot.sim.SimulationRuntimeSpec("gobot.rl.tracking.playback:create", runtime)
        self.runtime = gobot.sim.AsyncSimulationSession(
            recipe, fields=("policy", "reference", "status"), environments=(0,), max_hz=50)
        self.session = gobot.sim.ProviderPlaySession(
            self.context, self.runtime, fixed_dt=settings["control_dt"], max_sub_steps=1,
            sync_scene=self._sync, contact_roots={0: self.roots["Policy"]}).start()
        self.session.set_status("Loading tracking policy")

    def _sync(self, snapshot):
        if "policy" not in snapshot.fields:
            return
        target = np.asarray(snapshot.buffer("reference"))[0]
        center = target[0, :2]
        for name, field, offset in (("Policy", "policy", -1.1), ("Reference", "reference", 1.1)):
            translation = np.array([-center[0] + offset, -center[1], 0])
            self.roots[name].position = translation
            poses = np.asarray(snapshot.buffer(field))[0].copy()
            poses[:, :3] += translation
            self.context.apply_link_poses(self.links[name], poses)
        steps, seconds, duration, error, finished, failed = np.asarray(snapshot.buffer("status"))[0]
        if int(seconds) != self.last_second or finished:
            self.last_second = int(seconds)
            state = "Failed" if failed else "Completed" if finished else "Running"
            self.session.set_status(f"{state} {seconds:.2f}/{duration:.2f}s | "
                f"policy steps {int(steps)} | torso error {error * 100:.1f}cm")

    def _exit_tree(self):
        if self.session is not None:
            self.session.close()
        elif self.runtime is not None:
            self.runtime.close()
        self.session = self.runtime = None
