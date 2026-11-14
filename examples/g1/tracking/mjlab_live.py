"""Run native mjlab inference/physics on a worker; render its state in Gobot."""
from pathlib import Path
import json
import numpy as np
import gobot


def walk(root):
    yield root
    for child in root.children:
        yield from walk(child)


class Script(gobot.NodeScript):
    def _ready(self):
        self.session = self.runtime = None
        self.pending_push = False
        self.contact_snapshot = None
        self.contact_offset = np.zeros(3)
        self.last_debug_settings = None
        self.last_second = -1
        project = Path(self.context.project_path)
        settings = json.loads((project / "live.json").read_text())
        self.duration = settings["duration"]
        self.links = {}
        for name in ("Policy", "Reference"):
            robot = self.get_root().find(name)
            robot.mode = gobot.RobotMode.Motion
            nodes = {n.name: n for n in walk(robot) if n.type_name == "Link3D"}
            self.links[name] = tuple(nodes[n] for n in settings["link_names"])
        recipe = gobot.sim.SimulationRuntimeSpec("mjlab_live_runtime:create", settings["runtime"])
        self.runtime = gobot.sim.AsyncSimulationSession(
            recipe, fields=("policy", "reference", "status", "contacts"), environments=(0,), max_hz=50)
        self.session = gobot.sim.ProviderPlaySession(
            self.context, self.runtime, fixed_dt=0.02, max_sub_steps=1,
            before_step=self._commands, sync_scene=self._sync,
        ).start()
        self.session.set_status("Loading LIVE mjlab checkpoint and MuJoCo Warp")
        print("LIVE inference: BLUE policy + physical ground contact; GREEN kinematic reference. "
              "Toolbar: play/pause/stop/step. R: reset. F: horizontal velocity impulse.", flush=True)

    def _commands(self, dt):
        if self.pending_push:
            self.pending_push = False
            return {"push_velocity": np.array([[0.8, 0.0, 0.0]], dtype=np.float32)}
        return {}

    def _process(self, delta):
        # Refresh toggles between snapshot publications while Play is running.
        self._refresh_contacts()
        inputs = self.context.input
        if self.runtime is None or not self.runtime.ready or not inputs.has_control_focus:
            return
        if inputs.is_key_pressed("R"):
            self.contact_snapshot = None
            gobot.render.clear_debug_arrows()
            self.session.reset()
            self.last_second = -1
        if inputs.is_key_pressed("F"):
            self.pending_push = True

    def _sync(self, snapshot):
        if "policy" not in snapshot.fields:
            return
        reference = np.asarray(snapshot.buffer("reference"))[0]
        center = reference[0, :2].copy()
        for name, field, offset in (("Policy", "policy", -1.1), ("Reference", "reference", 1.1)):
            poses = np.asarray(snapshot.buffer(field))[0].copy()
            poses[:, :2] -= center
            poses[:, 0] += offset
            self.context.apply_link_poses(self.links[name], poses)
        if "contacts" in snapshot.fields:
            self.contact_snapshot = np.asarray(snapshot.buffer("contacts"))[0].copy()
            self.contact_offset[:] = (-center[0] - 1.1, -center[1], 0)
            self._refresh_contacts(force=True)
        count, seconds, error, terminated, completed, action_rms, pushes, contacts, left_fz, right_fz = np.asarray(snapshot.buffer("status"))[0]
        second = int(seconds)
        if second != self.last_second or terminated or completed:
            self.last_second = second
            state = "TERMINATED" if terminated else "COMPLETE" if completed else "LIVE"
            self.session.set_status(f"{state} {seconds:.2f}/{self.duration:.2f}s | "
                f"PPO steps {int(count)} | error {error * 100:.1f}cm | "
                f"action RMS {action_rms:.2f} | pushes {int(pushes)} | "
                f"contacts {int(contacts)} | foot Fz L {left_fz:.1f} N / R {right_fz:.1f} N")

    def _refresh_contacts(self, force=False):
        if self.contact_snapshot is None:
            return
        settings = self.context.get_physics_debug_settings()
        if force or settings != self.last_debug_settings:
            from mjlab_contacts import contact_arrows
            gobot.render.set_debug_arrows(contact_arrows(
                self.contact_snapshot, offset=self.contact_offset, settings=settings))
            self.last_debug_settings = dict(settings)

    def _exit_tree(self):
        gobot.render.clear_debug_arrows()
        if self.session is not None:
            self.session.close()
            self.session = None
        elif self.runtime is not None:
            self.runtime.close()
        self.runtime = None
