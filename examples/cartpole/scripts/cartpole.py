"""Editor playback of the single-pole plan; the solver owns no scene objects."""
from __future__ import annotations

from queue import Empty, Queue
import threading

import gobot

if __package__:
    from .. import cartpole_task as task
    from ..cartpole_swingup import PlanningCancelled, solve
else:
    import cartpole_task as task
    from cartpole_swingup import PlanningCancelled, solve


def _plan_worker(artifact, cancelled, messages):
    try:
        plan = solve(artifact, cancelled=cancelled.is_set,
                     progress=lambda text: messages.put(("progress", text)))
        if not cancelled.is_set():
            messages.put(("plan", plan))
    except PlanningCancelled:
        pass
    except Exception as error:
        if not cancelled.is_set():
            messages.put(("error", str(error)))


class Script(gobot.NodeScript):
    def _ready(self):
        self.joints = task.configure_context(self.context)
        artifact = self.context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
        self.cancelled = threading.Event()
        self.messages = Queue()
        self.plan = None
        self.tick = 0
        self.started = False
        self.force = 0.0
        self.worker = threading.Thread(target=_plan_worker,
                                       args=(artifact, self.cancelled, self.messages), daemon=True)
        self.worker.start()
        print("Cartpole: solving a four-second swing-up from the hanging pose.")

    def _process(self, delta):
        del delta
        while True:
            try:
                kind, value = self.messages.get_nowait()
            except Empty:
                break
            if kind == "progress":
                print(value)
            elif kind == "error":
                raise RuntimeError(f"Cartpole planning failed: {value}")
            else:
                self.plan = value
                print(f"Cartpole: playing swing-up, cost={value.cost:.4f}.")

    def _physics_process(self, delta):
        del delta
        if not self.context.has_world:
            return
        if self.plan is None or not self.started:
            task.reset(self.joints)
            if self.plan is None:
                return
            self.started = True
        if self.tick % task.SUBSTEPS == 0:
            self.force = self.plan.force(self.tick // task.SUBSTEPS, task.state(self.joints))
        self.joints[0].set_effort_target(self.force)
        self.joints[1].set_passive()
        if self.tick == task.HORIZON * task.SUBSTEPS:
            print("Cartpole: swing-up complete; LQR balance active.")
        self.tick += 1

    def _exit_tree(self):
        if hasattr(self, "cancelled"):
            self.cancelled.set()
        self.plan = None
