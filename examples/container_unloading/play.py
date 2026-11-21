"""Compare the straight joint seed and the optimized loaded motion in editor Play."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

import gobot


def seed_visuals(root, links):
    group = root.add_child(gobot.create_node("Node3D", "InitialTrajectory"))
    frames = []
    for link in links:
        frame = group.add_child(gobot.create_node("Node3D", link.name))
        for child in link.children:
            if isinstance(child, gobot.MeshInstance3D):
                mesh = child.create_visual_instance("res://seed_material.jres")
                mesh.cast_shadow = False
                frame.add_child(mesh)
        frames.append(frame)
    return frames


class Script(gobot.NodeScript):
    def _ready(self):
        self.poses = None
        self.worker = None
        self.folder = tempfile.TemporaryDirectory(prefix="gobot-unload-")
        snapshot = self.context.compile_physics_scene_snapshot()
        robot = next(robot for robot in snapshot["robots"] if robot["name"] == "M10iA")
        self.links_by_name = {
            link["name"]: self.get_node(link["path"].removeprefix(self.node.path + "/"))
            for link in robot["links"]
        }
        self.previous_backend = self.context.backend_type
        self.context.backend_type = gobot.PhysicsBackendType.Null
        pick, drop = self.get_node("Pick"), self.get_node("Drop")
        request = Path(self.folder.name) / "scene.json"
        self.result = request.with_name("plan.npz")
        request.write_text(json.dumps({
            "snapshot": snapshot,
            "pick": [pick.position, pick.rotation_degrees],
            "drop": [drop.position, drop.rotation_degrees],
        }, default=lambda array: array.tolist()))
        worker = Path(self.context.project_path) / "plan.py"
        self.worker = subprocess.Popen([sys.executable, str(worker), str(request), str(self.result)])
        print("Planning the loaded FANUC motion between the stacks and the conveyor...")

    def _load_plan(self):
        with np.load(self.result) as plan:
            if "error" in plan:
                print(f"Container TrajOpt failed: {plan['error']}")
                return
            self.links = [self.links_by_name[name] for name in plan["names"]]
            self.seed, self.poses = plan["poses"]
            self.dt = float(plan["dt"])
            self.handoff = plan["handoff"]
            seed_mm, optimized_mm = 1000 * plan["carton_clearance"]
            seed_pair, plan_pair = (str(path).split("/")[-2] for path in plan["nearest"])
        self.ghost = seed_visuals(self.node, self.links)
        separation = np.linalg.norm(self.seed[..., :3] - self.poses[..., :3], axis=-1)
        alignment = np.abs((self.seed[..., 3:] * self.poses[..., 3:]).sum(axis=-1))
        self.visible = (separation > 1e-5) | (alignment < 1 - 1e-6)
        self.elapsed = -0.40  # Give the viewer a moment to see the common pickup pose.
        self.last_frame = -1
        print(f"Minimum sampled carton clearance over {len(self.poses)} poses. "
              f"Translucent seed {seed_mm:+.1f} mm ({seed_pair}). "
              f"Solid optimized arm {optimized_mm:+.1f} mm ({plan_pair}).")

    def _process(self, delta):
        if self.worker is not None:
            status = self.worker.poll()
            if status is None:
                return
            self.worker = None
            if status:
                raise RuntimeError(f"Container planner exited with status {status}")
            self._load_plan()
        if self.poses is None:
            return

        last = len(self.poses) - 1
        frame = min(last, max(0, round(self.elapsed / self.dt)))
        if frame != self.last_frame:
            self.context.apply_link_poses(self.links, self.poses[frame])
            for node, pose, visible in zip(self.ghost, self.seed[frame], self.visible[frame], strict=True):
                node.visible = bool(visible)
                if visible:
                    node.set_global_transform(pose[:3], (pose[6], *pose[3:6]))
            self.last_frame = frame

        release_time = self.elapsed - last * self.dt
        if release_time > 0:
            if not len(self.handoff):
                print("Loaded motion complete. Drop is outside the conveyor handoff region.")
                self.poses = None
                return
            # Only the carton moves after release. Stop with its full footprint on the rollers.
            settle, travel = self.handoff
            settle_time = 0.15
            travel_fraction = max(0.0, (release_time - settle_time) * 0.55 / max(np.linalg.norm(travel), 1e-9))
            pose = self.poses[-1, -1]
            center = pose[:3] + min(1.0, release_time / settle_time) * settle + min(1.0, travel_fraction) * travel
            self.links_by_name["Parcel"].set_global_transform(center, (pose[6], *pose[3:6]))
            if travel_fraction >= 1:
                print("Unload complete. The released carton reached the end of the rollers.")
                self.poses = None
        self.elapsed += 0.5 * delta

    def _exit_tree(self):
        if self.worker is not None:
            self.worker.kill()
            self.worker.wait()
        self.folder.cleanup()
        if hasattr(self, "previous_backend"):
            self.context.backend_type = self.previous_backend
