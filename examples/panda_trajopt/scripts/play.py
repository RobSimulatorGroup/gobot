"""Solve from the live scene and replay the validated joint trajectory in Play."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

import gobot
from gobot import render


SEED_COLOR = (1.0, 0.35, 0.67, 1.0)
PLAN_COLOR = (0.10, 0.95, 0.90, 1.0)


def path_segments(points, color):
    return [render.DebugArrow(a, b - a, color=color, show_head=False)
            for a, b in zip(points, points[1:]) if np.any(a != b)]


def create_seed_visuals(parent, links):
    group = parent.add_child(gobot.create_node("Node3D", "InitialTrajectory"))
    frames = []
    for link in links:
        frame = group.add_child(gobot.create_node("Node3D", link.name))
        for child in link.children:
            if isinstance(child, gobot.MeshInstance3D):
                visual = child.create_visual_instance("res://seed_material.jres")
                visual.cast_shadow = False
                frame.add_child(visual)
        frames.append(frame)
    return frames


class Script(gobot.NodeScript):
    def _ready(self):
        self.worker = None
        self.folder = tempfile.TemporaryDirectory(prefix="gobot-panda-")
        self.poses = None
        snapshot = self.context.compile_physics_scene_snapshot()
        robot = next(robot for robot in snapshot["robots"] if robot["name"] == "panda")
        self.links_by_name = {
            link["name"]: self.get_node(link["path"].removeprefix(self.node.path + "/"))
            for link in robot["links"]
        }
        self.previous_backend = self.context.backend_type
        self.context.backend_type = gobot.PhysicsBackendType.Null
        request = Path(self.folder.name) / "request.json"
        self.result = request.with_name("result.npz")
        request.write_text(json.dumps({
            "snapshot": snapshot,
            "start": self.get_node("Start").position,
            "goal": self.get_node("Goal").position,
        }, default=lambda array: array.tolist()))
        script = Path(self.context.project_path) / "scripts" / "plan.py"
        self.worker = subprocess.Popen([sys.executable, str(script), str(request), str(self.result)])
        print("Panda TrajOpt: planning from Start to Goal...")

    def _process(self, delta):
        if self.worker is not None:
            status = self.worker.poll()
            if status is None:
                return
            self.worker = None
            if status:
                raise RuntimeError(f"Panda planner exited with status {status}")
            with np.load(self.result) as plan:
                if "error" in plan:
                    print(f"Panda TrajOpt failed: {plan['error']}")
                    return
                hits = str(plan["seed_hits"]) or "none"
                clearance_mm = 1000 * float(plan["world_clearance"])
                self.links = [self.links_by_name[name] for name in plan["links"]]
                self.seed_poses, self.poses = plan["poses"]
                self.dt = float(plan["dt"])
                seed_path, optimized_path = plan["paths"]
            self.seed_frames = create_seed_visuals(self.node, self.links)
            # Hide coincident surfaces, including at the common start and goal.
            separation = np.linalg.norm(self.seed_poses[..., :3] - self.poses[..., :3], axis=-1)
            alignment = np.abs((self.seed_poses[..., 3:] * self.poses[..., 3:]).sum(axis=-1))
            self.seed_visible = (separation > 1e-5) | (alignment < 1 - 1e-6)
            render.set_debug_arrows(path_segments(seed_path, SEED_COLOR)
                                    + path_segments(optimized_path, PLAN_COLOR))
            self.elapsed = 0.0
            print(f"Panda TrajOpt (Coal): {len(optimized_path)} waypoints. Seed intersections: {hits}. "
                  f"Optimized sampled world clearance: {clearance_mm:.1f} mm. Translucent pink arm and path = seed. "
                  "Solid arm and cyan path = optimized. Playing at quarter speed.")
        if self.poses is None:
            return
        # Hold at each endpoint before following the same path in reverse.
        last = len(self.poses) - 1
        duration = last * self.dt
        hold = 0.30
        phase = self.elapsed % (2 * (duration + hold))
        time = min(duration, max(0.0, min(phase, 2 * duration + hold - phase)))
        frame = round(time / self.dt)
        self.context.apply_link_poses(self.links, self.poses[frame])
        for node, pose, visible in zip(self.seed_frames, self.seed_poses[frame],
                                       self.seed_visible[frame], strict=True):
            node.visible = bool(visible)
            if visible:
                node.set_global_transform(pose[:3], (pose[6], *pose[3:6]))
        self.elapsed += 0.25 * delta

    def _exit_tree(self):
        if self.worker is not None:
            self.worker.kill()
            self.worker.wait()
        self.folder.cleanup()
        render.clear_debug_arrows()
        if hasattr(self, "previous_backend"):
            self.context.backend_type = self.previous_backend
