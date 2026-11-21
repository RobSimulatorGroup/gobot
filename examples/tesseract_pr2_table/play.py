"""Play the PR2 seed and the path exported by native Tesseract on one clock."""

from pathlib import Path

import gobot
import numpy as np


class Script(gobot.NodeScript):
    def _ready(self):
        self.previous_backend = self.context.backend_type
        self.context.backend_type = gobot.PhysicsBackendType.Null
        project = Path(self.context.project_path)
        with np.load(project / "generated/replay.npz") as replay:
            names = replay["links"].tolist()
            self.seed, self.solution = replay["poses"]
            hits = replay["seed_hits"].tolist()
        snapshot = self.context.compile_physics_scene_snapshot()
        paths = {link["name"]: link["path"] for robot in snapshot["robots"] for link in robot["links"]}
        self.links = [self.get_node(paths[name].removeprefix(self.node.path + "/")) for name in names]
        self.ghosts = []
        for i, link in enumerate(self.links):
            distance = np.linalg.norm(self.seed[:, i, :3] - self.solution[:, i, :3], axis=-1)
            alignment = np.abs((self.seed[:, i, 3:] * self.solution[:, i, 3:]).sum(axis=-1))
            different = (distance > 1e-5) | (alignment < 1 - 1e-6)
            if not different.any():
                continue
            ghost = self.node.add_child(gobot.create_node("Node3D", "Seed_" + names[i]))
            for child in link.children:
                if isinstance(child, gobot.MeshInstance3D):
                    mesh = child.create_visual_instance("res://seed_material.jres")
                    mesh.cast_shadow = False
                    ghost.add_child(mesh)
            self.ghosts.append((i, ghost, different))
        self.elapsed = 0
        self._process(0)
        print(f"Tesseract PR2 replay with {len(self.links)} collision links. "
              f"Seed collision segments {hits} (zero-based). Optimized collision segments [].")

    def _process(self, delta):
        last = len(self.solution) - 1
        phase = round(self.elapsed * 32) % (2 * last)
        frame = min(phase, 2 * last - phase)
        self.context.apply_link_poses(self.links, self.solution[frame])
        for i, ghost, different in self.ghosts:
            ghost.visible = bool(different[frame])
            if ghost.visible:
                pose = self.seed[frame, i]
                ghost.set_global_transform(pose[:3], (pose[6], *pose[3:6]))
        self.elapsed += delta

    def _exit_tree(self):
        self.context.backend_type = self.previous_backend
