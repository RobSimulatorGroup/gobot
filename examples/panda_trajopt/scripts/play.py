"""Solve from the live scene in a background thread and replay the retimed trajectory in Play."""

from __future__ import annotations

from pathlib import Path
import runpy
import threading

import numpy as np
from scipy.spatial.transform import Rotation

import gobot
from gobot import render

_plan = runpy.run_path(str(Path(__file__).with_name("plan.py")))
HAND, solve = _plan["HAND"], _plan["solve"]


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


def sample_piecewise_linear(waypoints: np.ndarray, fraction: float) -> np.ndarray:
    u = float(np.clip(fraction, 0.0, 1.0)) * (len(waypoints) - 1)
    idx = min(int(u), len(waypoints) - 2)
    alpha = u - idx
    return (1.0 - alpha) * waypoints[idx] + alpha * waypoints[idx + 1]


def fk_poses_xyzw(world, q_batch: np.ndarray, link_ids: list[int]) -> np.ndarray:
    pos, rot = world.forward_kinematics(q_batch)
    pos_sel = pos[:, link_ids, :]
    rot_sel = rot[:, link_ids, :, :]
    quat = Rotation.from_matrix(rot_sel.reshape(-1, 3, 3)).as_quat().reshape(*pos_sel.shape[:-1], 4)
    return np.concatenate((pos_sel, quat), axis=-1).astype(np.float32)


class Script(gobot.NodeScript):
    def _ready(self):
        self.plan_out = None
        self.plan_error = None
        self.plan_ready = threading.Event()
        self.seed_frames = None
        self.link_ids = []
        self.links = []

        snapshot = self.context.compile_physics_scene_snapshot()
        robot = next(robot for robot in snapshot["robots"] if robot["name"] == "panda")
        self.links_by_name = {
            link["name"]: self.get_node(link["path"].removeprefix(self.node.path + "/"))
            for link in robot["links"]
        }
        self.previous_backend = self.context.backend_type
        self.context.backend_type = gobot.PhysicsBackendType.Null

        start = np.asarray(self.get_node("Start").position, dtype=np.float64)
        goal = np.asarray(self.get_node("Goal").position, dtype=np.float64)
        inspection = np.asarray(self.get_node("Inspection").position, dtype=np.float64)

        def _worker():
            try:
                self.plan_out = solve(snapshot, start, goal, inspection, backend="coal")
            except Exception as error:
                self.plan_error = str(error)
            finally:
                self.plan_ready.set()

        self.worker = threading.Thread(target=_worker, daemon=True)
        self.worker.start()
        print("Panda TrajOpt: planning through Inspection to Goal in background thread...")

    def _finish_planning(self):
        if self.plan_error is not None:
            print(f"Panda TrajOpt failed: {self.plan_error}")
            self.plan_error = None
            return
        out = self.plan_out
        world = out["world"]
        planner = out["planner"]
        result = out["result"]
        diag = result.diagnostics

        self.world = world
        self.result = result
        self.links = [self.links_by_name[name] for name in out["links"]]
        self.link_ids = [world.links[("panda", name)] for name in out["links"]]
        self.seed_frames = create_seed_visuals(self.node, self.links)
        planner.debug_draw(render, result, link=HAND, samples=96, draw_witnesses=True)
        self.elapsed = 0.0

        hits = out["seed_hits"] or "none"
        clearance_mm = 1000.0 * out["world_clearance"]
        inspection_mm = 1000.0 * out["inspection_error"]
        inspection_deg = np.degrees(out["inspection_angle"])
        print(
            f"Panda TrajOpt (C++ SQP / Coal, {diag.total_ms:.1f} ms, {diag.qp_calls} QPs, "
            f"strategy={diag.seed_strategy}): duration={result.trajectory.duration:.2f} s. "
            f"Seed hits: {hits}. Inspection error: {inspection_mm:.3f} mm, {inspection_deg:.3f} deg. "
            f"World clearance: {clearance_mm:.1f} mm. "
            "Pink = seed, Cyan = retimed trajectory, Red = collision witnesses."
        )

    def _process(self, delta):
        if self.seed_frames is None:
            if not self.plan_ready.is_set():
                return
            self._finish_planning()
            if self.seed_frames is None:
                return

        traj = self.result.trajectory
        duration = max(traj.duration, 1e-6)
        hold = 0.35
        phase = self.elapsed % (2.0 * (duration + hold))
        time = min(duration, max(0.0, min(phase, 2.0 * duration + hold - phase)))

        q_opt, _, _ = traj.sample(time)
        q_seed = sample_piecewise_linear(self.result.initial_positions, time / duration)
        seed_poses, opt_poses = fk_poses_xyzw(self.world, np.stack((q_seed, q_opt), axis=0), self.link_ids)

        self.context.apply_link_poses(self.links, opt_poses)
        sep = np.linalg.norm(seed_poses[:, :3] - opt_poses[:, :3], axis=-1)
        align = np.abs((seed_poses[:, 3:] * opt_poses[:, 3:]).sum(axis=-1))
        visible = (sep > 1e-5) | (align < 1.0 - 1e-6)
        for node, pose, show in zip(self.seed_frames, seed_poses, visible, strict=True):
            node.visible = bool(show)
            if show:
                node.set_global_transform(pose[:3], (pose[6], *pose[3:6]))

        self.elapsed += 0.5 * delta

    def _exit_tree(self):
        render.clear_debug_arrows()
        if hasattr(self, "previous_backend"):
            self.context.backend_type = self.previous_backend
