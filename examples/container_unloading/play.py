"""Compare the straight joint seed and the retimed optimized loaded motion in editor Play."""

from __future__ import annotations

from pathlib import Path
import runpy
import threading

import numpy as np
from scipy.spatial.transform import Rotation

import gobot
from gobot import render

_plan = runpy.run_path(str(Path(__file__).with_name("plan.py")))
PAYLOAD, ROBOT, solve = _plan["PAYLOAD"], _plan["ROBOT"], _plan["solve"]


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
        self.active = False

        snapshot = self.context.compile_physics_scene_snapshot()
        robot = next(robot for robot in snapshot["robots"] if robot["name"] == ROBOT)
        self.links_by_name = {
            link["name"]: self.get_node(link["path"].removeprefix(self.node.path + "/"))
            for link in robot["links"]
        }
        self.previous_backend = self.context.backend_type
        self.context.backend_type = gobot.PhysicsBackendType.Null

        pick_node, drop_node = self.get_node("Pick"), self.get_node("Drop")
        pick = (np.asarray(pick_node.position, dtype=np.float64), np.asarray(pick_node.rotation_degrees, dtype=np.float64))
        drop = (np.asarray(drop_node.position, dtype=np.float64), np.asarray(drop_node.rotation_degrees, dtype=np.float64))

        def _worker():
            try:
                self.plan_out = solve(snapshot, pick, drop, backend="coal", first_feasible=True, retime=True)
            except Exception as error:
                self.plan_error = str(error)
            finally:
                self.plan_ready.set()

        self.worker = threading.Thread(target=_worker, daemon=True)
        self.worker.start()
        print("Planning the loaded FANUC motion between the stacks and the conveyor...")

    def _finish_planning(self):
        if self.plan_error is not None:
            print(f"Container TrajOpt failed: {self.plan_error}")
            self.plan_error = None
            return
        out = self.plan_out
        self.world = out["world"]
        self.result = out["result"]
        self.handoff = out["handoff"]
        self.links = [self.links_by_name[name] for name in out["names"]]
        self.link_ids = [self.world.links[(ROBOT, name)] for name in out["names"]]
        self.ghost = seed_visuals(self.node, self.links)
        out["planner"].debug_draw(render, self.result, link=PAYLOAD, samples=84, draw_witnesses=True)

        # Precompute final drop pose for post-release conveyor animation.
        end_p, end_r = self.world.link_pose(self.result.positions[-1], PAYLOAD)
        self.drop_pose = np.concatenate((end_p, Rotation.from_matrix(end_r).as_quat()))
        self.elapsed = -0.35  # Hold briefly at the common pickup pose.
        self.done_arm = False
        self.active = True

        diag = self.result.diagnostics
        seed_mm, optimized_mm = 1000.0 * out["carton_clearance"]
        seed_pair, plan_pair = (str(path).split("/")[-2] for path in out["nearest"])
        print(
            f"Container TrajOpt (C++ SQP / Coal, {diag.total_ms:.1f} ms, {diag.qp_calls} QPs, "
            f"duration={self.result.trajectory.duration:.2f} s). "
            f"Min carton clearance: seed {seed_mm:+.1f} mm ({seed_pair}), "
            f"optimized {optimized_mm:+.1f} mm ({plan_pair})."
        )

    def _process(self, delta):
        if not self.active:
            if not self.plan_ready.is_set():
                return
            self._finish_planning()
            if not self.active:
                return

        traj = self.result.trajectory
        duration = max(traj.duration, 1e-6)
        t_arm = float(np.clip(self.elapsed, 0.0, duration))

        if not self.done_arm:
            q_opt, _, _ = traj.sample(t_arm)
            q_seed = sample_piecewise_linear(self.result.initial_positions, t_arm / duration)
            seed_poses, opt_poses = fk_poses_xyzw(self.world, np.stack((q_seed, q_opt), axis=0), self.link_ids)
            self.context.apply_link_poses(self.links, opt_poses)
            sep = np.linalg.norm(seed_poses[:, :3] - opt_poses[:, :3], axis=-1)
            align = np.abs((seed_poses[:, 3:] * opt_poses[:, 3:]).sum(axis=-1))
            visible = (sep > 1e-5) | (align < 1.0 - 1e-6)
            for node, pose, show in zip(self.ghost, seed_poses, visible, strict=True):
                node.visible = bool(show)
                if show:
                    node.set_global_transform(pose[:3], (pose[6], *pose[3:6]))
            if self.elapsed >= duration:
                self.done_arm = True

        release_time = self.elapsed - duration
        if release_time > 0.0:
            if not len(self.handoff):
                print("Loaded motion complete. Drop is outside the conveyor handoff region.")
                self.active = False
                return
            settle, travel = self.handoff
            settle_time = 0.15
            travel_fraction = max(0.0, (release_time - settle_time) * 0.55 / max(float(np.linalg.norm(travel)), 1e-9))
            pose = self.drop_pose
            center = pose[:3] + min(1.0, release_time / settle_time) * settle + min(1.0, travel_fraction) * travel
            self.links_by_name[PAYLOAD[1]].set_global_transform(center, (pose[6], *pose[3:6]))
            if travel_fraction >= 1.0:
                print("Unload complete. The released carton reached the end of the rollers.")
                self.active = False

        self.elapsed += 0.65 * delta

    def _exit_tree(self):
        render.clear_debug_arrows()
        if hasattr(self, "previous_backend"):
            self.context.backend_type = self.previous_backend
