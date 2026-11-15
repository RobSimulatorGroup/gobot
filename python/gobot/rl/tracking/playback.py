"""Worker-owned policy inference and physics for the tracking editor player."""
import numpy as np
import torch

from gobot.rl.kinematics import SceneKinematics
from .inference import load_policy
from .math import matrix, quaternion


class TrackingPolicy:
    num_envs = 1

    def __init__(self, checkpoint, project_path=None, device="cuda:0"):
        self.env, self.wrapper, self.policy = load_policy(
            checkpoint, project_path=project_path, backend="mujoco-warp", device=device)
        try:
            env = self.env
            self.fixed_time_step = env.control_dt
            self.names = env.batch.presentation_link_names
            self.kinematics = SceneKinematics(
                env.batch.snapshot, robot_name=env.cfg_obj.robot_name, base_link=env.cfg_obj.base_link,
                joint_names=env.cfg_obj.joint_names, tracked_links=self.names, device=env.device)
            env.batch.enable_contact_snapshots()
            self._last_robot_view_joint_count = env.num_actions
            self.reset([True])
        except BaseException:
            self.env.close()
            raise

    @property
    def capabilities(self):
        from gobot.sim import ProviderCapabilities
        return ProviderCapabilities(name="Tracking PPO", device=self.env.device,
            device_native=True, graph_capture=True, masked_reset=True, fixed_capacity=True)

    @property
    def graph_captured(self):
        return self.env.cfg_obj.capture_graphs

    @property
    def capacities(self):
        return dict(nconmax=self.env.cfg_obj.nconmax, njmax=self.env.cfg_obj.njmax)

    @torch.inference_mode()
    def reset(self, mask):
        if not any(mask):
            return
        self.env.reset()
        self.steps = 0
        self.finished = self.failed = False
        self.error = 0.0
        self._snapshot(0)

    @torch.inference_mode()
    def step(self, *, nsteps=1):
        for _ in range(nsteps):
            if self.finished:
                break
            result = self.env.step(self.policy(self.wrapper.get_observations()))
            frame = result.info["target_frame"]
            self.error = float((self.env.motion.body_pos_w[frame, self.env.anchor]
                               - self.env.reward_bodies[0][:, self.env.anchor]).norm())
            self.steps += 1
            self.failed = bool(result.done.any())
            self.finished = self.failed or bool(result.info["motion_wrap"].any())
            self._snapshot(int(frame[0]))

    def _snapshot(self, frame):
        env = self.env
        motion = env.motion
        indices = torch.tensor([frame], device=env.device)
        p, q, v, w = (value[:, 0] for value in motion.bodies(indices))
        poses, rotations, _, _ = self.kinematics(
            p, matrix(q), v, w, motion.joint_pos[indices], motion.joint_vel[indices])
        rotations = quaternion(rotations)[..., (1, 2, 3, 0)]
        self.arrays = dict(
            policy=env.batch.read_presentation_poses().cpu().numpy().copy(),
            reference=torch.cat((poses, rotations), -1).cpu().numpy(),
            status=np.array([[self.steps, self.steps * env.control_dt,
                              (motion.frames - 1) * env.control_dt,
                              self.error, self.finished, self.failed]], dtype=np.float64),
        )

    def contact_snapshot(self, environments):
        return self.env.batch.contact_snapshot(environments)

    def close(self):
        self.env.close()


def create(checkpoint, project_path=None, device="cuda:0"):
    return TrackingPolicy(checkpoint, project_path, device)
