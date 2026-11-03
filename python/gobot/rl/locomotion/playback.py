"""Policy playback through the shared velocity task and external Play session."""
from pathlib import Path
import numpy as np
import gobot
from . import make_velocity_env
from .manifest import validate_policy
from gobot.rl.policy import policy_manifest_from_onnx_metadata

class VelocityPolicyPlayback:
    def __init__(self, context, root, cfg, policy_path, *, device='cuda:0'):
        import onnxruntime as ort
        import torch
        self.session = None
        self.env = None
        self.context = context
        self.cfg = cfg
        self.command = (0., 0., 0.)
        self.actor = ort.InferenceSession(str(policy_path), providers=['CPUExecutionProvider'])
        manifest = policy_manifest_from_onnx_metadata(self.actor.get_modelmeta().custom_metadata_map)
        if manifest is None:
            raise ValueError('policy has no Gobot manifest')
        cfg.randomize_reset_yaw = False
        cfg.spawn_jitter = 0.
        cfg.command.rel_standing_envs = cfg.command.rel_heading_envs = cfg.command.rel_forward_envs = 0.
        cfg.command.heading_command = False
        cfg.command.resampling_time_range = (1.e9, 1.e9)
        cfg.command.ranges.lin_vel_x = cfg.command.ranges.lin_vel_y = cfg.command.ranges.ang_vel_z = (0., 0.)
        # Compile the active Play clone; only the provider owns simulation state.
        self.env = make_velocity_env(cfg, backend='mujoco-warp', num_envs=1, device=device,
                                     context=context, reuse_scene=True, collect_step_extras=False)
        try:
            validate_policy(manifest, self.env, cfg)
            # Keep reset near the authored origin and camera rather than sampling
            # a distant training cell on each reset.
            origins = self.env.terrain_origins.detach().cpu().numpy()
            cell = int(np.argmin(np.square(origins[:, :2]).sum(axis=1)))
            _, columns = self.env.terrain_grid_shape
            self.env.set_terrain_assignments([cell // columns], [cell % columns], fixed=True)
            self.env.reset()
            links = []
            stack = [root]
            while stack:
                node = stack.pop()
                if node.type_name == 'Link3D':
                    links.append(node)
                stack.extend(node.children)
            self.view = self.env.provider.create_robot_view(robot_name=cfg.robot_name, base_link=cfg.base_link,
                joint_names=cfg.joint_names, link_names=tuple(link.name for link in links),
                scene_context=context, scene_links=tuple(links))
            self.session = gobot.sim.ProviderPlaySession(context, self, fixed_dt=self.env.step_dt,
                reset=self.reset, sync_scene=self.view.sync_scene, close_provider=False).start()
        except Exception:
            self.close()
            raise
    def step(self, *, nsteps=1):
        import torch
        for _ in range(nsteps):
            self.env.set_command(self.command)
            observation = self.env.get_observations()['actor'].detach().cpu().numpy()
            action = self.actor.run(None, {self.actor.get_inputs()[0].name: observation})[0]
            self.env.step(action)
    def reset(self):
        self.env.reset()
    def close(self):
        session, self.session = self.session, None
        if session is not None:
            session.close()
        env, self.env = self.env, None
        if env is not None:
            env.close()
