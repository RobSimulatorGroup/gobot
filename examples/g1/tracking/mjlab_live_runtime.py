"""Worker-owned native mjlab policy and physics, for Gobot Editor visualization."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def create(source, checkpoint, motion, support, extra_python_paths=(), status_file=None):
    # Resolve the verified numerical stack before optional viewer dependencies.
    import torch
    import rsl_rl.runners
    import mujoco
    import mujoco_warp
    import warp
    import numpy
    import scipy

    for path in extra_python_paths:
        if path not in sys.path:
            sys.path.append(path)
    if support not in sys.path:
        sys.path.insert(0, support)
    from mjlab_tracking_reference import reference, sha256, configure_deterministic_physics
    reference(source)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
    from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
    from mjlab.utils.torch import configure_torch_backends
    from gobot.sim import RuntimeComponents

    torch.set_num_threads(4)
    configure_torch_backends(allow_tf32=False)
    saved = json.loads(Path(checkpoint).parent.joinpath("comparison.json").read_text())
    if sha256(motion) != saved["motion_sha256"]:
        raise ValueError("motion differs from the trained checkpoint")
    cfg = unitree_g1_flat_tracking_env_cfg()
    cfg.scene.num_envs = 1
    cfg.seed = saved["seed"]
    cfg.scene.sensors[0].reduce = saved.get("self_contact_reduction", "none")
    cfg.auto_reset = False
    cfg.sim.nconmax, cfg.sim.njmax = saved["nconmax"], saved["njmax"]
    cfg.episode_length_s = 1e9
    cfg.events = {}
    cfg.observations["actor"].enable_corruption = False
    command = cfg.commands["motion"]
    command.motion_file = str(Path(motion).resolve())
    command.sampling_mode = "start"
    command.pose_range = {}
    command.velocity_range = {}
    command.joint_position_range = (0.0, 0.0)
    configure_deterministic_physics(cfg.sim, saved.get("deterministic_physics", False))
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        wrapped = RslRlVecEnvWrapper(env)
        agent = unitree_g1_tracking_ppo_runner_cfg()
        agent.logger, agent.upload_model = "tensorboard", False
        runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device="cuda:0")
        payload = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
        runner.alg.load(payload, {"actor": True}, True)
        provider = LivePolicy(env, wrapped, runner.get_inference_policy("cuda:0"),
                              status_file=status_file, checkpoint_sha256=sha256(checkpoint))
        return RuntimeComponents(provider, LiveCommands(provider))
    except BaseException:
        env.close()
        raise


class LivePolicy:
    num_envs = 1
    fixed_time_step = 0.02
    _last_robot_view_joint_count = 29
    capabilities = SimpleNamespace(name="LIVE mjlab PPO + MuJoCo Warp", device="cuda:0",
                                   graph_capture=True)
    graph_captured = True

    def __init__(self, env, wrapped, policy, *, status_file, checkpoint_sha256):
        import mujoco
        import torch
        self.env, self.wrapped, self.policy = env, wrapped, policy
        self.command = env.command_manager.get_term("motion")
        self.robot = env.scene["robot"]
        self.status_file = Path(status_file) if status_file else None
        self.checkpoint_sha256 = checkpoint_sha256
        self.capacities = {"nconmax": env.cfg.sim.nconmax, "njmax": env.cfg.sim.njmax}
        self.host = mujoco.MjData(env.sim.mj_model)
        self.body_ids = [env.sim.mj_model.body("robot/" + n).id for n in self.robot.body_names]
        from gobot.sim.providers.mujoco_warp_contacts import MuJoCoWarpContactSource
        self.contact_source = MuJoCoWarpContactSource(
            env.sim.mj_model, env.sim.wp_model, env.sim.wp_data,
            bodies={"robot/" + name: ("robot", name) for name in self.robot.body_names},
            sample_dt=env.physics_dt, device=env.sim.wp_device)
        self.push_count = 0
        self.final_qpos = None
        self.last_error = 0.0
        original = env.reward_manager.compute

        def capture(dt):
            reward = original(dt)
            # Read the force actually solved by the last mj_step before the
            # env's forward/reset recomputes constraints for the next state.
            self.contact_source.capture()
            self.last_error = float((self.command.anchor_pos_w - self.command.robot_anchor_pos_w).norm())
            # Preserve the final physical state before the command wraps its reference.
            if int(self.command.time_steps[0]) == self.command.motion.time_step_total - 1:
                self.final_qpos = self.env.sim.data.qpos[0].cpu().numpy().copy()
            return reward

        env.reward_manager.compute = capture
        self.reset([True])
        print("LIVE mjlab: checkpoint loaded; policy inference at 50 Hz, physics at 200 Hz; "
              "no recorded trajectory is used.", flush=True)

    def reset(self, mask):
        if not any(mask):
            return
        import torch
        with torch.inference_mode():
            self.obs, _ = self.wrapped.reset()
        self.steps = 0
        self.frame = 0
        self.terminated = self.completed = False
        self.action_rms = self.last_error = 0.0
        self.final_qpos = None
        self.contact_source.clear()
        self._update_arrays()
        self._write_status()

    def step(self, *, nsteps=1):
        import torch
        with torch.inference_mode():
            for _ in range(nsteps):
                if self.terminated or self.completed:
                    return
                self.frame = int(self.command.time_steps[0])
                action = self.policy(self.obs)
                self.action_rms = float(action.square().mean().sqrt())
                self.obs, _, done, _ = self.wrapped.step(action)
                self.steps += 1
                self.terminated = bool(done.any())
                self.completed = (self.frame == self.command.motion.time_step_total - 1
                                  and not self.terminated)
                self._update_arrays()
                if self.steps % 50 == 0 or self.terminated or self.completed:
                    self._write_status()

    def push(self, velocity):
        import torch
        if self.terminated or self.completed:
            return
        with torch.inference_mode():
            state = self.robot.data.root_link_vel_w.clone()
            state[:, :3] += torch.tensor(velocity, device=self.env.device)
            self.robot.write_root_link_velocity_to_sim(state)
            self.env.sim.forward()
            self.env.sim.sense()
            self.obs = self.wrapped.get_observations()
            self.push_count += 1
        print(f"LIVE mjlab push #{self.push_count}: delta velocity {velocity}", flush=True)

    def contact_snapshot(self, environments):
        return self.contact_source.snapshot(environments)

    def _update_arrays(self):
        import numpy as np
        import torch
        import mujoco
        if self.final_qpos is not None:
            self.host.qpos[:] = self.final_qpos
            mujoco.mj_kinematics(self.env.sim.mj_model, self.host)
            actual = np.concatenate((self.host.xpos[self.body_ids],
                    self.host.xquat[self.body_ids][:, [1, 2, 3, 0]]), axis=-1)[None]
        else:
            actual = torch.cat((self.robot.data.body_link_pos_w,
                                self.robot.data.body_link_quat_w[..., [1, 2, 3, 0]]), -1).cpu().numpy()
        motion = self.command.motion
        reference = torch.cat((motion._body_pos_w[self.frame],
                               motion._body_quat_w[self.frame][:, [1, 2, 3, 0]]), -1).cpu().numpy()[None]
        self.arrays = {
            "policy": np.asarray(actual, dtype=np.float32),
            "reference": np.asarray(reference, dtype=np.float32),
            "status": np.asarray([[self.steps, self.steps * self.fixed_time_step,
                                   self.last_error, self.terminated, self.completed,
                                   self.action_rms, self.push_count]], dtype=np.float64),
        }

    def _write_status(self):
        if self.status_file is None:
            return
        value = dict(mode="live_policy_inference", inference_steps=self.steps,
                     physics_steps=self.steps * self.env.cfg.decimation,
                     seconds=self.steps * self.fixed_time_step, action_rms=self.action_rms,
                     anchor_position_error_m=self.last_error, terminated=self.terminated,
                     completed=self.completed, pushes=self.push_count,
                     checkpoint_sha256=self.checkpoint_sha256)
        temp = self.status_file.with_suffix(".tmp")
        temp.write_text(json.dumps(value, indent=2) + "\n")
        temp.replace(self.status_file)

    def close(self):
        self.env.close()


class LiveCommands:
    def __init__(self, provider):
        self.provider = provider

    def step(self, dt):
        return None

    def reset(self, environments):
        pass

    def apply_commands(self, commands):
        if set(commands) - {"push_velocity"}:
            raise ValueError("unknown live policy command")
        if "push_velocity" in commands:
            self.provider.push(commands["push_velocity"][0])
