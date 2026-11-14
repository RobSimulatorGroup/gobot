"""mjlab tracking MDP over Gobot's scene and articulation services.

No editor, mjlab environment, or backend model pointers enter this task.
"""

from dataclasses import asdict
import math
from pathlib import Path
import torch
from gobot.rl.articulation import ArticulationBatch
from gobot.rl.batch import BatchEnvState
from gobot.rl.kinematics import SceneKinematics
from gobot.rl.spec import ActionSpec, ObservationSpec, SpecField
from gobot.sim.providers.base import RobotBatchSpec
from . import mdp
from .math import matrix, quaternion, multiply, from_euler
from .motion import Motion
from .sampler import AdaptiveSampler


class TrackingEnv:
    is_vector_env = True
    accepts_device_actions = True

    def __init__(self, cfg, *, autoreset=True):
        cfg.validate()
        self.cfg_obj = cfg
        self.cfg = asdict(cfg)
        self.project_path = Path(cfg.project_path).resolve()
        self.scene_path = cfg.scene_path
        self.num_envs = cfg.num_envs
        self.num_actions = len(cfg.joint_names)
        self.device = str(
            cfg.device or ("cuda:0" if cfg.backend == "mujoco-warp" else "cpu")
        )
        self.seed = cfg.seed
        self.generator = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.autoreset = autoreset
        self.control_dt = cfg.physics_dt * cfg.decimation
        self.max_episode_length = math.ceil(cfg.episode_length_s / self.control_dt)
        self.anchor = cfg.body_names.index(cfg.anchor_link)
        self.imu = cfg.body_names.index(cfg.imu_link)
        self.ee = [cfg.body_names.index(n) for n in cfg.end_effectors]

        def path(p):
            return p if Path(p).is_absolute() else self.project_path / p

        self.motion = Motion(
            path(cfg.motion_file),
            cfg.joint_names,
            cfg.body_names,
            dt=self.control_dt,
            device=self.device,
            layout=None if cfg.motion_layout is None else path(cfg.motion_layout),
        )
        self.batch = ArticulationBatch(
            project_path=self.project_path,
            scene_path=cfg.scene_path,
            spec=RobotBatchSpec(
                cfg.robot_name, cfg.base_link, cfg.joint_names, cfg.body_names
            ),
            num_envs=cfg.num_envs,
            backend=cfg.backend,
            device=self.device,
            physics_dt=cfg.physics_dt,
            decimation=cfg.decimation,
            solver_settings=cfg.solver_settings,
            capture_graphs=cfg.capture_graphs,
            nconmax=cfg.nconmax,
            njmax=cfg.njmax,
            self_contact_threshold=10.0,
        )
        try:
            self._initialize()
        except BaseException:
            self.close()
            raise

    def _initialize(self):
        cfg = self.cfg_obj
        self.kin = SceneKinematics(
            self.batch.snapshot,
            robot_name=cfg.robot_name,
            base_link=cfg.base_link,
            joint_names=cfg.joint_names,
            tracked_links=cfg.body_names,
            device=self.device,
        )
        robot = next(
            r for r in self.batch.snapshot["robots"] if r["name"] == cfg.robot_name
        )
        joints = {j["name"]: j for j in robot["joints"]}
        ordered = [joints[n] for n in cfg.joint_names]
        if any(j["drive_stiffness"] <= 0 or j["effort_limit"] <= 0 for j in ordered):
            raise ValueError(
                "tracking requires authored position drives with positive stiffness and effort limits"
            )
        self.default = torch.tensor(
            [j["initial_position"] for j in ordered], device=self.device
        )
        self.action_scale = torch.tensor(
            [0.25 * j["effort_limit"] / j["drive_stiffness"] for j in ordered],
            device=self.device,
        )
        midpoint = (self.kin.q_lo + self.kin.q_hi) / 2
        extent = (self.kin.q_hi - self.kin.q_lo) * cfg.soft_joint_limit_factor / 2
        self.soft_limits = torch.stack((midpoint - extent, midpoint + extent), -1)
        if not torch.isfinite(self.soft_limits).all():
            raise ValueError("tracking joint limits must be finite")
        self.bias = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.imu_position = torch.tensor(cfg.imu_position, device=self.device)
        self.imu_orientation = torch.tensor(cfg.imu_orientation, device=self.device)
        self._startup_randomization(robot)
        self.sampler = AdaptiveSampler(
            self.motion.frames,
            self.control_dt,
            device=self.device,
            alpha=cfg.adaptive_alpha,
            uniform=cfg.adaptive_uniform_ratio,
            kernel_size=cfg.adaptive_kernel_size,
            decay=cfg.adaptive_lambda,
        )
        self.frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.steps = torch.zeros_like(self.frame)
        self.episode_length_buf = self.steps
        self.action = torch.zeros_like(self.bias)
        self.pending_reset = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.ids = torch.arange(self.num_envs, device=self.device)
        self.push_time = 1 + 2 * self._rand((self.num_envs,))
        self.episode_terms = {
            k: torch.zeros(self.num_envs, device=self.device)
            for k in cfg.reward_weights
        }
        self.common_step_counter = 0
        self.state = None
        self.action_spec = ActionSpec(
            "gobot_tracking_position_offset_v1",
            tuple(SpecField(n, 1) for n in cfg.joint_names),
            -torch.finfo(torch.float32).max,
            torch.finfo(torch.float32).max,
        )
        self.reset()
        self.num_obs = self.state.obs["actor"].shape[1]
        self.num_privileged_obs = self.state.obs["critic"].shape[1]

    def _rand(self, shape):
        return torch.rand(shape, device=self.device, generator=self.generator)

    def _startup_randomization(self, robot):
        if not self.cfg_obj.domain_randomization:
            return
        offsets = (self._rand((self.num_envs, 3)) * 2 - 1) * torch.tensor(
            [0.025, 0.05, 0.05], device=self.device
        )
        self.bias.copy_((self._rand(self.bias.shape) * 2 - 1) * 0.01)
        slide = 0.3 + 0.9 * self._rand((self.num_envs,))
        friction = {}
        feet = {n for n in self.cfg_obj.end_effectors if "ankle" in n}
        for link in robot["links"]:
            if link["name"] not in feet:
                continue
            for shape in link.get("shapes", ()):
                material = shape["material"]
                v = torch.empty(self.num_envs, 3, device=self.device)
                v[:, 0] = slide
                v[:, 1] = material["torsional_friction"]
                v[:, 2] = material["rolling_friction"]
                friction[shape["name"]] = v
        if not friction:
            raise ValueError(
                "tracking friction randomization requires named foot collision shapes"
            )
        self.batch.configure_physics(
            link_center_of_mass_offsets={self.cfg_obj.anchor_link: offsets},
            shape_friction=friction,
        )

    def _bodies(self, state):
        p, r, v, w = self.kin(
            state.base_pose[:, :3],
            matrix(state.base_pose[:, (6, 3, 4, 5)]),
            state.base_velocity[:, :3],
            state.base_velocity[:, 3:],
            state.joint_position,
            state.joint_velocity,
        )
        return p, quaternion(r), v, w

    def _refresh_reference(self):
        self.reference = self.motion.bodies(self.frame)
        self.relative = mdp.reanchor(
            self.reference, self._bodies(self.batch.read_state()), self.anchor
        )

    def _observations(self, *, buffered=True):
        s = self.batch.read_state()
        cfg = self.cfg_obj
        actor, critic = mdp.observations(
            self.reference,
            self._bodies(s),
            torch.cat(
                (self.motion.joint_pos[self.frame], self.motion.joint_vel[self.frame]),
                -1,
            ),
            s.joint_position,
            s.joint_velocity,
            self.action,
            self.default,
            self.bias,
            anchor=self.anchor,
            imu=self.imu,
            imu_position=self.imu_position,
            imu_orientation=self.imu_orientation,
            state_estimation=cfg.state_estimation,
        )
        self.observation_spec = ObservationSpec(
            "gobot_mjlab_tracking_actor_v1",
            tuple(SpecField(k, v.shape[1]) for k, v in actor.items()),
        )
        self.critic_spec = ObservationSpec(
            "gobot_mjlab_tracking_critic_v1",
            tuple(SpecField(k, v.shape[1]) for k, v in critic.items()),
        )
        if cfg.observation_noise and buffered:
            ranges = dict(
                motion_anchor_pos_b=0.25,
                motion_anchor_ori_b=0.05,
                base_lin_vel=0.5,
                base_ang_vel=0.2,
                joint_pos=0.01,
                joint_vel=0.5,
            )
            actor = {
                k: v + (self._rand(v.shape) * 2 - 1) * ranges[k] if k in ranges else v
                for k, v in actor.items()
            }
        result = {
            "actor": torch.cat(tuple(actor.values()), -1),
            "critic": torch.cat(tuple(critic.values()), -1),
        }
        if not buffered:
            return result
        if not hasattr(self, "_observation_buffers"):
            self._observation_buffers = [
                {k: torch.empty_like(v) for k, v in result.items()} for _ in range(3)
            ]
            self._observation_index = 0
        target = self._observation_buffers[self._observation_index]
        self._observation_index = (self._observation_index + 1) % len(
            self._observation_buffers
        )
        for key, value in result.items():
            target[key].copy_(value)
        return target

    def _reset_rows(self, ids, frames, *, noise=True, episode=True):
        if not len(ids):
            return
        self.frame[ids] = frames
        p, r, v, w = (x.clone() for x in self.motion.bodies(frames))
        p, r, v, w = p[:, 0], r[:, 0], v[:, 0], w[:, 0]
        q = self.motion.joint_pos[frames].clone()
        qd = self.motion.joint_vel[frames]
        if noise and self.cfg_obj.reset_noise:
            delta = (self._rand((len(ids), 6)) * 2 - 1) * torch.tensor(
                [0.05, 0.05, 0.01, 0.1, 0.1, 0.2], device=self.device
            )
            speed = (self._rand((len(ids), 6)) * 2 - 1) * torch.tensor(
                [0.5, 0.5, 0.2, 0.52, 0.52, 0.78], device=self.device
            )
            p += delta[:, :3]
            r = multiply(from_euler(delta[:, 3:]), r)
            v += speed[:, :3]
            w += speed[:, 3:]
            q += (self._rand(q.shape) * 2 - 1) * 0.1
        s = self.batch.read_state()
        bp, bv, jp, jv, targets = (
            x.clone()
            for x in (
                s.base_pose,
                s.base_velocity,
                s.joint_position,
                s.joint_velocity,
                s.joint_control,
            )
        )
        bp[ids] = torch.cat((p, r[:, (1, 2, 3, 0)]), -1)
        bv[ids] = torch.cat((v, w), -1)
        jp[ids] = q.clamp(self.soft_limits[:, 0], self.soft_limits[:, 1])
        jv[ids] = qd
        targets[ids] = 0.0
        mask = torch.zeros_like(self.pending_reset)
        mask[ids] = True
        self.batch.reset(
            mask,
            base_pose=bp,
            base_velocity=bv,
            joint_position=jp,
            joint_velocity=jv,
            targets=targets,
        )
        if episode:
            self.action[ids] = 0
            self.steps[ids] = 0
            self.pending_reset[ids] = False
            self.push_time[ids] = 1 + 2 * self._rand((len(ids),))
            for value in self.episode_terms.values():
                value[ids] = 0

    @torch.no_grad()
    def reset(self, *, seed=None, mask=None, frames=None):
        if seed is not None:
            self.seed = seed
            self.generator.manual_seed(seed)
        mask = (
            torch.ones_like(self.pending_reset)
            if mask is None
            else torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        )
        if mask.shape != (self.num_envs,):
            raise ValueError("reset mask has wrong shape")
        ids = mask.nonzero().flatten()
        sampled = frames is None
        if sampled:
            frames = self.sampler.sample(
                len(ids), self.generator, self.cfg_obj.sampling_mode
            )
        else:
            frames = torch.as_tensor(frames, device=self.device)
            if frames.dtype not in (torch.int32, torch.int64):
                raise ValueError("frames must be integer indices")
            if (
                frames.shape != (len(ids),)
                or ((frames < 0) | (frames >= self.motion.frames)).any()
            ):
                raise ValueError("invalid reset frames")
        self._reset_rows(ids, frames, noise=sampled)
        # Upstream explicit reset samples frame k then computes command k+1.
        if sampled:
            self.frame[ids] += 1
            wrap = (self.frame >= self.motion.frames).nonzero().flatten()
            self._reset_rows(
                wrap,
                self.sampler.sample(
                    len(wrap), self.generator, self.cfg_obj.sampling_mode
                ),
                episode=False,
            )
        self._refresh_reference()
        zero = torch.zeros(self.num_envs, device=self.device)
        self.state = BatchEnvState(
            self._observations(),
            zero,
            zero.bool(),
            zero.bool(),
            {"steps": self.steps.clone()},
        )
        return self.state

    def _before_substep(self, tick):
        if tick == self.cfg_obj.decimation - 1:
            self.batch.refresh_kinematics()
            # Upstream reward and termination see pre-final-integration body state.
            self.reward_bodies = self._bodies(self.batch.read_state())

    def _push(self):
        if not self.cfg_obj.pushes:
            return
        self.push_time -= self.control_dt
        mask = self.push_time <= 0
        ids = mask.nonzero().flatten()
        if not len(ids):
            return
        s = self.batch.read_state()
        velocity = s.base_velocity.clone()
        velocity[ids] += (self._rand((len(ids), 6)) * 2 - 1) * torch.tensor(
            [0.5, 0.5, 0.2, 0.52, 0.52, 0.78], device=self.device
        )
        self.batch.set_base_velocity(mask, velocity)
        self.push_time[ids] = 1 + 2 * self._rand((len(ids),))

    @torch.no_grad()
    def step(self, action):
        if self.pending_reset.any():
            raise RuntimeError("reset finished environments before stepping")
        action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        if action.shape != self.action.shape or not torch.isfinite(action).all():
            raise ValueError("actions must be finite [num_envs, num_joints]")
        previous = self.action.clone()
        self.action.copy_(action)
        target_frame = self.frame.clone()
        self.batch.set_joint_targets(
            self.default + self.action_scale * action - self.bias
        )
        state = self.batch.step_control(before_substep=self._before_substep)
        self.batch.assert_no_overflow()
        if not all(torch.isfinite(v).all() for v in vars(state).values()):
            raise FloatingPointError("nonfinite tracking simulation state")
        terms = mdp.rewards(
            self.reference,
            self.reward_bodies,
            self.relative,
            state.joint_position,
            self.soft_limits,
            action,
            previous,
            self.batch.read_self_contact_substeps(),
            anchor=self.anchor,
            weights=self.cfg_obj.reward_weights,
            dt=self.control_dt,
        )
        conditions = mdp.terminations(
            self.reference,
            self.reward_bodies,
            self.relative,
            anchor=self.anchor,
            end_effectors=self.ee,
        )
        terminated = torch.stack(tuple(conditions.values())).any(0)
        self.steps += 1
        self.common_step_counter += 1
        truncated = self.steps >= self.max_episode_length
        done = terminated | truncated
        for key, value in terms.items():
            self.episode_terms[key] += value
        reward = sum(terms.values())
        final = self._observations(buffered=False) if done.any() else None
        info = {
            "steps": self.steps.clone(),
            "target_frame": target_frame,
            "reward_terms": terms,
            "termination_terms": conditions,
            "log": {},
        }
        ids = done.nonzero().flatten()
        if len(ids):
            info["log"] = {
                f"Episode_Reward/{k}": v[ids].mean() / self.cfg_obj.episode_length_s
                for k, v in self.episode_terms.items()
            }
            info["log"]["Episode/length"] = self.steps[ids].float().mean()
        if self.cfg_obj.sampling_mode == "adaptive":
            self.sampler.record(target_frame, terminated)
        self._push()
        if self.autoreset:
            self._reset_rows(
                ids,
                self.sampler.sample(
                    len(ids), self.generator, self.cfg_obj.sampling_mode
                ),
            )
        else:
            self.pending_reset.copy_(done)
        self.frame += 1
        wrap = (self.frame >= self.motion.frames) & ~self.pending_reset
        wrapped = wrap.nonzero().flatten()
        self._reset_rows(
            wrapped,
            self.sampler.sample(
                len(wrapped), self.generator, self.cfg_obj.sampling_mode
            ),
            episode=False,
        )
        # Keep a valid reference for terminal observations during manual-reset mode.
        self.frame.clamp_(max=self.motion.frames - 1)
        self._refresh_reference()
        if self.cfg_obj.sampling_mode == "adaptive":
            self.sampler.update()
        info["motion_wrap"] = wrap
        info["steps"] = self.steps.clone()
        self.state = BatchEnvState(
            self._observations(), reward, terminated, truncated, info, final
        )
        return self.state

    def training_state_dict(self):
        return dict(
            failures=self.sampler.failed.cpu().clone(),
            generator=self.generator.get_state().cpu(),
            bias=self.bias.cpu().clone(),
            physics=self.batch.physics_parameter_state(),
            common_step_counter=self.common_step_counter,
        )

    def load_training_state_dict(self, state):
        f = state["failures"].to(self.device)
        if (
            f.shape != self.sampler.failed.shape
            or not torch.isfinite(f).all()
            or (f < 0).any()
        ):
            raise ValueError("invalid sampler checkpoint")
        self.sampler.failed.copy_(f)
        self.sampler.pending.zero_()
        self.bias.copy_(state["bias"].to(self.device))
        self.generator.set_state(state["generator"].cpu())
        self.batch.configure_physics(**state["physics"])
        self.common_step_counter = int(state["common_step_counter"])
        self.reset()
        return {"common_step_counter": self.common_step_counter}

    def close(self):
        self.batch.close()
