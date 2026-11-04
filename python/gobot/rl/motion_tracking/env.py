"""SuperMimic-Reproduced task on a scene-authored Gobot articulation batch.

The control clock, reward/reference order, episode sampler and terminal replay
rows follow the pinned reference. Simulator access stays behind ArticulationBatch.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from gobot.rl.articulation import ArticulationBatch
from gobot.rl.kinematics import SceneKinematics
from gobot.rl.spec import ObservationSpec, ActionSpec, SpecField
from gobot.sim.providers.base import RobotBatchSpec
from gobot.rl.supermimic.heads import AnalyticHeads, OBS_TERMS, analytic_observation, anchor_relative_obs, flatten_history_term_major
from gobot.rl.supermimic.motion_library import MotionLibrary, discover_npz
from gobot.rl.supermimic.replay import ReplayBuffer
from gobot.rl.supermimic.rewards import reanchor_reference, tracking_reward_terms
from gobot.rl.supermimic.rotations import quat_apply, quat_apply_inverse, quat_mul, quat_normalize
from gobot.rl.supermimic.rollout import OBS_NOISE_HALF, ANCHOR_NOISE_HALF
from gobot.rl.supermimic.sampler import ReturnSampler, SamplerCfg
from gobot.rl.supermimic.wm_trainer import reference_at
from .kinematics import SceneStateKinematics


def euler_quat(angles):
    half = angles * .5
    c, s = half.cos(), half.sin()
    return torch.stack((c[:,0]*c[:,1]*c[:,2]+s[:,0]*s[:,1]*s[:,2],
                        s[:,0]*c[:,1]*c[:,2]-c[:,0]*s[:,1]*s[:,2],
                        c[:,0]*s[:,1]*c[:,2]+s[:,0]*c[:,1]*s[:,2],
                        c[:,0]*c[:,1]*s[:,2]-s[:,0]*s[:,1]*c[:,2]), -1)


class MotionTrackingEnv:
    def __init__(self, cfg):
        self.cfg, self.num_envs = cfg, cfg.env.num_envs
        cfg.validate()
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.fp32_precision = "tf32"
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        self.profile = json.loads((Path(cfg.project)/'assets/supermimic/robot_profile.json').read_text())
        p = self.profile
        if p.get('schema_version') != 2:
            raise ValueError('prepare the scene from the pinned SuperMimic-Reproduced export first')
        self.joint_names, self.body_names = tuple(p['joint_names']), tuple(p['tracked_links'])
        self.batch = ArticulationBatch(project_path=cfg.project, scene_path=cfg.scene,
            spec=RobotBatchSpec(p['robot_name'], p['base_link'], self.joint_names, self.body_names),
            num_envs=self.num_envs, backend=cfg.backend, device=cfg.device,
            physics_dt=p['physics_dt'], decimation=p['decimation'], solver_settings=p['solver_settings'],
            capture_graphs=cfg.capture_graphs, deterministic=cfg.deterministic,
            nconmax=cfg.nconmax, njmax=cfg.njmax)
        try:
            self._initialize()
        except BaseException:
            self.batch.close()
            raise

    def _initialize(self):
        cfg, p = self.cfg, self.profile
        self.device = self.batch.device
        self.step_dt = self.batch.control_dt
        self.max_episode_steps = math.ceil(cfg.env.episode_length_s/self.step_dt)
        self.task_version = 'supermimic_reproduced_g1_v1'
        self.observation_spec = ObservationSpec(self.task_version + '_policy',
            tuple(SpecField(name, cfg.policy.history*dim) for name,dim in OBS_TERMS))
        goal_fields = (SpecField('future_frames', cfg.env.future_frames*68),)
        if cfg.env.anchor_obs:
            goal_fields += (SpecField('anchor_pose',9),)
        self.goal_observation_spec = ObservationSpec(self.task_version + '_goal',goal_fields)
        self.action_spec = ActionSpec(self.task_version + '_action',
            tuple(SpecField(name,1,'scaled joint-position offset') for name in self.joint_names),
            lower=-float('inf'),upper=float('inf'))
        self.default_q = torch.tensor(p['default_joint_position'], device=self.device)
        self.action_scale = torch.tensor(p['action_scale'], device=self.device)
        scene_fk = SceneKinematics(self.batch.snapshot, robot_name=p['robot_name'], base_link=p['base_link'],
            joint_names=self.joint_names, tracked_links=self.body_names, device=self.device)
        self.fk = SceneStateKinematics(scene_fk)
        middle, half = (scene_fk.q_lo+scene_fk.q_hi)/2, (scene_fk.q_hi-scene_fk.q_lo)*p['soft_joint_limit_factor']/2
        self.soft_limits = torch.stack((middle-half, middle+half), -1)
        self.heads = AnalyticHeads(self.fk, self.body_names, p['anchor_link'], p['base_link'], self.soft_limits, self.default_q)
        self.files = [Path(cfg.data.library)] if cfg.data.library else discover_npz(
            cfg.data.npz_dir, names=cfg.data.clips, limit=cfg.data.limit, manifest=cfg.data.manifest)
        self.library = MotionLibrary(self.files, self.body_names, self.default_q, self.device,
            compact=bool(cfg.data.library or cfg.env.compact_library), fk=self.fk)
        if bool((self.library.clip_len < 3).any()):
            raise ValueError('tracking clips must contain at least three 50 Hz frames')
        self.sampler = ReturnSampler(self.library, self.num_envs, self.max_episode_steps, SamplerCfg(sampling_mode=cfg.env.sampling_mode))
        self.buffer = ReplayBuffer(self.num_envs, cfg.env.replay_capacity, 71, 29, 33, self.device)
        self.ids = torch.arange(self.num_envs, device=self.device)
        self.frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.clip_end = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ep_step = torch.zeros_like(self.frame)
        self.ep_start = torch.ones_like(self.clip_end)
        self.pending_reset = torch.zeros_like(self.clip_end)
        self.action = torch.zeros(self.num_envs, 29, device=self.device)
        self._reset_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.xi = torch.zeros(self.num_envs, 33, device=self.device)
        self.xi[:,0] = 1.
        self.history = torch.zeros(cfg.policy.history, self.num_envs, 93, device=self.device)
        self.obs_noise_half = OBS_NOISE_HALF.to(self.device)
        self.anchor_noise_half = ANCHOR_NOISE_HALF.to(self.device)
        self.forced_frame = torch.full_like(self.frame, -1)
        if cfg.env.domain_randomization:
            self.xi[:,30:33] = (2*torch.rand(self.num_envs, 3, device=self.device)-1)*torch.tensor([.025,.05,.05], device=self.device)
        self._apply_physics_context(include_com=True)

    def _apply_physics_context(self, *, include_com=False):
        # Only the 14 foot shapes change friction; torso COM is sampled once.
        robot = next(r for r in self.batch.snapshot['robots'] if r['name']==self.profile['robot_name'])
        shapes = {s['name']:s for link in robot['links'] for s in link.get('shapes', ())}
        friction = {}
        for name in self.profile['foot_shapes']:
            m = shapes[name]['material']
            v = torch.empty(self.num_envs, 3, device=self.device)
            v[:,0], v[:,1], v[:,2] = self.xi[:,0], m['torsional_friction'], m['rolling_friction']
            friction[name] = v
        kwargs = {'shape_friction':friction}
        if include_com:
            kwargs['link_center_of_mass_offsets'] = {self.profile['anchor_link']:self.xi[:,30:33]}
        self.batch.configure_physics(**kwargs)

    def physical_state(self):
        r = self.batch.read_state()
        q = r.base_pose[:, (6,3,4,5)]
        return torch.cat((r.base_pose[:,:3], q, r.joint_position, r.base_velocity[:,:3],
                          quat_apply_inverse(q, r.base_velocity[:,3:]), r.joint_velocity), -1)

    def learning_state(self):
        state = self.physical_state()
        # Compact reference quaternions have fp16 norm error. mjlab replay stores
        # that raw reset qpos, while its body observations use normalized xquat.
        state[self.ep_start,3:7] = self._reset_quat[self.ep_start]
        return state

    def _before_substep(self, substep):
        if substep == self.batch.decimation-1:
            self.batch.refresh_kinematics()
            self._lag_state = self.physical_state()

    def _write_reset(self, ids, frame, noise):
        self.frame[ids] = frame
        # mjlab reset events run before command resampling writes the pose.
        if self.cfg.env.domain_randomization:
            self.xi[ids,1:30] = (2*torch.rand(len(ids),29,device=self.device)-1)*.01
            self.xi[ids,0] = .3+.9*torch.rand(len(ids),device=self.device)
            self._apply_physics_context()
        r = self.library.ref(frame)
        pos, ori, lin, ang, q, qd = (r.root_pos_w.clone(), r.root_quat_w.clone(), r.root_lin_vel_w.clone(),
                                   r.root_ang_vel_w.clone(), r.joint_pos.clone(), r.joint_vel.clone())
        if noise:
            pose = (2*torch.rand(len(ids), 6, device=self.device)-1)*torch.tensor([.05,.05,.01,.1,.1,.2],device=self.device)
            vel = (2*torch.rand(len(ids), 6, device=self.device)-1)*torch.tensor([.5,.5,.2,.52,.52,.78],device=self.device)
            pos += pose[:,:3]
            ori = quat_mul(euler_quat(pose[:,3:]), ori)
            lin += vel[:,:3]
            ang += vel[:,3:]
            q += (2*torch.rand_like(q)-1)*.1
        self._reset_quat[ids] = ori
        # Preserve reference body angular coordinates while providing a unit
        # orientation to the backend-neutral scene reset API.
        angular_body = quat_apply_inverse(ori, ang)
        ori = quat_normalize(ori)
        ang = quat_apply(ori, angular_body)
        q = q.clamp(self.soft_limits[:,0], self.soft_limits[:,1])
        state = self.batch.read_state()
        bp, bv = state.base_pose.clone(), state.base_velocity.clone()
        jp, jv, targets = state.joint_position.clone(), state.joint_velocity.clone(), state.joint_control.clone()
        bp[ids] = torch.cat((pos, ori[:,(1,2,3,0)]),-1)
        bv[ids] = torch.cat((lin, ang),-1)
        jp[ids], jv[ids] = q, qd
        targets[ids] = 0.  # mjlab resets actuator data before processing the next action.
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[ids] = True
        self.batch.reset(mask, base_pose=bp, base_velocity=bv, joint_position=jp, joint_velocity=jv, targets=targets)
        self.action[ids] = 0.
        self.ep_step[ids] = 0
        self.ep_start[ids] = True
        self.clip_end[ids] = False
        self.pending_reset[ids] = False
        self.sampler.ep_len[ids] = 0

    def _advance(self):
        last = self.library.last_frame_of(self.frame)
        self.frame += 1
        self.clip_end |= self.frame > last
        self.frame.copy_(torch.minimum(self.frame,last))
        self.ref = reference_at(self.library, self.frame)
        i = self.heads.anchor_index
        anchor = self.batch.read_state().link_pose[:,i]
        self.relative_pose = reanchor_reference(self.ref.body_pos_w, self.ref.body_quat_w,
            self.ref.body_pos_w[:,i], self.ref.body_quat_w[:,i], anchor[:,:3], anchor[:,(6,3,4,5)])

    def _observe(self, reset_ids):
        s = self.physical_state()
        obs = analytic_observation(s, self.action, self.xi, self.default_q)
        if self.cfg.env.obs_noise:
            obs += (2*torch.rand_like(obs)-1)*self.obs_noise_half
        self.history = torch.cat((self.history[1:], obs[None]),0)
        self.history[:,reset_ids] = obs[reset_ids][None]
        goal = self.library.goal_window(self.frame, self.cfg.env.future_frames)
        if self.cfg.env.anchor_obs:
            i = self.heads.anchor_index
            pose = self.batch.read_state().link_pose[:,i]
            anchor = anchor_relative_obs(pose[:,:3], pose[:,(6,3,4,5)], self.ref.body_pos_w[:,i], self.ref.body_quat_w[:,i])
            if self.cfg.env.obs_noise:
                anchor += (2*torch.rand_like(anchor)-1)*self.anchor_noise_half
            goal = torch.cat((goal, anchor),-1)
        return {'policy':flatten_history_term_major(self.history), 'goal':goal}

    @torch.no_grad()
    def reset(self, *, frames=None):
        if frames is None:
            frames = self.sampler.sample(self.num_envs)
        self._write_reset(self.ids, frames, self.cfg.env.reset_noise)
        # Pinned command.compute advances once even on a freshly reset environment.
        self._advance()
        return self._observe(self.ids), {}

    @torch.no_grad()
    def step(self, action, *, autoreset=True, record=True):
        if self.pending_reset.any():
            raise RuntimeError('reset terminated environments before stepping')
        if action.shape != self.action.shape or not torch.isfinite(action).all():
            raise ValueError('actions must be finite [num_envs, 29] tensors')
        s, xi, frame = self.learning_state(), self.xi.clone(), self.frame.clone()
        previous = self.action.clone()
        self.action.copy_(action)
        executed = action + self.cfg.env.action_noise_std*torch.randn_like(action) if self.cfg.env.action_noise_std else action.clone()
        self.batch.set_joint_targets(self.default_q + executed*self.action_scale - self.xi[:,1:30])
        self.batch.step_control(before_substep=self._before_substep)
        snext = self.physical_state()
        # mjlab reward/termination read link kinematics before the final integration,
        # while joint limits and replay read the final qpos/qvel.
        bodies = self.heads.robot_bodies(self._lag_state)
        terms = tracking_reward_terms(robot_body_pos_w=bodies['pos'], robot_body_quat_w=bodies['quat'],
            robot_body_lin_vel_w=bodies['lin_vel'], robot_body_ang_vel_w=bodies['ang_vel'],
            ref_body_pos_w=self.ref.body_pos_w, ref_body_quat_w=self.ref.body_quat_w,
            ref_body_lin_vel_w=self.ref.body_lin_vel_w, ref_body_ang_vel_w=self.ref.body_ang_vel_w,
            anchor_index=self.heads.anchor_index, action=action, prev_action=previous,
            joint_pos=snext[:,7:36], soft_limits=self.soft_limits, cfg=self.heads.reward_cfg,
            relative_body_pose=self.relative_pose)
        reward = terms['total']
        term = self.heads.terminated(self._lag_state, self.ref)
        self.ep_step += 1
        timeout = (self.ep_step >= self.max_episode_steps) | self.clip_end
        done = term | timeout
        done_ids = done.nonzero().flatten()
        if record:
            self.buffer.add(self.ids, s, action, executed, xi, reward, term.float(), frame,
                self.library.frame_clip[frame], torch.zeros_like(done), self.ep_start)
            self.ep_start[:] = False
            if len(done_ids):
                z = torch.zeros(len(done_ids),29,device=self.device)
                self.buffer.add(done_ids, snext[done_ids], z, z, xi[done_ids], torch.zeros(len(done_ids),device=self.device),
                    term[done_ids].float(), frame[done_ids], self.library.frame_clip[frame[done_ids]],
                    torch.ones_like(done[done_ids]), torch.zeros_like(done[done_ids]))
                self.buffer.end_episode(done_ids)
        self.sampler.log_step(reward, frame)
        if len(done_ids):
            self.sampler.finalize(done_ids, term, self.clip_end)
            if autoreset:
                sampled = self.sampler.sample(len(done_ids))
                sampled = torch.where(self.forced_frame[done_ids]>=0, self.forced_frame[done_ids], sampled)
                self._write_reset(done_ids, sampled, self.cfg.env.reset_noise)
            else:
                self.pending_reset[done_ids] = True
        self._advance()
        obs = self._observe(done_ids if autoreset else self.ids[:0])
        return obs, reward, term, timeout, {'reward_terms':terms, 'terminal_state':snext, 'target_frame':frame}

    def close(self):
        self.batch.close()
