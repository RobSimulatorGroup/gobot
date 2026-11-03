"""Motion tracking on scene-compiled CPU/Warp articulation batches."""
from __future__ import annotations
import json
import re
from dataclasses import asdict
from pathlib import Path
import torch
import numpy as np
from gobot.sim.providers.base import RobotBatchSpec
from gobot.rl.articulation import ArticulationBatch
from gobot.rl.batch import BatchEnvState
from gobot.rl.spec import ObservationSpec, ActionSpec, SpecField
from gobot.rl.kinematics import SceneKinematics
from gobot.rl.experiment_identity import canonical_digest, file_digest
from gobot.rl.supermimic.geometry import flatten_state, mat_to_quat, quat_to_mat, so3_exp
from gobot.rl.supermimic.contracts import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
from .config import MotionTrackingCfg
from .motion import MotionLib, BIN_FRAMES
from .mdp import MDP, MAX_STEP_REWARD
from .robot import G1_JOINT_NAMES, TRACKED_BODIES


class MotionTrackingEnv:
    """Reference tracking with scene-bound physics and explicit randomization."""
    def __init__(self, cfg: MotionTrackingCfg, *, num_envs=1, backend='mujoco-cpu', device=None,
                 seed=42, capture_graphs=True):
        self.cfg, self.num_envs = cfg, int(num_envs)
        profile = json.loads((Path(cfg.project_path)/cfg.robot_profile).read_text())
        self.profile = profile
        self.joint_names = tuple(profile['joint_names'])
        if self.joint_names != tuple(G1_JOINT_NAMES):
            raise ValueError('reference G1 control order differs from the authored profile')
        files = [str(Path(path).resolve()) for path in cfg.motion_files]
        if cfg.split != 'coverage40':
            allowed = {'train': (1,2,3), 'validation': (4,), 'holdout': (5,)}[cfg.split]
            for path in files:
                match = re.search(r'subject(\d+)', Path(path).stem)
                if match is None or int(match.group(1)) not in allowed:
                    raise ValueError(f'motion {path} violates split {cfg.split}')
        self.batch = ArticulationBatch(project_path=cfg.project_path, scene_path=cfg.scene_path,
            spec=RobotBatchSpec(cfg.robot_name, cfg.base_link, self.joint_names, tuple(TRACKED_BODIES)),
            num_envs=self.num_envs, backend=backend, device=device, physics_dt=cfg.physics_dt,
            decimation=cfg.decimation, solver_settings=profile['solver_settings'],
            contact_links=cfg.contact_links, capture_graphs=capture_graphs,
            sliding_friction=None if cfg.randomization.enabled else 1.0, deterministic=cfg.deterministic)
        self.device = self.batch.device
        self.step_dt = self.batch.control_dt
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.state = None
        try:
            kin = SceneKinematics(self.batch.snapshot, robot_name=cfg.robot_name, base_link=cfg.base_link,
                joint_names=self.joint_names, tracked_links=TRACKED_BODIES, device=self.device)
            motion = MotionLib(files, kin, self.device, src_fps=cfg.source_fps)
            self.default_q = torch.tensor(profile['default_joint_position'], device=self.device)
            self.action_scale = torch.tensor(profile['action_scale'], device=self.device)
            self.supermimic_mdp = MDP(kin, motion, self.default_q)
            self.f = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.xi = torch.zeros((self.num_envs, 33), device=self.device); self.xi[:,0] = 1.
            self.a_prev = torch.zeros((self.num_envs, 29), device=self.device)
            self.ep_step = torch.zeros_like(self.f)
            self.ep_ret = torch.zeros(self.num_envs, device=self.device)
            self.ep_disc_ret = torch.zeros_like(self.ep_ret)
            self.start_bin = torch.zeros_like(self.f)
            self.bin_G = torch.zeros(motion.num_bins, device=self.device)
            self.bin_w_avg = torch.full_like(self.bin_G, 1/motion.num_bins)
            self._task_signature = canonical_digest({
                'config':{k:v for k,v in asdict(cfg).items() if k not in ('project_path','motion_files')},
                'data':[(Path(path).name,file_digest(path)) for path in files],
                'physics':self.batch.artifact.digest,'kinematics':kin.digest,
            })
            if cfg.randomization.enabled:
                self._sample_startup_context()
                self._apply_physics_context()
            self.observation_spec = ObservationSpec('g1_tracking_proprio_v1', (
                SpecField('joint_position_offset',29,'rad'), SpecField('joint_velocity',29,'rad/s'),
                SpecField('base_angular_velocity_body',3,'rad/s'), SpecField('projected_gravity',3),
                SpecField('previous_action',29), SpecField('reference_orientation_6d',6)))
            self.goal_observation_spec = ObservationSpec('g1_tracking_future_v1', (SpecField('future',73*cfg.goal_frames),))
            self.action_spec = ActionSpec('g1_tracking_position_offset_v1',
                tuple(SpecField(name,1,'normalized_offset') for name in self.joint_names),
                lower=-np.finfo(np.float32).max, upper=np.finfo(np.float32).max)
            self.reset()
            self.critic_observation_spec = ObservationSpec('g1_tracking_critic_v1',
                (SpecField('tracking_features_and_future',self.state.obs['critic'].shape[-1]),))
        except BaseException:
            self.batch.close(); raise

    def _uniform(self, shape, low, high):
        lower=torch.as_tensor(low,device=self.device,dtype=torch.float32)
        upper=torch.as_tensor(high,device=self.device,dtype=torch.float32)
        return lower+(upper-lower)*torch.rand(shape,device=self.device,generator=self.generator)

    def _sample_startup_context(self):
        cfg=self.cfg.randomization
        self.xi[:,0]=self._uniform((self.num_envs,),*cfg.sliding_friction)
        self.xi[:,30:33]=self._uniform((self.num_envs,3),cfg.com_offset_min,cfg.com_offset_max)

    def _apply_physics_context(self):
        if not self.cfg.randomization.enabled:
            return
        robot=next(r for r in self.batch.snapshot['robots'] if r['name']==self.cfg.robot_name)
        friction={}
        for link in robot['links']:
            for shape in link.get('shapes',()):
                if shape['disabled']:
                    continue
                material=shape['material']
                value=torch.empty((self.num_envs,3),device=self.device)
                value[:,0]=self.xi[:,0]
                value[:,1]=material['torsional_friction'];value[:,2]=material['rolling_friction']
                friction[shape['name']]=value
        self.batch.configure_physics(
            link_center_of_mass_offsets={self.cfg.randomization.torso_link:self.xi[:,30:33]},
            shape_friction=friction)

    def training_state_dict(self):
        """Resume preserves sampling/model context and starts fresh episodes."""
        return {'version':1,'signature':self._task_signature,'num_envs':self.num_envs,
                'backend':self.batch.backend,'rng_device':torch.device(self.device).type,
                'generator':self.generator.get_state().cpu().clone(),
                'context':self.xi.detach().cpu().clone(),
                'bin_G':self.bin_G.detach().cpu().clone(),'bin_w_avg':self.bin_w_avg.detach().cpu().clone(),
                'resume_episodes':'fresh'}

    def load_training_state_dict(self, saved):
        if saved.get('version')!=1 or saved.get('signature')!=self._task_signature:
            raise ValueError('tracking checkpoint configuration/data changed; use an explicit fork')
        if saved.get('num_envs')!=self.num_envs or saved.get('backend')!=self.batch.backend or saved.get('rng_device')!=torch.device(self.device).type:
            raise ValueError('tracking resume requires the same batch size and backend')
        replacements={}
        for name,target in (('context',self.xi),('bin_G',self.bin_G),('bin_w_avg',self.bin_w_avg)):
            value=torch.as_tensor(saved[name],device=self.device,dtype=target.dtype)
            if value.shape!=target.shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f'invalid tracking checkpoint field {name}')
            replacements[name]=value
        if bool((replacements['bin_w_avg']<0).any()) or not bool(replacements['bin_w_avg'].sum()>0):
            raise ValueError('invalid adaptive frame-sampling distribution')
        context=replacements['context']
        if self.cfg.randomization.enabled:
            dr=self.cfg.randomization
            for values,low,high in ((context[:,0],dr.sliding_friction[0],dr.sliding_friction[1]),
                                    (context[:,1:30],dr.joint_offset[0],dr.joint_offset[1]),
                                    (context[:,30:33],dr.com_offset_min,dr.com_offset_max)):
                if bool(((values<torch.as_tensor(low,device=self.device)-1.e-6)|
                         (values>torch.as_tensor(high,device=self.device)+1.e-6)).any()):
                    raise ValueError('tracking context is outside configured randomization ranges')
        elif not bool(torch.all(context[:,0]==1) and torch.all(context[:,1:]==0)):
            raise ValueError('fixed-physics checkpoint has randomized context')
        generator_state=saved['generator'].cpu()
        probe_generator=torch.Generator(device=self.device)
        probe_generator.set_state(generator_state)
        self.xi.copy_(replacements['context']);self.bin_G.copy_(replacements['bin_G'])
        self.bin_w_avg.copy_(replacements['bin_w_avg'])
        self.generator.set_state(generator_state)
        self._apply_physics_context()
        self.reset()

    def physical_state(self):
        state = self.batch.read_state()
        return (state.base_pose[:,:3], quat_to_mat(state.base_pose[:,(6,3,4,5)]),
                state.base_velocity[:,:3], state.base_velocity[:,3:], state.joint_position, state.joint_velocity)

    def _current(self):
        return {'s': flatten_state(self.physical_state()).detach().clone(), 'f': self.f.clone(),
                'xi': self.xi.clone(), 'a_prev': self.a_prev.clone(), 'ep_step': self.ep_step.clone()}

    def _observations(self):
        mdp = self.supermimic_mdp
        state = self.physical_state()
        goal = mdp.goal(self.f, self.cfg.goal_frames)
        actor = mdp.obs(state, self.a_prev, self.xi, self.f)
        critic = torch.cat((mdp.critic_features(state,self.f,mdp.normalize_context(self.xi)),goal),-1)
        return {'actor': actor, 'goal': goal, 'critic': critic}

    def _ids(self, env_ids):
        if env_ids is None:
            return torch.arange(self.num_envs,device=self.device)
        ids = torch.as_tensor(env_ids,device=self.device)
        if ids.dtype == torch.bool:
            if ids.shape != (self.num_envs,): raise ValueError('invalid reset mask')
            ids = ids.nonzero().flatten()
        elif ids.dtype not in (torch.int32,torch.int64):
            raise TypeError('reset indices must be integer or boolean')
        ids = ids.long()
        if ids.ndim != 1 or ids.unique().numel()!=ids.numel() or bool(((ids<0)|(ids>=self.num_envs)).any()):
            raise ValueError('invalid reset indices')
        return ids

    def _sample_frames(self, count):
        motion = self.supermimic_mdp.motion
        soft = torch.softmax(-25*self.bin_G/(MAX_STEP_REWARD/(1-self.cfg.gamma)),0)
        self.bin_w_avg.mul_(.999).add_(soft,alpha=.001)
        probabilities = .5/motion.num_bins + .4*soft + .1*self.bin_w_avg/self.bin_w_avg.sum()
        bins = torch.multinomial(probabilities,count,replacement=True,generator=self.generator)
        start = motion.bin_start[bins]
        end = torch.minimum(start+BIN_FRAMES,motion.clip_end[motion.bin_clip[bins]])
        return start+(torch.rand(count,device=self.device,generator=self.generator)*(end-start)).long()

    @torch.no_grad()
    def reset(self, env_ids=None, *, seed=None, frames=None):
        ids = self._ids(env_ids)
        if not ids.numel() and self.state is not None:
            return self.state.obs, self.state.info
        if seed is not None:
            self.generator.manual_seed(seed)
        if self.cfg.randomization.enabled:
            self.xi[ids,1:30]=self._uniform((len(ids),29),*self.cfg.randomization.joint_offset)
        motion = self.supermimic_mdp.motion
        if frames is None:
            frames = motion.clip_start[ids % len(motion.names)] if self.cfg.eval_mode else self._sample_frames(len(ids))
        else:
            frames = torch.as_tensor(frames,device=self.device)
            if frames.dtype not in (torch.int32,torch.int64) or frames.shape != ids.shape:
                raise ValueError('reset frames must be integer and match selected environments')
            if bool(((frames<0)|(frames>=motion.num_frames)).any()):
                raise ValueError('reset frames outside motion library')
        self.f[ids] = frames
        p,r,v,w,q,qd = (value.clone() for value in motion.ref_state(self.f))
        if self.cfg.reset_noise and len(ids):
            def noise(lo,hi,width):
                low = torch.as_tensor(lo,device=self.device)
                return low+(torch.as_tensor(hi,device=self.device)-low)*torch.rand((len(ids),width),device=self.device,generator=self.generator)
            p[ids] += noise([-.05,-.05,-.01],[.05,.05,.01],3)
            r[ids] = so3_exp(noise([-.1,-.1,-.2],[.1,.1,.2],3)) @ r[ids]
            v[ids] += noise([-.5,-.5,-.2],[.5,.5,.2],3)
            w[ids] += noise([-.52,-.52,-.78],[.52,.52,.78],3)
            q[ids] += noise(-.1,.1,29)
        q = q.clamp(self.supermimic_mdp.kin.q_lo,self.supermimic_mdp.kin.q_hi)
        quaternion = mat_to_quat(r)
        pose = torch.cat((p,quaternion[:,(1,2,3,0)]),-1)
        mask = torch.zeros(self.num_envs,dtype=torch.bool,device=self.device); mask[ids] = True
        self.batch.reset(mask,base_pose=pose,base_velocity=torch.cat((v,w),-1),
                         joint_position=q,joint_velocity=qd,targets=q)
        self.ep_step[ids] = 0; self.a_prev[ids] = 0
        self.ep_ret[ids] = 0; self.ep_disc_ret[ids] = 0
        self.start_bin[ids] = motion.frame_bin[frames]
        self.state = BatchEnvState(self._observations(),torch.zeros(self.num_envs,device=self.device),
            torch.zeros_like(mask),torch.zeros_like(mask),{SUPERMIMIC_STATE_KEY:self._current()})
        return self.state.obs,self.state.info

    def reset_to_frames(self, frames, env_ids=None):
        return self.reset(env_ids,frames=frames)

    def init_state(self):
        if self.state is None: self.reset()
        return self.state

    def get_observations(self):
        return self.state.obs

    @torch.no_grad()
    def step(self, action):
        action = torch.as_tensor(action,dtype=torch.float32,device=self.device)
        if action.shape != self.a_prev.shape or not bool(torch.isfinite(action).all()):
            raise ValueError('tracking actions must be finite and match joint order')
        current = self._current()
        mdp, motion = self.supermimic_mdp, self.supermimic_mdp.motion
        self.batch.set_joint_targets(self.default_q+self.xi[:,1:30]+self.action_scale*action)
        self.batch.step_control()
        after = self.physical_state()
        clip_end = motion.clip_end[motion.clip_id[self.f]]
        next_frame = torch.minimum(self.f+1,clip_end)
        reward = mdp.reward(after,next_frame,action,self.a_prev)
        terminated = mdp.termination(after,next_frame)
        truncated = (self.f+1>=clip_end)
        if not self.cfg.eval_mode:
            truncated |= self.ep_step+1>=self.cfg.max_episode_steps
        done = terminated|truncated
        transition = {**current,'s1':flatten_state(after).clone(),'a':action.clone(),
                      'r':reward.clone(),'term':terminated.clone(),'done':done.clone()}
        self.ep_disc_ret.add_(self.cfg.gamma**self.ep_step.float()*reward)
        self.ep_ret.add_(reward); self.ep_step.add_(1)
        self.f.copy_(next_frame); self.a_prev.copy_(action)
        if not self.cfg.eval_mode and bool(done.any()):
            lengths = self.ep_step[done].float()
            boot = torch.where(terminated[done],0.,self.cfg.gamma**lengths*self.ep_ret[done]/lengths/(1-self.cfg.gamma))
            bins = self.start_bin[done]
            self.bin_G.index_put_((bins,),.8*self.bin_G[bins]+.2*(self.ep_disc_ret[done]+boot))
            transition.update(ep_len=lengths.clone(),ep_ret_norm=self.ep_ret[done]/(lengths*MAX_STEP_REWARD))
        final_observation = self._observations()
        if self.cfg.autoreset and bool(done.any()):
            self.reset(done)
        self.state = BatchEnvState(self._observations(),reward,terminated,truncated,
            {SUPERMIMIC_STATE_KEY:self._current(),SUPERMIMIC_TRANSITION_KEY:transition,
             'final_observation':final_observation,'_final_observation':done,
             'terminal_frames':next_frame.clone()},final_observation=final_observation)
        return self.state

    def close(self):
        self.batch.close()


def make_motion_tracking_env(cfg, **kwargs):
    return MotionTrackingEnv(cfg, **kwargs)
