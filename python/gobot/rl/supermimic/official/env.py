"""Official GenCoord task on Gobot-owned scenes and articulation services."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

from gobot.rl.articulation import ArticulationBatch
from gobot.rl.kinematics import SceneKinematics
from gobot.sim.providers.base import RobotBatchSpec
from gobot.rl.supermimic.geometry import mat_to_quat
from gobot.rl.supermimic.rewards import TrackingRewardCfg, reanchor_reference, tracking_reward_terms
from gobot.rl.motion_tracking.env import euler_quat
from .motion import MotionLibrary, digest
from .rotations import matrix_from_quat, quat_apply, quat_apply_inverse, quat_mul
from .sampler import ReturnSampler

MODEL_PROTOCOL = 'gobot_supermimic_official_model_v1'


class SceneKinematicsAdapter(nn.Module):
    def __init__(self, scene):
        super().__init__()
        self.scene = scene

    def forward(self, root_pos, root_quat, joints, linear_world, angular_world, velocities):
        p,r,v,w = self.scene(root_pos,matrix_from_quat(root_quat),linear_world,angular_world,joints,velocities)
        return p,mat_to_quat(r),v,w


class TrackingEnv:
    def __init__(self, cfg):
        self.cfg=cfg
        self.profile_path=Path(cfg.project)/'assets/supermimic/robot_profile.json'
        self.profile=json.loads(self.profile_path.read_text())
        p=self.profile
        if p.get('protocol')!=MODEL_PROTOCOL or p['anchor_link']!='pelvis':
            raise ValueError('prepare an audited official G1 scene; reproduced profiles are not interchangeable')
        self.profile_digest=digest(self.profile_path)
        audit=json.loads(self.profile_path.with_name('world_physics_audit.json').read_text())
        if not audit.get('passed') or audit.get('profile_sha256')!=self.profile_digest:
            raise ValueError('official scene profile must have a passing static model audit')
        self.audited_artifact_digest=audit['artifact_digest']
        self.num_envs=cfg.num_envs
        self.joint_names=tuple(p['joint_names']); self.body_names=tuple(p['tracked_links'])
        if self.body_names[0]!='pelvis' or len(self.joint_names)!=29:
            raise ValueError('official GenCoord requires pelvis first and 29 named joints')
        self.batch=ArticulationBatch(project_path=cfg.project,scene_path=cfg.scene,
            spec=RobotBatchSpec(p['robot_name'],p['base_link'],self.joint_names,self.body_names),
            num_envs=cfg.num_envs,backend=cfg.backend,device=cfg.device,physics_dt=p['physics_dt'],
            decimation=p['decimation'],solver_settings=p['solver_settings'],
            capture_graphs=cfg.capture_graphs,nconmax=cfg.nconmax,njmax=cfg.njmax)
        try:
            if self.batch.artifact.content_digest != self.audited_artifact_digest:
                raise ValueError('compiled official scene differs from the audited model; re-audit scene changes')
            self._initialize()
        except BaseException:
            self.batch.close()
            raise

    def _initialize(self):
        cfg,p=self.cfg,self.profile
        self.device=self.batch.device; self.step_dt=self.batch.control_dt
        if not math.isclose(self.step_dt,.02):raise ValueError('official G1 motions require 50 Hz')
        self.default_q=torch.tensor(p['default_joint_position'],device=self.device)
        self.action_scale=torch.tensor(p['action_scale'],device=self.device)
        scene=SceneKinematics(self.batch.snapshot,robot_name=p['robot_name'],base_link=p['base_link'],
            joint_names=self.joint_names,tracked_links=self.body_names,device=self.device)
        self.fk=SceneKinematicsAdapter(scene)
        midpoint=(scene.q_lo+scene.q_hi)/2
        half=(scene.q_hi-scene.q_lo)*p['soft_joint_limit_factor']/2
        self.soft_limits=torch.stack((midpoint-half,midpoint+half),-1)
        self.motion=MotionLibrary(cfg.library,self.fk,self.joint_names,self.device,verify_hashes=cfg.verify_data)
        self.max_episode_steps=math.ceil(cfg.episode_length_s/self.step_dt)
        self.ids=torch.arange(self.num_envs,device=self.device)
        self.frame=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.episode_steps=torch.zeros_like(self.frame)
        self.action=torch.zeros(self.num_envs,29,device=self.device)
        self.terminated=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.truncated=torch.zeros_like(self.terminated)
        self.motion_end=torch.zeros_like(self.terminated)
        self.pending_reset=torch.zeros_like(self.terminated)
        self.bin_count=int(self.motion.time_step_total//(.2/self.step_dt))+1
        self.distribution=torch.full((self.bin_count,),1./self.bin_count,device=self.device)
        self.sampler=ReturnSampler(cfg.sampler,self.bin_count,self.motion.time_step_total,self.num_envs,
                                  self.step_dt*5.,self.device) if cfg.sampling_mode=='return' else None
        self.reward_cfg=TrackingRewardCfg(dt=self.step_dt)
        self.context_mean=torch.tensor(p['context_mean'],device=self.device)
        self.context_scale=torch.tensor(p['context_scale'],device=self.device)
        self.context=torch.zeros(self.num_envs,33,device=self.device)
        self.context[:,0]=p['nominal_evaluation_foot_friction']
        self.context[:,1:4]=torch.tensor(p['com_nominal'],device=self.device)
        if cfg.domain_randomization:
            # All three events are startup events in the pinned official task.
            self.context[:,1:4]+=(torch.rand(self.num_envs,3,device=self.device)*2-1)*torch.tensor([.025,.05,.05],device=self.device)
            self.context[:,4:]=(torch.rand(self.num_envs,29,device=self.device)*2-1)*.01
            self.context[:,0]=.3+.9*torch.rand(self.num_envs,device=self.device)
        self.apply_context()

    def apply_context(self):
        p=self.profile
        robot=next(r for r in self.batch.snapshot['robots'] if r['name']==p['robot_name'])
        shapes={s['name']:s for link in robot['links'] for s in link.get('shapes',())}
        friction={}
        for name in p['foot_shapes']:
            material=shapes[name]['material']
            value=torch.empty(self.num_envs,3,device=self.device)
            value[:,0]=self.context[:,0]
            value[:,1]=material['torsional_friction']; value[:,2]=material['rolling_friction']
            friction[name]=value
        offsets=self.context[:,1:4]-torch.tensor(p['com_nominal'],device=self.device)
        self.batch.configure_physics(link_center_of_mass_offsets={p['com_link']:offsets},shape_friction=friction)

    def normalized_context(self):
        return (self.context-self.context_mean)/self.context_scale

    def full_state(self):
        state=self.batch.read_state()
        return torch.cat((state.base_pose[:,:3],state.base_pose[:,(6,3,4,5)],state.joint_position,
                          state.base_velocity[:,:3],state.base_velocity[:,3:],state.joint_velocity),-1)

    def body_state(self, state):
        return self.fk(state[:,:3],state[:,3:7],state[:,7:36],state[:,36:39],state[:,39:42],state[:,42:])

    def sensor(self, state):
        # Official full_state is privileged; the optional proprio actor instead
        # reads the real biased joint sensor and configured IMU site frame.
        down=torch.zeros_like(state[:,:3]);down[:,2]=-1
        bodies=self.body_state(state)
        imu=self.profile['imu']
        index=self.body_names.index(imu['body'])
        orientation=quat_mul(bodies[1][:,index],torch.tensor(imu['quaternion'],device=self.device).expand(self.num_envs,-1))
        return torch.cat((quat_apply_inverse(state[:,3:7],down),quat_apply_inverse(orientation,bodies[3][:,index]),
                          state[:,7:36]+self.context[:,4:]-self.default_q,state[:,42:],self.action),-1)

    def sample_frames(self, count):
        total=self.motion.time_step_total
        if self.cfg.sampling_mode=='start':
            frames=torch.zeros(count,dtype=torch.long,device=self.device)
        elif self.cfg.sampling_mode=='uniform':
            frames=torch.randint(0,total-1,(count,),device=self.device)
        else:
            bins=torch.multinomial(self.distribution,count,replacement=True).clamp_(max=self.bin_count-1)
            lo=bins*(total-2)//self.bin_count
            span=((bins+1)*(total-2)//self.bin_count-lo).clamp_(min=1)
            frames=(lo+(torch.rand(count,device=self.device)*span).long()).clamp_(max=total-2)
        ends=self.motion.motion_ends
        index=torch.searchsorted(ends,frames+1).clamp(max=len(ends)-1)
        return torch.where(ends[index]==frames+1,frames-1,frames)

    def _reset(self, ids, frames):
        self.frame[ids]=frames
        motion=self.motion
        pos=motion.root_pos_w[frames].clone(); ori=motion.root_quat_w[frames].clone()
        lin=motion.root_lin_vel_w[frames].clone(); ang=motion.root_ang_vel_w[frames].clone()
        q=motion.joint_pos[frames].clone(); qd=motion.joint_vel[frames].clone()
        if self.cfg.reset_noise:
            pose=(torch.rand(len(ids),6,device=self.device)*2-1)*torch.tensor([.05,.05,.01,.1,.1,.2],device=self.device)
            vel=(torch.rand(len(ids),6,device=self.device)*2-1)*torch.tensor([.5,.5,.2,.52,.52,.78],device=self.device)
            pos+=pose[:,:3];ori=quat_mul(euler_quat(pose[:,3:]),ori)
            lin+=vel[:,:3];ang+=vel[:,3:]
            q+=(torch.rand_like(q)*2-1)*.1
        # Preserve MuJoCo's world->free-joint angular conversion for quantized
        # reference quaternions, then provide the scene API a unit orientation.
        angular_body=quat_apply_inverse(ori,ang)
        ori=ori/ori.norm(dim=-1,keepdim=True)
        ang=quat_apply(ori,angular_body)
        q=q.clamp(self.soft_limits[:,0],self.soft_limits[:,1])
        state=self.batch.read_state()
        bp,bv,jp,jv,controls=(x.clone() for x in (state.base_pose,state.base_velocity,state.joint_position,state.joint_velocity,state.joint_control))
        bp[ids]=torch.cat((pos,ori[:,(1,2,3,0)]),-1);bv[ids]=torch.cat((lin,ang),-1)
        jp[ids]=q;jv[ids]=qd;controls[ids]=0.
        mask=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device);mask[ids]=True
        self.batch.reset(mask,base_pose=bp,base_velocity=bv,joint_position=jp,joint_velocity=jv,targets=controls)
        self.action[ids]=0.;self.episode_steps[ids]=0;self.pending_reset[ids]=False

    def _advance(self):
        self.frame+=1
        # At clip ends terminated lanes have already been reset. The final
        # array boundary should never be reached by a valid training transition.
        if bool((self.frame>=self.motion.time_step_total).any()):
            raise RuntimeError('motion advanced beyond library; reset completed clips before stepping')
        self.reference=self.motion.body_state(self.frame)
        state=self.full_state()
        self.relative_pose=reanchor_reference(self.reference[0],self.reference[1],self.reference[0][:,0],
            self.reference[1][:,0],state[:,:3],state[:,3:7])

    @torch.no_grad()
    def reset(self, frames=None):
        if frames is None:frames=self.sample_frames(self.num_envs)
        frames=torch.as_tensor(frames,dtype=torch.long,device=self.device)
        if frames.shape!=(self.num_envs,) or bool(((frames<0)|(frames>=self.motion.time_step_total-1)).any()):
            raise ValueError('reset frames must have one valid pre-final frame per environment')
        self._reset(self.ids,frames)
        self._advance()
        return self.full_state()

    def _before_substep(self, substep):
        if substep==self.batch.decimation-1:
            self.batch.refresh_kinematics()
            self.lag_state=self.full_state()

    @torch.no_grad()
    def step(self, actions):
        if self.pending_reset.any():raise RuntimeError('reset required before another step')
        if actions.shape!=self.action.shape or not torch.isfinite(actions).all():
            raise ValueError('actions must be finite [num_envs,29]')
        previous=self.action.clone(); frame=self.frame.clone()
        self.action.copy_(actions)
        self.batch.set_joint_targets(self.default_q+actions*self.action_scale-self.context[:,4:])
        self.batch.step_control(before_substep=self._before_substep)
        self.batch.assert_no_overflow()
        terminal=self.full_state()
        body=self.body_state(self.lag_state)
        terms=tracking_reward_terms(robot_body_pos_w=body[0],robot_body_quat_w=body[1],
            robot_body_lin_vel_w=body[2],robot_body_ang_vel_w=body[3],ref_body_pos_w=self.reference[0],
            ref_body_quat_w=self.reference[1],ref_body_lin_vel_w=self.reference[2],ref_body_ang_vel_w=self.reference[3],
            anchor_index=0,action=actions,prev_action=previous,joint_pos=terminal[:,7:36],
            soft_limits=self.soft_limits,cfg=self.reward_cfg,relative_body_pose=self.relative_pose)
        self.terminated=(body[0][:,0,2]-self.reference[0][:,0,2]).abs()>self.profile['height_threshold']
        down=torch.zeros_like(terminal[:,:3]);down[:,2]=-1.
        self.terminated|=(quat_apply_inverse(body[1][:,0],down)[:,2]-quat_apply_inverse(self.reference[1][:,0],down)[:,2]).abs()>self.profile['tilt_threshold']
        self.episode_steps+=1
        ends=self.motion.motion_ends
        index=torch.searchsorted(ends,frame+1).clamp(max=len(ends)-1)
        self.motion_end=ends[index]==frame+1
        self.truncated=(self.episode_steps>=self.max_episode_steps)|self.motion_end
        if not torch.isfinite(terminal).all() or not torch.isfinite(terms['total']).all():
            raise FloatingPointError('nonfinite official scene transition')
        done=self.terminated|self.truncated
        ids=done.nonzero().flatten()
        if len(ids):self._reset(ids,self.sample_frames(len(ids)))
        self._advance()
        # Official runner updates its distribution after env.step/autoreset.
        if self.sampler is not None:
            probs=self.sampler.record(frame,terms['total'],self.terminated&~self.truncated,
                                      self.motion_end,self.truncated&~self.motion_end)
            if probs is not None:self.distribution=probs.detach()/probs.sum()
        return self.full_state(),terms['total'],self.terminated.clone(),self.truncated.clone(),{
            'terminal_state':terminal,'target_frame':frame,'reward_terms':terms,'reset_ids':ids}

    def close(self):
        self.batch.close()
