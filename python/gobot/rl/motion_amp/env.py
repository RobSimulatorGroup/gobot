"""AMP transitions over Gobot's typed articulation/state/contact boundaries."""
from __future__ import annotations
import json
import math
import warnings
from pathlib import Path
import torch
from gobot.rl.articulation import ArticulationBatch
from gobot.rl.batch import BatchEnvState
from gobot.rl.kinematics import SceneKinematics
from gobot.sim.providers.base import RobotBatchSpec
from .features import Features
from .motion import MotionLibrary


class MotionAMPEnv:
    is_vector_env = True
    accepts_device_actions = True

    def __init__(self, project_path, *, scene_path='res://world.jscn', motion='walk',
                 num_envs=16, backend='mujoco-cpu', device=None, seed=1,
                 capture_graphs=True, autoreset=False, velocity_reference='base_body_com', diagnostics_path=None):
        self.project_path = Path(project_path).resolve()
        self.scene_path = scene_path
        self.profile = p = json.loads((self.project_path/'motion_amp/profile.json').read_text())
        if p.get('schema') != 'gobot_motion_amp_export_v1':
            raise ValueError('unsupported motion AMP profile')
        self.motion_name = motion
        self.num_envs = num_envs
        self.seed = seed
        self.autoreset = autoreset
        self.diagnostics_path = Path(diagnostics_path) if diagnostics_path is not None else None
        self.numerical_failure_count = 0
        self._control_steps = 0
        self.device = device or ('cuda:0' if backend == 'mujoco-warp' else 'cpu')
        if velocity_reference!='base_body_com':
            raise ValueError('the aligned AMP task requires base_body_com velocity')
        self.velocity_reference = velocity_reference
        # The archived motion velocity is the derivative of root-origin position.
        # Legacy exports mislabeled it as COM velocity; only Newton's live state
        # and interpretation at reset use COM velocity.
        if p.get('motion_root_velocity_reference', 'link_origin') != 'link_origin':
            raise ValueError('unsupported archived motion velocity convention')
        self.history_steps = int(p['history_steps'])
        self.control_dt = p['physics_dt']*p['decimation']
        self.max_steps = round(p['episode_length_s']/self.control_dt)
        if self.history_steps < 2 or self.max_steps < 1 or not math.isclose(self.max_steps*self.control_dt, p['episode_length_s']):
            raise ValueError('invalid AMP history or episode duration')
        self.max_episode_length = self.max_steps
        self.cfg = {'task':'g1-motion-amp','motion':motion,'physics_dt':p['physics_dt'],'decimation':p['decimation']}
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        spec = RobotBatchSpec(p['robot_name'], p['base_link'], tuple(p['joint_names']), tuple(p['body_names']))
        self.batch = ArticulationBatch(project_path=self.project_path, scene_path=scene_path,
            spec=spec, num_envs=num_envs, backend=backend, device=self.device,
            physics_dt=p['physics_dt'], decimation=p['decimation'], solver_settings=p['solver_settings'],
            contact_links=p['body_names'], capture_graphs=capture_graphs, contact_scope='static_world')
        try:
            robot = next(r for r in self.batch.snapshot['robots'] if r['name'] == p['robot_name'])
            base = next(link for link in robot['links'] if link['name'] == p['base_link'])
            self.features = Features(p, self.device, base_center_of_mass=base['center_of_mass'],
                                     velocity_reference=velocity_reference)
            self.observation_spec, self.action_spec = self.features.obs_spec, self.features.action_spec
            self.num_obs, self.num_actions = self.observation_spec.dim, self.action_spec.dim
            kin = SceneKinematics(self.batch.snapshot, robot_name=p['robot_name'], base_link=p['base_link'],
                joint_names=p['joint_names'], device=self.device)
            self.motion = MotionLibrary(self.project_path/'motion_amp/motions.npz', p, motion, self.features, kin, self.device)
            self.history = torch.zeros(num_envs, self.history_steps, self.features.frame_dim, device=self.device)
            self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
            self.episode_length_buf = self.steps
            self._actor_buffers = [torch.empty(num_envs,self.num_obs,device=self.device) for _ in range(3)]
            self._buffer_index = 0
            self.state = None
            self.pending_reset = torch.ones(num_envs, dtype=torch.bool, device=self.device)
            self.fail_links = [i for i,n in enumerate(p['body_names']) if n not in p['contact_bodies']]
            self.low = torch.tensor(p['action_low'], device=self.device)
            self.high = torch.tensor(p['action_high'], device=self.device)
            self.reset(seed=seed)
        except BaseException:
            self.close()
            raise

    def _result(self, terminated=None, truncated=None, *, actor=None, valid=None):
        if actor is None:actor, _ = self.features.live(self.batch.read_state())
        disc = self.features.discriminator(self.history)
        zero = torch.zeros(self.num_envs, device=self.device)
        terminated = zero.bool() if terminated is None else terminated
        truncated = zero.bool() if truncated is None else truncated
        if self.autoreset:
            # RSL-RL stores the current observation before the next env.step.
            # Alternate persistent buffers so stepping cannot overwrite it.
            buffer = self._actor_buffers[self._buffer_index]
            self._buffer_index = (self._buffer_index+1)%3
            buffer.copy_(actor)
            obs = {'actor':buffer}
        else:
            obs = {'actor':actor, 'disc':disc}
        valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device) if valid is None else valid
        info = {'steps':self.steps.clone(), 'transition_observation':{'actor':actor,'disc':disc,'valid':valid}}
        return BatchEnvState(obs,zero,terminated,truncated,info,
            final_observation={'actor':actor.clone()} if bool((terminated|truncated).any()) else None)

    def _record_numerical_failure(self, invalid, checks, physical, contacts, action):
        ids=invalid.nonzero().flatten()
        first=self.numerical_failure_count==0
        self.numerical_failure_count+=ids.numel()
        if self.diagnostics_path is not None:
            self.diagnostics_path.mkdir(parents=True,exist_ok=True)
            record={'control_step':self._control_steps,'env_ids':ids.cpu().tolist(),
                'total':self.numerical_failure_count,
                'nonfinite_fields':{name:(~valid).nonzero().flatten().cpu().tolist()
                                    for name,valid in checks.items() if not bool(valid.all())}}
            with (self.diagnostics_path/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            if first:
                selected=ids[:16]
                snapshot={'record':record,'sample_env_ids':selected.cpu(),'steps':self.steps[selected].cpu(),
                    'velocity_reference':self.velocity_reference,'action':action[selected].cpu(),
                    'previous_actor_observation':self.state.obs['actor'][selected].cpu(),
                    'previous_history':self.history[selected].cpu(),
                    'physical':{k:v[selected].cpu() for k,v in physical.__dict__.items()},
                    'contacts':{k:v[selected].cpu() for k,v in contacts.items()}}
                torch.save(snapshot,self.diagnostics_path/'first.pt')
        if first:
            warnings.warn(f'AMP numerical failure in {ids.numel()} environments; terminating affected episodes. '
                          f'Diagnostics: {self.diagnostics_path}',RuntimeWarning)

    @torch.no_grad()
    def reset(self, *, seed=None, mask=None, phase=None):
        if seed is not None:
            self.generator.manual_seed(seed)
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        if mask.shape != (self.num_envs,):
            raise ValueError('reset mask must have one value per lane')
        ids = mask.nonzero().flatten()
        if ids.numel():
            times = self.motion.sample_times(ids.numel(), self.generator) if phase is None else torch.as_tensor(phase, device=self.device, dtype=torch.float32)
            if times.shape != (ids.numel(),) or not bool(torch.isfinite(times).all()):
                raise ValueError('phase must be finite seconds for each selected lane')
            p, r, v, w, j, qd = self.motion.sample(times)
            q = self.features.joint_coordinates(j)
            state = self.batch.read_state()
            pose, velocity = state.base_pose.clone(), state.base_velocity.clone()
            position, speed = state.joint_position.clone(), state.joint_velocity.clone()
            targets = state.joint_control.clone()
            # Preserve the trained COM-v2 task's reset interpretation while the
            # generic ArticulationBatch continues to use link-origin velocity.
            origin_velocity = self.features.root_origin_velocity(r, v, w)
            pose[ids] = torch.cat((p,r), -1); velocity[ids] = torch.cat((origin_velocity,w), -1)
            position[ids], speed[ids], targets[ids] = q, qd, q
            self.batch.reset(mask, base_pose=pose, base_velocity=velocity,
                joint_position=position, joint_velocity=speed, targets=targets)
            self.history[ids] = self.motion.history(times, self.history_steps, self.control_dt)
            self.steps[ids] = 0
            self.pending_reset[ids] = False
        self.state = self._result()
        return self.state

    @torch.no_grad()
    def step(self, action):
        if bool(self.pending_reset.any()):
            raise RuntimeError('reset finished lanes explicitly before the next step')
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.shape != (self.num_envs,self.action_spec.dim) or not bool(torch.isfinite(action).all()):
            raise ValueError('actions must be finite with shape [num_envs, num_joints]')
        self.batch.set_joint_targets(action.clamp(self.low, self.high))
        physical = self.batch.step_control()
        contacts = self.batch.read_contacts()
        actor, frame = self.features.live(physical)
        values={**physical.__dict__,**{f'contact_{k}':v for k,v in contacts.items()},
                'actor':actor,'amp_frame':frame,'history':self.history}
        checks={name:torch.isfinite(value).reshape(self.num_envs,-1).all(-1) for name,value in values.items()}
        valid=torch.stack(list(checks.values())).all(0)
        self._control_steps+=1
        if not bool(valid.all()):
            self._record_numerical_failure(~valid,checks,physical,contacts,action)
            # Finite placeholders are never used as expert/policy AMP samples or
            # timeout bootstrap values. Keep valid lanes' observations unchanged.
            actor=actor.masked_fill(~valid[:,None],0.)
            frame=frame.masked_fill(~valid[:,None],0.)
            self.history[~valid]=0.
        self.history = torch.cat((self.history[:, 1:], frame[:, None]), dim=1)
        self.steps += 1
        force = contacts['force'][:, self.fail_links]
        terminated = (force.abs() > .1).any(dim=(-1,-2)) | ~valid
        truncated = self.steps >= self.max_steps
        self.pending_reset.copy_(terminated|truncated)
        state = self._result(terminated,truncated,actor=actor,valid=valid)
        if self.autoreset and bool(state.done.any()):
            # Keep the physical terminal AMP history and critic observation.
            # Reset observations enter only the next policy action.
            reset = self.reset(mask=state.done)
            state = state.replace(obs=reset.obs)
        self.state = state
        return state

    @torch.no_grad()
    def sample_demonstrations(self, count):
        times = self.motion.sample_times(count, self.generator)
        return self.features.discriminator(self.motion.history(times, self.history_steps, self.control_dt))

    def close(self):
        self.batch.close()
