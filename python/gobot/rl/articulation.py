"""A control clock over Gobot's existing native/provider batch boundaries."""
from __future__ import annotations
import math
import numpy as np
import torch
import gobot
from gobot.sim.providers.base import CompiledSceneArtifact, RobotBatchSpec, RobotBatchState
from .runtime import BatchSimulationRuntime


class ArticulationBatch:
    """Named PD targets, world link-origin state, and explicit masked reset.

    State/contact tensors are borrowed stable storage. Tasks must copy transitions
    to replay before stepping or resetting. No reward or episode logic lives here.
    """

    def __init__(self, *, project_path, scene_path, spec: RobotBatchSpec, num_envs,
                 backend='mujoco-cpu', device=None, physics_dt=.005, decimation=4,
                 solver_settings=None, contact_links=(), capture_graphs=True, nconmax=128, njmax=500,
                 sliding_friction=None, deterministic=False):
        if int(num_envs) != num_envs or num_envs <= 0 or int(decimation) != decimation or decimation <= 0:
            raise ValueError('batch size and decimation must be positive integers')
        if not math.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError('physics_dt must be finite and positive')
        if backend not in ('mujoco-cpu', 'mujoco-warp'):
            raise ValueError(f'unsupported articulation backend {backend!r}')
        self.spec, self.num_envs = spec, int(num_envs)
        self.physics_dt, self.decimation = float(physics_dt), int(decimation)
        self.control_dt = self.physics_dt * self.decimation
        self.backend = backend
        self.device = str(device or ('cuda:0' if backend=='mujoco-warp' else 'cpu'))
        if backend == 'mujoco-cpu' and self.device != 'cpu':
            raise ValueError('native CPU articulation state uses CPU tensors')
        self.contact_links = tuple(contact_links)
        self._physics_parameters = {"link_center_of_mass_offsets": {}, "shape_friction": {}}
        self._cpu_parameters = {}
        self._provider = None; self._closed = False
        self.context = gobot.app.create_context()
        try:
            self.context.set_project_path(str(project_path))
            self.context.load_scene(str(scene_path))
            if sliding_friction is not None:
                if not math.isfinite(sliding_friction) or sliding_friction < 0:
                    raise ValueError('sliding friction must be finite and nonnegative')
                pending = [self.context.root]
                while pending:
                    node = pending.pop(); pending.extend(node.children)
                    if node.type_name == 'CollisionShape3D':
                        material = dict(node.physics_material or {})
                        material = dict(material.get('properties', material))
                        material['sliding_friction'] = float(sliding_friction)
                        node.physics_material = material
            self.context.fixed_time_step = self.physics_dt
            if solver_settings:
                self.context.set_mujoco_solver_settings(dict(solver_settings))
            self.snapshot = self.context.compile_physics_scene_snapshot()
            self.artifact = CompiledSceneArtifact.from_mapping(
                self.context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu))
            self._targets = torch.zeros((self.num_envs, len(spec.joint_names)), device=self.device)
            shape = (self.num_envs, len(self.contact_links))
            self._contacts = {'force': torch.zeros((*shape, 3), device=self.device),
                              'impulse': torch.zeros((*shape, 3), device=self.device),
                              'peak_force_norm': torch.zeros(shape, device=self.device),
                              'found': torch.zeros(shape, dtype=torch.bool, device=self.device)}
            if backend == 'mujoco-warp':
                from gobot.sim.providers.mujoco_warp import MuJoCoWarpProvider, MuJoCoWarpContactSensorSpec
                prefix = self.artifact.robot_prefix(spec.robot_name)
                sensors = () if not self.contact_links else (MuJoCoWarpContactSensorSpec(
                    name='articulation_contacts', primary_type='body',
                    primary_names=tuple(prefix+name for name in self.contact_links), reduce='netforce'),)
                self._provider = MuJoCoWarpProvider(self.artifact, num_envs=self.num_envs, device=self.device,
                    nconmax=nconmax, njmax=njmax, capture_graphs=capture_graphs, deterministic=deterministic, contact_sensors=sensors)
                self._view = self._provider.create_robot_view(spec)
                self._state = self._view.read_state()
            else:
                self.context.build_world(gobot.PhysicsBackendType.MuJoCoCpu)
                robot = next(r for r in self.snapshot['robots'] if r['name']==spec.robot_name)
                self._physical_links = tuple(link['name'] for link in robot['links'] if link.get('role')!='virtual_root')
                links = tuple(dict.fromkeys((*spec.link_names, *self.contact_links, *self._physical_links)))
                self._runtime = BatchSimulationRuntime(self.context, robot=spec.robot_name,
                    base_link=spec.base_link, joint_names=spec.joint_names, link_names=links)
                self._runtime.configure(self.num_envs)
                self._link_indices = [links.index(name) for name in spec.link_names]
                self._contact_indices = [links.index(name) for name in self.contact_links]
                self._state = RobotBatchState(torch.empty(self.num_envs, 7), torch.empty(self.num_envs, 6),
                    torch.empty_like(self._targets), torch.empty_like(self._targets), torch.empty_like(self._targets),
                    torch.empty(self.num_envs, len(spec.link_names), 7))
                self._targets.copy_(torch.as_tensor(self._runtime.robot_state()['joint_target_position']))
                self._read_cpu()
            self._targets.copy_(self._state.joint_control)
        except BaseException:
            self.close()
            raise

    def _require_open(self):
        if self._closed:
            raise RuntimeError('articulation is closed')

    def _tensor(self, value, shape, name):
        value = torch.as_tensor(value, dtype=self._targets.dtype, device=self.device)
        if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f'{name} must have finite shape {shape}')
        return value

    def _read_cpu(self, raw=None):
        # The typed batch service applies authored actuator parameters directly.
        # A zero-tick read refreshes kinematics without advancing physics time.
        if raw is None:
            raw = self._runtime.step_and_robot_state(self._targets.numpy(), 0, kinematic_readback=True, **self._cpu_parameters)
        state = self._state
        def array(name):
            return torch.as_tensor(np.asarray(raw[name]), dtype=torch.float32)
        state.base_pose[:, :3].copy_(array('base_position'))
        state.base_pose[:, 3:].copy_(array('base_quaternion')[:, (1, 2, 3, 0)])
        state.base_velocity[:, :3].copy_(array('base_linear_velocity'))
        state.base_velocity[:, 3:].copy_(array('base_angular_velocity'))
        state.joint_position.copy_(array('joint_position'))
        state.joint_velocity.copy_(array('joint_velocity'))
        state.joint_control.copy_(array('joint_target_position'))
        state.link_pose[..., :3].copy_(array('link_position')[:, self._link_indices])
        state.link_pose[..., 3:].copy_(array('link_quaternion')[:, self._link_indices][..., (1, 2, 3, 0)])
        self._raw_cpu = raw
        return state

    def read_state(self):
        self._require_open()
        return self._view.read_state() if self._provider is not None else self._state

    @torch.no_grad()
    def refresh_kinematics(self):
        """Refresh link state between ticks without a contact/force solve."""
        self._require_open()
        if self._provider is not None:
            self._provider.refresh_kinematics()
        else:
            self._read_cpu()
        return self.read_state()

    @torch.no_grad()
    def set_joint_targets(self, targets):
        self._require_open()
        self._targets.copy_(self._tensor(targets, tuple(self._targets.shape), 'joint targets'))
        if self._provider is not None:
            self._view.set_position_targets(self._targets)
        else:
            self._runtime.set_joint_position_targets(self._targets.numpy())

    def _sample_contacts(self):
        if not self.contact_links:
            return
        force = self._contacts['force']; force.zero_()
        if self._provider is not None:
            source = self._provider.contact_sensor('articulation_contacts')
            force.copy_(source['force'].reshape_as(force))
            self._contacts['found'].logical_or_(source['found'].reshape_as(self._contacts['found']) > 0)
        else:
            raw = self._raw_cpu
            link_ids = np.asarray(raw['contact_link_index'])
            counts = np.asarray(raw['contact_count'])
            valid = np.arange(link_ids.shape[1])[None, :] < counts[:, None]
            values = np.asarray(raw['contact_force'])
            for i, link in enumerate(self._contact_indices):
                # Typed batch contacts contain one pair, with force on side B.
                side_a = valid & (link_ids[..., 0] == link)
                side_b = valid & (link_ids[..., 1] == link)
                sign = side_b.astype(np.int8) - side_a.astype(np.int8)
                force[:, i].copy_(torch.as_tensor((values*sign[..., None]).sum(1), dtype=force.dtype))
                self._contacts['found'][:, i].logical_or_(torch.as_tensor((side_a|side_b).any(1)))
        self._contacts['impulse'].add_(force, alpha=self.physics_dt)
        torch.maximum(self._contacts['peak_force_norm'], force.norm(dim=-1), out=self._contacts['peak_force_norm'])

    @torch.no_grad()
    def step_control(self, *, before_substep=None):
        self._require_open()
        for value in self._contacts.values():
            value.zero_()
        for substep in range(self.decimation):
            if before_substep is not None:
                before_substep(substep)
            if self._provider is not None:
                self._provider.step(nsteps=1)
                if self.contact_links:
                    self._provider.sense()
            else:
                self._read_cpu(self._runtime.step_and_robot_state(self._targets.numpy(), 1, **self._cpu_parameters))
            self._sample_contacts()
        if self._provider is not None:
            self._provider.refresh_kinematics()
        else:
            self._read_cpu()
        return self.read_state()

    @torch.no_grad()
    def reset(self, mask, *, base_pose, base_velocity, joint_position, joint_velocity, targets):
        self._require_open()
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        if mask.shape != (self.num_envs,):
            raise ValueError('reset mask must have one entry per environment')
        pose = self._tensor(base_pose, (self.num_envs, 7), 'base pose')
        velocity = self._tensor(base_velocity, (self.num_envs, 6), 'base velocity')
        q = self._tensor(joint_position, tuple(self._targets.shape), 'joint position')
        qd = self._tensor(joint_velocity, tuple(self._targets.shape), 'joint velocity')
        targets = self._tensor(targets, tuple(self._targets.shape), 'reset targets')
        if not bool(torch.allclose(pose[:, 3:].norm(dim=-1), torch.ones(self.num_envs, device=self.device), atol=1.e-4)):
            raise ValueError('reset orientation must be a unit xyzw quaternion')
        if self._provider is not None:
            self._view.reset(mask, base_pose=pose, base_velocity=velocity,
                             joint_position=q, joint_velocity=qd, controls=targets, kinematics_only=True)
        else:
            ids = torch.nonzero(mask).flatten()
            self._runtime.reset_robot_states(ids.tolist(), base_positions=pose[ids, :3].numpy(),
                base_orientations=pose[ids][:, (6, 3, 4, 5)].numpy(),
                base_linear_velocities=velocity[ids, :3].numpy(), base_angular_velocities=velocity[ids, 3:].numpy(),
                joint_positions=q[ids].numpy(), joint_velocities=qd[ids].numpy(), joint_position_targets=targets[ids].numpy())
        self._targets[mask] = targets[mask]
        if self._provider is None:
            self._read_cpu()
        for value in self._contacts.values():
            value[mask] = 0
        return self.read_state()

    def read_contacts(self):
        self._require_open()
        return self._contacts

    @torch.no_grad()
    def configure_physics(self, *, link_center_of_mass_offsets=None, shape_friction=None):
        """Set named, per-world parameters relative to the authored model.

        Each value is a [num_envs, 3] array in link coordinates or SI friction
        coefficients (sliding, torsional, rolling). Offsets never accumulate.
        Topology/storage expansion is a cold-path operation, before rollouts.
        """
        self._require_open()
        robot = next(r for r in self.snapshot['robots'] if r['name']==self.spec.robot_name)
        links = {r['name']:r for r in robot['links'] if r.get('role')!='virtual_root'}
        shapes = {s['name']:s for link in robot['links'] for s in link.get('shapes', ())}
        pending = {key:dict(value) for key,value in self._physics_parameters.items()}
        for key, values, known in (("link_center_of_mass_offsets",link_center_of_mass_offsets,links),
                                   ("shape_friction",shape_friction,shapes)):
            if values is None:
                continue
            for name, value in values.items():
                if name not in known:
                    raise ValueError(f'unknown {key} entry {name!r}')
                tensor = self._tensor(value,(self.num_envs,3),name).detach().clone()
                if key=='shape_friction' and bool((tensor<0).any()):
                    raise ValueError('friction coefficients must be nonnegative')
                pending[key][name] = tensor
        com, friction = pending['link_center_of_mass_offsets'], pending['shape_friction']
        if self._provider is None:
            parameters = {
                'override_link_names':tuple(com),
                'link_center_of_mass_offset':torch.stack(tuple(com.values()),1).numpy() if com else None,
                'override_shape_names':tuple(friction),
                'shape_friction':torch.stack(tuple(friction.values()),1).numpy() if friction else None,
            }
            raw = self._runtime.step_and_robot_state(self._targets.numpy(),0,**parameters)
            self._cpu_parameters = parameters
            self._read_cpu(raw)
        else:
            provider = self._provider
            # Only fields supplied in this call require writes/recomputation.
            # Previously stored COM overrides remain in the model across resets.
            com = {name:com[name] for name in (link_center_of_mass_offsets or {})}
            friction = {name:friction[name] for name in (shape_friction or {})}
            fields = ['geom_friction'] if friction else []
            if com:
                # set_const launches over derived output dimensions. Expanding
                # only body_ipos leaves shared inverse weights across worlds.
                fields += ['body_ipos','body_subtreemass','dof_invweight0','body_invweight0',
                           'tendon_length0','tendon_invweight0','actuator_acc0',
                           'eq_data','cam_pos0','cam_poscom0','cam_mat0',
                           'light_pos0','light_poscom0','light_dir0']
            provider.expand_model_fields(fields)
            prefix = self.artifact.robot_prefix(self.spec.robot_name)
            if com:
                data = provider.model_array('body_ipos')
                baseline = provider.model_constant('body_ipos').to(data.dtype)
                for name,value in com.items():
                    index = provider.resolve_object_ids('body',(prefix+name,))[0]
                    data[:,index] = baseline[index]+value
                provider.recompute_constants()
            if friction:
                data = provider.model_array('geom_friction')
                for name,value in friction.items():
                    index = provider.resolve_object_ids('geom',(prefix+name,))[0]
                    data[:,index] = value
            provider.refresh_kinematics()
        self._physics_parameters = pending

    def physics_parameter_state(self):
        """Owned CPU copies of the explicit overrides for checkpoint provenance."""
        return {key:{name:value.detach().cpu().clone() for name,value in values.items()}
                for key,values in self._physics_parameters.items()}

    @torch.no_grad()
    def sync_scene(self, env_index=0):
        """Present one batch lane through the scene pose API, without an editor."""
        self._require_open()
        if isinstance(env_index, bool) or not isinstance(env_index, int) or not 0 <= env_index < self.num_envs:
            raise IndexError('scene presentation environment is out of range')
        if not hasattr(self, '_presentation_links'):
            from .locomotion.math import find_node_by_name
            robot = next(r for r in self.snapshot['robots'] if r['name']==self.spec.robot_name)
            names = tuple(link['name'] for link in robot['links'] if link.get('role')!='virtual_root')
            root = find_node_by_name(self.context.root, self.spec.robot_name)
            self._presentation_links = tuple(find_node_by_name(root,name) for name in names)
            if any(link is None for link in self._presentation_links):
                raise RuntimeError('scene presentation links differ from compiled articulation')
            if self._provider is not None:
                self._presentation_view = self._provider.create_robot_view(
                    RobotBatchSpec(self.spec.robot_name,self.spec.base_link,self.spec.joint_names,names))
                self._presentation_view.bind_scene(self.context,self._presentation_links)
            else:
                self._presentation_indices = [self._runtime.link_names.index(name) for name in names]
        if self._provider is not None:
            self._presentation_view.sync_scene(env_index=env_index)
        else:
            ids = self._presentation_indices
            positions = self._raw_cpu['link_position'][env_index,ids]
            quaternions = self._raw_cpu['link_quaternion'][env_index,ids][:,(1,2,3,0)]
            self.context.apply_link_poses(self._presentation_links,np.concatenate((positions,quaternions),axis=-1))

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._provider is not None:
            self._provider.close()
        self.context.clear_scene()
