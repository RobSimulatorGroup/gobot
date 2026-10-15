"""Unified Newton VBD simulation of Gobot rigid and deformable scene data."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .. import ProviderCapabilities, ProviderUnavailableError
from ._newton_vbd_model import ModelCompiler


@dataclass(frozen=True)
class NewtonVBDConfig:
    device: str = "cuda:0"
    fixed_time_step: float = .002
    substeps: int = 4
    iterations: int = 20
    particle_radius: float = .001
    contact_gap: float = .002
    contact_stiffness: float = 2.e5
    contact_damping: float = 10.
    # Particle contacts act on grams of film and need separate impedance
    # from the rigid bodies. Newton averages these
    # particle material values with shape materials for rigid/soft contacts.
    soft_contact_stiffness: float = 1.e3
    soft_contact_damping: float = .1
    # Optional separate impedance for gram-scale film contacting rigid shapes.
    # None retains Newton's arithmetic mean of particle and shape damping.
    rigid_soft_contact_damping: float | None = None
    # Kelvin strain damping in seconds, scaled by each body's elastic modulus.
    # Unlike mass damping this dissipates deformation, not rigid translation.
    strain_relaxation_time: float = 0.
    soft_friction: float = .85
    belt_friction: float = .92
    joint_limit_stiffness: float = 1.e5
    joint_linear_damping: float = 0.
    joint_angular_damping: float = 0.
    rigid_contact_capacity: int = 16384
    soft_contact_capacity: int = 131072
    body_particle_contact_capacity: int = 32768
    self_contact: bool = True
    capture_graph: bool = True
    rigid_soft_dat: bool = False

    def __post_init__(self):
        for name in ("fixed_time_step", "particle_radius", "contact_gap", "contact_stiffness", "soft_contact_stiffness", "joint_limit_stiffness"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("contact_damping", "soft_contact_damping", "strain_relaxation_time", "soft_friction", "belt_friction",
                     "joint_linear_damping", "joint_angular_damping"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.rigid_soft_contact_damping is not None and (
                not math.isfinite(self.rigid_soft_contact_damping) or self.rigid_soft_contact_damping < 0):
            raise ValueError("rigid_soft_contact_damping must be finite and nonnegative")
        for name in ("substeps", "iterations", "rigid_contact_capacity", "soft_contact_capacity", "body_particle_contact_capacity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class NewtonVBDProvider:
    """One worker-owned Newton world, with Gobot names and SI units at its boundary.

    The snapshot is an owned copy from AppContext.compile_physics_scene_snapshot().
    Presentation copies are explicit; simulation never reads or mutates scene nodes.
    """
    num_envs = 1

    def __init__(self, snapshot: dict[str, Any], *, config: NewtonVBDConfig | None = None):
        self.config = config or NewtonVBDConfig()
        if snapshot.get("schema_version") != 1:
            raise ValueError("unsupported physics snapshot schema")
        self.snapshot = deepcopy(snapshot)
        self.fixed_time_step = self.config.fixed_time_step
        self.session_substeps = self.config.substeps
        self.tick = 0
        self.microsteps = 0
        self.closed = False
        try:
            import newton
            import warp as wp
            from . import _newton_vbd_kernels as kernels
        except (ImportError, AttributeError) as exc:
            raise ProviderUnavailableError("Newton VBD requires Newton 1.6 with full-surface soft contacts") from exc
        wp.init()
        if self.config.device.startswith("cuda") and not wp.is_cuda_available():
            raise ProviderUnavailableError("Newton VBD requested CUDA but no CUDA device is available")
        self.wp, self.newton, self.kernels = wp, newton, kernels
        from .warp_cache import prepare_warp_kernel_cache
        prepare_warp_kernel_cache(wp)
        with wp.ScopedDevice(self.config.device):
            self.compiler = ModelCompiler(self.snapshot, self.config, newton, wp)
            self.model = self.compiler.compile()
            self.pipeline = newton.CollisionPipeline(self.model, broad_phase="sap",
                    rigid_contact_max=self.config.rigid_contact_capacity,
                    soft_contact_max=self.config.soft_contact_capacity,
                    soft_contact_gap=self.config.contact_gap,
                    enable_rigid_soft_full_surface_contact=True)
            self.compiler.filter_contacts(self.pipeline)
            self.contacts = self.pipeline.contacts()
            vf, ee = self.compiler.self_contact_filters() if self.config.self_contact else ({}, {})
            dat_options = {}
            if self.config.rigid_soft_dat:
                import inspect
                if "rigid_soft_enable_dat" not in inspect.signature(newton.solvers.SolverVBD).parameters:
                    raise ProviderUnavailableError("rigid_soft_dat requires Newton 1.7 or newer")
                dat_options = dict(collision_pipeline=self.pipeline, rigid_soft_enable_dat=True,
                                   rigid_soft_contact_use_log_barrier=True)
            solver_type = newton.solvers.SolverVBD
            owner = self
            rigid_soft_damping = self.config.rigid_soft_contact_damping

            class SurfaceContactVBD(solver_type):
                # Keep Newton-specific contact material and surface-velocity
                # handling below the provider boundary, inside captured steps.
                def _refresh_body_particle_contact_state(self, contacts, refresh):
                    super()._refresh_body_particle_contact_state(contacts, refresh)
                    if self.model.particle_count and contacts is not None and contacts.soft_contact_max:
                        if rigid_soft_damping is not None:
                            wp.launch(kernels.set_rigid_soft_damping, contacts.soft_contact_max,
                                      [contacts.soft_contact_count, self.body_particle_contact_material_kd,
                                       rigid_soft_damping], device=self.device)
                        wp.launch(kernels.set_soft_surface_velocity, contacts.soft_contact_max,
                                  [contacts.soft_contact_count, contacts.soft_contact_shape, owner.surface_active,
                                   owner.surface_velocity, self.model.shape_body, owner.state.body_q,
                                   contacts.soft_contact_normal, contacts.soft_contact_body_vel,
                                   self.body_particle_contact_material_mu, owner.config.belt_friction], device=self.device)

            solver_type = SurfaceContactVBD
            self.solver = solver_type(self.model,
                    iterations=self.config.iterations, rigid_compliant_alm=True,
                    rigid_joint_linear_kd=self.config.joint_linear_damping,
                    rigid_joint_angular_kd=self.config.joint_angular_damping,
                    particle_enable_self_contact=bool(self.model.particle_count and self.config.self_contact),
                    particle_self_contact_margin=2 * self.config.particle_radius,
                    particle_self_contact_gap=self.config.contact_gap,
                    particle_external_vertex_contact_filtering_map=vf,
                    particle_external_edge_contact_filtering_map=ee,
                    rigid_body_contact_buffer_size=512,
                    rigid_body_particle_contact_buffer_size=self.config.body_particle_contact_capacity,
                    particle_vertex_contact_buffer_size=64, particle_edge_contact_buffer_size=128,
                    **dat_options)
            if self.solver.particle_enable_self_contact:
                self.compiler.finalize_self_contact_filters(self.solver.trimesh_collision_detector)
            if self.config.rigid_soft_dat:
                self.contacts = self.solver.contacts
            self.state = self.model.state()
            self.next_state = self.model.state()
            self.control = self.model.control()
            self.body_previous = (wp.clone(self.model.body_q) if self.model.body_count
                                  else wp.empty(0, dtype=wp.transform))
            m, capacity = self.model, self.config.soft_contact_capacity
            self.contact_force = wp.zeros(capacity, dtype=wp.vec3)
            self.contact_position = wp.zeros(capacity, dtype=wp.vec3)
            self.nodal_force = wp.zeros(m.particle_count, dtype=wp.vec3)
            self.minimum_tet_volume = wp.ones(1, dtype=float)
            self.maximum_soft_penetration = wp.zeros(1, dtype=float)
            self.soft_drive = wp.zeros(m.particle_count, dtype=wp.vec3)
            self.particle_external = wp.zeros(m.particle_count, dtype=wp.vec3)
            self.body_external = wp.zeros(m.body_count, dtype=wp.spatial_vector)
            self.body_drive = wp.zeros(m.body_count, dtype=wp.spatial_vector)
            self.surface_velocity = wp.zeros(m.shape_count, dtype=wp.vec3)
            self.surface_active = wp.zeros(m.shape_count, dtype=int)
            damping = [body["damping"] for body in self.compiler.soft_bodies
                       for _ in range(body["element_count"])]
            self.damping = wp.array(damping, dtype=float)
            self._targets = self.control.joint_target_q.numpy()
            self.requested_targets = wp.clone(self.control.joint_target_q)
            joints = list(self.compiler.joints.values())
            self.joint_q_indices = wp.array([j["q"] for j in joints], dtype=int)
            self.joint_dofs = wp.array([j["dof"] for j in joints], dtype=int)
            self.joint_kinds = wp.array([j["type"] for j in joints], dtype=int)
            self.joint_gains = wp.array([(j["drive_stiffness"] if j["drive_mode"] == 2 else 0.,
                                         j["drive_damping"] if j["drive_mode"] == 2 else j["damping"],
                                         j["effective_mass"]) for j in joints], dtype=wp.vec3)
            limits = []
            for joint in joints:
                maximum = joint["effort_limit"] if joint["effort_limit"] > 0 else 1.e20
                lower, upper = -maximum, maximum
                if joint["force_lower_limit"] < joint["force_upper_limit"]:
                    lower, upper = max(lower, joint["force_lower_limit"]), min(upper, joint["force_upper_limit"])
                limits.append((lower, upper, joint["velocity_limit"]))
            self.joint_limits = wp.array(limits, dtype=wp.vec3)
            self.limited_targets = wp.zeros(len(joints), dtype=float)
            self.rigid_contact_output = None
            self.graphs = []
            self.graph_states = []
            self.reset()
            if self.config.capture_graph and self.model.device.is_cuda:
                # Warm lazy SDK allocations before capture; keep the two ping-pong graphs
                # explicit so published state always points at the last completed solve.
                self._simulate(self.fixed_time_step / self.session_substeps)
                wp.synchronize_device(self.model.device)
                self.reset()
                for _ in range(2):
                    with wp.ScopedCapture(device=self.model.device) as capture:
                        self._simulate(self.fixed_time_step / self.session_substeps)
                    self.graphs.append(capture.graph)
                    self.graph_states.append((self.state, self.next_state))
                self.reset()
        self.capabilities = ProviderCapabilities(name="NewtonVBD", device=str(self.model.device),
                device_native=self.model.device.is_cuda, graph_capture=bool(self.graphs), masked_reset=False,
                fixed_capacity=True, solver_substeps=True, exact_contact_wrench=False)
        self.capacities = {"rigid_contacts": self.config.rigid_contact_capacity,
                           "soft_contacts": self.config.soft_contact_capacity}
        self.graph_captured = bool(self.graphs)
        self.exact_effort_limits = False

    def _require_open(self):
        if self.closed:
            raise RuntimeError("Newton VBD provider is closed")

    def set_joint_position_targets(self, robot_name, joint_names, values):
        self._require_open()
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(values) != len(joint_names) or not np.isfinite(values).all():
            raise ValueError("joint targets must match named joints and be finite")
        resolved = [self.compiler.joints[(robot_name, name)] for name in joint_names]
        for joint, target in zip(resolved, values, strict=True):
            if joint["type"] != 2:
                target = np.clip(target, joint["lower_limit"], joint["upper_limit"])
            self._targets[joint["dof"]] = target - joint["offset"]
        self.requested_targets.assign(self._targets)

    def set_surface_velocity(self, robot_name, link_name, velocity):
        self._require_open()
        value = np.asarray(velocity, dtype=np.float32)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("surface velocity must be a finite world-space vector")
        ids = [i for i, (_, names) in enumerate(self.compiler.shapes) if names == (robot_name, link_name)]
        if not ids:
            raise KeyError((robot_name, link_name))
        velocities = self.surface_velocity.numpy()
        velocities[ids] = value
        self.surface_velocity.assign(velocities)
        active = self.surface_active.numpy()
        active[ids] = 1
        self.surface_active.assign(active)

    def _simulate(self, dt):
        wp, k, m, c, s = self.wp, self.kernels, self.model, self.contacts, self.solver
        self.state.clear_forces()
        if self.compiler.joints:
            self.newton.eval_ik(m, self.state, self.state.joint_q, self.state.joint_qd)
            wp.launch(k.joint_drive_targets, len(self.compiler.joints),
                    [dt, self.joint_q_indices, self.joint_dofs, self.joint_kinds, self.joint_gains, self.joint_limits,
                     self.state.joint_q, self.state.joint_qd, self.requested_targets,
                     self.limited_targets, self.control.joint_target_q, m.joint_target_ke, m.joint_target_kd], device=m.device)
        if m.body_count:
            wp.launch(k.copy_body_forces, m.body_count,
                      [self.body_drive, self.body_external, self.state.body_f], device=m.device)
            wp.copy(self.body_previous, self.state.body_q)
        if m.particle_count:
            wp.launch(k.soft_external_forces, m.particle_count,
                    [self.state.particle_qd, m.particle_mass, self.damping,
                     self.particle_external, self.state.particle_f], device=m.device)
        if not self.config.rigid_soft_dat:
            self.pipeline.collide(self.state, c)
        s.step(self.state, self.next_state, self.control, None if self.config.rigid_soft_dat else c, dt)
        self.state, self.next_state = self.next_state, self.state
        if m.tet_count:
            wp.launch(k.track_tet_volume, m.tet_count,
                      [self.state.particle_q, m.tet_indices, m.tet_poses, self.minimum_tet_volume], device=m.device)
        self.nodal_force.zero_()
        self.soft_drive.zero_()
        self.body_drive.zero_()
        if m.particle_count and m.shape_count:
            wp.launch(k.soft_contact_forces, self.config.soft_contact_capacity,
                    [dt, c.soft_contact_count, c.soft_contact_indices, c.soft_contact_barycentric,
                     c.soft_contact_shape, c.soft_contact_body_pos, c.soft_contact_body_vel, c.soft_contact_normal,
                     self.state.particle_q, s.particle_q_prev, m.particle_radius,
                     self.state.body_q, self.body_previous, self.state.body_qd, m.body_com, m.shape_body,
                     m.shape_margin, s.body_particle_contact_penalty_k, s.body_particle_contact_material_kd,
                     s.body_particle_contact_material_mu, float(s.friction_epsilon), self.config.rigid_soft_dat,
                     self.contact_force, self.nodal_force, self.contact_position,
                     self.maximum_soft_penetration], device=m.device)
            wp.launch(k.soft_belt_traction, self.config.soft_contact_capacity,
                    [c.soft_contact_count, c.soft_contact_indices, c.soft_contact_barycentric, c.soft_contact_shape,
                     c.soft_contact_normal, self.contact_force, self.surface_active, self.soft_drive], device=m.device)
        if m.body_count:
            self.rigid_contact_output = s.collect_rigid_contact_forces(self.state.body_q, self.body_previous, c, dt)
            b0, b1, p0, p1, force, count = self.rigid_contact_output
            wp.launch(k.rigid_belt_load, self.config.rigid_contact_capacity,
                    [dt, count, c.rigid_contact_shape0, c.rigid_contact_shape1, b0, b1, p0, p1,
                     c.rigid_contact_normal, force, self.surface_velocity, self.surface_active, self.state.body_q,
                     self.state.body_qd, m.body_com, m.body_mass, self.config.belt_friction,
                     self.body_drive], device=m.device)

    def advance_microstep(self, dt, actions=None):
        from ..._core import SimulationMicrostepResult
        self._require_open()
        expected = self.fixed_time_step / self.session_substeps
        if not math.isclose(dt, expected, rel_tol=1e-6):
            raise ValueError("Newton VBD microstep differs from its fixed-step configuration")
        if actions is not None:
            raise ValueError("use named joint targets for Newton VBD actions")
        if self.graphs:
            index = self.microsteps % 2
            self.wp.capture_launch(self.graphs[index])
            self.state, self.next_state = self.graph_states[index]
        else:
            self._simulate(dt)
        self.microsteps += 1
        self.tick = self.microsteps // self.session_substeps
        result = SimulationMicrostepResult()
        result.completed, result.advanced_time = True, dt
        return [result]

    def step(self, actions=None, *, nsteps=1):
        self._require_open()
        if not isinstance(nsteps, int) or isinstance(nsteps, bool) or nsteps <= 0:
            raise ValueError("nsteps must be a positive integer")
        for _ in range(nsteps * self.session_substeps):
            self.advance_microstep(self.fixed_time_step / self.session_substeps, actions)

    def reset(self, reset_mask=None, **state):
        self._require_open()
        if state:
            raise ValueError("Newton VBD reset restores the authored snapshot")
        if reset_mask is not None and not bool(np.asarray(reset_mask).reshape(-1)[0]):
            return
        wp = self.wp
        self.solver.reset(self.state)
        self.solver.reset(self.next_state)
        self.microsteps = self.tick = 0
        self._targets.fill(0.)
        self.requested_targets.assign(self._targets)
        self.control.joint_target_q.zero_()
        self.control.joint_target_qd.zero_()
        self.control.joint_f.zero_()
        self.limited_targets.zero_()
        for value in (self.nodal_force, self.soft_drive,
                      self.particle_external, self.body_external, self.body_drive, self.surface_velocity, self.surface_active,
                      self.contact_force, self.contact_position):
            value.zero_()
        self.contacts.soft_contact_count.zero_()
        self.contacts.rigid_contact_count.zero_()
        self.minimum_tet_volume.fill_(1.)
        self.maximum_soft_penetration.zero_()
        if self.model.body_count:
            wp.copy(self.body_previous, self.model.body_q)
        self.rigid_contact_output = None

    def snapshot_arrays(self, fields, environments):
        self._require_open()
        environments = tuple(environments)
        if environments != (0,):
            raise ValueError("Newton VBD currently exports environment 0 only")
        requested = set(fields)
        output = {}
        poses = None
        for robot in self.snapshot["robots"]:
            key = f"robot.{robot['name']}.link_pose"
            if key in requested:
                if poses is None:
                    poses = self.state.body_q.numpy()
                ids = [self.compiler.links[(robot["name"], link["name"])] for link in robot["links"]]
                output[key] = poses[ids][None].copy()
        if "deformable.local_vertices" in requested:
            world = self.state.particle_q.numpy() if self.model.particle_count else np.empty((0, 3))
            bodies = self.compiler.soft_bodies
            width = max((b["element_count"] for b in bodies), default=0)
            local = np.zeros((1, len(bodies), width, 3), dtype=np.float64)
            for i, body in enumerate(bodies):
                start, count = body["element_offset"], body["element_count"]
                inverse = body["inverse_transform"]
                local[0, i, :count] = world[start:start + count] @ inverse[:3, :3].T + inverse[:3, 3]
            output["deformable.local_vertices"] = local
        unknown = requested - output.keys()
        if unknown:
            raise KeyError(f"unknown Newton VBD snapshot fields: {sorted(unknown)}")
        if any(not np.isfinite(value).all() for value in output.values()):
            raise RuntimeError("Newton VBD produced non-finite simulation state; reset required")
        self._check_capacity()
        if float(self.minimum_tet_volume.numpy()[0]) <= 0:
            raise RuntimeError("Newton VBD tetrahedron inversion; reset required")
        return output

    def _check_capacity(self):
        for name, array, capacity in (("soft", self.contacts.soft_contact_count, self.config.soft_contact_capacity),
                                      ("rigid", self.contacts.rigid_contact_count, self.config.rigid_contact_capacity)):
            if int(array.numpy()[0]) >= capacity:
                raise RuntimeError(f"Newton VBD {name} contact capacity exhausted ({capacity}); increase capacity")
        if self.model.body_count and self.model.particle_count:
            counts = self.solver.body_particle_contact_counts.numpy()
            if counts.size and counts.max() > self.config.body_particle_contact_capacity:
                raise RuntimeError("Newton VBD per-body soft contact capacity exhausted; increase body_particle_contact_capacity")
        if self.config.self_contact and self.model.particle_count:
            if self.config.rigid_soft_dat:
                info = self.pipeline._soft_self_contact_detector.collision_info
            else:
                info = self.solver.trimesh_collision_detector.collision_info
            for kind in ("vertex_colliding_triangles", "triangle_colliding_vertices", "edge_colliding_edges"):
                counts = getattr(info, kind + "_count")
                if counts is None:
                    continue  # This SDK solve path does not allocate the inverse adjacency.
                count = counts.numpy()
                capacity = getattr(info, kind + "_buffer_sizes").numpy()
                if np.any(count > capacity):
                    raise RuntimeError(f"Newton VBD {kind} capacity exhausted; rebuild with larger self-contact buffers")

    @property
    def scene_fields(self):
        return tuple(f"robot.{robot['name']}.link_pose" for robot in self.snapshot["robots"]) + ("deformable.local_vertices",)

    def read_state(self):
        """Explicit CPU copy for diagnostics, in world meters, seconds and newtons.

        Soft ``drive_forces`` report the moving-surface tangent component already
        included in ``contact_forces``; they are not an additional applied load.
        """
        self._require_open()
        self._check_capacity()
        q = self.state.particle_q.numpy() if self.model.particle_count else np.empty((0, 3))
        velocity = self.state.particle_qd.numpy() if self.model.particle_count else np.empty((0, 3))
        forces, drive = self.nodal_force.numpy(), self.soft_drive.numpy()
        poses = self.state.body_q.numpy() if self.model.body_count else np.empty((0, 7))
        if not all(np.isfinite(value).all() for value in (q, velocity, forces, poses)):
            raise RuntimeError("Newton VBD produced non-finite state; reset required")
        bodies = {}
        for body in self.compiler.soft_bodies:
            start, count = body["element_offset"], body["element_count"]
            selection = slice(start, start + count)
            bodies[body["name"]] = {"positions": q[selection].copy(), "velocities": velocity[selection].copy(),
                                    "contact_forces": forces[selection].copy(), "drive_forces": drive[selection].copy()}
        contacts = self.contacts
        count = min(int(contacts.soft_contact_count.numpy()[0]), self.config.soft_contact_capacity)
        indices = contacts.soft_contact_indices.numpy()[:count]
        weights = contacts.soft_contact_barycentric.numpy()[:count]
        points = np.sum(q[np.maximum(indices, 0)] * weights[:, :, None] * (indices >= 0)[:, :, None], axis=1)
        radius = self.model.particle_radius.numpy() if self.model.particle_count else np.empty(0)
        radii = np.max(radius[np.maximum(indices, 0)] * (indices >= 0), axis=1) if count else np.empty(0)
        rigid_points = self.contact_position.numpy()[:count]
        normals = contacts.soft_contact_normal.numpy()[:count]
        shape_indices = contacts.soft_contact_shape.numpy()[:count]
        margins = self.model.shape_margin.numpy()[shape_indices]
        gap = np.sum((points - rigid_points) * normals, axis=1) - radii - margins
        efforts = np.zeros(self.model.joint_dof_count)
        joints = {}
        if self.compiler.joints:
            self.newton.eval_ik(self.model, self.state, self.state.joint_q, self.state.joint_qd)
            joint_q, joint_qd = self.state.joint_q.numpy(), self.state.joint_qd.numpy()
            targets = self.control.joint_target_q.numpy()
            ke, kd = self.model.joint_target_ke.numpy(), self.model.joint_target_kd.numpy()
            for (robot, name), joint in self.compiler.joints.items():
                dof, qi = joint["dof"], joint["q"]
                error = targets[dof] - joint_q[qi]
                if joint["type"] != 3:
                    error = math.atan2(math.sin(error), math.cos(error))
                efforts[dof] = ke[dof] * error - kd[dof] * joint_qd[dof]
                joints[f"{robot}/{name}"] = {"position": float(joint_q[qi] + joint["offset"]),
                    "velocity": float(joint_qd[dof]), "target": float(targets[dof] + joint["offset"]),
                    "tracking_error": float(error), "estimated_drive_effort": float(efforts[dof])}
        return {"time": self.microsteps * self.fixed_time_step / self.session_substeps,
                "minimum_tet_volume_ratio": float(self.minimum_tet_volume.numpy()[0]),
                "peak_soft_penetration": float(self.maximum_soft_penetration.numpy()[0]),
                "deformables": bodies,
                "links": {f"{robot}/{link}": poses[i].copy()
                          for i, (robot, link) in enumerate(self.compiler.body_names)},
                "soft_contact_count": count,
                "max_soft_penetration": float(max(0., -np.min(gap))) if count else 0.,
                "soft_contact_forces": self.contact_force.numpy()[:count],
                "soft_contact_shapes": tuple(self.compiler.shapes[i][1] for i in shape_indices),
                "joints": joints, "estimated_drive_efforts": efforts}

    def close(self):
        if self.closed:
            return
        self.wp.synchronize_device(self.model.device)
        self.closed = True
