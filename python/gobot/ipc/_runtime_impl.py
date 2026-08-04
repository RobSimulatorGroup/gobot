# SPDX-FileCopyrightText: Copyright (c) 2026 Gobot contributors
# SPDX-License-Identifier: Apache-2.0
"""Private Newton 1.4 / Warp 1.15 IPC correctness runtime.

This first executable runtime intentionally keeps CUDA graph capture disabled.
It uses Newton's public model/state and GPU FK contracts while owning the f64
tetrahedral and IPC storage below Gobot's provider boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import newton
import numpy as np
import torch
import warp as wp
from warp.sparse import bsr_set_from_triplets

from gobot.rl.providers.base import (
    ProviderUnavailableError,
    RobotBatchState,
    RobotBatchSpec,
    SimulationCapacityError,
)

from ._views import DeformableBatchState, TactileBatchState
from ._warp_kernels import (
    FixedCgSolver,
    assemble_bsr33_from_triplets,
    tet_neo_hookean_kernel,
)
from ._runtime_kernels import (
    apply_newton_step_kernel,
    apply_ccd_fraction_kernel,
    accumulate_tetrahedra_kernel,
    initialize_ccd_fraction_kernel,
    copy_particles_to_newton_kernel,
    finalize_particles_kernel,
    gather_tetrahedra_kernel,
    initialize_newton_kernel,
    point_triangle_contact_kernel,
    point_triangle_line_search_kernel,
    predict_kernel,
    reset_masked_particles_kernel,
    blend_line_search_kernel,
    tetrahedron_validity_kernel,
    update_attached_targets_kernel,
    write_diagonal_system_kernel,
    write_point_triangle_system_kernel,
    write_tetrahedron_system_kernel,
)


@dataclass(frozen=True)
class _Mesh:
    vertices: np.ndarray
    tetrahedra: np.ndarray
    surface_triangles: np.ndarray


@dataclass(frozen=True)
class _SoftLayout:
    kind: str
    path: str
    name: str
    entry: Mapping[str, Any]
    mesh: _Mesh
    particle_start: int
    tetrahedron_start: int
    world_vertices: np.ndarray

    @property
    def vertex_count(self) -> int:
        return int(self.mesh.vertices.shape[0])

    @property
    def tetrahedron_count(self) -> int:
        return int(self.mesh.tetrahedra.shape[0])


@dataclass(frozen=True)
class _RobotLayout:
    name: str
    path: str
    base_link_name: str
    base_body_index: int
    base_q_start: int
    base_qd_start: int
    joint_q_by_name: Mapping[str, int]
    joint_qd_by_name: Mapping[str, int]
    joint_limits_by_name: Mapping[str, tuple[float, float]]
    body_by_name: Mapping[str, int]
    body_by_path: Mapping[str, int]


def _decode_mesh(data: bytes) -> _Mesh:
    if len(data) < 24 or data[:8] != b"GOBTIPC1":
        raise ValueError("compiled IPC mesh blob is truncated or has invalid magic")
    version, vertex_count, tetrahedron_count, surface_count = struct.unpack_from(
        "<IIII", data, 8
    )
    if version != 1:
        raise ValueError(f"unsupported compiled IPC mesh version {version}")
    vertex_offset = 24
    tetrahedron_offset = vertex_offset + vertex_count * 3 * 8
    surface_offset = tetrahedron_offset + tetrahedron_count * 4 * 4
    expected = surface_offset + surface_count * 3 * 4
    if len(data) != expected:
        raise ValueError("compiled IPC mesh dimensions do not match its byte length")
    vertices = np.frombuffer(
        data, dtype="<f8", count=vertex_count * 3, offset=vertex_offset
    ).reshape(vertex_count, 3).copy()
    tetrahedra = np.frombuffer(
        data, dtype="<u4", count=tetrahedron_count * 4, offset=tetrahedron_offset
    ).astype(np.int32, copy=True).reshape(tetrahedron_count, 4)
    surface = np.frombuffer(
        data, dtype="<u4", count=surface_count * 3, offset=surface_offset
    ).astype(np.int32, copy=True).reshape(surface_count, 3)
    return _Mesh(vertices, tetrahedra, surface)


def _matrix(value: Mapping[str, Any]) -> np.ndarray:
    result = np.asarray(value["matrix_row_major"], dtype=np.float64).reshape(4, 4)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError("compiled IPC transform is not a finite 4x4 matrix")
    return result


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _quaternion_xyzw(matrix: np.ndarray) -> tuple[float, float, float, float]:
    rotation = matrix[:3, :3]
    scales = np.linalg.norm(rotation, axis=0)
    rotation = rotation / scales
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            ],
            dtype=np.float64,
        )
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        quaternion = np.array(
            [
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        quaternion = np.array(
            [
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def _pose_from_matrix(matrix: np.ndarray) -> tuple[float, ...]:
    return (
        float(matrix[0, 3]),
        float(matrix[1, 3]),
        float(matrix[2, 3]),
        *_quaternion_xyzw(matrix),
    )


def _wp_transform(matrix: np.ndarray) -> wp.transform:
    pose = _pose_from_matrix(matrix)
    return wp.transform(
        wp.vec3(float(pose[0]), float(pose[1]), float(pose[2])),
        wp.quat(float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])),
    )


def _lame_parameters(young_modulus: float, poisson_ratio: float) -> tuple[float, float]:
    shear = young_modulus / (2.0 * (1.0 + poisson_ratio))
    lame = young_modulus * poisson_ratio / (
        (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    )
    return shear, lame


def _resolve_unique(
    values: Sequence[Mapping[str, Any]], requested: str, description: str
) -> Mapping[str, Any]:
    path_matches = [value for value in values if str(value.get("path", "")) == requested]
    matches = path_matches or [
        value for value in values if str(value.get("name", "")) == requested
    ]
    if not matches:
        raise KeyError(f"compiled IPC artifact has no {description} {requested!r}")
    if len(matches) != 1:
        raise ValueError(f"compiled IPC artifact {description} {requested!r} is ambiguous")
    return matches[0]


def _as_device_tensor(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    description: str,
) -> torch.Tensor:
    result = torch.as_tensor(value, device=device, dtype=dtype)
    if tuple(int(item) for item in result.shape) != shape:
        raise ValueError(
            f"Warp IPC {description} has shape {tuple(result.shape)}, expected {shape}"
        )
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"Warp IPC {description} contains a non-finite value")
    return result


def _quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    imaginary = quaternion[..., :3]
    real = quaternion[..., 3:4]
    cross = 2.0 * torch.cross(imaginary, vector, dim=-1)
    return vector + real * cross + torch.cross(imaginary, cross, dim=-1)


def _quat_inverse_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    inverse = torch.cat((-quaternion[..., :3], quaternion[..., 3:4]), dim=-1)
    return _quat_apply(inverse, vector)


def _quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_xyz, left_w = left[..., :3], left[..., 3:4]
    right_xyz, right_w = right[..., :3], right[..., 3:4]
    xyz = (
        left_w * right_xyz
        + right_w * left_xyz
        + torch.cross(left_xyz, right_xyz, dim=-1)
    )
    w = left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True)
    return torch.cat((xyz, w), dim=-1)


class _SolverIPC(newton.solvers.SolverBase):
    """BSR/CG f64 Newton correctness solver over Newton storage."""

    def __init__(self, runtime: "_WarpIpcSession") -> None:
        super().__init__(runtime.model)
        self.runtime = runtime

    def step(
        self,
        state_in: Any,
        state_out: Any,
        control: Any,
        contacts: Any,
        dt: float,
    ) -> None:
        del control, contacts
        runtime = self.runtime
        runtime._update_robot_fk(state_out, dt)
        runtime._update_attached_targets(state_out)
        wp.copy(runtime.contact_velocity, runtime.velocity)
        device = runtime.wp_device
        count = runtime.particle_count
        wp.launch(
            predict_kernel,
            dim=count,
            inputs=[
                runtime.position,
                runtime.velocity,
                runtime.mass,
                runtime.kinematic,
                runtime.kinematic_target,
                wp.vec3d(*runtime.config.gravity),
                runtime.damping,
                wp.float64(dt),
            ],
            outputs=[runtime.predicted, runtime.iterate],
            device=device,
        )
        runtime._accept_positive_iterate(runtime.position, "prediction")
        for _iteration in range(runtime.config.newton_iterations):
            runtime.active_contact_count.zero_()
            wp.launch(
                initialize_newton_kernel,
                dim=count,
                inputs=[
                    runtime.iterate,
                    runtime.predicted,
                    runtime.mass,
                    runtime.kinematic,
                    runtime.kinematic_target,
                    wp.float64(runtime.config.kinematic_stiffness),
                    wp.float64(dt),
                ],
                outputs=[runtime.gradient, runtime.hessian, runtime.contact_force],
                device=device,
            )
            wp.launch(
                gather_tetrahedra_kernel,
                dim=(runtime.tetrahedron_count, 4),
                inputs=[runtime.iterate, runtime.tetrahedron_indices],
                outputs=[runtime.tetrahedron_position],
                device=device,
            )
            wp.launch(
                tet_neo_hookean_kernel,
                dim=runtime.tetrahedron_count,
                inputs=[
                    runtime.tetrahedron_position,
                    runtime.inverse_rest_matrix,
                    runtime.rest_volume,
                    runtime.shear_modulus,
                    runtime.lame_lambda,
                ],
                outputs=[
                    runtime.tetrahedron_energy,
                    runtime.tetrahedron_gradient,
                    runtime.tetrahedron_hessian,
                    runtime.tetrahedron_valid,
                ],
                device=device,
            )
            wp.launch(
                accumulate_tetrahedra_kernel,
                dim=(runtime.tetrahedron_count, 4),
                inputs=[
                    runtime.tetrahedron_indices,
                    runtime.tetrahedron_gradient,
                    runtime.tetrahedron_hessian,
                ],
                outputs=[runtime.gradient, runtime.hessian],
                device=device,
            )
            if runtime.pt_count:
                wp.launch(
                    point_triangle_contact_kernel,
                    dim=runtime.pt_count,
                    inputs=[
                        runtime.iterate,
                        runtime.contact_velocity,
                        runtime.pt_point,
                        runtime.pt_a,
                        runtime.pt_b,
                        runtime.pt_c,
                        wp.float64(runtime.config.barrier_distance**2),
                        wp.float64(runtime.config.barrier_stiffness),
                        wp.float64(runtime.config.friction_coefficient),
                        wp.float64(max(runtime.config.barrier_distance / dt, 1.0e-6)),
                    ],
                    outputs=[
                        runtime.gradient,
                        runtime.contact_force,
                        runtime.active_contact_count,
                        runtime.pt_barycentric,
                        runtime.pt_hessian,
                    ],
                    device=device,
                )
            wp.copy(runtime.line_search_origin, runtime.iterate)
            wp.launch(
                write_diagonal_system_kernel,
                dim=count,
                inputs=[
                    runtime.hessian,
                    runtime.gradient,
                    wp.float64(1.0e-8),
                ],
                outputs=[
                    runtime.system_values,
                    runtime.system_rhs,
                    runtime.system_solution,
                ],
                device=device,
            )
            wp.launch(
                write_tetrahedron_system_kernel,
                dim=(runtime.tetrahedron_count, 4, 4),
                inputs=[
                    runtime.tetrahedron_indices,
                    runtime.tetrahedron_hessian,
                    wp.int32(runtime.particle_count),
                ],
                outputs=[runtime.system_values],
                device=device,
            )
            if runtime.pt_count:
                wp.launch(
                    write_point_triangle_system_kernel,
                    dim=(runtime.pt_count, 4, 4),
                    inputs=[
                        runtime.pt_barycentric,
                        runtime.pt_hessian,
                        wp.int32(
                            runtime.particle_count + 16 * runtime.tetrahedron_count
                        ),
                    ],
                    outputs=[runtime.system_values],
                    device=device,
                )
            bsr_set_from_triplets(
                runtime.system_matrix,
                runtime.system_rows,
                runtime.system_columns,
                runtime.system_values,
                prune_numerical_zeros=False,
                topology="masked",
            )
            runtime._cg_solver.solve()
            wp.launch(
                apply_newton_step_kernel,
                dim=count,
                inputs=[
                    runtime.iterate,
                    runtime.system_solution,
                    wp.float64(max(0.5 * runtime.config.barrier_distance, 1.0e-4)),
                ],
                device=device,
            )
            runtime._accept_positive_iterate(runtime.line_search_origin, "Newton")

        runtime.invalid_state.zero_()
        wp.launch(
            finalize_particles_kernel,
            dim=count,
            inputs=[runtime.position, runtime.iterate, wp.float64(dt)],
            outputs=[
                runtime.position,
                runtime.velocity,
                runtime.public_position,
                runtime.public_velocity,
                runtime.public_contact_force,
                runtime.contact_force,
                runtime.invalid_state,
            ],
            device=device,
        )
        wp.launch(
            copy_particles_to_newton_kernel,
            dim=count,
            inputs=[runtime.position, runtime.velocity],
            outputs=[state_out.particle_q, state_out.particle_qd],
            device=device,
        )
        runtime._finish_tick_checks()


class _WarpIpcSession:
    graph_captured = False

    def __init__(self, artifact: Any, config: Any, num_envs: int) -> None:
        if config.capture_graphs:
            raise ProviderUnavailableError(
                "the Warp IPC correctness runtime requires capture_graphs=False; "
                "fixed-address CUDA graph admission is the next implementation phase"
            )
        self.artifact = artifact
        self.config = config
        self.num_envs = int(num_envs)
        self.device_name = str(config.device)
        if not self.device_name.startswith("cuda"):
            raise ProviderUnavailableError(
                "Warp IPC uses CUDA-resident f64 storage; configure a CUDA device"
            )
        wp.init()
        if not wp.is_cuda_available():
            raise ProviderUnavailableError("Warp IPC requested but Warp cannot access CUDA")
        self.wp_device = wp.get_device(self.device_name)
        if not self.wp_device.is_cuda:
            raise ProviderUnavailableError(
                f"Warp IPC device {self.device_name!r} is not CUDA-capable"
            )
        self.torch_device = torch.device(self.device_name)
        if not torch.cuda.is_available():
            raise ProviderUnavailableError("Warp IPC requested but Torch cannot access CUDA")
        self._closed = False
        self._step_count = 0
        self._last_latency_ms = 0.0
        self._last_active_contacts = 0
        self._last_cg_iteration = 0
        self._last_cg_status = "not-run"
        self._last_line_search_backtracks = 0
        self._last_line_search_fallbacks: dict[str, int] = {}
        self._last_ccd_clamps = 0
        self._build()

    def _build(self) -> None:
        with wp.ScopedDevice(self.wp_device):
            blueprint = newton.ModelBuilder()
            self.robot_layouts = self._add_robots(blueprint)
            self.soft_layouts = self._add_soft_bodies(blueprint)
            self.particles_per_environment = len(blueprint.particle_q)
            self.tetrahedra_per_environment = len(blueprint.tet_indices)
            self.bodies_per_environment = len(blueprint.body_q)
            self.joint_coordinates_per_environment = len(blueprint.joint_q)
            self.joint_dofs_per_environment = len(blueprint.joint_qd)
            if self.particles_per_environment <= 0 or self.tetrahedra_per_environment <= 0:
                raise ValueError("compiled IPC artifact contains no tetrahedral state")

            builder = newton.ModelBuilder()
            builder.replicate(blueprint, self.num_envs, spacing=(0.0, 0.0, 0.0))
            self.model = builder.finalize(device=self.wp_device)
            self.state = self.model.state()
            self.control = self.model.control()
            self._allocate_solver_storage()
            self._solver = _SolverIPC(self)
            self._initialize_robot_targets()
            self._update_robot_fk(self.state, float(self.config.fixed_time_step))
            self._update_attached_targets(self.state)
            wp.launch(
                copy_particles_to_newton_kernel,
                dim=self.particle_count,
                inputs=[self.position, self.velocity],
                outputs=[self.state.particle_q, self.state.particle_qd],
                device=self.wp_device,
            )
            self._step_count = 0

    def _add_robots(self, blueprint: Any) -> tuple[_RobotLayout, ...]:
        layouts: list[_RobotLayout] = []
        for robot in self.artifact.robots:
            links = tuple(robot["links"])
            joints = tuple(robot["joints"])
            link_by_name = {str(link["name"]): link for link in links}
            body_by_name: dict[str, int] = {}
            body_by_path: dict[str, int] = {}
            for link in links:
                matrix = _matrix(link["transform"])
                inertia_values = tuple(float(value) for value in link["inertia_diagonal"])
                inertia = wp.mat33(
                    max(inertia_values[0], 1.0e-6), 0.0, 0.0,
                    0.0, max(inertia_values[1], 1.0e-6), 0.0,
                    0.0, 0.0, max(inertia_values[2], 1.0e-6),
                )
                # Newton 1.4 add_body() creates a complete one-link free
                # articulation. Robot links must stay unattached until the
                # compiled Gobot joint tree below is assembled.
                body = blueprint.add_link(
                    xform=_wp_transform(matrix),
                    com=wp.vec3(*tuple(float(value) for value in link["center_of_mass"])),
                    inertia=inertia,
                    mass=max(float(link["mass"]), 1.0e-6),
                    label=str(link["name"]),
                )
                body_by_name[str(link["name"])] = body
                body_by_path[str(link["path"])] = body

            joints_by_parent: dict[str, list[Mapping[str, Any]]] = {}
            child_names: set[str] = set()
            for joint in joints:
                joints_by_parent.setdefault(str(joint["parent_link"]), []).append(joint)
                child_names.add(str(joint["child_link"]))
            root_links = [
                link for link in links if str(link["name"]) not in child_names
            ]
            if len(root_links) != 1:
                raise NotImplementedError(
                    "the Warp IPC correctness runtime requires one root link per robot"
                )
            root = root_links[0]
            root_name = str(root["name"])
            articulation_joints: list[int] = []
            free_joint = blueprint.add_joint_free(
                body_by_name[root_name], label=f"{robot['name']}/free"
            )
            articulation_joints.append(free_joint)
            base_q_start = int(blueprint.joint_q_start[free_joint])
            base_qd_start = int(blueprint.joint_qd_start[free_joint])
            joint_q_by_name: dict[str, int] = {}
            joint_qd_by_name: dict[str, int] = {}
            joint_limits: dict[str, tuple[float, float]] = {}

            def add_children(parent_name: str) -> None:
                for joint in joints_by_parent.get(parent_name, ()):
                    child_name = str(joint["child_link"])
                    parent_matrix = _matrix(joint["local_transform"])
                    child_matrix = np.linalg.inv(
                        _matrix(link_by_name[child_name]["local_transform"])
                    )
                    joint_type = int(joint["joint_type"])
                    common = {
                        "parent_xform": _wp_transform(parent_matrix),
                        "child_xform": _wp_transform(child_matrix),
                        "label": str(joint["name"]),
                    }
                    if joint_type == 0:
                        joint_id = blueprint.add_joint_fixed(
                            body_by_name[parent_name], body_by_name[child_name], **common
                        )
                    elif joint_type in (1, 2):
                        lower = float(joint["lower_limit"])
                        upper = float(joint["upper_limit"])
                        if joint_type == 2:
                            lower, upper = -math.inf, math.inf
                        joint_id = blueprint.add_joint_revolute(
                            body_by_name[parent_name],
                            body_by_name[child_name],
                            axis=wp.vec3(*tuple(float(value) for value in joint["axis"])),
                            limit_lower=lower,
                            limit_upper=upper,
                            **common,
                        )
                    elif joint_type == 3:
                        lower = float(joint["lower_limit"])
                        upper = float(joint["upper_limit"])
                        joint_id = blueprint.add_joint_prismatic(
                            body_by_name[parent_name],
                            body_by_name[child_name],
                            axis=wp.vec3(*tuple(float(value) for value in joint["axis"])),
                            limit_lower=lower,
                            limit_upper=upper,
                            **common,
                        )
                    else:
                        raise NotImplementedError(
                            f"Warp IPC robot joint {joint['name']!r} uses unsupported type {joint_type}"
                        )
                    articulation_joints.append(joint_id)
                    if joint_type != 0:
                        q_start = int(blueprint.joint_q_start[joint_id])
                        qd_start = int(blueprint.joint_qd_start[joint_id])
                        name = str(joint["name"])
                        joint_q_by_name[name] = q_start
                        joint_qd_by_name[name] = qd_start
                        joint_limits[name] = (lower, upper)
                        blueprint.joint_q[q_start] = float(joint["initial_position"])
                    add_children(child_name)

            add_children(root_name)
            blueprint.add_articulation(articulation_joints, label=str(robot["name"]))
            layouts.append(
                _RobotLayout(
                    name=str(robot["name"]),
                    path=str(robot["path"]),
                    base_link_name=root_name,
                    base_body_index=body_by_name[root_name],
                    base_q_start=base_q_start,
                    base_qd_start=base_qd_start,
                    joint_q_by_name=MappingProxyType(joint_q_by_name),
                    joint_qd_by_name=MappingProxyType(joint_qd_by_name),
                    joint_limits_by_name=MappingProxyType(joint_limits),
                    body_by_name=MappingProxyType(body_by_name),
                    body_by_path=MappingProxyType(body_by_path),
                )
            )
        return tuple(layouts)

    def _allocate_solver_storage(self) -> None:
        particles_per_environment = self.particles_per_environment
        tetrahedra_per_environment = self.tetrahedra_per_environment
        initial_per_environment = np.empty(
            (particles_per_environment, 3), dtype=np.float64
        )
        mass_per_environment = np.zeros(particles_per_environment, dtype=np.float64)
        damping_per_environment = np.zeros(particles_per_environment, dtype=np.float64)
        kinematic_per_environment = np.zeros(particles_per_environment, dtype=np.int32)
        tetrahedron_indices_per_environment = np.empty(
            (tetrahedra_per_environment, 4), dtype=np.int32
        )
        inverse_rest_per_environment = np.empty(
            (tetrahedra_per_environment, 3, 3), dtype=np.float64
        )
        rest_volume_per_environment = np.empty(
            tetrahedra_per_environment, dtype=np.float64
        )
        shear_per_environment = np.empty(tetrahedra_per_environment, dtype=np.float64)
        lame_per_environment = np.empty(tetrahedra_per_environment, dtype=np.float64)

        attached_particle: list[int] = []
        attached_body: list[int] = []
        attached_local: list[np.ndarray] = []
        attached_reset: list[tuple[int, int, int, np.ndarray]] = []
        for layout in self.soft_layouts:
            particle_slice = slice(
                layout.particle_start, layout.particle_start + layout.vertex_count
            )
            tetrahedron_slice = slice(
                layout.tetrahedron_start,
                layout.tetrahedron_start + layout.tetrahedron_count,
            )
            initial_per_environment[particle_slice] = layout.world_vertices
            damping_per_environment[particle_slice] = float(layout.entry["damping"])
            tetrahedron_indices_per_environment[tetrahedron_slice] = (
                layout.mesh.tetrahedra + layout.particle_start
            )
            shear, lame = _lame_parameters(
                float(layout.entry["young_modulus"]),
                float(layout.entry["poisson_ratio"]),
            )
            shear_per_environment[tetrahedron_slice] = shear
            lame_per_environment[tetrahedron_slice] = lame
            for local_tetrahedron, tetrahedron in enumerate(layout.mesh.tetrahedra):
                positions = layout.world_vertices[tetrahedron]
                rest_matrix = np.column_stack(
                    (
                        positions[1] - positions[0],
                        positions[2] - positions[0],
                        positions[3] - positions[0],
                    )
                )
                determinant = float(np.linalg.det(rest_matrix))
                if not math.isfinite(determinant) or determinant <= 0.0:
                    raise ValueError(
                        f"compiled IPC mesh {layout.path!r} contains an inverted tetrahedron"
                    )
                global_tetrahedron = layout.tetrahedron_start + local_tetrahedron
                inverse_rest_per_environment[global_tetrahedron] = np.linalg.inv(rest_matrix)
                volume = determinant / 6.0
                rest_volume_per_environment[global_tetrahedron] = volume
                particle_mass = float(layout.entry["density"]) * volume / 4.0
                for vertex in tetrahedron:
                    mass_per_environment[layout.particle_start + int(vertex)] += particle_mass

            if layout.kind == "deformable" and bool(layout.entry["kinematic"]):
                kinematic_per_environment[particle_slice] = 1
            if layout.kind == "tactile":
                stick_indices = tuple(int(value) for value in layout.entry["stick_vertex_indices"])
                kinematic_per_environment[
                    np.asarray(stick_indices, dtype=np.int32) + layout.particle_start
                ] = 1
                attachment = layout.entry.get("attachment")
                if attachment is not None:
                    link_path = str(attachment["link_path"])
                    matches = [
                        robot.body_by_path[link_path]
                        for robot in self.robot_layouts
                        if link_path in robot.body_by_path
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            f"tactile attachment {link_path!r} does not resolve uniquely"
                        )
                    link_local_matrix = _matrix(attachment["transform"])
                    attached_reset.append(
                        (
                            layout.particle_start,
                            layout.vertex_count,
                            matches[0],
                            _transform_points(link_local_matrix, layout.mesh.vertices).astype(
                                np.float32
                            ),
                        )
                    )
                    for vertex in stick_indices:
                        attached_particle.append(layout.particle_start + vertex)
                        attached_body.append(matches[0])
                        attached_local.append(
                            _transform_points(
                                link_local_matrix,
                                layout.mesh.vertices[vertex : vertex + 1],
                            )[0]
                        )

        if np.any(mass_per_environment <= 0.0):
            raise ValueError("compiled IPC tetrahedral topology has a massless vertex")

        self.particle_count = particles_per_environment * self.num_envs
        self.tetrahedron_count = tetrahedra_per_environment * self.num_envs
        initial = np.tile(initial_per_environment, (self.num_envs, 1))
        masses = np.tile(mass_per_environment, self.num_envs)
        damping = np.tile(damping_per_environment, self.num_envs)
        kinematic = np.tile(kinematic_per_environment, self.num_envs)
        tetrahedron_indices = np.concatenate(
            [
                tetrahedron_indices_per_environment
                + environment * particles_per_environment
                for environment in range(self.num_envs)
            ]
        )
        inverse_rest = np.tile(inverse_rest_per_environment, (self.num_envs, 1, 1))
        rest_volume = np.tile(rest_volume_per_environment, self.num_envs)
        shear = np.tile(shear_per_environment, self.num_envs)
        lame = np.tile(lame_per_environment, self.num_envs)

        attached_particles = np.asarray(
            [
                particle + environment * particles_per_environment
                for environment in range(self.num_envs)
                for particle in attached_particle
            ],
            dtype=np.int32,
        )
        attached_bodies = np.asarray(
            [
                body + environment * self.bodies_per_environment
                for environment in range(self.num_envs)
                for body in attached_body
            ],
            dtype=np.int32,
        )
        attached_positions = np.asarray(
            attached_local * self.num_envs, dtype=np.float64
        ).reshape(-1, 3)

        pt_point_per_environment: list[int] = []
        pt_a_per_environment: list[int] = []
        pt_b_per_environment: list[int] = []
        pt_c_per_environment: list[int] = []
        deformables = [layout for layout in self.soft_layouts if layout.kind == "deformable"]
        sensors = [layout for layout in self.soft_layouts if layout.kind == "tactile"]
        for sensor in sensors:
            if not bool(sensor.entry["enabled"]):
                continue
            sensor_layer = int(sensor.entry["collision_layer"])
            sensor_mask = int(sensor.entry["collision_mask"])
            for body in deformables:
                body_layer = int(body.entry["collision_layer"])
                body_mask = int(body.entry["collision_mask"])
                if not (sensor_layer & body_mask and body_layer & sensor_mask):
                    continue
                for coat_vertex in sensor.entry["coat_vertex_indices"]:
                    point = sensor.particle_start + int(coat_vertex)
                    for triangle in body.mesh.surface_triangles:
                        pt_point_per_environment.append(point)
                        pt_a_per_environment.append(body.particle_start + int(triangle[0]))
                        pt_b_per_environment.append(body.particle_start + int(triangle[1]))
                        pt_c_per_environment.append(body.particle_start + int(triangle[2]))

        pt_per_environment = len(pt_point_per_environment)
        self.pt_count = pt_per_environment * self.num_envs
        if self.pt_count > int(self.config.pt_capacity):
            raise SimulationCapacityError(
                f"Warp IPC PT candidate capacity exceeded: {self.pt_count}/{self.config.pt_capacity}"
            )
        if int(self.config.ee_capacity) < 0:
            raise SimulationCapacityError("Warp IPC EE capacity is invalid")
        required_hessian = self.num_envs * (
            particles_per_environment
            + 16 * tetrahedra_per_environment
            + 16 * pt_per_environment
        )
        if required_hessian > int(self.config.hessian_capacity):
            raise SimulationCapacityError(
                "Warp IPC Hessian capacity exceeded: "
                f"{required_hessian}/{self.config.hessian_capacity}"
            )

        def replicated(values: Sequence[int]) -> np.ndarray:
            return np.asarray(
                [
                    int(value) + environment * particles_per_environment
                    for environment in range(self.num_envs)
                    for value in values
                ],
                dtype=np.int32,
            )

        self.initial_position = wp.array(initial, dtype=wp.vec3d, device=self.wp_device)
        self.position = wp.array(initial, dtype=wp.vec3d, device=self.wp_device)
        self.velocity = wp.zeros(self.particle_count, dtype=wp.vec3d, device=self.wp_device)
        self.contact_velocity = wp.zeros(
            self.particle_count, dtype=wp.vec3d, device=self.wp_device
        )
        self.predicted = wp.empty_like(self.position)
        self.iterate = wp.empty_like(self.position)
        self.line_search_origin = wp.empty_like(self.position)
        self.line_search_candidate = wp.empty_like(self.position)
        self.mass = wp.array(masses, dtype=wp.float64, device=self.wp_device)
        self.damping = wp.array(damping, dtype=wp.float64, device=self.wp_device)
        self.kinematic = wp.array(kinematic, dtype=wp.int32, device=self.wp_device)
        self.initial_kinematic = wp.array(
            kinematic, dtype=wp.int32, device=self.wp_device
        )
        self.kinematic_target = wp.array(
            initial, dtype=wp.vec3d, device=self.wp_device
        )
        self.gradient = wp.zeros(self.particle_count, dtype=wp.vec3d, device=self.wp_device)
        self.hessian = wp.zeros(
            (self.particle_count, 3, 3), dtype=wp.float64, device=self.wp_device
        )
        self.contact_force = wp.zeros(
            self.particle_count, dtype=wp.vec3d, device=self.wp_device
        )
        self.public_position = wp.array(
            initial.astype(np.float32), dtype=wp.vec3, device=self.wp_device
        )
        self.public_velocity = wp.zeros(
            self.particle_count, dtype=wp.vec3, device=self.wp_device
        )
        self.public_contact_force = wp.zeros(
            self.particle_count, dtype=wp.vec3, device=self.wp_device
        )
        self.tetrahedron_indices = wp.array(
            tetrahedron_indices, dtype=wp.int32, device=self.wp_device
        )
        self.inverse_rest_matrix = wp.array(
            inverse_rest, dtype=wp.mat33d, device=self.wp_device
        )
        self.rest_volume = wp.array(
            rest_volume, dtype=wp.float64, device=self.wp_device
        )
        self.shear_modulus = wp.array(shear, dtype=wp.float64, device=self.wp_device)
        self.lame_lambda = wp.array(lame, dtype=wp.float64, device=self.wp_device)
        self.tetrahedron_position = wp.zeros(
            (self.tetrahedron_count, 4), dtype=wp.vec3d, device=self.wp_device
        )
        self.tetrahedron_energy = wp.zeros(
            self.tetrahedron_count, dtype=wp.float64, device=self.wp_device
        )
        self.tetrahedron_gradient = wp.zeros(
            (self.tetrahedron_count, 4), dtype=wp.vec3d, device=self.wp_device
        )
        self.tetrahedron_hessian = wp.zeros(
            (self.tetrahedron_count, 12, 12), dtype=wp.float64, device=self.wp_device
        )
        self.tetrahedron_valid = wp.zeros(
            self.tetrahedron_count, dtype=wp.int32, device=self.wp_device
        )
        self.pt_point = wp.array(
            replicated(pt_point_per_environment), dtype=wp.int32, device=self.wp_device
        )
        self.pt_a = wp.array(
            replicated(pt_a_per_environment), dtype=wp.int32, device=self.wp_device
        )
        self.pt_b = wp.array(
            replicated(pt_b_per_environment), dtype=wp.int32, device=self.wp_device
        )
        self.pt_c = wp.array(
            replicated(pt_c_per_environment), dtype=wp.int32, device=self.wp_device
        )
        self.pt_environment = wp.array(
            np.repeat(
                np.arange(self.num_envs, dtype=np.int32), pt_per_environment
            ),
            dtype=wp.int32,
            device=self.wp_device,
        )
        self.ccd_fraction = wp.ones(
            self.num_envs, dtype=wp.float64, device=self.wp_device
        )
        self.active_contact_count = wp.zeros(1, dtype=wp.int32, device=self.wp_device)
        self.pt_barycentric = wp.zeros(
            self.pt_count, dtype=wp.vec3d, device=self.wp_device
        )
        self.pt_hessian = wp.zeros(
            self.pt_count, dtype=wp.mat33d, device=self.wp_device
        )
        self.invalid_state = wp.zeros(1, dtype=wp.int32, device=self.wp_device)
        diagonal_rows = np.arange(self.particle_count, dtype=np.int32)
        tetrahedron_rows = np.repeat(tetrahedron_indices, 4, axis=1).reshape(-1)
        tetrahedron_columns = np.tile(tetrahedron_indices, (1, 4)).reshape(-1)
        contact_indices = np.stack(
            (
                replicated(pt_point_per_environment),
                replicated(pt_a_per_environment),
                replicated(pt_b_per_environment),
                replicated(pt_c_per_environment),
            ),
            axis=1,
        )
        contact_rows = np.repeat(contact_indices, 4, axis=1).reshape(-1)
        contact_columns = np.tile(contact_indices, (1, 4)).reshape(-1)
        system_rows = np.concatenate(
            (diagonal_rows, tetrahedron_rows, contact_rows)
        )
        system_columns = np.concatenate(
            (diagonal_rows, tetrahedron_columns, contact_columns)
        )
        self.system_rows = wp.array(
            system_rows, dtype=wp.int32, device=self.wp_device
        )
        self.system_columns = wp.array(
            system_columns, dtype=wp.int32, device=self.wp_device
        )
        self.system_values = wp.zeros(
            system_rows.size, dtype=wp.mat33d, device=self.wp_device
        )
        self.system_rhs = wp.zeros(
            self.particle_count, dtype=wp.vec3d, device=self.wp_device
        )
        self.system_solution = wp.zeros(
            self.particle_count, dtype=wp.vec3d, device=self.wp_device
        )
        self.system_matrix = assemble_bsr33_from_triplets(
            self.particle_count,
            self.system_rows,
            self.system_columns,
            self.system_values,
        )
        self._cg_solver = FixedCgSolver(
            self.system_matrix,
            self.system_rhs,
            self.system_solution,
            max_iterations=int(self.config.cg_iterations),
            relative_tolerance=1.0e-5,
        )
        self.attached_particle = wp.array(
            attached_particles, dtype=wp.int32, device=self.wp_device
        )
        self.attached_body = wp.array(
            attached_bodies, dtype=wp.int32, device=self.wp_device
        )
        self.attached_local_position = wp.array(
            attached_positions, dtype=wp.vec3d, device=self.wp_device
        )
        self.attached_count = int(attached_particles.size)
        self.attached_reset = tuple(
            (
                particle_start,
                vertex_count,
                body,
                torch.as_tensor(local, dtype=torch.float32, device=self.torch_device),
            )
            for particle_start, vertex_count, body, local in attached_reset
        )

        self.position_tensor = wp.to_torch(self.public_position).reshape(
            self.num_envs, particles_per_environment, 3
        )
        self.velocity_tensor = wp.to_torch(self.public_velocity).reshape(
            self.num_envs, particles_per_environment, 3
        )
        self.contact_force_tensor = wp.to_torch(self.public_contact_force).reshape(
            self.num_envs, particles_per_environment, 3
        )
        self.position_f64_tensor = wp.to_torch(self.position).reshape(
            self.num_envs, particles_per_environment, 3
        )
        self.velocity_f64_tensor = wp.to_torch(self.velocity).reshape(
            self.num_envs, particles_per_environment, 3
        )
        self.contact_velocity_f64_tensor = wp.to_torch(self.contact_velocity).reshape(
            self.num_envs, particles_per_environment, 3
        )
        self.line_search_origin_f64_tensor = wp.to_torch(
            self.line_search_origin
        ).reshape(self.num_envs, particles_per_environment, 3)
        self.kinematic_tensor = wp.to_torch(self.kinematic).reshape(
            self.num_envs, particles_per_environment
        )
        self.initial_kinematic_tensor = wp.to_torch(self.initial_kinematic).reshape(
            self.num_envs, particles_per_environment
        )
        self.kinematic_target_tensor = wp.to_torch(self.kinematic_target).reshape(
            self.num_envs, particles_per_environment, 3
        )
        self.body_pose_tensor = wp.to_torch(self.state.body_q).reshape(
            self.num_envs, self.bodies_per_environment, 7
        )
        self.body_velocity_tensor = wp.to_torch(self.state.body_qd).reshape(
            self.num_envs, self.bodies_per_environment, 6
        )
        self.joint_q_tensor = wp.to_torch(self.model.joint_q).reshape(
            self.num_envs, self.joint_coordinates_per_environment
        )
        self.joint_qd_tensor = wp.to_torch(self.model.joint_qd).reshape(
            self.num_envs, self.joint_dofs_per_environment
        )
        self._arrays = MappingProxyType(
            {
                "particle_position": self.position_tensor,
                "particle_velocity": self.velocity_tensor,
                "particle_contact_force": self.contact_force_tensor,
                "body_q": self.body_pose_tensor,
                "body_qd": self.body_velocity_tensor,
                "joint_q": self.joint_q_tensor,
                "joint_qd": self.joint_qd_tensor,
            }
        )

    def _initialize_robot_targets(self) -> None:
        self.joint_target_tensor = self.joint_q_tensor.clone()
        self.initial_joint_q_tensor = self.joint_q_tensor.clone()
        self.initial_joint_qd_tensor = self.joint_qd_tensor.clone()

    @property
    def arrays(self) -> Mapping[str, Any]:
        return self._arrays

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        return {
            "newton_iteration": int(self.config.newton_iterations),
            "cg_iteration": self._last_cg_iteration,
            "newton_status": "fixed-iteration BSR Newton correctness solve",
            "cg_status": self._last_cg_status,
            "active_pt_contacts": self._last_active_contacts,
            "pt_candidates": self.pt_count,
            "ee_candidates": 0,
            "tetrahedra": self.tetrahedron_count,
            "particles": self.particle_count,
            "step_latency_ms": self._last_latency_ms,
            "line_search_backtracks": self._last_line_search_backtracks,
            "line_search_fallbacks": dict(self._last_line_search_fallbacks),
            "ccd_clamps": self._last_ccd_clamps,
            "graph_status": "disabled-correctness",
        }

    def _update_robot_fk(self, state: Any, dt: float) -> None:
        if not self.robot_layouts:
            return
        self.joint_qd_tensor.zero_()
        for layout in self.robot_layouts:
            for name, q_index in layout.joint_q_by_name.items():
                qd_index = layout.joint_qd_by_name[name]
                self.joint_qd_tensor[:, qd_index].copy_(
                    (self.joint_target_tensor[:, q_index] - self.joint_q_tensor[:, q_index])
                    / float(dt)
                )
            self.joint_qd_tensor[
                :, layout.base_qd_start : layout.base_qd_start + 6
            ].zero_()
        self.joint_q_tensor.copy_(self.joint_target_tensor)
        newton.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            state,
        )

    def _update_attached_targets(self, state: Any) -> None:
        if not self.attached_count:
            return
        wp.launch(
            update_attached_targets_kernel,
            dim=self.attached_count,
            inputs=[
                state.body_q,
                self.attached_particle,
                self.attached_body,
                self.attached_local_position,
            ],
            outputs=[self.kinematic_target],
            device=self.wp_device,
        )

    def _reset_attached_gels(self, reset_mask: torch.Tensor) -> None:
        for particle_start, vertex_count, body, local in self.attached_reset:
            pose = self.body_pose_tensor[:, body]
            local_batch = local[None].expand(self.num_envs, -1, -1)
            quaternion = pose[:, None, 3:7].expand(-1, vertex_count, -1)
            world = pose[:, None, :3] + _quat_apply(quaternion, local_batch)
            destination = slice(particle_start, particle_start + vertex_count)
            position = self.position_f64_tensor[:, destination]
            velocity = self.velocity_f64_tensor[:, destination]
            position.copy_(
                torch.where(reset_mask[:, None, None], world.to(torch.float64), position)
            )
            velocity.copy_(
                torch.where(reset_mask[:, None, None], torch.zeros_like(velocity), velocity)
            )
            public_position = self.position_tensor[:, destination]
            public_velocity = self.velocity_tensor[:, destination]
            public_force = self.contact_force_tensor[:, destination]
            public_position.copy_(
                torch.where(reset_mask[:, None, None], world, public_position)
            )
            public_velocity.copy_(
                torch.where(
                    reset_mask[:, None, None],
                    torch.zeros_like(public_velocity),
                    public_velocity,
                )
            )
            public_force.copy_(
                torch.where(
                    reset_mask[:, None, None],
                    torch.zeros_like(public_force),
                    public_force,
                )
            )
        wp.launch(
            copy_particles_to_newton_kernel,
            dim=self.particle_count,
            inputs=[self.position, self.velocity],
            outputs=[self.state.particle_q, self.state.particle_qd],
            device=self.wp_device,
        )

    def _finish_tick_checks(self) -> None:
        wp.synchronize_device(self.wp_device)
        if int(self.invalid_state.numpy()[0]) != 0:
            raise RuntimeError("Warp IPC produced a non-finite particle state")
        valid = self.tetrahedron_valid.numpy()
        if valid.size and not bool(np.all(valid == 1)):
            invalid_count = int(np.count_nonzero(valid != 1))
            raise RuntimeError(
                f"Warp IPC line search could not preserve {invalid_count} positive tetrahedra"
            )
        self._last_active_contacts = int(self.active_contact_count.numpy()[0])
        self._last_cg_iteration = int(self._cg_solver.iteration_count.numpy()[0])
        cg_invalid = int(self._cg_solver.invalid.numpy()[0])
        cg_active = int(self._cg_solver.active.numpy()[0])
        if cg_invalid:
            self._last_cg_status = "invalid"
            raise RuntimeError("Warp IPC preconditioned CG encountered a non-SPD system")
        self._last_cg_status = "converged" if not cg_active else "maximum-iterations"
        if self._last_active_contacts > int(self.config.pt_capacity):
            raise SimulationCapacityError(
                "Warp IPC active PT contact count exceeded its fixed capacity"
            )

    def _accept_positive_iterate(self, origin: Any, description: str) -> None:
        wp.copy(self.line_search_candidate, self.iterate)
        if self.pt_count:
            # IPC's barrier distance is an activation range, not collision
            # geometry thickness. Preserve only the configured micron-scale
            # numerical clearance so an unsigned PT normal never degenerates
            # at exactly zero distance.
            ccd_thickness = self.config.ccd_tolerance
            wp.launch(
                initialize_ccd_fraction_kernel,
                dim=self.num_envs,
                outputs=[self.ccd_fraction],
                device=self.wp_device,
            )
            wp.launch(
                point_triangle_line_search_kernel,
                dim=self.pt_count,
                inputs=[
                    origin,
                    self.line_search_candidate,
                    self.pt_point,
                    self.pt_a,
                    self.pt_b,
                    self.pt_c,
                    self.pt_environment,
                    wp.float64(ccd_thickness),
                    wp.float64(
                        max(min(1.0e-3 * self.config.ccd_tolerance, 1.0e-9), 1.0e-12)
                    ),
                ],
                outputs=[self.ccd_fraction],
                device=self.wp_device,
            )
            fractions = self.ccd_fraction.numpy()
            if bool(np.any(fractions < 1.0)):
                self._last_ccd_clamps += int(np.count_nonzero(fractions < 1.0))
                wp.launch(
                    apply_ccd_fraction_kernel,
                    dim=self.particle_count,
                    inputs=[
                        origin,
                        self.line_search_candidate,
                        self.ccd_fraction,
                        wp.int32(self.particles_per_environment),
                    ],
                    outputs=[self.iterate],
                    device=self.wp_device,
                )
                wp.copy(self.line_search_candidate, self.iterate)
        for backtracks in range(13):
            if backtracks:
                wp.launch(
                    blend_line_search_kernel,
                    dim=self.particle_count,
                    inputs=[
                        origin,
                        self.line_search_candidate,
                        wp.float64(0.5**backtracks),
                    ],
                    outputs=[self.iterate],
                    device=self.wp_device,
                )
            wp.launch(
                tetrahedron_validity_kernel,
                dim=self.tetrahedron_count,
                inputs=[
                    self.iterate,
                    self.tetrahedron_indices,
                    self.rest_volume,
                ],
                outputs=[self.tetrahedron_valid],
                device=self.wp_device,
            )
            valid = self.tetrahedron_valid.numpy()
            if bool(np.all(valid == 1)):
                self._last_line_search_backtracks += backtracks
                return
        wp.copy(self.iterate, origin)
        wp.launch(
            tetrahedron_validity_kernel,
            dim=self.tetrahedron_count,
            inputs=[origin, self.tetrahedron_indices, self.rest_volume],
            outputs=[self.tetrahedron_valid],
            device=self.wp_device,
        )
        invalid = int(np.count_nonzero(self.tetrahedron_valid.numpy() != 1))
        if invalid == 0:
            self._last_line_search_backtracks += 13
            self._last_line_search_fallbacks[description] = (
                self._last_line_search_fallbacks.get(description, 0) + 1
            )
            return
        raise RuntimeError(
            f"Warp IPC {description} line search has {invalid} invalid origin tetrahedra"
        )

    def step(self, actions: Any | None, *, nsteps: int) -> None:
        if self._closed:
            raise RuntimeError("Warp IPC runtime is closed")
        if actions is not None:
            if len(self.robot_layouts) != 1:
                raise ValueError(
                    "implicit Warp IPC actions require exactly one compiled robot"
                )
            layout = self.robot_layouts[0]
            names = tuple(layout.joint_q_by_name)
            adapter = _RobotAdapter(
                self,
                RobotBatchSpec(layout.name, layout.base_link_name, names),
                layout,
            )
            adapter.set_position_targets(actions)
        started = time.perf_counter()
        self._last_line_search_backtracks = 0
        self._last_line_search_fallbacks = {}
        self._last_ccd_clamps = 0
        for _ in range(int(nsteps)):
            self._solver.step(
                self.state,
                self.state,
                self.control,
                None,
                float(self.config.fixed_time_step),
            )
            self._step_count += 1
        self._last_latency_ms = (time.perf_counter() - started) * 1000.0 / float(nsteps)

    def _mask_tensor(self, reset_mask: Any) -> torch.Tensor:
        result = torch.as_tensor(
            reset_mask, device=self.torch_device, dtype=torch.bool
        )
        if tuple(result.shape) != (self.num_envs,):
            raise ValueError(
                f"Warp IPC reset mask has shape {tuple(result.shape)}, "
                f"expected {(self.num_envs,)}"
            )
        return result.contiguous()

    def _reset_particles(self, reset_mask: torch.Tensor) -> None:
        mask_array = wp.from_torch(reset_mask, dtype=wp.bool)
        wp.launch(
            reset_masked_particles_kernel,
            dim=self.particle_count,
            inputs=[
                mask_array,
                wp.int32(self.particles_per_environment),
                self.initial_position,
            ],
            outputs=[
                self.position,
                self.velocity,
                self.predicted,
                self.iterate,
                self.kinematic_target,
                self.public_position,
                self.public_velocity,
                self.public_contact_force,
            ],
            device=self.wp_device,
        )
        self.kinematic_tensor.copy_(
            torch.where(
                reset_mask[:, None],
                self.initial_kinematic_tensor,
                self.kinematic_tensor,
            )
        )
        wp.launch(
            copy_particles_to_newton_kernel,
            dim=self.particle_count,
            inputs=[self.position, self.velocity],
            outputs=[self.state.particle_q, self.state.particle_qd],
            device=self.wp_device,
        )

    def reset(self, reset_mask: Any, **state: Any) -> None:
        mask = self._mask_tensor(reset_mask)
        self._reset_particles(mask)
        self.joint_target_tensor.copy_(
            torch.where(mask[:, None], self.initial_joint_q_tensor, self.joint_target_tensor)
        )
        self.joint_q_tensor.copy_(
            torch.where(mask[:, None], self.initial_joint_q_tensor, self.joint_q_tensor)
        )
        self.joint_qd_tensor.copy_(
            torch.where(mask[:, None], self.initial_joint_qd_tensor, self.joint_qd_tensor)
        )
        if state:
            unknown = sorted(set(state) - {"joint_position", "joint_velocity", "base_pose"})
            if unknown:
                raise TypeError(f"unknown Warp IPC reset state fields: {unknown}")
        self._update_robot_fk(self.state, float(self.config.fixed_time_step))
        self._update_attached_targets(self.state)

    def create_robot_view_adapter(self, spec: RobotBatchSpec) -> Any:
        robot_entry = _resolve_unique(self.artifact.robots, spec.robot_name, "robot")
        matches = [
            layout
            for layout in self.robot_layouts
            if layout.path == str(robot_entry["path"])
        ]
        if len(matches) != 1:
            raise RuntimeError("compiled IPC robot has no private Newton layout")
        return _RobotAdapter(self, spec, matches[0])

    def create_deformable_view_adapter(
        self, spec: Any, entries: Sequence[Mapping[str, Any]]
    ) -> Any:
        layouts = tuple(
            next(
                layout
                for layout in self.soft_layouts
                if layout.kind == "deformable" and layout.path == str(entry["path"])
            )
            for entry in entries
        )
        return _DeformableAdapter(self, spec, layouts)

    def create_tactile_view_adapter(
        self, spec: Any, entries: Sequence[Mapping[str, Any]]
    ) -> Any:
        layouts = tuple(
            next(
                layout
                for layout in self.soft_layouts
                if layout.kind == "tactile" and layout.path == str(entry["path"])
            )
            for entry in entries
        )
        return _TactileAdapter(self, spec, layouts)

    def close(self) -> None:
        self._closed = True

    def _add_soft_bodies(self, blueprint: Any) -> tuple[_SoftLayout, ...]:
        layouts: list[_SoftLayout] = []
        values = [
            ("deformable", entry, "mesh_blob", "vertex_count", "tetrahedron_count")
            for entry in self.artifact.deformable_bodies
        ] + [
            ("tactile", entry, "gel_mesh_blob", "gel_vertex_count", "gel_tetrahedron_count")
            for entry in self.artifact.tactile_sensors
        ]
        for kind, entry, blob_field, vertex_field, tetrahedron_field in values:
            mesh = _decode_mesh(self.artifact.blobs[str(entry[blob_field])])
            if mesh.vertices.shape[0] != int(entry[vertex_field]) or mesh.tetrahedra.shape[0] != int(
                entry[tetrahedron_field]
            ):
                raise ValueError(f"compiled IPC {kind} mesh metadata is inconsistent")
            transform = _matrix(entry["transform"])
            world_vertices = _transform_points(transform, mesh.vertices)
            particle_start = len(blueprint.particle_q)
            tetrahedron_start = len(blueprint.tet_indices)
            shear, lame = _lame_parameters(
                float(entry["young_modulus"]), float(entry["poisson_ratio"])
            )
            blueprint.add_soft_mesh(
                pos=wp.vec3(0.0, 0.0, 0.0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=[wp.vec3(*tuple(float(value) for value in vertex)) for vertex in world_vertices],
                indices=mesh.tetrahedra.reshape(-1).tolist(),
                density=float(entry["density"]),
                k_mu=shear,
                k_lambda=lame,
                k_damp=float(entry["damping"]),
                add_surface_mesh_edges=False,
                label=str(entry["name"]),
            )
            layouts.append(
                _SoftLayout(
                    kind=kind,
                    path=str(entry["path"]),
                    name=str(entry["name"]),
                    entry=entry,
                    mesh=mesh,
                    particle_start=particle_start,
                    tetrahedron_start=tetrahedron_start,
                    world_vertices=world_vertices,
                )
            )
        return tuple(layouts)


class _RobotAdapter:
    def __init__(
        self,
        session: _WarpIpcSession,
        spec: RobotBatchSpec,
        layout: _RobotLayout,
    ) -> None:
        self.session = session
        self.spec = spec
        self.layout = layout
        if spec.base_link not in (layout.base_link_name,):
            base_matches = [
                name
                for name, index in layout.body_by_name.items()
                if index == layout.base_body_index and name == spec.base_link
            ]
            if not base_matches:
                raise KeyError(
                    f"compiled IPC robot {layout.name!r} has no base link {spec.base_link!r}"
                )
        missing_joints = [
            name for name in spec.joint_names if name not in layout.joint_q_by_name
        ]
        if missing_joints:
            raise KeyError(
                f"compiled IPC robot {layout.name!r} has no movable joints {missing_joints}"
            )
        link_indices: list[int] = []
        for requested in spec.link_names:
            if requested in layout.body_by_name:
                link_indices.append(layout.body_by_name[requested])
            elif requested in layout.body_by_path:
                link_indices.append(layout.body_by_path[requested])
            else:
                raise KeyError(
                    f"compiled IPC robot {layout.name!r} has no link {requested!r}"
                )
        device = session.torch_device
        self.joint_q = torch.as_tensor(
            [layout.joint_q_by_name[name] for name in spec.joint_names],
            dtype=torch.long,
            device=device,
        )
        self.joint_qd = torch.as_tensor(
            [layout.joint_qd_by_name[name] for name in spec.joint_names],
            dtype=torch.long,
            device=device,
        )
        self.links = torch.as_tensor(link_indices, dtype=torch.long, device=device)
        self.lower = torch.as_tensor(
            [layout.joint_limits_by_name[name][0] for name in spec.joint_names],
            dtype=torch.float32,
            device=device,
        )
        self.upper = torch.as_tensor(
            [layout.joint_limits_by_name[name][1] for name in spec.joint_names],
            dtype=torch.float32,
            device=device,
        )

    def read_state(self, state: RobotBatchState | None) -> RobotBatchState:
        session = self.session
        shape = (session.num_envs, len(self.spec.joint_names))
        link_shape = (session.num_envs, len(self.spec.link_names), 7)
        if state is None:
            state = RobotBatchState(
                torch.empty((session.num_envs, 7), dtype=torch.float32, device=session.torch_device),
                torch.empty((session.num_envs, 6), dtype=torch.float32, device=session.torch_device),
                torch.empty(shape, dtype=torch.float32, device=session.torch_device),
                torch.empty(shape, dtype=torch.float32, device=session.torch_device),
                torch.empty(shape, dtype=torch.float32, device=session.torch_device),
                torch.empty(link_shape, dtype=torch.float32, device=session.torch_device),
            )
        state.base_pose.copy_(
            session.body_pose_tensor[:, self.layout.base_body_index]
        )
        state.base_velocity.copy_(
            session.body_velocity_tensor[:, self.layout.base_body_index]
        )
        torch.index_select(session.joint_q_tensor, 1, self.joint_q, out=state.joint_position)
        torch.index_select(session.joint_qd_tensor, 1, self.joint_qd, out=state.joint_velocity)
        torch.index_select(
            session.joint_target_tensor, 1, self.joint_q, out=state.joint_control
        )
        if len(self.spec.link_names):
            torch.index_select(session.body_pose_tensor, 1, self.links, out=state.link_pose)
        return state

    def set_position_targets(self, targets: Any) -> None:
        values = _as_device_tensor(
            targets,
            device=self.session.torch_device,
            dtype=torch.float32,
            shape=(self.session.num_envs, len(self.spec.joint_names)),
            description="robot position targets",
        )
        values = torch.maximum(torch.minimum(values, self.upper), self.lower)
        self.session.joint_target_tensor[:, self.joint_q] = values

    def set_base_pose_targets(self, targets: Any) -> None:
        values = _as_device_tensor(
            targets,
            device=self.session.torch_device,
            dtype=torch.float32,
            shape=(self.session.num_envs, 7),
            description="robot base pose targets",
        )
        quaternion = values[:, 3:7]
        length = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
        if bool(torch.any(length <= 1.0e-8)):
            raise ValueError("Warp IPC base pose contains a zero quaternion")
        normalized = torch.cat((values[:, :3], quaternion / length), dim=-1)
        self.session.joint_target_tensor[
            :, self.layout.base_q_start : self.layout.base_q_start + 7
        ].copy_(normalized)

    def set_controls(self, controls: Any) -> None:
        self.set_position_targets(controls)

    def reset(self, reset_mask: Any, **state_values: Any) -> Mapping[str, Any]:
        session = self.session
        mask = session._mask_tensor(reset_mask)
        session.reset(mask)
        if "base_pose" in state_values:
            values = _as_device_tensor(
                state_values["base_pose"],
                device=session.torch_device,
                dtype=torch.float32,
                shape=(session.num_envs, 7),
                description="robot reset base_pose",
            )
            target = session.joint_target_tensor[
                :, self.layout.base_q_start : self.layout.base_q_start + 7
            ]
            target.copy_(torch.where(mask[:, None], values, target))
        if "joint_position" in state_values:
            values = _as_device_tensor(
                state_values["joint_position"],
                device=session.torch_device,
                dtype=torch.float32,
                shape=(session.num_envs, len(self.spec.joint_names)),
                description="robot reset joint_position",
            )
            values = torch.maximum(torch.minimum(values, self.upper), self.lower)
            current = session.joint_target_tensor[:, self.joint_q]
            session.joint_target_tensor[:, self.joint_q] = torch.where(
                mask[:, None], values, current
            )
        if "joint_velocity" in state_values:
            values = _as_device_tensor(
                state_values["joint_velocity"],
                device=session.torch_device,
                dtype=torch.float32,
                shape=(session.num_envs, len(self.spec.joint_names)),
                description="robot reset joint_velocity",
            )
            current = session.joint_qd_tensor[:, self.joint_qd]
            session.joint_qd_tensor[:, self.joint_qd] = torch.where(
                mask[:, None], values, current
            )
        unknown = sorted(
            set(state_values) - {"base_pose", "joint_position", "joint_velocity"}
        )
        if unknown:
            raise TypeError(f"unknown Warp IPC robot reset fields: {unknown}")
        session.joint_q_tensor.copy_(session.joint_target_tensor)
        session._update_robot_fk(session.state, float(session.config.fixed_time_step))
        session._reset_attached_gels(mask)
        session._update_attached_targets(session.state)
        return session.arrays


class _DeformableAdapter:
    def __init__(self, session: _WarpIpcSession, spec: Any, layouts: Sequence[_SoftLayout]) -> None:
        self.session = session
        self.spec = spec
        self.layouts = tuple(layouts)
        self.max_vertices = max(layout.vertex_count for layout in self.layouts)

    def read_state(self, state: DeformableBatchState | None) -> DeformableBatchState:
        shape = (
            self.session.num_envs,
            len(self.layouts),
            self.max_vertices,
            3,
        )
        if state is None:
            state = DeformableBatchState(
                torch.zeros(shape, dtype=torch.float32, device=self.session.torch_device),
                torch.zeros(shape, dtype=torch.float32, device=self.session.torch_device),
                torch.zeros(shape, dtype=torch.float32, device=self.session.torch_device),
            )
        for body, layout in enumerate(self.layouts):
            source = slice(layout.particle_start, layout.particle_start + layout.vertex_count)
            state.position[:, body, : layout.vertex_count].copy_(
                self.session.position_tensor[:, source]
            )
            state.velocity[:, body, : layout.vertex_count].copy_(
                self.session.velocity_tensor[:, source]
            )
            state.contact_force[:, body, : layout.vertex_count].copy_(
                self.session.contact_force_tensor[:, source]
            )
        return state

    def set_kinematic_targets(self, targets: Any, *, target_mask: Any | None = None) -> None:
        session = self.session
        shape = (
            session.num_envs,
            len(self.layouts),
            self.max_vertices,
            3,
        )
        values = _as_device_tensor(
            targets,
            device=session.torch_device,
            dtype=torch.float64,
            shape=shape,
            description="deformable kinematic targets",
        )
        if target_mask is None:
            mask = torch.ones(shape[:-1], dtype=torch.bool, device=session.torch_device)
        else:
            mask = torch.as_tensor(target_mask, dtype=torch.bool, device=session.torch_device)
            if tuple(mask.shape) == (session.num_envs, len(self.layouts)):
                mask = mask[:, :, None].expand(shape[:-1])
            if tuple(mask.shape) != shape[:-1]:
                raise ValueError(
                    f"Warp IPC deformable target mask has shape {tuple(mask.shape)}, "
                    f"expected {shape[:-1]}"
                )
        for body, layout in enumerate(self.layouts):
            destination = slice(
                layout.particle_start, layout.particle_start + layout.vertex_count
            )
            body_mask = mask[:, body, : layout.vertex_count]
            target = session.kinematic_target_tensor[:, destination]
            target.copy_(
                torch.where(
                    body_mask[..., None],
                    values[:, body, : layout.vertex_count],
                    target,
                )
            )
            active = session.kinematic_tensor[:, destination]
            authored = session.initial_kinematic_tensor[:, destination]
            active.copy_(
                torch.where(body_mask, torch.ones_like(active), authored)
            )

    def reset(self, reset_mask: Any, **state_values: Any) -> Mapping[str, Any]:
        session = self.session
        mask = session._mask_tensor(reset_mask)
        session.reset(mask)
        expected = (
            session.num_envs,
            len(self.layouts),
            self.max_vertices,
            3,
        )
        for field, target in (
            ("position", session.position_f64_tensor),
            ("velocity", session.velocity_f64_tensor),
        ):
            if field not in state_values:
                continue
            values = _as_device_tensor(
                state_values[field],
                device=session.torch_device,
                dtype=torch.float64,
                shape=expected,
                description=f"deformable reset {field}",
            )
            for body, layout in enumerate(self.layouts):
                destination = slice(
                    layout.particle_start, layout.particle_start + layout.vertex_count
                )
                current = target[:, destination]
                current.copy_(
                    torch.where(
                        mask[:, None, None],
                        values[:, body, : layout.vertex_count],
                        current,
                    )
                )
        unknown = sorted(set(state_values) - {"position", "velocity"})
        if unknown:
            raise TypeError(f"unknown Warp IPC deformable reset fields: {unknown}")
        wp.launch(
            copy_particles_to_newton_kernel,
            dim=session.particle_count,
            inputs=[session.position, session.velocity],
            outputs=[session.state.particle_q, session.state.particle_qd],
            device=session.wp_device,
        )
        session.position_tensor.copy_(session.position_f64_tensor)
        session.velocity_tensor.copy_(session.velocity_f64_tensor)
        return session.arrays


class _TactileAdapter:
    def __init__(self, session: _WarpIpcSession, spec: Any, layouts: Sequence[_SoftLayout]) -> None:
        self.session = session
        self.spec = spec
        self.layouts = tuple(layouts)
        first = self.layouts[0].entry
        self.height, self.width = (int(value) for value in first["resolution"])
        self.vertex_count = self.layouts[0].vertex_count
        self.marker_count = len(first["marker_positions"])
        self._prepare_sampling()
        shape_prefix = (session.num_envs, len(self.layouts))
        device = session.torch_device
        self.state = TactileBatchState(
            torch.zeros(
                (*shape_prefix, self.height, self.width, 3),
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(
                (*shape_prefix, self.height, self.width),
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(
                (*shape_prefix, self.height, self.width, 3),
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(
                (*shape_prefix, self.marker_count, 2),
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(
                (*shape_prefix, self.marker_count, 2),
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(
                (*shape_prefix, self.vertex_count, 3),
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros((*shape_prefix, 6), dtype=torch.float32, device=device),
        )

    def _prepare_sampling(self) -> None:
        pixel_indices: list[np.ndarray] = []
        pixel_weights: list[np.ndarray] = []
        rest_pixels: list[np.ndarray] = []
        marker_indices: list[np.ndarray] = []
        marker_weights: list[np.ndarray] = []
        rest_markers: list[np.ndarray] = []
        marker_pixels: list[np.ndarray] = []
        static_pose: list[tuple[float, ...]] = []
        attachment_pose: list[tuple[float, ...]] = []
        attachment_body: list[int] = []
        near_plane: list[float] = []
        far_plane: list[float] = []
        pixel_size: list[float] = []
        for layout in self.layouts:
            vertices = layout.mesh.vertices
            coat = np.asarray(layout.entry["coat_vertex_indices"], dtype=np.int32)
            coat_xy = vertices[coat, :2]
            x = np.linspace(float(coat_xy[:, 0].min()), float(coat_xy[:, 0].max()), self.width)
            y = np.linspace(float(coat_xy[:, 1].min()), float(coat_xy[:, 1].max()), self.height)
            grid_x, grid_y = np.meshgrid(x, y)
            samples = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
            distance_squared = np.sum(
                (samples[:, None, :] - coat_xy[None, :, :]) ** 2, axis=-1
            )
            nearest = np.argpartition(distance_squared, kth=3, axis=1)[:, :4]
            selected_distance = np.take_along_axis(distance_squared, nearest, axis=1)
            weights = 1.0 / np.maximum(selected_distance, 1.0e-16)
            weights /= weights.sum(axis=1, keepdims=True)
            vertex_indices = coat[nearest]
            pixel_indices.append(vertex_indices)
            pixel_weights.append(weights.astype(np.float32))
            rest_pixels.append(
                np.sum(vertices[vertex_indices] * weights[..., None], axis=1).astype(np.float32)
            )

            marker_tetrahedra = np.asarray(
                layout.entry["marker_tetrahedra"], dtype=np.int32
            )
            indices = layout.mesh.tetrahedra[marker_tetrahedra]
            weights_marker = np.asarray(
                layout.entry["marker_barycentric"], dtype=np.float32
            )
            marker_indices.append(indices)
            marker_weights.append(weights_marker)
            rest_markers.append(
                np.sum(vertices[indices] * weights_marker[..., None], axis=1).astype(np.float32)
            )
            marker_pixels.append(
                np.asarray(layout.entry["marker_positions"], dtype=np.float32)
            )
            static_pose.append(_pose_from_matrix(_matrix(layout.entry["transform"])))
            attachment = layout.entry.get("attachment")
            if attachment is None:
                attachment_pose.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
                attachment_body.append(-1)
            else:
                attachment_pose.append(_pose_from_matrix(_matrix(attachment["transform"])))
                link_path = str(attachment["link_path"])
                matches = [
                    robot.body_by_path[link_path]
                    for robot in self.session.robot_layouts
                    if link_path in robot.body_by_path
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"tactile attachment {link_path!r} does not resolve uniquely"
                    )
                attachment_body.append(matches[0])
            near_plane.append(float(layout.entry["near_plane"]))
            far_plane.append(float(layout.entry["far_plane"]))
            pixel_size.append(float(layout.entry["pixel_size"]))

        device = self.session.torch_device
        self.pixel_indices = torch.as_tensor(
            np.stack(pixel_indices), dtype=torch.long, device=device
        )
        self.pixel_weights = torch.as_tensor(
            np.stack(pixel_weights), dtype=torch.float32, device=device
        )
        self.rest_pixels = torch.as_tensor(
            np.stack(rest_pixels), dtype=torch.float32, device=device
        )
        self.marker_indices = torch.as_tensor(
            np.stack(marker_indices), dtype=torch.long, device=device
        )
        self.marker_weights = torch.as_tensor(
            np.stack(marker_weights), dtype=torch.float32, device=device
        )
        self.rest_markers = torch.as_tensor(
            np.stack(rest_markers), dtype=torch.float32, device=device
        )
        self.marker_pixels = torch.as_tensor(
            np.stack(marker_pixels), dtype=torch.float32, device=device
        )
        self.static_pose = torch.as_tensor(
            static_pose, dtype=torch.float32, device=device
        )
        self.attachment_pose = torch.as_tensor(
            attachment_pose, dtype=torch.float32, device=device
        )
        self.attachment_body = tuple(attachment_body)
        self.near_plane = torch.as_tensor(
            near_plane, dtype=torch.float32, device=device
        )
        self.far_plane = torch.as_tensor(
            far_plane, dtype=torch.float32, device=device
        )
        self.pixel_size = torch.as_tensor(
            pixel_size, dtype=torch.float32, device=device
        )

    def _sensor_poses(self) -> torch.Tensor:
        poses = self.static_pose[None].expand(self.session.num_envs, -1, -1).clone()
        for sensor, body in enumerate(self.attachment_body):
            if body < 0:
                continue
            body_pose = self.session.body_pose_tensor[:, body]
            attachment = self.attachment_pose[sensor].expand(self.session.num_envs, -1)
            position = body_pose[:, :3] + _quat_apply(
                body_pose[:, 3:7], attachment[:, :3]
            )
            quaternion = _quat_multiply(
                body_pose[:, 3:7], attachment[:, 3:7]
            )
            poses[:, sensor] = torch.cat((position, quaternion), dim=-1)
        return poses

    def read_state(self, previous: TactileBatchState | None) -> TactileBatchState:
        del previous
        return self.state

    def render(self) -> None:
        poses = self._sensor_poses()
        environments = torch.arange(
            self.session.num_envs, dtype=torch.long, device=self.session.torch_device
        )
        for sensor, layout in enumerate(self.layouts):
            source = slice(layout.particle_start, layout.particle_start + layout.vertex_count)
            world = self.session.position_tensor[:, source]
            force_world = self.session.contact_force_tensor[:, source]
            pose = poses[:, sensor]
            quaternion = pose[:, None, 3:7].expand(-1, layout.vertex_count, -1)
            local = _quat_inverse_apply(
                quaternion, world - pose[:, None, :3]
            )
            force_local = _quat_inverse_apply(quaternion, force_world)
            self.state.contact_force[:, sensor].copy_(force_local)
            resultant_force = force_local.sum(dim=1)
            resultant_torque = torch.cross(local, force_local, dim=-1).sum(dim=1)
            self.state.contact_wrench[:, sensor].copy_(
                torch.cat((resultant_force, resultant_torque), dim=-1)
            )

            indices = self.pixel_indices[sensor]
            weights = self.pixel_weights[sensor]
            pixels = (local[:, indices] * weights[None, :, :, None]).sum(dim=2)
            grid = pixels.reshape(self.session.num_envs, self.height, self.width, 3)
            rest_grid = self.rest_pixels[sensor].reshape(self.height, self.width, 3)
            depth = grid[..., 2] - rest_grid[None, ..., 2]
            depth = torch.clamp(
                depth,
                min=float(self.near_plane[sensor]),
                max=float(self.far_plane[sensor]),
            )
            self.state.depth[:, sensor].copy_(depth)

            left = torch.roll(grid, shifts=1, dims=2)
            right = torch.roll(grid, shifts=-1, dims=2)
            top = torch.roll(grid, shifts=1, dims=1)
            bottom = torch.roll(grid, shifts=-1, dims=1)
            left[:, :, 0] = grid[:, :, 0]
            right[:, :, -1] = grid[:, :, -1]
            top[:, 0] = grid[:, 0]
            bottom[:, -1] = grid[:, -1]
            normal = torch.cross(right - left, bottom - top, dim=-1)
            normal = torch.nn.functional.normalize(normal, dim=-1, eps=1.0e-8)
            normal = torch.where(normal[..., 2:3] < 0.0, -normal, normal)
            self.state.normal[:, sensor].copy_(normal)
            depth_scale = depth / max(float(self.far_plane[sensor]), 1.0e-6)
            rgb = torch.stack(
                (
                    0.45 + 0.35 * normal[..., 0] + 0.25 * depth_scale,
                    0.45 + 0.35 * normal[..., 1] + 0.15 * depth_scale,
                    0.55 + 0.25 * normal[..., 2] - 0.20 * depth_scale,
                ),
                dim=-1,
            ).clamp_(0.0, 1.0)
            self.state.rgb[:, sensor].copy_(rgb)

            marker_indices = self.marker_indices[sensor]
            marker_weights = self.marker_weights[sensor]
            marker_local = (
                local[:, marker_indices] * marker_weights[None, :, :, None]
            ).sum(dim=2)
            flow = (
                marker_local[..., :2] - self.rest_markers[sensor][None, :, :2]
            ) / max(float(self.pixel_size[sensor]), 1.0e-8)
            marker_position = self.marker_pixels[sensor][None] + flow
            self.state.marker_flow[:, sensor].copy_(flow)
            self.state.marker_position[:, sensor].copy_(marker_position)
            marker_x = marker_position[..., 0].round().long().clamp_(0, self.width - 1)
            marker_y = marker_position[..., 1].round().long().clamp_(0, self.height - 1)
            for marker in range(self.marker_count):
                self.state.rgb[
                    environments,
                    sensor,
                    marker_y[:, marker],
                    marker_x[:, marker],
                ] = 0.05

    def reset(self, reset_mask: Any, **state_values: Any) -> Mapping[str, Any]:
        if state_values:
            raise TypeError(
                f"unknown Warp IPC tactile reset fields: {sorted(state_values)}"
            )
        mask = self.session._mask_tensor(reset_mask)
        self.session.reset(mask)
        for value in (
            self.state.rgb,
            self.state.depth,
            self.state.normal,
            self.state.marker_position,
            self.state.marker_flow,
            self.state.contact_force,
            self.state.contact_wrench,
        ):
            value[mask] = 0.0
        return self.session.arrays


def create_session(artifact: Any, config: Any, num_envs: int) -> _WarpIpcSession:
    return _WarpIpcSession(artifact, config, num_envs)


__all__: list[str] = []
