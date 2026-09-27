"""Physical measurements and JSON reports for a rope-twist trial."""
from __future__ import annotations

from typing import Any
from statistics import median
import torch
from gobot.sim.providers import CompiledMuJoCoIpcArtifact
from rope_twist_config import ROBOT_NAMES, FIXTURE_BODY_NAMES
from controllers import (
    TWIST_START_TICK, TWIST_COMPLETE_TICK, absolute_joint_positions,
    body_transforms_in_reference_frames, fixture_wrenches_in_tool_frames,
    maximum_box_vertex_penetration, maximum_shape_deformation,
    relative_transform_errors, rope_endpoint_index_sets, rope_endpoints_in_affine_frames,
    rope_winding_turns,
)

def _range(values: Any) -> list[float]:
    return [float(values.min().item()), float(values.max().item())]


def _matrix_tensor(value: Any, reference: Any) -> Any:
    return torch.as_tensor(
        value["transform"]["matrix_row_major"],
        dtype=reference.dtype,
        device=reference.device,
    ).reshape(4, 4)


def _penetration_box_spec(
    artifact: CompiledMuJoCoIpcArtifact,
    initial_affine_transforms: Any,
) -> dict[str, Any]:
    manifest = artifact.ipc.manifest_data
    links_by_path = {
        str(link["path"]): link
        for robot in manifest["robots"]
        for link in robot["links"]
    }
    dynamic_indices = []
    dynamic_local_transforms = []
    dynamic_sizes = []
    reference = initial_affine_transforms
    for mapping in artifact.coupled_bodies:
        if mapping.mode != "OneWay":
            continue
        link = links_by_path[mapping.ipc_path]
        for shape in link["collision_shapes"]:
            if bool(shape["disabled"]):
                continue
            if str(shape["shape_type"]) != "box":
                raise RuntimeError(
                    "rope penetration metric requires box OneWay proxies"
                )
            shape_transform = _matrix_tensor(shape, reference)
            proxy_transform = reference[0, mapping.ipc_body_index]
            dynamic_indices.append(mapping.ipc_body_index)
            dynamic_local_transforms.append(
                torch.linalg.solve(proxy_transform, shape_transform)
            )
            dynamic_sizes.append(tuple(float(value) for value in shape["size"]))

    static_transforms = []
    static_sizes = []
    for shape in manifest["static_colliders"]:
        if bool(shape["disabled"]):
            continue
        if str(shape["shape_type"]) != "box":
            raise RuntimeError(
                "rope penetration metric requires box static colliders"
            )
        static_transforms.append(_matrix_tensor(shape, reference))
        static_sizes.append(tuple(float(value) for value in shape["size"]))

    def stack_transforms(values: list[Any]) -> Any:
        if values:
            return torch.stack(values)
        return torch.empty(
            (0, 4, 4), dtype=reference.dtype, device=reference.device
        )

    sizes = dynamic_sizes + static_sizes
    return {
        "dynamic_indices": torch.as_tensor(
            dynamic_indices, dtype=torch.long, device=reference.device
        ),
        "dynamic_local_transforms": stack_transforms(
            dynamic_local_transforms
        ),
        "static_transforms": stack_transforms(static_transforms),
        "sizes": torch.as_tensor(
            sizes, dtype=reference.dtype, device=reference.device
        ).reshape(-1, 3),
    }


def _penetration_box_transforms(
    affine_transforms: Any, spec: dict[str, Any]
) -> Any:
    dynamic = affine_transforms.index_select(
        1, spec["dynamic_indices"]
    ) @ spec["dynamic_local_transforms"].unsqueeze(0)
    static = spec["static_transforms"].unsqueeze(0).expand(
        affine_transforms.shape[0], -1, -1, -1
    )
    return torch.cat((dynamic, static), dim=1)


def _coupling_wrench_imbalance_ratio(
    rigid_wrenches: Any,
    ipc_wrenches: Any,
    scales: Any,
) -> Any:
    expected = ipc_wrenches * scales
    difference = torch.linalg.vector_norm(
        rigid_wrenches - expected, dim=(1, 2)
    )
    reference = torch.maximum(
        torch.linalg.vector_norm(rigid_wrenches, dim=(1, 2)),
        torch.linalg.vector_norm(expected, dim=(1, 2)),
    )
    return torch.where(
        reference > 1.0e-12,
        difference / reference.clamp_min(1.0e-12),
        torch.zeros_like(reference),
    )


class RopeMetrics:
    def __init__(self, provider, artifact, control, args, solver_config):
        self.provider = provider
        self.artifact = artifact
        self.control = control
        self.args = args
        self.solver_config = solver_config
        self.positions = self.provider.arrays["ipc_positions"]
        self.initial_positions = self.positions.clone()
        self.penetration_box_spec = _penetration_box_spec(
            self.artifact, self.provider.arrays["ipc_affine_transforms"]
        )
        self.coupling_wrench_scales = torch.as_tensor(
            tuple(
                (
                    *([mapping.force_scale * self.args.coupling_feedback_scale] * 3),
                    *([mapping.torque_scale * self.args.coupling_feedback_scale] * 3),
                )
                for mapping in self.control.mappings
            ),
            dtype=self.positions.dtype,
            device=self.positions.device,
        ).unsqueeze(0)
        self.position_pointer = self.positions.data_ptr()
        self.qpos_pointer = self.provider.arrays["qpos"].data_ptr()
        self.endpoint_indices = rope_endpoint_index_sets(
            self.provider.ipc_solver.deformable_bodies, self.positions.device
        )
        self.capture_attachment_reference()
        self.peak_raw_torque = torch.zeros(
            self.args.num_envs, dtype=self.positions.dtype, device=self.positions.device
        )
        self.peak_fixture_force = torch.zeros_like(self.peak_raw_torque)
        self.peak_fixture_forces = torch.zeros(
            (self.args.num_envs, 2),
            dtype=self.positions.dtype,
            device=self.positions.device,
        )
        self.peak_contact_force = torch.zeros_like(self.peak_raw_torque)
        self.peak_finger_contact_force = torch.zeros_like(self.peak_raw_torque)
        self.grip_preload_force = torch.zeros_like(self.peak_raw_torque)
        self.grip_preload_finger_forces = torch.zeros(
            (self.args.num_envs, 4),
            dtype=self.positions.dtype,
            device=self.positions.device,
        )
        self.preload_force_accumulator = torch.zeros_like(
            self.grip_preload_finger_forces
        )
        self.preload_force_samples = 0
        self.peak_grip_slip = torch.zeros_like(self.peak_raw_torque)
        self.peak_grip_slip_tick = torch.full(
            (self.args.num_envs,),
            -1,
            dtype=torch.int32,
            device=self.positions.device,
        )
        self.peak_grip_rotation_slip = torch.zeros_like(self.peak_raw_torque)
        self.peak_grip_slip_per_fixture = torch.zeros(
            (self.args.num_envs, 2),
            dtype=self.positions.dtype,
            device=self.positions.device,
        )
        self.peak_grip_rotation_slip_per_fixture = torch.zeros_like(
            self.peak_grip_slip_per_fixture
        )
        self.peak_grip_axis_slip = torch.zeros(
            (self.args.num_envs, 2, 3),
            dtype=self.positions.dtype,
            device=self.positions.device,
        )
        self.peak_attachment_error = torch.zeros_like(self.peak_raw_torque)
        self.peak_rope_vertex_penetration = torch.zeros_like(self.peak_raw_torque)
        self.peak_coupling_wrench_imbalance_ratio = torch.zeros_like(
            self.peak_raw_torque
        )
        self.minimum_grip_contact_distance = torch.full_like(
            self.peak_raw_torque, torch.inf
        )
        self.final_grip_contact_distance = torch.zeros_like(self.peak_raw_torque)
        self.grip_contact_seen = torch.zeros(
            self.args.num_envs,
            dtype=torch.bool,
            device=self.positions.device,
        )
        self.peak_deformation = torch.zeros_like(self.peak_raw_torque)
        self.grip_position_reference = None
        self.grip_rotation_reference = None
        self.evaluation_winding = rope_winding_turns(
            self.positions, self.provider.ipc_solver.deformable_bodies
        ).clone()
        self.evaluation_joint_position = None
        self.evaluation_joint_velocity = None
        self.evaluation_wrist_effort = None

    def capture_attachment_reference(self):
        self.attachment_reference = rope_endpoints_in_affine_frames(
            self.positions,
            self.endpoint_indices,
            self.provider.arrays["ipc_affine_targets"],
            self.control.fixture_proxy_indices,
        ).clone()

    def update(self):
        self._update_contacts()
        self._update_coupling()
        self._update_grasp()
        if self.control.controller.tick == min(self.args.steps, TWIST_COMPLETE_TICK):
            self.capture_evaluation()

    def _update_contacts(self):
        applied_wrenches = fixture_wrenches_in_tool_frames(
            self.provider.arrays, self.control.fixture_body_ids, self.control.tool_body_ids
        )
        raw_torque = applied_wrenches[..., 5].abs().amax(dim=1)
        torch.maximum(self.peak_raw_torque, raw_torque, out=self.peak_raw_torque)
        fixture_force = applied_wrenches[..., :3].norm(dim=2)
        torch.maximum(
            self.peak_fixture_forces,
            fixture_force,
            out=self.peak_fixture_forces,
        )
        torch.maximum(
            self.peak_fixture_force,
            fixture_force.amax(dim=1),
            out=self.peak_fixture_force,
        )
        finger_contact_force = torch.cat(
            tuple(sensor["force"].norm(dim=2) for sensor in self.control.grip_sensors),
            dim=1,
        )
        torch.maximum(
            self.peak_finger_contact_force,
            finger_contact_force.amax(dim=1),
            out=self.peak_finger_contact_force,
        )
        if TWIST_START_TICK - 20 <= self.control.controller.tick <= TWIST_START_TICK:
            self.preload_force_accumulator.add_(finger_contact_force)
            self.preload_force_samples += 1
        found = torch.cat(
            tuple(sensor["found"] > 0.5 for sensor in self.control.grip_sensors),
            dim=1,
        )
        distances = torch.cat(
            tuple(sensor["dist"] for sensor in self.control.grip_sensors), dim=1
        )
        seen_now = found.any(dim=1)
        self.grip_contact_seen.logical_or_(seen_now)
        observed_minimum = torch.where(
            found,
            distances,
            torch.full_like(distances, torch.inf),
        ).amin(dim=1)
        torch.minimum(
            self.minimum_grip_contact_distance,
            observed_minimum,
            out=self.minimum_grip_contact_distance,
        )
        self.final_grip_contact_distance.copy_(
            torch.where(
                seen_now,
                observed_minimum,
                torch.zeros_like(observed_minimum),
            )
        )
        if self.solver_config.export_deformable_contact_forces:
            contact_force = self.provider.arrays["ipc_contact_forces"].norm(
                dim=2
            ).amax(dim=1)
            torch.maximum(
                self.peak_contact_force,
                contact_force,
                out=self.peak_contact_force,
            )

    def _update_coupling(self):
        box_transforms = _penetration_box_transforms(
            self.provider.arrays["ipc_affine_transforms"],
            self.penetration_box_spec,
        )
        rope_vertex_penetration = maximum_box_vertex_penetration(
            self.positions,
            box_transforms,
            self.penetration_box_spec["sizes"],
        )
        torch.maximum(
            self.peak_rope_vertex_penetration,
            rope_vertex_penetration,
            out=self.peak_rope_vertex_penetration,
        )
        coupling_wrench_imbalance_ratio = (
            _coupling_wrench_imbalance_ratio(
                self.provider.arrays["xfrc_applied"][
                    :, list(self.control.fixture_body_ids)
                ],
                self.provider.arrays["ipc_affine_contact_wrenches"][
                    :, list(self.control.fixture_proxy_indices)
                ],
                self.coupling_wrench_scales,
            )
        )
        torch.maximum(
            self.peak_coupling_wrench_imbalance_ratio,
            coupling_wrench_imbalance_ratio,
            out=self.peak_coupling_wrench_imbalance_ratio,
        )
        local_endpoints = rope_endpoints_in_affine_frames(
            self.positions,
            self.endpoint_indices,
            self.provider.arrays["ipc_affine_targets"],
            self.control.fixture_proxy_indices,
        )
        attachment_error = (
            local_endpoints - self.attachment_reference
        ).norm(dim=3).amax(dim=(1, 2))
        torch.maximum(
            self.peak_attachment_error,
            attachment_error,
            out=self.peak_attachment_error,
        )

    def _update_grasp(self):
        grip_position, grip_rotation = body_transforms_in_reference_frames(
            self.provider.arrays, self.control.fixture_body_ids, self.control.tool_body_ids
        )
        if self.control.controller.tick == TWIST_START_TICK:
            self.grip_position_reference = grip_position.clone()
            self.grip_rotation_reference = grip_rotation.clone()
            self.grip_preload_finger_forces.copy_(
                self.preload_force_accumulator / self.preload_force_samples
            )
            self.grip_preload_force.copy_(
                self.grip_preload_finger_forces.amin(dim=1)
            )
        elif self.grip_position_reference is not None:
            grip_slip, grip_rotation_slip = relative_transform_errors(
                grip_position,
                grip_rotation,
                self.grip_position_reference,
                self.grip_rotation_reference,
            )
            grip_slip_maximum = grip_slip.amax(dim=1)
            self.peak_grip_slip_tick.masked_fill_(
                grip_slip_maximum > self.peak_grip_slip,
                self.control.controller.tick,
            )
            torch.maximum(
                self.peak_grip_slip,
                grip_slip_maximum,
                out=self.peak_grip_slip,
            )
            torch.maximum(
                self.peak_grip_slip_per_fixture,
                grip_slip,
                out=self.peak_grip_slip_per_fixture,
            )
            torch.maximum(
                self.peak_grip_rotation_slip,
                grip_rotation_slip.amax(dim=1),
                out=self.peak_grip_rotation_slip,
            )
            torch.maximum(
                self.peak_grip_rotation_slip_per_fixture,
                grip_rotation_slip,
                out=self.peak_grip_rotation_slip_per_fixture,
            )
            axis_slip = (grip_position - self.grip_position_reference).abs()
            torch.maximum(
                self.peak_grip_axis_slip,
                axis_slip,
                out=self.peak_grip_axis_slip,
            )
        deformation = maximum_shape_deformation(
            self.initial_positions, self.positions
        )
        torch.maximum(
            self.peak_deformation, deformation, out=self.peak_deformation
        )

    def capture_evaluation(self):
        self.evaluation_joint_position = torch.stack(
            tuple(
                robot_view.read_state().joint_position
                for robot_view in self.control.robot_views
            ),
            dim=1,
        )
        self.evaluation_winding = rope_winding_turns(
            self.positions, self.provider.ipc_solver.deformable_bodies
        ).clone()
        self.evaluation_joint_velocity = torch.stack(
            tuple(
                robot_view.read_state().joint_velocity
                for robot_view in self.control.robot_views
            ),
            dim=1,
        )
        self.evaluation_wrist_effort = self.provider.arrays["actuator_force"][
            :, list(self.control.wrist_actuator_ids)
        ].clone()

    def _validate(self):
        self.provider.rigid_solver.assert_no_overflow()
        for name in (
            "qpos",
            "qfrc_applied",
            "xfrc_applied",
            "ipc_positions",
            "ipc_affine_contact_wrenches",
        ):
            if not bool(torch.isfinite(self.provider.arrays[name]).all().item()):
                raise RuntimeError(f"non-finite values in {name}")

    def _trial_reports(self):
        trials = []
        for environment in range(self.args.num_envs):
            trials.append(
                {
                    "environment": environment,
                    "controller": self.args.drive_mode + "_velocity_drive",
                    "joint7_positions_radians": [
                        float(value)
                        for value in self.absolute_positions[environment, :, 6].tolist()
                    ],
                    "finger_positions_meters": [
                        [float(value) for value in robot]
                        for robot in self.absolute_positions[
                            environment, :, 7:9
                        ].tolist()
                    ],
                    "joint7_velocities_radians_per_second": [
                        float(value)
                        for value in self.evaluation_joint_velocity[
                            environment, :, 6
                        ].tolist()
                    ],
                    "wrist_actuator_efforts_newton_meters": [
                        float(value)
                        for value in self.evaluation_wrist_effort[
                            environment
                        ].tolist()
                    ],
                    "peak_commanded_speed_radians_per_second": float(
                        self.control.controller.peak_commanded_speed[environment].item()
                    ),
                    "peak_actual_relative_rotation_radians": float(
                        self.control.controller.peak_actual_relative_rotation[
                            environment
                        ].item()
                    ),
                    "peak_actual_relative_turns": float(
                        self.control.controller.peak_actual_relative_rotation[
                            environment
                        ].item()
                        / (2.0 * 3.141592653589793)
                    ),
                    "stalled": bool(self.control.controller.stalled[environment].item()),
                    "stall_tick": int(
                        self.control.controller.stall_tick[environment].item()
                    ),
                    "stalled_relative_turns": float(
                        self.control.controller.stalled_relative_rotation[
                            environment
                        ].item()
                        / (2.0 * 3.141592653589793)
                    ),
                    "stall_wrist_speeds_radians_per_second": [
                        float(value)
                        for value in self.control.controller.stalled_wrist_speed[
                            environment
                        ].tolist()
                    ],
                    "stall_wrist_efforts_newton_meters": [
                        float(value)
                        for value in self.control.controller.stalled_wrist_effort[
                            environment
                        ].tolist()
                    ],
                    "stall_axial_torques_newton_meters": [
                        float(value)
                        for value in self.control.controller.stalled_axial_torque[
                            environment
                        ].tolist()
                    ],
                    "safety_stopped": bool(
                        self.control.controller.safety_stopped[environment].item()
                    ),
                    "peak_axial_torque_newton_meters": float(
                        self.control.controller.peak_axial_torque[environment].item()
                    ),
                    "raw_peak_axial_torque_newton_meters": float(
                        self.peak_raw_torque[environment].item()
                    ),
                    "peak_fixture_force_newtons": float(
                        self.peak_fixture_force[environment].item()
                    ),
                    "peak_fixture_forces_newtons": [
                        float(value)
                        for value in self.peak_fixture_forces[environment].tolist()
                    ],
                    "peak_contact_force_newtons": float(
                        self.peak_contact_force[environment].item()
                    ),
                    "grip_preload_minimum_finger_force_newtons": float(
                        self.grip_preload_force[environment].item()
                    ),
                    "grip_preload_finger_forces_newtons": [
                        float(value)
                        for value in self.grip_preload_finger_forces[
                            environment
                        ].tolist()
                    ],
                    "peak_finger_contact_force_newtons": float(
                        self.peak_finger_contact_force[environment].item()
                    ),
                    "maximum_grip_slip_meters": float(
                        self.peak_grip_slip[environment].item()
                    ),
                    "maximum_grip_slip_tick": int(
                        self.peak_grip_slip_tick[environment].item()
                    ),
                    "maximum_grip_slip_per_fixture_meters": [
                        float(value)
                        for value in self.peak_grip_slip_per_fixture[
                            environment
                        ].tolist()
                    ],
                    "maximum_grip_axis_slip_per_fixture_meters": [
                        [float(value) for value in fixture]
                        for fixture in self.peak_grip_axis_slip[
                            environment
                        ].tolist()
                    ],
                    "maximum_grip_rotation_slip_radians": float(
                        self.peak_grip_rotation_slip[environment].item()
                    ),
                    "maximum_grip_rotation_slip_per_fixture_radians": [
                        float(value)
                        for value in self.peak_grip_rotation_slip_per_fixture[
                            environment
                        ].tolist()
                    ],
                    "maximum_attachment_error_meters": float(
                        self.peak_attachment_error[environment].item()
                    ),
                    "maximum_rope_vertex_penetration_meters": float(
                        self.peak_rope_vertex_penetration[environment].item()
                    ),
                    "maximum_coupling_wrench_imbalance_ratio": float(
                        self.peak_coupling_wrench_imbalance_ratio[
                            environment
                        ].item()
                    ),
                    "minimum_grip_contact_distance_meters": float(
                        self.reported_minimum_grip_distance[environment].item()
                    ),
                    "final_grip_contact_distance_meters": float(
                        self.final_grip_contact_distance[environment].item()
                    ),
                    "grip_contact_seen": bool(
                        self.grip_contact_seen[environment].item()
                    ),
                    "strand_winding_turns": [
                        float(value)
                        for value in self.evaluation_winding[environment].tolist()
                    ],
                    "maximum_shape_deformation_meters": float(
                        self.peak_deformation[environment].item()
                    ),
                }
            )
        return trials

    def report(self, scene_path, timing):
        if self.evaluation_joint_position is None:
            self.capture_evaluation()
        self._validate()
        assert self.evaluation_joint_velocity is not None
        assert self.evaluation_wrist_effort is not None
        self.absolute_positions = absolute_joint_positions(
            self.evaluation_joint_position
        )
        self.reported_minimum_grip_distance = torch.where(
            self.grip_contact_seen,
            self.minimum_grip_contact_distance,
            torch.zeros_like(self.minimum_grip_contact_distance),
        )

        diagnostics = self.provider.diagnostics
        return {
            "artifact": self.artifact.digest,
            "scene": str(scene_path),
            "task": "dual_fr3_three_strand_rope_twist",
            "robots": list(ROBOT_NAMES),
            "fixtures": list(FIXTURE_BODY_NAMES),
            "robot_count": 2,
            "robot_arm_joint_count": 14,
            "robot_gripper_joint_count": 4,
            "deformable_strand_count": 3,
            "attachment_count": len(
                self.artifact.ipc.manifest_data["deformable_attachments"]
            ),
            "fixture_coupling_count": len(self.control.mappings),
            "one_way_proxy_count": sum(
                mapping.mode == "OneWay" for mapping in self.artifact.coupled_bodies
            ),
            "proxy_count": diagnostics["proxy_count"],
            "static_collider_count": diagnostics["static_collider_count"],
            "grip_mode": "mujoco_fixture_friction_contact",
            "drive_mode": self.args.drive_mode,
            "grip_friction_scale": self.args.grip_friction_scale,
            "device": self.args.device,
            "steps": timing.executed_steps,
            "requested_steps": self.args.steps,
            "warmup_steps": self.args.warmup_steps,
            "admission_completed": (
                not timing.admission_aborted and timing.executed_steps == self.args.steps
            ),
            "admission_aborted": timing.admission_aborted,
            "admission_abort_reason": timing.admission_abort_reason,
            "maximum_step_latency_seconds": timing.maximum_step_latency_seconds,
            "median_physics_step_latency_seconds": (
                median(timing.step_latency_samples_seconds)
                if timing.step_latency_samples_seconds
                else 0.0
            ),
            "step_latency_limit_seconds": (
                self.args.maximum_step_latency_seconds
            ),
            "environments": self.args.num_envs,
            "environments_per_shard": self.args.environments_per_shard,
            "shards": self.provider.ipc_solver.shard_count,
            "elapsed_seconds": timing.elapsed,
            "environment_steps_per_second": (
                self.args.num_envs * timing.executed_steps / timing.elapsed
            ),
            "coupling_feedback_scale": self.args.coupling_feedback_scale,
            "coupling_solver": diagnostics["coupling_solver"],
            "coupling_iterations": diagnostics["coupling_iterations"],
            "actual_coupling_iterations": diagnostics[
                "actual_coupling_iterations"
            ],
            "relaxation_mode": diagnostics["relaxation_mode"],
            "interface_residual": diagnostics["interface_residual"],
            "interface_residual_l2": diagnostics["interface_residual_l2"],
            "aitken_coefficient": diagnostics["aitken_coefficient"],
            "coupler_graph_captured": diagnostics[
                "coupler_graph_captured"
            ],
            "coupler_graph_capture_reason": diagnostics[
                "coupler_graph_capture_reason"
            ],
            "phase_latency_ms": dict(diagnostics["phase_latency_ms"]),
            "newton_max_iterations": self.solver_config.newton_max_iterations,
            "line_search_max_iterations": (
                self.solver_config.line_search_max_iterations
            ),
            "linear_system_tolerance_rate": (
                self.solver_config.linear_system_tolerance_rate
            ),
            "strict_convergence": self.solver_config.strict_convergence,
            "convergence_guard": dict(diagnostics["convergence_guard"]),
            "deformable_contact_forces_exported": (
                self.solver_config.export_deformable_contact_forces
            ),
            "feedback_source": diagnostics["feedback_source"],
            "exact_contact_wrench": self.provider.capabilities.exact_contact_wrench,
            "collision_ownership": dict(self.artifact.collision_ownership),
            "wrist_drive_torque_limit_newton_meters": self.control.drive_torque_limit,
            "stalled_count": int(self.control.controller.stalled.count_nonzero().item()),
            "safety_stopped_count": int(
                self.control.controller.safety_stopped.count_nonzero().item()
            ),
            "peak_actual_relative_rotation_range_radians": _range(
                self.control.controller.peak_actual_relative_rotation
            ),
            "stalled_relative_rotation_range_radians": _range(
                self.control.controller.stalled_relative_rotation
            ),
            "stall_wrist_speed_range_radians_per_second": _range(
                self.control.controller.stalled_wrist_speed
            ),
            "stall_wrist_effort_range_newton_meters": _range(
                self.control.controller.stalled_wrist_effort
            ),
            "stall_axial_torque_range_newton_meters": _range(
                self.control.controller.stalled_axial_torque
            ),
            "raw_peak_axial_torque_range_newton_meters": _range(
                self.peak_raw_torque
            ),
            "peak_fixture_force_range_newtons": _range(
                self.peak_fixture_force
            ),
            "peak_contact_force_range_newtons": _range(
                self.peak_contact_force
            ),
            "grip_preload_minimum_finger_force_range_newtons": _range(
                self.grip_preload_force
            ),
            "peak_finger_contact_force_range_newtons": _range(
                self.peak_finger_contact_force
            ),
            "maximum_grip_slip_range_meters": _range(
                self.peak_grip_slip
            ),
            "maximum_grip_slip_tick_range": [
                int(self.peak_grip_slip_tick.min().item()),
                int(self.peak_grip_slip_tick.max().item()),
            ],
            "maximum_grip_rotation_slip_range_radians": _range(
                self.peak_grip_rotation_slip
            ),
            "maximum_attachment_error_range_meters": _range(
                self.peak_attachment_error
            ),
            "maximum_rope_vertex_penetration_range_meters": _range(
                self.peak_rope_vertex_penetration
            ),
            "maximum_coupling_wrench_imbalance_ratio_range": _range(
                self.peak_coupling_wrench_imbalance_ratio
            ),
            "minimum_grip_contact_distance_range_meters": _range(
                self.reported_minimum_grip_distance
            ),
            "final_grip_contact_distance_range_meters": _range(
                self.final_grip_contact_distance
            ),
            "maximum_shape_deformation_range_meters": _range(
                self.peak_deformation
            ),
            "trials": self._trial_reports(),
            "ipc_position_storage_stable": (
                self.provider.arrays["ipc_positions"].data_ptr() == self.position_pointer
            ),
            "qpos_storage_stable": (
                self.provider.arrays["qpos"].data_ptr() == self.qpos_pointer
            ),
        }
