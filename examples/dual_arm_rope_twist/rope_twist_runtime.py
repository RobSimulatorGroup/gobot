"""Worker-owned rope physics/control; accepts compiled data and exports numeric snapshots."""
from __future__ import annotations

import math
import numpy as np
# Torch must load its CUDA runtime before the native solver.
import torch
from gobot.ipc import LibuipcBatchConfig, LibuipcConfig
from gobot.sim import RuntimeComponents, SceneStateOutput
from gobot.sim.providers import (CompiledMuJoCoIpcArtifact, MuJoCoIpcProvider,
                                MuJoCoIpcConfig, MuJoCoIpcConvergencePolicy)
from rope_twist_config import (
    ROBOT_NAMES, ROBOT_LINK_NAMES, FIXTURE_BODY_NAMES, TOOL_LINK_NAME,
    JOINT_NAMES, NUM_ENVS, FIXED_DT, _grip_sensor_specs,
    SCENE_FIELDS, BASE_FIELDS, CONTACT_FIELDS,
)
import controllers
from rope_twist_control import RopeControl


class RopeRuntime:
    def __init__(self, provider, artifact, *, drive_mode, wrist_torque_limit, stall_detection_enabled):
        self.provider = provider
        self.controllers_module = controllers
        self.drive_mode = drive_mode
        self.wrist_torque_limit = wrist_torque_limit
        self.grip_position_reference = None
        self.grip_rotation_reference = None
        self.maximum_grip_slip = 0.
        self.maximum_attachment_error = 0.
        layout = controllers.stall_detection_layout(NUM_ENVS)
        self.control = RopeControl(
            provider, artifact, layout, fixed_dt=FIXED_DT,
            drive_torque_limit=wrist_torque_limit,
            feedback_enabled=stall_detection_enabled,
            link_names=ROBOT_LINK_NAMES,
        )
        deformable_entries = tuple(provider.ipc_solver.deformable_bodies)
        self.endpoint_indices = controllers.rope_endpoint_index_sets(
            deformable_entries,
            self.provider.arrays["ipc_positions"].device,
        )
        self.attachment_reference = (
            controllers.rope_endpoints_in_affine_frames(
                self.provider.arrays["ipc_positions"],
                self.endpoint_indices,
                self.provider.arrays["ipc_affine_targets"],
                self.control.fixture_proxy_indices,
            ).clone()
        )
        self.reset((0,))
        self.output = SceneStateOutput(provider,
            robots=tuple(dict(field=f"robot.{name}.link_pose", robot_name=name,
                base_link="fr3_link0", joint_names=JOINT_NAMES, link_names=ROBOT_LINK_NAMES)
                for name in ROBOT_NAMES) + tuple(dict(field=f"robot.{name}.link_pose",
                robot_name=name, base_link=name, joint_names=(), link_names=(name,))
                for name in FIXTURE_BODY_NAMES),
            deformables=artifact.ipc.deformable_bodies, source_bodies=deformable_entries,
            positions_field="ipc_positions", refresh_positions=provider.refresh_state)

    def step(self, fixed_dt: float) -> None:
        self.control.step()

    def reset(self, environments) -> None:
        if tuple(environments) != (0,):
            raise ValueError("rope runtime resets its single environment")
        self.control.reset()
        self.grip_position_reference = None
        self.grip_rotation_reference = None
        self.maximum_grip_slip = 0.0
        self.maximum_attachment_error = 0.0

    def _update_physical_debug_metrics(self) -> None:
        if self.control.controller.tick < self.controllers_module.TWIST_START_TICK:
            return

        local_endpoints = self.controllers_module.rope_endpoints_in_affine_frames(
            self.provider.arrays["ipc_positions"],
            self.endpoint_indices,
            self.provider.arrays["ipc_affine_targets"],
            self.control.fixture_proxy_indices,
        )
        attachment_error = float(
            (local_endpoints - self.attachment_reference)
            .norm(dim=3)
            .amax()
            .item()
        )
        self.maximum_attachment_error = max(
            self.maximum_attachment_error, attachment_error
        )
        grip_position, grip_rotation = (
            self.controllers_module.body_transforms_in_reference_frames(
                self.provider.arrays,
                self.control.fixture_body_ids,
                self.control.tool_body_ids,
            )
        )
        if self.grip_position_reference is None:
            self.grip_position_reference = grip_position.clone()
            self.grip_rotation_reference = grip_rotation.clone()
            return
        slip, _ = self.controllers_module.relative_transform_errors(
            grip_position,
            grip_rotation,
            self.grip_position_reference,
            self.grip_rotation_reference,
        )
        self.maximum_grip_slip = max(
            self.maximum_grip_slip, float(slip.amax().item())
        )


    def apply_commands(self, commands):
        if commands:
            raise ValueError("rope controller has no external command fields")

    def snapshot(self, fields, environments):
        requested = set(fields)
        unknown = requested - set(BASE_FIELDS) - set(CONTACT_FIELDS)
        if unknown:
            raise KeyError(f"unknown rope snapshot fields: {sorted(unknown)}")
        indices = list(environments)
        scene = requested & (set(SCENE_FIELDS) | {"deformable.local_vertices"})
        result = self.output.snapshot(scene, environments)
        # Metrics and contact display use the same completed physical state.
        if "deformable.local_vertices" not in scene and requested & ({"rope.metrics"} | set(CONTACT_FIELDS)):
            self.provider.refresh_state()
        if "rope.metrics" in requested:
            self._update_physical_debug_metrics()
            winding = controllers.rope_winding_turns(self.provider.arrays["ipc_positions"],
                                                    self.provider.ipc_solver.deformable_bodies)
            metrics = torch.stack((
                self.control.controller.peak_actual_relative_rotation / (2 * math.pi),
                winding.abs().mean(dim=1),
                self.control.controller.filtered_wrist_speed[:, 0], self.control.controller.filtered_wrist_speed[:, 1],
                self.control.controller.filtered_wrist_effort[:, 0], self.control.controller.filtered_wrist_effort[:, 1],
                self.control.controller.peak_axial_torque, self.control.controller.stalled.to(winding.dtype),
                self.control.controller.complete.to(winding.dtype),
            ), dim=1)[indices].detach().cpu().numpy()
            result["rope.metrics"] = np.concatenate((metrics, np.tile(
                [self.maximum_grip_slip, self.maximum_attachment_error, self.control.controller.tick],
                (len(indices), 1))), axis=1)
        if "rope.wrenches" in requested:
            local = controllers.fixture_wrenches_in_tool_frames(self.provider.arrays,
                self.control.fixture_body_ids, self.control.tool_body_ids)
            rotations = self.provider.arrays["xmat"][:, list(self.control.tool_body_ids)].reshape(-1, 2, 3, 3)
            world = local.clone()
            world[..., :3] = (rotations @ local[..., :3, None])[..., 0]
            world[..., 3:] = (rotations @ local[..., 3:, None])[..., 0]
            result["rope.wrenches"] = world[indices].detach().cpu().numpy()
        if requested & set(CONTACT_FIELDS):
            self.provider.refresh_deformable_contact_forces()
            self.provider.sense()
            values = {
                "rope.contact_positions": self.provider.arrays["ipc_positions"],
                "rope.contact_forces": self.provider.arrays["ipc_contact_forces"],
            }
            if requested & {"grip.positions", "grip.forces", "grip.found"}:
                sensor = {key: torch.cat(tuple(value[key] for value in self.control.grip_sensors), dim=1)
                          for key in ("pos", "force", "normal", "tangent", "found")}
                n, t, f = sensor["normal"], sensor["tangent"], sensor["force"]
                world = f[..., 0, None] * n + f[..., 1, None] * t + f[..., 2, None] * torch.cross(n, t, dim=-1)
                values.update({"grip.positions": sensor["pos"], "grip.forces": world, "grip.found": sensor["found"]})
            for name in requested & set(CONTACT_FIELDS):
                result[name] = values[name][indices].detach().cpu().numpy()
        return result


def create(artifact, solver_config, quality, coupling_iterations, drive_mode, wrist_torque_limit):
    compiled = CompiledMuJoCoIpcArtifact.from_mapping(artifact)
    options = dict(solver_config)
    options["solver"] = LibuipcConfig(**options["solver"])
    provider = MuJoCoIpcProvider(compiled,
        config=MuJoCoIpcConfig(num_envs=NUM_ENVS, device="cuda:0", environments_per_shard=1,
            coupling_iterations=coupling_iterations, relaxation_mode=quality["relaxation_mode"],
            relaxation_factor=1., capture_mujoco_graphs=True, capture_coupler_graphs=True,
            convergence_policy=MuJoCoIpcConvergencePolicy(enabled=quality["name"] == "accurate")),
        libuipc_config=LibuipcBatchConfig(**options),
        mujoco_options={"nconmax": 512, "njmax": 2048, "contact_sensor_maxmatch": 32,
                        "contact_sensors": _grip_sensor_specs(), "overflow_check_interval": 0})
    try:
        provider.reset(torch.ones(NUM_ENVS, dtype=torch.bool, device=provider.arrays["ctrl"].device))
        runtime = RopeRuntime(provider, compiled, drive_mode=drive_mode, wrist_torque_limit=wrist_torque_limit,
            stall_detection_enabled=drive_mode == controllers.FINITE_TORQUE_DRIVE_MODE)
        return RuntimeComponents(provider, runtime, runtime)
    except Exception:
        provider.close()
        raise
