"""Robot bindings and control shared by headless trials and editor workers."""
from __future__ import annotations

import torch
from controllers import (
    BatchedGravityCompensator, BatchedTwistController, WRIST_INDEX,
    configure_wrist_torque_limit, fixture_wrenches_in_tool_frames,
    gravity_compensation_schedule,
)
from rope_twist_config import ROBOT_NAMES, JOINT_NAMES, TOOL_LINK_NAME, FIXTURE_BODY_NAMES


class RopeControl:
    def __init__(self, provider, artifact, layout, *, fixed_dt, drive_torque_limit, feedback_enabled, link_names=(TOOL_LINK_NAME,)):
        self.provider = provider
        self.drive_torque_limit = drive_torque_limit
        self.robot_views = tuple(
            self.provider.create_robot_view(
                robot_name=robot_name,
                base_link="fr3_link0",
                joint_names=JOINT_NAMES,
                link_names=link_names,
            )
            for robot_name in ROBOT_NAMES
        )
        self.wrist_actuator_ids = tuple(
            self.provider.rigid_solver.resolve_robot_layout(
                robot_name,
                base_link="fr3_link0",
                joint_names=JOINT_NAMES,
            ).actuator_ids[WRIST_INDEX]
            for robot_name in ROBOT_NAMES
        )
        configure_wrist_torque_limit(
            self.provider.rigid_solver,
            self.wrist_actuator_ids,
            self.drive_torque_limit,
        )
        self.mappings = tuple(
            next(
                mapping
                for mapping in artifact.coupled_bodies
                if mapping.robot_name == fixture_name
                and mapping.link_name == fixture_name
            )
            for fixture_name in FIXTURE_BODY_NAMES
        )
        self.tool_body_ids = tuple(
            self.provider.rigid_solver.resolve_object_ids(
                "body", (f"{robot_name}_{TOOL_LINK_NAME}",)
            )[0]
            for robot_name in ROBOT_NAMES
        )
        self.fixture_body_ids = tuple(
            self.provider.rigid_solver.resolve_object_ids(
                "body", (mapping.mujoco_body_name,)
            )[0]
            for mapping in self.mappings
        )
        self.fixture_proxy_indices = tuple(
            mapping.ipc_body_index for mapping in self.mappings
        )
        self.grip_sensors = tuple(
            self.provider.rigid_solver.contact_sensor(f"{side}_fixture_grip")
            for side in ("left", "right")
        )
        command_template = torch.zeros(
            (self.provider.num_envs, 2, len(JOINT_NAMES)),
            dtype=self.provider.arrays["ctrl"].dtype,
            device=self.provider.arrays["ctrl"].device,
        )
        self.controller = BatchedTwistController(
            command_template,
            layout,
            fixed_dt=fixed_dt,
            feedback_enabled=feedback_enabled,
            drive_torque_limit=self.drive_torque_limit,
        )
        self.gravity_compensator = BatchedGravityCompensator(
            gravity_compensation_schedule(
                artifact.mujoco.content, ROBOT_NAMES
            ),
            self.provider.arrays["qfrc_applied"],
        )

    def step(self):
        applied_wrenches = fixture_wrenches_in_tool_frames(
            self.provider.arrays, self.fixture_body_ids, self.tool_body_ids
        )
        robot_states = tuple(view.read_state() for view in self.robot_views)
        joint_positions = torch.stack(
            tuple(state.joint_position for state in robot_states), dim=1
        )
        joint_velocities = torch.stack(
            tuple(state.joint_velocity for state in robot_states), dim=1
        )
        wrist_efforts = self.provider.arrays["actuator_force"][
            :, list(self.wrist_actuator_ids)
        ]
        command = self.controller.step(
            applied_wrenches,
            joint_positions,
            joint_velocities,
            wrist_efforts,
        )
        for robot_index, robot_view in enumerate(self.robot_views):
            robot_view.set_controls(command[:, robot_index])
        self.gravity_compensator.apply(
            self.provider.arrays["qfrc_applied"],
            joint_positions,
            self.controller.gravity_schedule_indices,
        )

    def reset(self):
        command = self.controller.reset().clone()
        for robot_index, robot_view in enumerate(self.robot_views):
            robot_view.set_controls(command[:, robot_index])
        initial_robot_states = tuple(
            view.read_state() for view in self.robot_views
        )
        initial_joint_positions = torch.stack(
            tuple(
                state.joint_position for state in initial_robot_states
            ),
            dim=1,
        )
        self.gravity_compensator.apply(
            self.provider.arrays["qfrc_applied"], initial_joint_positions, 0
        )
