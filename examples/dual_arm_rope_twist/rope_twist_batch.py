"""Run batched dual-FR3 rope twisting with MuJoCo Warp and libuipc."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from typing import Any

# Torch initializes CUDA before the native libuipc module is loaded.
import torch

import gobot
from gobot.ipc import LibuipcBatchConfig, LibuipcBatchSolver, LibuipcConfig
from gobot.rl import (
    CompiledMuJoCoIpcArtifact,
    MuJoCoIpcConfig,
    MuJoCoIpcProvider,
    MuJoCoWarpContactSensorSpec,
    MuJoCoWarpProvider,
)

from build_scene import (
    FIXTURE_LINK_NAME,
    FIXTURE_ROBOT_NAMES,
    HERE,
    JOINT_NAMES,
    ROBOT_NAMES,
    SCENE_NAME,
    TOOL_LINK_NAME,
    build_scene,
)
from controllers import (
    BatchedGravityCompensator,
    BatchedTwistController,
    CYCLE_TICKS,
    FINITE_TORQUE_DRIVE_MODE,
    SHOWCASE_DRIVE_MODE,
    TWIST_COMPLETE_TICK,
    TWIST_START_TICK,
    TwistTrialLayout,
    WRIST_INDEX,
    absolute_joint_positions,
    body_transforms_in_reference_frames,
    configure_wrist_torque_limit,
    fixture_wrenches_in_tool_frames,
    gravity_compensation_schedule,
    make_trial_layout,
    maximum_shape_deformation,
    relative_transform_errors,
    rope_endpoint_index_sets,
    rope_endpoints_in_body_frames,
    rope_winding_turns,
    wrist_drive_torque_limit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run dual-FR3 rope twisting as a finite-torque experiment or "
            "constant-speed showcase."
        )
    )
    parser.add_argument("--scene", type=Path, default=HERE / SCENE_NAME)
    parser.add_argument("--rebuild-scene", action="store_true")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--environments-per-shard", type=int, default=1)
    parser.add_argument("--steps", type=int, default=CYCLE_TICKS)
    parser.add_argument("--fixed-dt", type=float, default=0.002)
    parser.add_argument("--rigid-substeps", type=int, default=1)
    parser.add_argument("--ipc-substeps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--drive-mode",
        choices=(FINITE_TORQUE_DRIVE_MODE, SHOWCASE_DRIVE_MODE),
        default=FINITE_TORQUE_DRIVE_MODE,
        help="finite physical stall experiment or strong constant-speed showcase",
    )
    parser.add_argument("--module-path", default="")
    parser.add_argument("--friction", type=float, default=1.25)
    parser.add_argument("--contact-resistance", type=float, default=1.0e7)
    parser.add_argument("--coupling-feedback-scale", type=float, default=1.0)
    parser.add_argument(
        "--grip-friction-scale",
        type=float,
        default=1.0,
        help="scale MuJoCo pad/fixture friction for grasp ablations",
    )
    parser.add_argument("--no-mujoco-graph", action="store_true")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(tempfile.gettempdir()) / "gobot-dual-arm-rope-twist",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.environments_per_shard <= 0:
        raise ValueError("--environments-per-shard must be positive")
    if args.num_envs % args.environments_per_shard:
        raise ValueError(
            "--num-envs must be divisible by --environments-per-shard"
        )
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.fixed_dt <= 0.0:
        raise ValueError("--fixed-dt must be positive")
    if args.rigid_substeps <= 0 or args.ipc_substeps <= 0:
        raise ValueError("solver substep counts must be positive")
    if args.friction < 0.0:
        raise ValueError("--friction must be non-negative")
    if args.contact_resistance <= 0.0:
        raise ValueError("--contact-resistance must be positive")
    if args.coupling_feedback_scale < 0.0:
        raise ValueError("--coupling-feedback-scale must be non-negative")
    if args.grip_friction_scale < 0.0:
        raise ValueError("--grip-friction-scale must be non-negative")


def _load_artifact(
    scene_path: Path,
) -> tuple[Any, CompiledMuJoCoIpcArtifact]:
    context = gobot.app.create_context()
    context.set_project_path(str(scene_path.parent))
    context.load_scene("res://" + scene_path.name)
    settings = context.get_mujoco_solver_settings()
    settings["integrator"] = gobot.PhysicsIntegratorType.ImplicitFast
    settings["cone"] = gobot.PhysicsFrictionConeType.Elliptic
    settings["impedance_ratio"] = 10.0
    context.set_mujoco_solver_settings(settings)
    return context, CompiledMuJoCoIpcArtifact.from_context(context)


def _range(values: Any) -> list[float]:
    return [float(values.min().item()), float(values.max().item())]


def _pad_geom_names(robot_name: str) -> tuple[str, str]:
    return tuple(
        f"{robot_name}_{robot_name}_{side}_rubber_pad_collision"
        for side in ("left", "right")
    )


def _fixture_geom_name(fixture_name: str) -> str:
    return f"{fixture_name}_fixture_body_collision"


def _grip_sensor_specs() -> tuple[MuJoCoWarpContactSensorSpec, ...]:
    return tuple(
        MuJoCoWarpContactSensorSpec(
            name=f"{side}_fixture_grip",
            primary_type="geom",
            primary_names=_pad_geom_names(robot_name),
            secondary_type="geom",
            secondary_name=_fixture_geom_name(fixture_name),
            fields=("found", "force", "dist"),
            reduce="maxforce",
            num_slots=1,
        )
        for side, robot_name, fixture_name in zip(
            ("left", "right"),
            ROBOT_NAMES,
            FIXTURE_ROBOT_NAMES,
            strict=True,
        )
    )


def run(
    args: argparse.Namespace,
    *,
    layout: TwistTrialLayout | None = None,
) -> dict[str, Any]:
    """Run one fixed-capacity batch and return JSON-compatible metrics."""

    _validate_args(args)
    scene_path = args.scene.expanduser().resolve()
    if args.rebuild_scene:
        scene_path = build_scene(scene_path.parent)
    elif not scene_path.is_file():
        raise FileNotFoundError(
            f"scene does not exist: {scene_path}; run build_scene.py first"
        )
    if layout is None:
        layout = make_trial_layout(args.num_envs, seed=args.seed)
    elif len(layout.stall_detection) != args.num_envs:
        raise ValueError("the supplied trial layout does not match --num-envs")

    solver_config = LibuipcBatchConfig(
        solver=LibuipcConfig(
            fixed_time_step=(
                args.fixed_dt * args.rigid_substeps / args.ipc_substeps
            ),
            gravity=(0.0, 0.0, -9.81),
            friction_coefficient=args.friction,
            contact_activation_distance=8.0e-4,
            contact_resistance=args.contact_resistance,
            affine_stiffness=1.0e8,
            module_path=args.module_path,
            workspace=str(args.workspace.expanduser().resolve()),
        ),
        environments_per_shard=args.environments_per_shard,
    )
    rigid_availability = MuJoCoWarpProvider.availability()
    if not rigid_availability.available:
        raise RuntimeError(rigid_availability.reason)
    ipc_availability = LibuipcBatchSolver.availability(solver_config)
    if not ipc_availability.available:
        raise RuntimeError(ipc_availability.reason)

    context, artifact = _load_artifact(scene_path)
    provider = MuJoCoIpcProvider(
        artifact,
        config=MuJoCoIpcConfig(
            num_envs=args.num_envs,
            device=args.device,
            environments_per_shard=args.environments_per_shard,
            force_scale=args.coupling_feedback_scale,
            torque_scale=args.coupling_feedback_scale,
            rigid_substeps=args.rigid_substeps,
            ipc_substeps=args.ipc_substeps,
            capture_mujoco_graphs=not args.no_mujoco_graph,
        ),
        libuipc_config=solver_config,
        mujoco_options={
            "nconmax": 512,
            "njmax": 2048,
            "contact_sensor_maxmatch": 32,
            "contact_sensors": _grip_sensor_specs(),
            "overflow_check_interval": 0,
        },
    )
    try:
        robot_views = tuple(
            provider.create_robot_view(
                robot_name=robot_name,
                base_link="fr3_link0",
                joint_names=JOINT_NAMES,
                link_names=(TOOL_LINK_NAME,),
            )
            for robot_name in ROBOT_NAMES
        )
        wrist_actuator_ids = tuple(
            provider.rigid_solver.resolve_robot_layout(
                robot_name,
                base_link="fr3_link0",
                joint_names=JOINT_NAMES,
            ).actuator_ids[WRIST_INDEX]
            for robot_name in ROBOT_NAMES
        )
        drive_torque_limit = wrist_drive_torque_limit(args.drive_mode)
        configure_wrist_torque_limit(
            provider.rigid_solver,
            wrist_actuator_ids,
            drive_torque_limit,
        )
        mappings = tuple(
            next(
                mapping
                for mapping in artifact.coupled_bodies
                if mapping.robot_name == fixture_name
                and mapping.link_name == FIXTURE_LINK_NAME
            )
            for fixture_name in FIXTURE_ROBOT_NAMES
        )
        tool_body_ids = tuple(
            provider.rigid_solver.resolve_object_ids(
                "body", (f"{robot_name}_{TOOL_LINK_NAME}",)
            )[0]
            for robot_name in ROBOT_NAMES
        )
        fixture_body_ids = tuple(
            provider.rigid_solver.resolve_object_ids(
                "body", (mapping.mujoco_body_name,)
            )[0]
            for mapping in mappings
        )
        grip_sensors = tuple(
            provider.rigid_solver.contact_sensor(f"{side}_fixture_grip")
            for side in ("left", "right")
        )

        grip_geom_ids = provider.rigid_solver.resolve_object_ids(
            "geom",
            tuple(
                name
                for pair in (
                    _pad_geom_names(ROBOT_NAMES[0]),
                    (_fixture_geom_name(FIXTURE_ROBOT_NAMES[0]),),
                    _pad_geom_names(ROBOT_NAMES[1]),
                    (_fixture_geom_name(FIXTURE_ROBOT_NAMES[1]),),
                )
                for name in pair
            ),
        )
        if args.grip_friction_scale != 1.0:
            geom_friction = provider.rigid_solver.model_array("geom_friction")
            geom_indices = torch.as_tensor(
                grip_geom_ids,
                dtype=torch.long,
                device=geom_friction.device,
            )
            scaled_friction = geom_friction.index_select(1, geom_indices)
            scaled_friction.mul_(args.grip_friction_scale)
            geom_friction.index_copy_(1, geom_indices, scaled_friction)
            if args.grip_friction_scale == 0.0:
                geom_condim = provider.rigid_solver.model_array("geom_condim")
                condim_indices = geom_indices.to(device=geom_condim.device)
                frictionless_condim = torch.ones_like(
                    geom_condim.index_select(0, condim_indices)
                )
                geom_condim.index_copy_(
                    0, condim_indices, frictionless_condim
                )
            provider.rigid_solver.recompute_constants()

        positions = provider.arrays["ipc_positions"]
        initial_positions = positions.clone()
        initial_qpos = provider.arrays["qpos"].clone()
        position_pointer = positions.data_ptr()
        qpos_pointer = provider.arrays["qpos"].data_ptr()
        command_template = torch.zeros(
            (args.num_envs, 2, len(JOINT_NAMES)),
            dtype=provider.arrays["ctrl"].dtype,
            device=positions.device,
        )
        controller = BatchedTwistController(
            command_template,
            layout,
            fixed_dt=args.fixed_dt,
            feedback_enabled=(
                args.drive_mode == FINITE_TORQUE_DRIVE_MODE
                and args.coupling_feedback_scale > 0.0
            ),
            drive_torque_limit=drive_torque_limit,
        )
        gravity_compensator = BatchedGravityCompensator(
            gravity_compensation_schedule(
                artifact.mujoco.content, ROBOT_NAMES
            ),
            provider.arrays["qfrc_applied"],
        )
        all_envs = torch.ones(
            args.num_envs, dtype=torch.bool, device=positions.device
        )
        command = controller.reset().clone()
        provider.reset(
            all_envs,
            qpos=initial_qpos,
            qvel=torch.zeros_like(provider.arrays["qvel"]),
            ctrl=torch.zeros_like(provider.arrays["ctrl"]),
        )
        for robot_index, robot_view in enumerate(robot_views):
            robot_view.set_controls(command[:, robot_index])
        initial_robot_states = tuple(
            view.read_state() for view in robot_views
        )
        initial_joint_positions = torch.stack(
            tuple(
                state.joint_position for state in initial_robot_states
            ),
            dim=1,
        )
        gravity_compensator.apply(
            provider.arrays["qfrc_applied"], initial_joint_positions, 0
        )
        endpoint_indices = rope_endpoint_index_sets(
            provider.ipc_solver.deformable_bodies, positions.device
        )
        attachment_reference = rope_endpoints_in_body_frames(
            positions,
            endpoint_indices,
            provider.arrays,
            fixture_body_ids,
        ).clone()

        peak_raw_torque = torch.zeros(
            args.num_envs, dtype=positions.dtype, device=positions.device
        )
        peak_fixture_force = torch.zeros_like(peak_raw_torque)
        peak_fixture_forces = torch.zeros(
            (args.num_envs, 2),
            dtype=positions.dtype,
            device=positions.device,
        )
        peak_contact_force = torch.zeros_like(peak_raw_torque)
        peak_finger_contact_force = torch.zeros_like(peak_raw_torque)
        grip_preload_force = torch.zeros_like(peak_raw_torque)
        grip_preload_finger_forces = torch.zeros(
            (args.num_envs, 4),
            dtype=positions.dtype,
            device=positions.device,
        )
        preload_force_accumulator = torch.zeros_like(
            grip_preload_finger_forces
        )
        preload_force_samples = 0
        peak_grip_slip = torch.zeros_like(peak_raw_torque)
        peak_grip_rotation_slip = torch.zeros_like(peak_raw_torque)
        peak_grip_slip_per_fixture = torch.zeros(
            (args.num_envs, 2),
            dtype=positions.dtype,
            device=positions.device,
        )
        peak_grip_rotation_slip_per_fixture = torch.zeros_like(
            peak_grip_slip_per_fixture
        )
        peak_grip_axis_slip = torch.zeros(
            (args.num_envs, 2, 3),
            dtype=positions.dtype,
            device=positions.device,
        )
        peak_attachment_error = torch.zeros_like(peak_raw_torque)
        minimum_grip_contact_distance = torch.full_like(
            peak_raw_torque, torch.inf
        )
        final_grip_contact_distance = torch.zeros_like(peak_raw_torque)
        grip_contact_seen = torch.zeros(
            args.num_envs,
            dtype=torch.bool,
            device=positions.device,
        )
        peak_deformation = torch.zeros_like(peak_raw_torque)
        grip_position_reference = None
        grip_rotation_reference = None
        evaluation_winding = rope_winding_turns(
            positions, provider.ipc_solver.deformable_bodies
        ).clone()
        evaluation_joint_position = None
        evaluation_joint_velocity = None
        evaluation_wrist_effort = None
        executed_steps = 0
        started = time.perf_counter()
        for _ in range(args.steps):
            applied_wrenches = fixture_wrenches_in_tool_frames(
                provider.arrays, fixture_body_ids, tool_body_ids
            )
            robot_states = tuple(view.read_state() for view in robot_views)
            joint_positions = torch.stack(
                tuple(state.joint_position for state in robot_states), dim=1
            )
            joint_velocities = torch.stack(
                tuple(state.joint_velocity for state in robot_states), dim=1
            )
            wrist_efforts = provider.arrays["actuator_force"][
                :, list(wrist_actuator_ids)
            ]
            command = controller.step(
                applied_wrenches,
                joint_positions,
                joint_velocities,
                wrist_efforts,
            )
            for robot_index, robot_view in enumerate(robot_views):
                robot_view.set_controls(command[:, robot_index])
            gravity_compensator.apply(
                provider.arrays["qfrc_applied"],
                joint_positions,
                controller.gravity_schedule_indices,
            )
            provider.step()
            provider.sense()
            executed_steps += 1

            applied_wrenches = fixture_wrenches_in_tool_frames(
                provider.arrays, fixture_body_ids, tool_body_ids
            )
            raw_torque = applied_wrenches[..., 5].abs().amax(dim=1)
            torch.maximum(peak_raw_torque, raw_torque, out=peak_raw_torque)
            fixture_force = applied_wrenches[..., :3].norm(dim=2)
            torch.maximum(
                peak_fixture_forces,
                fixture_force,
                out=peak_fixture_forces,
            )
            torch.maximum(
                peak_fixture_force,
                fixture_force.amax(dim=1),
                out=peak_fixture_force,
            )
            finger_contact_force = torch.cat(
                tuple(sensor["force"].norm(dim=2) for sensor in grip_sensors),
                dim=1,
            )
            torch.maximum(
                peak_finger_contact_force,
                finger_contact_force.amax(dim=1),
                out=peak_finger_contact_force,
            )
            if TWIST_START_TICK - 20 <= controller.tick <= TWIST_START_TICK:
                preload_force_accumulator.add_(finger_contact_force)
                preload_force_samples += 1
            found = torch.cat(
                tuple(sensor["found"] > 0.5 for sensor in grip_sensors),
                dim=1,
            )
            distances = torch.cat(
                tuple(sensor["dist"] for sensor in grip_sensors), dim=1
            )
            seen_now = found.any(dim=1)
            grip_contact_seen.logical_or_(seen_now)
            observed_minimum = torch.where(
                found,
                distances,
                torch.full_like(distances, torch.inf),
            ).amin(dim=1)
            torch.minimum(
                minimum_grip_contact_distance,
                observed_minimum,
                out=minimum_grip_contact_distance,
            )
            final_grip_contact_distance.copy_(
                torch.where(
                    seen_now,
                    observed_minimum,
                    torch.zeros_like(observed_minimum),
                )
            )
            contact_force = provider.arrays["ipc_contact_forces"].norm(
                dim=2
            ).amax(dim=1)
            torch.maximum(
                peak_contact_force,
                contact_force,
                out=peak_contact_force,
            )
            local_endpoints = rope_endpoints_in_body_frames(
                positions,
                endpoint_indices,
                provider.arrays,
                fixture_body_ids,
            )
            attachment_error = (
                local_endpoints - attachment_reference
            ).norm(dim=3).amax(dim=(1, 2))
            torch.maximum(
                peak_attachment_error,
                attachment_error,
                out=peak_attachment_error,
            )
            grip_position, grip_rotation = body_transforms_in_reference_frames(
                provider.arrays, fixture_body_ids, tool_body_ids
            )
            if controller.tick == TWIST_START_TICK:
                grip_position_reference = grip_position.clone()
                grip_rotation_reference = grip_rotation.clone()
                grip_preload_finger_forces.copy_(
                    preload_force_accumulator / preload_force_samples
                )
                grip_preload_force.copy_(
                    grip_preload_finger_forces.amin(dim=1)
                )
            elif grip_position_reference is not None:
                grip_slip, grip_rotation_slip = relative_transform_errors(
                    grip_position,
                    grip_rotation,
                    grip_position_reference,
                    grip_rotation_reference,
                )
                torch.maximum(
                    peak_grip_slip,
                    grip_slip.amax(dim=1),
                    out=peak_grip_slip,
                )
                torch.maximum(
                    peak_grip_slip_per_fixture,
                    grip_slip,
                    out=peak_grip_slip_per_fixture,
                )
                torch.maximum(
                    peak_grip_rotation_slip,
                    grip_rotation_slip.amax(dim=1),
                    out=peak_grip_rotation_slip,
                )
                torch.maximum(
                    peak_grip_rotation_slip_per_fixture,
                    grip_rotation_slip,
                    out=peak_grip_rotation_slip_per_fixture,
                )
                axis_slip = (grip_position - grip_position_reference).abs()
                torch.maximum(
                    peak_grip_axis_slip,
                    axis_slip,
                    out=peak_grip_axis_slip,
                )
            deformation = maximum_shape_deformation(
                initial_positions, positions
            )
            torch.maximum(
                peak_deformation, deformation, out=peak_deformation
            )
            if controller.tick == min(args.steps, TWIST_COMPLETE_TICK):
                evaluation_winding = rope_winding_turns(
                    positions, provider.ipc_solver.deformable_bodies
                ).clone()
                evaluation_joint_position = torch.stack(
                    tuple(
                        robot_view.read_state().joint_position
                        for robot_view in robot_views
                    ),
                    dim=1,
                )
                evaluation_joint_velocity = torch.stack(
                    tuple(
                        robot_view.read_state().joint_velocity
                        for robot_view in robot_views
                    ),
                    dim=1,
                )
                evaluation_wrist_effort = provider.arrays["actuator_force"][
                    :, list(wrist_actuator_ids)
                ].clone()
            if (
                controller.tick >= TWIST_START_TICK
                and controller.tick % 50 == 0
                and controller.cycle_complete
            ):
                break
        provider.synchronize()
        elapsed = time.perf_counter() - started
        if evaluation_joint_position is None:
            evaluation_joint_position = torch.stack(
                tuple(
                    robot_view.read_state().joint_position
                    for robot_view in robot_views
                ),
                dim=1,
            )
            evaluation_winding = rope_winding_turns(
                positions, provider.ipc_solver.deformable_bodies
            ).clone()
            evaluation_joint_velocity = torch.stack(
                tuple(
                    robot_view.read_state().joint_velocity
                    for robot_view in robot_views
                ),
                dim=1,
            )
            evaluation_wrist_effort = provider.arrays["actuator_force"][
                :, list(wrist_actuator_ids)
            ].clone()
        assert evaluation_joint_velocity is not None
        assert evaluation_wrist_effort is not None
        absolute_positions = absolute_joint_positions(
            evaluation_joint_position
        )
        reported_minimum_grip_distance = torch.where(
            grip_contact_seen,
            minimum_grip_contact_distance,
            torch.zeros_like(minimum_grip_contact_distance),
        )

        provider.rigid_solver.assert_no_overflow()
        for name in (
            "qpos",
            "qfrc_applied",
            "xfrc_applied",
            "ipc_positions",
            "ipc_affine_contact_wrenches",
        ):
            if not bool(torch.isfinite(provider.arrays[name]).all().item()):
                raise RuntimeError(f"non-finite values in {name}")

        trials = []
        for environment in range(args.num_envs):
            trials.append(
                {
                    "environment": environment,
                    "controller": args.drive_mode + "_velocity_drive",
                    "joint7_positions_radians": [
                        float(value)
                        for value in absolute_positions[environment, :, 6].tolist()
                    ],
                    "finger_positions_meters": [
                        [float(value) for value in robot]
                        for robot in absolute_positions[
                            environment, :, 7:9
                        ].tolist()
                    ],
                    "joint7_velocities_radians_per_second": [
                        float(value)
                        for value in evaluation_joint_velocity[
                            environment, :, 6
                        ].tolist()
                    ],
                    "wrist_actuator_efforts_newton_meters": [
                        float(value)
                        for value in evaluation_wrist_effort[
                            environment
                        ].tolist()
                    ],
                    "peak_commanded_speed_radians_per_second": float(
                        controller.peak_commanded_speed[environment].item()
                    ),
                    "peak_actual_relative_rotation_radians": float(
                        controller.peak_actual_relative_rotation[
                            environment
                        ].item()
                    ),
                    "peak_actual_relative_turns": float(
                        controller.peak_actual_relative_rotation[
                            environment
                        ].item()
                        / (2.0 * 3.141592653589793)
                    ),
                    "stalled": bool(controller.stalled[environment].item()),
                    "stall_tick": int(
                        controller.stall_tick[environment].item()
                    ),
                    "stalled_relative_turns": float(
                        controller.stalled_relative_rotation[
                            environment
                        ].item()
                        / (2.0 * 3.141592653589793)
                    ),
                    "stall_wrist_speeds_radians_per_second": [
                        float(value)
                        for value in controller.stalled_wrist_speed[
                            environment
                        ].tolist()
                    ],
                    "stall_wrist_efforts_newton_meters": [
                        float(value)
                        for value in controller.stalled_wrist_effort[
                            environment
                        ].tolist()
                    ],
                    "stall_axial_torques_newton_meters": [
                        float(value)
                        for value in controller.stalled_axial_torque[
                            environment
                        ].tolist()
                    ],
                    "safety_stopped": bool(
                        controller.safety_stopped[environment].item()
                    ),
                    "peak_axial_torque_newton_meters": float(
                        controller.peak_axial_torque[environment].item()
                    ),
                    "raw_peak_axial_torque_newton_meters": float(
                        peak_raw_torque[environment].item()
                    ),
                    "peak_fixture_force_newtons": float(
                        peak_fixture_force[environment].item()
                    ),
                    "peak_fixture_forces_newtons": [
                        float(value)
                        for value in peak_fixture_forces[environment].tolist()
                    ],
                    "peak_contact_force_newtons": float(
                        peak_contact_force[environment].item()
                    ),
                    "grip_preload_minimum_finger_force_newtons": float(
                        grip_preload_force[environment].item()
                    ),
                    "grip_preload_finger_forces_newtons": [
                        float(value)
                        for value in grip_preload_finger_forces[
                            environment
                        ].tolist()
                    ],
                    "peak_finger_contact_force_newtons": float(
                        peak_finger_contact_force[environment].item()
                    ),
                    "maximum_grip_slip_meters": float(
                        peak_grip_slip[environment].item()
                    ),
                    "maximum_grip_slip_per_fixture_meters": [
                        float(value)
                        for value in peak_grip_slip_per_fixture[
                            environment
                        ].tolist()
                    ],
                    "maximum_grip_axis_slip_per_fixture_meters": [
                        [float(value) for value in fixture]
                        for fixture in peak_grip_axis_slip[
                            environment
                        ].tolist()
                    ],
                    "maximum_grip_rotation_slip_radians": float(
                        peak_grip_rotation_slip[environment].item()
                    ),
                    "maximum_grip_rotation_slip_per_fixture_radians": [
                        float(value)
                        for value in peak_grip_rotation_slip_per_fixture[
                            environment
                        ].tolist()
                    ],
                    "maximum_attachment_error_meters": float(
                        peak_attachment_error[environment].item()
                    ),
                    "minimum_grip_contact_distance_meters": float(
                        reported_minimum_grip_distance[environment].item()
                    ),
                    "final_grip_contact_distance_meters": float(
                        final_grip_contact_distance[environment].item()
                    ),
                    "grip_contact_seen": bool(
                        grip_contact_seen[environment].item()
                    ),
                    "strand_winding_turns": [
                        float(value)
                        for value in evaluation_winding[environment].tolist()
                    ],
                    "maximum_shape_deformation_meters": float(
                        peak_deformation[environment].item()
                    ),
                }
            )

        return {
            "artifact": artifact.digest,
            "scene": str(scene_path),
            "task": "dual_fr3_three_strand_rope_twist",
            "robots": list(ROBOT_NAMES),
            "fixtures": list(FIXTURE_ROBOT_NAMES),
            "robot_count": 2,
            "robot_arm_joint_count": 14,
            "robot_gripper_joint_count": 4,
            "deformable_strand_count": 3,
            "attachment_count": len(
                artifact.ipc.manifest_data["deformable_attachments"]
            ),
            "fixture_coupling_count": len(artifact.coupled_bodies),
            "grip_mode": "mujoco_fixture_friction_contact",
            "drive_mode": args.drive_mode,
            "grip_friction_scale": args.grip_friction_scale,
            "device": args.device,
            "steps": executed_steps,
            "requested_steps": args.steps,
            "environments": args.num_envs,
            "environments_per_shard": args.environments_per_shard,
            "shards": provider.ipc_solver.shard_count,
            "elapsed_seconds": elapsed,
            "environment_steps_per_second": (
                args.num_envs * executed_steps / elapsed
            ),
            "coupling_feedback_scale": args.coupling_feedback_scale,
            "feedback_source": provider.diagnostics["feedback_source"],
            "exact_contact_wrench": provider.capabilities.exact_contact_wrench,
            "collision_ownership": dict(artifact.collision_ownership),
            "wrist_drive_torque_limit_newton_meters": drive_torque_limit,
            "stalled_count": int(controller.stalled.count_nonzero().item()),
            "safety_stopped_count": int(
                controller.safety_stopped.count_nonzero().item()
            ),
            "peak_actual_relative_rotation_range_radians": _range(
                controller.peak_actual_relative_rotation
            ),
            "stalled_relative_rotation_range_radians": _range(
                controller.stalled_relative_rotation
            ),
            "stall_wrist_speed_range_radians_per_second": _range(
                controller.stalled_wrist_speed
            ),
            "stall_wrist_effort_range_newton_meters": _range(
                controller.stalled_wrist_effort
            ),
            "stall_axial_torque_range_newton_meters": _range(
                controller.stalled_axial_torque
            ),
            "raw_peak_axial_torque_range_newton_meters": _range(
                peak_raw_torque
            ),
            "peak_fixture_force_range_newtons": _range(
                peak_fixture_force
            ),
            "peak_contact_force_range_newtons": _range(
                peak_contact_force
            ),
            "grip_preload_minimum_finger_force_range_newtons": _range(
                grip_preload_force
            ),
            "peak_finger_contact_force_range_newtons": _range(
                peak_finger_contact_force
            ),
            "maximum_grip_slip_range_meters": _range(
                peak_grip_slip
            ),
            "maximum_grip_rotation_slip_range_radians": _range(
                peak_grip_rotation_slip
            ),
            "maximum_attachment_error_range_meters": _range(
                peak_attachment_error
            ),
            "minimum_grip_contact_distance_range_meters": _range(
                reported_minimum_grip_distance
            ),
            "final_grip_contact_distance_range_meters": _range(
                final_grip_contact_distance
            ),
            "maximum_shape_deformation_range_meters": _range(
                peak_deformation
            ),
            "trials": trials,
            "ipc_position_storage_stable": (
                positions.data_ptr() == position_pointer
            ),
            "qpos_storage_stable": (
                provider.arrays["qpos"].data_ptr() == qpos_pointer
            ),
        }
    finally:
        provider.close()
        context.clear_scene()


def main() -> None:
    print(json.dumps(run(_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
