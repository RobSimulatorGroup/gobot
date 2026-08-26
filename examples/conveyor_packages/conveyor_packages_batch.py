"""Run the mixed rigid/deformable package conveyor without the editor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
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
    MuJoCoIpcConvergencePolicy,
    MuJoCoIpcProvider,
    MuJoCoWarpContactSensorSpec,
    MuJoCoWarpProvider,
)

from build_scene import (
    BELT_CENTER_X,
    BELT_CENTER_Y,
    BELT_SURFACE_LENGTH,
    BELT_TOP_Z,
    BELT_WIDTH,
    HAND_BASE_LINK_NAMES,
    HAND_JOINT_NAMES_BY_SIDE,
    HAND_LINK_NAMES_BY_SIDE,
    HAND_SIDES,
    HERE,
    LEAP_ROBOT_NAMES,
    RIGID_BOX_SPECS,
    SCENE_NAME,
    SOFT_PACKAGE_SPECS,
    build_scene,
)
from conveyor_forces import (
    ConveyorForceModel,
    DeformableConveyorForceModel,
    configure_mujoco_velocity_field_belt,
)
from conveyor_profile import (
    CYCLE_TICKS,
    FIXED_DT,
    HAND_MOTION_SEGMENTS,
    SOFT_PACKAGE_MASSES,
    IPC_CONTACT_ACTIVATION_DISTANCE,
    IPC_CONTACT_FRICTION,
    IPC_CONTACT_RESISTANCE,
    belt_speed_at_tick,
    cycle_phase,
    finger_close_fraction_at_tick,
    hand_controls_at_tick,
    quality_profile,
)


BELT_ROBOT_NAME = "conveyor"
BELT_LINK_NAME = "belt_surface"
BELT_GEOM_NAME = "conveyor_moving_belt_collision"
BELT_DRIVE_FRICTION = 0.92
SOLVER_MODULE_NAME = "libgobot_libuipc_solver.so"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run two floating LEAP Hands that flip and convey a blue mailer, "
            "then flip a yellow mailer and a rigid carton."
        )
    )
    parser.add_argument("--scene", type=Path, default=HERE / SCENE_NAME)
    parser.add_argument("--rebuild-scene", action="store_true")
    parser.add_argument(
        "--quality",
        choices=("interactive", "accurate"),
        default="interactive",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--environments-per-shard", type=int, default=1)
    parser.add_argument("--steps", type=int, default=CYCLE_TICKS)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--module-path", default="")
    parser.add_argument("--no-mujoco-graph", action="store_true")
    parser.add_argument("--no-coupler-graph", action="store_true")
    parser.add_argument(
        "--refresh-contact-forces",
        action="store_true",
        help="export per-vertex libuipc contact forces after the final step",
    )
    parser.add_argument(
        "--trace-force-flow",
        action="store_true",
        help=(
            "record per-step deformable contact/external resultant forces; "
            "intended for physics diagnostics, not latency measurement"
        ),
    )
    parser.add_argument(
        "--phase-diagnostics",
        action="store_true",
        help=(
            "record device-side object state at manipulation phase boundaries"
        ),
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(tempfile.gettempdir()) / "gobot-conveyor-packages",
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
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")


def _load_artifact(
    scene_path: Path,
) -> tuple[Any, CompiledMuJoCoIpcArtifact]:
    context = gobot.app.create_context()
    context.set_project_path(str(scene_path.parent))
    context.load_scene("res://" + scene_path.name)
    settings = context.get_mujoco_solver_settings()
    settings["integrator"] = gobot.PhysicsIntegratorType.ImplicitFast
    settings["cone"] = gobot.PhysicsFrictionConeType.Elliptic
    context.set_mujoco_solver_settings(settings)
    return context, CompiledMuJoCoIpcArtifact.from_context(context)


def _discover_solver_module(repository_root: Path) -> str:
    candidates = {
        repository_root
        / "build"
        / "libuipc-novcpkg"
        / "python"
        / "gobot"
        / SOLVER_MODULE_NAME,
        repository_root / "build" / "python" / "gobot" / SOLVER_MODULE_NAME,
    }
    candidates.update(
        (repository_root / "build").glob(
            "*/python/gobot/" + SOLVER_MODULE_NAME
        )
    )
    resolved_candidates = sorted(
        (candidate.resolve() for candidate in candidates if candidate.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for resolved in resolved_candidates:
        availability = LibuipcBatchSolver.availability(
            LibuipcBatchConfig(
                solver=LibuipcConfig(module_path=str(resolved))
            )
        )
        if availability.available:
            return str(resolved)
    return ""


def _contact_sensor_specs() -> tuple[MuJoCoWarpContactSensorSpec, ...]:
    return tuple(
        MuJoCoWarpContactSensorSpec(
            name=f"{spec['name']}_belt_contact",
            primary_type="geom",
            primary_names=(BELT_GEOM_NAME,),
            secondary_type="body",
            secondary_name=f"{spec['name']}_{spec['name']}",
            fields=("found", "force"),
            reduce="none",
            num_slots=8,
        )
        for spec in RIGID_BOX_SPECS
    )
def _range(values: Any) -> list[float]:
    return [float(values.min().item()), float(values.max().item())]


def _body_centers(positions: Any, entries: tuple[dict[str, Any], ...]) -> Any:
    return torch.stack(
        tuple(
            positions[
                :,
                int(entry["element_offset"]) : int(entry["element_offset"])
                + int(entry["element_count"]),
            ].mean(dim=1)
            for entry in entries
        ),
        dim=1,
    )


def _body_heights(positions: Any, entries: tuple[dict[str, Any], ...]) -> Any:
    heights = []
    for entry in entries:
        vertices = positions[
            :,
            int(entry["element_offset"]) : int(entry["element_offset"])
            + int(entry["element_count"]),
            2,
        ]
        heights.append(vertices.amax(dim=1) - vertices.amin(dim=1))
    return torch.stack(tuple(heights), dim=1)


def _body_extents(positions: Any, entries: tuple[dict[str, Any], ...]) -> Any:
    extents = []
    for entry in entries:
        vertices = positions[
            :,
            int(entry["element_offset"]) : int(entry["element_offset"])
            + int(entry["element_count"]),
        ]
        extents.append(vertices.amax(dim=1) - vertices.amin(dim=1))
    return torch.stack(tuple(extents), dim=1)


def _mailer_layer_axis(positions: Any, entry: dict[str, Any]) -> Any:
    begin = int(entry["element_offset"])
    count = int(entry["element_count"])
    if count % 2:
        raise RuntimeError("thin-shell mailer must have two equal vertex layers")
    half = count // 2
    bottom = positions[:, begin : begin + half].mean(dim=1)
    top = positions[:, begin + half : begin + count].mean(dim=1)
    return torch.nn.functional.normalize(top - bottom, dim=-1)


def _soft_body_diagnostics(
    positions: Any,
    velocities: Any,
    contact_forces: Any | None,
    entries: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    diagnostics = []
    for spec, entry in zip(SOFT_PACKAGE_SPECS, entries, strict=True):
        begin = int(entry["element_offset"])
        end = begin + int(entry["element_count"])
        body_positions = positions[0, begin:end]
        body_velocities = velocities[0, begin:end]
        center_velocity = body_velocities.mean(dim=0)
        minimum_height = body_positions[:, 2].amin()
        support_vertices = body_positions[
            body_positions[:, 2] <= minimum_height + 3.0e-3
        ]
        support_extent = support_vertices.amax(dim=0) - support_vertices.amin(
            dim=0
        )
        body_diagnostics = {
            "name": str(spec["name"]),
            "aabb_min_meters": [
                float(value)
                for value in body_positions.amin(dim=0).tolist()
            ],
            "aabb_max_meters": [
                float(value)
                for value in body_positions.amax(dim=0).tolist()
            ],
            "vertex_speed_peak_meters_per_second": float(
                torch.linalg.vector_norm(body_velocities, dim=-1)
                .amax()
                .item()
            ),
            "center_velocity_meters_per_second": [
                float(value) for value in center_velocity.tolist()
            ],
            "center_speed_meters_per_second": float(
                torch.linalg.vector_norm(center_velocity).item()
            ),
            "support_vertices_within_3mm": int(support_vertices.shape[0]),
            "support_patch_extent_meters": [
                float(value) for value in support_extent.tolist()
            ],
        }
        if contact_forces is not None:
            body_contact_forces = contact_forces[0, begin:end]
            body_diagnostics["contact_force_peak_newtons"] = float(
                torch.linalg.vector_norm(
                    body_contact_forces, dim=-1
                )
                .amax()
                .item()
            )
            body_diagnostics["contact_force_resultant_newtons"] = [
                float(value) for value in body_contact_forces.sum(dim=0).tolist()
            ]
        diagnostics.append(body_diagnostics)
    return diagnostics


def _force_flow_diagnostics(
    contact_force_trace: Any,
    external_force_trace: Any,
    center_trace: Any,
    extent_trace: Any,
    hand_wrench_trace: Any,
    initial_centers: Any,
    initial_extents: Any,
) -> list[dict[str, Any]]:
    contact = contact_force_trace[:, 0].detach().cpu()
    external = external_force_trace[:, 0].detach().cpu()
    centers = center_trace[:, 0].detach().cpu()
    extents = extent_trace[:, 0].detach().cpu()
    hand_reactions = (
        hand_wrench_trace[:, 0, :, :3].sum(dim=1).detach().cpu()
    )
    initial = initial_centers[0].detach().cpu()
    reference_extents = initial_extents[0].detach().cpu()
    diagnostics = []
    for body_index, spec in enumerate(SOFT_PACKAGE_SPECS):
        across_belt = contact[:, body_index, 1]
        peak_tick = int(torch.argmax(across_belt).item())
        reverse_peak_tick = int(torch.argmin(across_belt).item())
        across_belt_impulse = float(
            (across_belt.sum() * FIXED_DT).item()
        )
        across_belt_displacement = (
            centers[:, body_index, 1] - initial[body_index, 1]
        )
        forward_displacement_tick = int(
            torch.argmax(across_belt_displacement).item()
        )
        horizontal = torch.linalg.vector_norm(
            contact[:, body_index, :2], dim=-1
        )
        horizontal_peak_tick = int(torch.argmax(horizontal).item())
        external_magnitude = torch.linalg.vector_norm(
            external[:, body_index], dim=-1
        )
        external_peak_tick = int(torch.argmax(external_magnitude).item())
        external_active = external_magnitude > 1.0e-6
        external_active_ticks = torch.nonzero(
            external_active, as_tuple=False
        ).flatten()
        external_first_tick = (
            int(external_active_ticks[0].item())
            if external_active_ticks.numel()
            else -1
        )
        external_last_tick = (
            int(external_active_ticks[-1].item())
            if external_active_ticks.numel()
            else -1
        )
        extent_ratios = extents[:, body_index] / reference_extents[body_index]
        minimum_thickness_tick = int(
            torch.argmin(extent_ratios[:, 2]).item()
        )
        diagnostics.append(
            {
                "name": str(spec["name"]),
                "peak_across_belt_contact_force_newtons": float(
                    across_belt[peak_tick].item()
                ),
                "peak_across_belt_tick": peak_tick,
                "peak_across_belt_phase": cycle_phase(peak_tick),
                "reverse_peak_across_belt_contact_force_newtons": float(
                    across_belt[reverse_peak_tick].item()
                ),
                "reverse_peak_across_belt_tick": reverse_peak_tick,
                "hand_proxy_reaction_at_reverse_peak_newtons": [
                    float(value)
                    for value in hand_reactions[reverse_peak_tick].tolist()
                ],
                "net_across_belt_contact_impulse_newton_seconds": (
                    across_belt_impulse
                ),
                "maximum_forward_displacement_meters": float(
                    across_belt_displacement[
                        forward_displacement_tick
                    ].item()
                ),
                "maximum_forward_displacement_tick": (
                    forward_displacement_tick
                ),
                "belt_speed_at_peak_meters_per_second": float(
                    belt_speed_at_tick(peak_tick)
                ),
                "contact_resultant_at_peak_newtons": [
                    float(value)
                    for value in contact[peak_tick, body_index].tolist()
                ],
                "hand_proxy_reaction_at_peak_newtons": [
                    float(value) for value in hand_reactions[peak_tick].tolist()
                ],
                "external_resultant_at_contact_peak_newtons": [
                    float(value)
                    for value in external[peak_tick, body_index].tolist()
                ],
                "center_displacement_at_peak_meters": [
                    float(value)
                    for value in (
                        centers[peak_tick, body_index] - initial[body_index]
                    ).tolist()
                ],
                "peak_horizontal_contact_force_newtons": float(
                    horizontal[horizontal_peak_tick].item()
                ),
                "peak_horizontal_contact_tick": horizontal_peak_tick,
                "peak_external_force_newtons": float(
                    external_magnitude[external_peak_tick].item()
                ),
                "peak_external_force_tick": external_peak_tick,
                "external_force_active_steps": int(
                    external_active_ticks.numel()
                ),
                "external_force_first_tick": external_first_tick,
                "external_force_last_tick": external_last_tick,
                "external_force_impulse_newton_seconds": [
                    float(value)
                    for value in (
                        external[:, body_index].sum(dim=0) * FIXED_DT
                    ).tolist()
                ],
                "external_peak_resultant_newtons": [
                    float(value)
                    for value in external[external_peak_tick, body_index].tolist()
                ],
                "extent_ratio_at_contact_peak": [
                    float(value)
                    for value in extent_ratios[peak_tick].tolist()
                ],
                "minimum_thickness_ratio": float(
                    extent_ratios[minimum_thickness_tick, 2].item()
                ),
                "minimum_thickness_tick": minimum_thickness_tick,
                "minimum_thickness_phase": cycle_phase(
                    minimum_thickness_tick
                ),
            }
        )
    return diagnostics


def _rigid_body_diagnostics(body_states: tuple[Any, ...]) -> list[dict[str, Any]]:
    diagnostics = []
    for spec, state in zip(RIGID_BOX_SPECS, body_states, strict=True):
        velocity = state.base_velocity[0]
        diagnostics.append(
            {
                "name": str(spec["name"]),
                "position_meters": [
                    float(value) for value in state.base_pose[0, :3].tolist()
                ],
                "linear_speed_meters_per_second": float(
                    torch.linalg.vector_norm(velocity[:3]).item()
                ),
                "angular_speed_radians_per_second": float(
                    torch.linalg.vector_norm(velocity[3:]).item()
                ),
            }
        )
    return diagnostics


def _hand_diagnostics(
    hand_states: tuple[Any, ...], hand_command: Any
) -> list[dict[str, Any]]:
    diagnostics = []
    for side, state, command in zip(
        HAND_SIDES, hand_states, hand_command[0], strict=True
    ):
        joint_error = state.joint_position[0] - command
        diagnostics.append(
            {
                "side": side,
                "stage_control_error_peak": float(
                    joint_error[:6].abs().amax().item()
                ),
                "finger_joint_error_peak_radians": float(
                    joint_error[6:].abs().amax().item()
                ),
                "contact_link_positions_meters": [
                    [float(value) for value in position]
                    for position in state.link_pose[0, 5:, :3].tolist()
                ],
            }
        )
    return diagnostics


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run one fixed-capacity batch and return JSON-compatible metrics."""

    _validate_args(args)
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Torch cannot initialize the requested CUDA device")
    # Establish Torch's CUDA primary context before module discovery dlopens
    # the native IPC solver.
    torch.cuda.init()
    scene_path = args.scene.expanduser().resolve()
    if args.rebuild_scene:
        scene_path = build_scene(scene_path.parent)
    elif not scene_path.is_file():
        raise FileNotFoundError(
            f"scene does not exist: {scene_path}; run build_scene.py first"
        )

    profile = quality_profile(args.quality)
    context, artifact = _load_artifact(scene_path)
    module_path = (
        str(Path(args.module_path).expanduser().resolve())
        if args.module_path
        else ""
    )
    if not module_path:
        repository_root = HERE.parents[1]
        module_path = _discover_solver_module(repository_root)
    solver_config = LibuipcBatchConfig(
        solver=LibuipcConfig(
            fixed_time_step=FIXED_DT,
            gravity=(0.0, 0.0, -9.81),
            friction_coefficient=IPC_CONTACT_FRICTION,
            contact_activation_distance=IPC_CONTACT_ACTIVATION_DISTANCE,
            contact_resistance=IPC_CONTACT_RESISTANCE,
            affine_stiffness=1.0e8,
            module_path=module_path,
            workspace=str(args.workspace.expanduser().resolve()),
        ),
        environments_per_shard=args.environments_per_shard,
        newton_max_iterations=profile.newton_max_iterations,
        line_search_max_iterations=profile.line_search_max_iterations,
        linear_system_tolerance_rate=profile.linear_system_tolerance_rate,
        strict_convergence=profile.strict_convergence,
        # The velocity-field force model consumes these device buffers every
        # step. The visualization flag below only controls host readback.
        export_deformable_state=True,
        export_affine_state=profile.scene_sync_interval == 1,
        export_deformable_contact_forces=True,
    )
    rigid_availability = MuJoCoWarpProvider.availability()
    if not rigid_availability.available:
        context.clear_scene()
        raise RuntimeError(rigid_availability.reason)
    ipc_availability = LibuipcBatchSolver.availability(solver_config)
    if not ipc_availability.available:
        context.clear_scene()
        raise RuntimeError(
            ipc_availability.reason
            + "; pass --module-path or build the in-tree libuipc module"
        )

    provider = MuJoCoIpcProvider(
        artifact,
        config=MuJoCoIpcConfig(
            num_envs=args.num_envs,
            device=args.device,
            environments_per_shard=args.environments_per_shard,
            coupling_iterations=profile.coupling_iterations,
            relaxation_mode=profile.relaxation_mode,
            relaxation_factor=1.0,
            capture_mujoco_graphs=not args.no_mujoco_graph,
            capture_coupler_graphs=not args.no_coupler_graph,
            convergence_policy=MuJoCoIpcConvergencePolicy(
                enabled=profile.strict_convergence
            ),
        ),
        libuipc_config=solver_config,
        mujoco_options={
            "nconmax": 512,
            "njmax": 4096,
            "contact_sensor_maxmatch": 64,
            "contact_sensors": _contact_sensor_specs(),
            "overflow_check_interval": 0,
        },
    )
    try:
        configure_mujoco_velocity_field_belt(provider, BELT_GEOM_NAME)
        box_names = tuple(str(spec["name"]) for spec in RIGID_BOX_SPECS)
        box_views = tuple(
            provider.create_robot_view(
                robot_name=name,
                base_link=name,
                joint_names=(),
                link_names=(name,),
            )
            for name in box_names
        )
        box_body_ids = tuple(
            provider.rigid_solver.resolve_object_ids(
                "body", (f"{name}_{name}",)
            )[0]
            for name in box_names
        )
        hand_views = tuple(
            provider.create_robot_view(
                robot_name=robot_name,
                base_link=base_link,
                joint_names=joint_names,
                link_names=link_names,
            )
            for robot_name, base_link, joint_names, link_names in zip(
                LEAP_ROBOT_NAMES,
                HAND_BASE_LINK_NAMES,
                HAND_JOINT_NAMES_BY_SIDE,
                HAND_LINK_NAMES_BY_SIDE,
                strict=True,
            )
        )
        force_model = ConveyorForceModel(
            provider,
            box_views,
            box_body_ids,
            tuple(f"{name}_belt_contact" for name in box_names),
            tuple(float(spec["mass"]) for spec in RIGID_BOX_SPECS),
            friction_coefficient=BELT_DRIVE_FRICTION,
            fixed_dt=FIXED_DT,
        )
        belt_speed = torch.zeros(
            args.num_envs,
            dtype=provider.arrays["qpos"].dtype,
            device=provider.arrays["qpos"].device,
        )
        belt_twist = torch.zeros(
            (args.num_envs, 6),
            dtype=belt_speed.dtype,
            device=belt_speed.device,
        )
        hand_command = torch.zeros(
            (args.num_envs, len(HAND_SIDES), 22),
            dtype=belt_speed.dtype,
            device=belt_speed.device,
        )
        control_ticks = args.warmup_steps + args.steps
        hand_control_trajectory = torch.as_tensor(
            tuple(hand_controls_at_tick(tick) for tick in range(control_ticks)),
            dtype=belt_speed.dtype,
            device=belt_speed.device,
        ).unsqueeze(1).expand(-1, args.num_envs, -1, -1).contiguous()
        belt_travel = 0.0
        reset_mask = torch.ones(
            args.num_envs, dtype=torch.bool, device=belt_speed.device
        )
        initial_qpos = provider.arrays["qpos"].clone()
        initial_qvel = torch.zeros_like(provider.arrays["qvel"])
        initial_ctrl = torch.zeros_like(provider.arrays["ctrl"])
        ipc_position_pointer = provider.arrays["ipc_positions"].data_ptr()
        peak_drive_force = torch.zeros(
            len(box_names),
            dtype=belt_speed.dtype,
            device=belt_speed.device,
        )
        peak_normal_force = torch.zeros(
            len(box_names),
            dtype=belt_speed.dtype,
            device=belt_speed.device,
        )

        provider.refresh_state()
        provider.synchronize()
        deformable_entries = tuple(provider.ipc_solver.deformable_bodies)
        if len(deformable_entries) != len(SOFT_PACKAGE_SPECS):
            raise RuntimeError("conveyor deformable body count changed")
        affine_entries = tuple(provider.ipc_solver.affine_bodies)
        hand_proxy_indices = tuple(
            index
            for index, entry in enumerate(affine_entries)
            if "leap_" in str(entry["path"])
        )
        peak_hand_proxy_wrench = torch.zeros(
            len(hand_proxy_indices),
            dtype=belt_speed.dtype,
            device=belt_speed.device,
        )
        soft_force_model = DeformableConveyorForceModel(
            provider,
            deformable_entries,
            SOFT_PACKAGE_MASSES,
            friction_coefficient=BELT_DRIVE_FRICTION,
            fixed_dt=FIXED_DT,
            belt_half_length=0.5 * BELT_SURFACE_LENGTH,
            belt_half_width=0.5 * BELT_WIDTH,
            belt_top=BELT_TOP_Z,
            belt_center_x=BELT_CENTER_X,
            belt_center_y=BELT_CENTER_Y,
            velocity_damping_rates=tuple(
                float(spec["damping"]) for spec in SOFT_PACKAGE_SPECS
            ),
        )
        peak_soft_drive_force = torch.zeros(
            (), dtype=belt_speed.dtype, device=belt_speed.device
        )
        initial_ipc_positions = provider.arrays["ipc_positions"].clone()
        mailer_entry_indices = (0, 2)
        initial_mailer_layer_axes = torch.stack(
            tuple(
                _mailer_layer_axis(
                    initial_ipc_positions, deformable_entries[index]
                )
                for index in mailer_entry_indices
            ),
            dim=1,
        )
        minimum_mailer_layer_dots = torch.ones(
            (args.num_envs, len(mailer_entry_indices)),
            dtype=initial_ipc_positions.dtype,
            device=initial_ipc_positions.device,
        )
        initial_soft_centers = _body_centers(
            initial_ipc_positions, deformable_entries
        )
        initial_soft_heights = _body_heights(
            initial_ipc_positions, deformable_entries
        )
        initial_soft_extents = _body_extents(
            initial_ipc_positions, deformable_entries
        )
        initial_rigid_x = torch.stack(
            tuple(view.read_state().base_pose[:, 0] for view in box_views),
            dim=1,
        ).clone()
        initial_carton_quaternion = (
            box_views[0].read_state().base_pose[:, 3:7].clone()
        )
        minimum_carton_quaternion_dot = torch.ones(
            args.num_envs,
            dtype=initial_ipc_positions.dtype,
            device=initial_ipc_positions.device,
        )
        phase_snapshot_indices: dict[int, int] = {}
        phase_snapshot_centers = None
        phase_snapshot_mailer_axes = None
        phase_snapshot_carton_poses = None
        if args.phase_diagnostics:
            phase_end_tick = -1
            for phase_index, segment in enumerate(HAND_MOTION_SEGMENTS):
                phase_end_tick += segment.duration
                if phase_end_tick >= control_ticks:
                    break
                phase_snapshot_indices[phase_end_tick] = phase_index
            snapshot_count = len(phase_snapshot_indices)
            phase_snapshot_centers = torch.empty(
                (
                    snapshot_count,
                    args.num_envs,
                    len(deformable_entries),
                    3,
                ),
                dtype=initial_ipc_positions.dtype,
                device=initial_ipc_positions.device,
            )
            phase_snapshot_mailer_axes = torch.empty(
                (
                    snapshot_count,
                    args.num_envs,
                    len(mailer_entry_indices),
                    3,
                ),
                dtype=initial_ipc_positions.dtype,
                device=initial_ipc_positions.device,
            )
            phase_snapshot_carton_poses = torch.empty(
                (snapshot_count, args.num_envs, 7),
                dtype=initial_ipc_positions.dtype,
                device=initial_ipc_positions.device,
            )
        trace_step_count = control_ticks
        contact_force_trace = None
        external_force_trace = None
        center_trace = None
        extent_trace = None
        hand_wrench_trace = None
        if args.trace_force_flow:
            trace_shape = (
                trace_step_count,
                args.num_envs,
                len(deformable_entries),
                3,
            )
            contact_force_trace = torch.empty(
                trace_shape,
                dtype=initial_ipc_positions.dtype,
                device=initial_ipc_positions.device,
            )
            external_force_trace = torch.empty_like(contact_force_trace)
            center_trace = torch.empty_like(contact_force_trace)
            extent_trace = torch.empty_like(contact_force_trace)
            hand_wrench_trace = torch.empty(
                (
                    trace_step_count,
                    args.num_envs,
                    len(hand_proxy_indices),
                    6,
                ),
                dtype=initial_ipc_positions.dtype,
                device=initial_ipc_positions.device,
            )

        latency_samples = []
        for tick in range(trace_step_count):
            speed = belt_speed_at_tick(tick)
            belt_speed.fill_(speed)
            belt_twist.zero_()
            belt_twist[:, 0].copy_(belt_speed)
            belt_travel += speed * FIXED_DT
            force_model.apply(belt_speed)
            soft_force_model.apply(belt_speed)
            hand_command.copy_(hand_control_trajectory[tick])
            for hand_index, hand_view in enumerate(hand_views):
                hand_view.set_controls(hand_command[:, hand_index])
            torch.maximum(
                peak_drive_force,
                force_model.drive_force.abs().amax(dim=0),
                out=peak_drive_force,
            )
            torch.maximum(
                peak_normal_force,
                force_model.normal_force.amax(dim=0),
                out=peak_normal_force,
            )
            torch.maximum(
                peak_soft_drive_force,
                soft_force_model.drive_force.abs().amax(),
                out=peak_soft_drive_force,
            )
            if tick >= args.warmup_steps:
                provider.synchronize()
                started = time.perf_counter()
            provider.step()
            if contact_force_trace is not None:
                for body_index, entry in enumerate(deformable_entries):
                    begin = int(entry["element_offset"])
                    end = begin + int(entry["element_count"])
                    torch.sum(
                        provider.arrays["ipc_contact_forces"][:, begin:end],
                        dim=1,
                        out=contact_force_trace[tick, :, body_index],
                    )
                    torch.sum(
                        provider.arrays["ipc_external_forces"][:, begin:end],
                        dim=1,
                        out=external_force_trace[tick, :, body_index],
                    )
                    torch.mean(
                        provider.arrays["ipc_positions"][:, begin:end],
                        dim=1,
                        out=center_trace[tick, :, body_index],
                    )
                    body_positions = provider.arrays["ipc_positions"][
                        :, begin:end
                    ]
                    extent_trace[tick, :, body_index].copy_(
                        body_positions.amax(dim=1)
                        - body_positions.amin(dim=1)
                    )
                hand_wrench_trace[tick].copy_(
                    provider.arrays["ipc_affine_contact_wrenches"][
                        :, hand_proxy_indices
                    ]
                )
            current_mailer_layer_axes = torch.stack(
                tuple(
                    _mailer_layer_axis(
                        provider.arrays["ipc_positions"],
                        deformable_entries[index],
                    )
                    for index in mailer_entry_indices
                ),
                dim=1,
            )
            torch.minimum(
                minimum_mailer_layer_dots,
                (current_mailer_layer_axes * initial_mailer_layer_axes).sum(
                    dim=-1
                ),
                out=minimum_mailer_layer_dots,
            )
            carton_quaternion = box_views[0].read_state().base_pose[:, 3:7]
            carton_quaternion_dot = torch.sum(
                carton_quaternion * initial_carton_quaternion, dim=-1
            ).abs()
            torch.minimum(
                minimum_carton_quaternion_dot,
                carton_quaternion_dot,
                out=minimum_carton_quaternion_dot,
            )
            phase_snapshot_index = phase_snapshot_indices.get(tick)
            if phase_snapshot_index is not None:
                phase_snapshot_centers[phase_snapshot_index].copy_(
                    _body_centers(
                        provider.arrays["ipc_positions"], deformable_entries
                    )
                )
                phase_snapshot_mailer_axes[phase_snapshot_index].copy_(
                    current_mailer_layer_axes
                )
                phase_snapshot_carton_poses[phase_snapshot_index].copy_(
                    box_views[0].read_state().base_pose
                )
            hand_proxy_wrench = torch.linalg.vector_norm(
                provider.arrays["ipc_affine_contact_wrenches"][
                    :, hand_proxy_indices
                ],
                dim=-1,
            ).amax(dim=0)
            torch.maximum(
                peak_hand_proxy_wrench,
                hand_proxy_wrench,
                out=peak_hand_proxy_wrench,
            )
            if tick >= args.warmup_steps:
                provider.synchronize()
                latency_samples.append(time.perf_counter() - started)

        provider.refresh_state()
        provider.synchronize()
        final_ipc_positions = provider.arrays["ipc_positions"].clone()
        final_mailer_layer_axes = torch.stack(
            tuple(
                _mailer_layer_axis(
                    final_ipc_positions, deformable_entries[index]
                )
                for index in mailer_entry_indices
            ),
            dim=1,
        )
        final_mailer_layer_dots = (
            final_mailer_layer_axes * initial_mailer_layer_axes
        ).sum(dim=-1).clamp(-1.0, 1.0)
        final_mailer_flip_degrees = torch.rad2deg(
            torch.acos(final_mailer_layer_dots)
        )
        maximum_mailer_flip_degrees = torch.rad2deg(
            torch.acos(minimum_mailer_layer_dots.clamp(-1.0, 1.0))
        )
        final_soft_centers = _body_centers(
            final_ipc_positions, deformable_entries
        )
        final_soft_heights = _body_heights(
            final_ipc_positions, deformable_entries
        )
        final_rigid_states = tuple(view.read_state() for view in box_views)
        final_carton_quaternion_dot = torch.sum(
            final_rigid_states[0].base_pose[:, 3:7]
            * initial_carton_quaternion,
            dim=-1,
        ).abs().clamp(0.0, 1.0)
        final_carton_flip_degrees = torch.rad2deg(
            2.0 * torch.acos(final_carton_quaternion_dot)
        )
        maximum_carton_flip_degrees = torch.rad2deg(
            2.0
            * torch.acos(
                minimum_carton_quaternion_dot.clamp(0.0, 1.0)
            )
        )
        final_hand_states = tuple(view.read_state() for view in hand_views)
        hand_diagnostics = _hand_diagnostics(final_hand_states, hand_command)
        hand_proxy_transforms = provider.arrays[
            "ipc_affine_transforms"
        ][0, hand_proxy_indices]
        hand_proxy_diagnostics = [
            {
                "path": str(affine_entries[index]["path"]),
                "position_meters": [
                    float(value)
                    for value in transform[:3, 3].tolist()
                ],
                "peak_contact_wrench": float(wrench),
                "final_contact_wrench": [
                    float(value)
                    for value in provider.arrays[
                        "ipc_affine_contact_wrenches"
                    ][0, index].tolist()
                ],
            }
            for index, transform, wrench in zip(
                hand_proxy_indices,
                hand_proxy_transforms,
                peak_hand_proxy_wrench.tolist(),
                strict=True,
            )
        ]
        final_rigid_x = torch.stack(
            tuple(state.base_pose[:, 0] for state in final_rigid_states),
            dim=1,
        )
        rigid_body_diagnostics = _rigid_body_diagnostics(
            final_rigid_states
        )
        soft_body_diagnostics = _soft_body_diagnostics(
            final_ipc_positions,
            provider.arrays["ipc_velocities"],
            (
                provider.arrays["ipc_contact_forces"]
                if args.refresh_contact_forces
                else None
            ),
            deformable_entries,
        )
        elapsed = sum(latency_samples)
        diagnostics = provider.diagnostics
        contact_force_peak = 0.0
        if args.refresh_contact_forces:
            contact_force_peak = float(
                torch.linalg.vector_norm(
                    provider.arrays["ipc_contact_forces"], dim=-1
                )
                .amax()
                .item()
            )
        force_flow_diagnostics = []
        if contact_force_trace is not None:
            force_flow_diagnostics = _force_flow_diagnostics(
                contact_force_trace,
                external_force_trace,
                center_trace,
                extent_trace,
                hand_wrench_trace,
                initial_soft_centers,
                initial_soft_extents,
            )
        phase_diagnostics = []
        if phase_snapshot_centers is not None:
            snapshot_mailer_dots = (
                phase_snapshot_mailer_axes
                * initial_mailer_layer_axes.unsqueeze(0)
            ).sum(dim=-1).clamp(-1.0, 1.0)
            snapshot_mailer_angles = torch.rad2deg(
                torch.acos(snapshot_mailer_dots)
            )
            snapshot_carton_dots = torch.sum(
                phase_snapshot_carton_poses[..., 3:7]
                * initial_carton_quaternion.unsqueeze(0),
                dim=-1,
            ).abs().clamp(0.0, 1.0)
            snapshot_carton_angles = torch.rad2deg(
                2.0 * torch.acos(snapshot_carton_dots)
            )
            completed_segments = HAND_MOTION_SEGMENTS[
                : len(phase_snapshot_indices)
            ]
            for phase_index, segment in enumerate(completed_segments):
                phase_diagnostics.append(
                    {
                        "phase": segment.phase,
                        "soft_centers_meters": {
                            str(spec["name"]): [
                                float(value) for value in center
                            ]
                            for spec, center in zip(
                                SOFT_PACKAGE_SPECS,
                                phase_snapshot_centers[
                                    phase_index, 0
                                ].tolist(),
                                strict=True,
                            )
                        },
                        "blue_mailer_flip_degrees": float(
                            snapshot_mailer_angles[phase_index, 0, 0]
                        ),
                        "yellow_mailer_flip_degrees": float(
                            snapshot_mailer_angles[phase_index, 0, 1]
                        ),
                        "carton_small_flip_degrees": float(
                            snapshot_carton_angles[phase_index, 0]
                        ),
                    }
                )

        peak_drive_force_newtons = {
            name: float(value)
            for name, value in zip(
                box_names, peak_drive_force.tolist(), strict=True
            )
        }
        peak_normal_force_newtons = {
            name: float(value)
            for name, value in zip(
                box_names, peak_normal_force.tolist(), strict=True
            )
        }
        peak_soft_drive_force_newtons = float(peak_soft_drive_force.item())
        force_model.clear()
        soft_force_model.clear()
        hand_command.zero_()
        belt_speed.zero_()
        belt_twist.zero_()
        provider.reset(
            reset_mask,
            qpos=initial_qpos,
            qvel=initial_qvel,
            ctrl=initial_ctrl,
        )
        for hand_index, hand_view in enumerate(hand_views):
            hand_view.set_controls(hand_command[:, hand_index])
        provider.refresh_state()
        provider.synchronize()
        reset_qpos_error = float(
            (provider.arrays["qpos"] - initial_qpos).abs().amax().item()
        )
        reset_ipc_error = float(
            (
                provider.arrays["ipc_positions"] - initial_ipc_positions
            )
            .abs()
            .amax()
            .item()
        )

        rigid_displacement = final_rigid_x - initial_rigid_x
        soft_x_displacement = (
            final_soft_centers[..., 0] - initial_soft_centers[..., 0]
        )
        soft_y_displacement = (
            final_soft_centers[..., 1] - initial_soft_centers[..., 1]
        )
        soft_center_displacement = final_soft_centers - initial_soft_centers
        soft_height_ratio = final_soft_heights / initial_soft_heights.clamp_min(
            1.0e-8
        )
        return {
            "quality": profile.name,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "belt_surface_commanded_travel_meters": belt_travel,
            "environments": args.num_envs,
            "environments_per_shard": args.environments_per_shard,
            "elapsed_seconds": elapsed,
            "environment_steps_per_second": (
                args.num_envs * args.steps / elapsed if elapsed > 0.0 else 0.0
            ),
            "median_step_latency_seconds": (
                median(latency_samples) if latency_samples else 0.0
            ),
            "rigid_x_displacement_range_meters": _range(rigid_displacement),
            "rigid_body_diagnostics_environment_0": rigid_body_diagnostics,
            "soft_x_displacement_range_meters": _range(soft_x_displacement),
            "soft_y_displacement_range_meters": _range(soft_y_displacement),
            "soft_center_displacement_meters_environment_0": {
                str(spec["name"]): [float(value) for value in displacement]
                for spec, displacement in zip(
                    SOFT_PACKAGE_SPECS,
                    soft_center_displacement[0].tolist(),
                    strict=True,
                )
            },
            "soft_height_ratio_range": _range(soft_height_ratio),
            "hand_diagnostics_environment_0": hand_diagnostics,
            "hand_proxy_diagnostics_environment_0": hand_proxy_diagnostics,
            "mailer_final_flip_degrees_range": _range(
                final_mailer_flip_degrees[:, 0]
            ),
            "mailer_maximum_flip_degrees_range": _range(
                maximum_mailer_flip_degrees[:, 0]
            ),
            "blue_mailer_final_flip_degrees_range": _range(
                final_mailer_flip_degrees[:, 0]
            ),
            "blue_mailer_maximum_flip_degrees_range": _range(
                maximum_mailer_flip_degrees[:, 0]
            ),
            "yellow_mailer_final_flip_degrees_range": _range(
                final_mailer_flip_degrees[:, 1]
            ),
            "yellow_mailer_maximum_flip_degrees_range": _range(
                maximum_mailer_flip_degrees[:, 1]
            ),
            "carton_small_final_flip_degrees_range": _range(
                final_carton_flip_degrees
            ),
            "carton_small_maximum_flip_degrees_range": _range(
                maximum_carton_flip_degrees
            ),
            "peak_rigid_drive_force_newtons": peak_drive_force_newtons,
            "peak_rigid_normal_force_newtons": peak_normal_force_newtons,
            "peak_soft_drive_force_newtons": peak_soft_drive_force_newtons,
            "final_finger_close_fraction": float(
                finger_close_fraction_at_tick(
                    args.warmup_steps + args.steps - 1
                )
            ),
            "final_deformable_contact_force_peak_newtons": contact_force_peak,
            "soft_body_diagnostics_environment_0": soft_body_diagnostics,
            "deformable_contact_forces_refreshed": bool(
                args.refresh_contact_forces
            ),
            "force_flow_trace_environment_0": force_flow_diagnostics,
            "phase_diagnostics_environment_0": phase_diagnostics,
            "coupling_solver": diagnostics["coupling_solver"],
            "exact_contact_wrench": bool(
                provider.capabilities.exact_contact_wrench
            ),
            "coupling_iterations": diagnostics["coupling_iterations"],
            "actual_coupling_iterations": diagnostics[
                "actual_coupling_iterations"
            ],
            "interface_residual": diagnostics["interface_residual"],
            "interface_residual_l2": diagnostics["interface_residual_l2"],
            "coupler_graph_captured": diagnostics[
                "coupler_graph_captured"
            ],
            "coupler_graph_capture_reason": diagnostics[
                "coupler_graph_capture_reason"
            ],
            "mujoco_graph_captured": bool(
                getattr(provider.rigid_solver, "graph_captured", False)
            ),
            "reset_qpos_max_error": reset_qpos_error,
            "reset_ipc_position_max_error": reset_ipc_error,
            "ipc_position_storage_stable": (
                provider.arrays["ipc_positions"].data_ptr()
                == ipc_position_pointer
            ),
        }
    finally:
        provider.close()
        context.clear_scene()


def main() -> None:
    print(json.dumps(run(_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
