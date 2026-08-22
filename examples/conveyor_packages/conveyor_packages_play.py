"""Editor Play entry point for the mixed-package conveyor demo."""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import gobot

# Load Torch's CUDA runtime before the native libuipc solver module.
import torch
from gobot.ipc import LibuipcBatchConfig, LibuipcBatchSolver, LibuipcConfig
from gobot.render import DebugArrow, clear_debug_arrows, set_debug_arrows
from gobot.rl import (
    CompiledMuJoCoIpcArtifact,
    MuJoCoIpcConfig,
    MuJoCoIpcConvergencePolicy,
    MuJoCoIpcProvider,
    MuJoCoWarpContactSensorSpec,
    MuJoCoWarpProvider,
)


SCENE_ROOT_NAME = "conveyor_packages"
BELT_ROBOT_NAME = "conveyor"
BELT_LINK_NAME = "belt_surface"
BELT_GEOM_NAME = "conveyor_moving_belt_collision"
RIGID_BOX_NAMES = ("carton_small", "carton_wide", "carton_tall")
RIGID_BOX_BODY_NAMES = tuple(f"{name}_{name}" for name in RIGID_BOX_NAMES)
RIGID_BOX_MASSES = (0.62, 0.84, 0.76)
SOFT_PACKAGE_NAMES = (
    "soft_mailer_blue",
    "soft_mailer_blue_fill",
    "soft_pouch_yellow",
)
OPENARM_ROBOT_NAME = "openarm_bimanual"
ARM_SIDES = ("left", "right")
ARM_BASE_LINK_NAMES = tuple(
    f"openarm_{side}_base_link" for side in ARM_SIDES
)
ARM_JOINT_NAMES_BY_SIDE = tuple(
    tuple(f"openarm_{side}_joint{index}" for index in range(1, 8))
    + tuple(f"openarm_{side}_finger_joint{index}" for index in range(1, 3))
    for side in ARM_SIDES
)
ARM_LINK_NAMES_BY_SIDE = tuple(
    (f"openarm_{side}_base_link",)
    + tuple(f"openarm_{side}_link{index}" for index in range(1, 7))
    + (
        f"openarm_{side}_ee_base_link",
        f"openarm_{side}_ee_link1",
        f"openarm_{side}_ee_link2",
    )
    for side in ARM_SIDES
)
BELT_CONTACT_SENSOR_NAMES = tuple(
    name + "_belt_contact" for name in RIGID_BOX_NAMES
)
NUM_ENVS = 1
ENVIRONMENTS_PER_SHARD = 1
SOLVER_MODULE_NAME = "libgobot_libuipc_solver.so"
BELT_DRIVE_FRICTION = 0.92
BELT_MARKER_PERIOD = 0.22
BELT_MARKER_WRAP_LENGTH = 2.70
BELT_CENTER_X = 0.0
BELT_CENTER_Y = 0.58
BELT_HALF_WIDTH = 0.36
BELT_TOP_Z = 0.56
CONTACT_FORCE_ARROW_MIN_NEWTONS = 1.0e-3
CONTACT_FORCE_ARROW_MIN_LENGTH = 0.012
CONTACT_FORCE_ARROW_COLOR = (1.0, 0.12, 0.68, 1.0)
CONTACT_HORIZONTAL_RESULTANT_COLOR = (0.08, 0.82, 1.0, 1.0)
EXTERNAL_FORCE_RESULTANT_COLOR = (1.0, 0.58, 0.08, 1.0)
RESULTANT_ARROW_HEIGHT_OFFSET = 0.025


def _nodes_by_name(root: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    pending = [root]
    while pending:
        node = pending.pop()
        if node.name in result:
            raise RuntimeError(
                f"conveyor scene has duplicate node name {node.name!r}"
            )
        result[node.name] = node
        pending.extend(node.children)
    return result


def _repository_root(project_path: str) -> Path | None:
    current = Path(project_path).expanduser().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "CMakeLists.txt").is_file() and (
            candidate / "python" / "gobot"
        ).is_dir():
            return candidate
    return None


def _solver_module_path(project_path: str) -> str:
    configured = os.environ.get("GOBOT_LIBUIPC_SOLVER_MODULE", "").strip()
    if configured:
        return str(Path(configured).expanduser().resolve())
    repository = _repository_root(project_path)
    if repository is None:
        return ""
    candidates = {
        repository
        / "build"
        / "libuipc-novcpkg"
        / "python"
        / "gobot"
        / SOLVER_MODULE_NAME,
        repository / "build" / "python" / "gobot" / SOLVER_MODULE_NAME,
    }
    candidates.update(
        (repository / "build").glob(
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


def _load_project_module(
    project_path: str, filename: str, module_name: str
) -> Any:
    path = Path(project_path).expanduser().resolve() / filename
    if not path.is_file():
        raise FileNotFoundError(f"conveyor runtime module does not exist: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load conveyor runtime module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _batch_config(
    context: Any, profile_module: Any, profile: Any, fixed_dt: float
) -> LibuipcBatchConfig:
    return LibuipcBatchConfig(
        solver=LibuipcConfig(
            fixed_time_step=fixed_dt,
            gravity=(0.0, 0.0, -9.81),
            friction_coefficient=profile_module.IPC_CONTACT_FRICTION,
            contact_activation_distance=(
                profile_module.IPC_CONTACT_ACTIVATION_DISTANCE
            ),
            contact_resistance=profile_module.IPC_CONTACT_RESISTANCE,
            affine_stiffness=1.0e8,
            module_path=_solver_module_path(context.project_path),
            workspace=str(
                Path(tempfile.gettempdir()) / "gobot-conveyor-packages-editor"
            ),
        ),
        environments_per_shard=ENVIRONMENTS_PER_SHARD,
        newton_max_iterations=profile.newton_max_iterations,
        line_search_max_iterations=profile.line_search_max_iterations,
        linear_system_tolerance_rate=(
            profile.linear_system_tolerance_rate
        ),
        strict_convergence=profile.strict_convergence,
        export_deformable_state=True,
        export_affine_state=profile.scene_sync_interval == 1,
        export_deformable_contact_forces=True,
    )


def _belt_contact_sensor_specs() -> tuple[MuJoCoWarpContactSensorSpec, ...]:
    return tuple(
        MuJoCoWarpContactSensorSpec(
            name=sensor_name,
            primary_type="geom",
            primary_names=(BELT_GEOM_NAME,),
            secondary_type="body",
            secondary_name=body_name,
            fields=("found", "force"),
            reduce="none",
            num_slots=8,
        )
        for body_name, sensor_name in zip(
            RIGID_BOX_BODY_NAMES, BELT_CONTACT_SENSOR_NAMES, strict=True
        )
    )


def _contact_force_arrows(
    positions: Any,
    forces: Any,
    *,
    force_scale: float,
    max_force_length: float,
) -> list[DebugArrow]:
    import numpy as np

    points = np.asarray(positions, dtype=np.float64)
    values = np.asarray(forces, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or values.shape != points.shape
    ):
        raise RuntimeError("deformable contact arrays must have shape [count,3]")
    magnitudes = np.linalg.norm(values, axis=1)
    indices = np.flatnonzero(magnitudes >= CONTACT_FORCE_ARROW_MIN_NEWTONS)
    indices = indices[np.argsort(-magnitudes[indices], kind="stable")]
    arrows = []
    for index in indices:
        magnitude = float(magnitudes[index])
        length = min(
            max_force_length,
            max(
                CONTACT_FORCE_ARROW_MIN_LENGTH,
                force_scale * math.log1p(magnitude),
            ),
        )
        if length <= 0.0:
            continue
        arrows.append(
            DebugArrow(
                start=points[index],
                vector=values[index] / magnitude,
                color=CONTACT_FORCE_ARROW_COLOR,
                scale=length,
                label=f"package contact {magnitude:.3g} N",
            )
        )
    return arrows


def _body_resultant_force_arrows(
    positions: Any,
    forces: Any,
    body_ranges: tuple[tuple[int, int], ...],
    body_names: tuple[str, ...],
    *,
    horizontal_only: bool,
    color: tuple[float, float, float, float],
    label: str,
    force_scale: float,
    max_force_length: float,
) -> list[DebugArrow]:
    import numpy as np

    points = np.asarray(positions, dtype=np.float64)
    values = np.asarray(forces, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or values.shape != points.shape
    ):
        raise RuntimeError("deformable force arrays must have shape [count,3]")
    if len(body_ranges) != len(body_names):
        raise RuntimeError("deformable force ranges and names must align")

    arrows = []
    for name, (begin, end) in zip(body_names, body_ranges, strict=True):
        if begin < 0 or end <= begin or end > len(points):
            raise RuntimeError("deformable force range is out of bounds")
        vector = values[begin:end].sum(axis=0)
        if horizontal_only:
            vector[2] = 0.0
        magnitude = float(np.linalg.norm(vector))
        if magnitude < CONTACT_FORCE_ARROW_MIN_NEWTONS:
            continue
        start = points[begin:end].mean(axis=0)
        start[2] = points[begin:end, 2].max() + RESULTANT_ARROW_HEIGHT_OFFSET
        length = min(
            max_force_length,
            max(
                CONTACT_FORCE_ARROW_MIN_LENGTH,
                force_scale * math.log1p(magnitude),
            ),
        )
        arrows.append(
            DebugArrow(
                start=start,
                vector=vector / magnitude,
                color=color,
                scale=length,
                label=f"{name} {label} {magnitude:.3g} N",
            )
        )
    return arrows


class Script(gobot.NodeScript):
    """Run one repeating mixed rigid/deformable conveyor cycle."""

    def _ready(self) -> None:
        self.provider = None
        self.play_session = None
        self.profile_module = None
        self.profile = None
        self.force_model = None
        self.soft_force_model = None
        self.box_views = ()
        self.arm_views = ()
        self.arm_links = ()
        self.arm_command = None
        self.arm_clear_target = None
        self.arm_retract_target = None
        self.arm_grip_target = None
        self.arm_push_target = None
        self.gripper_offsets = None
        self.box_bodies = ()
        self.belt_markers = ()
        self.belt_marker_origins = ()
        self.deformable_bodies = ()
        self.deformable_counts = ()
        self.deformable_ranges = ()
        self.deformable_buffer = None
        self.belt_speed = None
        self.belt_twist = None
        self.belt_proxy_index = -1
        self.reset_mask = None
        self.initial_qpos = None
        self.tick = 0
        self.visual_belt_offset = 0.0
        self.last_scene_sync_frame = -1
        self.last_contact_refresh_frame = -1
        self.contact_arrows_enabled = False
        self.cached_contact_arrows: list[DebugArrow] = []
        self.drop_only = os.environ.get(
            "GOBOT_CONVEYOR_DROP_ONLY", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        try:
            import numpy as np

            root = self.get_root()
            if root is None or root.name != SCENE_ROOT_NAME:
                raise RuntimeError("unexpected conveyor packages scene root")
            nodes = _nodes_by_name(root)
            self.box_bodies = tuple(nodes[name] for name in RIGID_BOX_NAMES)
            self.arm_links = tuple(
                tuple(
                    nodes[link_name]
                    for link_name in link_names
                )
                for link_names in ARM_LINK_NAMES_BY_SIDE
            )
            self.deformable_bodies = tuple(
                nodes[name] for name in SOFT_PACKAGE_NAMES
            )
            markers = tuple(
                node
                for name, node in sorted(nodes.items())
                if name.startswith("belt_marker_")
            )
            self.belt_markers = markers
            self.belt_marker_origins = tuple(
                float(marker.position[0]) for marker in markers
            )

            self.profile_module = _load_project_module(
                self.context.project_path,
                "conveyor_profile.py",
                "gobot_conveyor_packages_runtime_profile",
            )
            forces_module = _load_project_module(
                self.context.project_path,
                "conveyor_forces.py",
                "gobot_conveyor_packages_force_model",
            )
            self.profile = self.profile_module.quality_profile()
            fixed_dt = float(self.profile_module.FIXED_DT)
            solver_config = _batch_config(
                self.context, self.profile_module, self.profile, fixed_dt
            )
            rigid_availability = MuJoCoWarpProvider.availability()
            if not rigid_availability.available:
                raise RuntimeError(rigid_availability.reason)
            ipc_availability = LibuipcBatchSolver.availability(solver_config)
            if not ipc_availability.available:
                raise RuntimeError(
                    ipc_availability.reason
                    + "; set GOBOT_LIBUIPC_SOLVER_MODULE to the built module"
                )

            settings = self.context.get_mujoco_solver_settings()
            settings["integrator"] = gobot.PhysicsIntegratorType.ImplicitFast
            settings["cone"] = gobot.PhysicsFrictionConeType.Elliptic
            self.context.set_mujoco_solver_settings(settings)
            artifact = CompiledMuJoCoIpcArtifact.from_context(self.context)
            self.provider = MuJoCoIpcProvider(
                artifact,
                config=MuJoCoIpcConfig(
                    num_envs=NUM_ENVS,
                    device="cuda:0",
                    environments_per_shard=ENVIRONMENTS_PER_SHARD,
                    coupling_iterations=self.profile.coupling_iterations,
                    relaxation_mode=self.profile.relaxation_mode,
                    relaxation_factor=1.0,
                    capture_mujoco_graphs=True,
                    capture_coupler_graphs=True,
                    convergence_policy=MuJoCoIpcConvergencePolicy(
                        enabled=self.profile.strict_convergence
                    ),
                ),
                libuipc_config=solver_config,
                mujoco_options={
                    "nconmax": 512,
                    "njmax": 4096,
                    "contact_sensor_maxmatch": 64,
                    "contact_sensors": _belt_contact_sensor_specs(),
                    "overflow_check_interval": 0,
                },
            )
            forces_module.configure_mujoco_velocity_field_belt(
                self.provider, BELT_GEOM_NAME
            )
            self.box_views = tuple(
                self.provider.create_robot_view(
                    robot_name=name,
                    base_link=name,
                    joint_names=(),
                    link_names=(name,),
                )
                for name in RIGID_BOX_NAMES
            )
            self.arm_views = tuple(
                self.provider.create_robot_view(
                    robot_name=OPENARM_ROBOT_NAME,
                    base_link=base_link,
                    joint_names=joint_names,
                    link_names=link_names,
                )
                for base_link, joint_names, link_names in zip(
                    ARM_BASE_LINK_NAMES,
                    ARM_JOINT_NAMES_BY_SIDE,
                    ARM_LINK_NAMES_BY_SIDE,
                    strict=True,
                )
            )
            belt_mapping = next(
                mapping
                for mapping in artifact.coupled_bodies
                if mapping.robot_name == BELT_ROBOT_NAME
                and mapping.link_name == BELT_LINK_NAME
            )
            self.belt_proxy_index = belt_mapping.ipc_body_index
            box_body_ids = tuple(
                self.provider.rigid_solver.resolve_object_ids(
                    "body", (runtime_name,)
                )[0]
                for runtime_name in RIGID_BOX_BODY_NAMES
            )
            self.force_model = forces_module.ConveyorForceModel(
                self.provider,
                self.box_views,
                box_body_ids,
                BELT_CONTACT_SENSOR_NAMES,
                RIGID_BOX_MASSES,
                friction_coefficient=BELT_DRIVE_FRICTION,
                fixed_dt=fixed_dt,
            )

            entries = tuple(self.provider.ipc_solver.deformable_bodies)
            counts = tuple(int(entry["element_count"]) for entry in entries)
            if len(entries) != len(self.deformable_bodies):
                raise RuntimeError("conveyor deformable body count changed")
            self.soft_force_model = (
                forces_module.DeformableConveyorForceModel(
                    self.provider,
                    entries,
                    self.profile_module.SOFT_PACKAGE_MASSES,
                    friction_coefficient=BELT_DRIVE_FRICTION,
                    fixed_dt=fixed_dt,
                    belt_half_length=0.5 * BELT_MARKER_WRAP_LENGTH,
                    belt_half_width=BELT_HALF_WIDTH,
                    belt_top=BELT_TOP_Z,
                    belt_center_x=BELT_CENTER_X,
                    belt_center_y=BELT_CENTER_Y,
                    velocity_damping_rates=(
                        self.profile_module.SOFT_PACKAGE_DAMPING_RATES
                    ),
                )
            )
            self.deformable_counts = counts
            self.deformable_ranges = tuple(
                (
                    int(entry["element_offset"]),
                    int(entry["element_offset"])
                    + int(entry["element_count"]),
                )
                for entry in entries
            )
            self.deformable_buffer = np.zeros(
                (len(counts), max(counts), 3), dtype=np.float32
            )
            device = self.provider.arrays["qpos"].device
            dtype = self.provider.arrays["qpos"].dtype
            self.belt_speed = torch.zeros(
                NUM_ENVS, dtype=dtype, device=device
            )
            self.belt_twist = torch.zeros(
                (NUM_ENVS, 6), dtype=dtype, device=device
            )
            self.arm_command = torch.zeros(
                (NUM_ENVS, len(ARM_SIDES), 9),
                dtype=dtype,
                device=device,
            )
            self.arm_retract_target = torch.as_tensor(
                self.profile_module.ARM_RETRACT_TARGETS,
                dtype=dtype,
                device=device,
            ).reshape(1, len(ARM_SIDES), 7).expand(NUM_ENVS, -1, -1)
            self.arm_clear_target = torch.as_tensor(
                self.profile_module.ARM_CLEAR_TARGETS,
                dtype=dtype,
                device=device,
            ).reshape(1, len(ARM_SIDES), 7).expand(NUM_ENVS, -1, -1)
            self.arm_grip_target = torch.as_tensor(
                self.profile_module.ARM_GRIP_TARGETS,
                dtype=dtype,
                device=device,
            ).reshape(1, len(ARM_SIDES), 7).expand(NUM_ENVS, -1, -1)
            self.arm_push_target = torch.as_tensor(
                self.profile_module.ARM_PUSH_TARGETS,
                dtype=dtype,
                device=device,
            ).reshape(1, len(ARM_SIDES), 7).expand(NUM_ENVS, -1, -1)
            self.gripper_offsets = torch.as_tensor(
                self.profile_module.FINGER_GRIP_OFFSETS,
                dtype=dtype,
                device=device,
            ).reshape(1, len(ARM_SIDES), 1).expand(NUM_ENVS, -1, 2)
            self.reset_mask = torch.ones(
                NUM_ENVS, dtype=torch.bool, device=device
            )
            self.initial_qpos = self.provider.arrays["qpos"].clone()

            self._reset_provider()
            self._sync_scene()
            self.play_session = gobot.sim.ProviderPlaySession(
                self.context,
                self.provider,
                fixed_dt=fixed_dt,
                max_sub_steps=1,
                before_step=self._before_step,
                reset=self._reset_provider,
                sync_scene=self._sync_scene,
            ).start()
            self._update_status()
            description = (
                "Drop-only mailer test started: a libuipc thin-shell mailer "
                "falls and settles under gravity while the arms and belt "
                "remain stationary on cuda:0; "
                if self.drop_only
                else "Mixed package workcell started: downward-facing "
                "bimanual palms sweep a libuipc thin-shell mailer from the "
                "static sorting table onto the front velocity-field outfeed "
                "on cuda:0; "
            )
            print(
                description
                + "SolverCoupledProxy "
                f"x{self.profile.coupling_iterations} ({self.profile.name})"
            )
        except Exception:
            self._close_play_session()
            raise

    def _before_step(self, fixed_dt: float) -> None:
        control_tick = self.tick
        if self.drop_only:
            control_tick = min(
                control_tick,
                self.profile_module.DROP_SETTLE_TICKS - 1,
            )
        speed = self.profile_module.belt_speed_at_tick(control_tick)
        self.belt_speed.fill_(speed)
        self.belt_twist.zero_()
        self.belt_twist[:, 0].copy_(self.belt_speed)
        self.visual_belt_offset += speed * fixed_dt
        self.provider.coupler.set_proxy_twist_override(
            self.belt_proxy_index, self.belt_twist
        )
        self.force_model.apply(self.belt_speed)
        self.soft_force_model.apply(self.belt_speed)
        self.arm_command.zero_()
        torch.mul(
            self.arm_grip_target,
            self.profile_module.arm_grip_fraction_at_tick(control_tick),
            out=self.arm_command[..., :7],
        )
        self.arm_command[..., :7].lerp_(
            self.arm_push_target,
            self.profile_module.arm_push_fraction_at_tick(control_tick),
        )
        self.arm_command[..., :7].lerp_(
            self.arm_clear_target,
            self.profile_module.arm_clear_fraction_at_tick(control_tick),
        )
        self.arm_command[..., :7].lerp_(
            self.arm_retract_target,
            self.profile_module.arm_retract_fraction_at_tick(control_tick),
        )
        torch.mul(
            self.gripper_offsets,
            self.profile_module.gripper_close_fraction_at_tick(control_tick),
            out=self.arm_command[..., -2:],
        )
        for arm_index, arm_view in enumerate(self.arm_views):
            arm_view.set_controls(self.arm_command[:, arm_index])
        self.tick += 1

    def _physics_process(self, delta: float) -> None:
        del delta
        if self.provider is None or self.play_session is None:
            return
        input_state = getattr(self.context, "input", None)
        if input_state is not None and input_state.is_key_pressed("P"):
            self.play_session.reset()
            return
        if (
            not self.drop_only
            and self.tick >= self.profile_module.CYCLE_TICKS
        ):
            self.play_session.reset()
            return
        if self.tick and self.tick % 30 == 0:
            self._update_status()

    def _process(self, delta: float) -> None:
        del delta

    def _reset_provider(self) -> None:
        if self.provider is None:
            return
        self.belt_speed.zero_()
        self.belt_twist.zero_()
        self.provider.coupler.set_proxy_twist_override(
            self.belt_proxy_index, self.belt_twist
        )
        self.force_model.clear()
        self.soft_force_model.clear()
        self.arm_command.zero_()
        self.provider.reset(
            self.reset_mask,
            qpos=self.initial_qpos,
            qvel=torch.zeros_like(self.provider.arrays["qvel"]),
            ctrl=torch.zeros_like(self.provider.arrays["ctrl"]),
        )
        for arm_index, arm_view in enumerate(self.arm_views):
            arm_view.set_controls(self.arm_command[:, arm_index])
        self.tick = 0
        self.visual_belt_offset = 0.0
        self.last_scene_sync_frame = -1
        self.last_contact_refresh_frame = -1
        self.cached_contact_arrows = []
        clear_debug_arrows()

    def _sync_scene(self) -> None:
        if self.provider is None:
            return
        settings = self.context.get_physics_debug_settings()
        draw_contacts = bool(settings["draw_contact_forces"])
        frame = self.provider.frame
        if draw_contacts != self.contact_arrows_enabled:
            self.contact_arrows_enabled = draw_contacts
            self.last_contact_refresh_frame = -1
            if not draw_contacts:
                self.cached_contact_arrows = []
                clear_debug_arrows()

        scene_due = (
            self.last_scene_sync_frame < 0
            or frame - self.last_scene_sync_frame
            >= self.profile.scene_sync_interval
        )
        contact_due = draw_contacts and (
            self.last_contact_refresh_frame < 0
            or frame - self.last_contact_refresh_frame
            >= self.profile.contact_refresh_interval
        )
        if not scene_due and not contact_due:
            return

        self.provider.refresh_state()
        self.provider.synchronize()
        positions = self.provider.arrays["ipc_positions"].detach().cpu().numpy()
        if scene_due:
            self._sync_render_state(positions)
            self.last_scene_sync_frame = frame
        if contact_due:
            contact_forces = (
                self.provider.arrays["ipc_contact_forces"][0]
                .detach()
                .cpu()
                .numpy()
            )
            external_forces = (
                self.provider.arrays["ipc_external_forces"][0]
                .detach()
                .cpu()
                .numpy()
            )
            arrows = _contact_force_arrows(
                positions[0],
                contact_forces,
                force_scale=float(settings["contact_force_scale"]),
                max_force_length=float(
                    settings["contact_force_max_length"]
                ),
            )
            arrows.extend(
                _body_resultant_force_arrows(
                    positions[0],
                    contact_forces,
                    self.deformable_ranges,
                    SOFT_PACKAGE_NAMES,
                    horizontal_only=True,
                    color=CONTACT_HORIZONTAL_RESULTANT_COLOR,
                    label="horizontal IPC resultant",
                    force_scale=float(settings["contact_force_scale"]),
                    max_force_length=float(
                        settings["contact_force_max_length"]
                    ),
                )
            )
            arrows.extend(
                _body_resultant_force_arrows(
                    positions[0],
                    external_forces,
                    self.deformable_ranges,
                    SOFT_PACKAGE_NAMES,
                    horizontal_only=False,
                    color=EXTERNAL_FORCE_RESULTANT_COLOR,
                    label="belt drive resultant",
                    force_scale=float(settings["contact_force_scale"]),
                    max_force_length=float(
                        settings["contact_force_max_length"]
                    ),
                )
            )
            self.cached_contact_arrows = arrows
            self.last_contact_refresh_frame = frame
        if draw_contacts:
            set_debug_arrows(self.cached_contact_arrows)

    def _sync_render_state(self, positions: Any) -> None:
        for body_index, entry in enumerate(
            self.provider.ipc_solver.deformable_bodies
        ):
            count = int(entry["element_count"])
            offset = int(entry["element_offset"])
            self.deformable_buffer[body_index, :count] = positions[
                0, offset : offset + count
            ]
        self.context.apply_deformable_vertices(
            self.deformable_bodies,
            self.deformable_buffer,
            self.deformable_counts,
        )

        for body, view in zip(self.box_bodies, self.box_views, strict=True):
            pose = view.read_state().link_pose[0, 0].detach().cpu().numpy()
            self.context.apply_link_poses((body,), pose.reshape(1, 7))

        for links, view in zip(self.arm_links, self.arm_views, strict=True):
            poses = view.read_state().link_pose[0].detach().cpu().numpy()
            self.context.apply_link_poses(links, poses)

        half_length = 0.5 * BELT_MARKER_WRAP_LENGTH
        for marker, origin in zip(
            self.belt_markers, self.belt_marker_origins, strict=True
        ):
            x = (
                (origin + self.visual_belt_offset + half_length)
                % BELT_MARKER_WRAP_LENGTH
            ) - half_length
            marker.position = (x, 0.0, 0.021)

    def _update_status(self) -> None:
        if self.play_session is None:
            return
        positions = self.provider.arrays["ipc_positions"][0]
        leading_soft_x = float(positions[:, 0].amax().item())
        drive_force = float(
            self.force_model.drive_force.abs().amax().item()
        )
        soft_drive_force = float(
            self.soft_force_model.drive_force.abs().amax().item()
        )
        phase_tick = self.tick
        if self.drop_only:
            phase_tick = min(
                phase_tick,
                self.profile_module.DROP_SETTLE_TICKS - 1,
            )
        phase = self.profile_module.cycle_phase(phase_tick)
        arrow_state = (
            "contact arrows on"
            if self.contact_arrows_enabled
            else "contact arrows off"
        )
        status_name = (
            "Thin-shell mailer drop test"
            if self.drop_only
            else "Mixed package conveyor"
        )
        self.play_session.set_status(
            status_name
            + " | "
            f"{phase} | belt {float(self.belt_speed[0].item()):.2f} m/s | "
            f"rigid drive peak {drive_force:.1f} N | "
            f"soft drive peak {soft_drive_force:.1f} N | "
            f"soft lead x={leading_soft_x:.2f} m | {self.profile.name} | "
            f"SolverCoupledProxy x{self.profile.coupling_iterations} | "
            + arrow_state
        )

    def _exit_tree(self) -> None:
        self._close_play_session()

    def _close_play_session(self) -> None:
        clear_debug_arrows()
        force_model = self.force_model
        self.force_model = None
        soft_force_model = self.soft_force_model
        self.soft_force_model = None
        if force_model is not None and self.provider is not None:
            try:
                force_model.clear()
            except Exception:
                pass
        if soft_force_model is not None and self.provider is not None:
            try:
                soft_force_model.clear()
            except Exception:
                pass
        play_session = self.play_session
        self.play_session = None
        provider = self.provider
        self.provider = None
        if play_session is not None:
            play_session.close()
        elif provider is not None:
            provider.close()
