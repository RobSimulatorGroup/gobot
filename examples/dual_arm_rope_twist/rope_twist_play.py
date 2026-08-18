"""Editor Play entry point for dual-FR3 rope twisting."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import gobot
# Keep the CUDA runtime load order identical to rope_twist_batch.py. Torch's
# packaged CUDA runtime must be resident before the native libuipc module.
import torch
from gobot.ipc import LibuipcBatchConfig, LibuipcBatchSolver, LibuipcConfig
from gobot.render import DebugArrow, clear_debug_arrows, set_debug_arrows
from gobot.rl import (
    CompiledMuJoCoIpcArtifact,
    MuJoCoIpcConfig,
    MuJoCoIpcProvider,
    MuJoCoWarpContactSensorSpec,
    MuJoCoWarpProvider,
)


SCENE_ROOT_NAME = "dual_arm_rope_twist"
ROBOT_NAMES = ("left_fr3", "right_fr3")
FIXTURE_BODY_NAMES = ("left_rope_fixture", "right_rope_fixture")
TOOL_LINK_NAME = "fr3_link7"
JOINT_NAMES = tuple(f"fr3_joint{index}" for index in range(1, 8)) + (
    "fr3_finger_joint1",
    "fr3_finger_joint2",
)
ROBOT_LINK_NAMES = tuple(f"fr3_link{index}" for index in range(8)) + (
    "fr3_leftfinger",
    "fr3_rightfinger",
)
FIXED_DT = 0.002
NUM_ENVS = 1
ENVIRONMENTS_PER_SHARD = 1
SOLVER_MODULE_NAME = "libgobot_libuipc_solver.so"
DRIVE_MODE_ENVIRONMENT_VARIABLE = "GOBOT_ROPE_TWIST_DRIVE_MODE"
COUPLING_ITERATIONS_ENVIRONMENT_VARIABLE = (
    "GOBOT_ROPE_TWIST_COUPLING_ITERATIONS"
)
QUALITY_ENVIRONMENT_VARIABLE = "GOBOT_ROPE_TWIST_QUALITY"
QUALITY_NAMES = ("interactive", "accurate")
DEFAULT_QUALITY = "interactive"
CONTACT_FORCE_ARROW_MIN_NEWTONS = 1.0e-3
IPC_CONTACT_FORCE_ARROW_MIN_LENGTH = 0.015
IPC_CONTACT_FORCE_ARROW_COLOR = (1.0, 0.12, 0.78, 1.0)
GRIP_CONTACT_FORCE_ARROW_COLOR = (0.16, 0.92, 0.34, 1.0)


@dataclass(frozen=True)
class RopeTwistQualityProfile:
    name: str
    coupling_iterations: int
    relaxation_mode: str
    scene_sync_interval: int
    contact_refresh_interval: int
    newton_max_iterations: int
    line_search_max_iterations: int
    linear_system_tolerance_rate: float


QUALITY_PROFILES = {
    "interactive": RopeTwistQualityProfile(
        name="interactive",
        coupling_iterations=1,
        relaxation_mode="fixed",
        scene_sync_interval=2,
        contact_refresh_interval=4,
        newton_max_iterations=16,
        line_search_max_iterations=8,
        linear_system_tolerance_rate=1.0e-3,
    ),
    "accurate": RopeTwistQualityProfile(
        name="accurate",
        coupling_iterations=2,
        relaxation_mode="aitken",
        scene_sync_interval=1,
        contact_refresh_interval=1,
        newton_max_iterations=16,
        line_search_max_iterations=8,
        linear_system_tolerance_rate=1.0e-3,
    ),
}
COUPLING_ITERATIONS = QUALITY_PROFILES[DEFAULT_QUALITY].coupling_iterations


def _nodes_by_name(root: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    pending = [root]
    while pending:
        node = pending.pop()
        if node.name in result:
            raise RuntimeError(
                f"robot subtree has duplicate node name {node.name!r}"
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
    candidates = [
        repository / "build" / "python" / "gobot" / SOLVER_MODULE_NAME
    ]
    candidates.extend(
        sorted(
            (repository / "build").glob(
                "*/python/gobot/" + SOLVER_MODULE_NAME
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
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
        raise FileNotFoundError(f"rope-twist module does not exist: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load rope-twist module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _batch_config(
    context: Any,
    profile: RopeTwistQualityProfile | None = None,
) -> LibuipcBatchConfig:
    profile = profile or _quality_profile()
    return LibuipcBatchConfig(
        solver=LibuipcConfig(
            fixed_time_step=FIXED_DT,
            gravity=(0.0, 0.0, -9.81),
            friction_coefficient=1.25,
            contact_activation_distance=8.0e-4,
            contact_resistance=1.0e7,
            affine_stiffness=1.0e8,
            module_path=_solver_module_path(context.project_path),
            workspace=str(
                Path(tempfile.gettempdir()) / "gobot-dual-arm-rope-editor"
            ),
        ),
        environments_per_shard=ENVIRONMENTS_PER_SHARD,
        newton_max_iterations=profile.newton_max_iterations,
        line_search_max_iterations=profile.line_search_max_iterations,
        linear_system_tolerance_rate=profile.linear_system_tolerance_rate,
        # Interactive rendering reads these buffers only at its display cadence.
        export_deformable_state=profile.scene_sync_interval == 1,
        export_affine_state=profile.scene_sync_interval == 1,
        # Contact-force visualization is a debug output. Both editor profiles
        # refresh it on demand when the Physics-panel flag is enabled.
        export_deformable_contact_forces=False,
    )


def _drive_mode(controllers: Any) -> tuple[str, float, bool]:
    mode = os.environ.get(
        DRIVE_MODE_ENVIRONMENT_VARIABLE, controllers.SHOWCASE_DRIVE_MODE
    ).strip().lower()
    torque_limit = controllers.wrist_drive_torque_limit(mode)
    return (
        mode,
        float(torque_limit),
        mode == controllers.FINITE_TORQUE_DRIVE_MODE,
    )


def _quality_profile() -> RopeTwistQualityProfile:
    quality = os.environ.get(
        QUALITY_ENVIRONMENT_VARIABLE, DEFAULT_QUALITY
    ).strip().lower()
    if quality not in QUALITY_PROFILES:
        raise ValueError(
            f"{QUALITY_ENVIRONMENT_VARIABLE} must be one of "
            + ", ".join(repr(value) for value in QUALITY_NAMES)
        )
    return QUALITY_PROFILES[quality]


def _coupling_iterations(
    profile: RopeTwistQualityProfile | None = None,
) -> int:
    profile = profile or _quality_profile()
    value = os.environ.get(
        COUPLING_ITERATIONS_ENVIRONMENT_VARIABLE,
        str(profile.coupling_iterations),
    ).strip()
    try:
        iterations = int(value)
    except ValueError as error:
        raise ValueError(
            f"{COUPLING_ITERATIONS_ENVIRONMENT_VARIABLE} must be an integer"
        ) from error
    if iterations < 1:
        raise ValueError(
            f"{COUPLING_ITERATIONS_ENVIRONMENT_VARIABLE} must be at least 1"
        )
    return iterations


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
            fields=("found", "force", "pos", "normal", "tangent"),
            reduce="maxforce",
            num_slots=1,
        )
        for side, robot_name, fixture_name in zip(
            ("left", "right"),
            ROBOT_NAMES,
            FIXTURE_BODY_NAMES,
            strict=True,
        )
    )


def _contact_force_arrows(
    positions: Any,
    forces: Any,
    *,
    found: Any | None = None,
    color: tuple[float, float, float, float],
    label: str,
    force_scale: float,
    max_force_length: float,
    min_force_length: float = 0.0,
    max_count: int | None = None,
) -> list[DebugArrow]:
    import numpy as np

    points = np.asarray(positions, dtype=np.float64)
    values = np.asarray(forces, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or values.shape != points.shape
    ):
        raise RuntimeError("contact-force arrays must have shape [count,3]")
    if not np.isfinite(points).all() or not np.isfinite(values).all():
        raise RuntimeError("contact-force arrays contain non-finite values")
    if not math.isfinite(force_scale) or force_scale < 0.0:
        raise ValueError("contact-force arrow scale must be finite and non-negative")
    if not math.isfinite(max_force_length) or max_force_length < 0.0:
        raise ValueError(
            "contact-force arrow maximum length must be finite and non-negative"
        )
    if not math.isfinite(min_force_length) or min_force_length < 0.0:
        raise ValueError(
            "contact-force arrow minimum length must be finite and non-negative"
        )
    if max_count is not None and (
        isinstance(max_count, bool) or int(max_count) <= 0
    ):
        raise ValueError("contact-force arrow maximum count must be positive")

    visible = np.ones(points.shape[0], dtype=bool)
    if found is not None:
        matches = np.asarray(found)
        if matches.shape != (points.shape[0],):
            raise RuntimeError("contact-found array must have shape [count]")
        visible &= matches.astype(bool)

    magnitudes = np.linalg.norm(values, axis=1)
    indices = np.flatnonzero(
        visible & (magnitudes >= CONTACT_FORCE_ARROW_MIN_NEWTONS)
    )
    if max_count is not None and len(indices) > int(max_count):
        selected = np.argpartition(
            magnitudes[indices], -int(max_count)
        )[-int(max_count) :]
        indices = indices[selected]
    indices = indices[
        np.argsort(-magnitudes[indices], kind="stable")
    ]

    arrows = []
    for index in indices:
        magnitude = float(magnitudes[index])
        length = min(
            max_force_length,
            max(min_force_length, force_scale * math.log1p(magnitude)),
        )
        if length <= 0.0:
            continue
        arrows.append(
            DebugArrow(
                start=points[index],
                vector=values[index] / magnitude,
                color=color,
                scale=length,
                label=f"{label} {magnitude:.3g} N",
            )
        )
    return arrows


def _contact_frame_forces_to_world(
    forces: Any,
    normals: Any,
    tangents: Any,
) -> Any:
    """Transform MuJoCo contact-frame force components into world vectors."""

    torch_module = __import__("torch")
    if (
        forces.shape != normals.shape
        or forces.shape != tangents.shape
        or forces.ndim != 2
        or forces.shape[1] != 3
    ):
        raise RuntimeError("contact force/frame tensors must have shape [count,3]")
    bitangents = torch_module.cross(normals, tangents, dim=1)
    return (
        forces[:, 0, None] * normals
        + forces[:, 1, None] * tangents
        + forces[:, 2, None] * bitangents
    )


def _torque_arrows(tool_poses: Any, wrenches: Any) -> list[DebugArrow]:
    arrows = []
    colors = ((0.98, 0.72, 0.12, 1.0), (0.12, 0.82, 0.88, 1.0))
    for index, (pose, wrench) in enumerate(zip(tool_poses, wrenches, strict=True)):
        torque = wrench[3:]
        magnitude = float(sum(float(value) ** 2 for value in torque) ** 0.5)
        if not math.isfinite(magnitude) or magnitude < 0.002:
            continue
        direction = tuple(float(value) / magnitude for value in torque)
        arrows.append(
            DebugArrow(
                start=tuple(float(value) for value in pose[:3]),
                vector=direction,
                color=colors[index],
                scale=min(0.18, 0.045 + 0.050 * math.log1p(magnitude * 8.0)),
                label=f"wrist reaction {magnitude:.2f} N m",
            )
        )
    return arrows


class Script(gobot.NodeScript):
    """Run and render one complete dual-arm twisting cycle."""

    def _ready(self) -> None:
        self.provider = None
        self.play_session = None
        self.robot_views = ()
        self.robot_links = ()
        self.fixture_views = ()
        self.fixture_bodies = ()
        self.deformable_bodies = ()
        self.deformable_counts = ()
        self.deformable_buffer = None
        self.controller = None
        self.reset_mask = None
        self.tool_body_ids = ()
        self.fixture_body_ids = ()
        self.fixture_proxy_indices = ()
        self.wrist_actuator_ids = ()
        self.grip_contact_sensors = ()
        self.last_scene_sync_frame = -1
        self.last_contact_refresh_frame = -1
        self.contact_arrows_enabled = False
        self.cached_torque_arrows = []
        self.cached_contact_arrows = []
        self.endpoint_indices = ()
        self.initial_qpos = None
        self.attachment_reference = None
        self.grip_position_reference = None
        self.grip_rotation_reference = None
        self.maximum_grip_slip = 0.0
        self.maximum_attachment_error = 0.0
        self.gravity_compensator = None
        self.drive_mode = "showcase"
        self.quality_profile = QUALITY_PROFILES[DEFAULT_QUALITY]
        self.coupling_iterations = self.quality_profile.coupling_iterations
        self.wrist_torque_limit = 0.0
        try:
            import numpy as np

            root = self.get_root()
            if root is None or root.name != SCENE_ROOT_NAME:
                raise RuntimeError("unexpected dual-arm rope scene root")
            robot_roots = tuple(root.find(name) for name in ROBOT_NAMES)
            if any(robot is None for robot in robot_roots):
                raise RuntimeError("dual-arm rope scene is missing an FR3")
            self.robot_links = tuple(
                tuple(_nodes_by_name(robot)[name] for name in ROBOT_LINK_NAMES)
                for robot in robot_roots
            )
            fixture_bodies = tuple(
                root.find(name) for name in FIXTURE_BODY_NAMES
            )
            if any(fixture is None for fixture in fixture_bodies):
                raise RuntimeError("dual-arm rope scene is missing a fixture")
            self.fixture_bodies = fixture_bodies

            controllers = _load_project_module(
                self.context.project_path,
                "controllers.py",
                "gobot_dual_arm_rope_runtime_controllers",
            )
            self.controllers_module = controllers
            layout = controllers.stall_detection_layout(NUM_ENVS)
            (
                self.drive_mode,
                self.wrist_torque_limit,
                stall_detection_enabled,
            ) = _drive_mode(controllers)
            self.quality_profile = _quality_profile()
            self.coupling_iterations = _coupling_iterations(self.quality_profile)

            solver_config = _batch_config(self.context, self.quality_profile)
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
            settings["impedance_ratio"] = 10.0
            self.context.set_mujoco_solver_settings(settings)
            artifact = CompiledMuJoCoIpcArtifact.from_context(self.context)
            self.provider = MuJoCoIpcProvider(
                artifact,
                config=MuJoCoIpcConfig(
                    num_envs=NUM_ENVS,
                    device="cuda:0",
                    environments_per_shard=ENVIRONMENTS_PER_SHARD,
                    coupling_iterations=self.coupling_iterations,
                    relaxation_mode=self.quality_profile.relaxation_mode,
                    relaxation_factor=1.0,
                    capture_mujoco_graphs=True,
                    capture_coupler_graphs=True,
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
            self.robot_views = tuple(
                self.provider.create_robot_view(
                    robot_name=robot_name,
                    base_link="fr3_link0",
                    joint_names=JOINT_NAMES,
                    link_names=ROBOT_LINK_NAMES,
                )
                for robot_name in ROBOT_NAMES
            )
            self.fixture_views = tuple(
                self.provider.create_robot_view(
                    robot_name=fixture_name,
                    base_link=fixture_name,
                    joint_names=(),
                    link_names=(fixture_name,),
                )
                for fixture_name in FIXTURE_BODY_NAMES
            )
            mappings = tuple(
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
                for mapping in mappings
            )
            self.fixture_proxy_indices = tuple(
                mapping.ipc_body_index for mapping in mappings
            )
            self.wrist_actuator_ids = tuple(
                self.provider.rigid_solver.resolve_robot_layout(
                    robot_name,
                    base_link="fr3_link0",
                    joint_names=JOINT_NAMES,
                ).actuator_ids[controllers.WRIST_INDEX]
                for robot_name in ROBOT_NAMES
            )
            self.grip_contact_sensors = tuple(
                self.provider.rigid_solver.contact_sensor(
                    f"{side}_fixture_grip"
                )
                for side in ("left", "right")
            )
            controllers.configure_wrist_torque_limit(
                self.provider.rigid_solver,
                self.wrist_actuator_ids,
                self.wrist_torque_limit,
            )

            deformable_entries = tuple(self.provider.ipc_solver.deformable_bodies)
            bodies = []
            counts = []
            for entry in deformable_entries:
                name = str(entry["path"]).rsplit("/", 1)[-1]
                body = root.find(name)
                if body is None or body.type_name != "DeformableBody3D":
                    raise RuntimeError(
                        f"rope scene is missing deformable strand {name!r}"
                    )
                bodies.append(body)
                counts.append(int(entry["element_count"]))
            self.deformable_bodies = tuple(bodies)
            self.deformable_counts = tuple(counts)
            self.deformable_buffer = np.zeros(
                (len(bodies), max(counts), 3), dtype=np.float32
            )
            self.endpoint_indices = controllers.rope_endpoint_index_sets(
                deformable_entries,
                self.provider.arrays["ipc_positions"].device,
            )
            self.initial_qpos = self.provider.arrays["qpos"].clone()
            self.attachment_reference = (
                controllers.rope_endpoints_in_affine_frames(
                    self.provider.arrays["ipc_positions"],
                    self.endpoint_indices,
                    self.provider.arrays["ipc_affine_targets"],
                    self.fixture_proxy_indices,
                ).clone()
            )
            self.reset_mask = torch.ones(
                NUM_ENVS,
                dtype=torch.bool,
                device=self.provider.arrays["ctrl"].device,
            )
            command_template = torch.zeros(
                (NUM_ENVS, 2, len(JOINT_NAMES)),
                dtype=self.provider.arrays["ctrl"].dtype,
                device=self.provider.arrays["ctrl"].device,
            )
            self.controller = controllers.BatchedTwistController(
                command_template,
                layout,
                fixed_dt=FIXED_DT,
                feedback_enabled=stall_detection_enabled,
                drive_torque_limit=self.wrist_torque_limit,
            )
            self.gravity_compensator = controllers.BatchedGravityCompensator(
                controllers.gravity_compensation_schedule(
                    artifact.mujoco.content, ROBOT_NAMES
                ),
                self.provider.arrays["qfrc_applied"],
            )

            self._reset_provider()
            self._sync_scene()
            self.play_session = gobot.sim.ProviderPlaySession(
                self.context,
                self.provider,
                fixed_dt=FIXED_DT,
                max_sub_steps=1,
                before_step=self._before_step,
                reset=self._reset_provider,
                sync_scene=self._sync_scene,
            ).start()
            self._update_status()
            print(
                "Dual FR3 rope twisting started: MuJoCo friction grasps two "
                "free fixtures; libuipc advances the attached rope on cuda:0; "
                f"quality={self.quality_profile.name}; "
                f"drive={self.drive_mode} ({self.wrist_torque_limit:g} N m); "
                f"coupling=SolverCoupledProxy x{self.coupling_iterations}"
            )
        except Exception:
            self._close_play_session()
            raise

    def _before_step(self, fixed_dt: float) -> None:
        del fixed_dt
        import torch

        applied_wrenches = self.controllers_module.fixture_wrenches_in_tool_frames(
            self.provider.arrays,
            self.fixture_body_ids,
            self.tool_body_ids,
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

    def _physics_process(self, delta: float) -> None:
        del delta
        if self.provider is None or self.play_session is None:
            return
        input_state = getattr(self.context, "input", None)
        if input_state is not None and input_state.is_key_pressed("P"):
            self.play_session.reset()
            return
        if self.controller.cycle_complete:
            self.play_session.reset()
            return
        if self.controller.tick and self.controller.tick % 20 == 0:
            self._update_status()

    def _process(self, delta: float) -> None:
        del delta

    def _reset_provider(self) -> None:
        if self.provider is None:
            return
        command = self.controller.reset().clone()
        import torch

        self.provider.reset(
            self.reset_mask,
            qpos=self.initial_qpos,
            qvel=torch.zeros_like(self.provider.arrays["qvel"]),
            ctrl=torch.zeros_like(self.provider.arrays["ctrl"]),
        )
        for robot_index, robot_view in enumerate(self.robot_views):
            robot_view.set_controls(command[:, robot_index])
        initial_states = tuple(view.read_state() for view in self.robot_views)
        initial_joint_positions = torch.stack(
            tuple(state.joint_position for state in initial_states), dim=1
        )
        self.gravity_compensator.apply(
            self.provider.arrays["qfrc_applied"],
            initial_joint_positions,
            0,
        )
        self.grip_position_reference = None
        self.grip_rotation_reference = None
        self.maximum_grip_slip = 0.0
        self.maximum_attachment_error = 0.0
        self.last_scene_sync_frame = -1
        self.last_contact_refresh_frame = -1
        self.cached_contact_arrows = []

    def _sync_scene(self) -> None:
        if self.provider is None:
            return
        settings = self.context.get_physics_debug_settings()
        draw_contact_forces = bool(settings["draw_contact_forces"])
        frame = self.provider.frame
        if draw_contact_forces != self.contact_arrows_enabled:
            self.contact_arrows_enabled = draw_contact_forces
            self.last_contact_refresh_frame = -1
            if not draw_contact_forces:
                self.cached_contact_arrows = []
                set_debug_arrows(self.cached_torque_arrows)

        scene_due = (
            self.last_scene_sync_frame < 0
            or frame - self.last_scene_sync_frame
            >= self.quality_profile.scene_sync_interval
        )
        contact_due = draw_contact_forces and (
            self.last_contact_refresh_frame < 0
            or frame - self.last_contact_refresh_frame
            >= self.quality_profile.contact_refresh_interval
        )
        if not scene_due and not contact_due:
            return

        self.provider.refresh_state()
        if contact_due:
            self.provider.refresh_deformable_contact_forces()
            self.provider.sense()
        self.provider.synchronize()

        positions = self.provider.arrays["ipc_positions"].detach().cpu().numpy()
        if scene_due:
            self._sync_render_state(positions)
            self.last_scene_sync_frame = frame
        if contact_due:
            self._refresh_contact_arrows(settings, positions)
            self.last_contact_refresh_frame = frame
        set_debug_arrows(
            self.cached_torque_arrows
            + (self.cached_contact_arrows if draw_contact_forces else [])
        )

    def _sync_render_state(self, positions: Any) -> None:
        for body_index, entry in enumerate(self.provider.ipc_solver.deformable_bodies):
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

        tool_poses = []
        for robot_index, robot_view in enumerate(self.robot_views):
            poses = robot_view.read_state().link_pose[0].detach().cpu().numpy()
            self.context.apply_link_poses(self.robot_links[robot_index], poses)
            tool_poses.append(poses[ROBOT_LINK_NAMES.index(TOOL_LINK_NAME)])
        for fixture_index, fixture_view in enumerate(self.fixture_views):
            fixture_pose = (
                fixture_view.read_state().link_pose[0, 0].detach().cpu().numpy()
            )
            self.context.apply_link_poses(
                (self.fixture_bodies[fixture_index],),
                fixture_pose.reshape(1, 7),
            )

        aggregate = self.controllers_module.fixture_wrenches_in_tool_frames(
            self.provider.arrays,
            self.fixture_body_ids,
            self.tool_body_ids,
        )
        local_wrenches = aggregate[0]
        tool_rotations = self.provider.arrays["xmat"][
            0, list(self.tool_body_ids)
        ].reshape(2, 3, 3)
        world_wrenches = local_wrenches.clone()
        world_wrenches[:, :3] = (
            tool_rotations @ local_wrenches[:, :3, None]
        )[..., 0]
        world_wrenches[:, 3:] = (
            tool_rotations @ local_wrenches[:, 3:, None]
        )[..., 0]
        self.cached_torque_arrows = _torque_arrows(
            tool_poses, world_wrenches.detach().cpu().numpy()
        )
        self._update_physical_debug_metrics()

    def _refresh_contact_arrows(self, settings: Any, positions: Any) -> None:
        import torch

        arrows = []
        force_scale = float(settings["contact_force_scale"])
        max_force_length = float(settings["contact_force_max_length"])
        arrows.extend(
            _contact_force_arrows(
                positions[0],
                self.provider.arrays["ipc_contact_forces"][0]
                .detach()
                .cpu()
                .numpy(),
                color=IPC_CONTACT_FORCE_ARROW_COLOR,
                label="rope contact",
                force_scale=force_scale,
                max_force_length=max_force_length,
                min_force_length=IPC_CONTACT_FORCE_ARROW_MIN_LENGTH,
            )
        )
        grip_positions = torch.cat(
            tuple(sensor["pos"] for sensor in self.grip_contact_sensors),
            dim=1,
        )[0]
        grip_forces = torch.cat(
            tuple(sensor["force"] for sensor in self.grip_contact_sensors),
            dim=1,
        )[0]
        grip_normals = torch.cat(
            tuple(sensor["normal"] for sensor in self.grip_contact_sensors),
            dim=1,
        )[0]
        grip_tangents = torch.cat(
            tuple(sensor["tangent"] for sensor in self.grip_contact_sensors),
            dim=1,
        )[0]
        grip_forces = _contact_frame_forces_to_world(
            grip_forces,
            grip_normals,
            grip_tangents,
        )
        grip_found = torch.cat(
            tuple(sensor["found"] for sensor in self.grip_contact_sensors),
            dim=1,
        )[0]
        arrows.extend(
            _contact_force_arrows(
                grip_positions.detach().cpu().numpy(),
                grip_forces.detach().cpu().numpy(),
                found=grip_found.detach().cpu().numpy(),
                color=GRIP_CONTACT_FORCE_ARROW_COLOR,
                label="grip contact",
                force_scale=force_scale,
                max_force_length=max_force_length,
                max_count=4,
            )
        )
        self.cached_contact_arrows = arrows

    def _update_physical_debug_metrics(self) -> None:
        if self.controller.tick < self.controllers_module.TWIST_START_TICK:
            return

        local_endpoints = self.controllers_module.rope_endpoints_in_affine_frames(
            self.provider.arrays["ipc_positions"],
            self.endpoint_indices,
            self.provider.arrays["ipc_affine_targets"],
            self.fixture_proxy_indices,
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
                self.fixture_body_ids,
                self.tool_body_ids,
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

    def _update_status(self) -> None:
        if self.play_session is None:
            return
        winding = self.controllers_module.rope_winding_turns(
            self.provider.arrays["ipc_positions"],
            self.provider.ipc_solver.deformable_bodies,
        )
        turns = float(winding[0].abs().mean().item())
        peak = float(self.controller.peak_axial_torque[0].item())
        relative_turns = float(
            self.controller.peak_actual_relative_rotation[0].item()
            / (2.0 * math.pi)
        )
        wrist_speed = self.controller.filtered_wrist_speed[0]
        wrist_effort = self.controller.filtered_wrist_effort[0]
        stalled = bool(self.controller.stalled[0].item())
        self.play_session.set_status(
            "Dual FR3 rope twist | "
            + self.controller.phase
            + f" | {self.drive_mode} {self.wrist_torque_limit:g} N m"
            + f" | {self.quality_profile.name}"
            + f" | SolverCoupledProxy x{self.coupling_iterations}"
            + f" | relative {relative_turns:.2f} turns | rope winding "
            + f"{turns:.2f} | wrist speed {float(wrist_speed[0]):.2f}/"
            + f"{float(wrist_speed[1]):.2f} rad/s | drive "
            + f"{float(wrist_effort[0]):.3f}/"
            + f"{float(wrist_effort[1]):.3f} N m | "
            + ("STALLED | " if stalled else "")
            + "fixture slip "
            + f"{1000.0 * self.maximum_grip_slip:.1f} mm | mount error "
            + f"{1000.0 * self.maximum_attachment_error:.1f} mm | "
            + f"rope reaction peak {peak:.3f} N m"
        )

    def _exit_tree(self) -> None:
        self._close_play_session()

    def _close_play_session(self) -> None:
        clear_debug_arrows()
        play_session = self.play_session
        self.play_session = None
        provider = self.provider
        self.provider = None
        if play_session is not None:
            play_session.close()
        elif provider is not None:
            provider.close()
