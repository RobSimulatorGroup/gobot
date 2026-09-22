"""Data-only configuration for the dual-FR3 rope example."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from gobot.ipc import LibuipcBatchConfig, LibuipcConfig
from gobot.sim.providers import MuJoCoWarpContactSensorSpec


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
    export_state_each_step: bool
    newton_max_iterations: int
    line_search_max_iterations: int
    linear_system_tolerance_rate: float


QUALITY_PROFILES = {
    "interactive": RopeTwistQualityProfile(
        name="interactive",
        coupling_iterations=1,
        relaxation_mode="fixed",
        export_state_each_step=False,
        newton_max_iterations=16,
        line_search_max_iterations=8,
        linear_system_tolerance_rate=1.0e-3,
    ),
    "accurate": RopeTwistQualityProfile(
        name="accurate",
        coupling_iterations=2,
        relaxation_mode="aitken",
        export_state_each_step=True,
        newton_max_iterations=16,
        line_search_max_iterations=8,
        linear_system_tolerance_rate=1.0e-3,
    ),
}
COUPLING_ITERATIONS = QUALITY_PROFILES[DEFAULT_QUALITY].coupling_iterations
SCENE_FIELDS = tuple(f"robot.{name}.link_pose" for name in (*ROBOT_NAMES, *FIXTURE_BODY_NAMES))
BASE_FIELDS = (*SCENE_FIELDS, "deformable.local_vertices", "rope.metrics", "rope.wrenches")
CONTACT_FIELDS = ("rope.contact_positions", "rope.contact_forces", "grip.positions", "grip.forces", "grip.found")


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


def _solver_module_path(project_path: str) -> str:
    del project_path
    configured = os.environ.get("GOBOT_LIBUIPC_SOLVER_MODULE", "").strip()
    return str(Path(configured).expanduser().resolve()) if configured else ""


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
        strict_convergence=profile.name == "accurate",
        # Interactive rendering reads these buffers only at its display cadence.
        export_deformable_state=profile.export_state_each_step,
        export_affine_state=profile.export_state_each_step,
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
