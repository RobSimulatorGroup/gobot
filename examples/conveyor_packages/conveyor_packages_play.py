"""Editor Play controller for the native SuperDex conveyor workcell."""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

import gobot
from gobot.render import DebugArrow, clear_debug_arrows, set_debug_arrows


SCENE_ROOT_NAME = "conveyor_packages"
BELT_ROBOT_NAME = "conveyor"
BELT_LINK_NAME = "belt_surface"
RIGID_BOX_NAMES = ("carton_small", "carton_wide", "carton_tall")
RIGID_BOX_MASSES = (0.62, 0.84, 0.76)
SOFT_PACKAGE_NAMES = (
    "soft_mailer_blue",
    "soft_mailer_blue_fill",
    "soft_pouch_yellow",
    "soft_pouch_yellow_fill",
)
SOFT_PACKAGE_RESULTANT_NAMES = (
    "soft_mailer_blue",
    "soft_pouch_yellow",
)
SOFT_PACKAGE_RESULTANT_BODY_GROUPS = ((0, 1), (2, 3))
HAND_SIDES = ("left", "right")
LEAP_ROBOT_NAMES = tuple(f"leap_{side}" for side in HAND_SIDES)
LEAP_FINGER_JOINT_NAMES = (
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
)
HAND_STAGE_DOF_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
HAND_STAGE_JOINT_NAMES_BY_SIDE = tuple(
    tuple(f"leap_{side}_wrist_{name}" for name in HAND_STAGE_DOF_NAMES)
    for side in HAND_SIDES
)
HAND_JOINT_NAMES_BY_SIDE = tuple(
    stage_names + LEAP_FINGER_JOINT_NAMES
    for stage_names in HAND_STAGE_JOINT_NAMES_BY_SIDE
)
BELT_DRIVE_FRICTION = 0.92
BELT_MARKER_WRAP_LENGTH = 2.70
BELT_CENTER_X = 0.0
BELT_CENTER_Y = 0.58
BELT_HALF_WIDTH = 0.36
BELT_TOP_Z = 0.56
CONTACT_FORCE_ARROW_MIN_NEWTONS = 1.0e-3
CONTACT_FORCE_ARROW_MIN_LENGTH = 0.012
CONTACT_FORCE_ARROW_COLOR = (1.0, 0.12, 0.68, 1.0)
EXTERNAL_FORCE_RESULTANT_COLOR = (1.0, 0.58, 0.08, 1.0)
PREVIEW_ENVIRONMENT_VARIABLE = "GOBOT_CONVEYOR_PREVIEW"
PREVIEW_SOFT_PACKAGE_CELLS = {
    "soft_mailer_blue": (12, 9),
    "soft_mailer_blue_fill": (5, 3, 2),
    "soft_pouch_yellow": (13, 10),
    "soft_pouch_yellow_fill": (6, 4, 3),
}
PREVIEW_NEWTON_ITERATIONS = 16
PREVIEW_LINE_SEARCH_ITERATIONS = 6
PREVIEW_PENETRATION_LIMIT_METERS = 1.0e-3
PREVIEW_PENETRATION_HOLD_STEPS = 3
PREVIEW_PENETRATION_LIMIT_PHASES = frozenset(
    {
        "blue_flip_contact",
        "blue_flip_grip",
        "blue_flip_stabilize",
        "blue_flip_rotate",
        "blue_flip_place",
        "blue_flip_turnover",
        "yellow_flip_contact",
        "yellow_flip_grip",
        "yellow_flip_stabilize",
        "yellow_flip_rotate",
        "yellow_flip_place",
    }
)


def _nodes_by_name(
    root: Any, *, allow_duplicate_names: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    pending = [root]
    while pending:
        node = pending.pop()
        if node.name in result:
            if allow_duplicate_names:
                pending.extend(node.children)
                continue
            raise RuntimeError(
                f"conveyor scene has duplicate node name {node.name!r}"
            )
        result[node.name] = node
        pending.extend(node.children)
    return result


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


def _environment_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _apply_preview_deformable_meshes(
    nodes: dict[str, Any], builder_module: Any
) -> int:
    specs = {
        str(spec["name"]): spec
        for spec in builder_module.SOFT_PACKAGE_SPECS
    }
    total_nodes = 0
    for name, cells in PREVIEW_SOFT_PACKAGE_CELLS.items():
        body = nodes[name]
        spec = specs[name]
        if spec.get("model", "volumetric") == "thin_shell":
            mesh = builder_module._soft_mailer_shell_mesh(
                spec["size"], cells
            )
            body.surface_mesh = mesh
            body.self_collision_enabled = False
        else:
            mesh = builder_module._soft_package_mesh(
                spec["size"],
                cells,
                side_rounding=float(spec.get("side_rounding", 0.0)),
            )
            body.mesh = mesh
        total_nodes += len(mesh.vertices)
    return total_nodes


def _force_arrow(
    start: Any,
    force: Any,
    *,
    color: tuple[float, float, float, float],
    label: str,
    force_scale: float,
    max_force_length: float,
) -> DebugArrow | None:
    vector = np.asarray(force, dtype=np.float64)
    magnitude = float(np.linalg.norm(vector))
    if magnitude < CONTACT_FORCE_ARROW_MIN_NEWTONS:
        return None
    length = min(
        max_force_length,
        max(CONTACT_FORCE_ARROW_MIN_LENGTH, force_scale * math.log1p(magnitude)),
    )
    return DebugArrow(
        start=np.asarray(start, dtype=np.float64),
        vector=vector / magnitude,
        color=color,
        scale=length,
        label=f"{label} {magnitude:.3g} N",
    )


def _contact_arrows(
    state: dict[str, Any],
    *,
    force_scale: float,
    max_force_length: float,
) -> list[DebugArrow]:
    arrows: list[DebugArrow] = []
    for contact in state["contacts"]:
        arrow = _force_arrow(
            contact["position"],
            contact["force"],
            color=CONTACT_FORCE_ARROW_COLOR,
            label="contact",
            force_scale=force_scale,
            max_force_length=max_force_length,
        )
        if arrow is not None:
            arrows.append(arrow)
    return arrows


def _max_contact_penetration(state: dict[str, Any]) -> float:
    return max(
        (
            max(0.0, -float(contact["distance"]))
            for contact in state["contacts"]
        ),
        default=0.0,
    )


def _soft_package_contact_links(state: dict[str, Any]) -> str:
    links: dict[str, set[str]] = {
        robot_name: set() for robot_name in LEAP_ROBOT_NAMES
    }
    soft_names = set(SOFT_PACKAGE_NAMES)
    for contact in state["contacts"]:
        endpoints = (
            (str(contact["robot_name"]), str(contact["link_name"])),
            (
                str(contact["other_robot_name"]),
                str(contact["other_link_name"]),
            ),
        )
        for hand, other in (endpoints, endpoints[::-1]):
            if hand[0] in links and (
                other[0] in soft_names or other[1] in soft_names
            ):
                links[hand[0]].add(hand[1])
    groups = [
        f"{robot_name}=[{','.join(sorted(names))}]"
        for robot_name, names in links.items()
        if names
    ]
    return ";".join(groups) if groups else "none"


class Script(gobot.NodeScript):
    """Run one repeating native rigid/deformable package cycle."""

    def _ready(self) -> None:
        self.profile_module = None
        self.forces_module = None
        self.profile = None
        self.hand_joints: tuple[tuple[Any, ...], ...] = ()
        self.force_model = None
        self.soft_force_model = None
        self.belt_markers: tuple[Any, ...] = ()
        self.belt_marker_origins: tuple[float, ...] = ()
        self.tick = 0
        self.initial_tick = 0
        self.peak_contact_penetration = 0.0
        self.penetration_hold_steps = 0
        self.visual_belt_offset = 0.0
        self.drop_only = _environment_flag("GOBOT_CONVEYOR_DROP_ONLY")
        self.preview = _environment_flag(PREVIEW_ENVIRONMENT_VARIABLE)

        root = self.get_root()
        if root is None or root.name != SCENE_ROOT_NAME:
            raise RuntimeError("unexpected conveyor packages scene root")
        nodes = _nodes_by_name(root, allow_duplicate_names=True)
        preview_node_count = 0
        if self.preview:
            builder_module = _load_project_module(
                self.context.project_path,
                "build_scene.py",
                "gobot_conveyor_packages_preview_builder",
            )
            preview_node_count = _apply_preview_deformable_meshes(
                nodes, builder_module
            )
        hand_nodes = tuple(
            _nodes_by_name(nodes[name]) for name in LEAP_ROBOT_NAMES
        )
        self.hand_joints = tuple(
            tuple(side_nodes[name] for name in joint_names)
            for side_nodes, joint_names in zip(
                hand_nodes, HAND_JOINT_NAMES_BY_SIDE, strict=True
            )
        )
        self.belt_markers = tuple(
            node
            for name, node in sorted(nodes.items())
            if name.startswith("belt_marker_")
        )
        self.belt_marker_origins = tuple(
            float(marker.position[0]) for marker in self.belt_markers
        )

        self.profile_module = _load_project_module(
            self.context.project_path,
            "conveyor_profile.py",
            "gobot_conveyor_packages_runtime_profile",
        )
        self.forces_module = _load_project_module(
            self.context.project_path,
            "conveyor_forces.py",
            "gobot_conveyor_packages_force_model",
        )
        self.profile = self.profile_module.quality_profile()
        if self.preview and not self.drop_only:
            self.initial_tick = self.profile_module.DROP_SETTLE_TICKS
            self.tick = self.initial_tick
        fixed_dt = float(self.profile_module.FIXED_DT)
        self.context.fixed_time_step = fixed_dt
        self.context.max_sub_steps = 1
        self.context.backend_type = gobot.PhysicsBackendType.SuperDex
        settings = self.context.get_superdex_solver_settings()
        settings.update(
            {
                "execution_mode": gobot.SuperDexExecutionMode.Cpu,
                "linear_solver": gobot.SuperDexLinearSolver.Auto,
                "newton_iterations": (
                    PREVIEW_NEWTON_ITERATIONS
                    if self.preview
                    else self.profile.newton_max_iterations
                ),
                "line_search_iterations": (
                    PREVIEW_LINE_SEARCH_ITERATIONS
                    if self.preview
                    else self.profile.line_search_max_iterations
                ),
                "linear_iterations": -1,
                "substeps": 1,
                "record_deformable_contact_forces": not self.preview,
            }
        )
        self.context.set_superdex_solver_settings(settings)

        self.force_model = self.forces_module.ConveyorForceModel(
            self.context,
            RIGID_BOX_NAMES,
            RIGID_BOX_MASSES,
            belt_robot=BELT_ROBOT_NAME,
            belt_link=BELT_LINK_NAME,
            friction_coefficient=BELT_DRIVE_FRICTION,
            fixed_dt=fixed_dt,
        )
        self.soft_force_model = self.forces_module.DeformableConveyorForceModel(
            self.context,
            SOFT_PACKAGE_NAMES,
            self.profile_module.SOFT_PACKAGE_MASSES,
            friction_coefficient=BELT_DRIVE_FRICTION,
            fixed_dt=fixed_dt,
            belt_half_length=0.5 * BELT_MARKER_WRAP_LENGTH,
            belt_half_width=BELT_HALF_WIDTH,
            belt_top=BELT_TOP_Z,
            belt_center_x=BELT_CENTER_X,
            belt_center_y=BELT_CENTER_Y,
            velocity_damping_rates=self.profile_module.SOFT_PACKAGE_DAMPING_RATES,
        )
        mode = (
            f"preview ({preview_node_count} deformable nodes, shell "
            "self-contact disabled)"
            if self.preview
            else "validation"
        )
        print(
            "Native SuperDex conveyor initialized: one rigid/articulated/"
            f"deformable contact scene on CPU, {mode} mode"
        )

    def _apply_hand_targets(self, control_tick: int) -> None:
        controls = self.profile_module.hand_controls_at_tick(control_tick)
        for joints, targets in zip(self.hand_joints, controls, strict=True):
            for joint, target in zip(joints, targets, strict=True):
                joint.set_position_target(float(target))

    def _move_belt_markers(self, speed: float, fixed_dt: float) -> None:
        self.visual_belt_offset += speed * fixed_dt
        half_length = 0.5 * BELT_MARKER_WRAP_LENGTH
        for marker, origin in zip(
            self.belt_markers, self.belt_marker_origins, strict=True
        ):
            x = (
                (origin + self.visual_belt_offset + half_length)
                % BELT_MARKER_WRAP_LENGTH
            ) - half_length
            marker.position = (x, 0.0, 0.021)

    def _refresh_debug_arrows(self, state: dict[str, Any]) -> None:
        settings = self.context.get_physics_debug_settings()
        if not bool(settings["draw_contact_forces"]):
            clear_debug_arrows()
            return
        force_scale = float(settings["contact_force_scale"])
        max_force_length = float(settings["contact_force_max_length"])
        arrows = _contact_arrows(
            state,
            force_scale=force_scale,
            max_force_length=max_force_length,
        )
        deformables = {
            str(body.get("name", "")): body for body in state["deformables"]
        }
        for result_name, group in zip(
            SOFT_PACKAGE_RESULTANT_NAMES,
            SOFT_PACKAGE_RESULTANT_BODY_GROUPS,
            strict=True,
        ):
            positions = []
            forces = []
            for body_index in group:
                name = SOFT_PACKAGE_NAMES[body_index]
                if name not in deformables:
                    continue
                positions.append(
                    np.asarray(deformables[name]["world_vertices"], dtype=np.float64)
                )
                applied = self.soft_force_model.applied_force.get(name)
                if applied is not None:
                    forces.append(np.asarray(applied, dtype=np.float64))
            if not positions or len(positions) != len(forces):
                continue
            points = np.concatenate(positions, axis=0)
            resultant = np.concatenate(forces, axis=0).sum(axis=0)
            start = points.mean(axis=0)
            start[2] = points[:, 2].max() + 0.025
            arrow = _force_arrow(
                start,
                resultant,
                color=EXTERNAL_FORCE_RESULTANT_COLOR,
                label=f"{result_name} belt drive",
                force_scale=force_scale,
                max_force_length=max_force_length,
            )
            if arrow is not None:
                arrows.append(arrow)
        set_debug_arrows(arrows)

    def _reset_cycle(self) -> None:
        self.context.clear_external_forces()
        self.context.reset_simulation()
        self.tick = self.initial_tick
        self.visual_belt_offset = 0.0
        self.peak_contact_penetration = 0.0
        self.penetration_hold_steps = 0
        clear_debug_arrows()

    def _physics_process(self, delta: float) -> None:
        del delta
        if not self.context.has_world:
            return
        input_state = getattr(self.context, "input", None)
        if input_state is not None and input_state.is_key_pressed("P"):
            self._reset_cycle()
            return
        if not self.drop_only and self.tick >= self.profile_module.CYCLE_TICKS:
            self._reset_cycle()
            return

        control_tick = self.tick
        if self.drop_only:
            control_tick = min(
                control_tick, self.profile_module.DROP_SETTLE_TICKS - 1
            )
        state = self.context.get_physics_state_view()
        penetration = _max_contact_penetration(state)
        self.peak_contact_penetration = max(
            self.peak_contact_penetration, penetration
        )
        phase = self.profile_module.cycle_phase(control_tick)
        speed = float(self.profile_module.belt_speed_at_tick(control_tick))
        self._apply_hand_targets(control_tick)
        self.force_model.apply(speed, state)
        self.soft_force_model.apply(
            speed,
            damping_scale=self.profile_module.soft_damping_scale_at_tick(
                control_tick
            ),
            state=state,
        )
        self._move_belt_markers(speed, float(self.profile_module.FIXED_DT))
        if self.tick % max(1, self.profile.contact_refresh_interval) == 0:
            self._refresh_debug_arrows(state)
        if self.tick and self.tick % 300 == 0:
            diagnostics = self.context.get_solver_diagnostics()
            print(
                "SuperDex conveyor "
                f"{phase} | "
                f"step={diagnostics['total_step_time_seconds'] * 1.0e3:.2f} ms | "
                f"residual={float(diagnostics['residual_norm']):.3g} | "
                f"penetration={penetration * 1.0e3:.3f} mm | "
                f"peak={self.peak_contact_penetration * 1.0e3:.3f} mm | "
                f"soft contacts={_soft_package_contact_links(state)} | "
                f"{diagnostics['convergence']}"
            )
        limit_preview_contact = (
            self.preview
            and phase in PREVIEW_PENETRATION_LIMIT_PHASES
            and penetration > PREVIEW_PENETRATION_LIMIT_METERS
            and self.penetration_hold_steps < PREVIEW_PENETRATION_HOLD_STEPS
        )
        if limit_preview_contact:
            self.penetration_hold_steps += 1
        else:
            self.penetration_hold_steps = 0
            self.tick += 1

    def _process(self, delta: float) -> None:
        del delta

    def _exit_tree(self) -> None:
        clear_debug_arrows()
        if self.context is not None and self.context.has_world:
            try:
                self.context.clear_external_forces()
            except Exception:
                pass
