"""Editor presentation for the worker-owned dual-FR3 rope simulation."""
from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any

import gobot
from gobot.render import DebugArrow, clear_debug_arrows, set_debug_arrows
from gobot.sim.providers import CompiledMuJoCoIpcArtifact
from rope_twist_config import (
    SCENE_ROOT_NAME, ROBOT_NAMES, ROBOT_LINK_NAMES, FIXTURE_BODY_NAMES, TOOL_LINK_NAME,
    FIXED_DT, NUM_ENVS, QUALITY_PROFILES, DEFAULT_QUALITY, COUPLING_ITERATIONS,
    QUALITY_ENVIRONMENT_VARIABLE, COUPLING_ITERATIONS_ENVIRONMENT_VARIABLE,
    CONTACT_FORCE_ARROW_MIN_NEWTONS, IPC_CONTACT_FORCE_ARROW_MIN_LENGTH,
    IPC_CONTACT_FORCE_ARROW_COLOR, GRIP_CONTACT_FORCE_ARROW_COLOR,
    BASE_FIELDS, CONTACT_FIELDS, _nodes_by_name, _load_project_module,
    _batch_config, _drive_mode, _quality_profile, _coupling_iterations, _grip_sensor_specs,
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
    """Own scene bindings only; physics and control live on the runtime worker."""

    def _ready(self):
        self.runtime = self.play_session = None
        self.contact_arrows_enabled = False
        self.cycle_complete = False
        self.cached_torque_arrows = []
        try:
            root = self.get_root()
            if root is None or root.name != SCENE_ROOT_NAME:
                raise RuntimeError("unexpected dual-arm rope scene root")
            self.controllers_module = _load_project_module(self.context.project_path,
                "controllers.py", "gobot_dual_arm_rope_display_controllers")
            self.drive_mode, self.wrist_torque_limit, _ = _drive_mode(self.controllers_module)
            self.quality_profile = _quality_profile()
            self.coupling_iterations = _coupling_iterations(self.quality_profile)
            settings = self.context.get_mujoco_solver_settings()
            settings["integrator"] = gobot.PhysicsIntegratorType.ImplicitFast
            settings["cone"] = gobot.PhysicsFrictionConeType.Elliptic
            settings["impedance_ratio"] = self.controllers_module.GRIP_IMPEDANCE_RATIO
            self.context.set_mujoco_solver_settings(settings)
            artifact = CompiledMuJoCoIpcArtifact.from_context(self.context)
            links = {}
            for name in ROBOT_NAMES:
                robot = root.find(name)
                if robot is None:
                    raise RuntimeError(f"missing rope robot {name!r}")
                nodes = _nodes_by_name(robot)
                links[f"robot.{name}.link_pose"] = [[nodes[link] for link in ROBOT_LINK_NAMES]]
            for name in FIXTURE_BODY_NAMES:
                body = root.find(name)
                if body is None:
                    raise RuntimeError(f"missing rope fixture {name!r}")
                links[f"robot.{name}.link_pose"] = [[body]]
            bodies = [root.find(str(entry["path"]).rsplit("/", 1)[-1])
                      for entry in artifact.ipc.deformable_bodies]
            if any(body is None or body.type_name != "DeformableBody3D" for body in bodies):
                raise RuntimeError("rope scene is missing a deformable strand")
            self.scene_sync = gobot.sim.SceneSnapshotSync(self.context, links=links,
                deformables=[bodies],
                vertex_counts=[entry["vertex_count"] for entry in artifact.ipc.deformable_bodies])
            recipe = gobot.sim.SimulationRuntimeSpec("rope_twist_runtime:create", {
                "artifact": artifact.to_mapping(),
                "solver_config": asdict(_batch_config(self.context, self.quality_profile)),
                "quality": asdict(self.quality_profile),
                "coupling_iterations": self.coupling_iterations,
                "drive_mode": self.drive_mode, "wrist_torque_limit": self.wrist_torque_limit,
            })
            self.runtime = gobot.sim.AsyncSimulationSession(recipe, fields=BASE_FIELDS)
            self.play_session = gobot.sim.ProviderPlaySession(self.context, self.runtime,
                fixed_dt=FIXED_DT, max_sub_steps=1, sync_scene=self._sync_scene).start()
            self.play_session.set_status("Starting dual FR3 rope physics worker")
            print(f"Dual FR3 rope worker starting: quality={self.quality_profile.name}; "
                  f"drive={self.drive_mode} ({self.wrist_torque_limit:g} N m); "
                  f"SolverCoupledProxy x{self.coupling_iterations}")
        except Exception:
            self._close_play_session()
            raise

    def _physics_process(self, delta):
        del delta
        if self.play_session is None:
            return
        input_state = getattr(self.context, "input", None)
        if ((input_state is not None and input_state.is_key_pressed("P")) or
                self.cycle_complete):
            self.play_session.reset()
            self.cycle_complete = False

    def _process(self, delta):
        del delta
        if self.runtime is None:
            return
        enabled = bool(self.context.get_physics_debug_settings()["draw_contact_forces"])
        if enabled != self.contact_arrows_enabled:
            self.contact_arrows_enabled = enabled
            self.runtime.subscribe(BASE_FIELDS + (CONTACT_FIELDS if enabled else ()))
            if not enabled:
                set_debug_arrows(self.cached_torque_arrows)

    def _sync_scene(self, snapshot):
        import numpy as np

        self.scene_sync(snapshot)
        fields = set(snapshot.fields)
        if "rope.metrics" in fields:
            values = np.asarray(snapshot.buffer("rope.metrics"))[0]
            relative, winding, speed0, speed1, effort0, effort1, peak, stalled, complete, slip, mount, tick = values
            self.cycle_complete = bool(complete)
            self.play_session.set_status(
                f"Dual FR3 rope twist | {self.controllers_module.phase_for_tick(int(tick))}"
                f" | {self.drive_mode} {self.wrist_torque_limit:g} N m | {self.quality_profile.name}"
                f" | SolverCoupledProxy x{self.coupling_iterations}"
                f" | relative {relative:.2f} turns | rope winding {winding:.2f}"
                f" | wrist speed {speed0:.2f}/{speed1:.2f} rad/s | drive {effort0:.3f}/{effort1:.3f} N m"
                + (" | STALLED" if stalled else "")
                + f" | fixture slip {slip * 1000:.1f} mm | mount error {mount * 1000:.1f} mm"
                f" | rope reaction peak {peak:.3f} N m")
        if "rope.wrenches" in fields:
            poses = [np.asarray(snapshot.buffer(f"robot.{name}.link_pose"))[0, ROBOT_LINK_NAMES.index(TOOL_LINK_NAME)]
                     for name in ROBOT_NAMES]
            self.cached_torque_arrows = _torque_arrows(poses, np.asarray(snapshot.buffer("rope.wrenches"))[0])
        arrows = list(self.cached_torque_arrows)
        if self.contact_arrows_enabled and set(CONTACT_FIELDS) <= fields:
            settings = self.context.get_physics_debug_settings()
            options = dict(force_scale=float(settings["contact_force_scale"]),
                           max_force_length=float(settings["contact_force_max_length"]))
            arrows.extend(_contact_force_arrows(
                np.asarray(snapshot.buffer("rope.contact_positions"))[0],
                np.asarray(snapshot.buffer("rope.contact_forces"))[0],
                color=IPC_CONTACT_FORCE_ARROW_COLOR, label="rope contact",
                min_force_length=IPC_CONTACT_FORCE_ARROW_MIN_LENGTH, **options))
            arrows.extend(_contact_force_arrows(
                np.asarray(snapshot.buffer("grip.positions"))[0],
                np.asarray(snapshot.buffer("grip.forces"))[0],
                found=np.asarray(snapshot.buffer("grip.found"))[0],
                color=GRIP_CONTACT_FORCE_ARROW_COLOR, label="grip contact", max_count=4, **options))
        set_debug_arrows(arrows)

    def _exit_tree(self):
        self._close_play_session()

    def _close_play_session(self):
        clear_debug_arrows()
        session, self.play_session = self.play_session, None
        runtime, self.runtime = self.runtime, None
        if session is not None:
            session.close()
        elif runtime is not None:
            runtime.close()
