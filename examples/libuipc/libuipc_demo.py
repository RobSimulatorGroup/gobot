"""Play and headless entry points for the native libuipc demos."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import gobot
from gobot.ipc import (
    CompiledIpcSceneArtifact,
    LibuipcConfig,
    LibuipcProvider,
)
from gobot.render import DebugArrow, clear_debug_arrows, set_debug_arrows


HERE = Path(__file__).resolve().parent
DEFAULT_SCENE = HERE / "fr3_brick_grasp.jscn"
FIXED_DT = 0.01
LOOP_SECONDS = 10.0
SCENE_ROOT_NAME = "libuipc_fr3_soft_grasp"
FR3_INITIAL_ARM = {
    "fr3_joint1": -0.0036802115,
    "fr3_joint2": 0.023901723,
    "fr3_joint3": 0.003680411,
    "fr3_joint4": -2.3683236,
    "fr3_joint5": -0.00012918962,
    "fr3_joint6": 2.3922248,
    "fr3_joint7": 0.785492,
}
FR3_ARM_LIFT_OFFSETS = {
    "fr3_joint1": 0.0,
    "fr3_joint2": -0.07,
    "fr3_joint3": 0.0,
    "fr3_joint4": 0.055,
    "fr3_joint5": 0.0,
    "fr3_joint6": 0.055,
    "fr3_joint7": 0.0,
}
FR3_FINGER_JOINTS = (
    "fr3_finger_joint1",
    "fr3_finger_joint2",
)
FR3_JOINT_NAMES = tuple(FR3_INITIAL_ARM) + FR3_FINGER_JOINTS
FR3_OPEN_FINGER = 0.017
FR3_CLOSED_FINGER = 0.0146
CONTACT_FORCE_ARROW_MAX_COUNT = 24
CONTACT_FORCE_ARROW_MIN_NEWTONS = 1.0e-3
CONTACT_FORCE_ARROW_LENGTH_SCALE = 0.08
CONTACT_FORCE_ARROW_MAX_LENGTH = 0.8
CONTACT_FORCE_ARROW_COLOR = (1.0, 0.16, 0.04, 1.0)


def _nodes_by_name(root: Any) -> dict[str, Any]:
    nodes: dict[str, Any] = {}
    pending = [root]
    while pending:
        node = pending.pop()
        if node.name in nodes:
            raise RuntimeError(f"libuipc demo has duplicate node name {node.name!r}")
        nodes[node.name] = node
        pending.extend(node.children)
    return nodes


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def _transition(time_seconds: float, start: float, end: float) -> float:
    if end <= start:
        raise ValueError("motion transition end must be greater than start")
    return _smoothstep((time_seconds - start) / (end - start))


def _fr3_motion_targets(time_seconds: float) -> dict[str, float]:
    time_seconds = max(0.0, float(time_seconds))
    if time_seconds < 1.0:
        grasp = 0.0
    elif time_seconds < 2.4:
        grasp = _transition(time_seconds, 1.0, 2.4)
    elif time_seconds < 8.5:
        grasp = 1.0
    elif time_seconds < 9.5:
        grasp = 1.0 - _transition(time_seconds, 8.5, 9.5)
    else:
        grasp = 0.0

    if time_seconds < 3.6:
        lift = 0.0
    elif time_seconds < 5.2:
        lift = _transition(time_seconds, 3.6, 5.2)
    elif time_seconds < 6.2:
        lift = 1.0
    elif time_seconds < 7.6:
        lift = 1.0 - _transition(time_seconds, 6.2, 7.6)
    else:
        lift = 0.0

    targets = {
        name: FR3_INITIAL_ARM[name] + offset * lift
        for name, offset in FR3_ARM_LIFT_OFFSETS.items()
    }
    finger_target = FR3_OPEN_FINGER + (
        FR3_CLOSED_FINGER - FR3_OPEN_FINGER
    ) * grasp
    targets.update(
        {name: finger_target for name in FR3_FINGER_JOINTS}
    )
    return targets


def _contact_force_arrows(
    positions: Any,
    forces: Any,
    *,
    force_scale: float = CONTACT_FORCE_ARROW_LENGTH_SCALE,
    max_force_length: float = CONTACT_FORCE_ARROW_MAX_LENGTH,
) -> list[DebugArrow]:
    import numpy as np

    points = np.asarray(positions, dtype=np.float64)
    values = np.asarray(forces, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or values.shape != points.shape:
        raise RuntimeError("libuipc contact-force arrays must have shape [vertex_count,3]")
    if not np.isfinite(points).all() or not np.isfinite(values).all():
        raise RuntimeError("libuipc contact-force arrays contain non-finite values")

    magnitudes = np.linalg.norm(values, axis=1)
    indices = np.flatnonzero(magnitudes >= CONTACT_FORCE_ARROW_MIN_NEWTONS)
    if len(indices) > CONTACT_FORCE_ARROW_MAX_COUNT:
        selected = np.argpartition(
            magnitudes[indices], -CONTACT_FORCE_ARROW_MAX_COUNT
        )[-CONTACT_FORCE_ARROW_MAX_COUNT :]
        indices = indices[selected]
    indices = indices[np.argsort(magnitudes[indices])[::-1]]

    arrows = []
    for index in indices:
        magnitude = float(magnitudes[index])
        direction = values[index] / magnitude
        length = min(
            max_force_length,
            force_scale * math.log1p(magnitude),
        )
        if length <= 0.0:
            continue
        arrows.append(
            DebugArrow(
                start=points[index],
                vector=direction,
                color=CONTACT_FORCE_ARROW_COLOR,
                scale=length,
                label=f"{magnitude:.3g} N",
            )
        )
    return arrows


def _libuipc_config() -> LibuipcConfig:
    return LibuipcConfig(
        fixed_time_step=FIXED_DT,
        friction_coefficient=0.8,
        contact_activation_distance=5.0e-4,
        contact_resistance=1.0e8,
        affine_stiffness=1.0e8,
        kinematic_strength=100.0,
    )


def _scene_provider(
    context: Any,
) -> LibuipcProvider:
    return LibuipcProvider.from_context(
        context,
        config=_libuipc_config(),
    )


class Script(gobot.NodeScript):
    """Editor Play entry point for the native libuipc FR3 scene."""

    def _ready(self) -> None:
        self.provider = None
        self.play_session = None
        self.tick = 0
        self.loop_ticks = max(1, round(LOOP_SECONDS / FIXED_DT))
        self.joint_targets = {}
        clear_debug_arrows()
        try:
            root = self.get_root()
            if root is None:
                raise RuntimeError("libuipc demo script has no scene root")
            if root.name != SCENE_ROOT_NAME:
                raise RuntimeError(f"unexpected libuipc demo scene {root.name!r}")

            self.provider = _scene_provider(self.context)
            nodes_by_name = _nodes_by_name(root)
            bodies = []
            for entry in self.provider.deformable_bodies:
                name = str(entry["path"]).rsplit("/", 1)[-1]
                body = nodes_by_name.get(name)
                if body is None or body.type_name != "DeformableBody3D":
                    raise RuntimeError(f"libuipc demo is missing deformable body {name!r}")
                bodies.append(body)
            affine_links = []
            for entry in self.provider.affine_bodies:
                name = str(entry["path"]).rsplit("/", 1)[-1]
                link = nodes_by_name.get(name)
                if link is None or link.type_name != "Link3D":
                    raise RuntimeError(f"libuipc demo is missing affine link {name!r}")
                affine_links.append(link)
            self.provider.bind_scene(self.context, bodies, affine_links)

            for name in FR3_JOINT_NAMES:
                matches = [
                    str(entry["path"])
                    for entry in self.provider.joints
                    if str(entry["path"]).endswith("/" + name)
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"FR3 grasp demo has no unique {name} joint"
                    )
                initial = (
                    FR3_INITIAL_ARM[name]
                    if name in FR3_INITIAL_ARM
                    else FR3_OPEN_FINGER
                )
                self.joint_targets[name] = (matches[0], initial)

            self.provider.sync_scene()
            self.play_session = gobot.sim.ProviderPlaySession(
                self.context,
                self.provider,
                fixed_dt=FIXED_DT,
                max_sub_steps=1,
                before_step=self._before_step,
                reset=self._reset,
                sync_scene=self._sync_scene,
            ).start()
            self.play_session.set_status("Running native libuipc FEM")
            print(
                f"libuipc demo started: scene={root.name} "
                f"deformables={len(self.provider.deformable_bodies)} "
                f"affine_bodies={len(self.provider.affine_bodies)}"
            )
        except Exception:
            self._close_play_session()
            raise

    def _before_step(self, fixed_dt: float) -> None:
        if self.provider is None:
            return
        for name, target in _fr3_motion_targets(self.tick * fixed_dt).items():
            self.provider.set_joint_target(self.joint_targets[name][0], target)
        self.tick += 1

    def _physics_process(self, delta: float) -> None:
        del delta
        if self.provider is None or self.play_session is None:
            return
        input_state = getattr(self.context, "input", None)
        if input_state is not None and input_state.is_key_pressed("P"):
            self.play_session.reset()
            return
        if self.tick >= self.loop_ticks:
            self.play_session.reset()
            return
        if self.tick % 10 == 0:
            diagnostics = self.provider.diagnostics
            self.play_session.set_status(
                f"Native libuipc | frame {int(diagnostics.get('frame', 0))} | "
                f"{float(diagnostics.get('last_step_latency_ms', 0.0)):.1f} ms"
            )

    def _process(self, delta: float) -> None:
        del delta

    def _reset(self) -> None:
        if self.provider is None:
            return
        self.provider.reset()
        self.tick = 0
        clear_debug_arrows()
        for path, initial in self.joint_targets.values():
            self.provider.set_joint_target(path, initial)

    def _sync_scene(self) -> None:
        if self.provider is not None:
            self.provider.sync_scene()
            settings = self.context.get_physics_debug_settings()
            draw_contact_forces = bool(settings["draw_contact_forces"])
            force_scale = float(settings["contact_force_scale"])
            max_force_length = float(settings["contact_force_max_length"])
            if not draw_contact_forces:
                clear_debug_arrows()
                return
            set_debug_arrows(
                _contact_force_arrows(
                    self.provider.arrays["positions"],
                    self.provider.arrays["contact_forces"],
                    force_scale=force_scale,
                    max_force_length=max_force_length,
                )
            )

    def _exit_tree(self) -> None:
        self._close_play_session()

    def _close_play_session(self) -> None:
        clear_debug_arrows()
        play_session = self.play_session
        self.play_session = None
        if play_session is not None:
            play_session.close()
        provider = self.provider
        self.provider = None
        if provider is not None and play_session is None:
            provider.close()


def _load_scene(scene_path: Path) -> tuple[Any, CompiledIpcSceneArtifact]:
    scene_path = scene_path.expanduser().resolve()
    context = gobot.app.create_context()
    context.set_project_path(str(scene_path.parent))
    context.load_scene("res://" + scene_path.name)
    artifact = CompiledIpcSceneArtifact.from_mapping(
        context.compile_ipc_scene_artifact()
    )
    return context, artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--steps", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be non-negative")
    context, artifact = _load_scene(args.scene)
    summary: dict[str, Any] = {
        "affine_bodies": sum(
            1
            for robot in artifact.robots
            for link in robot["links"]
            if any(not shape.get("disabled", False) for shape in link["collision_shapes"])
        ),
        "artifact": artifact.digest,
        "deformable_bodies": len(artifact.deformable_bodies),
        "provider": "libuipc",
        "scene": args.scene.name,
    }
    if args.steps:
        provider = LibuipcProvider(
            artifact,
            config=_libuipc_config(),
        )
        try:
            provider.step(nsteps=args.steps)
            import numpy as np

            positions = np.asarray(provider.arrays["positions"])
            if not np.isfinite(positions).all():
                raise RuntimeError("libuipc produced non-finite deformable positions")
            summary["diagnostics"] = dict(provider.diagnostics)
            summary["position_bounds"] = [
                positions.min(axis=0).tolist(),
                positions.max(axis=0).tolist(),
            ]
        finally:
            provider.close()
    print(json.dumps(summary, indent=2, sort_keys=True))
    del context


if __name__ == "__main__":
    main()
