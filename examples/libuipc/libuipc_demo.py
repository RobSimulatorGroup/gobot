"""Play and headless entry points for the native libuipc demos."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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


class Script(gobot.NodeScript):
    """Editor Play entry point for the native libuipc FR3 scene."""

    def _ready(self) -> None:
        self.runtime = None
        self.play_session = None
        self.loop_ticks = max(1, round(LOOP_SECONDS / FIXED_DT))
        self.output_fields = ("affine.link_pose", "deformable.local_vertices")
        clear_debug_arrows()
        try:
            root = self.get_root()
            if root is None:
                raise RuntimeError("libuipc demo script has no scene root")
            if root.name != SCENE_ROOT_NAME:
                raise RuntimeError(f"unexpected libuipc demo scene {root.name!r}")

            artifact = CompiledIpcSceneArtifact.from_mapping(self.context.compile_ipc_scene_artifact())
            nodes_by_name = _nodes_by_name(root)
            bodies = []
            for entry in artifact.deformable_bodies:
                name = str(entry["path"]).rsplit("/", 1)[-1]
                body = nodes_by_name.get(name)
                if body is None or body.type_name != "DeformableBody3D":
                    raise RuntimeError(f"libuipc demo is missing deformable body {name!r}")
                bodies.append(body)
            affine_links = []
            affine_entries = [link for robot in artifact.robots for link in robot["links"]
                    if any(not shape.get("disabled", False) for shape in link["collision_shapes"])]
            for entry in affine_entries:
                name = str(entry["path"]).rsplit("/", 1)[-1]
                link = nodes_by_name.get(name)
                if link is None or link.type_name != "Link3D":
                    raise RuntimeError(f"libuipc demo is missing affine link {name!r}")
                affine_links.append(link)
            self.scene_sync = gobot.sim.SceneSnapshotSync(self.context,
                links={"affine.link_pose": [affine_links]}, deformables=[bodies],
                vertex_counts=[int(entry["vertex_count"]) for entry in artifact.deformable_bodies])
            recipe = gobot.sim.SimulationRuntimeSpec("libuipc_runtime:create", {
                "artifact": artifact.to_mapping(), "config": asdict(_libuipc_config()),
                "affine_paths": [entry["path"] for entry in affine_entries]})
            self.runtime = gobot.sim.AsyncSimulationSession(recipe, fields=self.output_fields)
            self.play_session = gobot.sim.ProviderPlaySession(
                self.context,
                self.runtime,
                fixed_dt=FIXED_DT,
                max_sub_steps=1,
                sync_scene=self._sync_scene,
            ).start()
            self.play_session.set_status("Running native libuipc FEM")
            print(
                f"libuipc demo started: scene={root.name} "
                f"deformables={len(bodies)} "
                f"affine_bodies={len(affine_links)}"
            )
        except Exception:
            self._close_play_session()
            raise

    def _physics_process(self, delta: float) -> None:
        del delta
        if self.runtime is None or self.play_session is None:
            return
        input_state = getattr(self.context, "input", None)
        if input_state is not None and input_state.is_key_pressed("P"):
            self.play_session.reset()
            return
        if self.context.frame_count >= self.loop_ticks:
            self.play_session.reset()
            return
    def _process(self, delta: float) -> None:
        del delta
        if self.runtime is None:
            return
        fields = ("affine.link_pose", "deformable.local_vertices")
        if self.context.get_physics_debug_settings()["draw_contact_forces"]:
            fields += ("positions", "contact_forces")
        else:
            clear_debug_arrows()
        if fields != self.output_fields and self.runtime.subscribe(fields):
            self.output_fields = fields

    def _sync_scene(self, snapshot) -> None:
        import numpy as np

        self.scene_sync(snapshot)
        settings = self.context.get_physics_debug_settings()
        if not settings["draw_contact_forces"] or "contact_forces" not in snapshot.fields:
            clear_debug_arrows()
            return
        set_debug_arrows(_contact_force_arrows(
            np.asarray(snapshot.buffer("positions"))[0],
            np.asarray(snapshot.buffer("contact_forces"))[0],
            force_scale=float(settings["contact_force_scale"]),
            max_force_length=float(settings["contact_force_max_length"])))

    def _exit_tree(self) -> None:
        self._close_play_session()

    def _close_play_session(self) -> None:
        clear_debug_arrows()
        play_session = self.play_session
        self.play_session = None
        if play_session is not None:
            play_session.close()
        runtime, self.runtime = self.runtime, None
        if runtime is not None and play_session is None:
            runtime.close()


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
