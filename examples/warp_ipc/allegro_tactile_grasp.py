"""Run the Gobot-native Allegro grasp and export 2D/3D tactile signals.

The released Taccel examples do not contain the paper's multi-hand grasp
synthesis program. This example ports the observable Allegro sequence instead:
50 command frames close four fingers, 10 frames settle the grasp, and 20 frames
lift the kinematic hand.
It consumes only the checked-in ``.jscn`` scene and its compiled IPC artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import gobot
from gobot.ipc import CompiledIpcSceneArtifact, WarpIpcConfig, WarpIpcProvider


HERE = Path(__file__).resolve().parent
DEFAULT_SCENE = HERE / "allegro_tactile_grasp.jscn"
ASSET_MANIFEST = HERE / "allegro_assets.json"
ASSET_ROOT = HERE / "assets" / "wonik_allegro"

CLOSE_FRAMES = 60
LIFT_FRAMES = 20
CLOSE_MOTION_FRAMES = 50
FRAME_DT = 0.02
PHYSICS_SUBSTEPS = 4
LIFT_PER_FRAME = 0.0025

_EXPECTED_JOINTS = frozenset(
    {
        "joint_00",
        "joint_10",
        "joint_20",
        "joint_30",
        "joint_40",
        "joint_50",
        "joint_60",
        "joint_70",
        "joint_80",
        "joint_90",
        "joint_100",
        "joint_110",
        "joint_120",
        "joint_130",
        "joint_140",
        "joint_150",
    }
)
_EXPECTED_TIP_LINKS = frozenset(
    {"link_30_tip", "link_70_tip", "link_110_tip", "link_150_tip"}
)
_OPEN_POSE = {
    "joint_00": 0.0,
    "joint_10": 1.1033,
    "joint_20": 1.20863,
    "joint_30": 1.155965,
    "joint_40": 0.0,
    "joint_50": 1.1167,
    "joint_60": 1.22337,
    "joint_70": 1.170035,
    "joint_80": 0.0,
    "joint_90": 1.1033,
    "joint_100": 1.20863,
    "joint_110": 1.155965,
    "joint_120": 0.55,
    "joint_130": 0.05,
    "joint_140": 0.05,
    "joint_150": 0.05,
}

_CLOSED_POSE = {
    "joint_00": 0.0,
    "joint_10": 1.054264,
    "joint_20": 1.1546904,
    "joint_30": 1.1044772,
    "joint_40": 0.0,
    "joint_50": 1.0799,
    "joint_60": 1.18289,
    "joint_70": 1.131395,
    "joint_80": 0.0,
    "joint_90": 1.054264,
    "joint_100": 1.1546904,
    "joint_110": 1.1044772,
    "joint_120": 1.0512282,
    "joint_130": -0.0351211,
    "joint_140": 0.4569999,
    "joint_150": 0.6219878,
}


@dataclass(frozen=True)
class AllegroGraspFrame:
    index: int
    phase: str
    joint_position: tuple[float, ...]
    base_pose_xyzw: tuple[float, ...]


@dataclass(frozen=True)
class AllegroGraspProgram:
    robot_name: str
    base_link: str
    joint_names: tuple[str, ...]
    link_names: tuple[str, ...]
    tactile_sensor_paths: tuple[str, ...]
    deformable_body_paths: tuple[str, ...]
    initial_base_pose_xyzw: tuple[float, ...]
    joint_limits: tuple[tuple[float, float], ...]

    @property
    def frame_count(self) -> int:
        return CLOSE_FRAMES + LIFT_FRAMES

    @property
    def open_joint_position(self) -> tuple[float, ...]:
        return tuple(_OPEN_POSE[name] for name in self.joint_names)

    @property
    def closed_joint_position(self) -> tuple[float, ...]:
        return tuple(_CLOSED_POSE[name] for name in self.joint_names)

    def frame(self, index: int) -> AllegroGraspFrame:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Allegro grasp frame index must be an integer")
        if not 0 <= index < self.frame_count:
            raise IndexError(
                f"Allegro grasp frame {index} is outside [0, {self.frame_count})"
            )

        if index < CLOSE_FRAMES:
            phase = "close"
            alpha = min(1.0, float(index + 1) / float(CLOSE_MOTION_FRAMES))
            joints = tuple(
                start + (end - start) * alpha
                for name, start, end in zip(
                    self.joint_names,
                    self.open_joint_position,
                    self.closed_joint_position,
                    strict=True,
                )
            )
            lift = 0.0
        else:
            phase = "lift"
            joints = self.closed_joint_position
            lift = LIFT_PER_FRAME * float(index - CLOSE_FRAMES + 1)

        base_pose = list(self.initial_base_pose_xyzw)
        base_pose[2] += lift
        return AllegroGraspFrame(index, phase, joints, tuple(base_pose))


def _quaternion_xyzw(matrix: Sequence[float]) -> tuple[float, float, float, float]:
    m00, m01, m02 = matrix[0], matrix[1], matrix[2]
    m10, m11, m12 = matrix[4], matrix[5], matrix[6]
    m20, m21, m22 = matrix[8], matrix[9], matrix[10]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            (m21 - m12) / scale,
            (m02 - m20) / scale,
            (m10 - m01) / scale,
            0.25 * scale,
        )
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        quaternion = (
            0.25 * scale,
            (m01 + m10) / scale,
            (m02 + m20) / scale,
            (m21 - m12) / scale,
        )
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        quaternion = (
            (m01 + m10) / scale,
            0.25 * scale,
            (m12 + m21) / scale,
            (m02 - m20) / scale,
        )
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        quaternion = (
            (m02 + m20) / scale,
            (m12 + m21) / scale,
            0.25 * scale,
            (m10 - m01) / scale,
        )
    length = math.sqrt(sum(value * value for value in quaternion))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("Allegro robot transform contains an invalid rotation")
    return tuple(value / length for value in quaternion)


def _pose_xyzw(transform: Mapping[str, Any]) -> tuple[float, ...]:
    matrix = tuple(float(value) for value in transform["matrix_row_major"])
    if len(matrix) != 16:
        raise ValueError("Allegro robot transform must be a 4x4 matrix")
    quaternion = _quaternion_xyzw(matrix)
    return (matrix[3], matrix[7], matrix[11], *quaternion)


def build_grasp_program(
    artifact: CompiledIpcSceneArtifact,
) -> AllegroGraspProgram:
    if len(artifact.robots) != 1:
        raise ValueError(
            f"Allegro grasp scene must compile exactly one robot, got {len(artifact.robots)}"
        )
    robot = artifact.robots[0]
    joints = tuple(
        joint for joint in robot["joints"] if int(joint["joint_type"]) != 0
    )
    if any(int(joint["joint_type"]) != 1 for joint in joints):
        raise ValueError("Allegro grasp movable joints must all be revolute")
    joint_names = tuple(str(joint["name"]) for joint in joints)
    if len(joint_names) != 16 or frozenset(joint_names) != _EXPECTED_JOINTS:
        raise ValueError(
            "Allegro grasp scene must contain the pinned 16 movable Allegro joints"
        )
    limits = tuple(
        (float(joint["lower_limit"]), float(joint["upper_limit"])) for joint in joints
    )
    for name, target, limit in zip(
        joint_names,
        (_CLOSED_POSE[name] for name in joint_names),
        limits,
        strict=True,
    ):
        if not limit[0] <= target <= limit[1]:
            raise ValueError(
                f"Allegro closed target {target} for {name!r} exceeds limits {limit}"
            )

    links = tuple(robot["links"])
    root_paths = tuple(str(path) for path in robot["root_link_paths"])
    if len(root_paths) != 1:
        raise ValueError("Allegro grasp robot must have exactly one root link")
    base_matches = [link for link in links if str(link["path"]) == root_paths[0]]
    if len(base_matches) != 1:
        raise ValueError("Allegro root link path does not resolve uniquely")

    sensors = tuple(sorted(artifact.tactile_sensors, key=lambda item: str(item["name"])))
    if len(sensors) != 4:
        raise ValueError(
            f"Allegro grasp scene must contain four tactile sensors, got {len(sensors)}"
        )
    sensor_contracts = {
        (
            tuple(int(value) for value in sensor["resolution"]),
            str(sensor["gel_topology_sha256"]),
            int(sensor["gel_vertex_count"]),
            len(sensor["marker_positions"]),
        )
        for sensor in sensors
    }
    if len(sensor_contracts) != 1:
        raise ValueError(
            "Allegro tactile sensors must share resolution, gel topology, and markers"
        )
    if any(sensor.get("attachment") is None for sensor in sensors):
        raise ValueError("every Allegro tactile sensor must be attached to a fingertip link")
    resolution, _, gel_vertex_count, marker_count = next(iter(sensor_contracts))
    if resolution != (120, 160) or gel_vertex_count != 200 or marker_count != 64:
        raise ValueError(
            "Allegro tactile sensors must use the pinned 120x160, 200-vertex, "
            "8x8-marker gel contract"
        )
    attachment_links = frozenset(
        str(sensor["attachment"]["link_path"]).rsplit("/", 1)[-1]
        for sensor in sensors
    )
    if attachment_links != _EXPECTED_TIP_LINKS:
        raise ValueError("Allegro tactile sensors must attach once to each fingertip")

    deformables = tuple(
        sorted(artifact.deformable_bodies, key=lambda item: str(item["path"]))
    )
    if len(deformables) != 1:
        raise ValueError(
            "Allegro grasp scene must contain one independent deformable grasp object"
        )
    if str(deformables[0]["name"]) != "soft_grasp_object":
        raise ValueError("Allegro grasp deformable object has an unexpected stable name")

    return AllegroGraspProgram(
        robot_name=str(robot["name"]),
        base_link=str(base_matches[0]["name"]),
        joint_names=joint_names,
        link_names=tuple(str(link["name"]) for link in links),
        tactile_sensor_paths=tuple(str(sensor["path"]) for sensor in sensors),
        deformable_body_paths=tuple(str(body["path"]) for body in deformables),
        initial_base_pose_xyzw=_pose_xyzw(robot["transform"]),
        joint_limits=limits,
    )


def validate_assets(asset_root: Path = ASSET_ROOT) -> None:
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        relative = Path(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Allegro asset manifest contains unsafe path {relative}")
        path = asset_root / relative
        if not path.is_file():
            raise RuntimeError(
                f"Allegro asset is missing: {relative}; run download_allegro_assets.py"
            )
        if path.stat().st_size != int(entry["size"]):
            raise RuntimeError(f"Allegro asset has the wrong size: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != str(entry["sha256"]):
            raise RuntimeError(f"Allegro asset failed SHA-256 validation: {relative}")


def load_program(scene_path: Path) -> tuple[Any, CompiledIpcSceneArtifact, AllegroGraspProgram]:
    scene_path = scene_path.expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    if scene_path == DEFAULT_SCENE.resolve():
        validate_assets()
    context = gobot.app.create_context()
    context.set_project_path(str(scene_path.parent))
    context.load_scene("res://" + scene_path.name)
    artifact = CompiledIpcSceneArtifact.from_mapping(
        context.compile_ipc_scene_artifact()
    )
    return context, artifact, build_grasp_program(artifact)


def _repeat(values: Sequence[float], count: int) -> list[list[float]]:
    return [list(values) for _ in range(count)]


def _as_numpy(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if not callable(numpy):
        raise RuntimeError(
            f"tactile output {type(value).__name__} has no explicit NumPy readback"
        )
    return numpy()


def _write_frame(
    output_dir: Path,
    frame: AllegroGraspFrame,
    selected_env: int,
    tactile_state: Any,
    deformable_state: Any,
) -> None:
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    values = {
        name: _as_numpy(getattr(tactile_state, name))[selected_env]
        for name in (
            "rgb",
            "depth",
            "normal",
            "marker_position",
            "marker_flow",
            "contact_force",
            "contact_wrench",
        )
    }
    values["deformable_position"] = _as_numpy(deformable_state.position)[selected_env]
    values["deformable_velocity"] = _as_numpy(deformable_state.velocity)[selected_env]
    values["deformable_contact_force"] = _as_numpy(
        deformable_state.contact_force
    )[selected_env]
    values["joint_target"] = np.asarray(frame.joint_position, dtype=np.float32)
    values["base_target_xyzw"] = np.asarray(frame.base_pose_xyzw, dtype=np.float32)
    np.savez_compressed(output_dir / f"frame_{frame.index:04d}.npz", **values)


def run(
    context: Any,
    artifact: CompiledIpcSceneArtifact,
    program: AllegroGraspProgram,
    *,
    num_envs: int,
    device: str,
    capture_graphs: bool,
    output_dir: Path | None,
    record_every: int,
    selected_env: int,
) -> None:
    availability = WarpIpcProvider.availability()
    if not availability.available:
        raise RuntimeError(
            "the Allegro scene and grasp program validated, but simulation cannot start: "
            + availability.reason
        )

    import torch

    config = WarpIpcConfig(
        device=device,
        fixed_time_step=FRAME_DT / float(PHYSICS_SUBSTEPS),
        barrier_distance=2.0e-3,
        barrier_stiffness=1.5e6,
        kinematic_stiffness=1.0e5,
        friction_coefficient=0.5,
        newton_iterations=8,
        cg_iterations=64,
        capture_graphs=capture_graphs,
    )
    with WarpIpcProvider(artifact, num_envs=num_envs, config=config) as provider:
        robot_view = provider.create_robot_view(
            robot_name=program.robot_name,
            base_link=program.base_link,
            joint_names=program.joint_names,
        )
        tactile_view = provider.create_tactile_view(
            sensor_names=program.tactile_sensor_paths
        )
        deformable_view = provider.create_deformable_view(
            body_names=program.deformable_body_paths
        )

        target_device = torch.device(device)
        reset_mask = torch.ones(num_envs, dtype=torch.bool, device=target_device)
        open_joints = torch.as_tensor(
            _repeat(program.open_joint_position, num_envs),
            dtype=torch.float32,
            device=target_device,
        )
        base_pose = torch.as_tensor(
            _repeat(program.initial_base_pose_xyzw, num_envs),
            dtype=torch.float32,
            device=target_device,
        )
        robot_view.reset(
            reset_mask,
            base_pose=base_pose,
            joint_position=open_joints,
            joint_velocity=torch.zeros_like(open_joints),
        )
        previous_joint_targets = open_joints
        previous_base_targets = base_pose

        for index in range(program.frame_count):
            frame = program.frame(index)
            command_joint_targets = torch.as_tensor(
                _repeat(frame.joint_position, num_envs),
                dtype=torch.float32,
                device=target_device,
            )
            command_base_targets = torch.as_tensor(
                _repeat(frame.base_pose_xyzw, num_envs),
                dtype=torch.float32,
                device=target_device,
            )
            for substep in range(PHYSICS_SUBSTEPS):
                alpha = float(substep + 1) / float(PHYSICS_SUBSTEPS)
                joint_targets = torch.lerp(
                    previous_joint_targets, command_joint_targets, alpha
                )
                base_targets = torch.lerp(
                    previous_base_targets, command_base_targets, alpha
                )
                robot_view.set_position_targets(joint_targets)
                robot_view.set_base_pose_targets(base_targets)
                try:
                    provider.step(nsteps=1)
                except RuntimeError as error:
                    raise RuntimeError(
                        "Allegro Warp IPC failed at "
                        f"frame {index} ({frame.phase}), substep {substep}: {error}"
                    ) from error
            previous_joint_targets = command_joint_targets
            previous_base_targets = command_base_targets

            tactile_state = tactile_view.render()
            if output_dir is not None and (
                index % record_every == 0 or index == program.frame_count - 1
            ):
                _write_frame(
                    output_dir,
                    frame,
                    selected_env,
                    tactile_state,
                    deformable_view.read_state(),
                )

        if output_dir is not None:
            diagnostics = dict(provider.diagnostics)
            metadata = {
                "artifact": artifact.digest,
                "capture_graphs": provider.graph_captured,
                "device": device,
                "diagnostics": diagnostics,
                "dt": FRAME_DT,
                "frames": program.frame_count,
                "num_envs": num_envs,
                "preload_frames": CLOSE_MOTION_FRAMES,
                "physics_substeps": PHYSICS_SUBSTEPS,
                "selected_env": selected_env,
                "tactile_sensor_paths": list(program.tactile_sensor_paths),
            }
            (output_dir / "run.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    del context


def _summary(
    artifact: CompiledIpcSceneArtifact, program: AllegroGraspProgram
) -> Mapping[str, Any]:
    sensor = artifact.tactile_sensors[0]
    return {
        "artifact": artifact.digest,
        "deformable_bodies": len(program.deformable_body_paths),
        "frame_count": program.frame_count,
        "joint_count": len(program.joint_names),
        "phases": {"close": CLOSE_FRAMES, "lift": LIFT_FRAMES},
        "robot": program.robot_name,
        "sensor_count": len(program.tactile_sensor_paths),
        "sensor_resolution": list(sensor["resolution"]),
        "signals_2d": ["rgb", "depth", "normal", "marker_position", "marker_flow"],
        "signals_3d": [
            "contact_force",
            "contact_wrench",
            "deformable_position",
            "deformable_velocity",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--capture-graphs", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--selected-env", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.record_every <= 0:
        parser.error("--record-every must be positive")
    if not 0 <= args.selected_env < args.num_envs:
        parser.error("--selected-env must be within the environment batch")

    context, artifact, program = load_program(args.scene)
    print(json.dumps(_summary(artifact, program), indent=2, sort_keys=True))
    if args.validate_only:
        return
    run(
        context,
        artifact,
        program,
        num_envs=args.num_envs,
        device=args.device,
        capture_graphs=args.capture_graphs,
        output_dir=args.output_dir.expanduser().resolve() if args.output_dir else None,
        record_every=args.record_every,
        selected_env=args.selected_env,
    )


if __name__ == "__main__":
    main()
