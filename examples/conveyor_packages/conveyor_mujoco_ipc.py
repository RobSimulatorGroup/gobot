"""Headless two-hand mailer acceptance using MuJoCo Warp + libuipc."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np
# Load Torch's CUDA libraries before the native IPC module.
import torch

import gobot
from gobot.ipc import LibuipcBatchConfig, LibuipcConfig
from gobot.rl import CompiledMuJoCoIpcArtifact, MuJoCoIpcConfig, MuJoCoIpcProvider

from build_scene import (
    HAND_BASE_LINK_NAMES, HAND_JOINT_NAMES_BY_SIDE, HERE,
    LEAP_CONTACT_LINK_NAMES, LEAP_ROBOT_NAMES, SCENE_NAME, WORKTABLE_TOP_Z,
)
from conveyor_grasp_metrics import GraspMeasurements, GraspThresholds, proxy_surface_error
from conveyor_pinch_controller import ContactPinchController, PinchControlSettings
from conveyor_profile import (
    FIXED_DT, HAND_MOTION_SEGMENTS, HAND_STAGE_BLUE_FLIP_CONTACT_TARGETS,
    LEAP_BLUE_AIR_PINCH_TARGETS_BY_SIDE, cycle_phase,
    finger_close_fractions_at_tick, hand_controls_at_tick, smoothstep,
)


PACKAGE_NAMES = ("soft_mailer_blue", "soft_mailer_blue_fill")
TIP_NAMES = ("th_ds", "if_ds", "mf_ds", "rf_ds")
BLUE_SEGMENTS = []
for _segment in HAND_MOTION_SEGMENTS:
    BLUE_SEGMENTS.append(_segment)
    if _segment.phase == "blue_flip_settle":
        break
ACCEPTANCE_STEPS = sum(segment.duration for segment in BLUE_SEGMENTS)


def _walk(node: Any):
    yield node
    for child in node.children:
        yield from _walk(child)


def load_trial(scene_path: Path) -> tuple[Any, CompiledMuJoCoIpcArtifact, np.ndarray]:
    """Select an operation from the authored scene, without saving changes."""
    scene_path = scene_path.expanduser().resolve()
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(scene_path.parent))
        root = context.load_scene("res://" + scene_path.name)
        context.fixed_time_step = FIXED_DT
        keep = {"warehouse_frame", "conveyor", *LEAP_ROBOT_NAMES, *PACKAGE_NAMES}
        if not keep.issubset({child.name for child in root.children}):
            raise ValueError("scene is missing the two LEAP hands, blue mailer/fill, or worktable")
        for child in root.children:
            if child.name not in keep:
                root.remove_child(child, delete=True)
        for owner in root.children:
            links = [node for node in _walk(owner) if node.type_name == "Link3D"]
            for link in links:
                if owner.name in LEAP_ROBOT_NAMES and link.name not in LEAP_CONTACT_LINK_NAMES:
                    continue
                if not any(child.type_name == "CollisionShape3D" and not child.disabled
                           for child in link.children):
                    continue
                coupling = gobot.create_node("PhysicsCoupling", f"ipc_{owner.name}_{link.name}")
                coupling.target_body_path = "../" + link.path.removeprefix(root.path + "/")
                coupling.mode = (gobot.PhysicsCouplingMode.TwoWay if owner.name in LEAP_ROBOT_NAMES
                                 else gobot.PhysicsCouplingMode.OneWay)
                coupling.force_scale = coupling.torque_scale = 1.0
                root.add_child(coupling)
        shell = next(child for child in root.children if child.name == PACKAGE_NAMES[0])
        if not shell.self_collision_enabled:
            raise ValueError("acceptance requires authored shell self-contact")
        mesh = shell.surface_mesh
        triangles = np.asarray(mesh.triangles, dtype=np.int64).reshape(-1, 3)
        half = len(mesh.vertices) // 2
        face_triangles = triangles[np.all(triangles >= half, axis=1)]
        if not len(face_triangles):
            raise ValueError("blue mailer has no identifiable authored upper sheet")
        artifact = CompiledMuJoCoIpcArtifact.from_context(context)
        if artifact.ipc.deformable_attachments:
            raise ValueError("grasp acceptance forbids deformable attachments")
        return context, artifact, face_triangles
    except BaseException:
        context.clear_scene()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=HERE / SCENE_NAME)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--controller", choices=("contact", "open-loop"), default="contact")
    parser.add_argument("--grip-feedback", choices=("continuous", "fixed"), default="continuous")
    parser.add_argument("--grasp-height-offset", type=float, default=.006)
    parser.add_argument("--grasp-wait-steps", type=int, default=1000)
    parser.add_argument("--coupling-iterations", type=int, default=2)
    parser.add_argument("--newton-iterations", type=int, default=48)
    parser.add_argument("--line-search-iterations", type=int, default=16)
    parser.add_argument("--pinch-gap", type=float, default=0.0025,
                        help="Unloaded fingertip surface gap in meters; 0.0205 reproduces the authored pose.")
    parser.add_argument("--max-wall-seconds", type=float, default=1200.0)
    parser.add_argument("--module-path", default="")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path,
                        help="Optionally save measured states at the JSON trace boundaries.")
    parser.add_argument("--compile-only", action="store_true")
    return parser


def fit_pinch_targets(artifact: CompiledMuJoCoIpcArtifact, gap: float, *,
                      parallel_pads: bool = False) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    """Fit joint targets against the compiled Gobot fingertip collision boxes."""
    import mujoco
    from scipy.optimize import least_squares

    maximum_gap = .05 if parallel_pads else .0205
    if not np.isfinite(gap) or not 0.0005 <= gap <= maximum_gap:
        raise ValueError(f"pinch gap must be within [0.0005, {maximum_gap}] meters")
    model = mujoco.MjModel.from_xml_string(artifact.mujoco.content)
    data = mujoco.MjData(model)
    targets, offsets, diagnostics = [], [], []
    for side, name in enumerate(LEAP_ROBOT_NAMES):
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name + "_" + joint)
                  for joint in HAND_JOINT_NAMES_BY_SIDE[side]]
        if min(joints) < 0:
            raise ValueError("compiled scene is missing an authored LEAP joint")
        addresses = model.jnt_qposadr[joints]
        data.qpos[addresses[:6]] = HAND_STAGE_BLUE_FLIP_CONTACT_TARGETS[side]
        original = np.asarray(LEAP_BLUE_AIR_PINCH_TARGETS_BY_SIDE[side])
        data.qpos[addresses[6:]] = original
        geom_ids = [next(index for index in range(model.ngeom)
                         if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "")
                         .endswith(name + "_" + finger + "_tip_collision"))
                    for finger in ("mf", "th")]

        def geometry():
            mujoco.mj_kinematics(model, data)
            centers = data.geom_xpos[geom_ids].copy()
            delta = centers[0] - centers[1]
            distance = np.linalg.norm(delta)
            axis = delta / max(distance, 1.0e-12)
            rotations = data.geom_xmat[geom_ids].reshape(2, 3, 3)
            extent = sum(np.abs(axis @ rotation) @ model.geom_size[geom]
                         for geom, rotation in zip(geom_ids, rotations, strict=True))
            return centers.mean(axis=0), delta, float(distance - extent)

        center, original_delta, original_gap = geometry()
        slope = 0.0 if parallel_pads else original_delta[2] / original_delta[1]
        selected = [0, 2, 3, 12, 13, 14, 15]
        initial = original[selected]
        bounds = model.jnt_range[np.asarray(joints)[6:][selected]]

        def residual(values):
            pose = original.copy()
            for start in (0, 4, 8):
                pose[np.asarray([0, 2, 3]) + start] = values[:3]
            pose[12:] = values[3:]
            data.qpos[addresses[6:]] = pose
            _, delta, surface_gap = geometry()
            alignment = []
            if parallel_pads:
                thumb_rotation = data.geom_xmat[geom_ids[1]].reshape(3, 3)
                alignment = [10. * (sum(values[:3]) - np.pi / 2.),
                             * (10. * thumb_rotation[[0, 2], 0])]
            return np.concatenate((
                1000. * np.asarray([delta[0], delta[2] - slope * delta[1], surface_gap - gap]),
                alignment,
                .01 * (values - initial)))

        fit = least_squares(residual, initial, bounds=(bounds[:, 0], bounds[:, 1]),
                            max_nfev=500, xtol=1.0e-11, ftol=1.0e-11, gtol=1.0e-11)
        error = residual(fit.x)[:6 if parallel_pads else 3]
        if not fit.success or np.max(np.abs(error)) > .05:
            raise ValueError(f"{name} pinch fitting failed: scaled residual {error.tolist()}")
        pose = data.qpos[addresses[6:]].copy()
        targets.append(pose)
        offset = center - geometry()[0]
        offsets.append(offset)
        diagnostics.append({"robot": name, "authored_gap_meters": original_gap,
            "fitted_gap_meters": geometry()[2], "maximum_fit_error_meters": float(np.max(np.abs(error[:3]))) / 1000.,
            "pad_alignment_error": float(np.max(np.abs(error[3:]), initial=0.)) / 10.,
            "parallel_pads": parallel_pads,
            "fingertip_normals_world": (np.array([-1., 1., 1., 1.])[:, None]
                * (geometry()[1] / np.linalg.norm(geometry()[1]))).tolist(),
            "finger_targets_radians": pose.tolist(), "wrist_offset_meters": offset.tolist()})
    return targets, offsets, diagnostics


def contact_pinch_poses(artifact, gap):
    poses, diagnostics = [], []
    for opening in np.linspace(.05, gap, 17):
        targets, offsets, fits = fit_pinch_targets(artifact, float(opening), parallel_pads=True)
        commands = np.asarray([(*HAND_STAGE_BLUE_FLIP_CONTACT_TARGETS[side], *targets[side])
                               for side in range(2)])
        commands[:, :3] += np.asarray(offsets)
        poses.append(commands)
        diagnostics.append(fits)
    return np.asarray(poses), diagnostics


def collision_link_radii(artifact: CompiledMuJoCoIpcArtifact) -> dict[str, float]:
    """Bound each collision surface in its link frame, using the scene artifact."""
    radii = {}
    for robot in artifact.ipc.robots:
        for link in robot["links"]:
            radius = 0.0
            for shape in link["collision_shapes"]:
                if shape["disabled"]:
                    continue
                transform = np.asarray(shape["link_transform"]["matrix_row_major"]).reshape(4, 4)
                kind = shape["shape_type"]
                if kind == "box":
                    local_radius = np.linalg.norm(shape["size"]) / 2.0
                elif kind == "sphere":
                    local_radius = float(shape["radius"])
                elif kind in ("cylinder", "capsule"):
                    local_radius = float(shape["height"]) / 2.0 + float(shape["radius"])
                elif kind == "triangle_mesh":
                    points = np.frombuffer(artifact.ipc.blobs[shape["mesh_blob"]],
                        dtype="<f8", count=int(shape["vertex_count"]) * 3, offset=20).reshape(-1, 3)
                    local_radius = np.linalg.norm(points, axis=1).max()
                else:
                    raise ValueError(f"unsupported acceptance collision shape: {kind}")
                radius = max(radius, float(np.linalg.norm(transform[:3, 3])
                    + np.linalg.norm(transform[:3, :3], ord=2) * local_radius))
            radii[str(link["path"])] = radius
    return radii


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def joint_trajectory(
    steps: int, pinch_targets: list[np.ndarray], wrist_offsets: list[np.ndarray],
) -> np.ndarray:
    trajectory = np.asarray([hand_controls_at_tick(tick) for tick in range(steps)])
    for tick in range(steps):
        fraction = smoothstep(
            (tick - BLUE_SEGMENTS[0].duration) / BLUE_SEGMENTS[1].duration
        )
        for side, closure in enumerate(finger_close_fractions_at_tick(tick)):
            trajectory[tick, side, :3] += wrist_offsets[side] * fraction
            trajectory[tick, side, 6:] = pinch_targets[side] * closure
    return trajectory


def _timing_summary(samples: list[float]) -> dict[str, Any]:
    return {
        "samples": len(samples), "median": float(np.median(samples)),
        "p95": float(np.percentile(samples, 95)),
    }


def _record_failure(report: dict[str, Any], stage: str, error: Exception) -> None:
    message = f"{stage}: {type(error).__name__}: {error}"
    if report["error"] is None:
        report["error"] = message
    else:
        report.setdefault("secondary_errors", []).append(message)


def _capture_diagnostics(report: dict[str, Any], provider: Any) -> None:
    try:
        report["diagnostics"] = _jsonable(provider.diagnostics)
    except Exception as error:
        _record_failure(report, "diagnostics", error)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.grasp_wait_steps < 1 or args.grasp_wait_steps > 5000:
        raise ValueError("--grasp-wait-steps must be within [1, 5000]")
    if not np.isfinite(args.grasp_height_offset) or not -.02 <= args.grasp_height_offset <= .03:
        raise ValueError("--grasp-height-offset must be within [-.02, .03] meters")
    maximum_steps = ACCEPTANCE_STEPS + (args.grasp_wait_steps if args.controller == "contact" else 0)
    steps = maximum_steps if args.steps is None else args.steps
    if steps <= 0 or steps > maximum_steps:
        raise ValueError(f"--steps must be within [1, {maximum_steps}]")
    if not np.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0.0:
        raise ValueError("--max-wall-seconds must be finite and positive")
    thresholds = GraspThresholds()
    report: dict[str, Any] = {
        "schema_version": 1, "task": "blue_mailer_two_hand_pinch_flip",
        "provider": "MuJoCoWarp+libuipc", "num_envs": 1,
        "requested_controller": args.controller,
        "fixed_dt": FIXED_DT, "requested_steps": steps,
        "required_steps": ACCEPTANCE_STEPS, "passed": False,
        "scene": str(args.scene.resolve()),
        "source_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        "script_sha256": {
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in (Path(__file__).name, "conveyor_grasp_metrics.py", "conveyor_profile.py",
                         "conveyor_pinch_controller.py")
        },
        "thresholds": asdict(thresholds),
        "phase_samples": [], "error": None,
        "measurement_scope": (
            "IPC fingertip reaction wrenches; authored upper-sheet normal; all package nodes; "
            "proxy surface displacement bound. Table penetration is vertex-based, "
            "not a general triangle-intersection or visual-mesh penetration certificate. "
            "Precontact drift uses the shell vertex centroid, including initial settling."
        ),
        "timing_scope": (
            "Synchronized provider.step latency, including coupling iterations; excludes "
            "command submission and acceptance readback. No warmup samples excluded. "
            "Single-environment task diagnostic, not a batched throughput benchmark."
        ),
    }
    context = provider = metrics = workspace_owner = controller = None
    step_times: list[float] = []
    phase_times: dict[str, list[float]] = {}
    started = None
    try:
        context, artifact, face_triangles = load_trial(args.scene)
        report["artifact_digest"] = artifact.digest
        report["coupled_bodies"] = [mapping.to_mapping() for mapping in artifact.coupled_bodies]
        report["deformables"] = [dict(body) for body in artifact.ipc.deformable_bodies]
        report["collision_ownership"] = dict(artifact.collision_ownership)
        pinch_targets, wrist_offsets, report["pinch_fit"] = fit_pinch_targets(artifact, args.pinch_gap)
        if args.controller == "contact":
            control_poses, report["contact_pose_fits"] = contact_pinch_poses(artifact, args.pinch_gap)
        if args.compile_only:
            report["status"] = "compiled_not_simulated"
        else:
            report["gpu"] = torch.cuda.get_device_name(0)
            report["versions"] = {
                package: version(package) for package in ("torch", "warp-lang", "mujoco-warp", "mujoco")
            }
            free, total = torch.cuda.mem_get_info(0)
            memory = report["gpu_memory_bytes"] = {
                "total": total, "before_provider": total - free,
                "sampled_peak_device_used": total - free,
                "scope": "Whole-device used memory, sampled at phase/trace boundaries; includes other processes.",
            }
            workspace_owner = tempfile.TemporaryDirectory(prefix="gobot-conveyor-ipc-")
            build_started = time.perf_counter()
            provider = MuJoCoIpcProvider(
                artifact,
                config=MuJoCoIpcConfig(num_envs=1, environments_per_shard=1,
                                     coupling_iterations=args.coupling_iterations),
                libuipc_config=LibuipcBatchConfig(
                    solver=LibuipcConfig(fixed_time_step=FIXED_DT, workspace=workspace_owner.name,
                        module_path=args.module_path, contact_activation_distance=8.0e-4,
                        friction_coefficient=1.0, kinematic_strength=100.0),
                    environments_per_shard=1, newton_max_iterations=args.newton_iterations,
                    line_search_max_iterations=args.line_search_iterations,
                    strict_convergence=True,
                ),
                mujoco_options={"nconmax": 512, "njmax": 2048, "overflow_check_interval": 1},
            )
            provider.synchronize()
            report["build_seconds"] = time.perf_counter() - build_started
            memory["after_provider_build"] = total - torch.cuda.mem_get_info(0)[0]
            memory["sampled_peak_device_used"] = max(
                memory["sampled_peak_device_used"], memory["after_provider_build"]
            )
            report["solver_config"] = asdict(provider.ipc_solver.config)
            report["coupling_config"] = asdict(provider.config)
            report["capabilities"] = asdict(provider.capabilities)
            arrays = provider.arrays
            views = [provider.create_robot_view(robot_name=name, base_link=base,
                     joint_names=joints, link_names=LEAP_CONTACT_LINK_NAMES)
                     for name, base, joints in zip(LEAP_ROBOT_NAMES, HAND_BASE_LINK_NAMES,
                                                  HAND_JOINT_NAMES_BY_SIDE, strict=True)]
            trajectory = None if args.controller == "contact" else torch.as_tensor(
                joint_trajectory(ACCEPTANCE_STEPS, pinch_targets, wrist_offsets),
                device="cuda:0", dtype=arrays["ctrl"].dtype,
            )
            command_buffer = torch.empty((2, 22), device="cuda:0", dtype=arrays["ctrl"].dtype)
            shell_body = next(body for body in provider.ipc_solver.deformable_bodies
                              if str(body["path"]).endswith("/" + PACKAGE_NAMES[0]))
            offset = int(shell_body["element_offset"])
            shell_slice = slice(offset, offset + int(shell_body["element_count"]))
            shell = arrays["ipc_positions"][0, shell_slice].cpu().numpy().copy()
            metrics = GraspMeasurements(
                initial_shell=shell,
                face_triangles=face_triangles, table_height=WORKTABLE_TOP_Z,
                fixed_dt=FIXED_DT, required_steps=ACCEPTANCE_STEPS, thresholds=thresholds,
            )
            if args.controller == "contact":
                control_settings = PinchControlSettings(
                    maximum_wait_steps=args.grasp_wait_steps,
                    crest_height_offset_meters=args.grasp_height_offset,
                    continuous_finger_feedback=args.grip_feedback == "continuous",
                )
                report["control_settings"] = asdict(control_settings)
                controller = ContactPinchController(
                    control_poses, shell, face_triangles, BLUE_SEGMENTS, FIXED_DT, control_settings,
                    fingertip_normals=np.asarray([fit["fingertip_normals_world"]
                                                 for fit in report["contact_pose_fits"][-1]]),
                )
            hand_indices = [mapping.ipc_body_index for mapping in artifact.coupled_bodies
                            if mapping.robot_name in LEAP_ROBOT_NAMES]
            tip_indices = [[next(mapping.ipc_body_index for mapping in artifact.coupled_bodies
                                 if mapping.robot_name == name and mapping.link_name == tip)
                            for tip in TIP_NAMES] for name in LEAP_ROBOT_NAMES]
            link_radii = collision_link_radii(artifact)
            radii = np.asarray([link_radii[mapping.ipc_path]
                                for mapping in artifact.coupled_bodies
                                if mapping.robot_name in LEAP_ROBOT_NAMES])
            report["hand_collision_radii_meters"] = radii.tolist()
            palm_indices = [next(mapping.ipc_body_index for mapping in artifact.coupled_bodies
                                 if mapping.robot_name == name and mapping.link_name == "palm")
                            for name in LEAP_ROBOT_NAMES]
            initial_palms = [None, None]
            started = time.perf_counter()
            previous_phase = ""
            for tick in range(steps):
                if time.perf_counter() - started > args.max_wall_seconds:
                    raise TimeoutError("acceptance wall-time budget exceeded; trial is incomplete")
                phase = cycle_phase(tick) if controller is None else controller.phase
                if controller is not None:
                    command_buffer.copy_(torch.from_numpy(controller.command(shell)))
                for side, view in enumerate(views):
                    command = trajectory[tick, side] if controller is None else command_buffer[side]
                    view.set_position_targets(command.unsqueeze(0))
                step_started = time.perf_counter()
                provider.step()
                provider.synchronize()
                step_times.append((time.perf_counter() - step_started) * 1000.0)
                phase_times.setdefault(phase, []).append(step_times[-1])
                vertices = arrays["ipc_positions"][0].cpu().numpy()
                shell = vertices[shell_slice]
                velocities = arrays["ipc_velocities"][0].cpu().numpy()
                if not all(bool(torch.isfinite(arrays[key]).all()) for key in ("qpos", "qvel")):
                    raise ValueError("non-finite MuJoCo articulation state")
                wrenches = arrays["ipc_affine_contact_wrenches"][0].cpu().numpy()
                targets = arrays["ipc_affine_targets"][0].cpu().numpy()
                proxies = arrays["ipc_affine_transforms"][0].cpu().numpy()
                wrist_rotation = 0.0
                for side, palm_index in enumerate(palm_indices):
                    rotation = targets[palm_index, :3, :3]
                    if initial_palms[side] is None:
                        initial_palms[side] = rotation.copy()
                    cosine = (np.trace(rotation @ initial_palms[side].T) - 1.0) / 2.0
                    wrist_rotation = max(wrist_rotation, float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
                sample = metrics.observe(
                    phase=phase, shell=vertices[shell_slice], vertices=vertices,
                    velocities=velocities, fingertip_forces=wrenches[tip_indices, :3],
                    hand_contact_force=float(np.linalg.norm(wrenches[hand_indices, :3], axis=1).max()),
                    proxy_error=proxy_surface_error(targets[hand_indices], proxies[hand_indices], radii),
                    wrist_rotation_degrees=wrist_rotation,
                )
                control_finished = False
                if controller is not None:
                    controller.observe(wrenches[tip_indices, :3], sample["proxy_surface_error_bound_meters"])
                    sample["controller"] = controller.diagnostics()
                    control_finished = controller.completed or controller.failure is not None
                if (sample["phase"] != previous_phase or (tick + 1) % 50 == 0
                        or tick + 1 == steps or control_finished):
                    diagnostics = _jsonable(provider.diagnostics)
                    sample["diagnostics"] = diagnostics
                    report["phase_samples"].append(sample)
                    sample["fingertip_force_vectors_newtons"] = wrenches[tip_indices, :3].tolist()
                    sample["shell_bounds_meters"] = [vertices[shell_slice].min(axis=0).tolist(),
                                                      vertices[shell_slice].max(axis=0).tolist()]
                    if args.snapshot_dir is not None:
                        args.snapshot_dir.mkdir(parents=True, exist_ok=True)
                        path = args.snapshot_dir / f"step-{tick + 1:05d}.npz"
                        np.savez_compressed(
                            path, vertices=vertices, velocities=velocities,
                            shell_range=np.array([shell_slice.start, shell_slice.stop]),
                            face_triangles=face_triangles, targets=targets, proxies=proxies,
                            wrenches=wrenches, qpos=arrays["qpos"][0].cpu().numpy(),
                        )
                        sample["snapshot"] = str(path)
                    print(json.dumps({key: value for key, value in sample.items() if key != "diagnostics"}),
                          file=sys.stderr, flush=True)
                    previous_phase = sample["phase"]
                    memory["sampled_peak_device_used"] = max(
                        memory["sampled_peak_device_used"], total - torch.cuda.mem_get_info(0)[0]
                    )
                if control_finished:
                    break
    except Exception as error:
        _record_failure(report, "trial", error)
    finally:
        if started is not None:
            report["trial_wall_seconds"] = time.perf_counter() - started
        if provider is not None:
            _capture_diagnostics(report, provider)
        # Keep the scene and workspace alive until native solvers have closed.
        for resource, method in ((provider, "close"), (context, "clear_scene"),
                                 (workspace_owner, "cleanup")):
            if resource is not None:
                try:
                    getattr(resource, method)()
                except Exception as error:
                    _record_failure(report, method, error)
        if metrics is not None:
            exact = report.get("capabilities", {}).get("exact_contact_wrench", False)
            completed = None if controller is None else controller.completed
            report.update(metrics.result(error=report["error"], exact_feedback=exact,
                                         sequence_completed=completed))
        if controller is not None:
            report["controller"] = controller.diagnostics()
            report["task_failure"] = controller.failure
        if step_times:
            report["step_timing_ms"] = _timing_summary(step_times)
            report["phase_step_timing_ms"] = {
                phase: _timing_summary(samples) for phase, samples in phase_times.items()
            }
    return _jsonable(report)


def main() -> int:
    args = _parser().parse_args()
    report = run(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "error": report["error"],
                      "task_failure": report.get("task_failure"),
                      "report": str(args.report)}, allow_nan=False))
    return 0 if report["passed"] or (args.compile_only and report["error"] is None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
