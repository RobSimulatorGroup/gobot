"""Run the native SuperDex conveyor workcell without the editor."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
import time
from typing import Any, Callable, Sequence

import numpy as np

import gobot

from build_scene import (
    BELT_CENTER_X,
    BELT_CENTER_Y,
    BELT_SURFACE_LENGTH,
    BELT_TOP_Z,
    BELT_WIDTH,
    HAND_JOINT_NAMES_BY_SIDE,
    HAND_STAGE_JOINT_NAMES_BY_SIDE,
    HERE,
    LEAP_CONTACT_LINK_NAMES,
    LEAP_ROBOT_NAMES,
    RIGID_BOX_SPECS,
    SCENE_NAME,
    SOFT_PACKAGE_SPECS,
    build_scene,
)
from conveyor_forces import ConveyorForceModel, DeformableConveyorForceModel
from conveyor_profile import (
    CYCLE_TICKS,
    FIXED_DT,
    HAND_MOTION_SEGMENTS,
    SOFT_PACKAGE_DAMPING_RATES,
    SOFT_PACKAGE_MASSES,
    belt_speed_at_tick,
    cycle_phase,
    finger_close_fraction_at_tick,
    hand_controls_at_tick,
    quality_profile,
    soft_damping_scale_at_tick,
)


BELT_ROBOT_NAME = "conveyor"
BELT_LINK_NAME = "belt_surface"
BELT_DRIVE_FRICTION = 0.92


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run two floating LEAP Hands and rigid/thin-shell/volumetric "
            "packages in one native SuperDex physics world."
        )
    )
    parser.add_argument("--scene", type=Path, default=HERE / SCENE_NAME)
    parser.add_argument("--rebuild-scene", action="store_true")
    parser.add_argument(
        "--quality", choices=("interactive", "accurate"), default="interactive"
    )
    parser.add_argument("--steps", type=int, default=CYCLE_TICKS)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--execution", choices=("cpu", "cuda"), default="cpu"
    )
    parser.add_argument(
        "--linear-solver", choices=("auto", "cg", "gmres"), default="auto"
    )
    parser.add_argument("--trace-force-flow", action="store_true")
    parser.add_argument("--phase-diagnostics", action="store_true")
    parser.add_argument("--solver-timings", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.num_envs != 1:
        raise ValueError("native SuperDex currently supports exactly one environment")


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
            raise RuntimeError(f"scene subtree has duplicate node name {node.name!r}")
        result[node.name] = node
        pending.extend(node.children)
    return result


def _robot_table(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(robot["name"]): robot for robot in state["robots"]}


def _deformable_table(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(body.get("name", "")): body for body in state["deformables"]
    }


def _link_state(
    state: dict[str, Any], robot_name: str, link_name: str
) -> dict[str, Any]:
    robot = _robot_table(state)[robot_name]
    return next(link for link in robot["links"] if link["name"] == link_name)


def _deformable_center(body: dict[str, Any]) -> np.ndarray:
    return np.asarray(body["world_vertices"], dtype=np.float64).mean(axis=0)


def _deformable_bounds(body: dict[str, Any]) -> dict[str, list[float]]:
    vertices = np.asarray(body["world_vertices"], dtype=np.float64)
    return {
        "lower": [float(value) for value in vertices.min(axis=0)],
        "upper": [float(value) for value in vertices.max(axis=0)],
    }


def _hand_diagnostics(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    robots = _robot_table(state)
    result: dict[str, dict[str, Any]] = {}
    for robot_name, stage_names in zip(
        LEAP_ROBOT_NAMES, HAND_STAGE_JOINT_NAMES_BY_SIDE, strict=True
    ):
        robot = robots[robot_name]
        joints = {str(joint["name"]): joint for joint in robot["joints"]}
        links = {str(link["name"]): link for link in robot["links"]}
        result[robot_name] = {
            "stage_max_tracking_error": max(
                abs(float(joints[name]["tracking_error"]))
                for name in stage_names
            ),
            "stage_max_speed": max(
                abs(float(joints[name]["velocity"])) for name in stage_names
            ),
            "contact_link_positions_meters": {
                name: [
                    float(value)
                    for value in links[name]["global_transform"]["position"]
                ]
                for name in LEAP_CONTACT_LINK_NAMES
            },
        }
    return result


def _group_center(
    bodies: dict[str, dict[str, Any]], names: Sequence[str]
) -> np.ndarray:
    vertices = np.concatenate(
        tuple(
            np.asarray(bodies[name]["world_vertices"], dtype=np.float64)
            for name in names
        ),
        axis=0,
    )
    return vertices.mean(axis=0)


def _shell_axis(body: dict[str, Any]) -> np.ndarray:
    vertices = np.asarray(body["world_vertices"], dtype=np.float64)
    if len(vertices) % 2:
        raise RuntimeError("closed mailer shell must have two equal vertex sheets")
    half = len(vertices) // 2
    axis = vertices[half:].mean(axis=0) - vertices[:half].mean(axis=0)
    length = float(np.linalg.norm(axis))
    if length <= 1.0e-12:
        raise RuntimeError("closed mailer shell layer axis is degenerate")
    return axis / length


def _quaternion_angle_degrees(current: Any, initial: Any) -> float:
    left = np.asarray(current, dtype=np.float64)
    right = np.asarray(initial, dtype=np.float64)
    dot = float(np.clip(abs(np.dot(left, right)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def _contact_hands(
    contacts: Sequence[dict[str, Any]], target_names: set[str]
) -> set[str]:
    hands: set[str] = set()
    for contact in contacts:
        first = (str(contact["robot_name"]), str(contact["link_name"]))
        second = (
            str(contact["other_robot_name"]),
            str(contact["other_link_name"]),
        )
        for hand, target in ((first, second), (second, first)):
            if hand[0] in LEAP_ROBOT_NAMES and (
                target[0] in target_names or target[1] in target_names
            ):
                hands.add(hand[0])
    return hands


def _max_penetration(contacts: Sequence[dict[str, Any]]) -> float:
    return max(
        (max(0.0, -float(contact["distance"])) for contact in contacts),
        default=0.0,
    )


def _contact_pair_name(contact: dict[str, Any]) -> str:
    def endpoint(robot_key: str, link_key: str) -> str:
        robot = str(contact[robot_key])
        link = str(contact[link_key])
        return f"{robot}/{link}" if robot else link

    return " <-> ".join(
        sorted(
            (
                endpoint("robot_name", "link_name"),
                endpoint("other_robot_name", "other_link_name"),
            )
        )
    )


def _assert_finite_state(state: dict[str, Any]) -> None:
    values: list[np.ndarray] = []
    for robot in state["robots"]:
        for link in robot["links"]:
            values.extend(
                (
                    np.asarray(link["global_transform"]["matrix"]),
                    np.asarray(link["linear_velocity"]),
                    np.asarray(link["angular_velocity"]),
                )
            )
        for joint in robot["joints"]:
            values.append(
                np.asarray(
                    (joint["position"], joint["velocity"], joint["effort"])
                )
            )
    for body in state["deformables"]:
        values.extend(
            (
                np.asarray(body["local_vertices"]),
                np.asarray(body["local_velocities"]),
                np.asarray(body["contact_forces_world"]),
            )
        )
    if any(value.size and not np.isfinite(value).all() for value in values):
        raise RuntimeError("SuperDex produced non-finite conveyor state")


def _phase_end_ticks() -> dict[int, str]:
    result: dict[int, str] = {}
    end = 0
    for segment in HAND_MOTION_SEGMENTS:
        end += segment.duration
        result[end] = segment.phase
    return result


def _linear_solver(name: str) -> Any:
    return {
        "auto": gobot.SuperDexLinearSolver.Auto,
        "cg": gobot.SuperDexLinearSolver.CG,
        "gmres": gobot.SuperDexLinearSolver.GMRES,
    }[name]


def _execution_mode(name: str) -> Any:
    return {
        "cpu": gobot.SuperDexExecutionMode.Cpu,
        "cuda": gobot.SuperDexExecutionMode.Cuda,
    }[name]


def _apply_hand_targets(
    hand_joints: Sequence[Sequence[Any]], tick: int
) -> None:
    controls = hand_controls_at_tick(tick)
    for joints, targets in zip(hand_joints, controls, strict=True):
        for joint, target in zip(joints, targets, strict=True):
            joint.set_position_target(float(target))


def _reset_error(
    initial: dict[str, Any], restored: dict[str, Any]
) -> float:
    error = 0.0
    initial_robots = _robot_table(initial)
    restored_robots = _robot_table(restored)
    for robot_name, robot in initial_robots.items():
        restored_links = {
            str(link["name"]): link for link in restored_robots[robot_name]["links"]
        }
        for link in robot["links"]:
            current = np.asarray(link["global_transform"]["matrix"])
            reset = np.asarray(
                restored_links[str(link["name"])]["global_transform"]["matrix"]
            )
            error = max(error, float(np.max(np.abs(current - reset))))
    initial_bodies = _deformable_table(initial)
    restored_bodies = _deformable_table(restored)
    for name, body in initial_bodies.items():
        current = np.asarray(body["local_vertices"])
        reset = np.asarray(restored_bodies[name]["local_vertices"])
        error = max(error, float(np.max(np.abs(current - reset))))
    return error


def _step_with_diagnostics(
    context: Any,
    tick: int,
    control_tick: int,
    failure_observer: Callable[[dict[str, Any]], None] | None,
) -> float:
    started = time.perf_counter()
    try:
        context.step_once()
    except Exception as exc:
        if failure_observer is not None:
            # Capture before run() clears the failed world; never read invalid vertices.
            failure = {
                "tick": tick,
                "control_tick": control_tick,
                "phase": cycle_phase(control_tick),
                "step_once_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
            try:
                failure["solver"] = context.get_solver_diagnostics()
            except Exception as diagnostic_error:
                failure["diagnostic_error"] = str(diagnostic_error)
            failure_observer(failure)
        raise
    return time.perf_counter() - started


def run(
    args: argparse.Namespace,
    *,
    tick_observer: Callable[[dict[str, Any]], bool] | None = None,
    failure_observer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    _validate_args(args)
    scene_path = args.scene.expanduser().resolve()
    if args.rebuild_scene:
        scene_path = build_scene(scene_path.parent)
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)

    profile = quality_profile(args.quality)
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(scene_path.parent))
        root = context.load_scene("res://" + scene_path.name)
        context.fixed_time_step = FIXED_DT
        context.max_sub_steps = 1
        settings = context.get_superdex_solver_settings()
        settings.update(
            {
                "execution_mode": _execution_mode(args.execution),
                "linear_solver": _linear_solver(args.linear_solver),
                "newton_iterations": profile.newton_max_iterations,
                "line_search_iterations": profile.line_search_max_iterations,
                "linear_iterations": -1,
                "substeps": 1,
                "record_deformable_contact_forces": True,
                "record_solver_timings": (
                    tick_observer is not None or getattr(args, "solver_timings", False)
                ),
            }
        )
        context.set_superdex_solver_settings(settings)
        context.build_world(gobot.PhysicsBackendType.SuperDex)

        root_nodes = _nodes_by_name(root, allow_duplicate_names=True)
        hand_joints = tuple(
            tuple(
                _nodes_by_name(root_nodes[robot_name])[joint_name]
                for joint_name in joint_names
            )
            for robot_name, joint_names in zip(
                LEAP_ROBOT_NAMES, HAND_JOINT_NAMES_BY_SIDE, strict=True
            )
        )
        rigid_names = tuple(str(spec["name"]) for spec in RIGID_BOX_SPECS)
        rigid_masses = tuple(float(spec["mass"]) for spec in RIGID_BOX_SPECS)
        soft_names = tuple(str(spec["name"]) for spec in SOFT_PACKAGE_SPECS)
        rigid_forces = ConveyorForceModel(
            context,
            rigid_names,
            rigid_masses,
            belt_robot=BELT_ROBOT_NAME,
            belt_link=BELT_LINK_NAME,
            friction_coefficient=BELT_DRIVE_FRICTION,
            fixed_dt=FIXED_DT,
        )
        soft_forces = DeformableConveyorForceModel(
            context,
            soft_names,
            SOFT_PACKAGE_MASSES,
            friction_coefficient=BELT_DRIVE_FRICTION,
            fixed_dt=FIXED_DT,
            belt_half_length=0.5 * BELT_SURFACE_LENGTH,
            belt_half_width=0.5 * BELT_WIDTH,
            belt_top=BELT_TOP_Z,
            belt_center_x=BELT_CENTER_X,
            belt_center_y=BELT_CENTER_Y,
            velocity_damping_rates=SOFT_PACKAGE_DAMPING_RATES,
        )

        for _ in range(args.warmup_steps):
            state = context.get_physics_state_view()
            _apply_hand_targets(hand_joints, 0)
            rigid_forces.apply(0.0, state)
            soft_forces.apply(0.0, state=state)
            context.step_once()
        context.clear_external_forces()
        context.reset_simulation()

        initial_state = context.get_physics_state_view()
        _assert_finite_state(initial_state)
        initial_bodies = _deformable_table(initial_state)
        initial_shell_axes = {
            "soft_mailer_blue": _shell_axis(
                initial_bodies["soft_mailer_blue"]
            ),
            "soft_pouch_yellow": _shell_axis(
                initial_bodies["soft_pouch_yellow"]
            ),
        }
        initial_carton = _link_state(
            initial_state, "carton_small", "carton_small"
        )["global_transform"]["quaternion"]
        parcel_groups = {
            "blue_mailer": ("soft_mailer_blue", "soft_mailer_blue_fill"),
            "yellow_pouch": ("soft_pouch_yellow", "soft_pouch_yellow_fill"),
            "carton_small": ("carton_small",),
        }
        initial_centers = {
            "blue_mailer": _group_center(
                initial_bodies, parcel_groups["blue_mailer"]
            ),
            "yellow_pouch": _group_center(
                initial_bodies, parcel_groups["yellow_pouch"]
            ),
            "carton_small": np.asarray(
                _link_state(initial_state, "carton_small", "carton_small")
                ["global_transform"]["position"]
            ),
        }

        latency_samples: list[float] = []
        belt_travel = 0.0
        peak_penetration = 0.0
        peak_penetration_by_pair: dict[str, float] = {}
        peak_penetration_contact: dict[str, Any] | None = None
        peak_rigid_drive = np.zeros(len(rigid_names), dtype=np.float64)
        peak_soft_drive = 0.0
        maximum_flip = {
            "blue_mailer": 0.0,
            "yellow_pouch": 0.0,
            "carton_small": 0.0,
        }
        simultaneous_hand_contact = {name: False for name in parcel_groups}
        hand_contact_frames = {
            name: {hand: 0 for hand in LEAP_ROBOT_NAMES}
            for name in parcel_groups
        }
        first_hand_contact_tick = {
            name: {hand: None for hand in LEAP_ROBOT_NAMES}
            for name in parcel_groups
        }
        last_hand_contact_tick = {
            name: {hand: None for hand in LEAP_ROBOT_NAMES}
            for name in parcel_groups
        }
        maximum_simultaneous_hands = {name: 0 for name in parcel_groups}
        hand_contact_started = {name: False for name in parcel_groups}
        current_contacting_hands = {
            name: [] for name in parcel_groups
        }
        precontact_horizontal_drift = {name: 0.0 for name in parcel_groups}
        force_trace: list[dict[str, Any]] = []
        phase_trace: list[dict[str, Any]] = []
        phase_end_ticks = _phase_end_ticks()

        final_state = initial_state
        centers = initial_centers
        for tick in range(args.steps):
            tick_started = time.perf_counter()
            control_tick = min(tick, CYCLE_TICKS - 1)
            state = context.get_physics_state_view()
            state_view_seconds = time.perf_counter() - tick_started
            speed = float(belt_speed_at_tick(control_tick))
            _apply_hand_targets(hand_joints, control_tick)
            rigid_drive = rigid_forces.apply(speed, state)
            soft_forces.apply(
                speed,
                damping_scale=soft_damping_scale_at_tick(control_tick),
                state=state,
            )
            started = time.perf_counter()
            command_seconds = started - tick_started - state_view_seconds
            latency_samples.append(_step_with_diagnostics(
                context, tick + 1, control_tick, failure_observer
            ))
            belt_travel += speed * FIXED_DT
            read_started = time.perf_counter()
            final_state = context.get_physics_state_view()
            state_view_seconds += time.perf_counter() - read_started
            _assert_finite_state(final_state)
            for contact in final_state["contacts"]:
                penetration = max(0.0, -float(contact["distance"]))
                pair_name = _contact_pair_name(contact)
                peak_penetration_by_pair[pair_name] = max(
                    peak_penetration_by_pair.get(pair_name, 0.0),
                    penetration,
                )
                if penetration <= peak_penetration:
                    continue
                peak_penetration = penetration
                peak_penetration_contact = {
                    "tick": tick + 1,
                    "robot": str(contact["robot_name"]),
                    "link": str(contact["link_name"]),
                    "other_robot": str(contact["other_robot_name"]),
                    "other_link": str(contact["other_link_name"]),
                    "distance_meters": float(contact["distance"]),
                }
            peak_rigid_drive = np.maximum(
                peak_rigid_drive, np.abs(rigid_drive)
            )
            peak_soft_drive = max(
                peak_soft_drive,
                max(
                    (
                        float(np.max(np.abs(values)))
                        for values in soft_forces.drive_force.values()
                    ),
                    default=0.0,
                ),
            )

            bodies = _deformable_table(final_state)
            current_flip = {
                "blue_mailer": math.degrees(
                    math.acos(
                        float(
                            np.clip(
                                np.dot(
                                    _shell_axis(bodies["soft_mailer_blue"]),
                                    initial_shell_axes["soft_mailer_blue"],
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                ),
                "yellow_pouch": math.degrees(
                    math.acos(
                        float(
                            np.clip(
                                np.dot(
                                    _shell_axis(bodies["soft_pouch_yellow"]),
                                    initial_shell_axes["soft_pouch_yellow"],
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                ),
                "carton_small": _quaternion_angle_degrees(
                    _link_state(final_state, "carton_small", "carton_small")
                    ["global_transform"]["quaternion"],
                    initial_carton,
                ),
            }
            for name, angle in current_flip.items():
                maximum_flip[name] = max(maximum_flip[name], angle)

            centers = {
                "blue_mailer": _group_center(
                    bodies, parcel_groups["blue_mailer"]
                ),
                "yellow_pouch": _group_center(
                    bodies, parcel_groups["yellow_pouch"]
                ),
                "carton_small": np.asarray(
                    _link_state(final_state, "carton_small", "carton_small")
                    ["global_transform"]["position"]
                ),
            }
            for name, targets in parcel_groups.items():
                hands = _contact_hands(final_state["contacts"], set(targets))
                current_contacting_hands[name] = sorted(hands)
                maximum_simultaneous_hands[name] = max(
                    maximum_simultaneous_hands[name], len(hands)
                )
                for hand in hands:
                    hand_contact_frames[name][hand] += 1
                    if first_hand_contact_tick[name][hand] is None:
                        first_hand_contact_tick[name][hand] = tick + 1
                    last_hand_contact_tick[name][hand] = tick + 1
                if len(hands) == 2:
                    simultaneous_hand_contact[name] = True
                hand_contact_started[name] |= bool(hands)
                if not hand_contact_started[name]:
                    drift = float(
                        np.linalg.norm(
                            (centers[name] - initial_centers[name])[:2]
                        )
                    )
                    precontact_horizontal_drift[name] = max(
                        precontact_horizontal_drift[name], drift
                    )

            if args.trace_force_flow:
                force_trace.append(
                    {
                        "tick": tick + 1,
                        "phase": cycle_phase(control_tick),
                        "rigid_drive_resultant_newtons": float(
                            np.sum(rigid_drive)
                        ),
                        "soft_drive_resultant_newtons": float(
                            sum(
                                np.sum(values)
                                for values in soft_forces.drive_force.values()
                            )
                        ),
                    }
                )
            if args.phase_diagnostics and tick + 1 in phase_end_ticks:
                phase_trace.append(
                    {
                        "phase": phase_end_ticks[tick + 1],
                        "tick": tick + 1,
                        "flip_degrees": current_flip.copy(),
                        "centers_meters": {
                            name: [float(value) for value in center]
                            for name, center in centers.items()
                        },
                        "hand_diagnostics": _hand_diagnostics(final_state),
                        "deformable_bounds_meters": {
                            name: _deformable_bounds(body)
                            for name, body in bodies.items()
                        },
                        "contact_count": len(final_state["contacts"]),
                        "contacting_hands": {
                            name: list(hands)
                            for name, hands in current_contacting_hands.items()
                        },
                    }
                )

            # Observers consume completed physics ticks, never render frames or authored poses.
            if tick_observer is not None:
                sample = {
                    "tick": tick + 1,
                    "control_tick": control_tick,
                    "phase": cycle_phase(control_tick),
                    "simulation_time_seconds": float(final_state["simulation_time"]),
                    "step_once_seconds": latency_samples[-1],
                    "command_seconds": command_seconds,
                    "state_view_seconds": state_view_seconds,
                    "observed_tick_seconds": time.perf_counter() - tick_started,
                    "solver": context.get_solver_diagnostics(),
                    "contacting_hands": {
                        name: list(hands) for name, hands in current_contacting_hands.items()
                    },
                    "directed_contact_count": len(final_state["contacts"]),
                    "max_penetration_meters": _max_penetration(final_state["contacts"]),
                    "flip_degrees": current_flip.copy(),
                }
                if tick == 0:
                    sample["scene_complexity"] = {
                        "robots": len(final_state["robots"]),
                        "links": sum(len(robot["links"]) for robot in final_state["robots"]),
                        "joints": sum(len(robot["joints"]) for robot in final_state["robots"]),
                        "deformable_vertices": {
                            name: len(body["local_vertices"]) for name, body in bodies.items()
                        },
                    }
                if not tick_observer(sample):
                    break

        completed_steps = len(latency_samples)
        final_bodies = _deformable_table(final_state)
        final_flip = {
            "blue_mailer": math.degrees(
                math.acos(
                    float(
                        np.clip(
                            np.dot(
                                _shell_axis(final_bodies["soft_mailer_blue"]),
                                initial_shell_axes["soft_mailer_blue"],
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                )
            ),
            "yellow_pouch": math.degrees(
                math.acos(
                    float(
                        np.clip(
                            np.dot(
                                _shell_axis(final_bodies["soft_pouch_yellow"]),
                                initial_shell_axes["soft_pouch_yellow"],
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                )
            ),
            "carton_small": _quaternion_angle_degrees(
                _link_state(final_state, "carton_small", "carton_small")
                ["global_transform"]["quaternion"],
                initial_carton,
            ),
        }
        diagnostics = context.get_solver_diagnostics()
        context.clear_external_forces()
        context.reset_simulation()
        restored_state = context.get_physics_state_view()
        reset_max_error = _reset_error(initial_state, restored_state)
        latency_sorted = sorted(latency_samples)
        p95_index = max(
            0, math.ceil(0.95 * len(latency_sorted)) - 1
        )
        elapsed = sum(latency_samples)
        return {
            "backend": "SuperDex",
            "experimental": True,
            "quality": profile.name,
            "execution_requested": args.execution,
            "execution_device": diagnostics["execution_device"],
            "device_native": bool(diagnostics["device_native"]),
            "graph_capture": bool(diagnostics["graph_capture"]),
            "environment_batch": False,
            "masked_reset": False,
            "environments": 1,
            "steps": completed_steps,
            "warmup_steps": args.warmup_steps,
            "elapsed_seconds": elapsed,
            "steps_per_second": completed_steps / elapsed if elapsed else 0.0,
            "median_step_latency_seconds": (
                median(latency_samples) if latency_samples else 0.0
            ),
            "p95_step_latency_seconds": (
                latency_sorted[p95_index] if latency_sorted else 0.0
            ),
            "belt_surface_commanded_travel_meters": belt_travel,
            "final_flip_degrees": final_flip,
            "maximum_flip_degrees": maximum_flip,
            "simultaneous_hand_contact": simultaneous_hand_contact,
            "maximum_simultaneous_hands": maximum_simultaneous_hands,
            "hand_contact_frames": hand_contact_frames,
            "first_hand_contact_tick": first_hand_contact_tick,
            "last_hand_contact_tick": last_hand_contact_tick,
            "precontact_horizontal_drift_meters": precontact_horizontal_drift,
            "initial_centers_meters": {
                name: [float(value) for value in center]
                for name, center in initial_centers.items()
            },
            "final_centers_meters": {
                name: [float(value) for value in center]
                for name, center in centers.items()
            },
            "final_hand_diagnostics": _hand_diagnostics(final_state),
            "final_deformable_bounds_meters": {
                name: _deformable_bounds(body)
                for name, body in final_bodies.items()
            },
            "peak_contact_penetration_meters": peak_penetration,
            "peak_contact_penetration_by_pair_meters": dict(
                sorted(peak_penetration_by_pair.items())
            ),
            "peak_penetration_contact": peak_penetration_contact,
            "peak_rigid_drive_force_newtons": {
                name: float(value)
                for name, value in zip(
                    rigid_names, peak_rigid_drive, strict=True
                )
            },
            "peak_soft_nodal_drive_force_newtons": peak_soft_drive,
            "final_finger_close_fraction": float(
                finger_close_fraction_at_tick(min(completed_steps - 1, CYCLE_TICKS - 1))
            ),
            "solver": diagnostics,
            "reset_max_error": reset_max_error,
            "force_flow_trace": force_trace,
            "phase_diagnostics": phase_trace,
        }
    finally:
        context.clear_world()
        context.clear_scene()


def main() -> None:
    print(json.dumps(run(_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
