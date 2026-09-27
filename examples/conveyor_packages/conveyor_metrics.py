"""Conveyor trial geometry, contact measurements and report fields."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from conveyor_config import HAND_STAGE_JOINT_NAMES_BY_SIDE, LEAP_CONTACT_LINK_NAMES, LEAP_ROBOT_NAMES, RIGID_BOX_NAMES
from conveyor_profile import CYCLE_TICKS, HAND_MOTION_SEGMENTS, cycle_phase, finger_close_fraction_at_tick


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


def assert_finite_state(state: dict[str, Any]) -> None:
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


def reset_error(
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


class ConveyorMetrics:
    """Accumulate measurements from completed physics ticks."""

    def __init__(
        self, initial_state: dict[str, Any], *,
        trace_force_flow: bool = False, phase_diagnostics: bool = False,
    ) -> None:
        initial_bodies = _deformable_table(initial_state)
        self.initial_shell_axes = {
            "soft_mailer_blue": _shell_axis(
                initial_bodies["soft_mailer_blue"]
            ),
            "soft_pouch_yellow": _shell_axis(
                initial_bodies["soft_pouch_yellow"]
            ),
        }
        self.initial_carton = _link_state(
            initial_state, "carton_small", "carton_small"
        )["global_transform"]["quaternion"]
        self.parcel_groups = {
            "blue_mailer": ("soft_mailer_blue", "soft_mailer_blue_fill"),
            "yellow_pouch": ("soft_pouch_yellow", "soft_pouch_yellow_fill"),
            "carton_small": ("carton_small",),
        }
        self.initial_centers = {
            "blue_mailer": _group_center(
                initial_bodies, self.parcel_groups["blue_mailer"]
            ),
            "yellow_pouch": _group_center(
                initial_bodies, self.parcel_groups["yellow_pouch"]
            ),
            "carton_small": np.asarray(
                _link_state(initial_state, "carton_small", "carton_small")
                ["global_transform"]["position"]
            ),
        }

        self.belt_travel = 0.0
        self.peak_penetration = 0.0
        self.peak_penetration_by_pair: dict[str, float] = {}
        self.peak_penetration_contact: dict[str, Any] | None = None
        self.peak_rigid_drive = np.zeros(len(RIGID_BOX_NAMES), dtype=np.float64)
        self.peak_soft_drive = 0.0
        self.maximum_flip = {
            "blue_mailer": 0.0,
            "yellow_pouch": 0.0,
            "carton_small": 0.0,
        }
        self.simultaneous_hand_contact = {name: False for name in self.parcel_groups}
        self.hand_contact_frames = {
            name: {hand: 0 for hand in LEAP_ROBOT_NAMES}
            for name in self.parcel_groups
        }
        self.first_hand_contact_tick = {
            name: {hand: None for hand in LEAP_ROBOT_NAMES}
            for name in self.parcel_groups
        }
        self.last_hand_contact_tick = {
            name: {hand: None for hand in LEAP_ROBOT_NAMES}
            for name in self.parcel_groups
        }
        self.maximum_simultaneous_hands = {name: 0 for name in self.parcel_groups}
        self.hand_contact_started = {name: False for name in self.parcel_groups}
        self.current_contacting_hands = {
            name: [] for name in self.parcel_groups
        }
        self.precontact_horizontal_drift = {name: 0.0 for name in self.parcel_groups}
        self.force_trace: list[dict[str, Any]] = []
        self.phase_trace: list[dict[str, Any]] = []
        self.phase_end_ticks = _phase_end_ticks()

        self.centers = self.initial_centers
        self.current_flip = {name: 0.0 for name in self.parcel_groups}
        self.trace_force_flow = trace_force_flow
        self.phase_diagnostics = phase_diagnostics

    def update(
        self, final_state: dict[str, Any], *, tick: int, control_tick: int,
        rigid_drive: np.ndarray, soft_drive: dict[str, np.ndarray],
    ) -> None:
        self._update_penetration(final_state, tick)
        self._update_drive_forces(rigid_drive, soft_drive)
        self._update_parcels(final_state, tick)
        self._record_traces(final_state, tick, control_tick, rigid_drive, soft_drive)

    def _update_penetration(self, final_state: dict[str, Any], tick: int) -> None:
        for contact in final_state["contacts"]:
            penetration = max(0.0, -float(contact["distance"]))
            pair_name = _contact_pair_name(contact)
            self.peak_penetration_by_pair[pair_name] = max(
                self.peak_penetration_by_pair.get(pair_name, 0.0),
                penetration,
            )
            if penetration <= self.peak_penetration:
                continue
            self.peak_penetration = penetration
            self.peak_penetration_contact = {
                "tick": tick,
                "robot": str(contact["robot_name"]),
                "link": str(contact["link_name"]),
                "other_robot": str(contact["other_robot_name"]),
                "other_link": str(contact["other_link_name"]),
                "distance_meters": float(contact["distance"]),
            }

    def _update_drive_forces(self, rigid_drive, soft_drive) -> None:
        self.peak_rigid_drive = np.maximum(
            self.peak_rigid_drive, np.abs(rigid_drive)
        )
        self.peak_soft_drive = max(
            self.peak_soft_drive,
            max(
                (
                    float(np.max(np.abs(values)))
                    for values in soft_drive.values()
                ),
                default=0.0,
            ),
        )

    def _update_parcels(self, final_state: dict[str, Any], tick: int) -> None:
        bodies = _deformable_table(final_state)
        self.current_flip = {}
        for parcel, body_name in (("blue_mailer", "soft_mailer_blue"),
                                  ("yellow_pouch", "soft_pouch_yellow")):
            cosine = float(np.clip(np.dot(_shell_axis(bodies[body_name]),
                                          self.initial_shell_axes[body_name]), -1.0, 1.0))
            self.current_flip[parcel] = math.degrees(math.acos(cosine))
        self.current_flip["carton_small"] = _quaternion_angle_degrees(
            _link_state(final_state, "carton_small", "carton_small")["global_transform"]["quaternion"],
            self.initial_carton,
        )
        for name, angle in self.current_flip.items():
            self.maximum_flip[name] = max(self.maximum_flip[name], angle)

        self.centers = {
            "blue_mailer": _group_center(
                bodies, self.parcel_groups["blue_mailer"]
            ),
            "yellow_pouch": _group_center(
                bodies, self.parcel_groups["yellow_pouch"]
            ),
            "carton_small": np.asarray(
                _link_state(final_state, "carton_small", "carton_small")
                ["global_transform"]["position"]
            ),
        }
        for name, targets in self.parcel_groups.items():
            hands = _contact_hands(final_state["contacts"], set(targets))
            self.current_contacting_hands[name] = sorted(hands)
            self.maximum_simultaneous_hands[name] = max(
                self.maximum_simultaneous_hands[name], len(hands)
            )
            for hand in hands:
                self.hand_contact_frames[name][hand] += 1
                if self.first_hand_contact_tick[name][hand] is None:
                    self.first_hand_contact_tick[name][hand] = tick
                self.last_hand_contact_tick[name][hand] = tick
            if len(hands) == 2:
                self.simultaneous_hand_contact[name] = True
            self.hand_contact_started[name] |= bool(hands)
            if not self.hand_contact_started[name]:
                drift = float(
                    np.linalg.norm(
                        (self.centers[name] - self.initial_centers[name])[:2]
                    )
                )
                self.precontact_horizontal_drift[name] = max(
                    self.precontact_horizontal_drift[name], drift
                )

    def _record_traces(self, final_state, tick, control_tick, rigid_drive, soft_drive) -> None:
        if self.trace_force_flow:
            self.force_trace.append(
                {
                    "tick": tick,
                    "phase": cycle_phase(control_tick),
                    "rigid_drive_resultant_newtons": float(
                        np.sum(rigid_drive)
                    ),
                    "soft_drive_resultant_newtons": float(
                        sum(
                            np.sum(values)
                            for values in soft_drive.values()
                        )
                    ),
                }
            )
        if self.phase_diagnostics and tick in self.phase_end_ticks:
            self.phase_trace.append(
                {
                    "phase": self.phase_end_ticks[tick],
                    "tick": tick,
                    "flip_degrees": self.current_flip.copy(),
                    "centers_meters": {
                        name: [float(value) for value in center]
                        for name, center in self.centers.items()
                    },
                    "hand_diagnostics": _hand_diagnostics(final_state),
                    "deformable_bounds_meters": {
                        name: _deformable_bounds(body)
                        for name, body in _deformable_table(final_state).items()
                    },
                    "contact_count": len(final_state["contacts"]),
                    "contacting_hands": {
                        name: list(hands)
                        for name, hands in self.current_contacting_hands.items()
                    },
                }
            )

    def observation(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "contacting_hands": {name: list(hands) for name, hands in self.current_contacting_hands.items()},
            "directed_contact_count": len(state["contacts"]),
            "max_penetration_meters": _max_penetration(state["contacts"]),
            "flip_degrees": self.current_flip.copy(),
        }

    def report(self, final_state: dict[str, Any], completed_steps: int) -> dict[str, Any]:
        return {
            "belt_surface_commanded_travel_meters": self.belt_travel,
            "final_flip_degrees": self.current_flip.copy(),
            "maximum_flip_degrees": self.maximum_flip,
            "simultaneous_hand_contact": self.simultaneous_hand_contact,
            "maximum_simultaneous_hands": self.maximum_simultaneous_hands,
            "hand_contact_frames": self.hand_contact_frames,
            "first_hand_contact_tick": self.first_hand_contact_tick,
            "last_hand_contact_tick": self.last_hand_contact_tick,
            "precontact_horizontal_drift_meters": self.precontact_horizontal_drift,
            "initial_centers_meters": {
                name: [float(value) for value in center]
                for name, center in self.initial_centers.items()
            },
            "final_centers_meters": {
                name: [float(value) for value in center]
                for name, center in self.centers.items()
            },
            "final_hand_diagnostics": _hand_diagnostics(final_state),
            "final_deformable_bounds_meters": {
                name: _deformable_bounds(body)
                for name, body in _deformable_table(final_state).items()
            },
            "peak_contact_penetration_meters": self.peak_penetration,
            "peak_contact_penetration_by_pair_meters": dict(
                sorted(self.peak_penetration_by_pair.items())
            ),
            "peak_penetration_contact": self.peak_penetration_contact,
            "peak_rigid_drive_force_newtons": {
                name: float(value)
                for name, value in zip(
                    RIGID_BOX_NAMES, self.peak_rigid_drive, strict=True
                )
            },
            "peak_soft_nodal_drive_force_newtons": self.peak_soft_drive,
            "final_finger_close_fraction": float(
                finger_close_fraction_at_tick(min(completed_steps - 1, CYCLE_TICKS - 1))
            ),
            "force_flow_trace": self.force_trace,
            "phase_diagnostics": self.phase_trace,
        }
