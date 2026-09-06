"""Backend-neutral conveyor traction for Gobot's native physics world."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def _finite_positive(value: float, description: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{description} must be finite and positive")
    return result


def _finite_non_negative(value: float, description: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{description} must be finite and non-negative")
    return result


def _robot_table(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(robot["name"]): robot for robot in state["robots"]}


def _link_table(robot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(link["name"]): link for link in robot["links"]}


class ConveyorForceModel:
    """Drive standalone rigid packages using measured belt contact force."""

    def __init__(
        self,
        context: Any,
        body_names: Sequence[str],
        masses: Sequence[float],
        *,
        belt_robot: str,
        belt_link: str,
        friction_coefficient: float,
        fixed_dt: float,
        max_acceleration: float = 4.0,
    ) -> None:
        if not body_names or len(body_names) != len(masses):
            raise ValueError("rigid conveyor body names and masses must align")
        self.context = context
        self.body_names = tuple(str(name) for name in body_names)
        self.masses = tuple(
            _finite_positive(mass, "rigid package mass") for mass in masses
        )
        self.belt_robot = str(belt_robot)
        self.belt_link = str(belt_link)
        self.friction = _finite_non_negative(
            friction_coefficient, "conveyor friction"
        )
        self.fixed_dt = _finite_positive(fixed_dt, "conveyor fixed_dt")
        self.max_acceleration = _finite_positive(
            max_acceleration, "conveyor max acceleration"
        )
        self.normal_force = np.zeros(len(self.body_names), dtype=np.float64)
        self.drive_force = np.zeros(len(self.body_names), dtype=np.float64)

    def _belt_normal_force(
        self, contacts: Sequence[dict[str, Any]], body_name: str
    ) -> float:
        return sum(
            max(0.0, float(contact["normal_force"]))
            for contact in contacts
            if str(contact["robot_name"]) == body_name
            and str(contact["link_name"]) == body_name
            and str(contact["other_robot_name"]) == self.belt_robot
            and str(contact["other_link_name"]) == self.belt_link
        )

    def apply(
        self, target_speed: float, state: dict[str, Any] | None = None
    ) -> np.ndarray:
        speed = float(target_speed)
        if not math.isfinite(speed):
            raise ValueError("conveyor target speed must be finite")
        if state is None:
            state = self.context.get_physics_state_view()
        robots = _robot_table(state)
        contacts = state["contacts"]

        for index, (name, mass) in enumerate(
            zip(self.body_names, self.masses, strict=True)
        ):
            try:
                link = _link_table(robots[name])[name]
            except KeyError as error:
                raise RuntimeError(
                    f"physics state has no standalone rigid package {name!r}"
                ) from error
            normal = self._belt_normal_force(contacts, name)
            velocity_x = float(link["linear_velocity"][0])
            desired = mass * (speed - velocity_x) / self.fixed_dt
            limit = min(self.friction * normal, mass * self.max_acceleration)
            drive = float(np.clip(desired, -limit, limit))
            point = tuple(link["global_transform"]["position"])
            self.context.set_link_external_force(
                name, name, point, (drive, 0.0, 0.0)
            )
            self.normal_force[index] = normal
            self.drive_force[index] = drive
        return self.drive_force.copy()

    def clear(self, state: dict[str, Any] | None = None) -> None:
        if state is None:
            state = self.context.get_physics_state_view()
        robots = _robot_table(state)
        for name in self.body_names:
            link = _link_table(robots[name])[name]
            point = tuple(link["global_transform"]["position"])
            self.context.set_link_external_force(
                name, name, point, (0.0, 0.0, 0.0)
            )
        self.normal_force.fill(0.0)
        self.drive_force.fill(0.0)


class DeformableConveyorForceModel:
    """Apply Coulomb-limited belt traction as world-frame nodal forces."""

    def __init__(
        self,
        context: Any,
        body_names: Sequence[str],
        body_masses: Sequence[float],
        *,
        friction_coefficient: float,
        fixed_dt: float,
        belt_half_length: float,
        belt_half_width: float,
        belt_top: float,
        belt_center_x: float = 0.0,
        belt_center_y: float = 0.0,
        contact_height_tolerance: float = 0.012,
        max_acceleration: float = 4.0,
        velocity_damping_rates: Sequence[float] | None = None,
    ) -> None:
        if not body_names or len(body_names) != len(body_masses):
            raise ValueError("deformable conveyor body names and masses must align")
        self.context = context
        self.body_names = tuple(str(name) for name in body_names)
        self.body_masses = tuple(
            _finite_positive(mass, "deformable package mass")
            for mass in body_masses
        )
        self.friction = _finite_non_negative(
            friction_coefficient, "conveyor friction"
        )
        self.fixed_dt = _finite_positive(fixed_dt, "conveyor fixed_dt")
        self.belt_half_length = _finite_positive(
            belt_half_length, "belt half length"
        )
        self.belt_half_width = _finite_positive(
            belt_half_width, "belt half width"
        )
        self.belt_top = float(belt_top)
        self.belt_center_x = float(belt_center_x)
        self.belt_center_y = float(belt_center_y)
        self.contact_height_tolerance = _finite_non_negative(
            contact_height_tolerance, "belt contact height tolerance"
        )
        self.max_acceleration = _finite_positive(
            max_acceleration, "conveyor max acceleration"
        )
        if velocity_damping_rates is None:
            velocity_damping_rates = (0.0,) * len(self.body_names)
        if len(velocity_damping_rates) != len(self.body_names):
            raise ValueError("deformable damping rates must align with bodies")
        self.velocity_damping_rates = tuple(
            _finite_non_negative(value, "deformable damping rate")
            for value in velocity_damping_rates
        )
        scalar_fields = (
            self.belt_top,
            self.belt_center_x,
            self.belt_center_y,
        )
        if not all(math.isfinite(value) for value in scalar_fields):
            raise ValueError("belt bounds must be finite")
        self.normal_force: dict[str, np.ndarray] = {}
        self.drive_force: dict[str, np.ndarray] = {}
        self.applied_force: dict[str, np.ndarray] = {}

    def _body_table(
        self, state: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        bodies = {
            str(body.get("name", "")): body
            for body in state["deformables"]
        }
        missing = [name for name in self.body_names if name not in bodies]
        if missing:
            raise RuntimeError(
                "physics state is missing deformable bodies: "
                + ", ".join(missing)
            )
        return bodies

    def apply(
        self,
        target_speed: float,
        *,
        damping_scale: float = 1.0,
        state: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        speed = float(target_speed)
        damping_scale = _finite_non_negative(
            damping_scale, "deformable damping scale"
        )
        if not math.isfinite(speed):
            raise ValueError("conveyor target speed must be finite")
        if state is None:
            state = self.context.get_physics_state_view()
        bodies = self._body_table(state)

        for name, mass, damping_rate in zip(
            self.body_names,
            self.body_masses,
            self.velocity_damping_rates,
            strict=True,
        ):
            body = bodies[name]
            positions = np.asarray(body["world_vertices"], dtype=np.float64)
            local_velocities = np.asarray(
                body["local_velocities"], dtype=np.float64
            )
            rotation = np.asarray(
                body["global_transform"]["matrix"], dtype=np.float64
            )[:3, :3]
            velocities = local_velocities @ rotation.T
            contact_forces = np.asarray(
                body["contact_forces_world"], dtype=np.float64
            )
            if contact_forces.size == 0:
                contact_forces = np.zeros_like(positions)
            if (
                positions.ndim != 2
                or positions.shape[1:] != (3,)
                or velocities.shape != positions.shape
                or contact_forces.shape != positions.shape
            ):
                raise RuntimeError(
                    f"deformable runtime arrays for {name!r} are inconsistent"
                )

            normal = np.abs(contact_forces[:, 2])
            active = (
                (normal > 0.0)
                & (
                    np.abs(positions[:, 0] - self.belt_center_x)
                    <= self.belt_half_length
                )
                & (
                    np.abs(positions[:, 1] - self.belt_center_y)
                    <= self.belt_half_width
                )
                & (
                    positions[:, 2]
                    <= self.belt_top + self.contact_height_tolerance
                )
            )
            active_count = max(1, int(np.count_nonzero(active)))
            desired = mass * (speed - velocities[:, 0]) / (
                self.fixed_dt * active_count
            )
            limit = np.minimum(
                self.friction * normal,
                mass * self.max_acceleration / active_count,
            )
            drive = np.clip(desired, -limit, limit)
            drive[~active] = 0.0
            force = -(
                mass * damping_rate * damping_scale / len(positions)
            ) * velocities
            force[:, 0] += drive
            self.context.set_deformable_external_forces(
                int(body["stable_id"]), force
            )
            self.normal_force[name] = normal
            self.drive_force[name] = drive
            self.applied_force[name] = force
        return {name: values.copy() for name, values in self.applied_force.items()}

    def clear(self, state: dict[str, Any] | None = None) -> None:
        if state is None:
            state = self.context.get_physics_state_view()
        for name, body in self._body_table(state).items():
            count = len(body["local_vertices"])
            self.context.set_deformable_external_forces(
                int(body["stable_id"]), np.zeros((count, 3), dtype=np.float64)
            )
        self.normal_force.clear()
        self.drive_force.clear()
        self.applied_force.clear()
