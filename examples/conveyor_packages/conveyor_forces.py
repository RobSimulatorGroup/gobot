"""GPU-resident rigid and deformable conveyor force models.

The model follows Newton's ``basic_conveyor_forces`` structure: the belt is a
static collider, contact normal forces come from the owning solver, and a
Coulomb-limited tangential force drives each rigid body or deformable vertex
toward the prescribed surface velocity on the next step.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


def configure_mujoco_velocity_field_belt(
    provider: Any, geom_name: str
) -> None:
    """Make one MuJoCo belt geom normal-only for explicit surface traction."""

    rigid = provider.rigid_solver
    geom_id = rigid.resolve_object_ids("geom", (str(geom_name),))[0]
    torch = rigid._torch
    friction = rigid.model_array("geom_friction")
    geom_indices = torch.as_tensor(
        (geom_id,), dtype=torch.long, device=friction.device
    )
    frictionless = torch.zeros_like(friction.index_select(1, geom_indices))
    friction.index_copy_(1, geom_indices, frictionless)
    condim = rigid.model_array("geom_condim")
    condim_indices = geom_indices.to(device=condim.device)
    normal_only = torch.ones_like(condim.index_select(0, condim_indices))
    condim.index_copy_(0, condim_indices, normal_only)
    # MuJoCo otherwise combines equal-priority geom parameters and the parcel's
    # authored friction can keep the pair tangential. Give the velocity-field
    # surface precedence so condim=1 governs every belt contact.
    priority = rigid.model_array("geom_priority")
    priority_indices = geom_indices.to(device=priority.device)
    belt_priority = torch.ones_like(
        priority.index_select(0, priority_indices)
    )
    priority.index_copy_(0, priority_indices, belt_priority)
    rigid.recompute_constants()


class ConveyorForceModel:
    """Apply a constant +X surface-velocity field to rigid bodies."""

    def __init__(
        self,
        provider: Any,
        body_views: Sequence[Any],
        body_ids: Sequence[int],
        contact_sensor_names: Sequence[str],
        masses: Sequence[float],
        *,
        friction_coefficient: float,
        fixed_dt: float,
        max_acceleration: float = 4.0,
    ) -> None:
        if not body_views or len(body_views) != len(body_ids):
            raise ValueError("conveyor force model requires aligned rigid bodies")
        if len(body_ids) != len(contact_sensor_names) or len(body_ids) != len(
            masses
        ):
            raise ValueError("conveyor force model tables must have equal length")
        friction_coefficient = float(friction_coefficient)
        fixed_dt = float(fixed_dt)
        max_acceleration = float(max_acceleration)
        if not math.isfinite(friction_coefficient) or friction_coefficient < 0.0:
            raise ValueError("conveyor friction must be finite and non-negative")
        if not math.isfinite(fixed_dt) or fixed_dt <= 0.0:
            raise ValueError("conveyor fixed_dt must be finite and positive")
        if not math.isfinite(max_acceleration) or max_acceleration <= 0.0:
            raise ValueError(
                "conveyor max_acceleration must be finite and positive"
            )
        if any(not math.isfinite(float(mass)) or float(mass) <= 0.0 for mass in masses):
            raise ValueError("conveyor body masses must be finite and positive")

        self.provider = provider
        self.body_views = tuple(body_views)
        rigid = provider.rigid_solver
        self.sensors = tuple(
            rigid.contact_sensor(name) for name in contact_sensor_names
        )
        torch = rigid._torch
        self._torch = torch
        self._xfrc = provider.arrays["xfrc_applied"]
        self._body_ids = torch.as_tensor(
            tuple(int(value) for value in body_ids),
            dtype=torch.long,
            device=self._xfrc.device,
        )
        shape = (int(provider.num_envs), len(body_ids))
        self._masses = torch.as_tensor(
            tuple(float(value) for value in masses),
            dtype=self._xfrc.dtype,
            device=self._xfrc.device,
        ).reshape(1, len(body_ids))
        self._friction_coefficient = friction_coefficient
        self._inverse_dt = 1.0 / fixed_dt
        self._acceleration_force_limit = self._masses * max_acceleration
        self._normal_force = torch.zeros(
            shape, dtype=self._xfrc.dtype, device=self._xfrc.device
        )
        self._linear_velocity_x = torch.zeros_like(self._normal_force)
        self._desired_force = torch.zeros_like(self._normal_force)
        self._force_limit = torch.zeros_like(self._normal_force)
        self._negative_force_limit = torch.zeros_like(self._normal_force)
        self._drive_force = torch.zeros_like(self._normal_force)
        wrench_shape = (*shape, 6)
        self._selected_wrenches = torch.zeros(
            wrench_shape, dtype=self._xfrc.dtype, device=self._xfrc.device
        )
        self._previous_wrenches = torch.zeros_like(self._selected_wrenches)
        self._next_wrenches = torch.zeros_like(self._selected_wrenches)

    @property
    def normal_force(self) -> Any:
        return self._normal_force

    @property
    def drive_force(self) -> Any:
        return self._drive_force

    def apply(self, target_speed: Any) -> None:
        """Replace this model's prior wrench with the next belt wrench."""

        self.provider.sense()
        self._torch.index_select(
            self._xfrc, 1, self._body_ids, out=self._selected_wrenches
        )
        self._selected_wrenches.sub_(self._previous_wrenches)

        for body_index, (sensor, view) in enumerate(
            zip(self.sensors, self.body_views, strict=True)
        ):
            contact_force = sensor["force"]
            found = sensor["found"].to(dtype=contact_force.dtype)
            force_magnitude = self._torch.linalg.vector_norm(
                contact_force, dim=-1
            )
            self._normal_force[:, body_index].copy_(
                (force_magnitude * found).sum(dim=1)
            )
            self._linear_velocity_x[:, body_index].copy_(
                view.read_state().base_velocity[:, 0]
            )

        speed = self._torch.as_tensor(
            target_speed,
            dtype=self._xfrc.dtype,
            device=self._xfrc.device,
        )
        if tuple(speed.shape) == ():
            speed = speed.expand(int(self.provider.num_envs))
        if tuple(speed.shape) != (int(self.provider.num_envs),):
            raise ValueError(
                "conveyor target speed must be scalar or have shape "
                f"[{int(self.provider.num_envs)}]"
            )

        self._desired_force.copy_(speed[:, None])
        self._desired_force.sub_(self._linear_velocity_x)
        self._desired_force.mul_(self._masses).mul_(self._inverse_dt)
        self._force_limit.copy_(self._normal_force).mul_(
            self._friction_coefficient
        )
        self._torch.minimum(
            self._force_limit,
            self._acceleration_force_limit,
            out=self._force_limit,
        )
        self._negative_force_limit.copy_(self._force_limit).neg_()
        self._torch.maximum(
            self._desired_force,
            self._negative_force_limit,
            out=self._drive_force,
        )
        self._torch.minimum(
            self._drive_force,
            self._force_limit,
            out=self._drive_force,
        )
        self._next_wrenches.zero_()
        self._next_wrenches[..., 0].copy_(self._drive_force)
        self._selected_wrenches.add_(self._next_wrenches)
        self._xfrc.index_copy_(1, self._body_ids, self._selected_wrenches)
        self._previous_wrenches.copy_(self._next_wrenches)

    def clear(self) -> None:
        """Remove only wrenches previously contributed by this model."""

        self._torch.index_select(
            self._xfrc, 1, self._body_ids, out=self._selected_wrenches
        )
        self._selected_wrenches.sub_(self._previous_wrenches)
        self._xfrc.index_copy_(1, self._body_ids, self._selected_wrenches)
        self._previous_wrenches.zero_()
        self._next_wrenches.zero_()
        self._normal_force.zero_()
        self._drive_force.zero_()


class DeformableConveyorForceModel:
    """Apply damping and Newton-style traction to deformable vertices."""

    def __init__(
        self,
        provider: Any,
        body_entries: Sequence[Any],
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
        if not body_entries or len(body_entries) != len(body_masses):
            raise ValueError("deformable conveyor bodies and masses must align")
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in body_masses
        ):
            raise ValueError(
                "deformable conveyor masses must be finite and positive"
            )
        friction_coefficient = float(friction_coefficient)
        fixed_dt = float(fixed_dt)
        max_acceleration = float(max_acceleration)
        if (
            not math.isfinite(friction_coefficient)
            or friction_coefficient < 0.0
            or not math.isfinite(fixed_dt)
            or fixed_dt <= 0.0
            or not math.isfinite(max_acceleration)
            or max_acceleration <= 0.0
        ):
            raise ValueError("deformable conveyor friction/dt are invalid")
        belt_half_length = float(belt_half_length)
        belt_half_width = float(belt_half_width)
        belt_top = float(belt_top)
        belt_center_x = float(belt_center_x)
        belt_center_y = float(belt_center_y)
        contact_height_tolerance = float(contact_height_tolerance)
        if (
            not math.isfinite(belt_half_length)
            or not math.isfinite(belt_half_width)
            or min(belt_half_length, belt_half_width) <= 0.0
            or not math.isfinite(belt_top)
            or not math.isfinite(belt_center_x)
            or not math.isfinite(belt_center_y)
            or not math.isfinite(contact_height_tolerance)
            or contact_height_tolerance < 0.0
        ):
            raise ValueError(
                "deformable conveyor belt bounds must be finite and valid"
            )

        self.provider = provider
        self._torch = provider.rigid_solver._torch
        self._positions = provider.arrays["ipc_positions"]
        self._velocities = provider.arrays["ipc_velocities"]
        self._contact_forces = provider.arrays["ipc_contact_forces"]
        self._external_forces = provider.arrays["ipc_external_forces"]
        self._ranges = tuple(
            (
                int(entry["element_offset"]),
                int(entry["element_offset"]) + int(entry["element_count"]),
            )
            for entry in body_entries
        )
        self._masses = tuple(float(value) for value in body_masses)
        if velocity_damping_rates is None:
            velocity_damping_rates = (0.0,) * len(self._ranges)
        if len(velocity_damping_rates) != len(self._ranges) or any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in velocity_damping_rates
        ):
            raise ValueError(
                "deformable velocity damping rates must align and be finite"
            )
        self._velocity_damping_rates = tuple(
            float(value) for value in velocity_damping_rates
        )
        self._friction = friction_coefficient
        self._inverse_dt = 1.0 / fixed_dt
        self._max_acceleration = max_acceleration
        self._belt_half_length = belt_half_length
        self._belt_half_width = belt_half_width
        self._belt_center_x = belt_center_x
        self._belt_center_y = belt_center_y
        self._belt_contact_max_z = belt_top + contact_height_tolerance
        shape = self._positions.shape[:2]
        self._normal_force = self._torch.zeros(
            shape,
            dtype=self._positions.dtype,
            device=self._positions.device,
        )
        self._drive_force = self._torch.zeros_like(self._normal_force)

    @property
    def drive_force(self) -> Any:
        return self._drive_force

    def apply(self, target_speed: Any) -> None:
        speed = self._torch.as_tensor(
            target_speed,
            dtype=self._positions.dtype,
            device=self._positions.device,
        )
        if tuple(speed.shape) == ():
            speed = speed.expand(int(self.provider.num_envs))
        if tuple(speed.shape) != (int(self.provider.num_envs),):
            raise ValueError(
                "deformable conveyor target speed must be scalar or have "
                f"shape [{int(self.provider.num_envs)}]"
            )

        self._external_forces.zero_()
        self._drive_force.zero_()
        self._normal_force.copy_(self._contact_forces[..., 2].abs())
        for (begin, end), mass, damping_rate in zip(
            self._ranges,
            self._masses,
            self._velocity_damping_rates,
            strict=True,
        ):
            if damping_rate > 0.0:
                self._external_forces[:, begin:end].copy_(
                    self._velocities[:, begin:end]
                ).mul_(-mass * damping_rate / (end - begin))
            positions = self._positions[:, begin:end]
            normal = self._normal_force[:, begin:end]
            active = (
                (normal > 0.0)
                & (
                    (positions[..., 0] - self._belt_center_x).abs()
                    <= self._belt_half_length
                )
                & (
                    (positions[..., 1] - self._belt_center_y).abs()
                    <= self._belt_half_width
                )
                & (positions[..., 2] <= self._belt_contact_max_z)
            )
            active_float = active.to(dtype=self._positions.dtype)
            active_count = active_float.sum(dim=1, keepdim=True).clamp_min_(1.0)
            desired = speed[:, None] - self._velocities[:, begin:end, 0]
            desired.mul_(mass * self._inverse_dt).div_(active_count)
            limit = normal * self._friction
            acceleration_limit = (
                mass * self._max_acceleration / active_count
            )
            limit = self._torch.minimum(limit, acceleration_limit)
            drive = self._torch.clamp(desired, min=-limit, max=limit)
            drive.mul_(active_float)
            self._drive_force[:, begin:end].copy_(drive)
            self._external_forces[:, begin:end, 0].add_(drive)

    def clear(self) -> None:
        self._external_forces.zero_()
        self._normal_force.zero_()
        self._drive_force.zero_()
