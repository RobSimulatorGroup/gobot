"""Task-level measurements for the contact-driven, two-hand mailer trial."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GraspThresholds:
    fingertip_force_newtons: float = 0.01
    minimum_lift_meters: float = 0.02
    minimum_hold_seconds: float = 0.10
    maximum_proxy_surface_error_meters: float = 0.001
    maximum_table_penetration_meters: float = 0.001
    maximum_initial_drift_meters: float = 0.0005
    flip_tolerance_degrees: float = 10.0
    maximum_settled_speed_mps: float = 0.05
    maximum_placed_clearance_meters: float = 0.01
    minimum_settle_seconds: float = 0.10
    maximum_wrist_rotation_degrees: float = 10.0


def material_face_normal(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Track one authored sheet, not the arbitrary sign of a PCA eigenvector."""
    points = vertices[triangles]
    normal = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]).sum(axis=0)
    length = float(np.linalg.norm(normal))
    if length < 1.0e-10 or not math.isfinite(length):
        raise ValueError("the tracked material face has collapsed")
    return normal / length


def angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0))))


def opposed_pinch(forces: np.ndarray, threshold: float) -> tuple[bool, list[float]]:
    """Require the thumb and all three regular distal links to carry load."""
    if forces.shape != (4, 3) or not np.isfinite(forces).all():
        raise ValueError("fingertip forces must be finite [thumb,index,middle,ring,xyz]")
    magnitudes = np.linalg.norm(forces, axis=1)
    regular = forces[1:].sum(axis=0)
    opposed = float(np.dot(forces[0], regular)) < 0.0
    return bool(np.all(magnitudes >= threshold) and opposed), magnitudes.tolist()


def proxy_surface_error(targets: np.ndarray, proxies: np.ndarray, radii: np.ndarray) -> float:
    """Upper-bound point displacement, including affine shear and rotation."""
    translation = np.linalg.norm(targets[:, :3, 3] - proxies[:, :3, 3], axis=1)
    linear = np.linalg.norm(targets[:, :3, :3] - proxies[:, :3, :3], axis=(1, 2))
    return float(np.max(translation + linear * radii, initial=0.0))


class GraspMeasurements:
    def __init__(
        self, *, initial_shell: np.ndarray, face_triangles: np.ndarray,
        table_height: float, fixed_dt: float, required_steps: int,
        thresholds: GraspThresholds | None = None,
    ) -> None:
        self.thresholds = thresholds or GraspThresholds()
        self.initial_center = initial_shell.mean(axis=0)
        self.initial_normal = material_face_normal(initial_shell, face_triangles)
        self.face_triangles = face_triangles
        self.table_height = table_height
        self.fixed_dt = fixed_dt
        self.required_steps = required_steps
        self.steps = 0
        self.first_contact_step: int | None = None
        self.first_pinch_step: int | None = None
        self.hold_steps = 0
        self.longest_hold_steps = 0
        self.peak_clearance = 0.0
        self.peak_proxy_error = 0.0
        self.peak_table_penetration = 0.0
        self.peak_initial_drift = 0.0
        self.peak_wrist_rotation = 0.0
        self.final_flip_degrees = 0.0
        self.final_speed = 0.0
        self.settled_steps = 0
        self.last: dict[str, Any] = {}

    def observe(
        self, *, phase: str, shell: np.ndarray, vertices: np.ndarray,
        velocities: np.ndarray, fingertip_forces: np.ndarray,
        hand_contact_force: float, proxy_error: float, wrist_rotation_degrees: float,
    ) -> dict[str, Any]:
        if fingertip_forces.shape != (2, 4, 3):
            raise ValueError("grasp measurement requires both hands and four fingertips per hand")
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
            raise ValueError("package vertices must be a nonempty [vertex,xyz] array")
        if velocities.shape != vertices.shape:
            raise ValueError("package velocities must match the measured vertices")
        if not all(np.isfinite(value).all() for value in (
            shell, vertices, velocities, fingertip_forces,
            [hand_contact_force, proxy_error, wrist_rotation_degrees],
        )):
            raise ValueError("non-finite grasp state")
        self.steps += 1
        if self.first_contact_step is None:
            drift = float(np.linalg.norm(shell.mean(axis=0)[:2] - self.initial_center[:2]))
            self.peak_initial_drift = max(self.peak_initial_drift, drift)
            if hand_contact_force >= self.thresholds.fingertip_force_newtons:
                self.first_contact_step = self.steps
        pinches = [opposed_pinch(forces, self.thresholds.fingertip_force_newtons)
                   for forces in fingertip_forces]
        bilateral = all(pinch[0] for pinch in pinches)
        if bilateral and self.first_pinch_step is None:
            self.first_pinch_step = self.steps
        clearance = float(vertices[:, 2].min() - self.table_height)
        self.peak_clearance = max(self.peak_clearance, clearance)
        self.peak_table_penetration = max(self.peak_table_penetration, -clearance)
        self.peak_proxy_error = max(self.peak_proxy_error, proxy_error)
        self.peak_wrist_rotation = max(self.peak_wrist_rotation, wrist_rotation_degrees)
        self.hold_steps = self.hold_steps + 1 if (
            bilateral and clearance >= self.thresholds.minimum_lift_meters
        ) else 0
        self.longest_hold_steps = max(self.longest_hold_steps, self.hold_steps)
        self.final_flip_degrees = angle_degrees(
            self.initial_normal, material_face_normal(shell, self.face_triangles)
        )
        self.final_speed = float(np.linalg.norm(velocities, axis=1).max())
        placed_and_released = (
            phase == "blue_flip_settle"
            and clearance <= self.thresholds.maximum_placed_clearance_meters
            and clearance >= -self.thresholds.maximum_table_penetration_meters
            and hand_contact_force < self.thresholds.fingertip_force_newtons
            and self.final_speed <= self.thresholds.maximum_settled_speed_mps
            and abs(180.0 - self.final_flip_degrees) <= self.thresholds.flip_tolerance_degrees
        )
        self.settled_steps = self.settled_steps + 1 if placed_and_released else 0
        self.last = {
            "step": self.steps, "phase": phase,
            "bilateral_opposed_pinch": bilateral,
            "fingertip_force_newtons": [pinch[1] for pinch in pinches],
            "clearance_meters": clearance,
            "material_face_flip_degrees": self.final_flip_degrees,
            "proxy_surface_error_bound_meters": proxy_error,
            "maximum_speed_mps": self.final_speed,
        }
        return dict(self.last)

    def result(self, *, error: str | None, exact_feedback: bool,
               sequence_completed: bool | None = None) -> dict[str, Any]:
        limits = self.thresholds
        gates = {
            "completed": self.steps >= self.required_steps and sequence_completed is not False,
            "finite_and_solver_accepted": error is None,
            "exact_two_way_feedback": exact_feedback,
            "bilateral_thumb_and_three_fingers": self.first_pinch_step is not None,
            "sustained_airborne_pinch": self.longest_hold_steps * self.fixed_dt >= limits.minimum_hold_seconds,
            "flipped_material_face": abs(180.0 - self.final_flip_degrees) <= limits.flip_tolerance_degrees,
            "released_and_settled": self.settled_steps * self.fixed_dt >= limits.minimum_settle_seconds,
            "proxy_tracks_actual_hand": self.peak_proxy_error <= limits.maximum_proxy_surface_error_meters,
            "table_penetration": self.peak_table_penetration <= limits.maximum_table_penetration_meters,
            "precontact_horizontal_drift": self.peak_initial_drift <= limits.maximum_initial_drift_meters,
            "no_wrist_flip": self.peak_wrist_rotation <= limits.maximum_wrist_rotation_degrees,
        }
        return {
            "passed": all(gates.values()), "gates": gates,
            "failed_gates": [name for name, passed in gates.items() if not passed],
            "error": error, "completed_steps": self.steps,
            "required_steps": self.required_steps,
            "first_contact_step": self.first_contact_step,
            "first_bilateral_pinch_step": self.first_pinch_step,
            "longest_airborne_pinch_seconds": self.longest_hold_steps * self.fixed_dt,
            "maximum_clearance_meters": self.peak_clearance,
            "maximum_proxy_surface_error_bound_meters": self.peak_proxy_error,
            "maximum_table_penetration_meters": self.peak_table_penetration,
            "maximum_precontact_horizontal_drift_meters": self.peak_initial_drift,
            "maximum_wrist_rotation_degrees": self.peak_wrist_rotation,
            "final_material_face_flip_degrees": self.final_flip_degrees,
            "final_maximum_speed_mps": self.final_speed,
            "released_and_settled_seconds": self.settled_steps * self.fixed_dt,
        }
