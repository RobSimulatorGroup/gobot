"""Contact-gated joint control for the single-mailer acceptance task."""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import numpy as np

from conveyor_grasp_metrics import opposed_pinch
from conveyor_profile import (
    BLUE_FLIP_PINCH_WORLD_CENTERS, HAND_STAGE_BLUE_FLIP_CONTACT_TARGETS,
    cycle_phase, hand_controls_at_tick, smoothstep,
)


@dataclass(frozen=True)
class PinchControlSettings:
    closing_seconds: float = .65
    confirmed_pinch_seconds: float = .10
    lost_pinch_seconds: float = .02
    fingertip_force_newtons: float = .15
    maximum_fingertip_force_newtons: float = 20.
    maximum_proxy_error_meters: float = .001
    approach_clearance_meters: float = .16
    crest_height_offset_meters: float = .006
    target_tracking_speed_mps: float = .05
    maximum_target_correction_meters: float = .06
    maximum_wait_steps: int = 1000
    continuous_finger_feedback: bool = True
    normal_force_target_newtons: float = .75
    force_deadband_newtons: float = .15
    force_filter_seconds: float = .02
    finger_feedback_gain: float = .6
    finger_correction_rate: float = .6
    maximum_finger_correction: float = .15

    def __post_init__(self):
        try:
            wait_steps = operator.index(self.maximum_wait_steps)
        except TypeError as error:
            raise ValueError("maximum wait steps must be an integer") from error
        for value in (self.closing_seconds, self.confirmed_pinch_seconds, self.lost_pinch_seconds,
                      self.fingertip_force_newtons, self.maximum_fingertip_force_newtons,
                      self.maximum_proxy_error_meters, self.approach_clearance_meters,
                      self.target_tracking_speed_mps, self.maximum_target_correction_meters,
                      self.normal_force_target_newtons, self.force_deadband_newtons,
                      self.force_filter_seconds, self.finger_feedback_gain,
                      self.finger_correction_rate, self.maximum_finger_correction):
            if not math.isfinite(value) or value <= 0.:
                raise ValueError("pinch durations, force limits and clearances must be finite and positive")
        if (not math.isfinite(self.crest_height_offset_meters) or wait_steps < 1
                or isinstance(self.maximum_wait_steps, bool)
                or self.maximum_fingertip_force_newtons <= self.fingertip_force_newtons):
            raise ValueError("invalid pinch acquisition limits")
        if (not isinstance(self.continuous_finger_feedback, bool)
                or self.force_deadband_newtons >= self.normal_force_target_newtons
                or 3. * self.normal_force_target_newtons >= self.maximum_fingertip_force_newtons
                or self.maximum_finger_correction > .25):
            raise ValueError("invalid per-finger feedback limits")


class ContactPinchController:
    def __init__(self, poses, initial_shell, face_triangles, segments, fixed_dt,
                 settings: PinchControlSettings | None = None, *, fingertip_normals=None):
        self.settings = settings or PinchControlSettings()
        self.poses = np.array(poses, dtype=float, copy=True)
        if self.poses.ndim != 3 or self.poses.shape[1:] != (2, 22) or len(self.poses) < 2:
            raise ValueError("pinch poses must be [opening,hand,22 joint targets]")
        if not np.isfinite(self.poses).all():
            raise ValueError("pinch poses must be finite")
        if np.any(self.poses[:, :, 3:6] != 0.):
            raise ValueError("pinch pose table must preserve the authored palm-down wrist orientation")
        if not math.isfinite(fixed_dt) or fixed_dt <= 0.:
            raise ValueError("pinch fixed time step must be finite and positive")
        self.fixed_dt = fixed_dt
        self.fingertip_normals = None
        if fingertip_normals is not None:
            normals = np.array(fingertip_normals, dtype=float, copy=True)
            if (normals.shape != (2, 4, 3) or not np.isfinite(normals).all()
                    or not np.allclose(np.linalg.norm(normals, axis=2), 1., atol=1.e-5)):
                raise ValueError("fingertip normals must be two hands of four unit world vectors")
            self.fingertip_normals = normals
        elif self.settings.continuous_finger_feedback:
            raise ValueError("per-finger feedback requires geometry-derived fingertip normals")
        self.boundaries = {}
        end = 0
        for segment in segments:
            self.boundaries[segment.phase] = (end, end + segment.duration)
            end += segment.duration
        self.sequence_steps = end
        self.nominal_centers = np.asarray(BLUE_FLIP_PINCH_WORLD_CENTERS)
        self.shell_shape = initial_shell.shape
        if initial_shell.ndim != 2 or initial_shell.shape[1] != 3 or not np.isfinite(initial_shell).all():
            raise ValueError("initial shell must contain finite vertex positions")
        upper = np.unique(face_triangles)
        points = initial_shell[upper]
        rear = points[:, 1].min()
        length = np.ptp(points[:, 1])
        self.patches = []
        for center in self.nominal_centers:
            mask = ((abs(points[:, 0] - center[0]) < .03)
                    & (points[:, 1] > rear + .08 * length)
                    & (points[:, 1] < rear + .20 * length))
            if not mask.any():
                raise ValueError("authored upper sheet has no rear shoulder grasp patch")
            self.patches.append(upper[mask])
        self.centers = None
        self.initial_centers = None
        self.anchors = []
        self.last_pinches = [False, False]
        self.closure = np.zeros(2)
        self.finger_corrections = np.zeros((2, 4))
        self.normal_forces = np.zeros((2, 4))
        self.force_targets = np.tile(self.settings.normal_force_target_newtons * np.array([3., 1., 1., 1.]), (2, 1))
        self.feedback_updates = self.motion_hold_steps = 0
        self.tick = self.steps = self.wait_steps = 0
        self.confirmed_steps = self.lost_steps = 0
        self.lift_started_step = None
        self.failure = None

    @property
    def completed(self):
        return self.tick >= self.sequence_steps

    @property
    def phase(self):
        return cycle_phase(min(self.tick, self.sequence_steps - 1))

    def _pose(self, closure):
        indices = np.asarray(closure) * (len(self.poses) - 1)
        low = np.floor(indices).astype(int)
        high = np.minimum(low + 1, len(self.poses) - 1)
        fraction = indices - low
        pose = np.stack([
            self.poses[low[side], side] * (1. - fraction[side])
            + self.poses[high[side], side] * fraction[side] for side in range(2)
        ])
        if self.settings.continuous_finger_feedback:
            offsets = self.finger_corrections.copy()
            if self.phase in ("drop_settle", "blue_flip_approach", "blue_flip_contact",
                              "blue_flip_clear", "blue_flip_settle"):
                offsets[:] = 0.
            elif self.phase == "blue_flip_release":
                start, end = self.boundaries[self.phase]
                offsets *= 1. - smoothstep((self.tick - start + 1) / (end - start))
            finger_indices = np.clip(np.asarray(closure)[:, None] + offsets, 0., 1.) * (len(self.poses) - 1)
            # Keep the fitted wrist compensation at the shared aperture. Only the
            # four disjoint finger joint blocks receive force-feedback corrections.
            for side in range(2):
                for finger, block in enumerate((slice(18, 22), slice(6, 10), slice(10, 14), slice(14, 18))):
                    value = finger_indices[side, finger]
                    lower = int(np.floor(value))
                    upper = min(lower + 1, len(self.poses) - 1)
                    mix = value - lower
                    pose[side, block] = (self.poses[lower, side, block] * (1. - mix)
                                        + self.poses[upper, side, block] * mix)
        return pose

    def _update_finger_feedback(self, forces, tracked):
        limits = self.settings
        measured = np.maximum(0., np.sum(forces * self.fingertip_normals, axis=2))
        alpha = -math.expm1(-self.fixed_dt / limits.force_filter_seconds)
        self.normal_forces += alpha * (measured - self.normal_forces)
        error = self.force_targets - self.normal_forces
        error = np.sign(error) * np.maximum(0., np.abs(error) - limits.force_deadband_newtons)
        rate = np.clip(limits.finger_feedback_gain * error,
                       -limits.finger_correction_rate, limits.finger_correction_rate)
        if not tracked:
            rate[:] = 0.
        overloaded = np.linalg.norm(forces, axis=2) > limits.maximum_fingertip_force_newtons
        rate[overloaded] = -limits.finger_correction_rate
        offsets = self.finger_corrections + self.fixed_dt * rate
        # Clamp the controller state as well as the command, preventing wind-up
        # when a finger reaches either end of the calibrated pose family.
        lower = np.maximum(-limits.maximum_finger_correction, -self.closure[:, None])
        upper = np.minimum(limits.maximum_finger_correction, 1. - self.closure[:, None])
        self.finger_corrections = np.clip(offsets, lower, upper)
        self.feedback_updates += 1

    def command(self, shell):
        if self.failure is not None or self.completed:
            raise RuntimeError("pinch controller is no longer advancing")
        if shell.shape != self.shell_shape or not np.isfinite(shell).all():
            raise ValueError("measured shell must preserve its finite vertex layout")
        phase = self.phase
        if phase == "drop_settle":
            return np.asarray(hand_controls_at_tick(self.tick))
        if self.centers is None:
            self.anchors = [patch[shell[patch, 2] >= shell[patch, 2].max() - .002]
                            for patch in self.patches]
            self.centers = np.stack([
                shell[anchor].mean(axis=0) for anchor in self.anchors
            ])
            self.centers[:, 2] += self.settings.crest_height_offset_meters
            self.initial_centers = self.centers.copy()
        if phase in ("blue_flip_contact", "blue_flip_grip"):
            for side, anchor in enumerate(self.anchors):
                if self.last_pinches[side]:
                    continue
                desired = shell[anchor].mean(axis=0)
                desired[2] += self.settings.crest_height_offset_meters
                correction = desired - self.initial_centers[side]
                correction *= min(1., self.settings.maximum_target_correction_meters
                                  / max(np.linalg.norm(correction), 1.e-12))
                delta = self.initial_centers[side] + correction - self.centers[side]
                delta *= min(1., self.settings.target_tracking_speed_mps * self.fixed_dt
                             / max(np.linalg.norm(delta), 1.e-12))
                self.centers[side] += delta
        start, end = self.boundaries[phase]
        fraction = smoothstep((self.tick - start + 1) / (end - start))
        closure = self.closure.copy()
        if phase in ("blue_flip_approach", "blue_flip_contact", "blue_flip_clear", "blue_flip_settle"):
            closure[:] = 0.
        elif phase == "blue_flip_release":
            closure *= 1. - fraction
        pose = self._pose(closure)
        pose[:, :3] += self.centers - self.nominal_centers
        if phase == "blue_flip_approach":
            pose[:, 2] += self.settings.approach_clearance_meters
            home = np.asarray(hand_controls_at_tick(start))
            pose = home * (1. - fraction) + pose * fraction
        elif phase == "blue_flip_contact":
            pose[:, 2] += self.settings.approach_clearance_meters * (1. - fraction)
        elif phase != "blue_flip_grip":
            displacement = np.asarray(hand_controls_at_tick(self.tick))[:, :3]
            displacement = displacement - np.asarray(HAND_STAGE_BLUE_FLIP_CONTACT_TARGETS)[:, :3]
            pose[:, :3] += displacement
        return pose

    def observe(self, forces, proxy_error):
        if self.failure is not None or self.completed:
            raise RuntimeError("pinch controller is no longer advancing")
        forces = np.asarray(forces)
        if (forces.shape != (2, 4, 3) or not np.isfinite(forces).all()
                or not np.isfinite(proxy_error) or proxy_error < 0.):
            raise ValueError("pinch feedback must contain finite forces from both hands")
        self.steps += 1
        limits = self.settings
        peaks = np.linalg.norm(forces, axis=2).max(axis=1)
        pinches = [opposed_pinch(hand, limits.fingertip_force_newtons)[0]
                   and peaks[side] <= limits.maximum_fingertip_force_newtons
                   for side, hand in enumerate(forces)]
        tracked = proxy_error <= limits.maximum_proxy_error_meters
        self.last_pinches = [pinch and tracked for pinch in pinches]
        carrying = (self.boundaries["blue_flip_stabilize"][0] <= self.tick
                    < self.boundaries["blue_flip_release"][0])
        # Acquisition has its own aperture search. Arm independent force control
        # only once that search has established a sustained bilateral pinch.
        if self.settings.continuous_finger_feedback and carrying:
            self._update_finger_feedback(forces, tracked)
        if self.phase == "blue_flip_grip":
            self.confirmed_steps = self.confirmed_steps + 1 if all(pinches) and tracked else 0
            for side in range(2):
                if peaks[side] > limits.maximum_fingertip_force_newtons:
                    self.closure[side] = max(0., self.closure[side] - .5 * self.fixed_dt / limits.closing_seconds)
                elif not pinches[side] and tracked:
                    self.closure[side] = min(1., self.closure[side] + self.fixed_dt / limits.closing_seconds)
            if self.tick == self.boundaries["blue_flip_grip"][1] - 1:
                if self.confirmed_steps * self.fixed_dt < limits.confirmed_pinch_seconds:
                    self.wait_steps += 1
                    if self.wait_steps >= limits.maximum_wait_steps:
                        self.failure = "pinch acquisition timed out without sustained bilateral opposing load"
                    return
                self.lift_started_step = self.steps + 1
        elif carrying:
            self.lost_steps = 0 if all(pinches) and tracked else self.lost_steps + 1
            if self.lost_steps * self.fixed_dt >= limits.lost_pinch_seconds:
                self.failure = "bilateral pinch lost during lift/carry; trajectory halted"
                return
            if self.settings.continuous_finger_feedback and self.lost_steps:
                self.motion_hold_steps += 1
                return
        self.tick += 1

    def diagnostics(self):
        return {
            "type": ("per_finger_force_feedback" if self.settings.continuous_finger_feedback
                     else "contact_gated"), "sequence_tick": self.tick,
            "sequence_completed": self.completed, "physical_steps": self.steps,
            "wait_steps": self.wait_steps, "lift_started_step": self.lift_started_step,
            "confirmed_pinch_seconds": self.confirmed_steps * self.fixed_dt,
            "closure": self.closure.tolist(), "failure": self.failure,
            "finger_corrections": self.finger_corrections.tolist(),
            "normal_forces_newtons": self.normal_forces.tolist(),
            "normal_force_targets_newtons": self.force_targets.tolist(),
            "feedback_updates": self.feedback_updates, "motion_hold_steps": self.motion_hold_steps,
            "measured_grasp_centers": None if self.centers is None else self.centers.tolist(),
        }
