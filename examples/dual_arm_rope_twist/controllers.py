"""Dual-arm motion, wrist control, and rope metrics for the FR3 task."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


ARM_JOINT_COUNT = 7
MOTION_JOINT_COUNT = 6
JOINT_COUNT = 9
WRIST_INDEX = 6
FINGER_START_INDEX = 7
SETTLE_TICKS = 0
GRIP_TICKS = 60
GRIP_HOLD_TICKS = 300
TWIST_TICKS = 40000
STALL_HOLD_TICKS = 750
GRIP_START_TICK = SETTLE_TICKS
GRIP_COMPLETE_TICK = GRIP_START_TICK + GRIP_TICKS
TWIST_START_TICK = GRIP_COMPLETE_TICK + GRIP_HOLD_TICKS
TWIST_COMPLETE_TICK = TWIST_START_TICK + TWIST_TICKS
TASK_TICKS = TWIST_COMPLETE_TICK
CYCLE_TICKS = TWIST_COMPLETE_TICK + STALL_HOLD_TICKS

WRIST_TARGET_SPEED = 1.60
WRIST_DRIVE_TORQUE_LIMIT = 0.125
WRIST_SHOWCASE_TORQUE_LIMIT = 12.0
FINITE_TORQUE_DRIVE_MODE = "finite-torque"
SHOWCASE_DRIVE_MODE = "showcase"
STALL_TORQUE_FRACTION = 0.90
STALL_REACTION_TORQUE_FRACTION = 0.75
STALL_SPEED_THRESHOLD = 0.08
STALL_CONFIRM_TICKS = 150
STALL_DETECTION_DELAY_TICKS = 1500
STRETCH_PERIOD_TICKS = 2400
ROPE_SIDES = 6

FR3_ROPE_POSE = np.asarray(
    (
        -0.7433863055,
        -0.3703417127,
        0.7452846650,
        -1.8520545191,
        1.9125360536,
        3.1076412964,
        -0.8808,
    ),
    dtype=np.float64,
)
FR3_FINGER_OPEN_POSITION = 0.0240
FR3_FINGER_GRIP_POSITION = 0.0210
FINGER_GRIP_OFFSET = FR3_FINGER_GRIP_POSITION - FR3_FINGER_OPEN_POSITION

# Applying this offline IK offset to both facing FR3s moves their link7 frames
# about 8.3 mm outward in opposite world-X directions while preserving
# orientation. The 16.6 mm total breathing stroke remains visible without
# asking the friction grasp to absorb the rope's full axial stiffness at peak
# stretch.
# Joint7 is excluded so axial breathing and continuous twist remain independent.
SYMMETRIC_PULL_JOINT_TARGET = np.asarray(
    (
        0.04980316,
        -0.01703423,
        -0.04398707,
        -0.02414921,
        -0.00342362,
        0.00341121,
    ),
    dtype=np.float64,
)


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def wrist_drive_torque_limit(drive_mode: str) -> float:
    mode = str(drive_mode).strip().lower()
    if mode == FINITE_TORQUE_DRIVE_MODE:
        return WRIST_DRIVE_TORQUE_LIMIT
    if mode == SHOWCASE_DRIVE_MODE:
        return WRIST_SHOWCASE_TORQUE_LIMIT
    raise ValueError(
        f"drive mode must be {FINITE_TORQUE_DRIVE_MODE!r} or "
        f"{SHOWCASE_DRIVE_MODE!r}, got {drive_mode!r}"
    )


def configure_wrist_torque_limit(
    rigid_solver: Any,
    actuator_ids: Sequence[int],
    torque_limit: float,
) -> None:
    torch = __import__("torch")
    limit = float(torque_limit)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("torque_limit must be finite and positive")
    if len(actuator_ids) != 2:
        raise ValueError("wrist torque configuration expects two actuators")
    force_range = rigid_solver.model_array("actuator_forcerange")
    indices = torch.as_tensor(
        tuple(int(value) for value in actuator_ids),
        dtype=torch.long,
        device=force_range.device,
    )
    if force_range.ndim == 3:
        force_range[:, indices, 0] = -limit
        force_range[:, indices, 1] = limit
    elif force_range.ndim == 2:
        force_range[indices, 0] = -limit
        force_range[indices, 1] = limit
    else:
        raise RuntimeError(
            "unexpected actuator_forcerange shape " + str(tuple(force_range.shape))
        )
    rigid_solver.recompute_constants()


def nominal_wrist_target(tick: int) -> float:
    """Return the local joint7 velocity command shared by both wrists."""

    tick = max(0, int(tick))
    if TWIST_START_TICK <= tick < TWIST_COMPLETE_TICK:
        return WRIST_TARGET_SPEED
    return 0.0


def nominal_stretch_fraction(tick: int) -> float:
    """Cycle smoothly from relaxed to stretched and back while twisting."""

    tick = max(0, int(tick))
    if tick < TWIST_START_TICK:
        return 0.0
    phase = (tick - TWIST_START_TICK) % STRETCH_PERIOD_TICKS
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * phase / STRETCH_PERIOD_TICKS)


def nominal_arm_targets(tick: int) -> np.ndarray:
    """Return mirrored outward IK offsets for the two facing arms."""

    result = np.zeros((2, MOTION_JOINT_COUNT), dtype=np.float64)
    fraction = nominal_stretch_fraction(tick)
    result[:] = SYMMETRIC_PULL_JOINT_TARGET * fraction
    return result


def nominal_finger_target(tick: int) -> float:
    """Return the task-local target shared by all four gripping fingers."""

    tick = max(0, int(tick))
    if tick < GRIP_START_TICK:
        return 0.0
    if tick < GRIP_COMPLETE_TICK:
        progress = _smoothstep((tick - GRIP_START_TICK + 1) / GRIP_TICKS)
        return FINGER_GRIP_OFFSET * progress
    return FINGER_GRIP_OFFSET


def nominal_joint_targets(tick: int) -> np.ndarray:
    """Return complete task-local joint targets for both FR3 robots."""

    result = np.zeros((2, JOINT_COUNT), dtype=np.float64)
    result[:, :MOTION_JOINT_COUNT] = nominal_arm_targets(tick)
    result[:, WRIST_INDEX] = nominal_wrist_target(tick)
    result[:, FINGER_START_INDEX:] = nominal_finger_target(tick)
    return result


def phase_for_tick(tick: int) -> str:
    tick = max(0, int(tick))
    if tick < GRIP_START_TICK:
        return "RobotReady"
    if tick < GRIP_COMPLETE_TICK:
        return "GripFixtures"
    if tick < TWIST_START_TICK:
        return "GripPreload"
    if tick < TWIST_COMPLETE_TICK:
        return "TwistStretchRelax"
    if tick < CYCLE_TICKS:
        return "SafetyHold"
    return "CycleComplete"


@dataclass(frozen=True)
class TwistTrialLayout:
    stall_detection: np.ndarray
    speed_scales: np.ndarray

    def __post_init__(self) -> None:
        stall_detection = np.asarray(self.stall_detection, dtype=np.bool_)
        speed_scales = np.asarray(self.speed_scales, dtype=np.float64)
        if stall_detection.ndim != 1 or speed_scales.shape != stall_detection.shape:
            raise ValueError("twist trial arrays must be one-dimensional and equal")
        if not np.isfinite(speed_scales).all() or np.any(speed_scales <= 0.0):
            raise ValueError("speed scales must be finite and positive")
        object.__setattr__(self, "stall_detection", stall_detection)
        object.__setattr__(self, "speed_scales", speed_scales)


def make_trial_layout(num_envs: int, seed: int = 11) -> TwistTrialLayout:
    count = int(num_envs)
    if count <= 0:
        raise ValueError("num_envs must be positive")
    rng = np.random.default_rng(int(seed))
    scales = np.ones(count, dtype=np.float64)
    if count > 1:
        scales[1:] = rng.uniform(0.96, 1.04, count - 1)
    return TwistTrialLayout(
        stall_detection=np.ones(count, dtype=np.bool_),
        speed_scales=scales,
    )


def stall_detection_layout(num_envs: int) -> TwistTrialLayout:
    count = int(num_envs)
    if count <= 0:
        raise ValueError("num_envs must be positive")
    return TwistTrialLayout(
        stall_detection=np.ones(count, dtype=np.bool_),
        speed_scales=np.ones(count, dtype=np.float64),
    )


class BatchedTwistController:
    """Drive both wrists until finite actuator torque can no longer turn them."""

    def __init__(
        self,
        command_template: Any,
        layout: TwistTrialLayout,
        *,
        fixed_dt: float,
        feedback_enabled: bool = True,
        drive_torque_limit: float = WRIST_DRIVE_TORQUE_LIMIT,
    ) -> None:
        expected = (len(layout.stall_detection), 2, JOINT_COUNT)
        if tuple(command_template.shape) != expected:
            raise ValueError(f"command template must have shape {expected}")
        self._torch = __import__("torch")
        self.fixed_dt = float(fixed_dt)
        if not math.isfinite(self.fixed_dt) or self.fixed_dt <= 0.0:
            raise ValueError("fixed_dt must be finite and positive")
        self.drive_torque_limit = float(drive_torque_limit)
        if (
            not math.isfinite(self.drive_torque_limit)
            or self.drive_torque_limit <= 0.0
        ):
            raise ValueError("drive_torque_limit must be finite and positive")
        self.feedback_enabled = bool(feedback_enabled)
        self.command = command_template.clone().zero_()
        kwargs = {"dtype": self.command.dtype, "device": self.command.device}
        self.stall_detection = self._torch.as_tensor(
            layout.stall_detection,
            dtype=self._torch.bool,
            device=self.command.device,
        )
        self.active_feedback = self.stall_detection.clone()
        if not self.feedback_enabled:
            self.active_feedback.zero_()
        self.speed_scales = self._torch.as_tensor(
            layout.speed_scales, **kwargs
        )
        self.arm_target_table = self._torch.as_tensor(
            np.stack(
                [nominal_arm_targets(tick) for tick in range(CYCLE_TICKS)]
            ),
            **kwargs,
        )
        self.held_arm_target = self._torch.zeros(
            (self.command.shape[0], 2, MOTION_JOINT_COUNT), **kwargs
        )
        self.gravity_schedule_indices = self._torch.zeros(
            self.command.shape[0],
            dtype=self._torch.long,
            device=self.command.device,
        )
        self.filtered_wrench = self._torch.zeros(
            (self.command.shape[0], 2, 6), **kwargs
        )
        self.filtered_axial_torque = self._torch.zeros(
            (self.command.shape[0], 2), **kwargs
        )
        self.filtered_wrist_speed = self._torch.zeros(
            (self.command.shape[0], 2), **kwargs
        )
        self.filtered_wrist_effort = self._torch.zeros(
            (self.command.shape[0], 2), **kwargs
        )
        self.peak_axial_torque = self._torch.zeros(
            self.command.shape[0], **kwargs
        )
        self.peak_commanded_speed = self._torch.zeros_like(
            self.peak_axial_torque
        )
        self.peak_actual_relative_rotation = self._torch.zeros_like(
            self.peak_axial_torque
        )
        self.stalled_relative_rotation = self._torch.zeros_like(
            self.peak_axial_torque
        )
        self.stalled_wrist_speed = self._torch.zeros(
            (self.command.shape[0], 2), **kwargs
        )
        self.stalled_wrist_effort = self._torch.zeros(
            (self.command.shape[0], 2), **kwargs
        )
        self.stalled_axial_torque = self._torch.zeros(
            (self.command.shape[0], 2), **kwargs
        )
        self.stall_tick = self._torch.full(
            (self.command.shape[0],),
            -1,
            dtype=self._torch.int32,
            device=self.command.device,
        )
        self.stall_counter = self._torch.zeros(
            self.command.shape[0],
            dtype=self._torch.int32,
            device=self.command.device,
        )
        self.hold_counter = self._torch.zeros_like(self.stall_counter)
        self.stalled = self._torch.zeros_like(self.stall_detection)
        self.safety_stopped = self._torch.zeros_like(self.stall_detection)
        self.complete = self._torch.zeros_like(self.stall_detection)
        self.tick = 0
        self.reset()

    @property
    def phase(self) -> str:
        if bool(self.stalled.any().item()):
            return "TorqueStalled"
        if bool(self.safety_stopped.any().item()):
            return "SafetyHold"
        return phase_for_tick(self.tick)

    @property
    def cycle_complete(self) -> bool:
        return bool(self.complete.all().item())

    def reset(self) -> Any:
        self.command.zero_()
        self.filtered_wrench.zero_()
        self.held_arm_target.zero_()
        self.gravity_schedule_indices.zero_()
        self.filtered_axial_torque.zero_()
        self.filtered_wrist_speed.zero_()
        self.filtered_wrist_effort.zero_()
        self.peak_axial_torque.zero_()
        self.peak_commanded_speed.zero_()
        self.peak_actual_relative_rotation.zero_()
        self.stalled_relative_rotation.zero_()
        self.stalled_wrist_speed.zero_()
        self.stalled_wrist_effort.zero_()
        self.stalled_axial_torque.zero_()
        self.stall_tick.fill_(-1)
        self.stall_counter.zero_()
        self.hold_counter.zero_()
        self.stalled.zero_()
        self.safety_stopped.zero_()
        self.complete.zero_()
        self.tick = 0
        return self.command

    def step(
        self,
        end_effector_wrenches: Any,
        joint_positions: Any,
        joint_velocities: Any,
        wrist_efforts: Any,
    ) -> Any:
        if tuple(end_effector_wrenches.shape) != tuple(
            self.filtered_wrench.shape
        ):
            raise ValueError(
                "end-effector wrenches must have shape "
                f"{tuple(self.filtered_wrench.shape)}"
            )
        expected_joint_shape = tuple(self.command.shape)
        if tuple(joint_positions.shape) != expected_joint_shape or tuple(
            joint_velocities.shape
        ) != expected_joint_shape:
            raise ValueError(
                f"joint state tensors must have shape {expected_joint_shape}"
            )
        if tuple(wrist_efforts.shape) != tuple(self.filtered_wrist_effort.shape):
            raise ValueError(
                "wrist efforts must have shape "
                f"{tuple(self.filtered_wrist_effort.shape)}"
            )
        self.filtered_wrench.mul_(0.82).add_(end_effector_wrenches, alpha=0.18)
        # Input wrenches are expressed in their corresponding link7 frames.
        self.filtered_axial_torque.copy_(
            self._torch.abs(self.filtered_wrench[:, :, 5])
        )
        torque = self.filtered_axial_torque.amax(dim=1)
        self._torch.maximum(
            self.peak_axial_torque, torque, out=self.peak_axial_torque
        )
        actual_wrist_speed = joint_velocities[:, :, WRIST_INDEX].abs()
        self.filtered_wrist_speed.mul_(0.94).add_(
            actual_wrist_speed, alpha=0.06
        )
        self.filtered_wrist_effort.mul_(0.94).add_(
            wrist_efforts.abs(), alpha=0.06
        )
        actual_relative_rotation = joint_positions[:, :, WRIST_INDEX].abs().sum(
            dim=1
        )
        self._torch.maximum(
            self.peak_actual_relative_rotation,
            actual_relative_rotation,
            out=self.peak_actual_relative_rotation,
        )

        can_detect = (
            self.active_feedback
            & ~self.stalled
            & ~self.safety_stopped
            & (self.tick >= TWIST_START_TICK + STALL_DETECTION_DELAY_TICKS)
        )
        reaction_loaded = self.filtered_axial_torque.amin(dim=1) >= (
            self.drive_torque_limit * STALL_REACTION_TORQUE_FRACTION
        )
        effort_saturated = self.filtered_wrist_effort.amin(dim=1) >= (
            self.drive_torque_limit * STALL_TORQUE_FRACTION
        )
        barely_moving = self.filtered_wrist_speed.amax(dim=1) <= (
            STALL_SPEED_THRESHOLD
        )
        stalled_sample = (
            can_detect & reaction_loaded & effort_saturated & barely_moving
        )
        self.stall_counter.copy_(
            self._torch.where(
                stalled_sample,
                self.stall_counter + 1,
                self._torch.zeros_like(self.stall_counter),
            )
        )
        was_holding = self.stalled | self.safety_stopped
        newly_stalled = (self.stall_counter >= STALL_CONFIRM_TICKS) & ~self.stalled
        self.stalled.logical_or_(newly_stalled)
        self.stalled_relative_rotation.copy_(
            self._torch.where(
                newly_stalled,
                actual_relative_rotation,
                self.stalled_relative_rotation,
            )
        )
        self.stalled_wrist_speed.copy_(
            self._torch.where(
                newly_stalled[:, None],
                self.filtered_wrist_speed,
                self.stalled_wrist_speed,
            )
        )
        self.stalled_wrist_effort.copy_(
            self._torch.where(
                newly_stalled[:, None],
                self.filtered_wrist_effort,
                self.stalled_wrist_effort,
            )
        )
        self.stalled_axial_torque.copy_(
            self._torch.where(
                newly_stalled[:, None],
                self.filtered_axial_torque,
                self.stalled_axial_torque,
            )
        )
        self.stall_tick.copy_(
            self._torch.where(
                newly_stalled,
                self._torch.full_like(self.stall_tick, self.tick),
                self.stall_tick,
            )
        )
        if self.tick >= TWIST_COMPLETE_TICK:
            self.safety_stopped.logical_or_(~self.stalled)
        holding = self.stalled | self.safety_stopped
        newly_holding = holding & ~was_holding
        self.hold_counter.add_(holding.to(self.hold_counter.dtype))
        self.complete.logical_or_(self.hold_counter >= STALL_HOLD_TICKS)

        wrist_speed = nominal_wrist_target(self.tick) * self.speed_scales
        wrist_speed = self._torch.where(
            self.safety_stopped | self.complete,
            self._torch.zeros_like(wrist_speed),
            wrist_speed,
        )
        self.command.zero_()
        arm_target = self.arm_target_table[min(self.tick, CYCLE_TICKS - 1)]
        expanded_arm_target = arm_target.unsqueeze(0).expand_as(
            self.held_arm_target
        )
        self.held_arm_target.copy_(
            self._torch.where(
                newly_holding[:, None, None],
                expanded_arm_target,
                self.held_arm_target,
            )
        )
        self.gravity_schedule_indices.copy_(
            self._torch.where(
                ~was_holding,
                self._torch.full_like(
                    self.gravity_schedule_indices,
                    min(self.tick, CYCLE_TICKS - 1),
                ),
                self.gravity_schedule_indices,
            )
        )
        held = holding[:, None, None]
        self.command[:, :, :MOTION_JOINT_COUNT].copy_(
            self._torch.where(
                held,
                self.held_arm_target,
                expanded_arm_target,
            )
        )
        self.command[:, :, WRIST_INDEX].copy_(wrist_speed[:, None])
        self.command[:, :, FINGER_START_INDEX:].fill_(
            nominal_finger_target(self.tick)
        )
        self._torch.maximum(
            self.peak_commanded_speed,
            wrist_speed.abs(),
            out=self.peak_commanded_speed,
        )
        self.tick += 1
        return self.command


@dataclass(frozen=True)
class GravityCompensationSchedule:
    offset: np.ndarray
    cosine: np.ndarray
    sine: np.ndarray
    joint_dof_addresses: tuple[tuple[int, ...], ...]


def gravity_compensation_schedule(
    mjcf: str,
    robot_names: Sequence[str],
) -> GravityCompensationSchedule:
    """Fit each robot's gravity load as a sinusoid of its wrist angle."""

    mujoco = __import__("mujoco")
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    joint_qpos_addresses = []
    joint_dof_addresses = []
    for robot_name in robot_names:
        qpos_addresses = []
        dof_addresses = []
        for joint_name in (
            *(f"fr3_joint{index}" for index in range(1, 8)),
            "fr3_finger_joint1",
            "fr3_finger_joint2",
        ):
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{robot_name}_{joint_name}",
            )
            if joint_id < 0:
                raise RuntimeError(
                    f"MuJoCo model has no {joint_name!r} for {robot_name!r}"
                )
            qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
            dof_addresses.append(int(model.jnt_dofadr[joint_id]))
        joint_qpos_addresses.append(qpos_addresses)
        joint_dof_addresses.append(dof_addresses)
    shape = (CYCLE_TICKS, len(robot_names), JOINT_COUNT)
    offset = np.zeros(shape, dtype=np.float32)
    cosine = np.zeros(shape, dtype=np.float32)
    sine = np.zeros(shape, dtype=np.float32)
    for tick in range(CYCLE_TICKS):
        data.qpos[:] = model.qpos0
        targets = nominal_joint_targets(tick)
        for robot_index, addresses in enumerate(joint_qpos_addresses):
            data.qpos[addresses[:MOTION_JOINT_COUNT]] = targets[
                robot_index, :MOTION_JOINT_COUNT
            ]
            data.qpos[addresses[FINGER_START_INDEX:]] = targets[
                robot_index, FINGER_START_INDEX:
            ]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        at_zero = np.stack(
            tuple(data.qfrc_bias[addresses] for addresses in joint_dof_addresses)
        )
        for addresses in joint_qpos_addresses:
            data.qpos[addresses[WRIST_INDEX]] = 0.5 * math.pi
        mujoco.mj_forward(model, data)
        at_half_pi = np.stack(
            tuple(data.qfrc_bias[addresses] for addresses in joint_dof_addresses)
        )
        for addresses in joint_qpos_addresses:
            data.qpos[addresses[WRIST_INDEX]] = math.pi
        mujoco.mj_forward(model, data)
        at_pi = np.stack(
            tuple(data.qfrc_bias[addresses] for addresses in joint_dof_addresses)
        )
        offset[tick] = 0.5 * (at_zero + at_pi)
        cosine[tick] = 0.5 * (at_zero - at_pi)
        sine[tick] = at_half_pi - offset[tick]
    return GravityCompensationSchedule(
        offset=offset,
        cosine=cosine,
        sine=sine,
        joint_dof_addresses=tuple(
            tuple(addresses) for addresses in joint_dof_addresses
        ),
    )


class BatchedGravityCompensator:
    """Apply gravity-only generalized forces for both articulated robots."""

    def __init__(self, schedule: GravityCompensationSchedule, template: Any) -> None:
        torch = __import__("torch")
        if template.ndim != 2:
            raise ValueError("gravity compensation template must be two-dimensional")
        kwargs = {"dtype": template.dtype, "device": template.device}
        self.offset = torch.as_tensor(schedule.offset, **kwargs)
        self.cosine = torch.as_tensor(schedule.cosine, **kwargs)
        self.sine = torch.as_tensor(schedule.sine, **kwargs)
        self.joint_dof_addresses = tuple(
            torch.as_tensor(
                addresses, dtype=torch.long, device=template.device
            )
            for addresses in schedule.joint_dof_addresses
        )
        self.values = torch.empty(
            (template.shape[0], len(self.joint_dof_addresses), JOINT_COUNT),
            **kwargs,
        )
        self._torch = torch

    def apply(self, target: Any, joint_positions: Any, tick: Any) -> None:
        if tuple(joint_positions.shape) != tuple(self.values.shape):
            raise ValueError(
                "gravity compensation joint positions must have shape "
                f"{tuple(self.values.shape)}"
            )
        if isinstance(tick, int):
            index: Any = min(max(0, tick), self.offset.shape[0] - 1)
        else:
            index = tick.to(
                dtype=self._torch.long, device=self.offset.device
            ).clamp(0, self.offset.shape[0] - 1)
        wrist_angle = joint_positions[:, :, WRIST_INDEX, None]
        self.values.copy_(self.offset[index])
        self.values.add_(self.cosine[index] * wrist_angle.cos())
        self.values.add_(self.sine[index] * wrist_angle.sin())
        target.zero_()
        for robot_index, addresses in enumerate(self.joint_dof_addresses):
            target.index_copy_(1, addresses, self.values[:, robot_index])


def absolute_joint_positions(task_positions: Any) -> Any:
    torch = __import__("torch")
    result = task_positions.clone()
    result[..., :ARM_JOINT_COUNT].add_(
        torch.as_tensor(
            FR3_ROPE_POSE,
            dtype=result.dtype,
            device=result.device,
        )
    )
    result[..., ARM_JOINT_COUNT:].add_(FR3_FINGER_OPEN_POSITION)
    return result


def fixture_wrenches_in_tool_frames(
    rigid_arrays: Mapping[str, Any],
    fixture_body_ids: Sequence[int],
    tool_body_ids: Sequence[int],
) -> Any:
    """Express each fixture's applied wrench about its link7 origin."""

    torch = __import__("torch")
    if len(fixture_body_ids) != 2 or len(tool_body_ids) != 2:
        raise ValueError("fixture wrench conversion expects two fixture/tool pairs")
    fixtures = list(int(value) for value in fixture_body_ids)
    tools = list(int(value) for value in tool_body_ids)
    world = rigid_arrays["xfrc_applied"][:, fixtures]
    force_world = world[..., :3]
    moment_arm = rigid_arrays["xipos"][:, fixtures]
    moment_arm = moment_arm - rigid_arrays["xpos"][:, tools]
    torque_world = world[..., 3:] + torch.cross(
        moment_arm, force_world, dim=2
    )
    rotations = rigid_arrays["xmat"][:, tools]
    rotations = rotations.reshape(*rotations.shape[:2], 3, 3)
    force_local = torch.matmul(
        rotations.transpose(2, 3), force_world.unsqueeze(3)
    )[..., 0]
    torque_local = torch.matmul(
        rotations.transpose(2, 3), torque_world.unsqueeze(3)
    )[..., 0]
    return torch.cat((force_local, torque_local), dim=2)


def rope_endpoint_index_sets(
    deformable_bodies: Sequence[Mapping[str, Any]], device: Any
) -> Any:
    """Return global indices shaped as side, strand, end-section vertex."""

    torch = __import__("torch")
    section_size = ROPE_SIDES + 1
    if len(deformable_bodies) != 3:
        raise ValueError("rope grip metric expects exactly three strands")
    endpoint_indices = ([], [])
    for body in deformable_bodies:
        offset = int(body["element_offset"])
        count = int(body["element_count"])
        if count < 2 * section_size:
            raise ValueError("rope strand has too few vertices for end sections")
        endpoint_indices[0].append(
            list(range(offset, offset + section_size))
        )
        endpoint_indices[1].append(
            list(range(offset + count - section_size, offset + count))
        )
    return torch.as_tensor(
        endpoint_indices, dtype=torch.int64, device=device
    )


def rope_endpoints_in_body_frames(
    positions: Any,
    endpoint_index_sets: Any,
    rigid_arrays: Mapping[str, Any],
    body_ids: Sequence[int],
) -> Any:
    """Measure all six rope-end centroids in their fixture frames."""

    torch = __import__("torch")
    if tuple(endpoint_index_sets.shape[:2]) != (2, 3) or len(body_ids) != 2:
        raise ValueError("rope endpoint metric expects two fixtures and three strands")
    world_endpoints = positions[:, endpoint_index_sets].mean(dim=3)
    ids = list(int(value) for value in body_ids)
    body_positions = rigid_arrays["xpos"][:, ids].to(dtype=positions.dtype)
    rotations = rigid_arrays["xmat"][:, ids].to(dtype=positions.dtype)
    rotations = rotations.reshape(*rotations.shape[:2], 3, 3)
    delta = world_endpoints - body_positions[:, :, None, :]
    return torch.einsum("ebji,ebsj->ebsi", rotations, delta)


def rope_endpoints_in_affine_frames(
    positions: Any,
    endpoint_index_sets: Any,
    affine_targets: Any,
    body_indices: Sequence[int],
) -> Any:
    """Measure all six rope-end centroids in their IPC proxy frames."""

    torch = __import__("torch")
    if tuple(endpoint_index_sets.shape[:2]) != (2, 3) or len(body_indices) != 2:
        raise ValueError("rope endpoint metric expects two proxies and three strands")
    if affine_targets.ndim != 4 or tuple(affine_targets.shape[-2:]) != (4, 4):
        raise ValueError("affine target transforms must have shape [N, B, 4, 4]")
    world_endpoints = positions[:, endpoint_index_sets].mean(dim=3)
    transforms = affine_targets[:, list(int(value) for value in body_indices)].to(
        dtype=positions.dtype
    )
    body_positions = transforms[..., :3, 3]
    rotations = transforms[..., :3, :3]
    delta = world_endpoints - body_positions[:, :, None, :]
    return torch.einsum("ebji,ebsj->ebsi", rotations, delta)


def body_transforms_in_reference_frames(
    rigid_arrays: Mapping[str, Any],
    body_ids: Sequence[int],
    reference_body_ids: Sequence[int],
) -> tuple[Any, Any]:
    """Return body position and rotation in corresponding reference frames."""

    torch = __import__("torch")
    if len(body_ids) != 2 or len(reference_body_ids) != 2:
        raise ValueError("relative fixture pose expects two body/reference pairs")
    bodies = list(int(value) for value in body_ids)
    references = list(int(value) for value in reference_body_ids)
    body_position = rigid_arrays["xpos"][:, bodies]
    reference_position = rigid_arrays["xpos"][:, references]
    body_rotation = rigid_arrays["xmat"][:, bodies].reshape(-1, 2, 3, 3)
    reference_rotation = rigid_arrays["xmat"][:, references].reshape(
        -1, 2, 3, 3
    )
    reference_inverse = reference_rotation.transpose(2, 3)
    local_position = torch.matmul(
        reference_inverse,
        (body_position - reference_position).unsqueeze(3),
    )[..., 0]
    local_rotation = torch.matmul(reference_inverse, body_rotation)
    return local_position, local_rotation


def relative_transform_errors(
    position: Any,
    rotation: Any,
    reference_position: Any,
    reference_rotation: Any,
) -> tuple[Any, Any]:
    """Return translation norm and rotation angle from a reference transform."""

    torch = __import__("torch")
    translation = (position - reference_position).norm(dim=2)
    delta_rotation = torch.matmul(
        rotation, reference_rotation.transpose(2, 3)
    )
    cosine = (delta_rotation.diagonal(dim1=2, dim2=3).sum(dim=2) - 1.0) * 0.5
    angle = torch.acos(cosine.clamp(-1.0, 1.0))
    return translation, angle


def _unwrap_delta(delta: Any) -> Any:
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def rope_winding_turns(
    positions: Any,
    deformable_bodies: Sequence[Mapping[str, Any]],
) -> Any:
    """Measure each strand centerline winding around the bundle centerline."""

    torch = __import__("torch")
    if len(deformable_bodies) != 3:
        raise ValueError("rope winding metric expects exactly three strands")
    centers = []
    stride = ROPE_SIDES + 1
    for body in deformable_bodies:
        offset = int(body["element_offset"])
        count = int(body["element_count"])
        indices = torch.arange(
            offset,
            offset + count,
            stride,
            dtype=torch.int64,
            device=positions.device,
        )
        centers.append(positions.index_select(1, indices))
    strand_centers = torch.stack(centers, dim=1)
    bundle_center = strand_centers.mean(dim=1, keepdim=True)
    radial = strand_centers - bundle_center
    angles = torch.atan2(radial[..., 2], radial[..., 1])
    span = _unwrap_delta(angles[..., 1:] - angles[..., :-1]).sum(dim=-1)
    return span / (2.0 * math.pi)


def maximum_shape_deformation(initial_positions: Any, positions: Any) -> Any:
    displacement = positions - initial_positions
    displacement = displacement - displacement.mean(dim=1, keepdim=True)
    return displacement.norm(dim=2).amax(dim=1)


def maximum_box_vertex_penetration(
    positions: Any,
    box_transforms: Any,
    box_sizes: Any,
) -> Any:
    """Return each environment's deepest vertex inside any oriented box."""

    torch = __import__("torch")
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [N, V, 3]")
    if box_transforms.ndim != 4 or tuple(box_transforms.shape[-2:]) != (4, 4):
        raise ValueError("box transforms must have shape [N, B, 4, 4]")
    if box_sizes.ndim != 2 or box_sizes.shape[-1] != 3:
        raise ValueError("box sizes must have shape [B, 3]")
    if box_transforms.shape[0] != positions.shape[0]:
        raise ValueError("position and box environment counts must match")
    if box_transforms.shape[1] != box_sizes.shape[0]:
        raise ValueError("box transform and size counts must match")
    if box_sizes.shape[0] == 0:
        return torch.zeros(
            positions.shape[0],
            dtype=positions.dtype,
            device=positions.device,
        )

    rotations = box_transforms[..., :3, :3]
    translations = box_transforms[..., :3, 3]
    delta = positions[:, None, :, :] - translations[:, :, None, :]
    local = torch.einsum("nbji,nbvj->nbvi", rotations, delta)
    margins = box_sizes[None, :, None, :] * 0.5 - local.abs()
    inside_depth = margins.amin(dim=-1).clamp_min_(0.0)
    return inside_depth.amax(dim=(1, 2))


__all__ = [
    "BatchedGravityCompensator",
    "BatchedTwistController",
    "CYCLE_TICKS",
    "FINITE_TORQUE_DRIVE_MODE",
    "FINGER_GRIP_OFFSET",
    "FR3_FINGER_GRIP_POSITION",
    "FR3_FINGER_OPEN_POSITION",
    "GRIP_COMPLETE_TICK",
    "GRIP_HOLD_TICKS",
    "GRIP_START_TICK",
    "GRIP_TICKS",
    "SETTLE_TICKS",
    "SHOWCASE_DRIVE_MODE",
    "STALL_CONFIRM_TICKS",
    "STALL_DETECTION_DELAY_TICKS",
    "STALL_HOLD_TICKS",
    "STALL_REACTION_TORQUE_FRACTION",
    "STALL_SPEED_THRESHOLD",
    "STALL_TORQUE_FRACTION",
    "STRETCH_PERIOD_TICKS",
    "SYMMETRIC_PULL_JOINT_TARGET",
    "TASK_TICKS",
    "TWIST_COMPLETE_TICK",
    "TWIST_START_TICK",
    "TwistTrialLayout",
    "WRIST_DRIVE_TORQUE_LIMIT",
    "WRIST_SHOWCASE_TORQUE_LIMIT",
    "WRIST_TARGET_SPEED",
    "absolute_joint_positions",
    "body_transforms_in_reference_frames",
    "configure_wrist_torque_limit",
    "fixture_wrenches_in_tool_frames",
    "gravity_compensation_schedule",
    "make_trial_layout",
    "maximum_box_vertex_penetration",
    "maximum_shape_deformation",
    "nominal_arm_targets",
    "nominal_finger_target",
    "nominal_joint_targets",
    "nominal_stretch_fraction",
    "nominal_wrist_target",
    "phase_for_tick",
    "rope_endpoint_index_sets",
    "rope_endpoints_in_affine_frames",
    "rope_endpoints_in_body_frames",
    "relative_transform_errors",
    "rope_winding_turns",
    "stall_detection_layout",
    "wrist_drive_torque_limit",
]
