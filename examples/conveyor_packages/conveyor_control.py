"""Conveyor hand commands and traction shared by batch and editor Play."""

from typing import Any

from conveyor_config import (
    BELT_CENTER_X, BELT_CENTER_Y, BELT_DRIVE_FRICTION, BELT_LINK_NAME,
    BELT_ROBOT_NAME, BELT_SURFACE_LENGTH, BELT_TOP_Z, BELT_WIDTH,
    HAND_JOINT_NAMES_BY_SIDE, LEAP_ROBOT_NAMES, RIGID_BOX_MASSES,
    RIGID_BOX_NAMES, SOFT_PACKAGE_NAMES,
)
from conveyor_forces import ConveyorForceModel, DeformableConveyorForceModel
from conveyor_profile import (
    FIXED_DT, SOFT_PACKAGE_DAMPING_RATES, SOFT_PACKAGE_MASSES,
    belt_speed_at_tick, hand_controls_at_tick, soft_damping_scale_at_tick,
)


def _hand_joints(root: Any) -> tuple[tuple[Any, ...], ...]:
    hands = []
    for robot_name, names in zip(LEAP_ROBOT_NAMES, HAND_JOINT_NAMES_BY_SIDE, strict=True):
        robot = root.find(robot_name)
        if robot is None or robot.type_name != "Robot3D":
            raise ValueError(f"conveyor scene has no robot {robot_name!r}")
        joints = {}
        pending = [robot]
        while pending:
            node = pending.pop()
            if node.type_name == "Joint3D":
                if node.name in joints:
                    raise ValueError(f"{robot_name} has duplicate joint {node.name!r}")
                joints[node.name] = node
            pending.extend(node.children)
        missing = set(names) - joints.keys()
        if missing:
            raise ValueError(f"{robot_name} is missing joints: {sorted(missing)}")
        hands.append(tuple(joints[name] for name in names))
    return tuple(hands)


class ConveyorControl:
    """Apply commands before a physics tick; the caller owns stepping and timing."""

    def __init__(self, context: Any, root: Any) -> None:
        self.context = context
        self.hand_joints = _hand_joints(root)
        self.rigid_forces = ConveyorForceModel(
            context, RIGID_BOX_NAMES, RIGID_BOX_MASSES,
            belt_robot=BELT_ROBOT_NAME, belt_link=BELT_LINK_NAME,
            friction_coefficient=BELT_DRIVE_FRICTION, fixed_dt=FIXED_DT,
        )
        self.soft_forces = DeformableConveyorForceModel(
            context, SOFT_PACKAGE_NAMES, SOFT_PACKAGE_MASSES,
            friction_coefficient=BELT_DRIVE_FRICTION, fixed_dt=FIXED_DT,
            belt_half_length=0.5 * BELT_SURFACE_LENGTH,
            belt_half_width=0.5 * BELT_WIDTH, belt_top=BELT_TOP_Z,
            belt_center_x=BELT_CENTER_X, belt_center_y=BELT_CENTER_Y,
            velocity_damping_rates=SOFT_PACKAGE_DAMPING_RATES,
        )

    def _apply_hand_targets(self, tick: int) -> None:
        for joints, targets in zip(self.hand_joints, hand_controls_at_tick(tick), strict=True):
            for joint, target in zip(joints, targets, strict=True):
                joint.set_position_target(float(target))

    def apply(self, tick: int, state: dict[str, Any]) -> float:
        speed = float(belt_speed_at_tick(tick))
        self._apply_hand_targets(tick)
        self.rigid_forces.apply(speed, state)
        self.soft_forces.apply(speed, damping_scale=soft_damping_scale_at_tick(tick), state=state)
        return speed

    def warmup(self, state: dict[str, Any]) -> None:
        self._apply_hand_targets(0)
        self.rigid_forces.apply(0.0, state)
        self.soft_forces.apply(0.0, state=state)

    def reset(self) -> None:
        self.context.clear_external_forces()
        self.context.reset_simulation()
