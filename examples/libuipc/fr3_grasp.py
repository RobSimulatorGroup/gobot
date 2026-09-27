"""FR3 grasp task timing and trajectory, independent of Play and physics backends."""

FIXED_DT = 0.01
LOOP_SECONDS = 10.0
SCENE_ROOT_NAME = "libuipc_fr3_soft_grasp"
FR3_INITIAL_ARM = {
    "fr3_joint1": -0.0036802115,
    "fr3_joint2": 0.023901723,
    "fr3_joint3": 0.003680411,
    "fr3_joint4": -2.3683236,
    "fr3_joint5": -0.00012918962,
    "fr3_joint6": 2.3922248,
    "fr3_joint7": 0.785492,
}
FR3_ARM_LIFT_OFFSETS = {
    "fr3_joint1": 0.0,
    "fr3_joint2": -0.07,
    "fr3_joint3": 0.0,
    "fr3_joint4": 0.055,
    "fr3_joint5": 0.0,
    "fr3_joint6": 0.055,
    "fr3_joint7": 0.0,
}
FR3_FINGER_JOINTS = (
    "fr3_finger_joint1",
    "fr3_finger_joint2",
)
FR3_JOINT_NAMES = tuple(FR3_INITIAL_ARM) + FR3_FINGER_JOINTS
FR3_OPEN_FINGER = 0.017
FR3_CLOSED_FINGER = 0.0146


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def _transition(time_seconds: float, start: float, end: float) -> float:
    if end <= start:
        raise ValueError("motion transition end must be greater than start")
    return _smoothstep((time_seconds - start) / (end - start))


def motion_targets(time_seconds: float) -> dict[str, float]:
    time_seconds = max(0.0, float(time_seconds))
    if time_seconds < 1.0:
        grasp = 0.0
    elif time_seconds < 2.4:
        grasp = _transition(time_seconds, 1.0, 2.4)
    elif time_seconds < 8.5:
        grasp = 1.0
    elif time_seconds < 9.5:
        grasp = 1.0 - _transition(time_seconds, 8.5, 9.5)
    else:
        grasp = 0.0

    if time_seconds < 3.6:
        lift = 0.0
    elif time_seconds < 5.2:
        lift = _transition(time_seconds, 3.6, 5.2)
    elif time_seconds < 6.2:
        lift = 1.0
    elif time_seconds < 7.6:
        lift = 1.0 - _transition(time_seconds, 6.2, 7.6)
    else:
        lift = 0.0

    targets = {
        name: FR3_INITIAL_ARM[name] + offset * lift
        for name, offset in FR3_ARM_LIFT_OFFSETS.items()
    }
    finger_target = FR3_OPEN_FINGER + (
        FR3_CLOSED_FINGER - FR3_OPEN_FINGER
    ) * grasp
    targets.update(
        {name: finger_target for name in FR3_FINGER_JOINTS}
    )
    return targets
