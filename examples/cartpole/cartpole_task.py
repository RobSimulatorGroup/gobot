"""Single-pole scene and control contract shared by planning and playback."""
from __future__ import annotations

import numpy as np

ROBOT = "cartpole"
JOINT_NAMES = ("slider", "hinge")
JOINT_PATHS = ("rail/slider", "rail/slider/cart/hinge")
FIXED_DT = 1.0 / 240.0
SUBSTEPS = 6
HORIZON = 160
FORCE_LIMIT = 15.0
INITIAL_STATE = np.array([0.0, np.pi, 0.0, 0.0])


def configure_context(context):
    """Select the CPU world before compiling or starting editor playback."""
    import gobot

    context.fixed_time_step = FIXED_DT
    context.backend_type = gobot.PhysicsBackendType.MuJoCoCpu
    robot = context.root
    if robot is None or robot.name != ROBOT:
        raise ValueError("Cartpole requires the cartpole.jscn robot root")
    joints = tuple(robot.find(path) for path in JOINT_PATHS)
    if any(joint is None for joint in joints):
        raise ValueError("Cartpole scene must contain slider and hinge joints")
    slider, hinge = joints
    if slider.drive_mode != gobot.JointDriveMode.Motor or hinge.drive_mode != gobot.JointDriveMode.Passive:
        raise ValueError("Cartpole requires a motorized slider and a passive hinge")
    if not np.isclose(slider.effort_limit, FORCE_LIMIT):
        raise ValueError(f"Cartpole slider effort limit must be {FORCE_LIMIT:g} N")
    return joints


def reset(joints):
    for i, joint in enumerate(joints):
        joint.reset_runtime_state(float(INITIAL_STATE[i]), 0.0)
    joints[0].set_effort_target(0.0)
    joints[1].set_passive()


def state(joints):
    values = [joint.get_runtime_state() for joint in joints]
    return np.array([values[0]["position"], values[1]["position"],
                     values[0]["velocity"], values[1]["velocity"]], dtype=np.float64)


def wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
