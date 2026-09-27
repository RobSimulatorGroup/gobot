"""Scene and control contract shared by Cartpole training and playback."""
from __future__ import annotations

import numpy as np

ROBOT = "cartpole"
BASE_LINK = "rail"
JOINT_NAMES = ("slider", "hinge")
JOINT_PATHS = ("rail/slider", "rail/slider/cart/hinge")
FIXED_DT = 1.0 / 240.0
FORCE_LIMIT = 3.0
DISTURBANCE_STD = 0.05
DISTURBANCE_CLIP = 0.20
DISTURBANCE_INTERVAL_TICKS = 480
DISTURBANCE_DURATION_TICKS = 60
DISTURBANCE_START_TICK = 240


def configure_robot(robot, *, action_limit=FORCE_LIMIT, disturbance_clip=DISTURBANCE_CLIP):
    joints = tuple(robot.find(path) for path in JOINT_PATHS)
    if any(joint is None for joint in joints):
        raise ValueError("Cartpole scene must contain slider and hinge joints")
    slider, hinge = joints
    slider.effort_limit = action_limit
    slider.velocity_limit = max(float(slider.velocity_limit), 20.0)
    hinge.effort_limit = max(float(hinge.effort_limit), disturbance_clip)
    return joints


def observation(joint_position, joint_velocity, target):
    """Return the seven policy inputs for one state or a batch of states."""
    position = np.asarray(joint_position)
    velocity = np.asarray(joint_velocity)
    x, theta = position[..., 0], position[..., 1]
    target = np.broadcast_to(target, x.shape)
    return np.stack((np.cos(theta), np.sin(theta), x, velocity[..., 0],
                     velocity[..., 1], target, x - target), axis=-1).astype(np.float32)
