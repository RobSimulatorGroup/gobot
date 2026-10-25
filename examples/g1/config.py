"""G1 rough velocity preset, adapted from mjlab 8bb2bc9e (Apache-2.0).

The existing 43-joint asset retains its hands; the policy controls 29 body joints.
"""
from __future__ import annotations
import math
from pathlib import Path
from gobot.rl.locomotion.config import FootCfg, LocomotionRobotCfg, VelocityTaskCfg

REFERENCE_REVISION = "8bb2bc9e187797689ec82c78c35019afd2fd0376"
JOINT_NAMES = tuple(
    [f"{side}_{name}_joint" for side in ("left", "right")
     for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
    + [f"{side}_{name}_joint" for side in ("left", "right")
       for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
)
HAND_JOINTS = tuple(f"{side}_hand_{finger}_{segment}_joint"
    for side in ("left", "right") for finger, segments in (("index", 2), ("middle", 2), ("thumb", 3))
    for segment in range(segments))

def _inertia(rotors, gears):
    return rotors[0] * (gears[0] * gears[1]) ** 2 + rotors[1] * gears[1] ** 2 + rotors[2]

_MOTORS = {
    "5020": (_inertia((.139e-4, .017e-4, .169e-4), (1+46/18, 1+56/16)), 25., 37.),
    "7520_14": (_inertia((.489e-4, .098e-4, .533e-4), (4.5, 1+48/22)), 88., 32.),
    "7520_22": (_inertia((.489e-4, .109e-4, .738e-4), (4.5, 5.)), 139., 20.),
    "4010": (_inertia((.068e-4, 0., 0.), (5., 5.)), 5., 22.),
}

def _motor(name):
    key, factor = "5020", 1
    if "hip_pitch" in name or "hip_yaw" in name or "waist_yaw" in name:
        key = "7520_14"
    elif "hip_roll" in name or "knee" in name:
        key = "7520_22"
    elif "wrist_pitch" in name or "wrist_yaw" in name:
        key = "4010"
    elif "ankle" in name or "waist" in name:
        factor = 2
    inertia, effort, speed = _MOTORS[key]
    return inertia * factor, effort * factor, speed

def _initial(name):
    for part, value in (("hip_pitch", -.312), ("knee", .669), ("ankle_pitch", -.363),
                        ("elbow", .6), ("shoulder_pitch", .2)):
        if part in name:
            return value
    if "shoulder_roll" in name:
        return .2 if name.startswith("left") else -.2
    return 0.

def g1_velocity_cfg(*, project_path=None, play=False):
    motors = [_motor(name) for name in JOINT_NAMES]
    omega = 10 * 2 * math.pi
    kp = tuple(m[0] * omega**2 for m in motors)
    kd = tuple(4 * m[0] * omega for m in motors)
    offsets = ((0., 0., 0.),) + tuple((.03 * math.cos(i * math.tau / 6), .03 * math.sin(i * math.tau / 6), 0.) for i in range(6))
    profile = LocomotionRobotCfg(
        name="g1", base_link="pelvis", torso_link="torso_link", joint_names=JOINT_NAMES,
        default_joint_pos=tuple(map(_initial, JOINT_NAMES)), kp=kp, kd=kd,
        armature=tuple(m[0] for m in motors), effort_limit=tuple(m[1] for m in motors), velocity_limit=tuple(m[2] for m in motors),
        feet=tuple(FootCfg(side, f"{side}_ankle_roll_link", tuple(f"{side}_foot{i}_collision" for i in range(1, 8)),
            f"{side}_foot_contact", f"{side}_foot_height_scan", offsets)
            for side in ("left", "right")),
        held_joint_targets={name: 0. for name in HAND_JOINTS},
    )
    cfg = VelocityTaskCfg(robot=profile, name="g1_rough_velocity", task_version="g1_rough_velocity_v1",
        project_path=project_path or Path(__file__).parent, scene_path="res://g1_scene.jscn",
        base_clearance=.76, max_init_terrain_level=1, curriculum_mode="command_progress",
        fall_angle=math.radians(70), nconmax=100,
        action_scale={name: .25 * m[1] / gain for name, m, gain in zip(JOINT_NAMES, motors, kp, strict=True)})
    cfg.mujoco_solver_settings = {"solver": "Newton", "integrator": "ImplicitFast", "cone": "Elliptic", "iterations": 10,
        "line_search_iterations": 20, "convex_collision_iterations": 500, "impedance_ratio": 10.}
    cfg.illegal_contact.terminate_on_thigh = False
    cfg.illegal_contact.thigh_shape_patterns = ()
    cfg.illegal_contact.shank_shape_patterns = ()
    cfg.illegal_contact.trunk_head_shape_patterns = ()
    cfg.rewards.shank_collision = cfg.rewards.trunk_head_collision = 0.
    cfg.rewards.body_ang_vel = -.05
    cfg.rewards.angular_momentum = -.02
    cfg.rewards.self_collisions = -1.
    walking = (.3, .15, .15, .35, .25, .1, .2, .08, .1, .15, .15, .1, .15, .3)
    running = (.5, .2, .2, .6, .35, .15, .3, .08, .2, .5, .2, .15, .35, .3)
    parts = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll", "waist_yaw", "waist_roll", "waist_pitch",
             "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist")
    cfg.rewards.pose_std_standing = {".*": .05}
    cfg.rewards.pose_std_walking = {f".*{part}.*": value for part, value in zip(parts, walking, strict=True)}
    cfg.rewards.pose_std_running = {f".*{part}.*": value for part, value in zip(parts, running, strict=True)}
    if play:
        cfg.episode_length_s = 1_000_000_000.
        cfg.observations.actor_noise = False
        cfg.domain_randomization.enabled = False
        cfg.push_enabled = False
        cfg.terrain_curriculum = False
    return cfg
