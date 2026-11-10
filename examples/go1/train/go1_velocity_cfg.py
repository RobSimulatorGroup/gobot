"""Configuration for the Go1 rough-terrain velocity task."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

from gobot.rl.locomotion import VelocityCommandStage
from gobot.rl.locomotion.config import (
    FootCfg,
    LocomotionRobotCfg,
    NaturalGaitCfg,
    VelocityDomainRandomizationCfg,
    VelocityIllegalContactCfg,
    VelocityRewardCfg,
    VelocityTaskCfg,
)

from ..go1_profile import (
    GO1_ARMATURE,
    GO1_EFFORT_LIMIT,
    GO1_VELOCITY_LIMIT,
    GO1_DEFAULT_JOINT_POS,
    GO1_FOOT_LINK_NAMES,
    GO1_FOOT_NAMES,
    GO1_JOINT_NAMES,
    GO1_KD,
    GO1_KP,
)
from ..go1_policy_info import GO1_TASK_NAME


GO1_PHYSICS_DT = 0.005
GO1_DECIMATION = 4

GO1_MUJOCO_SOLVER_SETTINGS: Mapping[str, Any] = MappingProxyType(
    {
        "solver": "Newton",
        "integrator": "ImplicitFast",
        "cone": "Elliptic",
        "jacobian": "Auto",
        "iterations": 10,
        "line_search_iterations": 20,
        "convex_collision_iterations": 500,
        "tolerance": 1.0e-8,
        "line_search_tolerance": 0.01,
        "convex_collision_tolerance": 1.0e-6,
        "impedance_ratio": 10.0,
    }
)

GO1_ACTION_SCALE: Mapping[str, float] = {
    r".*_(hip|thigh)_joint": 0.3727530386870487,
    r".*_calf_joint": 0.24850202579136574,
}


def go1_gait_cfg():
    """Walk, trot and gallop phases in this profile's FR, FL, RR, RL order."""
    return NaturalGaitCfg(
        enabled=True,
        weight=0.5,
        phase_offsets=((0., .5, .75, .25), (0., .5, .5, 0.), (.6, .5, .1, 0.)),
    )


def _go1_rewards():
    return VelocityRewardCfg(
        shank_collision=-0.1,
        trunk_head_collision=-0.1,
        pose_std_standing={r".*_(hip|thigh)_joint": 0.05, r".*_calf_joint": 0.1},
        pose_std_walking={r".*_(hip|thigh)_joint": 0.3, r".*_calf_joint": 0.6},
        pose_std_running={r".*_(hip|thigh)_joint": 0.3, r".*_calf_joint": 0.6},
    )


def _go1_illegal_contacts():
    return VelocityIllegalContactCfg(
        terminate_on_thigh=True,
        thigh_shape_patterns=(r".*_thigh_collision[123]",),
        shank_shape_patterns=(r".*_calf_collision[12]",),
        trunk_head_shape_patterns=(r"trunk_collision", r"head_collision"),
    )


def go1_velocity_cfg(
    *,
    project_path: str | Path = "examples/go1",
    play: bool = False,
) -> VelocityTaskCfg:
    offsets = ((0., 0., 0.), (0.04, 0., 0.), (0., 0.04, 0.), (-0.04, 0., 0.), (0., -0.04, 0.))
    robot = LocomotionRobotCfg(
        name="go1", base_link="trunk", torso_link="trunk",
        joint_names=GO1_JOINT_NAMES,
        default_joint_pos=tuple(float(x) for x in GO1_DEFAULT_JOINT_POS),
        kp=tuple(float(x) for x in GO1_KP), kd=tuple(float(x) for x in GO1_KD),
        armature=GO1_ARMATURE, effort_limit=GO1_EFFORT_LIMIT, velocity_limit=GO1_VELOCITY_LIMIT,
        feet=tuple(FootCfg(name, link, (f"{name}_foot_collision",),
                          f"{name}_foot_contact", f"{name}_foot_height_scan", offsets)
                   for name, link in zip(GO1_FOOT_NAMES, GO1_FOOT_LINK_NAMES, strict=True)),
    )
    cfg = VelocityTaskCfg(
        robot=robot, name=GO1_TASK_NAME, project_path=project_path,
        scene_path="res://go1_scene.jscn", action_scale=dict(GO1_ACTION_SCALE),
        physics_dt=GO1_PHYSICS_DT, decimation=GO1_DECIMATION,
        mujoco_solver_settings=dict(GO1_MUJOCO_SOLVER_SETTINGS),
        rewards=_go1_rewards(), illegal_contact=_go1_illegal_contacts(),
        domain_randomization=VelocityDomainRandomizationCfg(
            startup_order=("encoder_bias", "body_com", "foot_friction")),
    )
    if play:
        cfg.episode_length_s = float(1_000_000_000)
        cfg.terrain_curriculum = False
        cfg.observations.actor_noise = False
        cfg.domain_randomization.enabled = False
        cfg.push_enabled = False
    return cfg


def apply_training_profile(
    cfg: VelocityTaskCfg,
    profile: Literal["balanced", "run"],
) -> VelocityTaskCfg:
    if profile not in {"balanced", "run"}:
        raise ValueError(f"unsupported Go1 training profile {profile!r}")
    cfg.training_profile = profile
    if profile == "run":
        cfg.command.rel_forward_envs = max(cfg.command.rel_forward_envs, 0.3)
        cfg.command.rel_run_envs = 0.6
        cfg.command.run_velocity_x = (1.5, 2.5)
        cfg.command_curriculum = cfg.command_curriculum[:2]
        cfg.run_command_curriculum = (
            VelocityCommandStage(step=0, run_velocity_x=(1.5, 2.5)),
            VelocityCommandStage(step=100 * 24, run_velocity_x=(2.0, 3.0)),
            VelocityCommandStage(step=250 * 24, run_velocity_x=(2.5, 3.5)),
        )
        cfg.gait_foot_indices = (0, 1, 2, 3)
        cfg.gait_joint_indices = tuple(tuple(range(i, i + 3)) for i in range(0, 12, 3))
        cfg.rewards.track_linear_velocity = 4.0
        cfg.rewards.run_progress = 4.0
        cfg.rewards.bound_gait = 0.0
    return cfg
