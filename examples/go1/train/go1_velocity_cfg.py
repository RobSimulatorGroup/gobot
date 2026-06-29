"""Configuration for the Go1 velocity training example."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

try:
    from ._repo_imports import prefer_repo_gobot
except ImportError:
    from _repo_imports import prefer_repo_gobot

prefer_repo_gobot()

from gobot.rl.locomotion import UniformVelocityCommandCfg, UniformVelocityCommandRanges, VelocityCommandStage

GO1_JOINT_NAMES: tuple[str, ...] = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)

GO1_DEFAULT_JOINT_POS = np.array(
    [
        0.0,
        0.9,
        -1.8,
        0.0,
        0.9,
        -1.8,
        0.0,
        1.0,
        -1.8,
        0.0,
        1.0,
        -1.8,
    ],
    dtype=np.float32,
)

GO1_FOOT_NAMES: tuple[str, ...] = ("FR", "FL", "RR", "RL")
GO1_UNILAB_FOOT_NAMES: tuple[str, ...] = ("FL", "FR", "RL", "RR")
GO1_UNILAB_FOOT_LINK_NAMES: tuple[str, ...] = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
GO1_ACTION_SCALE: Mapping[str, float] = {
    r".*_(hip|thigh)_joint": 0.3727530386870487,
    r".*_calf_joint": 0.24850202579136574,
}
GO1_UNILAB_ROUGH_ACTION_SCALE: Mapping[str, float] = {
    r".*_hip_joint": 0.125,
    "__default__": 0.25,
}

GO1_UNILAB_MUJOCO_SOLVER_SETTINGS: Mapping[str, Any] = {
    "cone": 1,  # mjCONE_ELLIPTIC
    "convex_collision_iterations": 500,
    "impedance_ratio": 100.0,
}


@dataclass
class UnilabNoiseCfg:
    level: float = 0.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 0.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1


@dataclass
class VelocityObservationCfg:
    height_scan_sensor: str | None = "terrain_scan"
    terrain_scan_max_distance: float = 5.0
    actor_noise: bool = True
    unilab_noise: UnilabNoiseCfg = field(default_factory=UnilabNoiseCfg)
    actor_noise_ranges: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "base_lin_vel": (-0.5, 0.5),
            "base_ang_vel": (-0.2, 0.2),
            "projected_gravity": (-0.05, 0.05),
            "joint_pos": (-0.01, 0.01),
            "joint_vel": (-1.5, 1.5),
            "height_scan": (-0.1, 0.1),
        }
    )


@dataclass
class VelocityRewardCfg:
    track_linear_velocity: float = 2.0
    track_angular_velocity: float = 2.0
    upright: float = 1.0
    action_rate_l2: float = -0.1
    air_time: float = 0.0
    foot_clearance: float = -2.0
    foot_slip: float = -0.1
    lin_vel_std: float = math.sqrt(0.25)
    ang_vel_std: float = math.sqrt(0.5)
    upright_std: float = math.sqrt(0.2)
    command_threshold: float = 0.05


@dataclass
class UnilabRewardCfg:
    scales: Mapping[str, float] = field(default_factory=dict)
    tracking_sigma: float = 0.25
    base_height_target: float = 0.3
    stand_still_command_threshold: float = 0.1
    joint_pos_penalty_stand_still_scale: float = 5.0
    joint_pos_penalty_velocity_threshold: float = 0.5
    joint_pos_penalty_command_threshold: float = 0.1
    contact_threshold: float = 1.0
    contact_forces_threshold: float = 100.0
    feet_air_time_threshold: float = 0.5
    feet_height_body_target: float = -0.2
    feet_height_body_tanh_mult: float = 2.0
    feet_gait_std: float = math.sqrt(0.5)
    feet_gait_max_err: float = 0.2
    feet_gait_velocity_threshold: float = 0.5
    feet_gait_command_threshold: float = 0.1


@dataclass
class VelocityIllegalContactCfg:
    """Best-effort collision classification from runtime contact link names."""

    enabled: bool = True
    terminate_on_thigh: bool = True
    ground_force_threshold: float = 1.0e-5
    self_collision_force_threshold: float = 1.0e-5
    thigh_link_patterns: tuple[str, ...] = (r".*_thigh",)
    shank_link_patterns: tuple[str, ...] = (r".*_calf",)
    trunk_head_link_patterns: tuple[str, ...] = (r"trunk", r".*head.*")


@dataclass
class VelocityDomainRandomizationCfg:
    """Domain randomization parameters for the Gobot velocity task."""

    enabled: bool = True
    randomize_base_mass: bool = True
    added_mass_range: tuple[float, float] = (-1.5, 1.5)
    random_com: bool = True
    com_offset_x: tuple[float, float] = (-0.05, 0.05)
    com_offset_y: tuple[float, float] | None = None
    com_offset_z: tuple[float, float] | None = None
    randomize_kp: bool = False
    kp_multiplier_range: tuple[float, float] = (0.9, 1.1)
    randomize_kd: bool = False
    kd_multiplier_range: tuple[float, float] = (0.9, 1.1)
    encoder_bias_range: tuple[float, float] = (-0.015, 0.015)
    reset_lin_vel_ranges: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.05, 0.05),
        }
    )
    reset_ang_vel_ranges: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.05, 0.05),
        }
    )
    foot_friction_range: tuple[float, float] = (0.3, 1.5)
    base_com_offset_ranges: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.025, 0.025),
            "y": (-0.025, 0.025),
            "z": (-0.03, 0.03),
        }
    )


def _default_push_velocity_ranges() -> Mapping[str, tuple[float, float]]:
    return {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.4, 0.4),
        "roll": (-0.52, 0.52),
        "pitch": (-0.52, 0.52),
        "yaw": (-0.78, 0.78),
    }


def _default_push_force_ranges() -> Mapping[str, tuple[float, float]]:
    return {
        "x": (-1.0, 1.0),
        "y": (-1.0, 1.0),
        "z": (-0.5, 0.5),
    }


@dataclass
class Go1VelocityCfg:
    """Go1 velocity task config for this example project."""

    name: str = "gobot_go1_unilab_rough"
    terrain_type: str = "rough"
    task_profile: Literal["gobot_velocity", "unilab_flat", "unilab_rough"] = "unilab_rough"
    project_path: str | Path = "examples/go1"
    scene_path: str = "res://go1_scene.jscn"
    terrain_scene_path: str = "terrain_scene.jscn"
    robot_name: str = "go1"
    base_link: str = "trunk"
    joint_names: tuple[str, ...] = GO1_JOINT_NAMES
    default_joint_pos: tuple[float, ...] = tuple(float(x) for x in GO1_DEFAULT_JOINT_POS)
    foot_names: tuple[str, ...] = GO1_UNILAB_FOOT_NAMES
    foot_link_names: tuple[str, ...] = GO1_UNILAB_FOOT_LINK_NAMES
    action_scale: float | Mapping[str, float] = field(default_factory=lambda: dict(GO1_UNILAB_ROUGH_ACTION_SCALE))
    action_clip: float = 100.0
    kp: float = 35.0
    kd: float = 0.5
    physics_dt: float = 0.01
    decimation: int = 2
    mujoco_solver_settings: Mapping[str, Any] = field(default_factory=lambda: dict(GO1_UNILAB_MUJOCO_SOLVER_SETTINGS))
    episode_length_s: float = 20.0
    base_clearance: float = 0.45
    min_base_clearance: float = 0.0
    spawn_jitter: float = 0.5
    terrain_curriculum: bool = False
    terrain_curriculum_steps: int = 21600
    spawn_difficulty_radius: float = 0.85
    terrain_out_of_bounds: bool = True
    terrain_distance_buffer: float = 3.0
    randomize_rough_reset_pose: bool = True
    use_unilab_reset_rng: bool = True
    illegal_contact: VelocityIllegalContactCfg = field(default_factory=VelocityIllegalContactCfg)
    domain_randomization: VelocityDomainRandomizationCfg = field(
        default_factory=lambda: VelocityDomainRandomizationCfg(enabled=True)
    )
    push_enabled: bool = True
    push_interval_steps: int = 750
    push_interval_mode: Literal["per_env_random", "global"] = "per_env_random"
    push_force_ranges: Mapping[str, tuple[float, float]] = field(default_factory=_default_push_force_ranges)
    push_interval_range_s: tuple[float, float] = (1.0, 3.0)
    push_velocity_ranges: Mapping[str, tuple[float, float]] = field(default_factory=_default_push_velocity_ranges)
    observations: VelocityObservationCfg = field(default_factory=VelocityObservationCfg)
    command: UniformVelocityCommandCfg = field(
        default_factory=lambda: UniformVelocityCommandCfg(
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.1,
            rel_heading_envs=1.0,
            rel_world_envs=0.0,
            rel_forward_envs=0.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            zero_small_xy_threshold=0.08,
            ranges=UniformVelocityCommandRanges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0),
                heading=(-math.pi, math.pi),
            ),
        )
    )
    command_curriculum: tuple[VelocityCommandStage, ...] = (
        VelocityCommandStage(step=0),
    )
    rewards: VelocityRewardCfg = field(default_factory=VelocityRewardCfg)
    unilab_rewards: UnilabRewardCfg = field(default_factory=lambda: _unilab_rough_reward_cfg())


def _unilab_flat_reward_cfg() -> UnilabRewardCfg:
    return UnilabRewardCfg(
        scales={
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -5.0,
            "ang_vel_xy": -0.1,
            "base_height": -100.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "contact": 0.24,
            "swing_feet_z": 4.0,
        },
        tracking_sigma=0.25,
        base_height_target=0.3,
    )


def _unilab_rough_reward_cfg() -> UnilabRewardCfg:
    return UnilabRewardCfg(
        scales={
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "joint_torques_l2": -2.5e-5,
            "joint_acc_l2": -2.5e-7,
            "joint_power": -2.0e-5,
            "stand_still": -2.0,
            "hip_pos": -0.5,
            "joint_pos_penalty": -1.0,
            "joint_mirror": -0.05,
            "action_rate": -0.01,
            "undesired_contacts": -1.0,
            "contact_forces": -1.5e-4,
            "tracking_lin_vel": 3.0,
            "tracking_ang_vel": 1.5,
            "feet_air_time": 0.5,
            "feet_air_time_variance": -1.0,
            "feet_contact_without_cmd": 0.1,
            "feet_slide": -0.1,
            "feet_height_body": -5.0,
            "feet_gait": 0.5,
            "upward": 1.0,
        },
        tracking_sigma=0.25,
        base_height_target=0.33,
        contact_threshold=1.0,
        contact_forces_threshold=100.0,
        feet_air_time_threshold=0.5,
        feet_height_body_target=-0.2,
        feet_height_body_tanh_mult=2.0,
        feet_gait_std=math.sqrt(0.5),
        feet_gait_max_err=0.2,
        feet_gait_velocity_threshold=0.5,
        feet_gait_command_threshold=0.1,
    )


def go1_rough_velocity_cfg(
    *,
    project_path: str | Path = "examples/go1",
    play: bool = False,
) -> Go1VelocityCfg:
    cfg = Go1VelocityCfg(project_path=project_path)
    cfg.name = "gobot_go1_unilab_rough"
    cfg.task_profile = "unilab_rough"
    cfg.foot_names = GO1_UNILAB_FOOT_NAMES
    cfg.foot_link_names = GO1_UNILAB_FOOT_LINK_NAMES
    cfg.action_scale = dict(GO1_UNILAB_ROUGH_ACTION_SCALE)
    cfg.action_clip = 100.0
    cfg.kp = 35.0
    cfg.kd = 0.5
    cfg.physics_dt = 0.005
    cfg.decimation = 4
    cfg.mujoco_solver_settings = dict(GO1_UNILAB_MUJOCO_SOLVER_SETTINGS)
    # UniLab's home qpos z is 0.27, but get_base_pos() observes the trunk body
    # 5 cm higher. Gobot resets the authored base body pose directly.
    cfg.base_clearance = 0.32
    cfg.min_base_clearance = 0.0
    cfg.terrain_curriculum = False
    cfg.terrain_out_of_bounds = True
    cfg.illegal_contact.ground_force_threshold = 1.0
    cfg.domain_randomization.enabled = True
    cfg.domain_randomization.randomize_base_mass = True
    cfg.domain_randomization.added_mass_range = (-1.0, 3.0)
    cfg.domain_randomization.random_com = True
    cfg.domain_randomization.randomize_kp = True
    cfg.domain_randomization.kp_multiplier_range = (0.9, 1.1)
    cfg.domain_randomization.randomize_kd = True
    cfg.domain_randomization.kd_multiplier_range = (0.9, 1.1)
    cfg.domain_randomization.encoder_bias_range = (0.0, 0.0)
    cfg.push_enabled = True
    cfg.push_interval_steps = 625
    cfg.push_interval_mode = "global"
    cfg.command = UniformVelocityCommandCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        rel_world_envs=0.0,
        rel_forward_envs=0.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        zero_small_xy_threshold=0.08,
        ranges=UniformVelocityCommandRanges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )
    cfg.command_curriculum = (VelocityCommandStage(step=0),)
    cfg.unilab_rewards = _unilab_rough_reward_cfg()
    if play:
        cfg.episode_length_s = float(1_000_000_000)
        cfg.terrain_curriculum = False
        cfg.randomize_rough_reset_pose = False
        cfg.observations.actor_noise = False
        cfg.domain_randomization.enabled = False
        cfg.push_enabled = False
    return cfg


def go1_flat_velocity_cfg(
    *,
    project_path: str | Path = "examples/go1",
    play: bool = False,
) -> Go1VelocityCfg:
    cfg = go1_rough_velocity_cfg(project_path=project_path, play=play)
    cfg.name = "gobot_go1_unilab_flat"
    cfg.terrain_type = "flat"
    cfg.task_profile = "unilab_flat"
    cfg.scene_path = "res://go1_flat_scene.jscn"
    cfg.terrain_scene_path = "flat_terrain_scene.jscn"
    cfg.action_scale = 0.25
    cfg.action_clip = 1.0
    cfg.kp = 35.0
    cfg.kd = 0.5
    cfg.physics_dt = 0.01
    cfg.decimation = 2
    cfg.base_clearance = 0.45
    cfg.min_base_clearance = 0.0
    cfg.randomize_rough_reset_pose = False
    cfg.observations.height_scan_sensor = None
    cfg.terrain_curriculum = False
    cfg.terrain_out_of_bounds = False
    cfg.illegal_contact.enabled = False
    cfg.domain_randomization.enabled = True
    cfg.domain_randomization.randomize_base_mass = True
    cfg.domain_randomization.random_com = True
    cfg.domain_randomization.randomize_kp = False
    cfg.domain_randomization.randomize_kd = False
    cfg.push_enabled = True
    cfg.push_interval_steps = 750
    cfg.command = UniformVelocityCommandCfg(
        resampling_time_range=(1_000_000_000.0, 1_000_000_000.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_world_envs=0.0,
        rel_forward_envs=0.0,
        heading_command=False,
        ranges=UniformVelocityCommandRanges(
            lin_vel_x=(-0.6, 1.0),
            lin_vel_y=(-0.4, 0.4),
            ang_vel_z=(-0.8, 0.8),
            heading=None,
        ),
    )
    cfg.command_curriculum = (VelocityCommandStage(step=0),)
    cfg.unilab_rewards = _unilab_flat_reward_cfg()
    if play:
        cfg.episode_length_s = float(1_000_000_000)
        cfg.terrain_curriculum = False
        cfg.observations.actor_noise = False
        cfg.domain_randomization.enabled = False
        cfg.push_enabled = False
    return cfg


def go1_velocity_cfg(name: str, *, project_path: str | Path | None = None, play: bool = False) -> Go1VelocityCfg:
    factories = {
        "go1_rough": go1_rough_velocity_cfg,
        "go1_unilab_rough": go1_rough_velocity_cfg,
        "gobot_go1_unilab_rough": go1_rough_velocity_cfg,
        "go1_velocity": go1_rough_velocity_cfg,
        "gobot_go1_velocity": go1_rough_velocity_cfg,
        "go1_flat": go1_flat_velocity_cfg,
        "go1_unilab_flat": go1_flat_velocity_cfg,
        "gobot_go1_unilab_flat": go1_flat_velocity_cfg,
        "gobot_go1_flat_velocity": go1_flat_velocity_cfg,
    }
    key = name.lower()
    if key not in factories:
        raise ValueError(f"unknown Go1 velocity task {name!r}; expected one of {sorted(factories)}")
    kwargs: dict[str, Any] = {"play": play}
    if project_path is not None:
        kwargs["project_path"] = project_path
    return factories[key](**kwargs)


def rsl_rl_train_cfg(
    *,
    experiment_name: str = "go1_velocity",
    max_iterations: int = 10_000,
    save_interval: int = 50,
    obs_normalization: bool = False,
    clip_actions: float | None = 1.0,
) -> dict[str, Any]:
    return {
        "num_steps_per_env": 24,
        "save_interval": save_interval,
        "max_iterations": max_iterations,
        "obs_groups": {
            "actor": ["actor"],
            "critic": ["critic"],
        },
        "experiment_name": experiment_name,
        "clip_actions": clip_actions,
        "actor": {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": obs_normalization,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": obs_normalization,
        },
        "algorithm": {
            "class_name": "gobot.rl.rsl_rl.FinalObservationAwarePPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "learning_rate": 1.0e-3,
            "max_grad_norm": 1.0,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


__all__ = [
    "GO1_DEFAULT_JOINT_POS",
    "GO1_FOOT_NAMES",
    "GO1_UNILAB_FOOT_NAMES",
    "GO1_JOINT_NAMES",
    "GO1_ACTION_SCALE",
    "GO1_UNILAB_ROUGH_ACTION_SCALE",
    "Go1VelocityCfg",
    "UnilabNoiseCfg",
    "UnilabRewardCfg",
    "VelocityObservationCfg",
    "VelocityRewardCfg",
    "VelocityDomainRandomizationCfg",
    "VelocityIllegalContactCfg",
    "go1_flat_velocity_cfg",
    "go1_rough_velocity_cfg",
    "go1_velocity_cfg",
    "rsl_rl_train_cfg",
]
