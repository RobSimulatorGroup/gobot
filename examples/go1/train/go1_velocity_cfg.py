"""Configuration for the Go1 rough-terrain velocity task."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

from gobot.rl.locomotion import UniformVelocityCommandCfg, UniformVelocityCommandRanges, VelocityCommandStage

from ..go1_profile import (
    GO1_ARMATURE,
    GO1_DEFAULT_JOINT_POS,
    GO1_EFFORT_LIMIT,
    GO1_FOOT_LINK_NAMES,
    GO1_FOOT_NAMES,
    GO1_JOINT_NAMES,
    GO1_KD,
    GO1_KP,
    GO1_VELOCITY_LIMIT,
)
from ..go1_velocity_contract import GO1_TASK_NAME, GO1_TASK_VERSION


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


@dataclass
class VelocityObservationCfg:
    height_scan_sensor: str = "terrain_scan"
    terrain_scan_max_distance: float = 5.0
    actor_noise: bool = True
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
    pose: float = 1.0
    body_ang_vel: float = 0.0
    angular_momentum: float = 0.0
    dof_pos_limits: float = -1.0
    action_rate_l2: float = -0.1
    air_time: float = 0.0
    foot_clearance: float = -2.0
    foot_swing_height: float = -0.25
    foot_slip: float = -0.1
    soft_landing: float = -1.0e-5
    self_collisions: float = -0.1
    shank_collision: float = -0.1
    trunk_head_collision: float = -0.1
    lin_vel_std: float = math.sqrt(0.25)
    ang_vel_std: float = math.sqrt(0.5)
    upright_std: float = math.sqrt(0.2)
    command_threshold: float = 0.05
    foot_clearance_target_height: float = 0.1
    soft_joint_pos_limit_factor: float = 0.9
    pose_walking_threshold: float = 0.05
    pose_running_threshold: float = 1.5
    pose_std_standing_hip_thigh: float = 0.05
    pose_std_standing_calf: float = 0.1
    pose_std_walking_hip_thigh: float = 0.3
    pose_std_walking_calf: float = 0.6
    pose_std_running_hip_thigh: float = 0.3
    pose_std_running_calf: float = 0.6


@dataclass
class VelocityIllegalContactCfg:
    enabled: bool = True
    terminate_on_thigh: bool = True
    ground_force_threshold: float = 10.0
    self_collision_force_threshold: float = 10.0
    thigh_shape_patterns: tuple[str, ...] = (r".*_thigh_collision[123]",)
    shank_shape_patterns: tuple[str, ...] = (r".*_calf_collision[12]",)
    trunk_head_shape_patterns: tuple[str, ...] = (r"trunk_collision", r"head_collision")


@dataclass
class VelocityDomainRandomizationCfg:
    enabled: bool = True
    randomize_base_mass: bool = False
    added_mass_range: tuple[float, float] = (-1.5, 1.5)
    random_com: bool = True
    com_offset_x: tuple[float, float] = (-0.025, 0.025)
    com_offset_y: tuple[float, float] | None = (-0.025, 0.025)
    com_offset_z: tuple[float, float] | None = (-0.03, 0.03)
    randomize_kp: bool = False
    kp_multiplier_range: tuple[float, float] = (0.9, 1.1)
    randomize_kd: bool = False
    kd_multiplier_range: tuple[float, float] = (0.9, 1.1)
    encoder_bias_range: tuple[float, float] = (-0.015, 0.015)
    randomize_foot_friction: bool = True
    foot_friction_slide_range: tuple[float, float] = (0.3, 1.5)
    foot_friction_spin_range: tuple[float, float] = (1.0e-4, 2.0e-2)
    foot_friction_roll_range: tuple[float, float] = (1.0e-5, 5.0e-3)


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
    """Single source of truth for Go1 rough-terrain training and playback."""

    name: str = GO1_TASK_NAME
    project_path: str | Path = "examples/go1"
    scene_path: str = "res://go1_scene.jscn"
    robot_name: str = "go1"
    base_link: str = "trunk"
    joint_names: tuple[str, ...] = GO1_JOINT_NAMES
    default_joint_pos: tuple[float, ...] = tuple(float(x) for x in GO1_DEFAULT_JOINT_POS)
    foot_names: tuple[str, ...] = GO1_FOOT_NAMES
    foot_link_names: tuple[str, ...] = GO1_FOOT_LINK_NAMES
    action_scale: float | Mapping[str, float] = field(default_factory=lambda: dict(GO1_ACTION_SCALE))
    simulate_action_latency: bool = False
    action_clip: float | None = None
    kp: float | tuple[float, ...] = tuple(float(x) for x in GO1_KP)
    kd: float | tuple[float, ...] = tuple(float(x) for x in GO1_KD)
    physics_dt: float = GO1_PHYSICS_DT
    decimation: int = GO1_DECIMATION
    mujoco_solver_settings: Mapping[str, Any] = field(default_factory=lambda: dict(GO1_MUJOCO_SOLVER_SETTINGS))
    episode_length_s: float = 20.0
    base_clearance: float = 0.278
    spawn_jitter: float = 0.5
    reset_z_range: tuple[float, float] = (0.01, 0.05)
    randomize_reset_yaw: bool = True
    # Controls spawn-level progression over the authored terrain grid. Geometry
    # is owned by the TerrainGeneratorConfig referenced from go1_scene.jscn.
    terrain_curriculum: bool = True
    max_init_terrain_level: int | None = 5
    terrain_out_of_bounds: bool = True
    terrain_distance_buffer: float = 0.3
    illegal_contact: VelocityIllegalContactCfg = field(default_factory=VelocityIllegalContactCfg)
    domain_randomization: VelocityDomainRandomizationCfg = field(default_factory=VelocityDomainRandomizationCfg)
    push_enabled: bool = True
    push_interval_steps: int = 750
    push_interval_mode: Literal["per_env_random", "global"] = "per_env_random"
    push_mode: Literal["force", "velocity"] = "velocity"
    push_force_ranges: Mapping[str, tuple[float, float]] = field(default_factory=_default_push_force_ranges)
    push_interval_range_s: tuple[float, float] = (1.0, 3.0)
    push_velocity_ranges: Mapping[str, tuple[float, float]] = field(default_factory=_default_push_velocity_ranges)
    observations: VelocityObservationCfg = field(default_factory=VelocityObservationCfg)
    command: UniformVelocityCommandCfg = field(
        default_factory=lambda: UniformVelocityCommandCfg(
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.1,
            rel_heading_envs=0.3,
            rel_world_envs=0.0,
            rel_forward_envs=0.2,
            heading_command=True,
            heading_control_stiffness=0.5,
            zero_small_xy_threshold=0.0,
            ranges=UniformVelocityCommandRanges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-0.5, 0.5),
                heading=(-math.pi, math.pi),
            ),
        )
    )
    command_curriculum: tuple[VelocityCommandStage, ...] = (
        VelocityCommandStage(step=0),
        VelocityCommandStage(step=120_000, lin_vel_x=(-1.5, 2.0), ang_vel_z=(-0.7, 0.7)),
        VelocityCommandStage(step=240_000, lin_vel_x=(-2.0, 3.0)),
    )
    rewards: VelocityRewardCfg = field(default_factory=VelocityRewardCfg)


def go1_velocity_cfg(
    *,
    project_path: str | Path = "examples/go1",
    play: bool = False,
) -> Go1VelocityCfg:
    cfg = Go1VelocityCfg(project_path=project_path)
    if play:
        cfg.episode_length_s = float(1_000_000_000)
        cfg.terrain_curriculum = False
        cfg.observations.actor_noise = False
        cfg.domain_randomization.enabled = False
        cfg.push_enabled = False
    return cfg


def rsl_rl_train_cfg(
    *,
    experiment_name: str = GO1_TASK_NAME,
    max_iterations: int = 10_000,
    save_interval: int = 50,
    obs_normalization: bool = False,
    clip_actions: float | None = None,
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
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "use_clipped_value_loss": True,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "learning_rate": 1.0e-3,
            "max_grad_norm": 1.0,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


__all__ = [
    "GO1_ACTION_SCALE",
    "GO1_ARMATURE",
    "GO1_DEFAULT_JOINT_POS",
    "GO1_EFFORT_LIMIT",
    "GO1_FOOT_LINK_NAMES",
    "GO1_FOOT_NAMES",
    "GO1_JOINT_NAMES",
    "GO1_KD",
    "GO1_KP",
    "GO1_MUJOCO_SOLVER_SETTINGS",
    "GO1_TASK_NAME",
    "GO1_TASK_VERSION",
    "GO1_VELOCITY_LIMIT",
    "Go1VelocityCfg",
    "VelocityDomainRandomizationCfg",
    "VelocityIllegalContactCfg",
    "VelocityObservationCfg",
    "VelocityRewardCfg",
    "go1_velocity_cfg",
    "rsl_rl_train_cfg",
]
