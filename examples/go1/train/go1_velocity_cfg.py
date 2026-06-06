"""Configuration for the Go1 velocity training example."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gobot.rl.locomotion import UniformVelocityCommandCfg, VelocityCommandStage

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
        0.9,
        -1.8,
        0.0,
        0.9,
        -1.8,
    ],
    dtype=np.float32,
)

GO1_FOOT_NAMES: tuple[str, ...] = ("FR", "FL", "RR", "RL")


@dataclass
class VelocityObservationCfg:
    height_scan_sensor: str | None = "terrain_scan"
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
    foot_target_height: float = 0.1
    command_threshold: float = 0.05
    pose_walking_threshold: float = 0.05
    pose_running_threshold: float = 1.5
    pose_std_standing: Mapping[str, float] = field(
        default_factory=lambda: {
            r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
            r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
        }
    )
    pose_std_walking: Mapping[str, float] = field(
        default_factory=lambda: {
            r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
            r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
        }
    )
    pose_std_running: Mapping[str, float] = field(
        default_factory=lambda: {
            r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
            r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
        }
    )


@dataclass
class VelocityTerrainNormalUprightCfg:
    """Use the local terrain plane instead of world-up for the upright reward."""

    enabled: bool = True
    min_hit_count: int = 3


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
    encoder_bias_range: tuple[float, float] = (-0.01, 0.01)
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
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (-0.25, 0.25),
    }


@dataclass
class Go1VelocityCfg:
    """Go1 velocity task config for this example project."""

    name: str = "gobot_go1_velocity"
    terrain_type: str = "rough"
    project_path: str | Path = "examples/go1"
    scene_path: str = "res://go1_scene.jscn"
    terrain_scene_path: str = "terrain_scene.jscn"
    robot_name: str = "go1"
    base_link: str = "trunk"
    joint_names: tuple[str, ...] = GO1_JOINT_NAMES
    default_joint_pos: tuple[float, ...] = tuple(float(x) for x in GO1_DEFAULT_JOINT_POS)
    foot_names: tuple[str, ...] = GO1_FOOT_NAMES
    foot_link_names: tuple[str, ...] = ("FR_calf", "FL_calf", "RR_calf", "RL_calf")
    action_scale: float | Mapping[str, float] = 0.35
    kp: float = 40.0
    kd: float = 1.0
    physics_dt: float = 0.002
    decimation: int = 10
    episode_length_s: float = 20.0
    base_clearance: float = 0.32
    min_base_clearance: float = 0.16
    spawn_jitter: float = 0.35
    terrain_curriculum: bool = True
    terrain_curriculum_steps: int = 21600
    spawn_difficulty_radius: float = 0.85
    terrain_normal_upright: VelocityTerrainNormalUprightCfg = field(default_factory=VelocityTerrainNormalUprightCfg)
    contact_history_length: int = 4
    illegal_contact: VelocityIllegalContactCfg = field(default_factory=VelocityIllegalContactCfg)
    domain_randomization: VelocityDomainRandomizationCfg = field(default_factory=VelocityDomainRandomizationCfg)
    push_enabled: bool = True
    push_interval_range_s: tuple[float, float] = (2.0, 6.0)
    push_velocity_ranges: Mapping[str, tuple[float, float]] = field(default_factory=_default_push_velocity_ranges)
    observations: VelocityObservationCfg = field(default_factory=VelocityObservationCfg)
    command: UniformVelocityCommandCfg = field(default_factory=UniformVelocityCommandCfg)
    command_curriculum: tuple[VelocityCommandStage, ...] = (
        VelocityCommandStage(step=0, lin_vel_x=(-1.0, 1.0), ang_vel_z=(-0.5, 0.5)),
        VelocityCommandStage(step=5000 * 24, lin_vel_x=(-1.5, 2.0), ang_vel_z=(-0.7, 0.7)),
        VelocityCommandStage(step=10000 * 24, lin_vel_x=(-2.0, 3.0)),
    )
    rewards: VelocityRewardCfg = field(default_factory=VelocityRewardCfg)


def go1_rough_velocity_cfg(
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


def go1_flat_velocity_cfg(
    *,
    project_path: str | Path = "examples/go1",
    play: bool = False,
) -> Go1VelocityCfg:
    cfg = go1_rough_velocity_cfg(project_path=project_path, play=play)
    cfg.name = "gobot_go1_flat_velocity"
    cfg.terrain_type = "flat"
    cfg.observations.height_scan_sensor = None
    cfg.terrain_curriculum = False
    cfg.terrain_normal_upright.enabled = False
    cfg.illegal_contact.enabled = False
    cfg.rewards.upright = 1.0
    return cfg


def go1_velocity_cfg(name: str, *, project_path: str | Path | None = None, play: bool = False) -> Go1VelocityCfg:
    factories = {
        "go1_rough": go1_rough_velocity_cfg,
        "go1_velocity": go1_rough_velocity_cfg,
        "gobot_go1_velocity": go1_rough_velocity_cfg,
        "go1_flat": go1_flat_velocity_cfg,
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
            "class_name": "rsl_rl.algorithms.PPO",
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
    "GO1_JOINT_NAMES",
    "Go1VelocityCfg",
    "VelocityObservationCfg",
    "VelocityRewardCfg",
    "VelocityDomainRandomizationCfg",
    "VelocityIllegalContactCfg",
    "VelocityTerrainNormalUprightCfg",
    "go1_flat_velocity_cfg",
    "go1_rough_velocity_cfg",
    "go1_velocity_cfg",
    "rsl_rl_train_cfg",
]
