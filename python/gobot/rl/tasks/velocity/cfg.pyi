from pathlib import Path
from typing import Any, Mapping

import numpy as np

GO1_JOINT_NAMES: tuple[str, ...]
GO1_DEFAULT_JOINT_POS: np.ndarray
GO1_FOOT_NAMES: tuple[str, ...]

class UniformVelocityCommandRanges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None

class UniformVelocityCommandCfg:
    name: str
    resampling_time_range: tuple[float, float]
    rel_standing_envs: float
    rel_heading_envs: float
    rel_world_envs: float
    rel_forward_envs: float
    heading_command: bool
    heading_control_stiffness: float
    init_velocity_prob: float
    ranges: UniformVelocityCommandRanges

class VelocityStage:
    step: int
    lin_vel_x: tuple[float, float] | None
    lin_vel_y: tuple[float, float] | None
    ang_vel_z: tuple[float, float] | None

class VelocityObservationCfg:
    height_scan_sensor: str | None
    terrain_scan_max_distance: float
    actor_noise: bool
    actor_noise_ranges: Mapping[str, tuple[float, float]]

class VelocityRewardCfg:
    track_linear_velocity: float
    track_angular_velocity: float
    upright: float
    pose: float
    body_ang_vel: float
    angular_momentum: float
    dof_pos_limits: float
    action_rate_l2: float
    air_time: float
    foot_clearance: float
    foot_swing_height: float
    foot_slip: float
    soft_landing: float
    self_collisions: float
    shank_collision: float
    trunk_head_collision: float
    lin_vel_std: float
    ang_vel_std: float
    upright_std: float
    foot_target_height: float
    command_threshold: float
    pose_walking_threshold: float
    pose_running_threshold: float
    pose_std_standing: Mapping[str, float]
    pose_std_walking: Mapping[str, float]
    pose_std_running: Mapping[str, float]

class VelocityTaskCfg:
    name: str
    robot_family: str
    terrain_type: str
    project_path: str | Path
    scene_path: str
    terrain_scene_path: str
    robot_name: str
    base_link: str
    joint_names: tuple[str, ...]
    default_joint_pos: tuple[float, ...]
    foot_names: tuple[str, ...]
    foot_link_names: tuple[str, ...]
    action_scale: float | Mapping[str, float]
    kp: float
    kd: float
    physics_dt: float
    decimation: int
    episode_length_s: float
    base_clearance: float
    min_base_clearance: float
    spawn_jitter: float
    terrain_curriculum: bool
    terrain_curriculum_steps: int
    spawn_difficulty_radius: float
    observations: VelocityObservationCfg
    command: UniformVelocityCommandCfg
    command_curriculum: tuple[VelocityStage, ...]
    rewards: VelocityRewardCfg

def unitree_go1_rough_velocity_cfg(*, project_path: str | Path = ..., play: bool = ...) -> VelocityTaskCfg: ...
def unitree_go1_flat_velocity_cfg(*, project_path: str | Path = ..., play: bool = ...) -> VelocityTaskCfg: ...
def unitree_g1_rough_velocity_cfg(*, project_path: str | Path = ..., play: bool = ...) -> VelocityTaskCfg: ...
def unitree_g1_flat_velocity_cfg(*, project_path: str | Path = ..., play: bool = ...) -> VelocityTaskCfg: ...
def velocity_task_cfg(name: str, *, project_path: str | Path | None = ..., play: bool = ...) -> VelocityTaskCfg: ...
def rsl_rl_train_cfg(
    *,
    experiment_name: str = ...,
    max_iterations: int = ...,
    save_interval: int = ...,
    obs_normalization: bool = ...,
) -> dict[str, Any]: ...

__all__: list[str]
