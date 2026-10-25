"""Typed scene, robot and velocity-task configuration; no backend handles."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Literal, Mapping
from .command import UniformVelocityCommandCfg, UniformVelocityCommandRanges, VelocityCommandStage

@dataclass(frozen=True)
class FootCfg:
    name: str
    link: str
    collision_shapes: tuple[str, ...]
    contact_sensor: str
    height_sensor: str
    sample_offsets: tuple[tuple[float, float, float], ...]

@dataclass(frozen=True)
class LocomotionRobotCfg:
    name: str
    base_link: str
    torso_link: str
    joint_names: tuple[str, ...]
    default_joint_pos: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    armature: tuple[float, ...]
    effort_limit: tuple[float, ...]
    velocity_limit: tuple[float, ...]
    feet: tuple[FootCfg, ...]
    imu_sensor: str = "imu"
    angular_momentum_sensor: str = "root_angmom"
    held_joint_targets: Mapping[str, float] = field(default_factory=dict)
    held_joint_kp: float = 10.0
    held_joint_kd: float = 2.0

    def __post_init__(self):
        n = len(self.joint_names)
        if not n or len(set(self.joint_names)) != n:
            raise ValueError("policy joint names must be nonempty and unique")
        for name in ("default_joint_pos", "kp", "kd", "armature", "effort_limit", "velocity_limit"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must have one value per policy joint")
        if set(self.joint_names) & set(self.held_joint_targets):
            raise ValueError("policy joints and held joints must be disjoint")
        shapes = [shape for foot in self.feet for shape in foot.collision_shapes]
        if not self.feet or any(not foot.collision_shapes for foot in self.feet) or len(shapes) != len(set(shapes)):
            raise ValueError("each foot requires a disjoint, nonempty collision group")

VELOCITY_REWARD_TERM_NAMES: tuple[str, ...] = (
    "track_linear_velocity",
    "track_angular_velocity",
    "upright",
    "pose",
    "body_ang_vel",
    "angular_momentum",
    "dof_pos_limits",
    "action_rate_l2",
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "soft_landing",
    "self_collisions",
    "shank_collision",
    "trunk_head_collision",
)


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
    run_progress: float = 0.0
    bound_gait: float = 0.0
    bound_gait_motion_std: float = 1.0
    bound_gait_action_std: float = 0.5
    bound_gait_height_sync_std: float = 0.04
    bound_gait_height_separation_std: float = 0.05
    bound_gait_trot_penalty: float = 0.5
    lin_vel_std: float = math.sqrt(0.25)
    ang_vel_std: float = math.sqrt(0.5)
    upright_std: float = math.sqrt(0.2)
    command_threshold: float = 0.05
    foot_clearance_target_height: float = 0.1
    soft_joint_pos_limit_factor: float = 0.9
    pose_walking_threshold: float = 0.05
    pose_running_threshold: float = 1.5
    pose_std_standing: Mapping[str, float] = field(default_factory=dict)
    pose_std_walking: Mapping[str, float] = field(default_factory=dict)
    pose_std_running: Mapping[str, float] = field(default_factory=dict)
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
class VelocityTaskCfg:
    """Single source of truth for robot rough-terrain training and playback."""

    robot: LocomotionRobotCfg | None = None
    task_version: str = "velocity_v1"
    post_step_sensing: bool = True
    terrain_node_path: str = "terrain_world/terrain"
    curriculum_mode: Literal["legacy", "command_progress"] = "legacy"
    fall_angle: float | None = None
    nconmax: int = 35
    njmax: int = 1500
    gait_foot_indices: tuple[int, ...] = ()
    gait_joint_indices: tuple[tuple[int, ...], ...] = ()
    name: str = "velocity"
    training_profile: Literal["balanced", "run"] = "balanced"
    project_path: str | Path = "."
    scene_path: str = "res://world.jscn"
    robot_name: str = "robot"
    base_link: str = "base"
    joint_names: tuple[str, ...] = ()
    default_joint_pos: tuple[float, ...] = ()
    foot_names: tuple[str, ...] = ()
    foot_link_names: tuple[str, ...] = ()
    action_scale: float | Mapping[str, float] = field(default_factory=lambda: {})
    simulate_action_latency: bool = False
    action_clip: float | None = None
    kp: float | tuple[float, ...] = 0.0
    kd: float | tuple[float, ...] = 0.0
    physics_dt: float = 0.005
    decimation: int = 4
    mujoco_solver_settings: Mapping[str, Any] = field(default_factory=lambda: {})
    episode_length_s: float = 20.0
    base_clearance: float = 0.278
    spawn_jitter: float = 0.5
    reset_z_range: tuple[float, float] = (0.01, 0.05)
    randomize_reset_yaw: bool = True
    # Controls spawn-level progression over the authored terrain grid. Geometry
    # is owned by the TerrainGeneratorConfig referenced from the authored scene.
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
            rel_run_envs=0.0,
            run_velocity_x=(1.5, 2.5),
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
        VelocityCommandStage(step=60_000, lin_vel_x=(-1.5, 2.0), ang_vel_z=(-0.7, 0.7)),
        VelocityCommandStage(step=120_000, lin_vel_x=(-2.0, 3.0)),
    )
    run_command_curriculum: tuple[VelocityCommandStage, ...] = (
        VelocityCommandStage(step=0, run_velocity_x=(1.5, 2.5)),
    )
    rewards: VelocityRewardCfg = field(default_factory=VelocityRewardCfg)


    def __post_init__(self):
        if self.robot is not None:
            profile = self.robot
            self.robot_name = profile.name
            self.base_link = profile.base_link
            self.joint_names = profile.joint_names
            self.default_joint_pos = profile.default_joint_pos
            self.kp, self.kd = profile.kp, profile.kd
            self.foot_names = tuple(foot.name for foot in profile.feet)
            self.foot_link_names = tuple(foot.link for foot in profile.feet)
        if self.curriculum_mode not in {"legacy", "command_progress"}:
            raise ValueError("unknown terrain curriculum mode")
