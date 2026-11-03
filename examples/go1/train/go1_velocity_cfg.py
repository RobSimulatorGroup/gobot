"""Configuration for the Go1 rough-terrain velocity task."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

from gobot.rl.locomotion import UniformVelocityCommandCfg, UniformVelocityCommandRanges, VelocityCommandStage

from ..go1_profile import (
    GO1_DEFAULT_JOINT_POS,
    GO1_FOOT_LINK_NAMES,
    GO1_FOOT_NAMES,
    GO1_JOINT_NAMES,
    GO1_KD,
    GO1_KP,
)
from ..go1_velocity_contract import GO1_TASK_NAME


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

GO1_ROUGH_REWARD_TERM_NAMES: tuple[str, ...] = (
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


from gobot.rl.locomotion.config import (
    FootCfg, LocomotionRobotCfg, VelocityTaskCfg, VelocityObservationCfg,
    VelocityRewardCfg, VelocityIllegalContactCfg, VelocityDomainRandomizationCfg,
    _default_push_force_ranges, _default_push_velocity_ranges,
)

@dataclass
class Go1VelocityCfg(VelocityTaskCfg):
    """Single source of truth for Go1 rough-terrain training and playback."""

    task_version: str = "go1_rough_velocity_numpy_v2"
    post_step_sensing: bool = False
    name: str = GO1_TASK_NAME
    training_profile: Literal["balanced", "run"] = "balanced"
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
        from ..go1_profile import GO1_ARMATURE, GO1_EFFORT_LIMIT, GO1_VELOCITY_LIMIT
        offsets = ((0., 0., 0.), (0.04, 0., 0.), (0., 0.04, 0.), (-0.04, 0., 0.), (0., -0.04, 0.))
        self.robot = LocomotionRobotCfg(
            name=self.robot_name, base_link=self.base_link, torso_link=self.base_link,
            joint_names=self.joint_names, default_joint_pos=self.default_joint_pos,
            kp=(float(self.kp),) * len(self.joint_names) if isinstance(self.kp, (int, float)) else tuple(self.kp),
            kd=(float(self.kd),) * len(self.joint_names) if isinstance(self.kd, (int, float)) else tuple(self.kd),
            armature=GO1_ARMATURE,
            effort_limit=GO1_EFFORT_LIMIT, velocity_limit=GO1_VELOCITY_LIMIT,
            feet=tuple(FootCfg(name, link, (f"{name}_foot_collision",),
                f"{name}_foot_contact", f"{name}_foot_height_scan",
                offsets)
                for name, link in zip(self.foot_names, self.foot_link_names, strict=True)),
        )
        if self.training_profile == "run":
            self.gait_foot_indices = (0, 1, 2, 3)
            self.gait_joint_indices = tuple(tuple(range(i, i + 3)) for i in range(0, 12, 3))

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


def apply_training_profile(
    cfg: Go1VelocityCfg,
    profile: Literal["balanced", "run"],
) -> Go1VelocityCfg:
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
        cfg.rewards.bound_gait = 3.0
    return cfg


from gobot.rl.locomotion.ppo import rsl_rl_train_cfg as _shared_ppo_cfg

def rsl_rl_train_cfg(**kwargs):
    kwargs.setdefault("experiment_name", GO1_TASK_NAME)
    return _shared_ppo_cfg(**kwargs)
