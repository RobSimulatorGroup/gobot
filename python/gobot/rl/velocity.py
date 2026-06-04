"""MJLab-style velocity locomotion tasks backed by Gobot runtime sensors.

This module ports the velocity-task shape from MJLab without importing the
``mjlab`` package or MuJoCo-Warp.  The source of truth remains a Gobot scene;
runtime observations come from Gobot link, joint, ``HeightScanner3D``, and
``ContactSensor3D`` state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

import gobot


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
class UniformVelocityCommandRanges:
    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: tuple[float, float] = (-1.0, 1.0)
    ang_vel_z: tuple[float, float] = (-0.5, 0.5)
    heading: tuple[float, float] | None = (-math.pi, math.pi)


@dataclass
class UniformVelocityCommandCfg:
    """Subset of MJLab's velocity command config used by Gobot tasks."""

    name: str = "twist"
    resampling_time_range: tuple[float, float] = (3.0, 8.0)
    rel_standing_envs: float = 0.1
    rel_heading_envs: float = 0.3
    rel_world_envs: float = 0.0
    rel_forward_envs: float = 0.2
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    init_velocity_prob: float = 0.0
    ranges: UniformVelocityCommandRanges = field(default_factory=UniformVelocityCommandRanges)


@dataclass
class VelocityStage:
    step: int
    lin_vel_x: tuple[float, float] | None = None
    lin_vel_y: tuple[float, float] | None = None
    ang_vel_z: tuple[float, float] | None = None


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
class VelocityTaskCfg:
    """Gobot-native equivalent of MJLab velocity task config."""

    name: str = "unitree_go1_rough_velocity"
    robot_family: str = "go1"
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
    observations: VelocityObservationCfg = field(default_factory=VelocityObservationCfg)
    command: UniformVelocityCommandCfg = field(default_factory=UniformVelocityCommandCfg)
    command_curriculum: tuple[VelocityStage, ...] = (
        VelocityStage(step=0, lin_vel_x=(-1.0, 1.0), ang_vel_z=(-0.5, 0.5)),
        VelocityStage(step=5000 * 24, lin_vel_x=(-1.5, 2.0), ang_vel_z=(-0.7, 0.7)),
        VelocityStage(step=10000 * 24, lin_vel_x=(-2.0, 3.0)),
    )
    rewards: VelocityRewardCfg = field(default_factory=VelocityRewardCfg)


def unitree_go1_rough_velocity_cfg(
    *,
    project_path: str | Path = "examples/go1",
    play: bool = False,
) -> VelocityTaskCfg:
    cfg = VelocityTaskCfg(project_path=project_path)
    if play:
        cfg.episode_length_s = float(1_000_000_000)
        cfg.terrain_curriculum = False
        cfg.observations.actor_noise = False
    return cfg


def unitree_go1_flat_velocity_cfg(
    *,
    project_path: str | Path = "examples/go1",
    play: bool = False,
) -> VelocityTaskCfg:
    cfg = unitree_go1_rough_velocity_cfg(project_path=project_path, play=play)
    cfg.name = "unitree_go1_flat_velocity"
    cfg.terrain_type = "flat"
    cfg.observations.height_scan_sensor = None
    cfg.terrain_curriculum = False
    cfg.rewards.upright = 1.0
    return cfg


def unitree_g1_rough_velocity_cfg(
    *,
    project_path: str | Path = "examples/g1",
    play: bool = False,
) -> VelocityTaskCfg:
    del play
    return VelocityTaskCfg(
        name="unitree_g1_rough_velocity",
        robot_family="g1",
        project_path=project_path,
        scene_path="res://g1_scene.jscn",
        terrain_scene_path="terrain_scene.jscn",
        robot_name="g1",
        base_link="torso_link",
        joint_names=(),
        default_joint_pos=(),
        foot_names=("left", "right"),
        foot_link_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        action_scale=0.25,
        kp=40.0,
        kd=1.0,
    )


def unitree_g1_flat_velocity_cfg(
    *,
    project_path: str | Path = "examples/g1",
    play: bool = False,
) -> VelocityTaskCfg:
    cfg = unitree_g1_rough_velocity_cfg(project_path=project_path, play=play)
    cfg.name = "unitree_g1_flat_velocity"
    cfg.terrain_type = "flat"
    cfg.observations.height_scan_sensor = None
    cfg.terrain_curriculum = False
    return cfg


def velocity_task_cfg(name: str, *, project_path: str | Path | None = None, play: bool = False) -> VelocityTaskCfg:
    factories = {
        "go1_rough": unitree_go1_rough_velocity_cfg,
        "unitree_go1_rough": unitree_go1_rough_velocity_cfg,
        "unitree_go1_rough_velocity": unitree_go1_rough_velocity_cfg,
        "go1_flat": unitree_go1_flat_velocity_cfg,
        "unitree_go1_flat": unitree_go1_flat_velocity_cfg,
        "unitree_go1_flat_velocity": unitree_go1_flat_velocity_cfg,
        "g1_rough": unitree_g1_rough_velocity_cfg,
        "unitree_g1_rough": unitree_g1_rough_velocity_cfg,
        "unitree_g1_rough_velocity": unitree_g1_rough_velocity_cfg,
        "g1_flat": unitree_g1_flat_velocity_cfg,
        "unitree_g1_flat": unitree_g1_flat_velocity_cfg,
        "unitree_g1_flat_velocity": unitree_g1_flat_velocity_cfg,
    }
    key = name.lower()
    if key not in factories:
        raise ValueError(f"unknown velocity task {name!r}; expected one of {sorted(factories)}")
    kwargs: dict[str, Any] = {"play": play}
    if project_path is not None:
        kwargs["project_path"] = project_path
    return factories[key](**kwargs)


@dataclass
class VelocityRuntimeState:
    robot: Mapping[str, Any]
    base: Mapping[str, Any]
    joints: Mapping[str, Mapping[str, Any]]
    links: Mapping[str, Mapping[str, Any]]
    sensors: Mapping[str, Mapping[str, Any]]
    contacts: Sequence[Mapping[str, Any]]


class TerrainSampler:
    """Small height-query fallback for reset placement."""

    def __init__(self, scene_path: str | Path, grid_resolution: float = 0.08) -> None:
        self._boxes: list[dict[str, Any]] = []
        self._heightfields: list[dict[str, Any]] = []
        self._grid_resolution = float(grid_resolution)
        self._grid_origin = np.zeros(2, dtype=np.float64)
        self._grid_heights: np.ndarray | None = None
        path = Path(scene_path)
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        terrain_node = next((node for node in data.get("__NODES__", []) if node.get("type") == "Terrain3D"), None)
        if terrain_node is None:
            return
        properties = terrain_node.get("properties", {})
        for box in properties.get("boxes", []):
            self._boxes.append(
                {
                    "center": _json_vec(box.get("center"), 3),
                    "size": _json_vec(box.get("size"), 3),
                    "rotation": _json_vec(box.get("rotation_degrees"), 3),
                }
            )
        for heightfield in properties.get("heightfields", []):
            rows = int(heightfield.get("rows", 0))
            cols = int(heightfield.get("cols", 0))
            heights = np.asarray(heightfield.get("heights", []), dtype=np.float64)
            if rows <= 1 or cols <= 1 or heights.size != rows * cols:
                continue
            self._heightfields.append(
                {
                    "center": _json_vec(heightfield.get("center"), 3),
                    "size": _json_vec(heightfield.get("size"), 2),
                    "rows": rows,
                    "cols": cols,
                    "heights": heights.reshape(rows, cols),
                    "z_offset": float(heightfield.get("z_offset", 0.0)),
                }
            )
        self._build_height_grid()

    def height_at(self, x: float, y: float) -> float:
        if self._grid_heights is not None:
            return self._grid_height_at(x, y)
        height = -np.inf
        for box in self._boxes:
            candidate = self._box_height(box, x, y)
            if candidate is not None:
                height = max(height, candidate)
        for heightfield in self._heightfields:
            candidate = self._heightfield_height(heightfield, x, y)
            if candidate is not None:
                height = max(height, candidate)
        return float(height if np.isfinite(height) else 0.0)

    def _build_height_grid(self) -> None:
        bounds = self._terrain_bounds()
        if bounds is None:
            return
        min_x, min_y, max_x, max_y = bounds
        padding = max(self._grid_resolution * 2.0, 0.25)
        xs = np.arange(min_x - padding, max_x + padding + self._grid_resolution * 0.5, self._grid_resolution)
        ys = np.arange(min_y - padding, max_y + padding + self._grid_resolution * 0.5, self._grid_resolution)
        if xs.size < 2 or ys.size < 2:
            return
        grid_x, grid_y = np.meshgrid(xs, ys)
        heights = np.full(grid_x.shape, -np.inf, dtype=np.float64)
        for box in self._boxes:
            candidate = self._box_height_grid(box, grid_x, grid_y)
            if candidate is not None:
                heights = np.maximum(heights, candidate)
        for heightfield in self._heightfields:
            candidate = self._heightfield_height_grid(heightfield, grid_x, grid_y)
            if candidate is not None:
                heights = np.maximum(heights, candidate)
        heights[~np.isfinite(heights)] = 0.0
        self._grid_origin = np.array([xs[0], ys[0]], dtype=np.float64)
        self._grid_heights = heights

    def _terrain_bounds(self) -> tuple[float, float, float, float] | None:
        bounds = []
        for box in self._boxes:
            center = box["center"]
            size = box["size"]
            bounds.append((center[0] - size[0] * 0.5, center[1] - size[1] * 0.5, center[0] + size[0] * 0.5, center[1] + size[1] * 0.5))
        for heightfield in self._heightfields:
            center = heightfield["center"]
            size = heightfield["size"]
            bounds.append((center[0] - size[0] * 0.5, center[1] - size[1] * 0.5, center[0] + size[0] * 0.5, center[1] + size[1] * 0.5))
        if not bounds:
            return None
        return (
            min(bound[0] for bound in bounds),
            min(bound[1] for bound in bounds),
            max(bound[2] for bound in bounds),
            max(bound[3] for bound in bounds),
        )

    def _grid_height_at(self, x: float, y: float) -> float:
        assert self._grid_heights is not None
        u = (float(x) - self._grid_origin[0]) / self._grid_resolution
        v = (float(y) - self._grid_origin[1]) / self._grid_resolution
        rows, cols = self._grid_heights.shape
        if u < 0.0 or v < 0.0 or u > cols - 1 or v > rows - 1:
            return 0.0
        c0 = int(np.clip(np.floor(u), 0, cols - 1))
        r0 = int(np.clip(np.floor(v), 0, rows - 1))
        c1 = min(c0 + 1, cols - 1)
        r1 = min(r0 + 1, rows - 1)
        fu = float(u - c0)
        fv = float(v - r0)
        h00 = self._grid_heights[r0, c0]
        h10 = self._grid_heights[r0, c1]
        h01 = self._grid_heights[r1, c0]
        h11 = self._grid_heights[r1, c1]
        return float((h00 * (1.0 - fu) + h10 * fu) * (1.0 - fv) + (h01 * (1.0 - fu) + h11 * fu) * fv)

    @staticmethod
    def _box_height(box: Mapping[str, Any], x: float, y: float) -> float | None:
        center = box["center"]
        size = box["size"]
        rotation = box["rotation"]
        yaw = -np.deg2rad(rotation[2])
        local_x = x - center[0]
        local_y = y - center[1]
        rx = np.cos(yaw) * local_x - np.sin(yaw) * local_y
        ry = np.sin(yaw) * local_x + np.cos(yaw) * local_y
        if abs(rx) > size[0] * 0.5 or abs(ry) > size[1] * 0.5:
            return None
        return float(center[2] + size[2] * 0.5)

    @staticmethod
    def _box_height_grid(box: Mapping[str, Any], grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray | None:
        center = box["center"]
        size = box["size"]
        rotation = box["rotation"]
        yaw = -np.deg2rad(rotation[2])
        local_x = grid_x - center[0]
        local_y = grid_y - center[1]
        rx = np.cos(yaw) * local_x - np.sin(yaw) * local_y
        ry = np.sin(yaw) * local_x + np.cos(yaw) * local_y
        mask = (np.abs(rx) <= size[0] * 0.5) & (np.abs(ry) <= size[1] * 0.5)
        if not np.any(mask):
            return None
        return np.where(mask, center[2] + size[2] * 0.5, -np.inf)

    @staticmethod
    def _heightfield_height(heightfield: Mapping[str, Any], x: float, y: float) -> float | None:
        center = heightfield["center"]
        size = heightfield["size"]
        local_x = x - center[0]
        local_y = y - center[1]
        if abs(local_x) > size[0] * 0.5 or abs(local_y) > size[1] * 0.5:
            return None
        cols = heightfield["cols"]
        rows = heightfield["rows"]
        u = (local_x / size[0] + 0.5) * (cols - 1)
        v = (local_y / size[1] + 0.5) * (rows - 1)
        c0 = int(np.clip(np.floor(u), 0, cols - 1))
        r0 = int(np.clip(np.floor(v), 0, rows - 1))
        c1 = min(c0 + 1, cols - 1)
        r1 = min(r0 + 1, rows - 1)
        fu = float(u - c0)
        fv = float(v - r0)
        heights = heightfield["heights"]
        h00 = heights[r0, c0]
        h10 = heights[r0, c1]
        h01 = heights[r1, c0]
        h11 = heights[r1, c1]
        return float(center[2] + heightfield["z_offset"] + (h00 * (1.0 - fu) + h10 * fu) * (1.0 - fv) + (h01 * (1.0 - fu) + h11 * fu) * fv)

    @staticmethod
    def _heightfield_height_grid(heightfield: Mapping[str, Any], grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray | None:
        center = heightfield["center"]
        size = heightfield["size"]
        local_x = grid_x - center[0]
        local_y = grid_y - center[1]
        mask = (np.abs(local_x) <= size[0] * 0.5) & (np.abs(local_y) <= size[1] * 0.5)
        if not np.any(mask):
            return None
        cols = heightfield["cols"]
        rows = heightfield["rows"]
        u = (local_x / size[0] + 0.5) * (cols - 1)
        v = (local_y / size[1] + 0.5) * (rows - 1)
        c0 = np.clip(np.floor(u).astype(np.int64), 0, cols - 1)
        r0 = np.clip(np.floor(v).astype(np.int64), 0, rows - 1)
        c1 = np.minimum(c0 + 1, cols - 1)
        r1 = np.minimum(r0 + 1, rows - 1)
        fu = u - c0
        fv = v - r0
        heights = heightfield["heights"]
        h00 = heights[r0, c0]
        h10 = heights[r0, c1]
        h01 = heights[r1, c0]
        h11 = heights[r1, c1]
        sampled = center[2] + heightfield["z_offset"] + (h00 * (1.0 - fu) + h10 * fu) * (1.0 - fv) + (h01 * (1.0 - fu) + h11 * fu) * fv
        return np.where(mask, sampled, -np.inf)


class UniformVelocityCommand:
    def __init__(self, cfg: UniformVelocityCommandCfg, env: "GobotVelocityEnv") -> None:
        self.cfg = cfg
        self.env = env
        self.command_b = np.zeros((env.num_envs, 3), dtype=np.float32)
        self.command_w = np.zeros((env.num_envs, 3), dtype=np.float32)
        self.heading_target = np.zeros(env.num_envs, dtype=np.float32)
        self.heading_error = np.zeros(env.num_envs, dtype=np.float32)
        self.is_heading_env = np.zeros(env.num_envs, dtype=bool)
        self.is_standing_env = np.zeros(env.num_envs, dtype=bool)
        self.is_world_env = np.zeros(env.num_envs, dtype=bool)
        self.is_forward_env = np.zeros(env.num_envs, dtype=bool)
        self.time_left = np.zeros(env.num_envs, dtype=np.float32)
        self.metrics: dict[str, np.ndarray] = {
            "error_vel_xy": np.zeros(env.num_envs, dtype=np.float32),
            "error_vel_yaw": np.zeros(env.num_envs, dtype=np.float32),
        }

    def reset(self, env_ids: np.ndarray) -> None:
        self._resample(env_ids)
        low, high = self.cfg.resampling_time_range
        self.time_left[env_ids] = self.env._rng.uniform(low, high, size=env_ids.shape).astype(np.float32)
        self.metrics["error_vel_xy"][env_ids] = 0.0
        self.metrics["error_vel_yaw"][env_ids] = 0.0

    def compute(self, dt: float, states: Sequence[VelocityRuntimeState | None]) -> None:
        self.time_left -= float(dt)
        resample_ids = np.flatnonzero(self.time_left <= 0.0).astype(np.int64)
        if resample_ids.size:
            self.reset(resample_ids)
        self._update_heading_and_world_commands(states)
        max_command_step = max(self.cfg.resampling_time_range[1] / max(self.env.step_dt, 1.0e-9), 1.0)
        for env_id, state in enumerate(states):
            if state is None:
                continue
            base_lin_vel_b = _rotate_vec_by_quat_inv(_as_vec(state.base.get("linear_velocity"), 3), _quat(state.base))
            base_ang_vel_b = _rotate_vec_by_quat_inv(_as_vec(state.base.get("angular_velocity"), 3), _quat(state.base))
            self.metrics["error_vel_xy"][env_id] += np.linalg.norm(self.command_b[env_id, :2] - base_lin_vel_b[:2]) / max_command_step
            self.metrics["error_vel_yaw"][env_id] += abs(self.command_b[env_id, 2] - base_ang_vel_b[2]) / max_command_step

    def _resample(self, env_ids: np.ndarray) -> None:
        ranges = self.cfg.ranges
        self.command_b[env_ids, 0] = self.env._rng.uniform(*ranges.lin_vel_x, size=env_ids.shape)
        self.command_b[env_ids, 1] = self.env._rng.uniform(*ranges.lin_vel_y, size=env_ids.shape)
        self.command_b[env_ids, 2] = self.env._rng.uniform(*ranges.ang_vel_z, size=env_ids.shape)
        if self.cfg.heading_command:
            if ranges.heading is None:
                raise ValueError("heading_command=True requires heading range")
            self.heading_target[env_ids] = self.env._rng.uniform(*ranges.heading, size=env_ids.shape)
            self.is_heading_env[env_ids] = self.env._rng.uniform(0.0, 1.0, size=env_ids.shape) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = self.env._rng.uniform(0.0, 1.0, size=env_ids.shape) <= self.cfg.rel_standing_envs
        self.is_world_env[env_ids] = self.env._rng.uniform(0.0, 1.0, size=env_ids.shape) <= self.cfg.rel_world_envs
        self.command_w[env_ids] = self.command_b[env_ids]
        self.is_forward_env[env_ids] = self.env._rng.uniform(0.0, 1.0, size=env_ids.shape) <= self.cfg.rel_forward_envs
        forward_ids = env_ids[self.is_forward_env[env_ids]]
        if forward_ids.size:
            self.command_b[forward_ids, 0] = np.maximum(np.abs(self.command_b[forward_ids, 0]), 0.3)
            self.command_b[forward_ids, 1] = 0.0
            self.command_b[forward_ids, 2] = 0.0

    def _update_heading_and_world_commands(self, states: Sequence[VelocityRuntimeState | None]) -> None:
        for env_id, state in enumerate(states):
            if state is None:
                continue
            heading = _quat_to_yaw(_quat(state.base))
            if self.cfg.heading_command and self.is_heading_env[env_id]:
                self.heading_error[env_id] = _wrap_to_pi(float(self.heading_target[env_id]) - heading)
                lo, hi = self.cfg.ranges.ang_vel_z
                self.command_b[env_id, 2] = np.clip(self.cfg.heading_control_stiffness * self.heading_error[env_id], lo, hi)
            if self.is_world_env[env_id]:
                vx_w, vy_w = self.command_w[env_id, :2]
                self.command_b[env_id, 0] = math.cos(heading) * vx_w + math.sin(heading) * vy_w
                self.command_b[env_id, 1] = -math.sin(heading) * vx_w + math.cos(heading) * vy_w
        self.command_b[self.is_standing_env] = 0.0
        self.command_w[self.is_standing_env] = 0.0


class GobotVelocityEnv:
    """rsl_rl-compatible vector env for MJLab-style velocity tasks."""

    is_vector_env = True

    def __init__(
        self,
        cfg: VelocityTaskCfg | None = None,
        *,
        num_envs: int = 64,
        device: str = "cpu",
        seed: int = 42,
        max_episode_length: int | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("GobotVelocityEnv requires gobot[train] dependencies.") from error
        self.torch = torch
        self.cfg_obj = cfg if cfg is not None else unitree_go1_rough_velocity_cfg()
        if self.cfg_obj.name != "unitree_go1_rough_velocity":
            raise NotImplementedError(
                f"{self.cfg_obj.name} is configured for API parity with MJLab velocity, "
                "but the current Gobot repository only ships a runnable Go1 rough velocity scene."
            )
        if self.cfg_obj.robot_family != "go1":
            raise NotImplementedError(
                f"{self.cfg_obj.name} is configured, but this repository currently ships only the Go1 Gobot asset."
            )
        self.num_envs = int(num_envs)
        self.device = device
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.num_actions = len(self.cfg_obj.joint_names)
        self.joint_names = tuple(self.cfg_obj.joint_names)
        self.default_joint_pos = np.asarray(self.cfg_obj.default_joint_pos, dtype=np.float32)
        if self.num_actions == 0 or self.default_joint_pos.shape != (self.num_actions,):
            raise ValueError("velocity task joint_names and default_joint_pos must be non-empty and have the same length")
        self.action_scale = self._action_scale_array(self.cfg_obj.action_scale)
        self.physics_dt = float(self.cfg_obj.physics_dt)
        self.decimation = int(self.cfg_obj.decimation)
        self.step_dt = self.physics_dt * self.decimation
        self.max_episode_length = int(max_episode_length or math.ceil(float(self.cfg_obj.episode_length_s) / self.step_dt))
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self._episode_start_xy = np.zeros((self.num_envs, 2), dtype=np.float32)
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._total_policy_steps = 0
        self._curriculum_progress = 0.0

        self._last_actions = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)
        self._previous_actions = np.zeros_like(self._last_actions)
        self._last_foot_contact = np.zeros((self.num_envs, len(self.cfg_obj.foot_names)), dtype=np.float32)
        self._foot_air_time = np.zeros_like(self._last_foot_contact)
        self._foot_peak_height = np.zeros_like(self._last_foot_contact)
        self._previous_foot_positions = np.zeros((self.num_envs, len(self.cfg_obj.foot_names), 3), dtype=np.float32)
        self._foot_velocities = np.zeros_like(self._previous_foot_positions)
        self._spawn_indices = np.zeros(self.num_envs, dtype=np.int64)
        self._terrain_levels = np.zeros(self.num_envs, dtype=np.float32)
        self._reset_reasons = np.zeros(self.num_envs, dtype=np.int64)

        self.project_path = Path(self.cfg_obj.project_path).resolve()
        self.context = gobot.app.context()
        self.context.set_project_path(str(self.project_path))
        self.context.load_scene(self.cfg_obj.scene_path)
        self._spawn_origins = self._load_spawn_origins()
        self._center_spawn_index = int(np.argmin(np.linalg.norm(self._spawn_origins[:, :2], axis=1)))
        self._terrain_sampler = TerrainSampler(self.project_path / self.cfg_obj.terrain_scene_path)
        self._spawn_difficulties = self._spawn_difficulty_scores()
        self._spawn_order = np.argsort(self._spawn_difficulties, kind="stable")
        max_difficulty = max(float(np.max(self._spawn_difficulties)), 1.0e-6)
        self._spawn_levels = np.clip(self._spawn_difficulties / max_difficulty, 0.0, 1.0).astype(np.float32)

        self.context.fixed_time_step = self.physics_dt
        self.context.set_default_joint_gains(
            {
                "position_stiffness": self.cfg_obj.kp,
                "velocity_damping": self.cfg_obj.kd,
                "integral_gain": 0.0,
                "integral_limit": 0.0,
            }
        )
        self._configure_robot_drives()
        self.context.build_world(gobot.PhysicsBackendType.MuJoCoCpu)
        self.context.configure_batch_world(self.num_envs)

        name_map = self.context.get_runtime_name_map()
        robot_map = next((robot for robot in name_map.get("robots", []) if robot.get("name") == self.cfg_obj.robot_name), None)
        if robot_map is None:
            raise RuntimeError(f"Gobot scene has no robot named {self.cfg_obj.robot_name!r}")
        self._height_scan_dim = self._sensor_dim(robot_map, self.cfg_obj.observations.height_scan_sensor)
        self._foot_count = len(self.cfg_obj.foot_names)
        self.num_obs = 3 + 3 + 3 + self.num_actions + self.num_actions + self.num_actions + 3 + self._height_scan_dim
        self.num_privileged_obs = self.num_obs + self._foot_count * (1 + 1 + 1 + 3)
        self.command_manager = UniformVelocityCommand(self.cfg_obj.command, self)

        self.cfg = {
            "name": self.cfg_obj.name,
            "source": "gobot.rl.velocity",
            "mjlab_task": "velocity",
            "robot": self.cfg_obj.robot_name,
            "scene_path": self.cfg_obj.scene_path,
            "project_path": str(self.project_path),
            "num_envs": self.num_envs,
            "num_obs": self.num_obs,
            "num_privileged_obs": self.num_privileged_obs,
            "num_actions": self.num_actions,
            "joint_names": self.joint_names,
            "height_scan_dim": self._height_scan_dim,
            "decimation": self.decimation,
            "physics_dt": self.physics_dt,
        }
        self.extras: dict[str, Any] = {}
        self._obs, self._critic_obs = self._reset_all()

    def get_observations(self):
        return self._tensor_dict(self._obs, self._critic_obs)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed = int(seed)
            self._rng = np.random.default_rng(self.seed)
        self.episode_length_buf.zero_()
        self._episode_returns[:] = 0.0
        self._obs, self._critic_obs = self._reset_all()
        return self.get_observations()

    def step(self, actions):
        torch = self.torch
        self._total_policy_steps += 1
        self._apply_command_curriculum()
        self._update_curriculum_progress()

        action_np = actions.detach().cpu().numpy() if hasattr(actions, "detach") else np.asarray(actions)
        action_np = np.asarray(action_np, dtype=np.float32).reshape(self.num_envs, self.num_actions)
        action_np = np.clip(action_np, -1.0, 1.0)

        states_after_step: list[VelocityRuntimeState | None] = [None] * self.num_envs
        for env_id in range(self.num_envs):
            self._apply_action(env_id, action_np[env_id])
            self.context.step_batch_env(env_id, self.decimation)
            states_after_step[env_id] = self._runtime_state(env_id)
        self.command_manager.compute(self.step_dt, states_after_step)

        obs_list = []
        critic_list = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        time_outs = np.zeros(self.num_envs, dtype=bool)
        reset_reason = np.zeros(self.num_envs, dtype=np.int64)
        metrics: dict[str, np.ndarray] = {
            "base_clearance": np.zeros(self.num_envs, dtype=np.float32),
            "velocity_error": np.zeros(self.num_envs, dtype=np.float32),
            "foot_clearance": np.zeros(self.num_envs, dtype=np.float32),
            "foot_contact_ratio": np.zeros(self.num_envs, dtype=np.float32),
            "foot_slip": np.zeros(self.num_envs, dtype=np.float32),
        }
        reward_terms: dict[str, np.ndarray] = {}

        for env_id, state in enumerate(states_after_step):
            assert state is not None
            self._update_foot_history(env_id, state)
            reward, breakdown = self._compute_reward(env_id, state, action_np[env_id])
            for name, value in breakdown.items():
                reward_terms.setdefault(name, np.zeros(self.num_envs, dtype=np.float32))[env_id] = value
            rewards[env_id] = reward
            terminated[env_id] = self._terminated(state)
            self.episode_length_buf[env_id] += 1
            time_outs[env_id] = bool(self.episode_length_buf[env_id] >= self.max_episode_length and not terminated[env_id])
            done = bool(terminated[env_id] or time_outs[env_id])
            reset_reason[env_id] = 1 if terminated[env_id] else (2 if time_outs[env_id] else 0)
            self._episode_returns[env_id] += rewards[env_id]
            self._previous_actions[env_id] = self._last_actions[env_id]
            self._last_actions[env_id] = action_np[env_id]

            if done:
                self._reset_env(env_id, reason=reset_reason[env_id])
                state = self._runtime_state(env_id)

            obs = self._actor_obs(env_id, state, corrupt=True)
            critic = self._critic_obs_for(env_id, state, obs)
            obs_list.append(obs)
            critic_list.append(critic)
            metrics["base_clearance"][env_id] = self._base_clearance(state)
            base_lin_vel_b = _rotate_vec_by_quat_inv(_as_vec(state.base.get("linear_velocity"), 3), _quat(state.base))
            metrics["velocity_error"][env_id] = float(np.linalg.norm(base_lin_vel_b[:2] - self.command_manager.command_b[env_id, :2]))
            foot_heights = self._foot_heights(state)
            foot_contacts = self._foot_contacts(state)
            metrics["foot_clearance"][env_id] = float(np.mean(foot_heights)) if foot_heights.size else 0.0
            metrics["foot_contact_ratio"][env_id] = float(np.mean(foot_contacts)) if foot_contacts.size else 0.0
            metrics["foot_slip"][env_id] = float(self._foot_slip_cost(env_id, foot_contacts))

        self._obs = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)
        self._critic_obs = torch.as_tensor(np.stack(critic_list), dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(terminated | time_outs, dtype=torch.bool, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        extras = {
            "time_outs": torch.as_tensor(time_outs, dtype=torch.bool, device=self.device),
            "log": {
                "/velocity/terrain_level": torch.as_tensor(float(np.mean(self._terrain_levels)), device=self.device),
                "/velocity/velocity_error": torch.as_tensor(float(np.mean(metrics["velocity_error"])), device=self.device),
                "/velocity/foot_clearance": torch.as_tensor(float(np.mean(metrics["foot_clearance"])), device=self.device),
                "/velocity/foot_contact_ratio": torch.as_tensor(float(np.mean(metrics["foot_contact_ratio"])), device=self.device),
                "/velocity/foot_slip": torch.as_tensor(float(np.mean(metrics["foot_slip"])), device=self.device),
                "/velocity/reset_reason": torch.as_tensor(float(np.mean(reset_reason)), device=self.device),
                "/velocity/command_vx": torch.as_tensor(float(np.mean(self.command_manager.command_b[:, 0])), device=self.device),
                "/velocity/command_vy": torch.as_tensor(float(np.mean(self.command_manager.command_b[:, 1])), device=self.device),
                "/velocity/command_yaw": torch.as_tensor(float(np.mean(self.command_manager.command_b[:, 2])), device=self.device),
            },
            "reward_terms": {name: torch.as_tensor(value, dtype=torch.float32, device=self.device) for name, value in reward_terms.items()},
        }
        self.extras = extras
        return self._tensor_dict(self._obs, self._critic_obs), rewards_t, done_t, extras

    def close(self) -> None:
        self.context.clear_world()

    def set_training_progress(self, policy_steps: int) -> None:
        self._total_policy_steps = max(0, int(policy_steps))
        self._apply_command_curriculum()
        self._update_curriculum_progress()

    def _tensor_dict(self, actor, critic):
        from tensordict import TensorDict

        return TensorDict(
            {"actor": actor.clone(), "critic": critic.clone(), "policy": actor.clone()},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def _action_scale_array(self, scale: float | Mapping[str, float]) -> np.ndarray:
        if isinstance(scale, Mapping):
            values = []
            for joint_name in self.cfg_obj.joint_names:
                matched = None
                for pattern, value in scale.items():
                    if re.fullmatch(pattern, joint_name) or pattern == joint_name:
                        matched = float(value)
                        break
                values.append(0.35 if matched is None else matched)
            return np.asarray(values, dtype=np.float32)
        return np.full((self.num_actions,), float(scale), dtype=np.float32)

    @staticmethod
    def _sensor_dim(robot_map: Mapping[str, Any], sensor_name: str | None) -> int:
        if sensor_name is None:
            return 0
        for sensor in robot_map.get("sensors", []):
            if sensor.get("sensor_name") == sensor_name or sensor.get("name") == sensor_name:
                return len(sensor.get("channel_names", []))
        return 0

    def _configure_robot_drives(self) -> None:
        root = self.context.root
        robot = root.find(self.cfg_obj.robot_name) if root is not None else None
        if robot is None:
            robot = _find_node_by_name(root, self.cfg_obj.robot_name)
        if robot is None:
            return
        for joint_name in self.joint_names:
            joint = robot.find(joint_name) or _find_node_by_name(robot, joint_name)
            if joint is None:
                continue
            joint.drive_mode = gobot.JointDriveMode.Position
            joint.drive_stiffness = self.cfg_obj.kp
            joint.drive_damping = self.cfg_obj.kd
            joint.damping = self.cfg_obj.kd

    def _load_spawn_origins(self) -> np.ndarray:
        root = self.context.root
        terrain_world = root.find("terrain_world") if root is not None else None
        terrain = terrain_world.find("terrain") if terrain_world is not None else None
        origins = np.asarray(getattr(terrain, "spawn_origins", []), dtype=np.float64)
        if origins.ndim != 2 or origins.shape[1] != 3 or origins.shape[0] == 0:
            return np.zeros((1, 3), dtype=np.float64)
        return origins

    def _spawn_difficulty_scores(self) -> np.ndarray:
        radius = float(self.cfg_obj.spawn_difficulty_radius)
        offsets = np.asarray([(x, y) for x in (-radius, 0.0, radius) for y in (-radius, 0.0, radius)], dtype=np.float64)
        scores = np.zeros(self._spawn_origins.shape[0], dtype=np.float32)
        for index, origin in enumerate(self._spawn_origins):
            heights = np.asarray([self._terrain_height(origin[0] + dx, origin[1] + dy) for dx, dy in offsets], dtype=np.float64)
            scores[index] = float(np.max(heights) - np.min(heights))
        return scores

    def _terrain_height(self, x: float, y: float) -> float:
        return self._terrain_sampler.height_at(float(x), float(y))

    def _sample_spawn_index(self) -> int:
        if not self.cfg_obj.terrain_curriculum:
            return int(self._rng.integers(0, self._spawn_origins.shape[0]))
        warmup_progress = 0.10
        if self._curriculum_progress < warmup_progress:
            return self._center_spawn_index
        difficulty_progress = (self._curriculum_progress - warmup_progress) / max(1.0 - warmup_progress, 1.0e-6)
        allowed_count = max(1, int(np.ceil(difficulty_progress * self._spawn_order.size)))
        candidates = self._spawn_order[:allowed_count]
        if self._center_spawn_index not in candidates:
            candidates = np.concatenate([candidates, np.asarray([self._center_spawn_index], dtype=np.int64)])
        return int(candidates[int(self._rng.integers(0, len(candidates)))])

    def _update_curriculum_progress(self) -> None:
        if not self.cfg_obj.terrain_curriculum:
            self._curriculum_progress = 1.0
            return
        self._curriculum_progress = float(np.clip(self._total_policy_steps / max(1, self.cfg_obj.terrain_curriculum_steps), 0.0, 1.0))

    def _apply_command_curriculum(self) -> None:
        ranges = self.cfg_obj.command.ranges
        for stage in self.cfg_obj.command_curriculum:
            if self._total_policy_steps < stage.step:
                continue
            if stage.lin_vel_x is not None:
                ranges.lin_vel_x = stage.lin_vel_x
            if stage.lin_vel_y is not None:
                ranges.lin_vel_y = stage.lin_vel_y
            if stage.ang_vel_z is not None:
                ranges.ang_vel_z = stage.ang_vel_z

    def _reset_all(self):
        obs_list = []
        critic_list = []
        for env_id in range(self.num_envs):
            self._reset_env(env_id, reason=0)
            state = self._runtime_state(env_id)
            obs = self._actor_obs(env_id, state, corrupt=True)
            obs_list.append(obs)
            critic_list.append(self._critic_obs_for(env_id, state, obs))
        torch = self.torch
        return (
            torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=self.device),
            torch.as_tensor(np.stack(critic_list), dtype=torch.float32, device=self.device),
        )

    def _reset_env(self, env_id: int, *, reason: int) -> None:
        self.context.reset_batch_env(env_id)
        spawn_index = self._sample_spawn_index()
        spawn = self._spawn_origins[spawn_index].copy()
        spawn[:2] += self._rng.uniform(-self.cfg_obj.spawn_jitter, self.cfg_obj.spawn_jitter, 2)
        terrain_height = self._terrain_height(spawn[0], spawn[1])
        base_z = max(spawn[2], terrain_height) + self.cfg_obj.base_clearance
        yaw = float(self._rng.uniform(-math.pi, math.pi))
        self.context.reset_batch_link_state(
            env_id,
            self.cfg_obj.robot_name,
            self.cfg_obj.base_link,
            [float(spawn[0]), float(spawn[1]), float(base_z)],
            _quat_from_yaw(yaw).tolist(),
            self._rng.uniform(-0.05, 0.05, 3).tolist(),
            self._rng.uniform(-0.05, 0.05, 3).tolist(),
        )
        for joint_name, default_pos in zip(self.joint_names, self.default_joint_pos, strict=True):
            self.context.reset_batch_joint_state(
                env_id,
                self.cfg_obj.robot_name,
                joint_name,
                float(default_pos + self._rng.uniform(-0.05, 0.05)),
                float(self._rng.uniform(-0.05, 0.05)),
            )
            self.context.set_batch_joint_position_target(env_id, self.cfg_obj.robot_name, joint_name, float(default_pos))
        self._spawn_indices[env_id] = spawn_index
        self._terrain_levels[env_id] = float(self._spawn_levels[spawn_index])
        self._episode_start_xy[env_id] = spawn[:2]
        self.episode_length_buf[env_id] = 0
        self._episode_returns[env_id] = 0.0
        self._previous_actions[env_id] = 0.0
        self._last_actions[env_id] = 0.0
        self._last_foot_contact[env_id] = 0.0
        self._foot_air_time[env_id] = 0.0
        self._foot_peak_height[env_id] = 0.0
        self._foot_velocities[env_id] = 0.0
        self._previous_foot_positions[env_id] = 0.0
        self._reset_reasons[env_id] = reason
        self.command_manager.reset(np.asarray([env_id], dtype=np.int64))

    def _apply_action(self, env_id: int, action: np.ndarray) -> None:
        target_pos = self.default_joint_pos + self.action_scale * action
        for joint_name, target in zip(self.joint_names, target_pos, strict=True):
            self.context.set_batch_joint_position_target(env_id, self.cfg_obj.robot_name, joint_name, float(target))

    def _runtime_state(self, env_id: int) -> VelocityRuntimeState:
        runtime = self.context.get_batch_runtime_state(env_id)
        robot = next((robot for robot in runtime.get("robots", []) if robot.get("name") == self.cfg_obj.robot_name), None)
        if robot is None:
            raise RuntimeError(f"Gobot runtime state has no robot {self.cfg_obj.robot_name!r}")
        links = {str(link.get("name") or link.get("link_name")): link for link in robot.get("links", [])}
        base = links.get(self.cfg_obj.base_link)
        if base is None:
            raise RuntimeError(f"Gobot runtime state has no base link {self.cfg_obj.base_link!r}")
        return VelocityRuntimeState(
            robot=robot,
            base=base,
            joints={str(joint.get("name")): joint for joint in robot.get("joints", [])},
            links=links,
            sensors={str(sensor.get("name") or sensor.get("sensor_name")): sensor for sensor in robot.get("sensors", [])},
            contacts=runtime.get("contacts", []),
        )

    def _actor_obs(self, env_id: int, state: VelocityRuntimeState, *, corrupt: bool) -> np.ndarray:
        base_quat = _quat(state.base)
        base_lin_vel = _rotate_vec_by_quat_inv(_as_vec(state.base.get("linear_velocity"), 3), base_quat)
        base_ang_vel = _rotate_vec_by_quat_inv(_as_vec(state.base.get("angular_velocity"), 3), base_quat)
        projected_gravity = _rotate_vec_by_quat_inv(np.array([0.0, 0.0, -1.0], dtype=np.float32), base_quat)
        joint_pos, joint_vel = self._joint_values(state)
        joint_pos_rel = joint_pos - self.default_joint_pos
        height_scan = self._height_scan(state)
        parts = [
            self._with_noise("base_lin_vel", base_lin_vel, corrupt),
            self._with_noise("base_ang_vel", base_ang_vel, corrupt),
            self._with_noise("projected_gravity", projected_gravity, corrupt),
            self._with_noise("joint_pos", joint_pos_rel, corrupt),
            self._with_noise("joint_vel", joint_vel, corrupt),
            self._last_actions[env_id],
            self.command_manager.command_b[env_id],
            self._with_noise("height_scan", height_scan, corrupt) / max(self.cfg_obj.observations.terrain_scan_max_distance, 1.0e-6),
        ]
        return np.concatenate([np.asarray(part, dtype=np.float32).reshape(-1) for part in parts])

    def _critic_obs_for(self, env_id: int, state: VelocityRuntimeState, actor_obs: np.ndarray) -> np.ndarray:
        foot_forces = self._foot_contact_forces(state).reshape(-1)
        extra = np.concatenate(
            [
                self._foot_heights(state),
                self._foot_air_time[env_id],
                self._foot_contacts(state),
                np.sign(foot_forces) * np.log1p(np.abs(foot_forces)),
            ]
        ).astype(np.float32)
        return np.concatenate([actor_obs.astype(np.float32), extra])

    def _with_noise(self, name: str, values: np.ndarray, corrupt: bool) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if not corrupt or not self.cfg_obj.observations.actor_noise:
            return values
        noise_range = self.cfg_obj.observations.actor_noise_ranges.get(name)
        if noise_range is None:
            return values
        return values + self._rng.uniform(noise_range[0], noise_range[1], size=values.shape).astype(np.float32)

    def _joint_values(self, state: VelocityRuntimeState) -> tuple[np.ndarray, np.ndarray]:
        pos = np.zeros(self.num_actions, dtype=np.float32)
        vel = np.zeros(self.num_actions, dtype=np.float32)
        for index, joint_name in enumerate(self.joint_names):
            joint = state.joints.get(joint_name, {})
            pos[index] = float(joint.get("position", self.default_joint_pos[index]))
            vel[index] = float(joint.get("velocity", 0.0))
        return pos, vel

    def _height_scan(self, state: VelocityRuntimeState) -> np.ndarray:
        sensor_name = self.cfg_obj.observations.height_scan_sensor
        if sensor_name is None or self._height_scan_dim == 0:
            return np.zeros(0, dtype=np.float32)
        sensor = state.sensors.get(sensor_name)
        if sensor is None:
            return np.zeros(self._height_scan_dim, dtype=np.float32)
        frame_z = _as_vec(sensor.get("global_transform", {}).get("position"), 3)[2]
        heights: list[float] = []
        for hit in sensor.get("hits", []):
            if not hit.get("hit", False):
                heights.append(self.cfg_obj.observations.terrain_scan_max_distance)
                continue
            point_z = _as_vec(hit.get("point"), 3)[2]
            heights.append(float(frame_z - point_z))
        if len(heights) != self._height_scan_dim:
            values = np.asarray(sensor.get("values", []), dtype=np.float32)
            if values.size == self._height_scan_dim:
                return values
            return np.zeros(self._height_scan_dim, dtype=np.float32)
        return np.asarray(heights, dtype=np.float32)

    def _foot_heights(self, state: VelocityRuntimeState) -> np.ndarray:
        values = []
        for foot in self.cfg_obj.foot_names:
            sensor = state.sensors.get(f"{foot}_foot_height_scan")
            if sensor is None:
                values.append(0.0)
                continue
            sensor_values = np.asarray(sensor.get("values", []), dtype=np.float32)
            values.append(float(sensor_values[0]) if sensor_values.size else 0.0)
        return np.asarray(values, dtype=np.float32)

    def _foot_contacts(self, state: VelocityRuntimeState) -> np.ndarray:
        values = []
        for foot in self.cfg_obj.foot_names:
            sensor = state.sensors.get(f"{foot}_foot_contact")
            sensor_values = np.asarray(sensor.get("values", []) if sensor is not None else [], dtype=np.float32)
            values.append(1.0 if sensor_values.size and float(sensor_values[0]) > 1.0e-5 else 0.0)
        return np.asarray(values, dtype=np.float32)

    def _foot_positions(self, state: VelocityRuntimeState) -> np.ndarray:
        values = []
        for foot, link_name in zip(self.cfg_obj.foot_names, self.cfg_obj.foot_link_names, strict=True):
            sensor = state.sensors.get(f"{foot}_foot_height_scan")
            if sensor is not None:
                values.append(_as_vec(sensor.get("global_transform", {}).get("position"), 3))
                continue
            link = state.links.get(link_name, {})
            values.append(_as_vec(link.get("global_transform", {}).get("position"), 3))
        return np.asarray(values, dtype=np.float32)

    def _foot_contact_forces(self, state: VelocityRuntimeState) -> np.ndarray:
        foot_positions = self._foot_positions(state)
        forces = np.zeros((self._foot_count, 3), dtype=np.float32)
        for contact in state.contacts:
            position = _as_vec(contact.get("position"), 3)
            if foot_positions.size == 0:
                continue
            index = int(np.argmin(np.linalg.norm(foot_positions - position.reshape(1, 3), axis=1)))
            if np.linalg.norm(foot_positions[index] - position) > 0.18:
                continue
            forces[index] += _as_vec(contact.get("force"), 3)
        return forces

    def _update_foot_history(self, env_id: int, state: VelocityRuntimeState) -> None:
        positions = self._foot_positions(state)
        if not np.any(self._previous_foot_positions[env_id]):
            self._previous_foot_positions[env_id] = positions
        self._foot_velocities[env_id] = (positions - self._previous_foot_positions[env_id]) / max(self.step_dt, 1.0e-6)
        self._previous_foot_positions[env_id] = positions
        contacts = self._foot_contacts(state)
        heights = self._foot_heights(state)
        in_air = contacts <= 0.0
        self._foot_air_time[env_id] = np.where(in_air, self._foot_air_time[env_id] + self.step_dt, 0.0)
        self._foot_peak_height[env_id] = np.where(in_air, np.maximum(self._foot_peak_height[env_id], heights), self._foot_peak_height[env_id])

    def _compute_reward(self, env_id: int, state: VelocityRuntimeState, action: np.ndarray) -> tuple[float, dict[str, float]]:
        reward_cfg = self.cfg_obj.rewards
        command = self.command_manager.command_b[env_id]
        base_quat = _quat(state.base)
        base_lin_vel_b = _rotate_vec_by_quat_inv(_as_vec(state.base.get("linear_velocity"), 3), base_quat)
        base_ang_vel_b = _rotate_vec_by_quat_inv(_as_vec(state.base.get("angular_velocity"), 3), base_quat)
        joint_pos, joint_vel = self._joint_values(state)
        foot_contacts = self._foot_contacts(state)
        foot_heights = self._foot_heights(state)

        lin_error = float(np.sum(np.square(command[:2] - base_lin_vel_b[:2])) + base_lin_vel_b[2] ** 2)
        ang_error = float((command[2] - base_ang_vel_b[2]) ** 2 + np.sum(np.square(base_ang_vel_b[:2])))
        projected_gravity = _rotate_vec_by_quat_inv(np.array([0.0, 0.0, -1.0], dtype=np.float32), base_quat)
        upright_error = float(np.sum(np.square(projected_gravity[:2])))
        action_rate = float(np.sum(np.square(action - self._previous_actions[env_id])))
        command_speed = float(np.linalg.norm(command[:2]) + abs(command[2]))

        pose_std = self._pose_std(command_speed)
        pose_error = float(np.mean(np.square(joint_pos - self.default_joint_pos) / np.square(np.maximum(pose_std, 1.0e-6))))
        foot_clearance_cost = float(np.sum(np.abs(foot_heights - reward_cfg.foot_target_height) * np.linalg.norm(self._foot_velocities[env_id, :, :2], axis=1)))
        active = 1.0 if command_speed > reward_cfg.command_threshold else 0.0
        foot_slip = self._foot_slip_cost(env_id, foot_contacts) * active
        forces = self._foot_contact_forces(state)
        first_contact = (foot_contacts > 0.0) & (self._last_foot_contact[env_id] <= 0.0)
        landing_force = float(np.sum(np.linalg.norm(forces, axis=1) * first_contact.astype(np.float32))) * active
        swing_error = np.square(self._foot_peak_height[env_id] / max(reward_cfg.foot_target_height, 1.0e-6) - 1.0)
        swing_cost = float(np.sum(swing_error * first_contact.astype(np.float32))) * active
        self._foot_peak_height[env_id] = np.where(first_contact, 0.0, self._foot_peak_height[env_id])
        self._last_foot_contact[env_id] = foot_contacts

        breakdown = {
            "track_linear_velocity": reward_cfg.track_linear_velocity * math.exp(-lin_error / reward_cfg.lin_vel_std**2),
            "track_angular_velocity": reward_cfg.track_angular_velocity * math.exp(-ang_error / reward_cfg.ang_vel_std**2),
            "upright": reward_cfg.upright * math.exp(-upright_error / reward_cfg.upright_std**2),
            "pose": reward_cfg.pose * math.exp(-pose_error),
            "body_ang_vel": reward_cfg.body_ang_vel * float(np.sum(np.square(base_ang_vel_b[:2]))),
            "dof_pos_limits": reward_cfg.dof_pos_limits * self._joint_limit_cost(state),
            "action_rate_l2": reward_cfg.action_rate_l2 * action_rate,
            "air_time": reward_cfg.air_time * float(np.sum((self._foot_air_time[env_id] > 0.05) & (self._foot_air_time[env_id] < 0.5))) * active,
            "foot_clearance": reward_cfg.foot_clearance * foot_clearance_cost * active,
            "foot_swing_height": reward_cfg.foot_swing_height * swing_cost,
            "foot_slip": reward_cfg.foot_slip * foot_slip,
            "soft_landing": reward_cfg.soft_landing * landing_force,
        }
        return float(sum(breakdown.values()) * self.step_dt), {key: float(value) for key, value in breakdown.items()}

    def _pose_std(self, command_speed: float) -> np.ndarray:
        cfg = self.cfg_obj.rewards
        if command_speed < cfg.pose_walking_threshold:
            table = cfg.pose_std_standing
        elif command_speed < cfg.pose_running_threshold:
            table = cfg.pose_std_walking
        else:
            table = cfg.pose_std_running
        values = np.full(self.num_actions, 0.3, dtype=np.float32)
        for index, joint_name in enumerate(self.joint_names):
            for pattern, value in table.items():
                if re.fullmatch(pattern, joint_name):
                    values[index] = float(value)
                    break
        return values

    def _joint_limit_cost(self, state: VelocityRuntimeState) -> float:
        cost = 0.0
        for joint_name in self.joint_names:
            joint = state.joints.get(joint_name, {})
            pos = float(joint.get("position", 0.0))
            lower = float(joint.get("lower_limit", -math.inf))
            upper = float(joint.get("upper_limit", math.inf))
            if math.isfinite(lower):
                cost += max(0.0, lower - pos)
            if math.isfinite(upper):
                cost += max(0.0, pos - upper)
        return cost

    def _foot_slip_cost(self, env_id: int, foot_contacts: np.ndarray) -> float:
        vel_xy = np.linalg.norm(self._foot_velocities[env_id, :, :2], axis=1)
        return float(np.sum(np.square(vel_xy) * foot_contacts))

    def _base_clearance(self, state: VelocityRuntimeState) -> float:
        position = _as_vec(state.base.get("global_transform", {}).get("position"), 3)
        return float(position[2] - self._terrain_height(position[0], position[1]))

    def _terminated(self, state: VelocityRuntimeState) -> bool:
        if self.cfg_obj.terrain_type == "rough":
            return self._base_clearance(state) < self.cfg_obj.min_base_clearance
        roll, pitch = _quat_to_rp(_quat(state.base))
        return bool(self._base_clearance(state) < self.cfg_obj.min_base_clearance or abs(roll) > math.radians(70.0) or abs(pitch) > math.radians(70.0))


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


def _json_vec(value: object, size: int) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("matrix_data", {}).get("storage", [])
    vector = np.zeros(size, dtype=np.float64)
    array = np.asarray(value if value is not None else [], dtype=np.float64).reshape(-1)
    vector[: min(size, array.size)] = array[:size]
    return vector


def _find_node_by_name(node: Any, name: str):
    if node is None:
        return None
    if getattr(node, "name", None) == name:
        return node
    for child in getattr(node, "children", []):
        found = _find_node_by_name(child, name)
        if found is not None:
            return found
    return None


def _as_vec(value: object, size: int) -> np.ndarray:
    if isinstance(value, Mapping) and "position" in value:
        value = value.get("position")
    array = np.asarray(value if value is not None else [], dtype=np.float32).reshape(-1)
    out = np.zeros(size, dtype=np.float32)
    out[: min(size, array.size)] = array[:size]
    return out


def _quat(base_or_quat: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
    if isinstance(base_or_quat, Mapping):
        transform = base_or_quat.get("global_transform", {})
        value = transform.get("quaternion", [1.0, 0.0, 0.0, 0.0])
    else:
        value = base_or_quat
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size < 4:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return array[:4]


def _quat_to_rp(q: np.ndarray | Sequence[float]) -> tuple[float, float]:
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return float(roll), float(pitch)


def _quat_to_yaw(q: np.ndarray | Sequence[float]) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _quat_from_yaw(yaw: float) -> np.ndarray:
    half_yaw = yaw * 0.5
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float32)


def _rotate_vec_by_quat_inv(v: np.ndarray, q: np.ndarray | Sequence[float]) -> np.ndarray:
    w, x, y, z = q
    return _quat_rotate(v, np.array([w, -x, -y, -z], dtype=np.float32))


def _quat_rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    xyz = np.array([x, y, z], dtype=np.float32)
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def _wrap_to_pi(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


__all__ = [
    "GobotVelocityEnv",
    "TerrainSampler",
    "UniformVelocityCommandCfg",
    "UniformVelocityCommandRanges",
    "VelocityObservationCfg",
    "VelocityRewardCfg",
    "VelocityStage",
    "VelocityTaskCfg",
    "rsl_rl_train_cfg",
    "unitree_g1_flat_velocity_cfg",
    "unitree_g1_rough_velocity_cfg",
    "unitree_go1_flat_velocity_cfg",
    "unitree_go1_rough_velocity_cfg",
    "velocity_task_cfg",
]
