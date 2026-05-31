"""Gobot-backed Go1 velocity-tracking environment for rsl_rl.

This intentionally uses Gobot's scene and simulation APIs instead of importing
MuJoCo directly. The editable source of truth is ``examples/go1/go1_scene.jscn``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

import gobot
from rsl_rl.env import VecEnv


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = "res://go1_scene.jscn"
ROBOT = "go1"
BASE_LINK = "trunk"

JOINT_NAMES = [
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
]

DEFAULT_POS = np.array(
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
    dtype=np.float64,
)

ACTION_SCALE = 0.35
KP = 40.0
KD = 1.0
DECIMATION = 10
FIXED_TIME_STEP = 0.002
BASE_CLEARANCE = 0.32
MIN_BASE_CLEARANCE = 0.16
TERRAIN_CURRICULUM_STEPS = 21600
HEIGHT_SCAN_POINTS = tuple((x, y) for x in (0.3, 0.6, 0.9, 1.2, 1.5) for y in (-0.45, 0.0, 0.45))
TERRAIN_SCAN_SENSOR = "terrain_scan"
FOOT_NAMES = ("FR", "FL", "RR", "RL")
SPAWN_DIFFICULTY_RADIUS = 0.85


class TerrainSampler:
    """Lightweight height queries for the authored Terrain3D scene."""

    def __init__(self, scene_path: Path, grid_resolution: float = 0.08) -> None:
        self._boxes: list[dict] = []
        self._heightfields: list[dict] = []
        self._grid_resolution = float(grid_resolution)
        self._grid_origin = np.zeros(2, dtype=np.float64)
        self._grid_heights: np.ndarray | None = None
        if not scene_path.exists():
            return
        data = json.loads(scene_path.read_text(encoding="utf-8"))
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
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding
        xs = np.arange(min_x, max_x + self._grid_resolution * 0.5, self._grid_resolution, dtype=np.float64)
        ys = np.arange(min_y, max_y + self._grid_resolution * 0.5, self._grid_resolution, dtype=np.float64)
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
        bounds: list[tuple[float, float, float, float]] = []
        for box in self._boxes:
            center = box["center"]
            size = box["size"]
            half_x = size[0] * 0.5
            half_y = size[1] * 0.5
            bounds.append((center[0] - half_x, center[1] - half_y, center[0] + half_x, center[1] + half_y))
        for heightfield in self._heightfields:
            center = heightfield["center"]
            size = heightfield["size"]
            half_x = size[0] * 0.5
            half_y = size[1] * 0.5
            bounds.append((center[0] - half_x, center[1] - half_y, center[0] + half_x, center[1] + half_y))
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
        h0 = h00 * (1.0 - fu) + h10 * fu
        h1 = h01 * (1.0 - fu) + h11 * fu
        return float(h0 * (1.0 - fv) + h1 * fv)

    @staticmethod
    def _box_height(box: dict, x: float, y: float) -> float | None:
        center = box["center"]
        size = box["size"]
        rotation = box["rotation"]
        local_x = x - center[0]
        local_y = y - center[1]
        yaw = -np.deg2rad(rotation[2])
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rx = cos_yaw * local_x - sin_yaw * local_y
        ry = sin_yaw * local_x + cos_yaw * local_y
        if abs(rx) > size[0] * 0.5 or abs(ry) > size[1] * 0.5:
            return None
        return float(center[2] + size[2] * 0.5)

    @staticmethod
    def _box_height_grid(box: dict, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray | None:
        center = box["center"]
        size = box["size"]
        rotation = box["rotation"]
        yaw = -np.deg2rad(rotation[2])
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        local_x = grid_x - center[0]
        local_y = grid_y - center[1]
        rx = cos_yaw * local_x - sin_yaw * local_y
        ry = sin_yaw * local_x + cos_yaw * local_y
        mask = (np.abs(rx) <= size[0] * 0.5) & (np.abs(ry) <= size[1] * 0.5)
        if not np.any(mask):
            return None
        height = center[2] + size[2] * 0.5
        return np.where(mask, height, -np.inf)

    @staticmethod
    def _heightfield_height(heightfield: dict, x: float, y: float) -> float | None:
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
        h0 = h00 * (1.0 - fu) + h10 * fu
        h1 = h01 * (1.0 - fu) + h11 * fu
        return float(center[2] + heightfield["z_offset"] + h0 * (1.0 - fv) + h1 * fv)

    @staticmethod
    def _heightfield_height_grid(heightfield: dict, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray | None:
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
        h0 = h00 * (1.0 - fu) + h10 * fu
        h1 = h01 * (1.0 - fu) + h11 * fu
        sampled = center[2] + heightfield["z_offset"] + h0 * (1.0 - fv) + h1 * fv
        return np.where(mask, sampled, -np.inf)


class Go1VecEnv(VecEnv):
    """Vectorized Go1 task backed by one Gobot MuJoCo model and many runtime states."""

    num_obs: int = 48

    def __init__(
        self,
        num_envs: int = 64,
        max_episode_length: int = 1000,
        device: str = "cpu",
        project_path: str | Path = EXAMPLE_ROOT,
        scene_path: str = SCENE_PATH,
        decimation: int = DECIMATION,
        seed: int = 42,
        terrain_curriculum: bool = True,
        terrain_curriculum_steps: int = TERRAIN_CURRICULUM_STEPS,
        spawn_jitter: float = 0.35,
        base_clearance: float = BASE_CLEARANCE,
        height_scan: bool = True,
    ) -> None:
        self.num_envs = int(num_envs)
        self.max_episode_length = int(max_episode_length)
        self.device = device
        self.project_path = Path(project_path)
        self.scene_path = scene_path
        self.decimation = int(decimation)
        self.num_actions = len(JOINT_NAMES)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.terrain_curriculum = bool(terrain_curriculum)
        self.terrain_curriculum_steps = max(1, int(terrain_curriculum_steps))
        self.spawn_jitter = float(spawn_jitter)
        self.base_clearance = float(base_clearance)
        self._height_scan_points = np.asarray(HEIGHT_SCAN_POINTS if height_scan else (), dtype=np.float64)
        self.num_obs = 48 + int(self._height_scan_points.shape[0])
        self.num_privileged_obs = self.num_obs + 11
        self._total_policy_steps = 0
        self._curriculum_progress = 0.0

        self._rng = np.random.default_rng(seed)
        self._cmds = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._last_actions = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)
        self._spawn_indices = np.zeros(self.num_envs, dtype=np.int64)
        self._terrain_levels = np.zeros(self.num_envs, dtype=np.float32)

        self.context = gobot.app.context()
        self.context.set_project_path(str(self.project_path))
        self.context.load_scene(self.scene_path)
        self._spawn_origins = self._load_spawn_origins()
        self._center_spawn_index = int(np.argmin(np.linalg.norm(self._spawn_origins[:, :2], axis=1)))
        self._terrain_sampler = TerrainSampler(self.project_path / "terrain_scene.jscn")
        self._spawn_difficulties = self._spawn_difficulty_scores(self._spawn_origins)
        self._spawn_order = np.argsort(self._spawn_difficulties, kind="stable")
        max_difficulty = max(float(np.max(self._spawn_difficulties)), 1.0e-6)
        self._spawn_levels = np.clip(self._spawn_difficulties / max_difficulty, 0.0, 1.0).astype(np.float32)
        self.context.fixed_time_step = FIXED_TIME_STEP
        self.context.set_default_joint_gains(
            {
                "position_stiffness": KP,
                "velocity_damping": KD,
                "integral_gain": 0.0,
                "integral_limit": 0.0,
            }
        )
        self._configure_robot_drives()
        self.context.build_world(gobot.PhysicsBackendType.MuJoCoCpu)
        self.context.configure_batch_world(self.num_envs)

        self.cfg = {
            "num_envs": self.num_envs,
            "max_episode_length": self.max_episode_length,
            "project_path": str(self.project_path),
            "scene_path": self.scene_path,
            "backend": "MuJoCoCpu",
            "robot": ROBOT,
            "base_link": BASE_LINK,
            "joint_names": JOINT_NAMES,
            "action_scale": ACTION_SCALE,
            "kp": KP,
            "kd": KD,
            "decimation": self.decimation,
            "fixed_time_step": FIXED_TIME_STEP,
            "seed": seed,
            "terrain_curriculum": self.terrain_curriculum,
            "terrain_curriculum_steps": self.terrain_curriculum_steps,
            "spawn_jitter": self.spawn_jitter,
            "base_clearance": self.base_clearance,
            "height_scan": bool(height_scan),
            "height_scan_points": self._height_scan_points.tolist(),
            "num_obs": self.num_obs,
        }
        self._obs, self._critic_obs = self._reset_all()

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {"policy": self._obs.clone(), "critic": self._critic_obs.clone()},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def reset(self, seed: int | None = None) -> TensorDict:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.episode_length_buf.zero_()
        self._obs, self._critic_obs = self._reset_all()
        return self.get_observations()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        self._total_policy_steps += 1
        self._update_curriculum_progress()
        actions_np = actions.detach().cpu().numpy().reshape(self.num_envs, self.num_actions)
        actions_np = np.clip(actions_np, -1.0, 1.0)

        obs_list = []
        critic_obs_list = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        time_outs = np.zeros(self.num_envs, dtype=bool)

        base_heights = np.zeros(self.num_envs, dtype=np.float32)
        base_clearances = np.zeros(self.num_envs, dtype=np.float32)
        terrain_heights = np.zeros(self.num_envs, dtype=np.float32)
        velocity_errors = np.zeros(self.num_envs, dtype=np.float32)
        foot_clearances = np.zeros(self.num_envs, dtype=np.float32)
        foot_contact_ratios = np.zeros(self.num_envs, dtype=np.float32)
        for env_id in range(self.num_envs):
            action = actions_np[env_id]
            self._apply_action(env_id, action)
            self.context.step_batch_env(env_id, self.decimation)

            state = self._robot_state(env_id)
            rewards[env_id] = self._compute_reward(state, self._cmds[env_id], action)
            terminated = self._terminated(state)

            self.episode_length_buf[env_id] += 1
            timed_out = bool(self.episode_length_buf[env_id] >= self.max_episode_length)
            done = terminated or timed_out
            dones[env_id] = done
            time_outs[env_id] = timed_out and not terminated

            self._last_actions[env_id] = action
            if done:
                self._reset_env(env_id)
                self.episode_length_buf[env_id] = 0
                state = self._robot_state(env_id)

            base = self._base_link(state)
            terrain_height = self._terrain_height(base["position"][0], base["position"][1])
            base_heights[env_id] = base["position"][2]
            terrain_heights[env_id] = terrain_height
            base_clearances[env_id] = base["position"][2] - terrain_height
            base_lin_vel = _rotate_vec_by_quat_inv(np.asarray(base["linear_velocity"], dtype=np.float64), base["quaternion"])
            velocity_errors[env_id] = float(np.linalg.norm(base_lin_vel[:2] - self._cmds[env_id, :2]))
            foot_clearance = self._foot_clearances(state)
            foot_contact = self._foot_contacts(state)
            foot_clearances[env_id] = float(np.mean(foot_clearance)) if foot_clearance.size else 0.0
            foot_contact_ratios[env_id] = float(np.mean(foot_contact)) if foot_contact.size else 0.0
            obs = self._get_obs(state, self._cmds[env_id], self._last_actions[env_id])
            obs_list.append(obs)
            critic_obs_list.append(self._get_critic_obs(state, obs, env_id))

        self._obs = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)
        self._critic_obs = torch.as_tensor(np.stack(critic_obs_list), dtype=torch.float32, device=self.device)
        obs_td = TensorDict(
            {"policy": self._obs.clone(), "critic": self._critic_obs.clone()},
            batch_size=[self.num_envs],
            device=self.device,
        )
        extras = {
            "time_outs": torch.as_tensor(time_outs, dtype=torch.bool, device=self.device),
            "log": {
                "/go1/command_vx": torch.as_tensor(float(np.mean(self._cmds[:, 0])), device=self.device),
                "/go1/base_height": torch.as_tensor(float(np.mean(base_heights)), device=self.device),
                "/go1/base_clearance": torch.as_tensor(float(np.mean(base_clearances)), device=self.device),
                "/go1/terrain_height": torch.as_tensor(float(np.mean(terrain_heights)), device=self.device),
                "/go1/terrain_level": torch.as_tensor(float(np.mean(self._terrain_levels)), device=self.device),
                "/go1/velocity_error": torch.as_tensor(float(np.mean(velocity_errors)), device=self.device),
                "/go1/foot_clearance": torch.as_tensor(float(np.mean(foot_clearances)), device=self.device),
                "/go1/foot_contact_ratio": torch.as_tensor(float(np.mean(foot_contact_ratios)), device=self.device),
            },
        }
        return (
            obs_td,
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(dones, dtype=torch.bool, device=self.device),
            extras,
        )

    def close(self) -> None:
        self.context.clear_world()

    def _apply_action(self, env_id: int, action: np.ndarray) -> None:
        target_pos = DEFAULT_POS + ACTION_SCALE * action
        for joint_name, target in zip(JOINT_NAMES, target_pos, strict=True):
            self.context.set_batch_joint_position_target(env_id, ROBOT, joint_name, float(target))

    def _compute_reward(self, state: dict, cmd: np.ndarray, action: np.ndarray) -> float:
        base = self._base_link(state)
        world_lin_vel = np.asarray(base["linear_velocity"], dtype=np.float64)
        base_lin_vel = _rotate_vec_by_quat_inv(world_lin_vel, base["quaternion"])
        base_ang_vel = np.asarray(base["angular_velocity"], dtype=np.float64)
        vx_cmd, vy_cmd, yaw_cmd = cmd

        lin_vel_err = (base_lin_vel[0] - vx_cmd) ** 2 + (base_lin_vel[1] - vy_cmd) ** 2
        ang_vel_err = (base_ang_vel[2] - yaw_cmd) ** 2
        command_speed = float(np.linalg.norm(cmd[:2]))
        measured_speed = float(np.linalg.norm(base_lin_vel[:2]))
        r_lin = 2.0 * np.exp(-lin_vel_err / 0.08)
        r_ang = 0.25 * np.exp(-ang_vel_err / 0.25)
        r_progress = 0.5 * float(np.clip(vx_cmd * base_lin_vel[0] + vy_cmd * base_lin_vel[1], -1.0, 1.0))
        r_stall = -1.2 * max(0.0, command_speed * 0.6 - measured_speed)
        r_z_vel = -1.5 * world_lin_vel[2] ** 2
        r_ang_xy = -0.05 * float(np.sum(base_ang_vel[:2] ** 2))
        r_action = -0.002 * float(np.sum(action**2))
        joint_vel = np.asarray(self._joint_values(state)[1], dtype=np.float64)
        r_joint_vel = -0.001 * float(np.sum(joint_vel**2))
        clearance = base["position"][2] - self._terrain_height(base["position"][0], base["position"][1])
        r_height = -2.0 * max(0.0, 0.27 - clearance)
        foot_clearance = self._foot_clearances(state)
        foot_contact = self._foot_contacts(state)
        r_foot_clearance = 0.0
        r_foot_slip = 0.0
        if foot_clearance.size:
            target_clearance = 0.08
            swing = 1.0 - foot_contact if foot_contact.size == foot_clearance.size else np.ones_like(foot_clearance)
            r_foot_clearance = -0.08 * float(np.mean(np.square(np.clip(target_clearance - foot_clearance, 0.0, 1.0)) * swing))
            foot_velocity = self._foot_linear_speeds(state)
            if foot_velocity.size == foot_contact.size:
                r_foot_slip = -0.02 * float(np.mean(foot_velocity * foot_contact))
        return float(
            r_lin
            + r_ang
            + r_progress
            + r_stall
            + r_z_vel
            + r_ang_xy
            + r_action
            + r_joint_vel
            + r_height
            + r_foot_clearance
            + r_foot_slip
        )

    def _terminated(self, state: dict) -> bool:
        base = self._base_link(state)
        roll, pitch = _quat_to_rp(base["quaternion"])
        clearance = base["position"][2] - self._terrain_height(base["position"][0], base["position"][1])
        return bool(clearance < MIN_BASE_CLEARANCE or abs(roll) > 0.8 or abs(pitch) > 0.8)

    def _get_obs(self, state: dict, cmd: np.ndarray, last_action: np.ndarray) -> np.ndarray:
        base = self._base_link(state)
        quat = base["quaternion"]
        base_lin_vel = _rotate_vec_by_quat_inv(np.asarray(base["linear_velocity"], dtype=np.float64), quat)
        base_ang_vel = np.asarray(base["angular_velocity"], dtype=np.float64)
        projected_gravity = _rotate_vec_by_quat_inv(np.array([0.0, 0.0, -1.0], dtype=np.float64), quat)
        joint_pos, joint_vel = self._joint_values(state)
        joint_pos_rel = np.asarray(joint_pos, dtype=np.float32) - DEFAULT_POS.astype(np.float32)
        height_scan = self._height_scan(state, base["position"], quat)

        return np.concatenate(
            [
                base_lin_vel.astype(np.float32),
                base_ang_vel.astype(np.float32),
                projected_gravity.astype(np.float32),
                cmd.astype(np.float32),
                joint_pos_rel.astype(np.float32),
                np.asarray(joint_vel, dtype=np.float32),
                last_action.astype(np.float32),
                height_scan.astype(np.float32),
            ]
        )

    def _get_critic_obs(self, state: dict, actor_obs: np.ndarray, env_id: int) -> np.ndarray:
        base = self._base_link(state)
        base_clearance = base["position"][2] - self._terrain_height(base["position"][0], base["position"][1])
        progress = float(self.episode_length_buf[env_id].item()) / max(float(self.max_episode_length), 1.0)
        extra = np.concatenate(
            [
                self._foot_clearances(state).astype(np.float32),
                self._foot_contacts(state).astype(np.float32),
                np.asarray(
                    [
                        float(self._terrain_levels[env_id]),
                        float(self._curriculum_progress),
                        float(base_clearance),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return np.concatenate([actor_obs.astype(np.float32), extra.astype(np.float32)])

    def _reset_env(self, env_id: int) -> None:
        self.context.reset_batch_env(env_id)
        spawn_index = self._sample_spawn_index()
        spawn = self._spawn_origins[spawn_index].copy()
        jitter = self._rng.uniform(-self.spawn_jitter, self.spawn_jitter, 2)
        spawn[:2] += jitter
        terrain_height = self._terrain_height(spawn[0], spawn[1])
        base_z = max(spawn[2], terrain_height) + self.base_clearance
        yaw = float(self._rng.uniform(-np.pi, np.pi))
        self._spawn_indices[env_id] = spawn_index
        self._terrain_levels[env_id] = self._terrain_level(spawn_index)
        self.context.reset_batch_link_state(
            env_id,
            ROBOT,
            BASE_LINK,
            [float(spawn[0]), float(spawn[1]), float(base_z)],
            _quat_from_yaw(yaw).tolist(),
            self._rng.uniform(-0.05, 0.05, 3).tolist(),
            self._rng.uniform(-0.05, 0.05, 3).tolist(),
        )
        for joint_name, default_pos in zip(JOINT_NAMES, DEFAULT_POS, strict=True):
            position = float(default_pos + self._rng.uniform(-0.05, 0.05))
            velocity = float(self._rng.uniform(-0.05, 0.05))
            self.context.reset_batch_joint_state(env_id, ROBOT, joint_name, position, velocity)
            self.context.set_batch_joint_position_target(env_id, ROBOT, joint_name, float(default_pos))

        self._cmds[env_id] = _sample_cmd(self._rng, self._command_scale(self._terrain_levels[env_id]))
        self._last_actions[env_id] = 0.0

    def _reset_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        obs_list = []
        critic_obs_list = []
        for env_id in range(self.num_envs):
            self._reset_env(env_id)
            state = self._robot_state(env_id)
            obs = self._get_obs(state, self._cmds[env_id], self._last_actions[env_id])
            obs_list.append(obs)
            critic_obs_list.append(self._get_critic_obs(state, obs, env_id))
        return (
            torch.as_tensor(
            np.stack(obs_list),
            dtype=torch.float32,
            device=self.device,
            ),
            torch.as_tensor(
                np.stack(critic_obs_list),
                dtype=torch.float32,
                device=self.device,
            ),
        )

    def set_training_progress(self, policy_steps: int) -> None:
        self._total_policy_steps = max(0, int(policy_steps))
        self._update_curriculum_progress()

    def _robot_state(self, env_id: int) -> dict:
        runtime = self.context.get_batch_runtime_state(env_id)
        for robot in runtime.get("robots", []):
            if robot.get("name") == ROBOT:
                return robot
        raise RuntimeError(f"Gobot runtime state has no robot '{ROBOT}'")

    def _base_link(self, state: dict) -> dict:
        for link in state.get("links", []):
            if link.get("name") == BASE_LINK:
                transform = link.get("global_transform", {})
                return {
                    "position": np.asarray(transform.get("position", [0.0, 0.0, 0.0]), dtype=np.float64),
                    "quaternion": np.asarray(transform.get("quaternion", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64),
                    "linear_velocity": np.asarray(link.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float64),
                    "angular_velocity": np.asarray(link.get("angular_velocity", [0.0, 0.0, 0.0]), dtype=np.float64),
                }
        raise RuntimeError(f"Gobot runtime state has no link '{BASE_LINK}'")

    def _joint_values(self, state: dict) -> tuple[list[float], list[float]]:
        joints = {joint.get("name"): joint for joint in state.get("joints", [])}
        joint_pos = []
        joint_vel = []
        for default_pos, joint_name in zip(DEFAULT_POS, JOINT_NAMES, strict=True):
            joint = joints.get(joint_name, {})
            joint_pos.append(float(joint.get("position", default_pos)))
            joint_vel.append(float(joint.get("velocity", 0.0)))
        return joint_pos, joint_vel

    def _sensor_map(self, state: dict) -> dict[str, dict]:
        return {sensor.get("name") or sensor.get("sensor_name"): sensor for sensor in state.get("sensors", [])}

    def _load_spawn_origins(self) -> np.ndarray:
        root = self.context.root
        terrain_world = root.find("terrain_world") if root is not None else None
        terrain = terrain_world.find("terrain") if terrain_world is not None else None
        origins = np.asarray(getattr(terrain, "spawn_origins", []), dtype=np.float64)
        if origins.ndim != 2 or origins.shape[1] != 3 or origins.shape[0] == 0:
            return np.zeros((1, 3), dtype=np.float64)
        return origins

    def _configure_robot_drives(self) -> None:
        root = self.context.root
        robot = root.find(ROBOT) if root is not None else None
        if robot is None:
            robot = _find_node_by_name(root, ROBOT)
        if robot is None:
            return
        for joint_name in JOINT_NAMES:
            joint = robot.find(joint_name) or _find_node_by_name(robot, joint_name)
            if joint is None:
                continue
            joint.drive_mode = gobot.JointDriveMode.Position
            joint.drive_stiffness = KP
            joint.drive_damping = KD
            joint.damping = KD

    def _spawn_difficulty_scores(self, spawn_origins: np.ndarray) -> np.ndarray:
        offsets = np.asarray(
            [(x, y) for x in (-SPAWN_DIFFICULTY_RADIUS, 0.0, SPAWN_DIFFICULTY_RADIUS) for y in (-SPAWN_DIFFICULTY_RADIUS, 0.0, SPAWN_DIFFICULTY_RADIUS)],
            dtype=np.float64,
        )
        scores = np.zeros(spawn_origins.shape[0], dtype=np.float32)
        for index, origin in enumerate(spawn_origins):
            heights = np.asarray(
                [self._terrain_sampler.height_at(origin[0] + dx, origin[1] + dy) for dx, dy in offsets],
                dtype=np.float64,
            )
            scores[index] = float(np.max(heights) - np.min(heights))
        return scores

    def _sample_spawn_index(self) -> int:
        if not self.terrain_curriculum:
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

    def _terrain_level(self, spawn_index: int) -> float:
        return float(self._spawn_levels[spawn_index])

    def _update_curriculum_progress(self) -> None:
        if not self.terrain_curriculum:
            self._curriculum_progress = 1.0
            return
        self._curriculum_progress = float(np.clip(self._total_policy_steps / self.terrain_curriculum_steps, 0.0, 1.0))

    def _command_scale(self, terrain_level: float) -> float:
        if not self.terrain_curriculum:
            return 1.0
        scale = 1.0 - 0.45 * terrain_level * (1.0 - self._curriculum_progress)
        return float(np.clip(scale, 0.35, 1.0))

    def _terrain_height(self, x: float, y: float) -> float:
        return self._terrain_sampler.height_at(float(x), float(y))

    def _height_scan(self, state: dict, base_position: np.ndarray, quat: np.ndarray | list[float]) -> np.ndarray:
        if self._height_scan_points.size == 0:
            return np.zeros(0, dtype=np.float32)
        sensor = self._sensor_map(state).get(TERRAIN_SCAN_SENSOR)
        if sensor is not None:
            values = np.asarray(sensor.get("values", []), dtype=np.float32)
            if values.size == self._height_scan_points.shape[0] and np.all(np.isfinite(values)):
                return np.clip(values, -1.0, 1.0)
        yaw = _quat_to_yaw(quat)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        samples = []
        for local_x, local_y in self._height_scan_points:
            world_x = base_position[0] + cos_yaw * local_x - sin_yaw * local_y
            world_y = base_position[1] + sin_yaw * local_x + cos_yaw * local_y
            samples.append(base_position[2] - self._terrain_height(world_x, world_y))
        return np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)

    def _foot_clearances(self, state: dict) -> np.ndarray:
        sensors = self._sensor_map(state)
        values = []
        for foot in FOOT_NAMES:
            sensor = sensors.get(f"{foot}_foot_height_scan")
            sensor_values = np.asarray(sensor.get("values", []) if sensor is not None else [], dtype=np.float32)
            values.append(float(sensor_values[0]) if sensor_values.size else 0.0)
        return np.asarray(values, dtype=np.float32)

    def _foot_contacts(self, state: dict) -> np.ndarray:
        sensors = self._sensor_map(state)
        values = []
        for foot in FOOT_NAMES:
            sensor = sensors.get(f"{foot}_foot_contact")
            sensor_values = np.asarray(sensor.get("values", []) if sensor is not None else [], dtype=np.float32)
            values.append(1.0 if sensor_values.size and float(sensor_values[0]) > 1.0e-5 else 0.0)
        return np.asarray(values, dtype=np.float32)

    def _foot_linear_speeds(self, state: dict) -> np.ndarray:
        links = {link.get("name") or link.get("link_name"): link for link in state.get("links", [])}
        values = []
        for foot in FOOT_NAMES:
            link = links.get(f"{foot}_calf", {})
            velocity = np.asarray(link.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
            values.append(float(np.linalg.norm(velocity[:2])))
        return np.asarray(values, dtype=np.float32)


def _sample_cmd(rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    vx = rng.uniform(0.25, 0.9)
    if rng.random() < 0.15:
        vx = rng.uniform(-0.20, 0.25)
    cmd = np.array(
        [
            vx,
            rng.uniform(-0.20, 0.20),
            rng.uniform(-0.55, 0.55),
        ],
        dtype=np.float32,
    )
    return cmd * float(scale)


def _json_vec(value: object, size: int) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("matrix_data", {}).get("storage", [])
    vector = np.zeros(size, dtype=np.float64)
    array = np.asarray(value if value is not None else [], dtype=np.float64).reshape(-1)
    vector[: min(size, array.size)] = array[:size]
    return vector


def _find_node_by_name(node, name: str):
    if node is None:
        return None
    if getattr(node, "name", None) == name:
        return node
    for child in getattr(node, "children", []):
        found = _find_node_by_name(child, name)
        if found is not None:
            return found
    return None


def _quat_to_rp(q: np.ndarray | list[float]) -> tuple[float, float]:
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return float(roll), float(pitch)


def _quat_to_yaw(q: np.ndarray | list[float]) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _quat_from_yaw(yaw: float) -> np.ndarray:
    half_yaw = yaw * 0.5
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float64)


def _rotate_vec_by_quat_inv(v: np.ndarray, q: np.ndarray | list[float]) -> np.ndarray:
    w, x, y, z = q
    return _quat_rotate(v, np.array([w, -x, -y, -z], dtype=np.float64))


def _quat_rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    t = 2.0 * np.cross(np.array([x, y, z], dtype=np.float64), v)
    return v + w * t + np.cross(np.array([x, y, z], dtype=np.float64), t)
