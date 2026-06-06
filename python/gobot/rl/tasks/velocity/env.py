"""Gobot runtime environment for velocity locomotion tasks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

import gobot

from .cfg import VelocityTaskCfg, unitree_go1_rough_velocity_cfg
from .command import UniformVelocityCommand
from .math_utils import (
    _as_vec,
    _find_node_by_name,
    _quat,
    _quat_from_yaw,
    _quat_to_rp,
    _rotate_vec_by_quat_inv,
)
from .terrain import TerrainSampler

@dataclass
class VelocityRuntimeState:
    robot: Mapping[str, Any]
    base: Mapping[str, Any]
    joints: Mapping[str, Mapping[str, Any]]
    links: Mapping[str, Mapping[str, Any]]
    sensors: Mapping[str, Mapping[str, Any]]
    contacts: Sequence[Mapping[str, Any]]


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

        self._foot_count = len(self.cfg_obj.foot_names)
        self._contact_history_length = max(1, int(self.cfg_obj.contact_history_length))
        self._last_actions = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)
        self._previous_actions = np.zeros_like(self._last_actions)
        self._last_foot_contact = np.zeros((self.num_envs, self._foot_count), dtype=np.float32)
        self._foot_air_time = np.zeros_like(self._last_foot_contact)
        self._foot_peak_height = np.zeros_like(self._last_foot_contact)
        self._previous_foot_positions = np.zeros((self.num_envs, self._foot_count, 3), dtype=np.float32)
        self._foot_velocities = np.zeros_like(self._previous_foot_positions)
        self._foot_contact_history = np.zeros(
            (self.num_envs, self._contact_history_length, self._foot_count),
            dtype=np.float32,
        )
        self._foot_force_history = np.zeros(
            (self.num_envs, self._contact_history_length, self._foot_count, 3),
            dtype=np.float32,
        )
        self._foot_history_cursor = np.zeros(self.num_envs, dtype=np.int64)
        self._first_contact = np.zeros((self.num_envs, self._foot_count), dtype=np.float32)
        self._landing_force = np.zeros((self.num_envs, self._foot_count), dtype=np.float32)
        self._terrain_normal_error = np.zeros(self.num_envs, dtype=np.float32)
        self._illegal_contact_counts = np.zeros(self.num_envs, dtype=np.float32)
        self._self_collision_counts = np.zeros(self.num_envs, dtype=np.float32)
        self._shank_collision_counts = np.zeros(self.num_envs, dtype=np.float32)
        self._trunk_head_collision_counts = np.zeros(self.num_envs, dtype=np.float32)
        self._encoder_bias = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)
        self._push_time_left = np.zeros(self.num_envs, dtype=np.float32)
        self._push_count = np.zeros(self.num_envs, dtype=np.int64)
        self._spawn_indices = np.zeros(self.num_envs, dtype=np.int64)
        self._terrain_levels = np.zeros(self.num_envs, dtype=np.float32)
        self._terrain_curriculum_limits = np.zeros(self.num_envs, dtype=np.float32)
        self._reset_reasons = np.zeros(self.num_envs, dtype=np.int64)
        self._current_command_stage = 0

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
        self.num_obs = 3 + 3 + 3 + self.num_actions + self.num_actions + self.num_actions + 3 + self._height_scan_dim
        self.num_privileged_obs = self.num_obs + self._foot_count * (1 + 1 + 1 + 3)
        self.command_manager = UniformVelocityCommand(self.cfg_obj.command, self)

        self.cfg = {
            "name": self.cfg_obj.name,
            "source": "gobot.rl.tasks.velocity",
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
            "terrain_normal_upright": self.cfg_obj.terrain_normal_upright.enabled,
            "contact_history_length": self._contact_history_length,
            "illegal_contact": self.cfg_obj.illegal_contact.enabled,
            "domain_randomization": self.cfg_obj.domain_randomization.enabled,
            "push_enabled": self.cfg_obj.push_enabled,
            "push_interval_range_s": self.cfg_obj.push_interval_range_s,
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
        self._terrain_curriculum_limits[:] = 0.0
        self._push_count[:] = 0
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
            self._maybe_apply_push(env_id)
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
            "terrain_normal_error": np.zeros(self.num_envs, dtype=np.float32),
            "illegal_contact_count": np.zeros(self.num_envs, dtype=np.float32),
            "self_collision_count": np.zeros(self.num_envs, dtype=np.float32),
            "shank_collision_count": np.zeros(self.num_envs, dtype=np.float32),
            "trunk_head_collision_count": np.zeros(self.num_envs, dtype=np.float32),
            "landing_force": np.zeros(self.num_envs, dtype=np.float32),
            "terrain_curriculum_limit": np.zeros(self.num_envs, dtype=np.float32),
            "encoder_bias_abs": np.zeros(self.num_envs, dtype=np.float32),
            "push_count": np.zeros(self.num_envs, dtype=np.float32),
        }
        reward_terms: dict[str, np.ndarray] = {}

        for env_id, state in enumerate(states_after_step):
            assert state is not None
            self._update_contact_summary(env_id, state)
            self._update_foot_history(env_id, state)
            reward, breakdown = self._compute_reward(env_id, state, action_np[env_id])
            for name, value in breakdown.items():
                reward_terms.setdefault(name, np.zeros(self.num_envs, dtype=np.float32))[env_id] = value
            rewards[env_id] = reward
            terminated[env_id] = self._terminated(env_id, state)
            self.episode_length_buf[env_id] += 1
            time_outs[env_id] = bool(self.episode_length_buf[env_id] >= self.max_episode_length and not terminated[env_id])
            done = bool(terminated[env_id] or time_outs[env_id])
            reset_reason[env_id] = 1 if terminated[env_id] else (2 if time_outs[env_id] else 0)
            self._episode_returns[env_id] += rewards[env_id]
            self._previous_actions[env_id] = self._last_actions[env_id]
            self._last_actions[env_id] = action_np[env_id]
            metrics["terrain_normal_error"][env_id] = self._terrain_normal_error[env_id]
            metrics["illegal_contact_count"][env_id] = self._illegal_contact_counts[env_id]
            metrics["self_collision_count"][env_id] = self._self_collision_counts[env_id]
            metrics["shank_collision_count"][env_id] = self._shank_collision_counts[env_id]
            metrics["trunk_head_collision_count"][env_id] = self._trunk_head_collision_counts[env_id]
            metrics["landing_force"][env_id] = float(np.sum(self._landing_force[env_id]))
            metrics["encoder_bias_abs"][env_id] = float(np.mean(np.abs(self._encoder_bias[env_id])))
            metrics["push_count"][env_id] = float(self._push_count[env_id])

            if done:
                self._update_terrain_curriculum_limit(env_id, state, reset_reason[env_id])
                self._reset_env(env_id, reason=reset_reason[env_id])
                state = self._runtime_state(env_id)
            metrics["terrain_curriculum_limit"][env_id] = self._terrain_curriculum_limits[env_id]

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
                "/velocity/terrain_normal_error": torch.as_tensor(float(np.mean(metrics["terrain_normal_error"])), device=self.device),
                "/velocity/illegal_contact_count": torch.as_tensor(float(np.mean(metrics["illegal_contact_count"])), device=self.device),
                "/velocity/self_collision_count": torch.as_tensor(float(np.mean(metrics["self_collision_count"])), device=self.device),
                "/velocity/shank_collision_count": torch.as_tensor(float(np.mean(metrics["shank_collision_count"])), device=self.device),
                "/velocity/trunk_head_collision_count": torch.as_tensor(float(np.mean(metrics["trunk_head_collision_count"])), device=self.device),
                "/velocity/landing_force": torch.as_tensor(float(np.mean(metrics["landing_force"])), device=self.device),
                "/velocity/terrain_curriculum_limit": torch.as_tensor(float(np.mean(metrics["terrain_curriculum_limit"])), device=self.device),
                "/velocity/encoder_bias_abs": torch.as_tensor(float(np.mean(metrics["encoder_bias_abs"])), device=self.device),
                "/velocity/push_count": torch.as_tensor(float(np.mean(metrics["push_count"])), device=self.device),
                "/velocity/command_stage": torch.as_tensor(float(self._current_command_stage), device=self.device),
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

    def _sample_spawn_index(self, env_id: int) -> int:
        if not self.cfg_obj.terrain_curriculum:
            return int(self._rng.integers(0, self._spawn_origins.shape[0]))
        warmup_progress = 0.10
        if self._curriculum_progress < warmup_progress and self._terrain_curriculum_limits[env_id] <= 0.0:
            return self._center_spawn_index
        difficulty_progress = (self._curriculum_progress - warmup_progress) / max(1.0 - warmup_progress, 1.0e-6)
        allowed_level = max(float(np.clip(difficulty_progress, 0.0, 1.0)), float(self._terrain_curriculum_limits[env_id]))
        candidates = np.flatnonzero(self._spawn_levels <= allowed_level + 1.0e-6)
        if candidates.size == 0:
            candidates = self._spawn_order[:1]
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
        current_stage = 0
        for index, stage in enumerate(self.cfg_obj.command_curriculum):
            if self._total_policy_steps < stage.step:
                continue
            current_stage = index
            if stage.lin_vel_x is not None:
                ranges.lin_vel_x = stage.lin_vel_x
            if stage.lin_vel_y is not None:
                ranges.lin_vel_y = stage.lin_vel_y
            if stage.ang_vel_z is not None:
                ranges.ang_vel_z = stage.ang_vel_z
        self._current_command_stage = current_stage

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
        spawn_index = self._sample_spawn_index(env_id)
        spawn = self._spawn_origins[spawn_index].copy()
        spawn[:2] += self._rng.uniform(-self.cfg_obj.spawn_jitter, self.cfg_obj.spawn_jitter, 2)
        terrain_height = self._terrain_height(spawn[0], spawn[1])
        base_z = max(spawn[2], terrain_height) + self.cfg_obj.base_clearance
        yaw = float(self._rng.uniform(-math.pi, math.pi))
        reset_lin_vel = (
            self._sample_range_vec(self.cfg_obj.domain_randomization.reset_lin_vel_ranges, ("x", "y", "z"))
            if self.cfg_obj.domain_randomization.enabled
            else np.zeros(3, dtype=np.float32)
        )
        reset_ang_vel = (
            self._sample_range_vec(self.cfg_obj.domain_randomization.reset_ang_vel_ranges, ("x", "y", "z"))
            if self.cfg_obj.domain_randomization.enabled
            else np.zeros(3, dtype=np.float32)
        )
        self.context.reset_batch_link_state(
            env_id,
            self.cfg_obj.robot_name,
            self.cfg_obj.base_link,
            [float(spawn[0]), float(spawn[1]), float(base_z)],
            _quat_from_yaw(yaw).tolist(),
            reset_lin_vel.tolist(),
            reset_ang_vel.tolist(),
        )
        if self.cfg_obj.domain_randomization.enabled:
            lo, hi = self.cfg_obj.domain_randomization.encoder_bias_range
            self._encoder_bias[env_id] = self._rng.uniform(lo, hi, self.num_actions).astype(np.float32)
        else:
            self._encoder_bias[env_id] = 0.0
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
        self._foot_contact_history[env_id] = 0.0
        self._foot_force_history[env_id] = 0.0
        self._foot_history_cursor[env_id] = 0
        self._first_contact[env_id] = 0.0
        self._landing_force[env_id] = 0.0
        self._terrain_normal_error[env_id] = 0.0
        self._illegal_contact_counts[env_id] = 0.0
        self._self_collision_counts[env_id] = 0.0
        self._shank_collision_counts[env_id] = 0.0
        self._trunk_head_collision_counts[env_id] = 0.0
        self._reset_push_timer(env_id)
        self._reset_reasons[env_id] = reason
        self.command_manager.reset(np.asarray([env_id], dtype=np.int64))

    def _sample_range_vec(self, ranges: Mapping[str, tuple[float, float]], names: Sequence[str]) -> np.ndarray:
        values = []
        for name in names:
            lo, hi = ranges.get(name, (0.0, 0.0))
            values.append(float(self._rng.uniform(lo, hi)))
        return np.asarray(values, dtype=np.float32)

    def _reset_push_timer(self, env_id: int) -> None:
        if not self.cfg_obj.push_enabled:
            self._push_time_left[env_id] = math.inf
            return
        lo, hi = self.cfg_obj.push_interval_range_s
        self._push_time_left[env_id] = float(self._rng.uniform(lo, hi))

    def _maybe_apply_push(self, env_id: int) -> None:
        if not self.cfg_obj.push_enabled:
            return
        self._push_time_left[env_id] -= self.step_dt
        if self._push_time_left[env_id] > 0.0:
            return
        state = self._runtime_state(env_id)
        position = _as_vec(state.base.get("global_transform", {}).get("position"), 3)
        orientation = _quat(state.base)
        linear_velocity = _as_vec(state.base.get("linear_velocity"), 3)
        angular_velocity = _as_vec(state.base.get("angular_velocity"), 3)
        linear_velocity += self._sample_range_vec(self.cfg_obj.push_velocity_ranges, ("x", "y", "z"))
        angular_velocity += self._sample_range_vec(self.cfg_obj.push_velocity_ranges, ("roll", "pitch", "yaw"))
        self.context.reset_batch_link_state(
            env_id,
            self.cfg_obj.robot_name,
            self.cfg_obj.base_link,
            position.tolist(),
            orientation.tolist(),
            linear_velocity.tolist(),
            angular_velocity.tolist(),
        )
        self._push_count[env_id] += 1
        self._reset_push_timer(env_id)

    def _update_terrain_curriculum_limit(
        self,
        env_id: int,
        state: VelocityRuntimeState,
        reset_reason: int,
    ) -> None:
        if not self.cfg_obj.terrain_curriculum:
            self._terrain_curriculum_limits[env_id] = 1.0
            return
        base_pos = _as_vec(state.base.get("global_transform", {}).get("position"), 3)
        distance = float(np.linalg.norm(base_pos[:2] - self._episode_start_xy[env_id]))
        survival = float(self.episode_length_buf[env_id]) / max(float(self.max_episode_length), 1.0)
        level_step = 1.0 / max(float(self._spawn_order.size - 1), 1.0)
        commanded_speed = float(np.linalg.norm(self.command_manager.command_b[env_id, :2]))
        expected_distance = max(0.25, commanded_speed * survival * self.max_episode_length * self.step_dt * 0.5)

        level = float(self._terrain_curriculum_limits[env_id])
        if reset_reason == 2 or survival > 0.75 or distance > expected_distance:
            level += level_step
        elif reset_reason == 1 and survival < 0.25:
            level -= level_step
        self._terrain_curriculum_limits[env_id] = float(np.clip(level, 0.0, 1.0))

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
        joint_pos, joint_vel = self._joint_values(state, env_id=env_id, apply_encoder_bias=True)
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

    def _joint_values(
        self,
        state: VelocityRuntimeState,
        *,
        env_id: int | None = None,
        apply_encoder_bias: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        pos = np.zeros(self.num_actions, dtype=np.float32)
        vel = np.zeros(self.num_actions, dtype=np.float32)
        for index, joint_name in enumerate(self.joint_names):
            joint = state.joints.get(joint_name, {})
            pos[index] = float(joint.get("position", self.default_joint_pos[index]))
            vel[index] = float(joint.get("velocity", 0.0))
        if apply_encoder_bias and env_id is not None:
            pos += self._encoder_bias[env_id]
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
            if not self._robot_contact_links(contact):
                continue
            position = _as_vec(contact.get("position"), 3)
            if foot_positions.size == 0:
                continue
            index = int(np.argmin(np.linalg.norm(foot_positions - position.reshape(1, 3), axis=1)))
            if np.linalg.norm(foot_positions[index] - position) > 0.18:
                continue
            forces[index] += _as_vec(contact.get("force"), 3)
        return forces

    def _update_contact_summary(self, env_id: int, state: VelocityRuntimeState) -> None:
        summary = self._contact_summary(state)
        self._illegal_contact_counts[env_id] = summary["illegal"]
        self._self_collision_counts[env_id] = summary["self_collision"]
        self._shank_collision_counts[env_id] = summary["shank"]
        self._trunk_head_collision_counts[env_id] = summary["trunk_head"]

    def _contact_summary(self, state: VelocityRuntimeState) -> dict[str, float]:
        cfg = self.cfg_obj.illegal_contact
        if not cfg.enabled:
            return {
                "illegal": 0.0,
                "self_collision": 0.0,
                "shank": 0.0,
                "trunk_head": 0.0,
            }
        summary = {
            "illegal": 0.0,
            "self_collision": 0.0,
            "shank": 0.0,
            "trunk_head": 0.0,
        }
        for contact in state.contacts:
            links = self._robot_contact_links(contact)
            if not links:
                continue
            magnitude = self._contact_force_magnitude(contact)
            robot_name = str(contact.get("robot_name") or "")
            other_robot = str(contact.get("other_robot_name") or "")
            is_self_collision = bool(robot_name == self.cfg_obj.robot_name and other_robot == self.cfg_obj.robot_name)
            if is_self_collision and magnitude >= cfg.self_collision_force_threshold:
                summary["self_collision"] += 1.0
                continue
            if magnitude < cfg.ground_force_threshold:
                continue
            for link_name in links:
                if cfg.terminate_on_thigh and self._matches_any(link_name, cfg.thigh_link_patterns):
                    summary["illegal"] += 1.0
                if self._matches_any(link_name, cfg.shank_link_patterns) and not self._contact_near_any_foot(state, contact):
                    summary["shank"] += 1.0
                if self._matches_any(link_name, cfg.trunk_head_link_patterns):
                    summary["trunk_head"] += 1.0
        return summary

    def _robot_contact_links(self, contact: Mapping[str, Any]) -> list[str]:
        links = []
        if str(contact.get("robot_name") or "") == self.cfg_obj.robot_name:
            link_name = str(contact.get("link_name") or "")
            if link_name:
                links.append(link_name)
        if str(contact.get("other_robot_name") or "") == self.cfg_obj.robot_name:
            link_name = str(contact.get("other_link_name") or "")
            if link_name:
                links.append(link_name)
        return links

    @staticmethod
    def _contact_force_magnitude(contact: Mapping[str, Any]) -> float:
        normal_force = contact.get("normal_force")
        if normal_force is not None:
            try:
                magnitude = abs(float(normal_force))
                if magnitude > 0.0:
                    return magnitude
            except (TypeError, ValueError):
                pass
        return float(np.linalg.norm(_as_vec(contact.get("force"), 3)))

    def _contact_near_any_foot(self, state: VelocityRuntimeState, contact: Mapping[str, Any]) -> bool:
        foot_positions = self._foot_positions(state)
        if foot_positions.size == 0:
            return False
        position = _as_vec(contact.get("position"), 3)
        return bool(np.min(np.linalg.norm(foot_positions - position.reshape(1, 3), axis=1)) <= 0.18)

    @staticmethod
    def _matches_any(name: str, patterns: Sequence[str]) -> bool:
        for pattern in patterns:
            if re.fullmatch(pattern, name) or re.search(pattern, name):
                return True
        return False

    def _update_foot_history(self, env_id: int, state: VelocityRuntimeState) -> None:
        positions = self._foot_positions(state)
        if not np.any(self._previous_foot_positions[env_id]):
            self._previous_foot_positions[env_id] = positions
        self._foot_velocities[env_id] = (positions - self._previous_foot_positions[env_id]) / max(self.step_dt, 1.0e-6)
        self._previous_foot_positions[env_id] = positions
        contacts = self._foot_contacts(state)
        heights = self._foot_heights(state)
        forces = self._foot_contact_forces(state)
        first_contact = (contacts > 0.0) & (self._last_foot_contact[env_id] <= 0.0)
        self._first_contact[env_id] = first_contact.astype(np.float32)
        self._landing_force[env_id] = np.linalg.norm(forces, axis=1) * self._first_contact[env_id]
        cursor = int(self._foot_history_cursor[env_id] % self._contact_history_length)
        self._foot_contact_history[env_id, cursor] = contacts
        self._foot_force_history[env_id, cursor] = forces
        self._foot_history_cursor[env_id] = (cursor + 1) % self._contact_history_length
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
        upright_error = self._upright_error(env_id, state, base_quat)
        action_rate = float(np.sum(np.square(action - self._previous_actions[env_id])))
        command_speed = float(np.linalg.norm(command[:2]) + abs(command[2]))

        pose_std = self._pose_std(command_speed)
        pose_error = float(np.mean(np.square(joint_pos - self.default_joint_pos) / np.square(np.maximum(pose_std, 1.0e-6))))
        foot_clearance_cost = float(np.sum(np.abs(foot_heights - reward_cfg.foot_target_height) * np.linalg.norm(self._foot_velocities[env_id, :, :2], axis=1)))
        active = 1.0 if command_speed > reward_cfg.command_threshold else 0.0
        foot_slip = self._foot_slip_cost(env_id, foot_contacts) * active
        first_contact = self._first_contact[env_id] > 0.0
        landing_force = float(np.sum(self._landing_force[env_id])) * active
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
            "self_collisions": reward_cfg.self_collisions * float(self._self_collision_counts[env_id]),
            "shank_collision": reward_cfg.shank_collision * float(self._shank_collision_counts[env_id]),
            "trunk_head_collision": reward_cfg.trunk_head_collision * float(self._trunk_head_collision_counts[env_id]),
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

    def _upright_error(self, env_id: int, state: VelocityRuntimeState, base_quat: np.ndarray) -> float:
        if self.cfg_obj.terrain_normal_upright.enabled and self._height_scan_dim > 0:
            normal_w = self._terrain_normal_from_scan(state)
            body_normal = _rotate_vec_by_quat_inv(normal_w, base_quat)
            error = float(np.sum(np.square(body_normal[:2])))
        else:
            projected_gravity = _rotate_vec_by_quat_inv(np.array([0.0, 0.0, -1.0], dtype=np.float32), base_quat)
            error = float(np.sum(np.square(projected_gravity[:2])))
        self._terrain_normal_error[env_id] = error
        return error

    def _terrain_normal_from_scan(self, state: VelocityRuntimeState) -> np.ndarray:
        sensor_name = self.cfg_obj.observations.height_scan_sensor
        if sensor_name is None:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        sensor = state.sensors.get(sensor_name)
        if sensor is None:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        points = []
        normals = []
        for hit in sensor.get("hits", []):
            if not hit.get("hit", False):
                continue
            point = _as_vec(hit.get("point"), 3)
            if np.all(np.isfinite(point)):
                points.append(point.astype(np.float64))
            normal = _as_vec(hit.get("normal"), 3)
            if np.all(np.isfinite(normal)) and np.linalg.norm(normal) > 1.0e-6:
                normals.append(normal.astype(np.float64))

        min_hits = max(3, int(self.cfg_obj.terrain_normal_upright.min_hit_count))
        if len(points) >= min_hits:
            cloud = np.asarray(points, dtype=np.float64)
            centered = cloud - np.mean(cloud, axis=0)
            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                normal = vh[-1]
                if normal[2] < 0.0:
                    normal = -normal
                length = np.linalg.norm(normal)
                if length > 1.0e-6 and np.all(np.isfinite(normal)):
                    return (normal / length).astype(np.float32)
            except np.linalg.LinAlgError:
                pass
        if normals:
            normal = np.mean(np.asarray(normals, dtype=np.float64), axis=0)
            if normal[2] < 0.0:
                normal = -normal
            length = np.linalg.norm(normal)
            if length > 1.0e-6 and np.all(np.isfinite(normal)):
                return (normal / length).astype(np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def _base_clearance(self, state: VelocityRuntimeState) -> float:
        position = _as_vec(state.base.get("global_transform", {}).get("position"), 3)
        return float(position[2] - self._terrain_height(position[0], position[1]))

    def _terminated(self, env_id: int, state: VelocityRuntimeState) -> bool:
        if self.cfg_obj.terrain_type == "rough":
            if self._base_clearance(state) < self.cfg_obj.min_base_clearance:
                return True
            if self.cfg_obj.illegal_contact.enabled and self._illegal_contact_counts[env_id] > 0.0:
                return True
            return False
        roll, pitch = _quat_to_rp(_quat(state.base))
        return bool(self._base_clearance(state) < self.cfg_obj.min_base_clearance or abs(roll) > math.radians(70.0) or abs(pitch) > math.radians(70.0))


__all__ = ["GobotVelocityEnv", "VelocityRuntimeState"]
