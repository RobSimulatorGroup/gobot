"""Go1 NumPy batch environment for velocity locomotion training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from ._repo_imports import prefer_repo_gobot
except ImportError:
    from _repo_imports import prefer_repo_gobot

prefer_repo_gobot()

import gobot

from gobot.rl import (
    ActionSpec,
    BatchEnvState,
    BatchSimulationRuntime,
    SpecField,
)
from gobot.rl.locomotion import (
    LocomotionBatchEnv,
    LocomotionBatchSpec,
    LocomotionControlCfg,
    LocomotionNoiseCfg,
    NativeLocomotionBatchBackend,
    TerrainSampler,
    UniformVelocityCommand,
    build_velocity_actor_observation,
    build_velocity_critic_observation,
    velocity_actor_observation_schema,
    velocity_critic_observation_schema,
)
from gobot.rl.locomotion.math import (
    _as_vec,
    _find_node_by_name,
    _quat,
    _quat_from_yaw,
    _quat_to_rp,
    _rotate_vec_by_quat_inv,
)

try:
    from .go1_velocity_cfg import Go1VelocityCfg, go1_rough_velocity_cfg
except ImportError:
    from go1_velocity_cfg import Go1VelocityCfg, go1_rough_velocity_cfg

_REWARD_TERM_NAMES: tuple[str, ...] = (
    "track_linear_velocity",
    "track_angular_velocity",
    "upright",
    "pose",
    "body_ang_vel",
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

_TASK_PARAM = {
    "step_dt": 0,
    "lin_vel_std2": 1,
    "ang_vel_std2": 2,
    "upright_std2": 3,
    "foot_target_height": 4,
    "command_threshold": 5,
    "min_base_clearance": 6,
    "flat_roll_pitch_limit": 7,
    "pose_walking_threshold": 8,
    "pose_running_threshold": 9,
    "height_scan_max_distance": 10,
}

_TASK_FLAG = {
    "rough_terrain": 0,
    "terrain_normal_upright": 1,
    "illegal_contact_enabled": 2,
}


def _iter_nodes(node):
    if node is None:
        return
    yield node
    for child in node.children:
        yield from _iter_nodes(child)


def _node_names_by_type(root, type_name: str) -> list[str]:
    return [node.name for node in _iter_nodes(root) if getattr(node, "type", "") == type_name]


def _node_names_by_base_type(root, base_type: str) -> list[str]:
    if base_type != "Sensor3D":
        return _node_names_by_type(root, base_type)
    return [
        node.name
        for node in _iter_nodes(root)
        if str(getattr(node, "type", "")).endswith("Sensor3D") or hasattr(node, "sensor_period")
    ]


@dataclass
class VelocityRuntimeState:
    robot: Mapping[str, Any]
    base: Mapping[str, Any]
    joints: Mapping[str, Mapping[str, Any]]
    links: Mapping[str, Mapping[str, Any]]
    sensors: Mapping[str, Mapping[str, Any]]
    contacts: Sequence[Mapping[str, Any]]

VelocityBatchRuntimeState = Any


class Go1VelocityEnv(LocomotionBatchEnv):
    """NumPy batch env for the Go1 velocity example."""

    is_vector_env = True

    def __init__(
        self,
        cfg: Go1VelocityCfg | None = None,
        *,
        num_envs: int = 64,
        device: str = "cpu",
        seed: int = 42,
        max_episode_length: int | None = None,
        sim_workers: int = 0,
        profile_step: bool = False,
        context: gobot.AppContext | None = None,
    ) -> None:
        self.cfg_obj = cfg if cfg is not None else go1_rough_velocity_cfg()
        joint_names = tuple(self.cfg_obj.joint_names)
        default_joint_pos = np.asarray(self.cfg_obj.default_joint_pos, dtype=np.float32)
        if len(joint_names) == 0 or default_joint_pos.shape != (len(joint_names),):
            raise ValueError("velocity task joint_names and default_joint_pos must be non-empty and have the same length")
        super().__init__(
            num_envs=int(num_envs),
            seed=int(seed),
            control_cfg=LocomotionControlCfg(action_scale=self.cfg_obj.action_scale),
            noise_cfg=LocomotionNoiseCfg(
                level=1.0 if self.cfg_obj.observations.actor_noise else 0.0,
                ranges=self.cfg_obj.observations.actor_noise_ranges,
            ),
            joint_names=joint_names,
            default_joint_pos=default_joint_pos,
            profile_step=profile_step,
        )
        self.device = device
        self.sim_workers = max(0, int(sim_workers))
        self.physics_dt = float(self.cfg_obj.physics_dt)
        self.decimation = int(self.cfg_obj.decimation)
        self.step_dt = self.physics_dt * self.decimation
        self.max_episode_length = int(max_episode_length or math.ceil(float(self.cfg_obj.episode_length_s) / self.step_dt))
        self._episode_length_np = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_length_buf = self._episode_length_np
        self._episode_start_xy = np.zeros((self.num_envs, 2), dtype=np.float32)
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._total_policy_steps = 0
        self._curriculum_progress = 0.0

        self._foot_count = len(self.cfg_obj.foot_names)
        self._contact_history_length = max(1, int(self.cfg_obj.contact_history_length))
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
        self.context = context if context is not None else gobot.app.context()
        self.context.set_project_path(str(self.project_path))
        self.context.load_scene(self.cfg_obj.scene_path)
        self.robot = self._find_robot_node()
        self._spawn_origins = self._load_spawn_origins()
        self._warmup_spawn_index = int(np.argmin(np.linalg.norm(self._spawn_origins[:, :2], axis=1)))
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

        self._scene_link_names = tuple(_node_names_by_type(self.robot, "Link3D"))
        self._height_scan_dim = self._sensor_dim(self.cfg_obj.observations.height_scan_sensor)
        if self.cfg_obj.observations.height_scan_sensor is not None and self._height_scan_dim == 0:
            raise RuntimeError(
                f"Gobot scene robot {self.cfg_obj.robot_name!r} has no usable height scan sensor "
                f"{self.cfg_obj.observations.height_scan_sensor!r}"
            )
        self._batch_link_names = tuple(
            dict.fromkeys(
                [str(name) for name in self._scene_link_names]
                or [self.cfg_obj.base_link, *self.cfg_obj.foot_link_names]
            )
        )
        self._batch_link_index = {name: index for index, name in enumerate(self._batch_link_names)}
        self._foot_link_indices = np.asarray(
            [self._batch_link_index[name] for name in self.cfg_obj.foot_link_names],
            dtype=np.int64,
        )
        sensor_names = [f"{foot}_foot_height_scan" for foot in self.cfg_obj.foot_names]
        sensor_names.extend(f"{foot}_foot_contact" for foot in self.cfg_obj.foot_names)
        if self.cfg_obj.observations.height_scan_sensor is not None:
            sensor_names.append(self.cfg_obj.observations.height_scan_sensor)
        self._batch_sensor_names = tuple(dict.fromkeys(sensor_names))
        self._batch_sensor_index = {name: index for index, name in enumerate(self._batch_sensor_names)}
        self._foot_height_sensor_indices = np.asarray(
            [self._batch_sensor_index.get(f"{foot}_foot_height_scan", -1) for foot in self.cfg_obj.foot_names],
            dtype=np.int64,
        )
        self._foot_contact_sensor_indices = np.asarray(
            [self._batch_sensor_index.get(f"{foot}_foot_contact", -1) for foot in self.cfg_obj.foot_names],
            dtype=np.int64,
        )
        self._height_scan_sensor_index = (
            self._batch_sensor_index.get(self.cfg_obj.observations.height_scan_sensor, -1)
            if self.cfg_obj.observations.height_scan_sensor is not None
            else -1
        )
        self.runtime = BatchSimulationRuntime(
            self.context,
            robot=self.cfg_obj.robot_name,
            base_link=self.cfg_obj.base_link,
            joint_names=self.joint_names,
            link_names=self._batch_link_names,
            sensor_names=self._batch_sensor_names,
        )
        self.backend = self._make_batch_backend()
        self.backend.configure(self.num_envs)
        self.resolved_sim_workers = self.backend.resolved_workers(self.sim_workers)
        self.actor_obs_schema = velocity_actor_observation_schema(self.num_actions, self._height_scan_dim)
        self.critic_obs_schema = velocity_critic_observation_schema(self.num_actions, self._height_scan_dim, self._foot_count)
        self.action_spec = self._make_action_spec()
        self.observation_spec = self.actor_obs_schema
        self.critic_observation_spec = self.critic_obs_schema
        self.num_obs = self.actor_obs_schema.dim
        self.num_privileged_obs = self.critic_obs_schema.dim
        self.command_manager = UniformVelocityCommand(self.cfg_obj.command, self)
        self._configure_native_task_buffers()

        self.cfg = {
            "name": self.cfg_obj.name,
            "source": "examples.go1.train.go1_velocity_env",
            "task": "gobot_go1_velocity",
            "obs_schema_version": self.actor_obs_schema.version,
            "obs_names": self.actor_obs_schema.names,
            "critic_obs_names": self.critic_obs_schema.names,
            "obs_spec": self.actor_obs_schema.metadata(),
            "critic_obs_spec": self.critic_obs_schema.metadata(),
            "action_spec": self.action_spec.metadata(),
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
            "fast_batch_view": bool(getattr(self.backend, "is_fast", False)),
        }
        self.extras: dict[str, Any] = {}
        self._obs, self._critic_obs = self._reset_all()
        self._state_obs_actor = self._obs.copy()
        self._state_obs_critic = self._critic_obs.copy()
        self._state_reward = np.zeros((self.num_envs,), dtype=np.float32)
        self._state_terminated = np.zeros((self.num_envs,), dtype=bool)
        self._state_truncated = np.zeros((self.num_envs,), dtype=bool)
        self._state_steps = self._episode_length_np.copy()
        self._state = BatchEnvState(
            obs={
                "actor": self._state_obs_actor,
                "critic": self._state_obs_critic,
            },
            reward=self._state_reward,
            terminated=self._state_terminated,
            truncated=self._state_truncated,
            info={"steps": self._state_steps},
        )

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"actor": self.num_obs, "critic": self.num_privileged_obs}

    def get_observations(self) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in self._state.obs.items()} if self._state is not None else {
            "actor": self._obs.copy(),
            "critic": self._critic_obs.copy(),
        }

    def init_state(self) -> BatchEnvState:
        return self._state

    def reset(
        self,
        env_ids: np.ndarray | None = None,
        *,
        seed: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if env_ids is None:
            self.reset_seed(seed)
            env_ids = np.arange(self.num_envs, dtype=np.int64)
        else:
            env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
            if seed is not None:
                self.reset_seed(seed)
        if env_ids.size == 0:
            return {
                "actor": np.zeros((0, self.num_obs), dtype=np.float32),
                "critic": np.zeros((0, self.num_privileged_obs), dtype=np.float32),
            }, {}
        if np.any(env_ids < 0) or np.any(env_ids >= self.num_envs):
            raise IndexError("reset env_ids contain an out-of-range environment id")
        full_reset = env_ids.size == self.num_envs and np.array_equal(env_ids, np.arange(self.num_envs, dtype=np.int64))
        if full_reset:
            self._episode_length_np[:] = 0
            self._episode_returns[:] = 0.0
            self._terrain_curriculum_limits[:] = 0.0
            self._push_count[:] = 0
        self._reset_envs(env_ids, np.zeros(env_ids.size, dtype=np.int64))
        batch_state = self.backend.state
        self.backend.compute_observations()
        obs = np.asarray(batch_state.actor_obs, dtype=np.float32).copy()
        critic = np.asarray(batch_state.critic_obs, dtype=np.float32).copy()
        if self.noise_cfg.level > 0.0 and self.cfg_obj.observations.actor_noise:
            obs = self._apply_actor_obs_noise(obs)
            critic[:, : self.num_obs] = obs
        self._obs = obs
        self._critic_obs = critic
        if self._state is not None:
            self._sync_batch_env_state(
                reward=self._state.reward,
                terminated=self._state.terminated,
                truncated=self._state.truncated,
                obs_actor=obs,
                obs_critic=critic,
            )
            self._state.terminated[env_ids] = False
            self._state.truncated[env_ids] = False
            self._state.info["steps"][env_ids] = 0
        reset_info = {
            "reset_env_ids": env_ids.copy(),
            "reset_reason": self._reset_reasons[env_ids].copy(),
            "terrain_level": self._terrain_levels[env_ids].copy(),
        }
        return {"actor": obs[env_ids].copy(), "critic": critic[env_ids].copy()}, reset_info

    def reset_all(self, seed: int | None = None) -> dict[str, np.ndarray]:
        obs, _ = self.reset(seed=seed)
        return obs

    def step(self, actions):
        step_t0 = self.perf_counter()
        self.clear_step_final_observation()
        profile_marks = self._new_profile_marks()
        self._total_policy_steps += self.num_envs
        self._apply_command_curriculum()
        self._update_curriculum_progress()

        t0 = self.perf_counter()
        action_np = self._prepare_actions(actions)
        apply_action_ms = (self.perf_counter() - t0) * 1000.0
        self._mark_profile(profile_marks, "action_prepare")

        t0 = self.perf_counter()
        self._apply_pushes()
        self._mark_profile(profile_marks, "action_apply")
        backend_action_ms = (self.perf_counter() - t0) * 1000.0
        t0 = self.perf_counter()
        self.backend.step_actions(
            action_np,
            self.decimation,
            workers=self.sim_workers,
            simulate_action_latency=self.control_cfg.simulate_action_latency,
        )
        backend_physics_ms = (self.perf_counter() - t0) * 1000.0
        self._mark_profile(profile_marks, "physics")
        t0 = self.perf_counter()
        batch_state = self.backend.state
        backend_refresh_cache_ms = (self.perf_counter() - t0) * 1000.0
        self._mark_profile(profile_marks, "state")
        step_core_ms = backend_action_ms + backend_physics_ms + backend_refresh_cache_ms

        t0 = self.perf_counter()
        self.command_manager.compute_batch(
            self.step_dt,
            base_quaternion=batch_state.base_quaternion,
            base_lin_vel_b=batch_state.base_linear_velocity_body,
            base_ang_vel_b=batch_state.base_angular_velocity_body,
            update_metrics=False,
        )
        np.copyto(batch_state.command, self.command_manager.command_b)
        self._mark_profile(profile_marks, "command")

        fused_task_observations = hasattr(self.backend, "compute_task_observations")
        if fused_task_observations:
            self.backend.compute_task_observations()
        else:
            self.backend.compute_task()
        foot_heights = batch_state.foot_height
        foot_contacts = batch_state.foot_contact
        self._mark_profile(profile_marks, "contact")
        rewards = np.asarray(batch_state.reward, dtype=np.float32).copy()
        reward_terms = {
            name: np.asarray(batch_state.reward_terms[:, index], dtype=np.float32)
            for index, name in enumerate(_REWARD_TERM_NAMES)
        }
        self._mark_profile(profile_marks, "reward")
        log_values = {
            "velocity_error": float(np.mean(batch_state.velocity_error)),
            "foot_clearance": float(np.mean(foot_heights)) if foot_heights.size else 0.0,
            "foot_contact_ratio": float(np.mean(foot_contacts)) if foot_contacts.size else 0.0,
            "foot_slip": float(np.mean(batch_state.foot_slip)),
            "terrain_normal_error": float(np.mean(batch_state.terrain_normal_error)),
            "illegal_contact_count": float(np.mean(batch_state.illegal_contact_count)),
            "self_collision_count": float(np.mean(batch_state.self_collision_count)),
            "shank_collision_count": float(np.mean(batch_state.shank_collision_count)),
            "trunk_head_collision_count": float(np.mean(batch_state.trunk_head_collision_count)),
            "landing_force": float(np.mean(np.sum(batch_state.landing_force, axis=1))),
            "encoder_bias_abs": float(np.mean(np.abs(batch_state.encoder_bias))),
            "push_count": float(np.mean(self._push_count)),
        }
        update_state_ms = (self.perf_counter() - t0) * 1000.0

        t0 = self.perf_counter()
        terminated = np.asarray(batch_state.terminated, dtype=bool).copy()
        self._episode_length_np += 1
        time_outs = (self._episode_length_np >= self.max_episode_length) & ~terminated
        done_envs = terminated | time_outs
        reset_reason = np.where(terminated, 1, np.where(time_outs, 2, 0)).astype(np.int64)
        self._episode_returns += rewards
        reset_env_ids = np.flatnonzero(done_envs).astype(np.int64)
        terminal_actor_obs: np.ndarray | None = None
        terminal_critic_obs: np.ndarray | None = None
        if reset_env_ids.size and not fused_task_observations:
            self.backend.compute_observations()
        if reset_env_ids.size:
            terminal_actor_obs = np.asarray(batch_state.actor_obs, dtype=np.float32).copy()
            terminal_critic_obs = np.asarray(batch_state.critic_obs, dtype=np.float32).copy()
            if self.noise_cfg.level > 0.0 and self.cfg_obj.observations.actor_noise:
                terminal_actor_obs = self._apply_actor_obs_noise(terminal_actor_obs)
                terminal_critic_obs[:, : self.num_obs] = terminal_actor_obs
        if reset_env_ids.size:
            self.capture_final_observation(reset_env_ids)
            for env_id in reset_env_ids:
                self._update_terrain_curriculum_limit_from_position(
                    int(env_id),
                    batch_state.base_position[int(env_id)],
                    int(reset_reason[int(env_id)]),
                )
            self._reset_envs(reset_env_ids, reset_reason[reset_env_ids])
        log_values["terrain_curriculum_limit"] = float(np.mean(self._terrain_curriculum_limits))

        active_envs = ~done_envs
        batch_state.previous_action[active_envs] = batch_state.last_action[active_envs]
        batch_state.last_action[active_envs] = batch_state.submitted_action[active_envs]
        self._previous_actions[active_envs] = batch_state.previous_action[active_envs]
        self._last_actions[active_envs] = batch_state.last_action[active_envs]

        if reset_env_ids.size:
            batch_state = self.backend.state
            np.copyto(batch_state.command, self.command_manager.command_b)
        self._mark_profile(profile_marks, "termination_reset")
        reset_done_ms = (self.perf_counter() - t0) * 1000.0

        t0 = self.perf_counter()
        if not fused_task_observations and not reset_env_ids.size:
            self.backend.compute_observations()
        elif reset_env_ids.size:
            self.backend.compute_observations()
        obs_np = np.asarray(batch_state.actor_obs, dtype=np.float32).copy()
        critic_np = np.asarray(batch_state.critic_obs, dtype=np.float32).copy()
        if self.noise_cfg.level > 0.0 and self.cfg_obj.observations.actor_noise:
            obs_np = self._apply_actor_obs_noise(obs_np)
            critic_np[:, : self.num_obs] = obs_np
        self._mark_profile(profile_marks, "obs_build")
        self._obs = obs_np
        self._critic_obs = critic_np
        self._mark_profile(profile_marks, "tensor_convert")
        observation_ms = (self.perf_counter() - t0) * 1000.0

        if reset_env_ids.size and self._state is not None:
            scratch = self._state.final_observation
            if scratch is not None and terminal_actor_obs is not None and terminal_critic_obs is not None:
                scratch["actor"][reset_env_ids] = terminal_actor_obs[reset_env_ids]
                scratch["critic"][reset_env_ids] = terminal_critic_obs[reset_env_ids]
                self._state.info["final_observation"] = scratch
        extras = {
            "time_outs": time_outs.copy(),
            "log": {
                "/velocity/terrain_level": np.asarray(float(np.mean(self._terrain_levels)), dtype=np.float32),
                "/velocity/velocity_error": np.asarray(log_values["velocity_error"], dtype=np.float32),
                "/velocity/foot_clearance": np.asarray(log_values["foot_clearance"], dtype=np.float32),
                "/velocity/foot_contact_ratio": np.asarray(log_values["foot_contact_ratio"], dtype=np.float32),
                "/velocity/foot_slip": np.asarray(log_values["foot_slip"], dtype=np.float32),
                "/velocity/terrain_normal_error": np.asarray(log_values["terrain_normal_error"], dtype=np.float32),
                "/velocity/illegal_contact_count": np.asarray(log_values["illegal_contact_count"], dtype=np.float32),
                "/velocity/self_collision_count": np.asarray(log_values["self_collision_count"], dtype=np.float32),
                "/velocity/shank_collision_count": np.asarray(log_values["shank_collision_count"], dtype=np.float32),
                "/velocity/trunk_head_collision_count": np.asarray(log_values["trunk_head_collision_count"], dtype=np.float32),
                "/velocity/landing_force": np.asarray(log_values["landing_force"], dtype=np.float32),
                "/velocity/terrain_curriculum_limit": np.asarray(log_values["terrain_curriculum_limit"], dtype=np.float32),
                "/velocity/encoder_bias_abs": np.asarray(log_values["encoder_bias_abs"], dtype=np.float32),
                "/velocity/push_count": np.asarray(log_values["push_count"], dtype=np.float32),
                "/velocity/command_stage": np.asarray(float(self._current_command_stage), dtype=np.float32),
                "/velocity/reset_reason": np.asarray(float(np.mean(reset_reason)), dtype=np.float32),
                "/velocity/command_vx": np.asarray(float(np.mean(self.command_manager.command_b[:, 0])), dtype=np.float32),
                "/velocity/command_vy": np.asarray(float(np.mean(self.command_manager.command_b[:, 1])), dtype=np.float32),
                "/velocity/command_yaw": np.asarray(float(np.mean(self.command_manager.command_b[:, 2])), dtype=np.float32),
            },
            "reward_terms": {name: value.copy() for name, value in reward_terms.items()},
        }
        self._mark_profile(profile_marks, "extras")
        if profile_marks is not None:
            for name, value in self._consume_profile_marks(profile_marks).items():
                extras["log"][f"/profile/{name}_ms"] = np.asarray(value, dtype=np.float32)
        self.extras = extras
        self._sync_batch_env_state(
            reward=rewards,
            terminated=terminated,
            truncated=time_outs,
            obs_actor=obs_np,
            obs_critic=critic_np,
            info_extra={
                "reward_terms": reward_terms,
                "reset_reason": reset_reason,
                "time_outs": time_outs,
                "log": extras["log"],
            },
        )
        self.step_counter += 1
        timing = self._state.info.setdefault("timing", {})
        timing["env_step_total_ms"] = (self.perf_counter() - step_t0) * 1000.0
        timing["apply_action_ms"] = apply_action_ms
        timing["step_core_ms"] = step_core_ms
        timing["backend_apply_action_ms"] = backend_action_ms
        timing["backend_physics_ms"] = backend_physics_ms
        timing["backend_refresh_cache_ms"] = backend_refresh_cache_ms
        timing["update_state_ms"] = update_state_ms + observation_ms
        timing["reset_done_ms"] = reset_done_ms
        return self._state

    def close(self) -> None:
        self.context.clear_world()
        self.context.clear_scene()

    def set_training_progress(self, policy_steps: int) -> None:
        self._total_policy_steps = max(0, int(policy_steps))
        self._apply_command_curriculum()
        self._update_curriculum_progress()

    def _apply_actor_obs_noise(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).copy()
        offset = 0
        obs[:, offset : offset + 3] = self._obs_noise(obs[:, offset : offset + 3], "base_lin_vel", corrupt=True)
        offset += 3
        obs[:, offset : offset + 3] = self._obs_noise(obs[:, offset : offset + 3], "base_ang_vel", corrupt=True)
        offset += 3
        obs[:, offset : offset + 3] = self._obs_noise(obs[:, offset : offset + 3], "projected_gravity", corrupt=True)
        offset += 3
        obs[:, offset : offset + self.num_actions] = self._obs_noise(
            obs[:, offset : offset + self.num_actions],
            "joint_pos",
            corrupt=True,
        )
        offset += self.num_actions
        obs[:, offset : offset + self.num_actions] = self._obs_noise(
            obs[:, offset : offset + self.num_actions],
            "joint_vel",
            corrupt=True,
        )
        offset += self.num_actions * 2 + 3
        if self._height_scan_dim > 0:
            scale = max(self.cfg_obj.observations.terrain_scan_max_distance, 1.0e-6)
            height = obs[:, offset : offset + self._height_scan_dim] * scale
            obs[:, offset : offset + self._height_scan_dim] = self._obs_noise(
                height,
                "height_scan",
                corrupt=True,
            ) / scale
        return obs

    def _make_action_spec(self) -> ActionSpec:
        return ActionSpec(
            version=f"{self.actor_obs_schema.version}_action_v1",
            fields=tuple(SpecField(str(name), 1) for name in self.joint_names),
            lower=-1.0,
            upper=1.0,
        )

    def _sync_batch_env_state(
        self,
        *,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        obs_actor: np.ndarray | None = None,
        obs_critic: np.ndarray | None = None,
        info_extra: Mapping[str, Any] | None = None,
    ) -> None:
        info: dict[str, Any] = {
            "steps": self._state_steps,
        }
        if self._state is not None:
            for key in ("final_observation", "_final_observation"):
                if key in self._state.info:
                    info[key] = self._state.info[key]
        if info_extra:
            info.update(info_extra)
        if obs_actor is None:
            obs_actor = self._obs
        if obs_critic is None:
            obs_critic = self._critic_obs
        np.copyto(self._state_obs_actor, obs_actor)
        np.copyto(self._state_obs_critic, obs_critic)
        np.copyto(self._state_reward, np.asarray(reward, dtype=np.float32).reshape(self.num_envs))
        np.copyto(self._state_terminated, np.asarray(terminated, dtype=bool).reshape(self.num_envs))
        np.copyto(self._state_truncated, np.asarray(truncated, dtype=bool).reshape(self.num_envs))
        np.copyto(self._state_steps, self._episode_length_np)
        if self._state is None:
            self._state = BatchEnvState(
                obs={
                    "actor": self._state_obs_actor,
                    "critic": self._state_obs_critic,
                },
                reward=self._state_reward,
                terminated=self._state_terminated,
                truncated=self._state_truncated,
                info=info,
            )
        else:
            self._state.info = info

    def _sensor_dim(self, sensor_name: str | None) -> int:
        if sensor_name is None:
            return 0
        sensor = self.robot.find(sensor_name) or _find_node_by_name(self.robot, sensor_name)
        if sensor is None:
            return 0
        if getattr(sensor, "pattern_mode", None) == gobot.RayPatternMode.Grid:
            grid_size = np.asarray(getattr(sensor, "grid_size"), dtype=np.float64)
            resolution = float(getattr(sensor, "grid_resolution"))
            if grid_size.shape == (2,) and resolution > 0.0:
                return (int(round(float(grid_size[0]) / resolution)) + 1) * (
                    int(round(float(grid_size[1]) / resolution)) + 1
                )
        if hasattr(sensor, "sample_offsets"):
            return len(getattr(sensor, "sample_offsets", []))
        if hasattr(sensor, "grid_size") and hasattr(sensor, "grid_resolution"):
            grid_size = np.asarray(getattr(sensor, "grid_size"), dtype=np.float64)
            resolution = float(getattr(sensor, "grid_resolution"))
            if grid_size.shape == (2,) and resolution > 0.0:
                return (int(round(float(grid_size[0]) / resolution)) + 1) * (
                    int(round(float(grid_size[1]) / resolution)) + 1
                )
        return 0

    def _configure_robot_drives(self) -> None:
        for joint_name in self.joint_names:
            joint = self.robot.find(joint_name) or _find_node_by_name(self.robot, joint_name)
            if joint is None:
                continue
            joint.drive_mode = gobot.JointDriveMode.Position
            joint.drive_stiffness = self.cfg_obj.kp
            joint.drive_damping = self.cfg_obj.kd
            joint.damping = self.cfg_obj.kd

    def _make_batch_backend(self):
        illegal_cfg = self.cfg_obj.illegal_contact
        if hasattr(self.context, "create_locomotion_batch_view"):
            spec = LocomotionBatchSpec(
                foot_link_names=self.cfg_obj.foot_link_names,
                foot_height_sensor_names=[f"{foot}_foot_height_scan" for foot in self.cfg_obj.foot_names],
                foot_contact_sensor_names=[f"{foot}_foot_contact" for foot in self.cfg_obj.foot_names],
                height_scan_sensor=self.cfg_obj.observations.height_scan_sensor,
                thigh_link_patterns=illegal_cfg.thigh_link_patterns,
                shank_link_patterns=illegal_cfg.shank_link_patterns,
                trunk_head_link_patterns=illegal_cfg.trunk_head_link_patterns,
                terminate_on_thigh_contact=illegal_cfg.terminate_on_thigh,
                ground_force_threshold=illegal_cfg.ground_force_threshold,
                self_collision_force_threshold=illegal_cfg.self_collision_force_threshold,
            )
            return NativeLocomotionBatchBackend(
                self.runtime,
                spec=spec,
            )
        raise RuntimeError("Gobot Go1 velocity training requires create_locomotion_batch_view native binding")

    def _configure_native_task_buffers(self) -> None:
        state = self.backend.state
        np.copyto(state.default_joint_position, self.default_joint_pos.astype(np.float32))
        np.copyto(state.action_scale, self.action_scale.astype(np.float32))
        np.copyto(state.pose_std_standing, self._pose_std_from_table(self.cfg_obj.rewards.pose_std_standing))
        np.copyto(state.pose_std_walking, self._pose_std_from_table(self.cfg_obj.rewards.pose_std_walking))
        np.copyto(state.pose_std_running, self._pose_std_from_table(self.cfg_obj.rewards.pose_std_running))

        reward_cfg = self.cfg_obj.rewards
        reward_weights = np.asarray(
            [
                reward_cfg.track_linear_velocity,
                reward_cfg.track_angular_velocity,
                reward_cfg.upright,
                reward_cfg.pose,
                reward_cfg.body_ang_vel,
                reward_cfg.dof_pos_limits,
                reward_cfg.action_rate_l2,
                reward_cfg.air_time,
                reward_cfg.foot_clearance,
                reward_cfg.foot_swing_height,
                reward_cfg.foot_slip,
                reward_cfg.soft_landing,
                reward_cfg.self_collisions,
                reward_cfg.shank_collision,
                reward_cfg.trunk_head_collision,
            ],
            dtype=np.float32,
        )
        np.copyto(state.reward_weights, reward_weights)

        params = np.zeros_like(state.task_params, dtype=np.float32)
        params[_TASK_PARAM["step_dt"]] = self.step_dt
        params[_TASK_PARAM["lin_vel_std2"]] = reward_cfg.lin_vel_std**2
        params[_TASK_PARAM["ang_vel_std2"]] = reward_cfg.ang_vel_std**2
        params[_TASK_PARAM["upright_std2"]] = reward_cfg.upright_std**2
        params[_TASK_PARAM["foot_target_height"]] = reward_cfg.foot_target_height
        params[_TASK_PARAM["command_threshold"]] = reward_cfg.command_threshold
        params[_TASK_PARAM["min_base_clearance"]] = self.cfg_obj.min_base_clearance
        params[_TASK_PARAM["flat_roll_pitch_limit"]] = math.radians(70.0)
        params[_TASK_PARAM["pose_walking_threshold"]] = reward_cfg.pose_walking_threshold
        params[_TASK_PARAM["pose_running_threshold"]] = reward_cfg.pose_running_threshold
        params[_TASK_PARAM["height_scan_max_distance"]] = self.cfg_obj.observations.terrain_scan_max_distance
        np.copyto(state.task_params, params)

        flags = np.zeros_like(state.task_flags, dtype=np.float32)
        flags[_TASK_FLAG["rough_terrain"]] = 1.0 if self.cfg_obj.terrain_type == "rough" else 0.0
        flags[_TASK_FLAG["terrain_normal_upright"]] = (
            1.0 if self.cfg_obj.terrain_normal_upright.enabled and self._height_scan_dim > 0 else 0.0
        )
        flags[_TASK_FLAG["illegal_contact_enabled"]] = 1.0 if self.cfg_obj.illegal_contact.enabled else 0.0
        np.copyto(state.task_flags, flags)

    def _pose_std_from_table(self, table: Mapping[str, float]) -> np.ndarray:
        values = np.full(self.num_actions, 0.3, dtype=np.float32)
        for index, joint_name in enumerate(self.joint_names):
            for pattern, value in table.items():
                if re.fullmatch(pattern, joint_name):
                    values[index] = float(value)
                    break
        return values

    def _find_robot_node(self):
        root = self.context.root
        robot = root.find(self.cfg_obj.robot_name) if root is not None else None
        if robot is None:
            robot = _find_node_by_name(root, self.cfg_obj.robot_name)
        if robot is None:
            raise RuntimeError(f"Gobot scene has no robot named {self.cfg_obj.robot_name!r}")
        return robot

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
        if self.cfg_obj.terrain_type == "flat":
            return 0.0
        return self._terrain_sampler.height_at(float(x), float(y))

    def _terrain_heights(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        if self.cfg_obj.terrain_type == "flat":
            return np.zeros((points.shape[0],), dtype=np.float32)
        return self._terrain_sampler.heights_at(points)

    def _sample_spawn_index(self, env_id: int) -> int:
        if not self.cfg_obj.terrain_curriculum:
            return int(self._rng.integers(0, self._spawn_origins.shape[0]))
        warmup_progress = 0.10
        if self._curriculum_progress < warmup_progress and self._terrain_curriculum_limits[env_id] <= 0.0:
            return self._warmup_spawn_index
        difficulty_progress = (self._curriculum_progress - warmup_progress) / max(1.0 - warmup_progress, 1.0e-6)
        allowed_level = max(float(np.clip(difficulty_progress, 0.0, 1.0)), float(self._terrain_curriculum_limits[env_id]))
        candidates = np.flatnonzero(self._spawn_levels <= allowed_level + 1.0e-6)
        if candidates.size == 0:
            candidates = self._spawn_order[:1]
        if self._warmup_spawn_index not in candidates:
            candidates = np.concatenate([candidates, np.asarray([self._warmup_spawn_index], dtype=np.int64)])
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
        env_ids = np.arange(self.num_envs, dtype=np.int64)
        self._reset_envs(env_ids, np.zeros(self.num_envs, dtype=np.int64))
        batch_state = self.backend.state
        self.backend.compute_observations()
        obs = np.asarray(batch_state.actor_obs, dtype=np.float32).copy()
        critic = np.asarray(batch_state.critic_obs, dtype=np.float32).copy()
        if self.noise_cfg.level > 0.0 and self.cfg_obj.observations.actor_noise:
            obs = self._apply_actor_obs_noise(obs)
            critic[:, : self.num_obs] = obs
        return obs, critic

    def _apply_pushes(self) -> None:
        for env_id in range(self.num_envs):
            self._maybe_apply_push(env_id)

    def _base_motion_batch(
        self,
        batch_state: VelocityBatchRuntimeState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base_lin_vel_b = _rotate_vec_batch_by_quat_inv(batch_state.base_linear_velocity, batch_state.base_quaternion)
        base_ang_vel_b = _rotate_vec_batch_by_quat_inv(batch_state.base_angular_velocity, batch_state.base_quaternion)
        projected_gravity = _rotate_vec_batch_by_quat_inv(
            np.broadcast_to(np.asarray([0.0, 0.0, -1.0], dtype=np.float32), (self.num_envs, 3)),
            batch_state.base_quaternion,
        )
        return base_lin_vel_b, base_ang_vel_b, projected_gravity

    def _reset_env(self, env_id: int, *, reason: int) -> None:
        self._reset_envs(np.asarray([env_id], dtype=np.int64), np.asarray([reason], dtype=np.int64))

    def _reset_envs(self, env_ids: np.ndarray, reasons: np.ndarray) -> None:
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        reasons = np.asarray(reasons, dtype=np.int64).reshape(-1)
        if env_ids.size == 0:
            return
        if reasons.size != env_ids.size:
            raise ValueError("reset reasons must have the same length as env_ids")

        reset_count = int(env_ids.size)
        base_positions = np.zeros((reset_count, 3), dtype=np.float32)
        base_orientations = np.zeros((reset_count, 4), dtype=np.float32)
        base_linear_velocities = np.zeros((reset_count, 3), dtype=np.float32)
        base_angular_velocities = np.zeros((reset_count, 3), dtype=np.float32)
        joint_positions = np.zeros((reset_count, self.num_actions), dtype=np.float32)
        joint_velocities = np.zeros((reset_count, self.num_actions), dtype=np.float32)
        joint_targets = np.broadcast_to(self.default_joint_pos.reshape(1, -1), (reset_count, self.num_actions)).astype(np.float32, copy=True)

        for row, env_id_value in enumerate(env_ids):
            env_id = int(env_id_value)
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
            if self.cfg_obj.domain_randomization.enabled:
                lo, hi = self.cfg_obj.domain_randomization.encoder_bias_range
                self._encoder_bias[env_id] = self._rng.uniform(lo, hi, self.num_actions).astype(np.float32)
            else:
                self._encoder_bias[env_id] = 0.0

            base_positions[row] = np.asarray([float(spawn[0]), float(spawn[1]), float(base_z)], dtype=np.float32)
            base_orientations[row] = _quat_from_yaw(yaw)
            base_linear_velocities[row] = reset_lin_vel
            base_angular_velocities[row] = reset_ang_vel
            joint_positions[row] = self.default_joint_pos + self._rng.uniform(-0.05, 0.05, self.num_actions).astype(np.float32)
            joint_velocities[row] = self._rng.uniform(-0.05, 0.05, self.num_actions).astype(np.float32)

            self._spawn_indices[env_id] = spawn_index
            self._terrain_levels[env_id] = float(self._spawn_levels[spawn_index])
            self._episode_start_xy[env_id] = spawn[:2]
            self._episode_length_np[env_id] = 0
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
            self._reset_reasons[env_id] = int(reasons[row])

        self.backend.reset_robot_states(
            env_ids.tolist(),
            base_positions=base_positions,
            base_orientations=base_orientations,
            base_linear_velocities=base_linear_velocities,
            base_angular_velocities=base_angular_velocities,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_position_targets=joint_targets,
        )
        state = self.backend.state
        rows = env_ids.astype(np.int64, copy=False)
        state.encoder_bias[rows] = self._encoder_bias[rows]
        state.previous_action[rows] = 0.0
        state.last_action[rows] = 0.0
        state.action[rows] = 0.0
        state.foot_air_time[rows] = 0.0
        state.foot_peak_height[rows] = 0.0
        state.last_foot_contact[rows] = 0.0
        state.previous_foot_position[rows] = 0.0
        self.command_manager.reset(env_ids)
        np.copyto(state.command, self.command_manager.command_b)
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
        if hasattr(self.backend, "set_base_velocity") and hasattr(self.backend.state, "base_linear_velocity"):
            batch = self.backend.state
            linear_velocity = np.asarray(batch.base_linear_velocity[env_id], dtype=np.float32).copy()
            angular_velocity = np.asarray(batch.base_angular_velocity[env_id], dtype=np.float32).copy()
            linear_velocity += self._sample_range_vec(self.cfg_obj.push_velocity_ranges, ("x", "y", "z"))
            angular_velocity += self._sample_range_vec(self.cfg_obj.push_velocity_ranges, ("roll", "pitch", "yaw"))
            self.backend.set_base_velocity(env_id, linear_velocity, angular_velocity)
            self._push_count[env_id] += 1
            self._reset_push_timer(env_id)
            return
        state = self._runtime_state(env_id)
        position = _as_vec(state.base.get("global_transform", {}).get("position"), 3)
        orientation = _quat(state.base)
        linear_velocity = _as_vec(state.base.get("linear_velocity"), 3)
        angular_velocity = _as_vec(state.base.get("angular_velocity"), 3)
        linear_velocity += self._sample_range_vec(self.cfg_obj.push_velocity_ranges, ("x", "y", "z"))
        angular_velocity += self._sample_range_vec(self.cfg_obj.push_velocity_ranges, ("roll", "pitch", "yaw"))
        self.backend.reset_link_state(
            env_id,
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
        self._update_terrain_curriculum_limit_from_position(
            env_id,
            _as_vec(state.base.get("global_transform", {}).get("position"), 3),
            reset_reason,
        )

    def _update_terrain_curriculum_limit_from_position(
        self,
        env_id: int,
        base_position: np.ndarray,
        reset_reason: int,
    ) -> None:
        if not self.cfg_obj.terrain_curriculum:
            self._terrain_curriculum_limits[env_id] = 1.0
            return
        base_pos = np.asarray(base_position, dtype=np.float32).reshape(3)
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
            self.backend.set_joint_position_target(env_id, joint_name, float(target))

    def _apply_actions(self, actions: np.ndarray) -> None:
        targets = self._target_positions_from_actions(actions)
        backend = getattr(self, "backend", None)
        if backend is not None:
            backend.set_position_targets(targets)
            return
        self.runtime.set_joint_position_targets(targets)

    def _batch_runtime_state(self) -> VelocityBatchRuntimeState:
        return self.backend.refresh()

    def _command_states_from_batch(self, batch: VelocityBatchRuntimeState) -> list[VelocityRuntimeState]:
        states: list[VelocityRuntimeState] = []
        for env_id in range(self.num_envs):
            states.append(
                VelocityRuntimeState(
                    robot={},
                    base={
                        "global_transform": {
                            "position": batch.base_position[env_id],
                            "quaternion": batch.base_quaternion[env_id],
                        },
                        "linear_velocity": batch.base_linear_velocity[env_id],
                        "angular_velocity": batch.base_angular_velocity[env_id],
                    },
                    joints={},
                    links={},
                    sensors={},
                    contacts=[],
                )
            )
        return states

    def _runtime_state(self, env_id: int) -> VelocityRuntimeState:
        runtime = self.backend.env_state(env_id)
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
        return build_velocity_actor_observation(
            base_lin_vel_b=self._obs_noise(base_lin_vel, "base_lin_vel", corrupt=corrupt),
            base_ang_vel_b=self._obs_noise(base_ang_vel, "base_ang_vel", corrupt=corrupt),
            projected_gravity=self._obs_noise(projected_gravity, "projected_gravity", corrupt=corrupt),
            joint_pos_rel=self._obs_noise(joint_pos_rel, "joint_pos", corrupt=corrupt),
            joint_vel=self._obs_noise(joint_vel, "joint_vel", corrupt=corrupt),
            last_action=self._last_actions[env_id],
            command=self.command_manager.command_b[env_id],
            height_scan=self._obs_noise(height_scan, "height_scan", corrupt=corrupt) / max(self.cfg_obj.observations.terrain_scan_max_distance, 1.0e-6),
        )

    def _critic_obs_for(self, env_id: int, state: VelocityRuntimeState, actor_obs: np.ndarray) -> np.ndarray:
        return build_velocity_critic_observation(
            actor_obs=actor_obs,
            foot_height=self._foot_heights(state),
            foot_air_time=self._foot_air_time[env_id],
            foot_contact=self._foot_contacts(state),
            foot_contact_forces=self._foot_contact_forces(state).reshape(-1),
        )

    def _actor_obs_batch(
        self,
        batch: VelocityBatchRuntimeState,
        base_lin_vel_b: np.ndarray,
        base_ang_vel_b: np.ndarray,
        projected_gravity: np.ndarray,
        *,
        corrupt: bool,
    ) -> np.ndarray:
        joint_pos_rel = batch.joint_position + self._encoder_bias - self.default_joint_pos.reshape(1, -1)
        height_scan = self._height_scan_batch(batch)
        parts = [
            self._obs_noise(base_lin_vel_b, "base_lin_vel", corrupt=corrupt),
            self._obs_noise(base_ang_vel_b, "base_ang_vel", corrupt=corrupt),
            self._obs_noise(projected_gravity, "projected_gravity", corrupt=corrupt),
            self._obs_noise(joint_pos_rel, "joint_pos", corrupt=corrupt),
            self._obs_noise(batch.joint_velocity, "joint_vel", corrupt=corrupt),
            self._last_actions,
            self.command_manager.command_b,
            self._obs_noise(height_scan, "height_scan", corrupt=corrupt)
            / max(self.cfg_obj.observations.terrain_scan_max_distance, 1.0e-6),
        ]
        return np.concatenate([np.asarray(part, dtype=np.float32).reshape(self.num_envs, -1) for part in parts], axis=1)

    def _critic_obs_batch(
        self,
        actor_obs: np.ndarray,
        foot_heights: np.ndarray,
        foot_contacts: np.ndarray,
        foot_forces: np.ndarray,
    ) -> np.ndarray:
        log_forces = np.sign(foot_forces) * np.log1p(np.abs(foot_forces))
        parts = [
            actor_obs,
            foot_heights,
            self._foot_air_time,
            foot_contacts,
            log_forces.reshape(self.num_envs, -1),
        ]
        return np.concatenate([np.asarray(part, dtype=np.float32).reshape(self.num_envs, -1) for part in parts], axis=1)

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
        if sensor_name is None:
            return np.zeros(0, dtype=np.float32)
        if self._height_scan_dim == 0:
            raise RuntimeError(f"Go1 height scan sensor {sensor_name!r} is configured but has no channels")
        sensor = state.sensors.get(sensor_name)
        if sensor is None:
            raise RuntimeError(f"Go1 runtime state is missing configured height scan sensor {sensor_name!r}")
        frame_z = _as_vec(sensor.get("global_transform", {}).get("position"), 3)[2]
        heights: list[float] = []
        for hit in sensor.get("hits", []):
            if not hit.get("hit", False):
                heights.append(self.cfg_obj.observations.terrain_scan_max_distance)
                continue
            point_z = _as_vec(hit.get("point"), 3)[2]
            heights.append(float(frame_z - point_z))
        if len(heights) != self._height_scan_dim:
            raise RuntimeError(
                f"Go1 height scan sensor {sensor_name!r} produced {len(heights)} hits, "
                f"expected {self._height_scan_dim}"
            )
        return np.asarray(heights, dtype=np.float32)

    def _height_scan_batch(self, batch: VelocityBatchRuntimeState) -> np.ndarray:
        sensor_name = self.cfg_obj.observations.height_scan_sensor
        if sensor_name is None:
            return np.zeros((self.num_envs, 0), dtype=np.float32)
        if self._height_scan_dim == 0:
            raise RuntimeError(f"Go1 height scan sensor {sensor_name!r} is configured but has no channels")
        if hasattr(batch, "height_scan"):
            height_scan = np.asarray(batch.height_scan, dtype=np.float32)
            if height_scan.shape[1] != self._height_scan_dim:
                raise RuntimeError(
                    f"Go1 height scan sensor {sensor_name!r} produced {height_scan.shape[1]} hits, "
                    f"expected {self._height_scan_dim}"
                )
            return height_scan
        sensor_index = int(self._height_scan_sensor_index)
        if sensor_index < 0:
            raise RuntimeError(f"Go1 runtime state is missing configured height scan sensor {sensor_name!r}")
        hit_count = int(batch.sensor_value_count[sensor_index])
        if hit_count != self._height_scan_dim:
            raise RuntimeError(
                f"Go1 height scan sensor {sensor_name!r} produced {hit_count} hits, expected {self._height_scan_dim}"
            )
        frame_z = batch.sensor_position[:, sensor_index, 2:3]
        point_z = batch.sensor_hit_point[:, sensor_index, : self._height_scan_dim, 2]
        heights = frame_z - point_z
        hit_mask = batch.sensor_hit[:, sensor_index, : self._height_scan_dim]
        return np.where(hit_mask, heights, self.cfg_obj.observations.terrain_scan_max_distance).astype(np.float32)

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

    def _foot_heights_batch(self, batch: VelocityBatchRuntimeState) -> np.ndarray:
        if hasattr(batch, "foot_height"):
            return np.asarray(batch.foot_height, dtype=np.float32)
        values = np.zeros((self.num_envs, self._foot_count), dtype=np.float32)
        for foot_index, sensor_index in enumerate(self._foot_height_sensor_indices):
            if sensor_index < 0 or batch.sensor_values.shape[2] == 0:
                continue
            values[:, foot_index] = batch.sensor_values[:, int(sensor_index), 0]
        return values

    def _foot_contacts(self, state: VelocityRuntimeState) -> np.ndarray:
        values = []
        for foot in self.cfg_obj.foot_names:
            sensor = state.sensors.get(f"{foot}_foot_contact")
            sensor_values = np.asarray(sensor.get("values", []) if sensor is not None else [], dtype=np.float32)
            values.append(1.0 if sensor_values.size and float(sensor_values[0]) > 1.0e-5 else 0.0)
        return np.asarray(values, dtype=np.float32)

    def _foot_contacts_batch(self, batch: VelocityBatchRuntimeState) -> np.ndarray:
        if hasattr(batch, "foot_contact"):
            return np.asarray(batch.foot_contact, dtype=np.float32)
        values = np.zeros((self.num_envs, self._foot_count), dtype=np.float32)
        for foot_index, sensor_index in enumerate(self._foot_contact_sensor_indices):
            if sensor_index < 0 or batch.sensor_values.shape[2] == 0:
                continue
            values[:, foot_index] = (batch.sensor_values[:, int(sensor_index), 0] > 1.0e-5).astype(np.float32)
        return values

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

    def _foot_positions_batch(self, batch: VelocityBatchRuntimeState) -> np.ndarray:
        if hasattr(batch, "foot_position"):
            return np.asarray(batch.foot_position, dtype=np.float32)
        return batch.link_position[:, self._foot_link_indices, :]

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

    def _foot_contact_forces_batch(self, batch: VelocityBatchRuntimeState) -> np.ndarray:
        if hasattr(batch, "foot_contact_force"):
            return np.asarray(batch.foot_contact_force, dtype=np.float32)
        foot_positions = self._foot_positions_batch(batch)
        forces = np.zeros((self.num_envs, self._foot_count, 3), dtype=np.float32)
        max_contacts = batch.contact_position.shape[1]
        for env_id in range(self.num_envs):
            contact_count = min(int(batch.contact_count[env_id]), max_contacts)
            for contact_index in range(contact_count):
                links = batch.contact_link_index[env_id, contact_index]
                if np.all(links < 0):
                    continue
                position = batch.contact_position[env_id, contact_index]
                distances = np.linalg.norm(foot_positions[env_id] - position.reshape(1, 3), axis=1)
                foot_index = int(np.argmin(distances))
                if distances[foot_index] > 0.18:
                    continue
                forces[env_id, foot_index] += batch.contact_force[env_id, contact_index]
        return forces

    def _update_contact_summary(self, env_id: int, state: VelocityRuntimeState) -> None:
        summary = self._contact_summary(state)
        self._illegal_contact_counts[env_id] = summary["illegal"]
        self._self_collision_counts[env_id] = summary["self_collision"]
        self._shank_collision_counts[env_id] = summary["shank"]
        self._trunk_head_collision_counts[env_id] = summary["trunk_head"]

    def _update_contact_summary_batch(self, batch: VelocityBatchRuntimeState) -> None:
        cfg = self.cfg_obj.illegal_contact
        self._illegal_contact_counts[:] = 0.0
        self._self_collision_counts[:] = 0.0
        self._shank_collision_counts[:] = 0.0
        self._trunk_head_collision_counts[:] = 0.0
        if not cfg.enabled:
            return
        if hasattr(batch, "illegal_contact_count"):
            self._illegal_contact_counts[:] = np.asarray(batch.illegal_contact_count, dtype=np.float32)
            self._self_collision_counts[:] = np.asarray(batch.self_collision_count, dtype=np.float32)
            self._shank_collision_counts[:] = np.asarray(batch.shank_collision_count, dtype=np.float32)
            self._trunk_head_collision_counts[:] = np.asarray(batch.trunk_head_collision_count, dtype=np.float32)
            return
        max_contacts = batch.contact_link_index.shape[1]
        for env_id in range(self.num_envs):
            contact_count = min(int(batch.contact_count[env_id]), max_contacts)
            for contact_index in range(contact_count):
                link_indices = [int(index) for index in batch.contact_link_index[env_id, contact_index] if int(index) >= 0]
                if not link_indices:
                    continue
                normal_force = abs(float(batch.contact_normal_force[env_id, contact_index]))
                magnitude = normal_force if normal_force > 0.0 else float(np.linalg.norm(batch.contact_force[env_id, contact_index]))
                is_self_collision = len(link_indices) == 2
                if is_self_collision and magnitude >= cfg.self_collision_force_threshold:
                    self._self_collision_counts[env_id] += 1.0
                    continue
                if magnitude < cfg.ground_force_threshold:
                    continue
                for link_index in link_indices:
                    link_name = batch.link_names[link_index] if link_index < len(batch.link_names) else ""
                    if cfg.terminate_on_thigh and self._matches_any(link_name, cfg.thigh_link_patterns):
                        self._illegal_contact_counts[env_id] += 1.0
                    if self._matches_any(link_name, cfg.shank_link_patterns) and not self._contact_near_any_foot_batch(batch, env_id, contact_index):
                        self._shank_collision_counts[env_id] += 1.0
                    if self._matches_any(link_name, cfg.trunk_head_link_patterns):
                        self._trunk_head_collision_counts[env_id] += 1.0

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

    def _contact_near_any_foot_batch(self, batch: VelocityBatchRuntimeState, env_id: int, contact_index: int) -> bool:
        foot_positions = self._foot_positions_batch(batch)[env_id]
        position = batch.contact_position[env_id, contact_index]
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

    def _update_foot_history_batch(self, batch: VelocityBatchRuntimeState) -> None:
        positions = self._foot_positions_batch(batch)
        uninitialized = ~np.any(self._previous_foot_positions, axis=(1, 2))
        if np.any(uninitialized):
            self._previous_foot_positions[uninitialized] = positions[uninitialized]
        self._foot_velocities = (positions - self._previous_foot_positions) / max(self.step_dt, 1.0e-6)
        self._previous_foot_positions = positions.copy()
        contacts = self._foot_contacts_batch(batch)
        heights = self._foot_heights_batch(batch)
        forces = self._foot_contact_forces_batch(batch)
        first_contact = (contacts > 0.0) & (self._last_foot_contact <= 0.0)
        self._first_contact = first_contact.astype(np.float32)
        self._landing_force = np.linalg.norm(forces, axis=2) * self._first_contact
        for env_id in range(self.num_envs):
            cursor = int(self._foot_history_cursor[env_id] % self._contact_history_length)
            self._foot_contact_history[env_id, cursor] = contacts[env_id]
            self._foot_force_history[env_id, cursor] = forces[env_id]
            self._foot_history_cursor[env_id] = (cursor + 1) % self._contact_history_length
        in_air = contacts <= 0.0
        self._foot_air_time = np.where(in_air, self._foot_air_time + self.step_dt, 0.0)
        self._foot_peak_height = np.where(in_air, np.maximum(self._foot_peak_height, heights), self._foot_peak_height)

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

    def _compute_reward_batch(
        self,
        batch: VelocityBatchRuntimeState,
        action: np.ndarray,
        base_lin_vel_b: np.ndarray,
        base_ang_vel_b: np.ndarray,
        projected_gravity: np.ndarray,
        foot_heights: np.ndarray,
        foot_contacts: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        reward_cfg = self.cfg_obj.rewards
        command = self.command_manager.command_b
        lin_error = np.sum(np.square(command[:, :2] - base_lin_vel_b[:, :2]), axis=1) + np.square(base_lin_vel_b[:, 2])
        ang_error = np.square(command[:, 2] - base_ang_vel_b[:, 2]) + np.sum(np.square(base_ang_vel_b[:, :2]), axis=1)
        upright_error = self._upright_error_batch(batch, projected_gravity)
        action_rate = np.sum(np.square(action - self._previous_actions), axis=1)
        command_speed = np.linalg.norm(command[:, :2], axis=1) + np.abs(command[:, 2])

        pose_std = self._pose_std_batch(command_speed)
        pose_error = np.mean(
            np.square(batch.joint_position - self.default_joint_pos.reshape(1, -1))
            / np.square(np.maximum(pose_std, 1.0e-6)),
            axis=1,
        )
        foot_clearance_cost = np.sum(
            np.abs(foot_heights - reward_cfg.foot_target_height)
            * np.linalg.norm(self._foot_velocities[:, :, :2], axis=2),
            axis=1,
        )
        active = (command_speed > reward_cfg.command_threshold).astype(np.float32)
        foot_slip = self._foot_slip_cost_batch(foot_contacts) * active
        first_contact = self._first_contact > 0.0
        landing_force = np.sum(self._landing_force, axis=1) * active
        swing_error = np.square(self._foot_peak_height / max(reward_cfg.foot_target_height, 1.0e-6) - 1.0)
        swing_cost = np.sum(swing_error * first_contact.astype(np.float32), axis=1) * active
        self._foot_peak_height = np.where(first_contact, 0.0, self._foot_peak_height)
        self._last_foot_contact = foot_contacts.copy()

        dof_limit_cost = self._joint_limit_cost_batch(batch)
        breakdown = {
            "track_linear_velocity": reward_cfg.track_linear_velocity * np.exp(-lin_error / reward_cfg.lin_vel_std**2),
            "track_angular_velocity": reward_cfg.track_angular_velocity * np.exp(-ang_error / reward_cfg.ang_vel_std**2),
            "upright": reward_cfg.upright * np.exp(-upright_error / reward_cfg.upright_std**2),
            "pose": reward_cfg.pose * np.exp(-pose_error),
            "body_ang_vel": reward_cfg.body_ang_vel * np.sum(np.square(base_ang_vel_b[:, :2]), axis=1),
            "dof_pos_limits": reward_cfg.dof_pos_limits * dof_limit_cost,
            "action_rate_l2": reward_cfg.action_rate_l2 * action_rate,
            "air_time": reward_cfg.air_time
            * np.sum((self._foot_air_time > 0.05) & (self._foot_air_time < 0.5), axis=1).astype(np.float32)
            * active,
            "foot_clearance": reward_cfg.foot_clearance * foot_clearance_cost * active,
            "foot_swing_height": reward_cfg.foot_swing_height * swing_cost,
            "foot_slip": reward_cfg.foot_slip * foot_slip,
            "soft_landing": reward_cfg.soft_landing * landing_force,
            "self_collisions": reward_cfg.self_collisions * self._self_collision_counts,
            "shank_collision": reward_cfg.shank_collision * self._shank_collision_counts,
            "trunk_head_collision": reward_cfg.trunk_head_collision * self._trunk_head_collision_counts,
        }
        reward = np.zeros(self.num_envs, dtype=np.float32)
        for value in breakdown.values():
            reward += np.asarray(value, dtype=np.float32)
        return reward * self.step_dt, {key: np.asarray(value, dtype=np.float32) for key, value in breakdown.items()}

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

    def _pose_std_batch(self, command_speed: np.ndarray) -> np.ndarray:
        cfg = self.cfg_obj.rewards
        values = np.full((self.num_envs, self.num_actions), 0.3, dtype=np.float32)
        standing = command_speed < cfg.pose_walking_threshold
        walking = (command_speed >= cfg.pose_walking_threshold) & (command_speed < cfg.pose_running_threshold)
        running = command_speed >= cfg.pose_running_threshold
        self._fill_pose_std(values, standing, cfg.pose_std_standing)
        self._fill_pose_std(values, walking, cfg.pose_std_walking)
        self._fill_pose_std(values, running, cfg.pose_std_running)
        return values

    def _fill_pose_std(self, values: np.ndarray, env_mask: np.ndarray, table: Mapping[str, float]) -> None:
        if not np.any(env_mask):
            return
        for joint_index, joint_name in enumerate(self.joint_names):
            for pattern, value in table.items():
                if re.fullmatch(pattern, joint_name):
                    values[env_mask, joint_index] = float(value)
                    break

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

    def _joint_limit_cost_batch(self, batch: VelocityBatchRuntimeState) -> np.ndarray:
        lower_cost = np.where(np.isfinite(batch.joint_lower_limit.reshape(1, -1)), batch.joint_lower_limit.reshape(1, -1) - batch.joint_position, 0.0)
        upper_cost = np.where(np.isfinite(batch.joint_upper_limit.reshape(1, -1)), batch.joint_position - batch.joint_upper_limit.reshape(1, -1), 0.0)
        return np.sum(np.maximum(lower_cost, 0.0) + np.maximum(upper_cost, 0.0), axis=1).astype(np.float32)

    def _foot_slip_cost(self, env_id: int, foot_contacts: np.ndarray) -> float:
        vel_xy = np.linalg.norm(self._foot_velocities[env_id, :, :2], axis=1)
        return float(np.sum(np.square(vel_xy) * foot_contacts))

    def _foot_slip_cost_batch(self, foot_contacts: np.ndarray) -> np.ndarray:
        vel_xy = np.linalg.norm(self._foot_velocities[:, :, :2], axis=2)
        return np.sum(np.square(vel_xy) * foot_contacts, axis=1).astype(np.float32)

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

    def _upright_error_batch(self, batch: VelocityBatchRuntimeState, projected_gravity: np.ndarray) -> np.ndarray:
        if self.cfg_obj.terrain_normal_upright.enabled and self._height_scan_dim > 0:
            normal_w = self._terrain_normal_from_scan_batch(batch)
            body_normal = _rotate_vec_batch_by_quat_inv(normal_w, batch.base_quaternion)
            error = np.sum(np.square(body_normal[:, :2]), axis=1).astype(np.float32)
        else:
            error = np.sum(np.square(projected_gravity[:, :2]), axis=1).astype(np.float32)
        self._terrain_normal_error = error
        return error

    def _terrain_normal_from_scan(self, state: VelocityRuntimeState) -> np.ndarray:
        sensor_name = self.cfg_obj.observations.height_scan_sensor
        if sensor_name is None:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        sensor = state.sensors.get(sensor_name)
        if sensor is None:
            raise RuntimeError(f"Go1 runtime state is missing configured height scan sensor {sensor_name!r}")
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

    def _terrain_normal_from_scan_batch(self, batch: VelocityBatchRuntimeState) -> np.ndarray:
        sensor_name = self.cfg_obj.observations.height_scan_sensor
        if sensor_name is None:
            return np.broadcast_to(np.asarray([0.0, 0.0, 1.0], dtype=np.float32), (self.num_envs, 3)).copy()
        if hasattr(batch, "height_scan_point"):
            points = np.asarray(batch.height_scan_point, dtype=np.float32)[:, : self._height_scan_dim, :]
            normals = np.asarray(batch.height_scan_normal, dtype=np.float32)[:, : self._height_scan_dim, :]
            hits = np.asarray(batch.height_scan_hit, dtype=bool)[:, : self._height_scan_dim]
        else:
            sensor_index = int(self._height_scan_sensor_index)
            if sensor_index < 0:
                raise RuntimeError(f"Go1 runtime state is missing configured height scan sensor {sensor_name!r}")
            points = batch.sensor_hit_point[:, sensor_index, : self._height_scan_dim, :]
            normals = batch.sensor_hit_normal[:, sensor_index, : self._height_scan_dim, :]
            hits = batch.sensor_hit[:, sensor_index, : self._height_scan_dim]
        min_hits = max(3, int(self.cfg_obj.terrain_normal_upright.min_hit_count))
        result = np.zeros((self.num_envs, 3), dtype=np.float32)
        result[:, 2] = 1.0
        for env_id in range(self.num_envs):
            hit_points = points[env_id, hits[env_id]]
            if hit_points.shape[0] >= min_hits:
                cloud = hit_points.astype(np.float64)
                centered = cloud - np.mean(cloud, axis=0)
                try:
                    _, _, vh = np.linalg.svd(centered, full_matrices=False)
                    normal = vh[-1]
                    if normal[2] < 0.0:
                        normal = -normal
                    length = np.linalg.norm(normal)
                    if length > 1.0e-6 and np.all(np.isfinite(normal)):
                        result[env_id] = (normal / length).astype(np.float32)
                        continue
                except np.linalg.LinAlgError:
                    pass
            valid_normals = normals[env_id, hits[env_id]]
            if valid_normals.size:
                lengths = np.linalg.norm(valid_normals, axis=1)
                valid_normals = valid_normals[(lengths > 1.0e-6) & np.all(np.isfinite(valid_normals), axis=1)]
                if valid_normals.size:
                    normal = np.mean(valid_normals.astype(np.float64), axis=0)
                    if normal[2] < 0.0:
                        normal = -normal
                    length = np.linalg.norm(normal)
                    if length > 1.0e-6 and np.all(np.isfinite(normal)):
                        result[env_id] = (normal / length).astype(np.float32)
        return result

    def _base_clearance(self, state: VelocityRuntimeState) -> float:
        position = _as_vec(state.base.get("global_transform", {}).get("position"), 3)
        return float(position[2] - self._terrain_height(position[0], position[1]))

    def _base_clearance_batch(self, batch: VelocityBatchRuntimeState) -> np.ndarray:
        base_position = np.asarray(batch.base_position, dtype=np.float32)
        return (base_position[:, 2] - self._terrain_heights(base_position[:, :2])).astype(np.float32)

    def _terminated(self, env_id: int, state: VelocityRuntimeState) -> bool:
        if self.cfg_obj.terrain_type == "rough":
            if self._base_clearance(state) < self.cfg_obj.min_base_clearance:
                return True
            if self.cfg_obj.illegal_contact.enabled and self._illegal_contact_counts[env_id] > 0.0:
                return True
            return False
        roll, pitch = _quat_to_rp(_quat(state.base))
        return bool(self._base_clearance(state) < self.cfg_obj.min_base_clearance or abs(roll) > math.radians(70.0) or abs(pitch) > math.radians(70.0))

    def _terminated_batch(
        self,
        batch: VelocityBatchRuntimeState,
        projected_gravity: np.ndarray,
        base_clearance: np.ndarray,
    ) -> np.ndarray:
        if self.cfg_obj.terrain_type == "rough":
            terminated = base_clearance < self.cfg_obj.min_base_clearance
            if self.cfg_obj.illegal_contact.enabled:
                terminated = terminated | (self._illegal_contact_counts > 0.0)
            return terminated.astype(bool)
        roll_pitch = _quat_to_rp_batch(batch.base_quaternion)
        return (
            (base_clearance < self.cfg_obj.min_base_clearance)
            | (np.abs(roll_pitch[:, 0]) > math.radians(70.0))
            | (np.abs(roll_pitch[:, 1]) > math.radians(70.0))
        ).astype(bool)


def _rotate_vec_batch_by_quat_inv(vectors: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    quaternions = np.asarray(quaternions, dtype=np.float32)
    xyz = -quaternions[:, 1:4]
    w = quaternions[:, 0:1]
    t = 2.0 * np.cross(xyz, vectors)
    return (vectors + w * t + np.cross(xyz, t)).astype(np.float32)


def _quat_to_rp_batch(quaternions: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions, dtype=np.float32)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return np.stack([roll, pitch], axis=1).astype(np.float32)


__all__ = ["Go1VelocityEnv", "VelocityRuntimeState", "VelocityBatchRuntimeState"]
