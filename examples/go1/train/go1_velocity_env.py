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
    RewardTermSpec,
    SpecField,
    TaskJitCompiler,
    TaskExpression,
    TaskLayout,
    TerminationSpec,
    task_buffer,
)
from gobot.rl.locomotion import (
    HeightScan,
    LocomotionBatchEnv,
    LocomotionBatchSpec,
    LocomotionControlCfg,
    LocomotionNoiseCfg,
    NativeLocomotionBatchBackend,
    TerrainSampler,
    velocity_actor_observation_schema,
    velocity_critic_observation_schema,
)
from gobot.rl.locomotion.math import (
    _find_node_by_name,
    _quat_from_yaw,
)

try:
    from .go1_velocity_cfg import Go1VelocityCfg, go1_rough_velocity_cfg
except ImportError:
    from go1_velocity_cfg import Go1VelocityCfg, go1_rough_velocity_cfg

try:
    from .go1_task_kernels import go1_velocity_task
except ImportError:
    from go1_task_kernels import go1_velocity_task

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
        collect_step_extras: bool = True,
        task_kernel: str = "jit",
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
        self.collect_step_extras = bool(collect_step_extras)
        self.task_kernel_mode = str(task_kernel)
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
        self.actor_obs_schema = velocity_actor_observation_schema(self.num_actions, self._height_scan_dim)
        self.critic_obs_schema = velocity_critic_observation_schema(self.num_actions, self._height_scan_dim, self._foot_count)
        self.action_spec = self._make_action_spec()
        self.observation_spec = self.actor_obs_schema
        self.critic_observation_spec = self.critic_obs_schema
        self.num_obs = self.actor_obs_schema.dim
        self.num_privileged_obs = self.critic_obs_schema.dim
        self.backend = self._make_batch_backend()
        self.backend.configure(self.num_envs)
        self.resolved_sim_workers = self.backend.resolved_workers(self.sim_workers)
        self._configure_native_task_buffers()
        self._configure_native_command()
        self.task_ir = self._make_task_ir()
        self.task_ir.validate_native_arrays(self.backend.state)
        self.compiled_task_kernel = None
        self._compiled_task_kernel_installed = False
        self.task_kernel_info = self._configure_task_kernel()

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
            "task_ir": self.task_ir.metadata(),
            "task_ir_version": self.task_ir.version,
            "task_ir_backend": self.task_ir.backend,
            "task_kernel": dict(self.task_kernel_info),
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

    def _configure_task_kernel(self) -> dict[str, Any]:
        mode = self.task_kernel_mode.lower()
        if mode != "jit":
            raise ValueError("task_kernel must be 'jit'")

        self.compiled_task_kernel = TaskJitCompiler().compile(
            self.task_ir,
            self.backend.state,
            kernel=go1_velocity_task,
        )
        self.backend.install_task_kernel(self.compiled_task_kernel)
        self._compiled_task_kernel_installed = True

        build_info = self.compiled_task_kernel.build_info
        info: dict[str, Any] = {
            "mode": mode,
            "compiled": True,
            "installed": True,
            "backend": getattr(build_info, "backend", mode),
            "cache_key": build_info.cache_key,
            "cache_hit": build_info.cache_hit,
            "compile_ms": build_info.compile_ms,
            "array_count": len(build_info.array_specs),
        }
        if hasattr(build_info, "compiler"):
            info["compiler"] = build_info.compiler
        if hasattr(build_info, "library_path"):
            info["library_path"] = str(build_info.library_path)
        if hasattr(build_info, "object_path"):
            info["object_path"] = str(build_info.object_path)
        if hasattr(build_info, "source_path"):
            info["source_path"] = str(build_info.source_path)
        return info

    @property
    def command_b(self) -> np.ndarray:
        backend = getattr(self, "backend", None)
        if backend is not None:
            return backend.state.command
        command_manager = getattr(self, "command_manager", None)
        if command_manager is not None and hasattr(command_manager, "command_b"):
            return command_manager.command_b
        return np.zeros((self.num_envs, 3), dtype=np.float32)

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
        self._run_task_kernel()
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
        backend_step_actions_ms = 0.0
        jit_task_kernel_ms = 0.0
        action_step_t0 = self.perf_counter()
        self.backend.step_task_kernel(
            action_np,
            self.decimation,
            workers=self.sim_workers,
            simulate_action_latency=self.control_cfg.simulate_action_latency,
        )
        backend_step_actions_ms = (self.perf_counter() - action_step_t0) * 1000.0
        native_step_total_ms = (self.perf_counter() - t0) * 1000.0
        self._mark_profile(profile_marks, "physics")
        t0 = self.perf_counter()
        batch_state = self.backend.state
        backend_refresh_cache_ms = (self.perf_counter() - t0) * 1000.0
        self._mark_profile(profile_marks, "state")
        step_core_ms = backend_action_ms + native_step_total_ms + backend_refresh_cache_ms
        native_profile = self.backend.step_profile()

        t0 = self.perf_counter()
        self._mark_profile(profile_marks, "command")

        foot_heights = batch_state.foot_height
        foot_contacts = batch_state.foot_contact
        self._mark_profile(profile_marks, "contact")
        rewards = np.asarray(batch_state.reward, dtype=np.float32).copy()
        reward_terms = {}
        if self.collect_step_extras:
            reward_terms = {
                name: np.asarray(batch_state.reward_terms[:, index], dtype=np.float32)
                for index, name in enumerate(_REWARD_TERM_NAMES)
            }
        self._mark_profile(profile_marks, "reward")
        log_values: dict[str, float] = {}
        if self.collect_step_extras:
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
        if self.collect_step_extras:
            log_values["terrain_curriculum_limit"] = float(np.mean(self._terrain_curriculum_limits))

        active_envs = ~done_envs
        batch_state.previous_action[active_envs] = batch_state.last_action[active_envs]
        batch_state.last_action[active_envs] = batch_state.submitted_action[active_envs]
        self._previous_actions[active_envs] = batch_state.previous_action[active_envs]
        self._last_actions[active_envs] = batch_state.last_action[active_envs]

        if reset_env_ids.size:
            batch_state = self.backend.state
        self._mark_profile(profile_marks, "termination_reset")
        reset_done_ms = (self.perf_counter() - t0) * 1000.0

        t0 = self.perf_counter()
        if reset_env_ids.size:
            self._run_task_kernel()
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
        log_info = self._step_log_info(batch_state, reset_reason, log_values) if self.collect_step_extras else {}
        extras = {
            "time_outs": time_outs.copy(),
            "log": log_info,
        }
        if reward_terms:
            extras["reward_terms"] = {name: value.copy() for name, value in reward_terms.items()}
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
                "reset_reason": reset_reason,
                "time_outs": time_outs,
                "log": extras["log"],
            },
        )
        if reward_terms:
            self._state.info["reward_terms"] = reward_terms
        self.step_counter += 1
        timing = self._state.info.setdefault("timing", {})
        timing["env_step_total_ms"] = (self.perf_counter() - step_t0) * 1000.0
        timing["apply_action_ms"] = apply_action_ms
        timing["step_core_ms"] = step_core_ms
        timing["backend_apply_action_ms"] = backend_action_ms
        timing["backend_physics_ms"] = native_step_total_ms
        timing["native_step_total_ms"] = native_step_total_ms
        timing["backend_step_actions_ms"] = backend_step_actions_ms
        timing["jit_task_kernel_ms"] = jit_task_kernel_ms
        timing["backend_refresh_cache_ms"] = backend_refresh_cache_ms
        timing["update_state_ms"] = update_state_ms + observation_ms
        timing["reset_done_ms"] = reset_done_ms
        for key, value in native_profile.items():
            timing[f"native_{key}"] = float(value)
        return self._state

    def close(self) -> None:
        self.context.clear_world()
        self.context.clear_scene()

    def set_training_progress(self, policy_steps: int) -> None:
        self._total_policy_steps = max(0, int(policy_steps))
        self._apply_command_curriculum()
        self._update_curriculum_progress()

    def _step_log_info(self, batch_state: Any, reset_reason: np.ndarray, log_values: Mapping[str, float]) -> dict[str, np.ndarray]:
        return {
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
            "/velocity/command_vx": np.asarray(float(np.mean(batch_state.command[:, 0])), dtype=np.float32),
            "/velocity/command_vy": np.asarray(float(np.mean(batch_state.command[:, 1])), dtype=np.float32),
            "/velocity/command_yaw": np.asarray(float(np.mean(batch_state.command[:, 2])), dtype=np.float32),
        }

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

    def _make_task_ir(self) -> TaskLayout:
        reward_cfg = self.cfg_obj.rewards
        reward_inputs = {
            "track_linear_velocity": ("command", "base_linear_velocity_body"),
            "track_angular_velocity": ("command", "base_angular_velocity_body"),
            "upright": ("projected_gravity", "height_scan_normal"),
            "pose": ("joint_position", "default_joint_position", "pose_std_standing", "pose_std_walking", "pose_std_running"),
            "body_ang_vel": ("base_angular_velocity_body",),
            "dof_pos_limits": ("joint_position", "joint_lower_limit", "joint_upper_limit"),
            "action_rate_l2": ("submitted_action", "last_action"),
            "air_time": ("foot_air_time", "first_contact", "command"),
            "foot_clearance": ("foot_height", "foot_contact"),
            "foot_swing_height": ("foot_peak_height", "foot_contact"),
            "foot_slip": ("foot_slip",),
            "soft_landing": ("landing_force",),
            "self_collisions": ("self_collision_count",),
            "shank_collision": ("shank_collision_count",),
            "trunk_head_collision": ("trunk_head_collision_count",),
        }
        reward_params = {
            "track_linear_velocity": {"std": reward_cfg.lin_vel_std},
            "track_angular_velocity": {"std": reward_cfg.ang_vel_std},
            "upright": {
                "std": reward_cfg.upright_std,
                "terrain_normal": self.cfg_obj.terrain_normal_upright.enabled and self._height_scan_dim > 0,
            },
            "pose": {
                "walking_threshold": reward_cfg.pose_walking_threshold,
                "running_threshold": reward_cfg.pose_running_threshold,
            },
            "air_time": {"command_threshold": reward_cfg.command_threshold},
            "foot_clearance": {"target_height": reward_cfg.foot_target_height},
            "foot_swing_height": {"target_height": reward_cfg.foot_target_height},
            "soft_landing": {"ground_force_threshold": self.cfg_obj.illegal_contact.ground_force_threshold},
        }
        reward_terms = tuple(
            RewardTermSpec(
                name=name,
                weight=weight,
                expression=TaskExpression(
                    f"go1_velocity.{name}",
                    reward_inputs.get(name, ()),
                    reward_params.get(name, {}),
                ),
                scale_by_dt=True,
            )
            for name, weight in zip(_REWARD_TERM_NAMES, self._reward_weights(), strict=True)
        )
        terminations = [
            TerminationSpec(
                "base_clearance",
                TaskExpression("less_than", ("base_clearance",), {"threshold": self.cfg_obj.min_base_clearance}),
            ),
        ]
        if self.cfg_obj.terrain_type == "flat":
            terminations.append(
                TerminationSpec(
                    "flat_roll_pitch_limit",
                    TaskExpression("roll_pitch_limit", ("projected_gravity",), {"threshold_rad": math.radians(70.0)}),
                )
            )
        if self.cfg_obj.illegal_contact.enabled:
            terminations.append(
                TerminationSpec(
                    "illegal_contact",
                    TaskExpression("greater_than", ("illegal_contact_count",), {"threshold": 0.0}),
                )
            )
        buffers = (
            task_buffer("target_position", "env", "dof", role="action_target"),
            task_buffer("action", "env", "dof", role="action"),
            task_buffer("submitted_action", "env", "dof", role="action"),
            task_buffer("previous_action", "env", "dof", role="history"),
            task_buffer("last_action", "env", "dof", role="history"),
            task_buffer("default_joint_position", "dof", role="config"),
            task_buffer("action_scale", "dof", role="config"),
            task_buffer("encoder_bias", "env", "dof", role="randomization"),
            task_buffer("command", "env", 3, role="command"),
            task_buffer("command_world", "env", 3, role="command"),
            task_buffer("heading_commands", "env", role="command"),
            task_buffer("command_heading_error", "env", role="command"),
            task_buffer("command_time_left", "env", role="command"),
            task_buffer("command_is_heading_env", "env", dtype="uint8", role="command"),
            task_buffer("command_is_standing_env", "env", dtype="uint8", role="command"),
            task_buffer("command_is_world_env", "env", dtype="uint8", role="command"),
            task_buffer("command_is_forward_env", "env", dtype="uint8", role="command"),
            task_buffer("command_ranges", "command_range", role="config"),
            task_buffer("gait_phase", "env", 2, role="locomotion_optional"),
            task_buffer("feet_phase_height_target", "env", 2, role="locomotion_optional"),
            task_buffer("pose_weights", "dof", role="config"),
            task_buffer("pose_std_standing", "dof", role="config"),
            task_buffer("pose_std_walking", "dof", role="config"),
            task_buffer("pose_std_running", "dof", role="config"),
            task_buffer("reward_weights", len(_REWARD_TERM_NAMES), role="config"),
            task_buffer("task_params", len(_TASK_PARAM), role="config"),
            task_buffer("task_flags", len(_TASK_FLAG), role="config"),
            task_buffer("reset_base_position", "env", 3, role="reset"),
            task_buffer("reset_base_quaternion", "env", 4, role="reset"),
            task_buffer("reset_base_linear_velocity", "env", 3, role="reset"),
            task_buffer("reset_base_angular_velocity", "env", 3, role="reset"),
            task_buffer("reset_joint_position", "env", "dof", role="reset"),
            task_buffer("reset_joint_velocity", "env", "dof", role="reset"),
            task_buffer("base_position", "env", 3, role="base_state"),
            task_buffer("base_quaternion", "env", 4, role="base_state"),
            task_buffer("base_linear_velocity", "env", 3, role="base_state"),
            task_buffer("base_angular_velocity", "env", 3, role="base_state"),
            task_buffer("base_linear_velocity_body", "env", 3, role="base_state"),
            task_buffer("base_angular_velocity_body", "env", 3, role="base_state"),
            task_buffer("projected_gravity", "env", 3, role="base_state"),
            task_buffer("base_height", "env", role="base_state"),
            task_buffer("joint_position", "env", "dof", role="dof_state"),
            task_buffer("joint_velocity", "env", "dof", role="dof_state"),
            task_buffer("qacc", "env", "dof", role="dof_state"),
            task_buffer("torques", "env", "dof", role="dof_state"),
            task_buffer("joint_lower_limit", "dof", role="dof_state"),
            task_buffer("joint_upper_limit", "dof", role="dof_state"),
            task_buffer("foot_position", "env", "foot", 3, role="foot_state"),
            task_buffer("feet_quat", "env", "foot", 4, role="foot_state"),
            task_buffer("foot_velocity", "env", "foot", 3, role="foot_state"),
            task_buffer("foot_height", "env", "foot", role="foot_state"),
            task_buffer("foot_contact", "env", "foot", role="foot_state"),
            task_buffer("foot_contact_force", "env", "foot", 3, role="foot_state"),
            task_buffer("height_scan", "env", self._height_scan_dim, role="terrain"),
            task_buffer("height_scan_hit", "env", self._height_scan_dim, dtype="uint8", role="terrain"),
            task_buffer("height_scan_point", "env", self._height_scan_dim, 3, role="terrain"),
            task_buffer("height_scan_normal", "env", self._height_scan_dim, 3, role="terrain"),
            task_buffer("illegal_contact_count", "env", role="termination"),
            task_buffer("self_collision_count", "env", role="reward"),
            task_buffer("shank_collision_count", "env", role="reward"),
            task_buffer("trunk_head_collision_count", "env", role="reward"),
            task_buffer("foot_air_time", "env", "foot", role="reward"),
            task_buffer("foot_peak_height", "env", "foot", role="reward"),
            task_buffer("last_foot_contact", "env", "foot", role="history"),
            task_buffer("first_contact", "env", "foot", role="reward"),
            task_buffer("landing_force", "env", "foot", role="reward"),
            task_buffer("previous_foot_position", "env", "foot", 3, role="history"),
            task_buffer("reward", "env", role="output"),
            task_buffer("terminated", "env", dtype="uint8", role="output"),
            task_buffer("base_clearance", "env", role="termination"),
            task_buffer("velocity_error", "env", role="reward"),
            task_buffer("foot_slip", "env", role="reward"),
            task_buffer("terrain_normal_error", "env", role="reward"),
            task_buffer("reward_terms", "env", len(_REWARD_TERM_NAMES), role="output"),
            task_buffer("actor_obs", "env", self.num_obs, role="output"),
            task_buffer("critic_obs", "env", self.num_privileged_obs, role="output"),
        )
        return TaskLayout(
            name=self.cfg_obj.name,
            version="go1_velocity_task_ir_v1",
            action_spec=self.action_spec,
            obs_groups={"actor": self.actor_obs_schema, "critic": self.critic_obs_schema},
            buffers=buffers,
            reward_terms=reward_terms,
            terminations=tuple(terminations),
            backend="gobot_native_cpu_fused",
        )

    def _apply_actions(self, actions: np.ndarray) -> None:
        targets = self.target_positions_from_actions(actions)
        backend = getattr(self, "backend", None)
        if backend is not None and hasattr(backend, "set_position_targets"):
            backend.set_position_targets(targets)
            return
        runtime = getattr(self, "runtime", None)
        if runtime is not None and hasattr(runtime, "set_joint_position_targets"):
            runtime.set_joint_position_targets(targets)
            return
        raise RuntimeError("Go1VelocityEnv has no backend/runtime action target sink")

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
                reward_term_count=len(_REWARD_TERM_NAMES),
                task_param_count=len(_TASK_PARAM),
                task_flag_count=len(_TASK_FLAG),
                actor_obs_dim=self.num_obs,
                critic_obs_dim=self.num_privileged_obs,
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
        reward_weights = np.asarray(self._reward_weights(), dtype=np.float32)
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

    def _reward_weights(self) -> tuple[float, ...]:
        reward_cfg = self.cfg_obj.rewards
        return (
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
        )

    def _configure_native_command(self) -> None:
        command_cfg = self.cfg_obj.command
        ranges = command_cfg.ranges
        self.backend.configure_command(
            step_dt=self.step_dt,
            resampling_time_range=command_cfg.resampling_time_range,
            lin_vel_x=ranges.lin_vel_x,
            lin_vel_y=ranges.lin_vel_y,
            ang_vel_z=ranges.ang_vel_z,
            heading=ranges.heading,
            rel_standing_envs=command_cfg.rel_standing_envs,
            rel_heading_envs=command_cfg.rel_heading_envs,
            rel_world_envs=command_cfg.rel_world_envs,
            rel_forward_envs=command_cfg.rel_forward_envs,
            heading_command=command_cfg.heading_command,
            heading_control_stiffness=command_cfg.heading_control_stiffness,
            seed=self.seed + 17_171,
        )
        self._apply_command_curriculum()

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
        if hasattr(self, "backend"):
            self.backend.set_command_ranges(
                lin_vel_x=ranges.lin_vel_x,
                lin_vel_y=ranges.lin_vel_y,
                ang_vel_z=ranges.ang_vel_z,
            )

    def _reset_all(self):
        env_ids = np.arange(self.num_envs, dtype=np.int64)
        self._reset_envs(env_ids, np.zeros(self.num_envs, dtype=np.int64))
        batch_state = self.backend.state
        self._run_task_kernel()
        obs = np.asarray(batch_state.actor_obs, dtype=np.float32).copy()
        critic = np.asarray(batch_state.critic_obs, dtype=np.float32).copy()
        if self.noise_cfg.level > 0.0 and self.cfg_obj.observations.actor_noise:
            obs = self._apply_actor_obs_noise(obs)
            critic[:, : self.num_obs] = obs
        return obs, critic

    def _run_task_kernel(self) -> None:
        self.backend.run_task_kernel()

    def _apply_pushes(self) -> None:
        for env_id in range(self.num_envs):
            self._maybe_apply_push(env_id)

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
        self.backend.reset_commands(env_ids.tolist())

    def _sample_range_vec(self, ranges: Mapping[str, tuple[float, float]], names: Sequence[str]) -> np.ndarray:
        values = []
        for name in names:
            lo, hi = ranges.get(name, (0.0, 0.0))
            values.append(float(self._rng.uniform(lo, hi)))
        return np.asarray(values, dtype=np.float32)

    def _height_scan(self, state: VelocityBatchRuntimeState) -> np.ndarray:
        if self._height_scan_dim <= 0:
            return np.zeros((0,), dtype=np.float32)
        sensor_name = self.cfg_obj.observations.height_scan_sensor
        if sensor_name is None:
            return np.zeros((0,), dtype=np.float32)
        sensors = getattr(state, "sensors", None) if not isinstance(state, Mapping) else state.get("sensors", {})
        sensor = sensors.get(sensor_name) if isinstance(sensors, Mapping) else None
        if sensor is None:
            raise RuntimeError(f"missing configured height scan sensor {sensor_name!r}")
        hits = sensor.get("hits", []) if isinstance(sensor, Mapping) else []
        if len(hits) != self._height_scan_dim:
            raise RuntimeError(f"height scan sensor {sensor_name!r} expected {self._height_scan_dim} hits, got {len(hits)}")
        values = np.zeros((self._height_scan_dim,), dtype=np.float32)
        for index, hit in enumerate(hits):
            if not isinstance(hit, Mapping) or not bool(hit.get("hit", False)):
                values[index] = 0.0
                continue
            distance = hit.get("distance")
            if distance is not None:
                values[index] = float(distance)
                continue
            point = np.asarray(hit.get("point", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
            values[index] = float(point[2])
        return HeightScan(self._height_scan_dim, self.cfg_obj.observations.terrain_scan_max_distance).normalize(values)

    def _terrain_normal_from_scan(self, state: VelocityBatchRuntimeState) -> np.ndarray:
        if self._height_scan_dim <= 0:
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        sensor_name = self.cfg_obj.observations.height_scan_sensor
        if sensor_name is None:
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        sensors = getattr(state, "sensors", None) if not isinstance(state, Mapping) else state.get("sensors", {})
        sensor = sensors.get(sensor_name) if isinstance(sensors, Mapping) else None
        if sensor is None:
            raise RuntimeError(f"missing configured height scan sensor {sensor_name!r}")
        hits = sensor.get("hits", []) if isinstance(sensor, Mapping) else []
        points: list[np.ndarray] = []
        mask: list[bool] = []
        for hit in hits:
            if not isinstance(hit, Mapping):
                continue
            points.append(np.asarray(hit.get("point", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3))
            mask.append(bool(hit.get("hit", False)))
        if not points:
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        return HeightScan(len(points)).terrain_normal(np.asarray(points, dtype=np.float32), np.asarray(mask, dtype=bool))

    def _contact_summary(self, state: VelocityBatchRuntimeState) -> dict[str, float]:
        contacts = getattr(state, "contacts", None) if not isinstance(state, Mapping) else state.get("contacts", [])
        if contacts is None:
            contacts = []
        cfg = self.cfg_obj.illegal_contact
        summary = {
            "illegal": 0.0,
            "self_collision": 0.0,
            "shank": 0.0,
            "trunk_head": 0.0,
        }
        for contact in contacts:
            if not isinstance(contact, Mapping):
                continue
            normal_force = float(contact.get("normal_force", 0.0))
            if normal_force < float(cfg.ground_force_threshold):
                continue
            robot_name = str(contact.get("robot_name", ""))
            other_robot_name = str(contact.get("other_robot_name", ""))
            link_name = str(contact.get("link_name", ""))
            other_link_name = str(contact.get("other_link_name", ""))
            if robot_name == self.cfg_obj.robot_name and other_robot_name == self.cfg_obj.robot_name:
                summary["self_collision"] += 1.0
                continue
            robot_link = link_name if robot_name == self.cfg_obj.robot_name else other_link_name
            if not robot_link:
                continue
            if cfg.terminate_on_thigh and self._matches_any(robot_link, cfg.thigh_link_patterns):
                summary["illegal"] += 1.0
            if self._matches_any(robot_link, cfg.shank_link_patterns):
                summary["shank"] += 1.0
            if self._matches_any(robot_link, cfg.trunk_head_link_patterns):
                summary["trunk_head"] += 1.0
        return summary

    @staticmethod
    def _matches_any(name: str, patterns: Sequence[str]) -> bool:
        return any(re.fullmatch(pattern, name) for pattern in patterns)

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
        batch = self.backend.state
        linear_velocity = np.asarray(batch.base_linear_velocity[env_id], dtype=np.float32).copy()
        angular_velocity = np.asarray(batch.base_angular_velocity[env_id], dtype=np.float32).copy()
        linear_velocity += self._sample_range_vec(self.cfg_obj.push_velocity_ranges, ("x", "y", "z"))
        angular_velocity += self._sample_range_vec(self.cfg_obj.push_velocity_ranges, ("roll", "pitch", "yaw"))
        self.backend.set_base_velocity(env_id, linear_velocity, angular_velocity)
        self._push_count[env_id] += 1
        self._reset_push_timer(env_id)

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
        commanded_speed = float(np.linalg.norm(self.command_b[env_id, :2]))
        expected_distance = max(0.25, commanded_speed * survival * self.max_episode_length * self.step_dt * 0.5)

        level = float(self._terrain_curriculum_limits[env_id])
        if reset_reason == 2 or survival > 0.75 or distance > expected_distance:
            level += level_step
        elif reset_reason == 1 and survival < 0.25:
            level -= level_step
        self._terrain_curriculum_limits[env_id] = float(np.clip(level, 0.0, 1.0))

    def _update_terrain_curriculum_limit(
        self,
        env_id: int,
        state: VelocityBatchRuntimeState,
        *,
        reset_reason: int,
    ) -> None:
        base = getattr(state, "base", None) if not isinstance(state, Mapping) else state.get("base", {})
        transform = base.get("global_transform", {}) if isinstance(base, Mapping) else {}
        position = transform.get("position", [0.0, 0.0, 0.0]) if isinstance(transform, Mapping) else [0.0, 0.0, 0.0]
        self._update_terrain_curriculum_limit_from_position(env_id, np.asarray(position, dtype=np.float32), reset_reason)

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


__all__ = ["Go1VelocityEnv", "VelocityRuntimeState", "VelocityBatchRuntimeState"]
