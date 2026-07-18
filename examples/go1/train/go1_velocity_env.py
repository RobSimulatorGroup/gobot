"""Go1 NumPy batch environment for velocity locomotion training."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import gobot

from gobot.rl import (
    ActionSpec,
    BatchEnvState,
    BatchSimulationRuntime,
    ObservationSpec,
    SpecField,
    TaskRuntimeMetadata,
)
from gobot.rl.locomotion import (
    LocomotionBatchEnv,
    LocomotionBatchSpec,
    LocomotionControlCfg,
    LocomotionNoiseCfg,
    NativeLocomotionBatchBackend,
    velocity_actor_observation_schema,
    velocity_critic_observation_schema,
)
from gobot.rl.locomotion.math import (
    _find_node_by_name,
    _quat_from_yaw,
)

from .go1_velocity_cfg import (
    GO1_ROUGH_REWARD_TERM_NAMES,
    GO1_TASK_VERSION,
    Go1VelocityCfg,
    go1_velocity_cfg,
)
from .go1_gait import bound_gait_score_numpy, gait_foot_indices, gait_joint_indices
from .go1_scene_runtime import prepare_go1_scene
from .go1_training_state import (
    build_terrain_curriculum_state,
    restore_terrain_curriculum_assignments,
)

_GO1_HIP_INDICES = np.asarray([0, 3, 6, 9], dtype=np.int64)

_TASK_PARAM = {
    "step_dt": 0,
    "lin_vel_std2": 1,
    "ang_vel_std2": 2,
    "upright_std2": 3,
    "command_threshold": 4,
    "height_scan_max_distance": 5,
}

_TASK_FLAG = {"rough_terrain": 0}


def _quat_rotate_inv_batch(vectors: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    quaternions = np.asarray(quaternions, dtype=np.float32)
    q_xyz = -quaternions[..., 1:4]
    q_w = quaternions[..., 0:1]
    t = 2.0 * np.cross(q_xyz, vectors)
    return (vectors + q_w * t + np.cross(q_xyz, t)).astype(np.float32, copy=False)


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
        task_runtime: str | None = None,
        context: gobot.AppContext | None = None,
    ) -> None:
        self.cfg_obj = cfg if cfg is not None else go1_velocity_cfg()
        joint_names = tuple(self.cfg_obj.joint_names)
        default_joint_pos = np.asarray(self.cfg_obj.default_joint_pos, dtype=np.float32)
        if len(joint_names) == 0 or default_joint_pos.shape != (len(joint_names),):
            raise ValueError("velocity task joint_names and default_joint_pos must be non-empty and have the same length")
        super().__init__(
            num_envs=int(num_envs),
            seed=int(seed),
            control_cfg=LocomotionControlCfg(
                action_scale=self.cfg_obj.action_scale,
                simulate_action_latency=bool(self.cfg_obj.simulate_action_latency),
            ),
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
        self.task_runtime_mode = self._resolve_task_runtime(task_runtime)
        self.physics_dt = float(self.cfg_obj.physics_dt)
        self.decimation = int(self.cfg_obj.decimation)
        self.step_dt = self.physics_dt * self.decimation
        self.max_episode_length = int(max_episode_length or math.ceil(float(self.cfg_obj.episode_length_s) / self.step_dt))
        self._episode_length_np = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_length_buf = self._episode_length_np
        self._episode_start_xy = np.zeros((self.num_envs, 2), dtype=np.float32)
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.common_step_counter = 0
        self.profile_step_counter = 0
        self._reward_term_names = GO1_ROUGH_REWARD_TERM_NAMES + (
            ("run_progress", "bound_gait")
            if self.cfg_obj.training_profile == "run"
            else ()
        )
        self._advance_task_time = False

        self._foot_count = len(self.cfg_obj.foot_names)
        self._gait_foot_indices = gait_foot_indices(self.cfg_obj.foot_names)
        self._gait_joint_indices = gait_joint_indices(self.joint_names)
        self._encoder_bias = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)
        self._push_time_left = np.zeros(self.num_envs, dtype=np.float32)
        self._push_step_left = np.zeros(self.num_envs, dtype=np.int64)
        self._push_count = np.zeros(self.num_envs, dtype=np.int64)
        self._spawn_indices = np.zeros(self.num_envs, dtype=np.int64)
        self._spawn_type_cols = np.zeros(self.num_envs, dtype=np.int64)
        self._spawn_env_levels = np.zeros(self.num_envs, dtype=np.int64)
        self._terrain_levels = np.zeros(self.num_envs, dtype=np.float32)
        self._terrain_curriculum_limits = np.zeros(self.num_envs, dtype=np.float32)
        self._reset_reasons = np.zeros(self.num_envs, dtype=np.int64)
        self._current_command_stage = 0
        self.project_path = Path(self.cfg_obj.project_path).resolve()
        self.context, self.robot, self._runtime_terrain_node = prepare_go1_scene(
            self.cfg_obj,
            context=context,
        )
        self._terrain_config = self._read_terrain_generator_config()
        self._spawn_origins = self._load_spawn_origins()
        self._spawn_grid_shape = self._infer_spawn_grid_shape()
        self._initialize_spawn_assignments()
        self._spawn_difficulties = self._spawn_difficulty_scores()
        max_difficulty = max(float(np.max(self._spawn_difficulties)), 1.0e-6)
        self._spawn_levels = np.clip(self._spawn_difficulties / max_difficulty, 0.0, 1.0).astype(np.float32)

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
        self.actor_obs_schema, self.critic_obs_schema = self._make_observation_schemas()
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
        self._apply_startup_domain_randomization()
        self.task_runtime_metadata = self._make_task_runtime_metadata()
        self.task_runtime_info = self._configure_task_runtime()

        self.cfg = {
            "name": self.cfg_obj.name,
            "source": "examples.go1.train.go1_velocity_env",
            "task": self.cfg_obj.name,
            "obs_schema_version": self.actor_obs_schema.version,
            "obs_names": self.actor_obs_schema.names,
            "critic_obs_names": self.critic_obs_schema.names,
            "obs_spec": self.actor_obs_schema.metadata(),
            "critic_obs_spec": self.critic_obs_schema.metadata(),
            "action_spec": self.action_spec.metadata(),
            "task_runtime_metadata": self.task_runtime_metadata.metadata(),
            "task_runtime": dict(self.task_runtime_info),
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
            "mujoco_solver_settings": dict(self.cfg_obj.mujoco_solver_settings),
            "step_dt": self.step_dt,
            "action_clip": self.cfg_obj.action_clip,
            "illegal_contact": self.cfg_obj.illegal_contact.enabled,
            "domain_randomization": self.cfg_obj.domain_randomization.enabled,
            "push_enabled": self.cfg_obj.push_enabled,
            "push_interval_steps": self.cfg_obj.push_interval_steps,
            "push_interval_range_s": self.cfg_obj.push_interval_range_s,
            "terrain_out_of_bounds": self.cfg_obj.terrain_out_of_bounds,
            "terrain_distance_buffer": self.cfg_obj.terrain_distance_buffer,
            "training_profile": self.cfg_obj.training_profile,
            "run_environment_ratio": self.cfg_obj.command.rel_run_envs,
            "run_velocity_x": tuple(self.cfg_obj.command.run_velocity_x),
            "run_velocity_curriculum": tuple(
                (int(stage.step), stage.run_velocity_x)
                for stage in self.cfg_obj.run_command_curriculum
            ),
            "bound_gait_reward": self.cfg_obj.rewards.bound_gait,
            "run_progress_reward": self.cfg_obj.rewards.run_progress,
            "fast_batch_view": bool(getattr(self.backend, "is_fast", False)),
        }
        self.extras: dict[str, Any] = {}
        self._obs, self._critic_obs = self._reset_all()
        self._state_obs_actor = self._obs.copy()
        self._state_obs_critic = self._critic_obs.copy()
        self._state_reward = np.zeros((self.num_envs,), dtype=np.float32)
        self._state_terminated = np.zeros((self.num_envs,), dtype=bool)
        self._state_truncated = np.zeros((self.num_envs,), dtype=bool)
        self._state_steps = self._episode_length_np
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

    @staticmethod
    def _resolve_task_runtime(task_runtime: str | None) -> str:
        if task_runtime is None:
            task_runtime = "numpy"
        mode = str(task_runtime).lower()
        aliases = {
            "np": "numpy",
            "python": "numpy",
            "cpu": "numpy",
            "native": "numpy",
        }
        mode = aliases.get(mode, mode)
        if mode != "numpy":
            raise ValueError("task_runtime must be 'numpy'")
        return mode

    def _configure_task_runtime(self) -> dict[str, Any]:
        return self._task_runtime_info(
            mode="numpy",
            backend="gobot_native_cpu_batch_numpy",
        )

    def _task_runtime_info(
        self,
        *,
        mode: str,
        backend: str,
    ) -> dict[str, Any]:
        metadata = self.task_runtime_metadata
        arrays = getattr(self.backend, "_arrays", {})
        return {
            "mode": mode,
            "compiled": False,
            "installed": True,
            "backend": backend,
            "array_count": len(arrays),
            "array_names": tuple(sorted(str(name) for name in arrays)),
            "metadata": metadata.metadata(),
            "name": metadata.name,
            "version": metadata.version,
            "obs_groups_spec": dict(metadata.obs_groups_spec),
            "reward_names": tuple(metadata.reward_names),
        }

    @property
    def command_b(self) -> np.ndarray:
        backend = getattr(self, "backend", None)
        if backend is not None:
            return backend.state.command
        command_manager = getattr(self, "command_manager", None)
        if command_manager is not None and hasattr(command_manager, "command_b"):
            return command_manager.command_b
        return np.zeros((self.num_envs, 3), dtype=np.float32)

    def reset_seed(self, seed: int | None) -> None:
        super().reset_seed(seed)

    def _prepare_actions(self, actions: Any) -> np.ndarray:
        action_np = actions.detach().cpu().numpy() if hasattr(actions, "detach") else np.asarray(actions)
        action_np = np.asarray(action_np, dtype=np.float32).reshape(self.num_envs, self.num_actions)
        clip = getattr(self.cfg_obj, "action_clip", 1.0)
        self._submitted_actions = action_np.copy() if clip is None else np.clip(action_np, -float(clip), float(clip))
        return self._submitted_actions

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
        self.backend.update_command_frames()
        batch_state = self.backend.state
        self._run_task_runtime(advance_time=False)
        obs = np.asarray(batch_state.actor_obs, dtype=np.float32).copy()
        critic = np.asarray(batch_state.critic_obs, dtype=np.float32).copy()
        if self.noise_cfg.level > 0.0 and self.cfg_obj.observations.actor_noise:
            obs = self._apply_actor_obs_noise(obs)
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
        self.common_step_counter += 1
        self.profile_step_counter += 1

        t0 = self.perf_counter()
        action_np = self._prepare_actions(actions)
        apply_action_ms = (self.perf_counter() - t0) * 1000.0
        self._mark_profile(profile_marks, "action_prepare")

        t0 = self.perf_counter()
        self._mark_profile(profile_marks, "action_apply")
        backend_action_ms = (self.perf_counter() - t0) * 1000.0
        t0 = self.perf_counter()
        backend_step_actions_ms = 0.0
        task_runtime_ms = 0.0
        numpy_task_ms = 0.0
        action_step_t0 = self.perf_counter()
        self.backend.step_task_inputs(
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
        task_t0 = self.perf_counter()
        self._run_task_runtime(advance_time=True)
        task_runtime_ms = (self.perf_counter() - task_t0) * 1000.0
        numpy_task_ms = task_runtime_ms
        self._mark_profile(profile_marks, "state")
        step_core_ms = backend_action_ms + native_step_total_ms + backend_refresh_cache_ms + task_runtime_ms
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
                for index, name in enumerate(self._reward_term_names)
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
        terrain_out = self._terrain_out_of_bounds(batch_state)
        time_outs = ((self._episode_length_np >= self.max_episode_length) | terrain_out) & ~terminated
        done_envs = terminated | time_outs
        reset_reason = np.where(terminated, 1, np.where(terrain_out, 3, np.where(time_outs, 2, 0))).astype(np.int64)
        self._episode_returns += rewards
        reset_env_ids = np.flatnonzero(done_envs).astype(np.int64)
        if reset_env_ids.size:
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
        self._previous_actions[active_envs] = batch_state.previous_action[active_envs]
        self._last_actions[active_envs] = batch_state.last_action[active_envs]

        if reset_env_ids.size:
            batch_state = self.backend.state
        self.backend.advance_commands()
        self._apply_pushes()
        batch_state = self.backend.state
        self._mark_profile(profile_marks, "termination_reset")
        reset_done_ms = (self.perf_counter() - t0) * 1000.0

        t0 = self.perf_counter()
        actor_obs, critic_obs = self._build_generic_velocity_observations(batch_state)
        self._copy_obs(batch_state, actor_obs, critic_obs)
        obs_np = np.asarray(batch_state.actor_obs, dtype=np.float32).copy()
        critic_np = np.asarray(batch_state.critic_obs, dtype=np.float32).copy()
        if self.noise_cfg.level > 0.0 and self.cfg_obj.observations.actor_noise:
            obs_np = self._apply_actor_obs_noise(obs_np)
        self._mark_profile(profile_marks, "obs_build")
        self._obs = obs_np
        self._critic_obs = critic_np
        self._mark_profile(profile_marks, "tensor_convert")
        observation_ms = (self.perf_counter() - t0) * 1000.0

        log_info = self._step_log_info(batch_state, reset_reason, log_values) if self.collect_step_extras else {}
        if reward_terms:
            for name, value in reward_terms.items():
                log_info[f"reward/{name}"] = np.asarray(float(np.mean(value)), dtype=np.float32)
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
        timing["task_runtime_ms"] = task_runtime_ms
        timing["numpy_task_ms"] = numpy_task_ms
        timing["backend_refresh_cache_ms"] = backend_refresh_cache_ms
        timing["update_state_ms"] = update_state_ms + observation_ms
        timing["reset_done_ms"] = reset_done_ms
        for key, value in native_profile.items():
            timing[f"native_{key}"] = float(value)
        return self._state

    def close(self) -> None:
        backend = getattr(self, "backend", None)
        if backend is not None and hasattr(backend, "close"):
            backend.close()
        self.context.clear_world()
        self.context.clear_scene()

    def set_training_progress(self, common_steps: int) -> None:
        self.common_step_counter = max(0, int(common_steps))
        self._apply_command_curriculum()

    def training_state_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "backend": "mujoco-cpu",
            "num_envs": self.num_envs,
            "common_step_counter": int(self.common_step_counter),
            "training_profile": self.cfg_obj.training_profile,
            "profile_step_counter": int(self.profile_step_counter),
            "terrain_curriculum": build_terrain_curriculum_state(
                self._spawn_env_levels,
                self._spawn_type_cols,
                rows=self._spawn_grid_shape[0],
                cols=self._spawn_grid_shape[1],
            ),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_training_state_dict(self, state: Mapping[str, Any]) -> dict[str, Any]:
        version = int(state.get("version", 0))
        if version not in (0, 1, 2):
            raise RuntimeError(f"unsupported Go1 CPU training checkpoint version {version}")
        self.set_training_progress(int(state.get("common_step_counter", 0)))
        saved_profile = state.get("training_profile")
        profile_restored = saved_profile == self.cfg_obj.training_profile
        self.profile_step_counter = (
            max(0, int(state.get("profile_step_counter", 0)))
            if profile_restored
            else 0
        )
        self._apply_command_curriculum()
        terrain_state = state.get("terrain_curriculum")
        if not isinstance(terrain_state, Mapping):
            return {
                "common_step_counter": int(self.common_step_counter),
                "profile_curriculum": "exact" if profile_restored else "restarted",
                "terrain_curriculum": "legacy_initial_levels",
            }

        rows, cols = self._spawn_grid_shape
        levels, terrain_types, exact = restore_terrain_curriculum_assignments(
            terrain_state,
            self._spawn_env_levels,
            self._spawn_type_cols,
            rows=rows,
            cols=cols,
        )
        np.copyto(self._spawn_env_levels, levels)
        np.copyto(self._spawn_type_cols, terrain_types)
        rng_state = state.get("rng_state")
        if isinstance(rng_state, Mapping):
            self._rng.bit_generator.state = dict(rng_state)
        self.reset()
        return {
            "common_step_counter": int(self.common_step_counter),
            "profile_curriculum": "exact" if profile_restored else "restarted",
            "terrain_curriculum": "exact" if exact else "resampled_for_num_envs",
            "mean_terrain_level": float(np.mean(self._spawn_env_levels)),
        }

    def _step_log_info(self, batch_state: Any, reset_reason: np.ndarray, log_values: Mapping[str, float]) -> dict[str, np.ndarray]:
        command = np.asarray(batch_state.command, dtype=np.float32)
        gravity_down = np.asarray(batch_state.projected_gravity, dtype=np.float32)
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
            "/velocity/run_command_stage": np.asarray(float(self._current_run_command_stage), dtype=np.float32),
            "/velocity/reset_reason": np.asarray(float(np.mean(reset_reason)), dtype=np.float32),
            "/velocity/command_speed": np.asarray(float(np.mean(np.linalg.norm(command[:, :2], axis=1))), dtype=np.float32),
            "/velocity/command_yaw_abs": np.asarray(float(np.mean(np.abs(command[:, 2]))), dtype=np.float32),
            "/velocity/base_clearance": np.asarray(float(np.mean(batch_state.base_clearance)), dtype=np.float32),
            "/velocity/upright": np.asarray(float(np.mean(-gravity_down[:, 2])), dtype=np.float32),
            "/velocity/command_vx": np.asarray(float(np.mean(command[:, 0])), dtype=np.float32),
            "/velocity/command_vy": np.asarray(float(np.mean(command[:, 1])), dtype=np.float32),
            "/velocity/command_yaw": np.asarray(float(np.mean(command[:, 2])), dtype=np.float32),
            "/velocity/run_env_ratio": np.asarray(
                float(np.mean(self.backend.state.command_is_run_env)), dtype=np.float32
            ),
        }

    def _make_observation_schemas(self) -> tuple[ObservationSpec, ObservationSpec]:
        return (
            velocity_actor_observation_schema(self.num_actions, self._height_scan_dim),
            velocity_critic_observation_schema(self.num_actions, self._height_scan_dim, self._foot_count),
        )

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
        clip = getattr(self.cfg_obj, "action_clip", None)
        lower = -math.inf if clip is None else -float(clip)
        upper = math.inf if clip is None else float(clip)
        return ActionSpec(
            version=f"{self.actor_obs_schema.version}_action_v1",
            fields=tuple(SpecField(str(name), 1) for name in self.joint_names),
            lower=lower,
            upper=upper,
        )

    def _make_task_runtime_metadata(self) -> TaskRuntimeMetadata:
        return TaskRuntimeMetadata(
            name=self.cfg_obj.name,
            version=GO1_TASK_VERSION,
            obs_groups_spec={"actor": self.actor_obs_schema.dim, "critic": self.critic_obs_schema.dim},
            reward_names=self._reward_term_names,
            backend="gobot_native_cpu_batch_numpy",
            cache_info={
                "scene_source": "jscn",
                "scene_path": self.cfg_obj.scene_path,
                "native_contact_detail": "substep_contact_history",
                "domain_randomization_backend": "per_env_mjmodel_pool",
            },
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
        if self._state is not None and self._state.final_observation is not None:
            terminal_mask = self._state.info.get("_final_observation")
            if isinstance(terminal_mask, np.ndarray) and bool(np.any(terminal_mask)):
                info["final_observation"] = self._state.final_observation
                info["_final_observation"] = terminal_mask
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

    def _joint_gain_array(self, value: float | Sequence[float]) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 0:
            return np.full((self.num_actions,), float(array), dtype=np.float32)
        array = array.reshape(-1)
        if array.shape != (self.num_actions,):
            raise ValueError(f"joint gain array must have shape ({self.num_actions},), got {array.shape}")
        return array.astype(np.float32, copy=True)

    def _make_batch_backend(self):
        illegal_cfg = self.cfg_obj.illegal_contact
        if hasattr(self.context, "create_locomotion_batch_view"):
            spec = LocomotionBatchSpec(
                foot_link_names=self.cfg_obj.foot_link_names,
                foot_height_sensor_names=[f"{foot}_foot_height_scan" for foot in self.cfg_obj.foot_names],
                foot_contact_sensor_names=[f"{foot}_foot_contact" for foot in self.cfg_obj.foot_names],
                height_scan_sensor=self.cfg_obj.observations.height_scan_sensor,
                thigh_shape_patterns=illegal_cfg.thigh_shape_patterns,
                shank_shape_patterns=illegal_cfg.shank_shape_patterns,
                trunk_head_shape_patterns=illegal_cfg.trunk_head_shape_patterns,
                terminate_on_thigh_contact=illegal_cfg.terminate_on_thigh,
                ground_force_threshold=illegal_cfg.ground_force_threshold,
                self_collision_force_threshold=illegal_cfg.self_collision_force_threshold,
                reward_term_count=len(self._reward_term_names),
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
        if "action_clip" in getattr(self.backend, "_arrays", {}):
            clip = getattr(self.cfg_obj, "action_clip", None)
            state.action_clip[:] = -1.0 if clip is None else float(clip)
        state.pose_std_standing[:] = 1.0
        state.pose_std_walking[:] = 1.0
        state.pose_std_running[:] = 1.0
        reward_cfg = self.cfg_obj.rewards
        standing = self._pose_std_array(
            float(reward_cfg.pose_std_standing_hip_thigh),
            float(reward_cfg.pose_std_standing_calf),
        )
        walking = self._pose_std_array(
            float(reward_cfg.pose_std_walking_hip_thigh),
            float(reward_cfg.pose_std_walking_calf),
        )
        running = self._pose_std_array(
            float(reward_cfg.pose_std_running_hip_thigh),
            float(reward_cfg.pose_std_running_calf),
        )
        np.copyto(state.pose_std_standing, standing)
        np.copyto(state.pose_std_walking, walking)
        np.copyto(state.pose_std_running, running)

        reward_weights = np.zeros_like(state.reward_weights, dtype=np.float32)
        reward_weights[: len(self._reward_weights())] = np.asarray(self._reward_weights(), dtype=np.float32)
        np.copyto(state.reward_weights, reward_weights)

        params = np.zeros_like(state.task_params, dtype=np.float32)
        params[_TASK_PARAM["step_dt"]] = self.step_dt
        params[_TASK_PARAM["lin_vel_std2"]] = reward_cfg.lin_vel_std**2
        params[_TASK_PARAM["ang_vel_std2"]] = reward_cfg.ang_vel_std**2
        params[_TASK_PARAM["upright_std2"]] = reward_cfg.upright_std**2
        params[_TASK_PARAM["command_threshold"]] = reward_cfg.command_threshold
        params[_TASK_PARAM["height_scan_max_distance"]] = self.cfg_obj.observations.terrain_scan_max_distance
        np.copyto(state.task_params, params)

        flags = np.zeros_like(state.task_flags, dtype=np.float32)
        flags[_TASK_FLAG["rough_terrain"]] = 1.0
        np.copyto(state.task_flags, flags)

    def _reward_weights(self) -> tuple[float, ...]:
        reward_cfg = self.cfg_obj.rewards
        return tuple(float(getattr(reward_cfg, name)) for name in self._reward_term_names)

    def _pose_std_array(self, hip_thigh: float, calf: float) -> np.ndarray:
        std = np.full((self.num_actions,), float(hip_thigh), dtype=np.float32)
        for index, joint_name in enumerate(self.joint_names):
            if "calf" in joint_name:
                std[index] = float(calf)
        return std

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
            rel_run_envs=command_cfg.rel_run_envs,
            run_velocity_x_min=command_cfg.run_velocity_x[0],
            run_velocity_x_max=command_cfg.run_velocity_x[1],
            heading_command=command_cfg.heading_command,
            heading_control_stiffness=command_cfg.heading_control_stiffness,
            zero_small_xy_threshold=command_cfg.zero_small_xy_threshold,
            seed=self.seed + 17_171,
        )
        self._apply_command_curriculum()

    def _read_terrain_generator_config(self) -> dict[str, Any]:
        config = self._runtime_terrain_node.generator_config
        properties = config.get("properties", config)
        if not isinstance(properties, Mapping):
            raise RuntimeError("Gobot scene terrain generator metadata is invalid")
        return dict(properties)

    def _load_spawn_origins(self) -> np.ndarray:
        root = self.context.root
        terrain_world = root.find("terrain_world") if root is not None else None
        terrain = terrain_world.find("terrain") if terrain_world is not None else None
        origins = np.asarray(getattr(terrain, "spawn_origins", []), dtype=np.float64)
        if origins.ndim != 2 or origins.shape[1] != 3 or origins.shape[0] == 0:
            return np.zeros((1, 3), dtype=np.float64)
        return origins

    def _infer_spawn_grid_shape(self) -> tuple[int, int]:
        unique_x = np.unique(np.round(self._spawn_origins[:, 0], 6))
        unique_y = np.unique(np.round(self._spawn_origins[:, 1], 6))
        if unique_x.size * unique_y.size == self._spawn_origins.shape[0]:
            return (int(unique_x.size), int(unique_y.size))
        return (int(self._spawn_origins.shape[0]), 1)

    @staticmethod
    def _proportional_counts(count: int, proportions: Sequence[float]) -> np.ndarray:
        proportions_np = np.asarray(proportions, dtype=np.float64).reshape(-1)
        if proportions_np.size == 0:
            return np.zeros((0,), dtype=np.int64)
        proportions_np = np.maximum(proportions_np, 0.0)
        if not np.any(proportions_np > 0.0):
            proportions_np[:] = 1.0
        proportions_np /= np.sum(proportions_np)
        if count >= proportions_np.size:
            counts = np.ones(proportions_np.size, dtype=np.int64)
            remaining = int(count) - proportions_np.size
        else:
            counts = np.zeros(proportions_np.size, dtype=np.int64)
            remaining = int(count)
        if remaining > 0:
            ideal = proportions_np * remaining
            floor = np.floor(ideal).astype(np.int64)
            counts += floor
            leftover = remaining - int(np.sum(floor))
            if leftover > 0:
                order = np.argsort(-(ideal - floor))
                counts[order[:leftover]] += 1
        return counts

    def _initialize_spawn_assignments(self) -> None:
        rows, cols = self._spawn_grid_shape
        if rows <= 0 or cols <= 0:
            return
        rng = np.random.default_rng(self.seed)
        if not self.cfg_obj.terrain_curriculum:
            self._spawn_type_cols[:] = rng.integers(0, cols, size=self.num_envs).astype(np.int64)
            self._spawn_env_levels[:] = rng.integers(0, rows, size=self.num_envs).astype(np.int64)
            return
        sub_terrains = self._terrain_config.get("sub_terrains", [])
        proportions = [
            float(sub_cfg.get("proportion", 1.0))
            for sub_cfg in sub_terrains[:cols]
            if isinstance(sub_cfg, Mapping)
        ]
        type_counts = self._proportional_counts(
            self.num_envs,
            proportions if len(proportions) == cols else [1.0] * cols,
        )
        types = np.repeat(np.arange(cols, dtype=np.int64), type_counts)
        if types.size < self.num_envs:
            types = np.pad(types, (0, self.num_envs - types.size), mode="wrap")
        self._spawn_type_cols[:] = types[: self.num_envs]
        max_init_level_cfg = self.cfg_obj.max_init_terrain_level
        max_init_level = rows - 1 if max_init_level_cfg is None else min(int(max_init_level_cfg), rows - 1)
        self._spawn_env_levels[:] = rng.integers(0, max_init_level + 1, size=self.num_envs).astype(np.int64)

    def _spawn_difficulty_scores(self) -> np.ndarray:
        rows, cols = self._spawn_grid_shape
        if rows > 0 and cols > 0 and rows * cols == self._spawn_origins.shape[0]:
            return np.repeat(np.arange(rows, dtype=np.float32), cols)
        return np.zeros(self._spawn_origins.shape[0], dtype=np.float32)

    def _terrain_patch_size(self) -> tuple[float, float]:
        patch_size = np.asarray(
            self._terrain_config.get("patch_size", (8.0, 8.0)),
            dtype=np.float64,
        ).reshape(-1)
        if patch_size.size != 2 or not np.all(np.isfinite(patch_size)):
            raise RuntimeError("Gobot scene terrain generator has an invalid patch_size")
        return (float(patch_size[0]), float(patch_size[1]))

    def _terrain_out_of_bounds(self, state: Any) -> np.ndarray:
        if not self.cfg_obj.terrain_out_of_bounds:
            return np.zeros((self.num_envs,), dtype=bool)
        buffer = float(self.cfg_obj.terrain_distance_buffer)
        base_pos = np.asarray(state.base_position, dtype=np.float32)
        if base_pos.shape[0] != self.num_envs or base_pos.shape[1] < 2:
            return np.zeros((self.num_envs,), dtype=bool)
        rows, cols = self._spawn_grid_shape
        patch_x, patch_y = self._terrain_patch_size()
        border = float(self._terrain_config.get("border_width", 0.0))
        limit_x = max(0.0, 0.5 * rows * patch_x + border - buffer)
        limit_y = max(0.0, 0.5 * cols * patch_y + border - buffer)
        return (np.abs(base_pos[:, 0]) > limit_x) | (np.abs(base_pos[:, 1]) > limit_y)

    def _sample_spawn_index(self, env_id: int) -> int:
        rows, cols = self._spawn_grid_shape
        if rows * cols == self._spawn_origins.shape[0] and rows > 0 and cols > 0:
            row = int(np.clip(self._spawn_env_levels[env_id], 0, rows - 1))
            col = int(np.clip(self._spawn_type_cols[env_id], 0, cols - 1))
            return row * cols + col
        return int(env_id % self._spawn_origins.shape[0])

    def _apply_command_curriculum(self) -> None:
        ranges = self.cfg_obj.command.ranges
        progress = int(self.common_step_counter)
        current_stage = 0
        for index, stage in enumerate(self.cfg_obj.command_curriculum):
            if progress < stage.step:
                continue
            current_stage = index
            if stage.lin_vel_x is not None:
                ranges.lin_vel_x = stage.lin_vel_x
            if stage.lin_vel_y is not None:
                ranges.lin_vel_y = stage.lin_vel_y
            if stage.ang_vel_z is not None:
                ranges.ang_vel_z = stage.ang_vel_z
        run_stage = 0
        for index, stage in enumerate(self.cfg_obj.run_command_curriculum):
            if self.profile_step_counter < int(stage.step):
                continue
            run_stage = index
            if stage.run_velocity_x is not None:
                self.cfg_obj.command.run_velocity_x = stage.run_velocity_x
        self._current_command_stage = current_stage
        self._current_run_command_stage = run_stage
        if hasattr(self, "backend"):
            self.backend.set_command_ranges(
                lin_vel_x=ranges.lin_vel_x,
                lin_vel_y=ranges.lin_vel_y,
                ang_vel_z=ranges.ang_vel_z,
            )
            self.backend.set_command_run_sampling(
                rel_run_envs=self.cfg_obj.command.rel_run_envs,
                run_velocity_x_min=self.cfg_obj.command.run_velocity_x[0],
                run_velocity_x_max=self.cfg_obj.command.run_velocity_x[1],
            )

    def _reset_all(self):
        env_ids = np.arange(self.num_envs, dtype=np.int64)
        self._reset_envs(env_ids, np.zeros(self.num_envs, dtype=np.int64))
        self.backend.update_command_frames()
        batch_state = self.backend.state
        self._run_task_runtime(advance_time=False)
        obs = np.asarray(batch_state.actor_obs, dtype=np.float32).copy()
        critic = np.asarray(batch_state.critic_obs, dtype=np.float32).copy()
        if self.noise_cfg.level > 0.0 and self.cfg_obj.observations.actor_noise:
            obs = self._apply_actor_obs_noise(obs)
        return obs, critic

    def _run_task_runtime(self, *, advance_time: bool = False) -> None:
        self._advance_task_time = bool(advance_time)
        try:
            self._run_batch_task_numpy()
        finally:
            self._advance_task_time = False

    def _run_batch_task_numpy(self) -> None:
        self._run_go1_rough_task_numpy()

    def _build_generic_velocity_observations(self, state: Any) -> tuple[np.ndarray, np.ndarray]:
        params = np.asarray(state.task_params, dtype=np.float32)
        height_scan_scale = 1.0 / max(float(params[_TASK_PARAM["height_scan_max_distance"]]), 1.0e-6)
        lin_vel = np.asarray(state.base_linear_velocity_body, dtype=np.float32)
        ang_vel = np.asarray(state.base_angular_velocity_body, dtype=np.float32)
        gravity = np.asarray(state.projected_gravity, dtype=np.float32)
        joint_pos = np.asarray(state.joint_position, dtype=np.float32)
        joint_vel = np.asarray(state.joint_velocity, dtype=np.float32)
        last_action = np.asarray(state.last_action, dtype=np.float32)
        command = np.asarray(state.command, dtype=np.float32)
        default_joint_position = np.asarray(state.default_joint_position, dtype=np.float32).reshape(1, -1)
        actor_parts = [
            lin_vel,
            ang_vel,
            gravity,
            joint_pos - default_joint_position,
            joint_vel,
            last_action,
            command,
        ]
        height_scan = np.asarray(state.height_scan, dtype=np.float32)
        if height_scan.shape[1] > 0:
            actor_parts.append(height_scan * height_scan_scale)
        actor_obs = np.concatenate(actor_parts, axis=1).astype(np.float32, copy=False)
        contact_force_log = np.sign(state.foot_contact_force) * np.log1p(np.abs(state.foot_contact_force))
        critic_obs = np.concatenate(
            [
                actor_obs,
                np.asarray(state.foot_height, dtype=np.float32),
                np.asarray(state.foot_air_time, dtype=np.float32),
                np.asarray(state.foot_contact, dtype=np.float32),
                np.asarray(contact_force_log, dtype=np.float32).reshape(self.num_envs, -1),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        return actor_obs, critic_obs

    def _run_go1_rough_task_numpy(self) -> None:
        state = self.backend.state
        params = np.asarray(state.task_params, dtype=np.float32)
        weights = np.asarray(state.reward_weights, dtype=np.float32)
        reward_cfg = self.cfg_obj.rewards

        step_dt = float(params[_TASK_PARAM["step_dt"]]) if params.size > _TASK_PARAM["step_dt"] else self.step_dt
        lin_vel_std2 = max(float(params[_TASK_PARAM["lin_vel_std2"]]), 1.0e-6)
        ang_vel_std2 = max(float(params[_TASK_PARAM["ang_vel_std2"]]), 1.0e-6)
        upright_std2 = max(float(params[_TASK_PARAM["upright_std2"]]), 1.0e-6)
        command_threshold = float(params[_TASK_PARAM["command_threshold"]])

        command = np.asarray(state.command, dtype=np.float32)
        base_quaternion = np.asarray(state.base_quaternion, dtype=np.float32)
        lin_vel = _quat_rotate_inv_batch(
            np.asarray(state.base_linear_velocity, dtype=np.float32),
            base_quaternion,
        )
        ang_vel = _quat_rotate_inv_batch(
            np.asarray(state.base_angular_velocity, dtype=np.float32),
            base_quaternion,
        )
        joint_pos = np.asarray(state.joint_position, dtype=np.float32)
        previous_action = np.asarray(state.previous_action, dtype=np.float32)
        submitted_action = np.asarray(state.submitted_action, dtype=np.float32)
        current_action = np.asarray(state.action, dtype=np.float32)
        default_joint_position = np.asarray(state.default_joint_position, dtype=np.float32).reshape(1, -1)
        diff = joint_pos - default_joint_position
        command_xy_norm = np.linalg.norm(command[:, :2], axis=1)
        command_speed = command_xy_norm + np.abs(command[:, 2])
        active = (command_speed > command_threshold).astype(np.float32)

        lin_error = np.sum(np.square(command[:, :2] - lin_vel[:, :2]), axis=1) + np.square(lin_vel[:, 2])
        ang_error = np.square(command[:, 2] - ang_vel[:, 2]) + np.sum(np.square(ang_vel[:, :2]), axis=1)
        terrain_up_b = self._terrain_up_body_from_scan(state)
        upright_error = np.sum(np.square(terrain_up_b[:, :2]), axis=1)
        pose_std = self._go1_pose_std(command_speed)
        pose_error = np.mean(np.square(diff / np.maximum(pose_std, 1.0e-6)), axis=1)
        dof_pos_limits = self._soft_joint_limit_penalty(state, float(reward_cfg.soft_joint_pos_limit_factor))
        action_rate_l2 = np.sum(np.square(current_action - previous_action), axis=1)

        foot_contact = np.asarray(state.foot_contact, dtype=np.float32)
        foot_contact_mask = foot_contact > 0.0
        foot_height = np.asarray(state.foot_height, dtype=np.float32)
        foot_vel = np.asarray(state.foot_velocity, dtype=np.float32)
        foot_vel_xy = np.linalg.norm(foot_vel[:, :, :2], axis=2) if foot_vel.size else np.zeros_like(foot_height)
        foot_clearance = (
            np.sum(np.abs(foot_height - float(reward_cfg.foot_clearance_target_height)) * foot_vel_xy, axis=1)
            if foot_height.size
            else np.zeros((self.num_envs,), dtype=np.float32)
        )
        foot_peak_height = np.asarray(state.foot_peak_height, dtype=np.float32)
        first_contact = np.asarray(state.first_contact, dtype=np.float32)
        foot_swing_height = (
            np.sum(np.square(foot_peak_height / max(float(reward_cfg.foot_clearance_target_height), 1.0e-6) - 1.0) * first_contact, axis=1)
            if foot_peak_height.size
            else np.zeros((self.num_envs,), dtype=np.float32)
        )
        foot_slip = np.sum(np.square(foot_vel_xy) * foot_contact_mask.astype(np.float32), axis=1)
        soft_landing = np.sum(np.asarray(state.landing_force, dtype=np.float32), axis=1)

        terms = self._zero_reward_terms(state)
        self._set_reward_term(terms, "track_linear_velocity", weights[0] * np.exp(-lin_error / lin_vel_std2))
        self._set_reward_term(terms, "track_angular_velocity", weights[1] * np.exp(-ang_error / ang_vel_std2))
        self._set_reward_term(terms, "upright", weights[2] * np.exp(-upright_error / upright_std2))
        self._set_reward_term(terms, "pose", weights[3] * np.exp(-pose_error))
        self._set_reward_term(terms, "body_ang_vel", weights[4] * np.sum(np.square(ang_vel[:, :2]), axis=1))
        self._set_reward_term(terms, "angular_momentum", weights[5] * np.zeros((self.num_envs,), dtype=np.float32))
        self._set_reward_term(terms, "dof_pos_limits", weights[6] * dof_pos_limits)
        self._set_reward_term(terms, "action_rate_l2", weights[7] * action_rate_l2)
        self._set_reward_term(terms, "air_time", weights[8] * np.zeros((self.num_envs,), dtype=np.float32))
        self._set_reward_term(terms, "foot_clearance", weights[9] * foot_clearance * active)
        self._set_reward_term(terms, "foot_swing_height", weights[10] * foot_swing_height * active)
        self._set_reward_term(terms, "foot_slip", weights[11] * foot_slip * active)
        self._set_reward_term(terms, "soft_landing", weights[12] * soft_landing * active)
        self._set_reward_term(terms, "self_collisions", weights[13] * np.asarray(state.self_collision_count, dtype=np.float32))
        self._set_reward_term(terms, "shank_collision", weights[14] * np.asarray(state.shank_collision_count, dtype=np.float32))
        self._set_reward_term(
            terms,
            "trunk_head_collision",
            weights[15] * np.asarray(state.trunk_head_collision_count, dtype=np.float32),
        )
        run_mask = np.asarray(
            getattr(state, "command_is_run_env", np.zeros(self.num_envs, dtype=bool)),
            dtype=bool,
        )
        if "run_progress" in self._reward_term_names:
            progress_index = self._reward_term_names.index("run_progress")
            run_progress = np.clip(
                lin_vel[:, 0] / np.maximum(command[:, 0], 0.1),
                0.0,
                1.0,
            ) * run_mask
            self._set_reward_term(
                terms,
                "run_progress",
                weights[progress_index] * run_progress,
            )
        if "bound_gait" in self._reward_term_names:
            gait_index = self._reward_term_names.index("bound_gait")
            gait_score = bound_gait_score_numpy(
                foot_contact,
                foot_vel,
                foot_height,
                current_action,
                lin_vel[:, 0],
                command[:, 0],
                run_mask,
                foot_indices=self._gait_foot_indices,
                joint_indices=self._gait_joint_indices,
                motion_std=reward_cfg.bound_gait_motion_std,
                action_std=reward_cfg.bound_gait_action_std,
                height_sync_std=reward_cfg.bound_gait_height_sync_std,
                height_separation_std=reward_cfg.bound_gait_height_separation_std,
                trot_penalty=reward_cfg.bound_gait_trot_penalty,
            )
            self._set_reward_term(terms, "bound_gait", weights[gait_index] * gait_score)
        np.nan_to_num(terms, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.copyto(state.reward, (np.sum(terms, axis=1) * step_dt).astype(np.float32))

        terminated = np.asarray(state.illegal_contact_count, dtype=np.float32) > 0.0
        np.copyto(state.terminated, terminated.astype(np.uint8))
        np.copyto(state.velocity_error, np.linalg.norm(command[:, :2] - lin_vel[:, :2], axis=1).astype(np.float32))
        np.copyto(state.foot_slip, (foot_slip * active).astype(np.float32))
        np.copyto(state.base_clearance, np.asarray(state.base_height, dtype=np.float32))
        np.copyto(state.terrain_normal_error, upright_error.astype(np.float32))

        actor_obs, critic_obs = self._build_generic_velocity_observations(state)
        self._copy_obs(state, actor_obs, critic_obs)

    def _terrain_up_body_from_scan(self, state: Any) -> np.ndarray:
        points = np.asarray(getattr(state, "height_scan_point", np.empty((self.num_envs, 0, 3))), dtype=np.float32)
        hits = np.asarray(getattr(state, "height_scan_hit", np.empty((self.num_envs, 0))), dtype=bool)
        terrain_up_w = np.broadcast_to(
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            (self.num_envs, 3),
        ).copy()
        if points.ndim == 3 and points.shape[0] == self.num_envs and points.shape[2] == 3 and points.shape[1] > 0:
            if points.shape[1] > 32:
                sample_ids = np.linspace(0, points.shape[1] - 1, 32).astype(np.int64)
                points = points[:, sample_ids]
                hits = hits[:, sample_ids] if hits.shape[0:2] == (self.num_envs, self._height_scan_dim) else hits
            valid = hits if hits.shape == points.shape[:2] else np.ones(points.shape[:2], dtype=bool)
            valid &= np.all(np.isfinite(points), axis=2)
            counts = np.sum(valid, axis=1)
            weights = valid.astype(np.float64)[:, :, None]
            points64 = points.astype(np.float64, copy=False)
            centroids = np.sum(points64 * weights, axis=1) / np.maximum(counts[:, None], 1)
            centered = (points64 - centroids[:, None, :]) * weights
            covariance = np.einsum("bni,bnj->bij", centered, centered)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            normals = eigenvectors[:, :, 0]
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            normals /= np.maximum(lengths, 1.0e-8)
            normals = np.where(normals[:, 2:3] < 0.0, -normals, normals)
            eps = np.finfo(eigenvalues.dtype).eps
            plane_like = eigenvalues[:, 0] / np.maximum(eigenvalues[:, 1], eps) < 0.1
            has_spread = eigenvalues[:, 1] > np.maximum(eigenvalues[:, 2], eps) * 1.0e-6
            reliable = (counts >= 3) & plane_like & has_spread & np.all(np.isfinite(normals), axis=1)
            terrain_up_w[reliable] = normals[reliable].astype(np.float32)
        return _quat_rotate_inv_batch(terrain_up_w.astype(np.float32, copy=False), np.asarray(state.base_quaternion, dtype=np.float32))

    def _go1_pose_std(self, command_speed: np.ndarray) -> np.ndarray:
        state = self.backend.state
        standing = np.asarray(state.pose_std_standing, dtype=np.float32).reshape(1, -1)
        walking = np.asarray(state.pose_std_walking, dtype=np.float32).reshape(1, -1)
        running = np.asarray(state.pose_std_running, dtype=np.float32).reshape(1, -1)
        reward_cfg = self.cfg_obj.rewards
        walk = float(reward_cfg.pose_walking_threshold)
        run = float(reward_cfg.pose_running_threshold)
        speed = np.asarray(command_speed, dtype=np.float32).reshape(-1, 1)
        return np.where(speed < walk, standing, np.where(speed < run, walking, running)).astype(np.float32, copy=False)

    def _soft_joint_limit_penalty(self, state: Any, factor: float) -> np.ndarray:
        joint_pos = np.asarray(state.joint_position, dtype=np.float32)
        lower = np.asarray(state.joint_lower_limit, dtype=np.float32).reshape(1, -1)
        upper = np.asarray(state.joint_upper_limit, dtype=np.float32).reshape(1, -1)
        finite = np.isfinite(lower) & np.isfinite(upper) & (upper > lower)
        mid = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower) * float(np.clip(factor, 0.0, 1.0))
        soft_lower = mid - half
        soft_upper = mid + half
        below = np.clip(soft_lower - joint_pos, 0.0, None)
        above = np.clip(joint_pos - soft_upper, 0.0, None)
        return np.sum(np.where(finite, below + above, 0.0), axis=1).astype(np.float32, copy=False)

    def _copy_obs(self, state: Any, actor_obs: np.ndarray, critic_obs: np.ndarray) -> None:
        if actor_obs.shape != state.actor_obs.shape:
            raise RuntimeError(f"Go1 numpy task built actor obs shape {actor_obs.shape}, expected {state.actor_obs.shape}")
        if critic_obs.shape != state.critic_obs.shape:
            raise RuntimeError(f"Go1 numpy task built critic obs shape {critic_obs.shape}, expected {state.critic_obs.shape}")
        np.copyto(state.actor_obs, actor_obs)
        np.copyto(state.critic_obs, critic_obs)

    def _zero_reward_terms(self, state: Any) -> np.ndarray:
        terms = np.asarray(state.reward_terms, dtype=np.float32)
        terms.fill(0.0)
        return terms

    def _set_reward_term(self, terms: np.ndarray, name: str, values: np.ndarray) -> None:
        try:
            index = self._reward_term_names.index(name)
        except ValueError:
            return
        terms[:, index] = np.asarray(values, dtype=np.float32).reshape(self.num_envs)

    def _apply_pushes(self) -> None:
        if not self.cfg_obj.push_enabled:
            return
        push_mode = str(getattr(self.cfg_obj, "push_mode", "force"))
        interval = max(1, int(self.cfg_obj.push_interval_steps))
        if getattr(self.cfg_obj, "push_interval_mode", "per_env_random") == "global":
            if self.step_counter % interval != 0:
                return
            env_ids = np.arange(self.num_envs, dtype=np.int64)
            if push_mode == "velocity":
                lin_vel, ang_vel = self._sample_pushed_base_velocities(env_ids)
                self.backend.set_base_velocities(env_ids.tolist(), lin_vel, ang_vel, refresh=False)
            else:
                push_force = self._sample_range_matrix(self.cfg_obj.push_force_ranges, ("x", "y", "z"), env_ids.size)
                self.backend.set_push_forces(env_ids.tolist(), push_force)
            self._push_count[env_ids] += 1
            return
        self._push_step_left -= 1
        env_ids = np.flatnonzero(self._push_step_left <= 0).astype(np.int64, copy=False)
        if env_ids.size == 0:
            return
        if push_mode == "velocity":
            lin_vel, ang_vel = self._sample_pushed_base_velocities(env_ids)
            self.backend.set_base_velocities(env_ids.tolist(), lin_vel, ang_vel, refresh=False)
        else:
            push_force = self._sample_range_matrix(self.cfg_obj.push_force_ranges, ("x", "y", "z"), env_ids.size)
            self.backend.set_push_forces(env_ids.tolist(), push_force)
        self._push_count[env_ids] += 1
        self._reset_push_timers(env_ids)

    def _sample_pushed_base_velocities(self, env_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        state = self.backend.state
        linear_velocity = np.asarray(state.base_linear_velocity[rows], dtype=np.float32).copy()
        angular_velocity = np.asarray(state.base_angular_velocity[rows], dtype=np.float32).copy()
        linear_velocity += self._sample_range_matrix(
            self.cfg_obj.push_velocity_ranges,
            ("x", "y", "z"),
            rows.size,
        )
        angular_velocity += self._sample_range_matrix(
            self.cfg_obj.push_velocity_ranges,
            ("roll", "pitch", "yaw"),
            rows.size,
        )
        return linear_velocity, angular_velocity

    def _reset_env(self, env_id: int, *, reason: int) -> None:
        self._reset_envs(np.asarray([env_id], dtype=np.int64), np.asarray([reason], dtype=np.int64))

    def _reset_envs(self, env_ids: np.ndarray, reasons: np.ndarray) -> None:
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        reasons = np.asarray(reasons, dtype=np.int64).reshape(-1)
        if env_ids.size == 0:
            return
        if reasons.size != env_ids.size:
            raise ValueError("reset reasons must have the same length as env_ids")

        self._apply_command_curriculum()

        if not self.cfg_obj.terrain_curriculum:
            rows, cols = self._spawn_grid_shape
            if rows > 0 and cols > 0:
                self._spawn_env_levels[env_ids] = self._rng.integers(
                    0, rows, size=env_ids.size
                ).astype(np.int64)
                self._spawn_type_cols[env_ids] = self._rng.integers(
                    0, cols, size=env_ids.size
                ).astype(np.int64)

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
            terrain_origin = spawn.copy()
            spawn[:2] += self._rng.uniform(-self.cfg_obj.spawn_jitter, self.cfg_obj.spawn_jitter, 2)
            z_lo, z_hi = self.cfg_obj.reset_z_range
            base_z = float(spawn[2]) + self.cfg_obj.base_clearance + float(self._rng.uniform(z_lo, z_hi))
            reset_lin_vel = np.zeros(3, dtype=np.float32)
            reset_ang_vel = np.zeros(3, dtype=np.float32)
            orientation = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            if self.cfg_obj.randomize_reset_yaw:
                yaw = float(self._rng.uniform(-3.14, 3.14))
                orientation = _quat_from_yaw(yaw)
            base_positions[row] = np.asarray([float(spawn[0]), float(spawn[1]), float(base_z)], dtype=np.float32)
            base_orientations[row] = orientation
            base_linear_velocities[row] = reset_lin_vel
            base_angular_velocities[row] = reset_ang_vel
            joint_positions[row] = self.default_joint_pos
            joint_velocities[row] = 0.0

            self._spawn_indices[env_id] = spawn_index
            self._terrain_levels[env_id] = float(self._spawn_levels[spawn_index])
            self._episode_start_xy[env_id] = terrain_origin[:2]
            self._episode_length_np[env_id] = 0
            self._episode_returns[env_id] = 0.0
            self._previous_actions[env_id] = 0.0
            self._last_actions[env_id] = 0.0
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
        if hasattr(self.backend, "clear_reset_contacts"):
            self.backend.clear_reset_contacts(env_ids.tolist())
        self.backend.reset_commands(env_ids.tolist())

    def _apply_startup_domain_randomization(self) -> None:
        env_ids = np.arange(self.num_envs, dtype=np.int64)
        dr = self.cfg_obj.domain_randomization
        count = self.num_envs
        base_mass_delta = np.zeros((count,), dtype=np.float32)
        base_com_offset = np.zeros((count, 3), dtype=np.float32)
        base_joint_kp = self._joint_gain_array(self.cfg_obj.kp)
        base_joint_kd = self._joint_gain_array(self.cfg_obj.kd)
        joint_kp = np.broadcast_to(base_joint_kp.reshape(1, -1), (count, self.num_actions)).astype(np.float32, copy=True)
        joint_kd = np.broadcast_to(base_joint_kd.reshape(1, -1), (count, self.num_actions)).astype(np.float32, copy=True)
        foot_friction = None
        if dr.enabled:
            lo, hi = dr.encoder_bias_range
            self._encoder_bias[:] = self._rng.uniform(lo, hi, (count, self.num_actions)).astype(np.float32)
            if dr.randomize_base_mass:
                lo, hi = dr.added_mass_range
                base_mass_delta[:] = self._rng.uniform(lo, hi, count).astype(np.float32)
            if dr.random_com:
                lo, hi = dr.com_offset_x
                base_com_offset[:, 0] = self._rng.uniform(lo, hi, count).astype(np.float32)
                if dr.com_offset_y is not None:
                    lo, hi = dr.com_offset_y
                    base_com_offset[:, 1] = self._rng.uniform(lo, hi, count).astype(np.float32)
                if dr.com_offset_z is not None:
                    lo, hi = dr.com_offset_z
                    base_com_offset[:, 2] = self._rng.uniform(lo, hi, count).astype(np.float32)
            if dr.randomize_kp:
                lo, hi = dr.kp_multiplier_range
                joint_kp[:] = base_joint_kp.reshape(1, -1) * self._rng.uniform(lo, hi, (count, 1)).astype(np.float32)
            if dr.randomize_kd:
                lo, hi = dr.kd_multiplier_range
                joint_kd[:] = base_joint_kd.reshape(1, -1) * self._rng.uniform(lo, hi, (count, 1)).astype(np.float32)
            if dr.randomize_foot_friction:
                foot_friction = np.zeros((count, 3), dtype=np.float32)
                lo, hi = dr.foot_friction_slide_range
                foot_friction[:, 0] = self._rng.uniform(lo, hi, count).astype(np.float32)
                foot_friction[:, 1] = self._sample_log_uniform(dr.foot_friction_spin_range, count)
                foot_friction[:, 2] = self._sample_log_uniform(dr.foot_friction_roll_range, count)
        else:
            self._encoder_bias.fill(0.0)
        if not hasattr(self.backend, "reset_domain_randomization"):
            return
        self.backend.reset_domain_randomization(
            env_ids.tolist(),
            base_mass_delta=base_mass_delta,
            base_com_offset=base_com_offset,
            joint_kp=joint_kp,
            joint_kd=joint_kd,
            foot_friction=foot_friction,
        )

    def _sample_log_uniform(self, value_range: tuple[float, float], count: int) -> np.ndarray:
        lo, hi = float(value_range[0]), float(value_range[1])
        if lo <= 0.0 or hi <= 0.0:
            return self._rng.uniform(lo, hi, int(count)).astype(np.float32)
        log_values = self._rng.uniform(math.log(lo), math.log(hi), int(count)).astype(np.float32)
        return np.exp(log_values).astype(np.float32)

    def _sample_range_vec(self, ranges: Mapping[str, tuple[float, float]], names: Sequence[str]) -> np.ndarray:
        values = []
        for name in names:
            lo, hi = ranges.get(name, (0.0, 0.0))
            values.append(float(self._rng.uniform(lo, hi)))
        return np.asarray(values, dtype=np.float32)

    def _sample_range_matrix(
        self,
        ranges: Mapping[str, tuple[float, float]],
        names: Sequence[str],
        count: int,
    ) -> np.ndarray:
        values = np.zeros((int(count), len(names)), dtype=np.float32)
        for column, name in enumerate(names):
            lo, hi = ranges.get(name, (0.0, 0.0))
            values[:, column] = self._rng.uniform(lo, hi, int(count)).astype(np.float32)
        return values

    def _reset_push_timer(self, env_id: int) -> None:
        if not self.cfg_obj.push_enabled:
            self._push_time_left[env_id] = math.inf
            self._push_step_left[env_id] = np.iinfo(np.int64).max
            return
        if getattr(self.cfg_obj, "push_interval_mode", "per_env_random") == "global":
            self._push_time_left[env_id] = math.inf
            self._push_step_left[env_id] = np.iinfo(np.int64).max
            return
        if str(getattr(self.cfg_obj, "push_mode", "force")) == "velocity":
            lo, hi = self.cfg_obj.push_interval_range_s
            interval_s = float(self._rng.uniform(float(lo), float(hi)))
            self._push_step_left[env_id] = max(1, int(math.ceil(interval_s / max(self.step_dt, 1.0e-9))))
            self._push_time_left[env_id] = float(self._push_step_left[env_id]) * self.step_dt
            return
        interval = max(1, int(self.cfg_obj.push_interval_steps))
        self._push_step_left[env_id] = int(self._rng.integers(1, interval + 1))
        self._push_time_left[env_id] = float(self._push_step_left[env_id]) * self.step_dt

    def _reset_push_timers(self, env_ids: np.ndarray) -> None:
        for env_id in np.asarray(env_ids, dtype=np.int64).reshape(-1):
            self._reset_push_timer(int(env_id))

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
        rows, _ = self._spawn_grid_shape
        commanded_distance = (
            float(np.linalg.norm(self.command_b[env_id, :2]))
            * float(self.cfg_obj.episode_length_s)
            * 0.5
        )
        move_up = distance > (float(self._terrain_patch_size()[0]) * 0.5)
        move_down = (distance < commanded_distance) and not move_up
        level = int(self._spawn_env_levels[env_id]) + (1 if move_up else 0) - (1 if move_down else 0)
        if level >= rows:
            level = int(self._rng.integers(0, rows))
        self._spawn_env_levels[env_id] = int(np.clip(level, 0, max(rows - 1, 0)))
        self._terrain_curriculum_limits[env_id] = float(self._spawn_env_levels[env_id]) / max(float(rows - 1), 1.0)

__all__ = ["Go1VelocityEnv"]
