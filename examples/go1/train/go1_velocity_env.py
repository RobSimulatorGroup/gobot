"""Go1 NumPy batch environment for velocity locomotion training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
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

_GOBOT_REWARD_TERM_NAMES: tuple[str, ...] = (
    "track_linear_velocity",
    "track_angular_velocity",
    "upright",
    "action_rate_l2",
    "air_time",
    "foot_clearance",
    "foot_slip",
)
_UNILAB_FLAT_REWARD_TERM_NAMES: tuple[str, ...] = (
    "tracking_lin_vel",
    "tracking_ang_vel",
    "lin_vel_z",
    "ang_vel_xy",
    "base_height",
    "action_rate",
    "similar_to_default",
    "contact",
    "swing_feet_z",
)
_UNILAB_ROUGH_REWARD_TERM_NAMES: tuple[str, ...] = (
    "lin_vel_z",
    "ang_vel_xy",
    "joint_torques_l2",
    "joint_acc_l2",
    "joint_power",
    "stand_still",
    "hip_pos",
    "joint_pos_penalty",
    "joint_mirror",
    "action_rate",
    "undesired_contacts",
    "contact_forces",
    "tracking_lin_vel",
    "tracking_ang_vel",
    "feet_air_time",
    "feet_air_time_variance",
    "feet_contact_without_cmd",
    "feet_slide",
    "feet_height_body",
    "feet_gait",
    "upward",
)
_REWARD_TERM_NAMES: tuple[str, ...] = _GOBOT_REWARD_TERM_NAMES

_GO1_HIP_INDICES = np.asarray([0, 3, 6, 9], dtype=np.int64)
_GO1_FRONT_LEFT = 0
_GO1_FRONT_RIGHT = 1
_GO1_REAR_LEFT = 2
_GO1_REAR_RIGHT = 3

R_TRACK_LINEAR_VELOCITY = 0
R_TRACK_ANGULAR_VELOCITY = 1
R_UPRIGHT = 2
R_ACTION_RATE_L2 = 3
R_AIR_TIME = 4
R_FOOT_CLEARANCE = 5
R_FOOT_SLIP = 6

_TASK_PARAM = {
    "step_dt": 0,
    "lin_vel_std2": 1,
    "ang_vel_std2": 2,
    "upright_std2": 3,
    "command_threshold": 4,
    "min_base_clearance": 5,
    "flat_roll_pitch_limit": 6,
    "height_scan_max_distance": 7,
}

_TASK_FLAG = {
    "rough_terrain": 0,
}


def _reward_names_for_profile(profile: str) -> tuple[str, ...]:
    if profile == "unilab_flat":
        return _UNILAB_FLAT_REWARD_TERM_NAMES
    if profile == "unilab_rough":
        return _UNILAB_ROUGH_REWARD_TERM_NAMES
    return _GOBOT_REWARD_TERM_NAMES


def _go1_unilab_flat_actor_schema(action_dim: int, foot_count: int) -> ObservationSpec:
    return ObservationSpec(
        version="gobot_go1_unilab_flat_actor_v1",
        fields=(
            SpecField("gyro", 3, "rad/s"),
            SpecField("projected_gravity", 3),
            SpecField("joint_pos_rel", int(action_dim), "rad"),
            SpecField("joint_vel", int(action_dim), "rad/s"),
            SpecField("current_action", int(action_dim)),
            SpecField("command", 3),
            SpecField("feet_phase", int(foot_count)),
        ),
    )


def _go1_unilab_flat_critic_schema(action_dim: int, foot_count: int) -> ObservationSpec:
    return ObservationSpec(
        version="gobot_go1_unilab_flat_critic_v1",
        fields=(
            SpecField("gyro", 3, "rad/s"),
            SpecField("projected_gravity", 3),
            SpecField("joint_pos_rel", int(action_dim), "rad"),
            SpecField("joint_vel", int(action_dim), "rad/s"),
            SpecField("current_action", int(action_dim)),
            SpecField("command", 3),
            SpecField("feet_phase", int(foot_count)),
            SpecField("base_lin_vel_b", 3, "m/s"),
        ),
    )


def _go1_unilab_rough_actor_schema(action_dim: int) -> ObservationSpec:
    return ObservationSpec(
        version="gobot_go1_unilab_rough_actor_v1",
        fields=(
            SpecField("gyro_scaled", 3, "rad/s"),
            SpecField("projected_gravity", 3),
            SpecField("command", 3),
            SpecField("joint_pos_rel", int(action_dim), "rad"),
            SpecField("joint_vel_scaled", int(action_dim), "rad/s"),
            SpecField("current_action", int(action_dim)),
        ),
    )


def _go1_unilab_rough_critic_schema(action_dim: int, height_scan_dim: int) -> ObservationSpec:
    return ObservationSpec(
        version="gobot_go1_unilab_rough_critic_v1",
        fields=(
            SpecField("base_lin_vel_b", 3, "m/s"),
            SpecField("gyro", 3, "rad/s"),
            SpecField("projected_gravity", 3),
            SpecField("command", 3),
            SpecField("joint_pos_rel", int(action_dim), "rad"),
            SpecField("joint_vel", int(action_dim), "rad/s"),
            SpecField("current_action", int(action_dim)),
            SpecField("height_scan", int(height_scan_dim)),
        ),
    )


def _quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )


def _quat_rotate_inv_batch(vectors: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    quaternions = np.asarray(quaternions, dtype=np.float32)
    q_xyz = -quaternions[..., 1:4]
    q_w = quaternions[..., 0:1]
    t = 2.0 * np.cross(q_xyz, vectors)
    return (vectors + q_w * t + np.cross(q_xyz, t)).astype(np.float32, copy=False)


def _gait_sync_reward(air: np.ndarray, contact: np.ndarray, foot_0: int, foot_1: int, std: float, max_err: float) -> np.ndarray:
    se_air = np.clip(np.square(air[:, foot_0] - air[:, foot_1]), 0.0, max_err**2)
    se_contact = np.clip(np.square(contact[:, foot_0] - contact[:, foot_1]), 0.0, max_err**2)
    return np.exp(-(se_air + se_contact) / std)


def _gait_async_reward(air: np.ndarray, contact: np.ndarray, foot_0: int, foot_1: int, std: float, max_err: float) -> np.ndarray:
    se_act_0 = np.clip(np.square(air[:, foot_0] - contact[:, foot_1]), 0.0, max_err**2)
    se_act_1 = np.clip(np.square(contact[:, foot_0] - air[:, foot_1]), 0.0, max_err**2)
    return np.exp(-(se_act_0 + se_act_1) / std)


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
        task_runtime: str | None = None,
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
        self.task_runtime_mode = self._resolve_task_runtime(task_runtime)
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
        self._task_profile = str(getattr(self.cfg_obj, "task_profile", "gobot_velocity"))
        self._reward_term_names = _reward_names_for_profile(self._task_profile)
        self._advance_task_time = False

        self._foot_count = len(self.cfg_obj.foot_names)
        self._encoder_bias = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)
        self._feet_phase = np.zeros((self.num_envs, self._foot_count), dtype=np.float32)
        self._gait_phase = np.zeros((self.num_envs,), dtype=np.float32)
        self._last_dof_vel_for_acc = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)
        self._current_air_time = np.zeros((self.num_envs, self._foot_count), dtype=np.float32)
        self._current_contact_time = np.zeros((self.num_envs, self._foot_count), dtype=np.float32)
        self._last_air_time = np.zeros((self.num_envs, self._foot_count), dtype=np.float32)
        self._last_contact_time = np.zeros((self.num_envs, self._foot_count), dtype=np.float32)
        self._last_foot_contact_mask = np.zeros((self.num_envs, self._foot_count), dtype=bool)
        self._first_foot_contact = np.zeros((self.num_envs, self._foot_count), dtype=bool)
        self._push_time_left = np.zeros(self.num_envs, dtype=np.float32)
        self._push_step_left = np.zeros(self.num_envs, dtype=np.int64)
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
        self._terrain_bounds = self._terrain_sampler.bounds()
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
        self.task_runtime_metadata = self._make_task_runtime_metadata()
        self.task_runtime_info = self._configure_task_runtime()

        self.cfg = {
            "name": self.cfg_obj.name,
            "source": "examples.go1.train.go1_velocity_env",
            "task": self.cfg_obj.name,
            "task_profile": self._task_profile,
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
            "step_dt": self.step_dt,
            "action_clip": self.cfg_obj.action_clip,
            "illegal_contact": self.cfg_obj.illegal_contact.enabled,
            "domain_randomization": self.cfg_obj.domain_randomization.enabled,
            "push_enabled": self.cfg_obj.push_enabled,
            "push_interval_steps": self.cfg_obj.push_interval_steps,
            "push_interval_range_s": self.cfg_obj.push_interval_range_s,
            "terrain_out_of_bounds": self.cfg_obj.terrain_out_of_bounds,
            "terrain_distance_buffer": self.cfg_obj.terrain_distance_buffer,
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

    def _prepare_actions(self, actions: Any) -> np.ndarray:
        action_np = actions.detach().cpu().numpy() if hasattr(actions, "detach") else np.asarray(actions)
        action_np = np.asarray(action_np, dtype=np.float32).reshape(self.num_envs, self.num_actions)
        clip = float(getattr(self.cfg_obj, "action_clip", 1.0))
        self._submitted_actions = np.clip(action_np, -clip, clip)
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
        batch_state = self.backend.state
        self._run_task_runtime(advance_time=False)
        obs = np.asarray(batch_state.actor_obs, dtype=np.float32).copy()
        critic = np.asarray(batch_state.critic_obs, dtype=np.float32).copy()
        if self.noise_cfg.level > 0.0 and self.cfg_obj.observations.actor_noise:
            obs = self._apply_actor_obs_noise(obs)
            if self._task_profile == "gobot_velocity":
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
        terminal_actor_obs: np.ndarray | None = None
        terminal_critic_obs: np.ndarray | None = None
        if reset_env_ids.size:
            terminal_actor_obs = np.asarray(batch_state.actor_obs, dtype=np.float32).copy()
            terminal_critic_obs = np.asarray(batch_state.critic_obs, dtype=np.float32).copy()
            if self.noise_cfg.level > 0.0 and self.cfg_obj.observations.actor_noise:
                terminal_actor_obs = self._apply_actor_obs_noise(terminal_actor_obs)
                if self._task_profile == "gobot_velocity":
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
            self._run_task_runtime(advance_time=False)
        obs_np = np.asarray(batch_state.actor_obs, dtype=np.float32).copy()
        critic_np = np.asarray(batch_state.critic_obs, dtype=np.float32).copy()
        if self.noise_cfg.level > 0.0 and self.cfg_obj.observations.actor_noise:
            obs_np = self._apply_actor_obs_noise(obs_np)
            if self._task_profile == "gobot_velocity":
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
        timing["task_runtime_ms"] = task_runtime_ms
        timing["numpy_task_ms"] = numpy_task_ms
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

    def _make_observation_schemas(self) -> tuple[ObservationSpec, ObservationSpec]:
        if self._task_profile == "unilab_flat":
            return (
                _go1_unilab_flat_actor_schema(self.num_actions, self._foot_count),
                _go1_unilab_flat_critic_schema(self.num_actions, self._foot_count),
            )
        if self._task_profile == "unilab_rough":
            return (
                _go1_unilab_rough_actor_schema(self.num_actions),
                _go1_unilab_rough_critic_schema(self.num_actions, self._height_scan_dim),
            )
        return (
            velocity_actor_observation_schema(self.num_actions, self._height_scan_dim),
            velocity_critic_observation_schema(self.num_actions, self._height_scan_dim, self._foot_count),
        )

    def _apply_actor_obs_noise(self, obs: np.ndarray) -> np.ndarray:
        if self._task_profile != "gobot_velocity":
            return np.asarray(obs, dtype=np.float32)
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
            lower=-float(self.cfg_obj.action_clip),
            upper=float(self.cfg_obj.action_clip),
        )

    def _make_task_runtime_metadata(self) -> TaskRuntimeMetadata:
        return TaskRuntimeMetadata(
            name=self.cfg_obj.name,
            version=f"go1_{self._task_profile}_numpy_v1",
            obs_groups_spec={"actor": self.actor_obs_schema.dim, "critic": self.critic_obs_schema.dim},
            reward_names=self._reward_term_names,
            backend="gobot_native_cpu_batch_numpy",
            cache_info={
                "scene_source": "jscn",
                "scene_path": self.cfg_obj.scene_path,
                "unilab_reference": self._task_profile in {"unilab_flat", "unilab_rough"},
                "native_contact_detail": "categorized" if self._task_profile == "unilab_rough" else "foot_sensors",
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
            state.action_clip[:] = float(self.cfg_obj.action_clip)
        state.pose_std_standing[:] = 1.0
        state.pose_std_walking[:] = 1.0
        state.pose_std_running[:] = 1.0

        reward_cfg = self.cfg_obj.rewards
        reward_weights = np.zeros_like(state.reward_weights, dtype=np.float32)
        reward_weights[: len(self._reward_weights())] = np.asarray(self._reward_weights(), dtype=np.float32)
        np.copyto(state.reward_weights, reward_weights)

        params = np.zeros_like(state.task_params, dtype=np.float32)
        params[_TASK_PARAM["step_dt"]] = self.step_dt
        params[_TASK_PARAM["lin_vel_std2"]] = reward_cfg.lin_vel_std**2
        params[_TASK_PARAM["ang_vel_std2"]] = reward_cfg.ang_vel_std**2
        params[_TASK_PARAM["upright_std2"]] = reward_cfg.upright_std**2
        params[_TASK_PARAM["command_threshold"]] = reward_cfg.command_threshold
        params[_TASK_PARAM["min_base_clearance"]] = self.cfg_obj.min_base_clearance
        params[_TASK_PARAM["flat_roll_pitch_limit"]] = math.radians(70.0)
        params[_TASK_PARAM["height_scan_max_distance"]] = self.cfg_obj.observations.terrain_scan_max_distance
        np.copyto(state.task_params, params)

        flags = np.zeros_like(state.task_flags, dtype=np.float32)
        flags[_TASK_FLAG["rough_terrain"]] = 1.0 if self.cfg_obj.terrain_type == "rough" else 0.0
        np.copyto(state.task_flags, flags)

    def _reward_weights(self) -> tuple[float, ...]:
        reward_cfg = self.cfg_obj.rewards
        return (
            reward_cfg.track_linear_velocity,
            reward_cfg.track_angular_velocity,
            reward_cfg.upright,
            reward_cfg.action_rate_l2,
            reward_cfg.air_time,
            reward_cfg.foot_clearance,
            reward_cfg.foot_slip,
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

    def _terrain_out_of_bounds(self, state: Any) -> np.ndarray:
        if self._task_profile != "unilab_rough" or not self.cfg_obj.terrain_out_of_bounds:
            return np.zeros((self.num_envs,), dtype=bool)
        if self._terrain_bounds is None:
            return np.zeros((self.num_envs,), dtype=bool)
        min_x, min_y, max_x, max_y = self._terrain_bounds
        buffer = float(self.cfg_obj.terrain_distance_buffer)
        base_pos = np.asarray(state.base_position, dtype=np.float32)
        if base_pos.shape[0] != self.num_envs or base_pos.shape[1] < 2:
            return np.zeros((self.num_envs,), dtype=bool)
        return (
            (base_pos[:, 0] < float(min_x) + buffer)
            | (base_pos[:, 0] > float(max_x) - buffer)
            | (base_pos[:, 1] < float(min_y) + buffer)
            | (base_pos[:, 1] > float(max_y) - buffer)
        )

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
        self._run_task_runtime(advance_time=False)
        obs = np.asarray(batch_state.actor_obs, dtype=np.float32).copy()
        critic = np.asarray(batch_state.critic_obs, dtype=np.float32).copy()
        if self.noise_cfg.level > 0.0 and self.cfg_obj.observations.actor_noise:
            obs = self._apply_actor_obs_noise(obs)
            if self._task_profile == "gobot_velocity":
                critic[:, : self.num_obs] = obs
        return obs, critic

    def _run_task_runtime(self, *, advance_time: bool = False) -> None:
        self._advance_task_time = bool(advance_time)
        try:
            self._run_batch_task_numpy()
        finally:
            self._advance_task_time = False

    def _run_batch_task_numpy(self) -> None:
        if self._task_profile == "unilab_flat":
            self._run_unilab_flat_task_numpy()
            return
        if self._task_profile == "unilab_rough":
            self._run_unilab_rough_task_numpy()
            return
        self._run_gobot_velocity_task_numpy()

    def _run_gobot_velocity_task_numpy(self) -> None:
        state = self.backend.state
        params = np.asarray(state.task_params, dtype=np.float32)
        flags = np.asarray(state.task_flags, dtype=np.float32)
        weights = np.asarray(state.reward_weights, dtype=np.float32)

        step_dt = float(params[_TASK_PARAM["step_dt"]]) if params.size > _TASK_PARAM["step_dt"] else self.step_dt
        lin_vel_std2 = max(float(params[_TASK_PARAM["lin_vel_std2"]]), 1.0e-6)
        ang_vel_std2 = max(float(params[_TASK_PARAM["ang_vel_std2"]]), 1.0e-6)
        upright_std2 = max(float(params[_TASK_PARAM["upright_std2"]]), 1.0e-6)
        command_threshold = float(params[_TASK_PARAM["command_threshold"]])
        min_base_clearance = float(params[_TASK_PARAM["min_base_clearance"]])
        flat_limit = float(params[_TASK_PARAM["flat_roll_pitch_limit"]])
        height_scan_scale = 1.0 / max(float(params[_TASK_PARAM["height_scan_max_distance"]]), 1.0e-6)

        command = np.asarray(state.command, dtype=np.float32)
        lin_vel = np.asarray(state.base_linear_velocity_body, dtype=np.float32)
        ang_vel = np.asarray(state.base_angular_velocity_body, dtype=np.float32)
        gravity = np.asarray(state.projected_gravity, dtype=np.float32)
        joint_pos = np.asarray(state.joint_position, dtype=np.float32)
        joint_vel = np.asarray(state.joint_velocity, dtype=np.float32)
        previous_action = np.asarray(state.previous_action, dtype=np.float32)
        submitted_action = np.asarray(state.submitted_action, dtype=np.float32)
        last_action = np.asarray(state.last_action, dtype=np.float32)
        foot_height = np.asarray(state.foot_height, dtype=np.float32)
        foot_air_time = np.asarray(state.foot_air_time, dtype=np.float32)
        foot_contact = np.asarray(state.foot_contact, dtype=np.float32)
        foot_contact_force = np.asarray(state.foot_contact_force, dtype=np.float32)
        default_joint_position = np.asarray(state.default_joint_position, dtype=np.float32).reshape(1, -1)

        lin_error = np.sum(np.square(command[:, :2] - lin_vel[:, :2]), axis=1) + np.square(lin_vel[:, 2])
        ang_error = np.square(command[:, 2] - ang_vel[:, 2]) + np.sum(np.square(ang_vel[:, :2]), axis=1)
        command_speed = np.sqrt(np.sum(np.square(command[:, :2]), axis=1)) + np.abs(command[:, 2])
        active = (command_speed > command_threshold).astype(np.float32)
        upright_error = np.sum(np.square(gravity[:, :2]), axis=1)
        action_rate_l2 = np.sum(np.square(submitted_action - previous_action), axis=1)

        foot_clearance_sum = np.sum(foot_height, axis=1) if foot_height.size else np.zeros(self.num_envs, dtype=np.float32)
        foot_slip_sum = np.sum(foot_contact > 0.0, axis=1).astype(np.float32) if foot_contact.size else np.zeros(self.num_envs, dtype=np.float32)
        air_time_count = (
            np.sum((foot_air_time > 0.05) & (foot_air_time < 0.5), axis=1).astype(np.float32)
            if foot_air_time.size
            else np.zeros(self.num_envs, dtype=np.float32)
        )

        np.copyto(state.foot_slip, (foot_slip_sum * active).astype(np.float32))
        np.copyto(state.base_clearance, np.asarray(state.base_height, dtype=np.float32))
        np.copyto(
            state.velocity_error,
            np.sqrt(np.sum(np.square(command[:, :2] - lin_vel[:, :2]), axis=1)).astype(np.float32),
        )

        reward_terms = np.asarray(state.reward_terms, dtype=np.float32)
        reward_terms.fill(0.0)
        reward_terms[:, R_TRACK_LINEAR_VELOCITY] = weights[R_TRACK_LINEAR_VELOCITY] * np.exp(-lin_error / lin_vel_std2)
        reward_terms[:, R_TRACK_ANGULAR_VELOCITY] = weights[R_TRACK_ANGULAR_VELOCITY] * np.exp(-ang_error / ang_vel_std2)
        reward_terms[:, R_UPRIGHT] = weights[R_UPRIGHT] * np.exp(-upright_error / upright_std2)
        reward_terms[:, R_ACTION_RATE_L2] = weights[R_ACTION_RATE_L2] * action_rate_l2
        reward_terms[:, R_AIR_TIME] = weights[R_AIR_TIME] * air_time_count * active
        reward_terms[:, R_FOOT_CLEARANCE] = weights[R_FOOT_CLEARANCE] * foot_clearance_sum * active
        reward_terms[:, R_FOOT_SLIP] = weights[R_FOOT_SLIP] * foot_slip_sum * active
        np.copyto(state.reward, (np.sum(reward_terms, axis=1) * step_dt).astype(np.float32))

        terminated = np.asarray(state.base_clearance, dtype=np.float32) < min_base_clearance
        rough_terrain = flags.size > _TASK_FLAG["rough_terrain"] and flags[_TASK_FLAG["rough_terrain"]] > 0.0
        if not rough_terrain:
            limit = math.sin(flat_limit)
            terminated |= np.abs(gravity[:, 0]) > limit
            terminated |= np.abs(gravity[:, 1]) > limit
        np.copyto(state.terminated, terminated.astype(np.uint8))

        actor_parts = [
            lin_vel,
            ang_vel,
            gravity,
            joint_pos + np.asarray(state.encoder_bias, dtype=np.float32) - default_joint_position,
            joint_vel,
            last_action,
            command,
        ]
        height_scan = np.asarray(state.height_scan, dtype=np.float32)
        if height_scan.shape[1] > 0:
            actor_parts.append(height_scan * height_scan_scale)
        actor_obs = np.concatenate(actor_parts, axis=1).astype(np.float32, copy=False)
        if actor_obs.shape != state.actor_obs.shape:
            raise RuntimeError(
                f"Go1 numpy task built actor obs shape {actor_obs.shape}, expected {state.actor_obs.shape}"
            )
        np.copyto(state.actor_obs, actor_obs)

        contact_force_log = np.sign(foot_contact_force) * np.log1p(np.abs(foot_contact_force))
        critic_parts = [
            actor_obs,
            foot_height,
            foot_air_time,
            foot_contact,
            contact_force_log.reshape(self.num_envs, -1),
        ]
        critic_obs = np.concatenate(critic_parts, axis=1).astype(np.float32, copy=False)
        if critic_obs.shape != state.critic_obs.shape:
            raise RuntimeError(
                f"Go1 numpy task built critic obs shape {critic_obs.shape}, expected {state.critic_obs.shape}"
            )
        np.copyto(state.critic_obs, critic_obs)

    def _run_unilab_flat_task_numpy(self) -> None:
        state = self.backend.state
        cfg = self.cfg_obj.unilab_rewards
        scales = dict(cfg.scales)
        linvel = np.asarray(state.base_linear_velocity_body, dtype=np.float32)
        gyro = np.asarray(state.base_angular_velocity_body, dtype=np.float32)
        gravity_down = np.asarray(state.projected_gravity, dtype=np.float32)
        gravity_up = -gravity_down
        dof_pos = np.asarray(state.joint_position, dtype=np.float32)
        dof_vel = np.asarray(state.joint_velocity, dtype=np.float32)
        default = np.asarray(state.default_joint_position, dtype=np.float32).reshape(1, -1)
        diff = dof_pos - default
        commands = np.asarray(state.command, dtype=np.float32)
        current_actions = np.asarray(state.action, dtype=np.float32)
        last_actions = np.asarray(state.last_action, dtype=np.float32)

        if self._advance_task_time:
            self._update_gait_phase()
        foot_force = np.asarray(state.foot_contact_force, dtype=np.float32)
        foot_contact = np.asarray(state.foot_contact, dtype=np.float32)
        contact_z = foot_force[:, :, 2]
        contact = (contact_z > 0.1) | (foot_contact > 0.0)
        feet_pos = np.asarray(state.foot_position, dtype=np.float32)

        actor_obs = np.concatenate(
            [
                self._unilab_obs_noise(gyro, "gyro"),
                self._unilab_obs_noise(gravity_down, "gravity"),
                self._unilab_obs_noise(diff, "joint_angle"),
                self._unilab_obs_noise(dof_vel, "joint_vel"),
                current_actions,
                commands,
                self._feet_phase,
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        critic_obs = np.concatenate(
            [gyro, gravity_down, diff, dof_vel, current_actions, commands, self._feet_phase, linvel],
            axis=1,
        ).astype(np.float32, copy=False)
        self._copy_obs(state, actor_obs, critic_obs)

        terms = self._zero_reward_terms(state)
        self._set_reward_term(terms, "tracking_lin_vel", np.exp(-np.sum(np.square(commands[:, :2] - linvel[:, :2]), axis=1) / cfg.tracking_sigma))
        self._set_reward_term(terms, "tracking_ang_vel", np.exp(-np.square(commands[:, 2] - gyro[:, 2]) / cfg.tracking_sigma))
        self._set_reward_term(terms, "lin_vel_z", np.square(linvel[:, 2]))
        self._set_reward_term(terms, "ang_vel_xy", np.sum(np.square(gyro[:, :2]), axis=1))
        self._set_reward_term(terms, "base_height", np.square(np.asarray(state.base_position, dtype=np.float32)[:, 2] - cfg.base_height_target))
        self._set_reward_term(terms, "action_rate", np.sum(np.square(current_actions - last_actions), axis=1))
        self._set_reward_term(terms, "similar_to_default", np.sum(np.abs(diff), axis=1))
        is_contact_phase = (self._feet_phase < 0.6).astype(bool)
        self._set_reward_term(terms, "contact", np.sum(~np.logical_xor(contact, is_contact_phase), axis=1))
        is_swing = self._feet_phase >= 0.6
        height_error = np.square(feet_pos[:, :, 2] - 0.1)
        swing_rew = np.exp(-height_error / 0.01) * is_swing
        self._set_reward_term(terms, "swing_feet_z", np.sum(swing_rew, axis=1) / max(self._foot_count, 1))
        self._finish_unilab_reward(state, terms, scales)

        terminated = gravity_up[:, 2] <= 0.5
        np.copyto(state.terminated, terminated.astype(np.uint8))
        np.copyto(state.velocity_error, np.linalg.norm(commands[:, :2] - linvel[:, :2], axis=1).astype(np.float32))
        np.copyto(state.base_clearance, np.asarray(state.base_position, dtype=np.float32)[:, 2])

    def _run_unilab_rough_task_numpy(self) -> None:
        state = self.backend.state
        cfg = self.cfg_obj.unilab_rewards
        scales = dict(cfg.scales)
        linvel = np.asarray(state.base_linear_velocity_body, dtype=np.float32)
        gyro = np.asarray(state.base_angular_velocity_body, dtype=np.float32)
        gravity_down = np.asarray(state.projected_gravity, dtype=np.float32)
        gravity_up = -gravity_down
        dof_pos = np.asarray(state.joint_position, dtype=np.float32)
        dof_vel = np.asarray(state.joint_velocity, dtype=np.float32)
        default = np.asarray(state.default_joint_position, dtype=np.float32).reshape(1, -1)
        diff = dof_pos - default
        commands = np.asarray(state.command, dtype=np.float32)
        current_actions = np.asarray(state.action, dtype=np.float32)
        last_actions = np.asarray(state.last_action, dtype=np.float32)
        torques = self._estimate_unilab_pd_torques(current_actions, dof_pos, dof_vel)
        qacc = np.asarray((dof_vel - self._last_dof_vel_for_acc) / max(self.step_dt, 1.0e-6), dtype=np.float32)
        self._last_dof_vel_for_acc[:] = dof_vel
        if self._advance_task_time:
            self._update_gait_phase()
            self._update_contact_timers(self._foot_contact_mask(state))

        actor_obs = np.concatenate(
            [
                self._unilab_obs_noise(gyro, "gyro") * 0.25,
                self._unilab_obs_noise(gravity_down, "gravity"),
                commands,
                self._unilab_obs_noise(diff, "joint_angle"),
                self._unilab_obs_noise(dof_vel, "joint_vel") * 0.05,
                current_actions,
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        critic_base = np.concatenate([linvel, gyro, gravity_down, commands, diff, dof_vel, current_actions], axis=1)
        height_scan = self._unilab_height_scan_obs(state)
        critic_obs = np.concatenate([critic_base, height_scan], axis=1).astype(np.float32, copy=False)
        self._copy_obs(state, actor_obs, critic_obs)

        upright_scale = np.clip(gravity_up[:, 2], 0.0, 0.7) / 0.7
        terms = self._zero_reward_terms(state)
        self._set_reward_term(terms, "lin_vel_z", np.square(linvel[:, 2]) * upright_scale)
        self._set_reward_term(terms, "ang_vel_xy", np.sum(np.square(gyro[:, :2]), axis=1) * upright_scale)
        self._set_reward_term(terms, "joint_torques_l2", np.sum(np.square(torques), axis=1) * upright_scale)
        self._set_reward_term(terms, "joint_acc_l2", np.sum(np.square(qacc), axis=1) * upright_scale)
        self._set_reward_term(terms, "joint_power", np.sum(np.abs(dof_vel * torques), axis=1) * upright_scale)
        command_norm = np.linalg.norm(commands, axis=1)
        body_xy_vel = np.linalg.norm(linvel[:, :2], axis=1)
        stopped = command_norm < cfg.stand_still_command_threshold
        self._set_reward_term(terms, "stand_still", np.sum(np.abs(diff), axis=1) * stopped * upright_scale)
        self._set_reward_term(terms, "hip_pos", np.sum(np.square(diff[:, _GO1_HIP_INDICES]), axis=1) * upright_scale)
        running_error = np.linalg.norm(diff, axis=1)
        moving = (command_norm > cfg.joint_pos_penalty_command_threshold) | (body_xy_vel > cfg.joint_pos_penalty_velocity_threshold)
        self._set_reward_term(terms, "joint_pos_penalty", np.where(moving, running_error, cfg.joint_pos_penalty_stand_still_scale * running_error) * upright_scale)
        fr_rl = dof_pos[:, 0:3] - dof_pos[:, 9:12]
        fl_rr = dof_pos[:, 3:6] - dof_pos[:, 6:9]
        self._set_reward_term(terms, "joint_mirror", 0.5 * (np.sum(np.square(fr_rl), axis=1) + np.sum(np.square(fl_rr), axis=1)) * upright_scale)
        self._set_reward_term(terms, "action_rate", np.sum(np.square(current_actions - last_actions), axis=1))
        categorized_undesired = (
            np.asarray(getattr(state, "base_collision_count", 0.0), dtype=np.float32)
            + np.asarray(getattr(state, "hip_collision_count", 0.0), dtype=np.float32)
            + np.asarray(getattr(state, "thigh_collision_count", 0.0), dtype=np.float32)
            + np.asarray(getattr(state, "calf_collision_count", 0.0), dtype=np.float32)
        )
        if np.asarray(categorized_undesired).shape != (self.num_envs,):
            categorized_undesired = np.asarray(state.illegal_contact_count, dtype=np.float32)
        self._set_reward_term(terms, "undesired_contacts", categorized_undesired * upright_scale)
        force_norm = np.linalg.norm(np.asarray(state.foot_contact_force, dtype=np.float32), axis=2)
        force_clip = np.minimum(force_norm, 1500.0)
        self._set_reward_term(terms, "contact_forces", np.sum(np.clip(force_clip - cfg.contact_forces_threshold, 0.0, None), axis=1) * upright_scale)
        moving_cmd = command_norm > 0.1
        self._set_reward_term(terms, "tracking_lin_vel", np.exp(-np.sum(np.square(commands[:, :2] - linvel[:, :2]), axis=1) / cfg.tracking_sigma) * upright_scale)
        self._set_reward_term(terms, "tracking_ang_vel", np.exp(-np.square(commands[:, 2] - gyro[:, 2]) / cfg.tracking_sigma) * upright_scale)
        self._set_reward_term(terms, "feet_air_time", np.sum((self._last_air_time - cfg.feet_air_time_threshold) * self._first_foot_contact, axis=1) * moving_cmd * upright_scale)
        self._set_reward_term(terms, "feet_air_time_variance", (np.var(np.clip(self._last_air_time, 0.0, 0.5), axis=1) + np.var(np.clip(self._last_contact_time, 0.0, 0.5), axis=1)) * upright_scale)
        self._set_reward_term(terms, "feet_contact_without_cmd", np.sum(self._first_foot_contact, axis=1) * (command_norm < 0.1) * upright_scale)
        foot_vel_body = self._relative_foot_velocity_body(state)
        foot_pos_body = self._relative_foot_position_body(state)
        contact_mask = self._foot_contact_mask(state)
        self._set_reward_term(terms, "feet_slide", np.sum(np.linalg.norm(foot_vel_body[:, :, :2], axis=2) * contact_mask, axis=1) * upright_scale)
        z_error = np.square(foot_pos_body[:, :, 2] - cfg.feet_height_body_target)
        velocity_tanh = np.tanh(cfg.feet_height_body_tanh_mult * np.linalg.norm(foot_vel_body[:, :, :2], axis=2))
        self._set_reward_term(terms, "feet_height_body", np.sum(z_error * velocity_tanh, axis=1) * moving_cmd * upright_scale)
        self._set_reward_term(terms, "feet_gait", self._feet_gait_reward(command_norm, body_xy_vel) * upright_scale)
        self._set_reward_term(terms, "upward", np.square(1.0 + gravity_up[:, 2]))
        self._finish_unilab_reward(state, terms, scales)

        np.copyto(state.terminated, np.zeros((self.num_envs,), dtype=np.uint8))
        np.copyto(state.velocity_error, np.linalg.norm(commands[:, :2] - linvel[:, :2], axis=1).astype(np.float32))
        np.copyto(state.foot_slip, np.sum(np.linalg.norm(foot_vel_body[:, :, :2], axis=2) * contact_mask, axis=1).astype(np.float32))
        np.copyto(state.base_clearance, self._unilab_base_height_from_scan(state))

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

    def _finish_unilab_reward(self, state: Any, terms: np.ndarray, scales: Mapping[str, float]) -> None:
        weighted = np.zeros((self.num_envs,), dtype=np.float32)
        for index, name in enumerate(self._reward_term_names):
            terms[:, index] *= float(scales.get(name, 0.0))
            weighted += terms[:, index]
        np.copyto(state.reward, (weighted * self.step_dt).astype(np.float32))

    def _unilab_obs_noise(self, values: np.ndarray, name: str) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if not self.cfg_obj.observations.actor_noise:
            return values
        noise_cfg = self.cfg_obj.observations.unilab_noise
        scale = {
            "joint_angle": noise_cfg.scale_joint_angle,
            "joint_vel": noise_cfg.scale_joint_vel,
            "gyro": noise_cfg.scale_gyro,
            "gravity": noise_cfg.scale_gravity,
            "linvel": noise_cfg.scale_linvel,
        }.get(name, 0.0)
        level = float(noise_cfg.level)
        if level <= 0.0 or scale <= 0.0:
            return values
        noise = self._rng.uniform(-1.0, 1.0, values.shape).astype(np.float32)
        return values + noise * level * float(scale)

    def _update_gait_phase(self) -> None:
        self._gait_phase = np.fmod(self._gait_phase + self.step_dt * 2.0, 1.0).astype(np.float32)
        if self._foot_count >= 4:
            self._feet_phase[:, 0] = self._gait_phase
            self._feet_phase[:, 3] = self._gait_phase
            self._feet_phase[:, 1] = np.fmod(self._gait_phase + 0.5, 1.0)
            self._feet_phase[:, 2] = np.fmod(self._gait_phase + 0.5, 1.0)

    def _foot_contact_mask(self, state: Any) -> np.ndarray:
        force = np.asarray(state.foot_contact_force, dtype=np.float32)
        force_norm = np.linalg.norm(force, axis=2)
        contact_sensor = np.asarray(state.foot_contact, dtype=np.float32) > 0.0
        return np.asarray((force_norm > self.cfg_obj.unilab_rewards.contact_threshold) | contact_sensor, dtype=bool)

    def _update_contact_timers(self, contact: np.ndarray) -> None:
        contact = np.asarray(contact, dtype=bool)
        first_contact = contact & ~self._last_foot_contact_mask
        first_air = ~contact & self._last_foot_contact_mask
        self._first_foot_contact[:] = first_contact
        self._last_air_time[first_contact] = self._current_air_time[first_contact]
        self._last_contact_time[first_air] = self._current_contact_time[first_air]
        self._current_air_time[contact] = 0.0
        self._current_air_time[~contact] += self.step_dt
        self._current_contact_time[~contact] = 0.0
        self._current_contact_time[contact] += self.step_dt
        self._last_foot_contact_mask[:] = contact

    def _reset_contact_timers(self, env_ids: np.ndarray) -> None:
        rows = np.asarray(env_ids, dtype=np.int64)
        self._current_air_time[rows] = 0.0
        self._current_contact_time[rows] = 0.0
        self._last_air_time[rows] = 0.0
        self._last_contact_time[rows] = 0.0
        self._first_foot_contact[rows] = False
        self._last_foot_contact_mask[rows] = False

    def _estimate_unilab_pd_torques(self, actions: np.ndarray, dof_pos: np.ndarray, dof_vel: np.ndarray) -> np.ndarray:
        targets = actions * self.action_scale.reshape(1, -1) + self.default_joint_pos.reshape(1, -1)
        return (float(self.cfg_obj.kp) * (targets - dof_pos) - float(self.cfg_obj.kd) * dof_vel).astype(np.float32)

    def _relative_foot_velocity_body(self, state: Any) -> np.ndarray:
        base_quat = np.asarray(state.base_quaternion, dtype=np.float32)
        base_vel = np.asarray(state.base_linear_velocity, dtype=np.float32)
        relative = np.asarray(state.foot_velocity, dtype=np.float32) - base_vel[:, None, :]
        flat = relative.reshape(self.num_envs * self._foot_count, 3)
        quat = np.repeat(base_quat, self._foot_count, axis=0)
        return _quat_rotate_inv_batch(flat, quat).reshape(self.num_envs, self._foot_count, 3)

    def _relative_foot_position_body(self, state: Any) -> np.ndarray:
        base_quat = np.asarray(state.base_quaternion, dtype=np.float32)
        base_pos = np.asarray(state.base_position, dtype=np.float32)
        relative = np.asarray(state.foot_position, dtype=np.float32) - base_pos[:, None, :]
        flat = relative.reshape(self.num_envs * self._foot_count, 3)
        quat = np.repeat(base_quat, self._foot_count, axis=0)
        return _quat_rotate_inv_batch(flat, quat).reshape(self.num_envs, self._foot_count, 3)

    def _unilab_height_scan_obs(self, state: Any) -> np.ndarray:
        if self._height_scan_dim <= 0:
            return np.zeros((self.num_envs, 0), dtype=np.float32)
        base_z = np.asarray(state.base_position, dtype=np.float32)[:, 2:3]
        raw_heights = self._height_scan_world_height(state)
        heights = np.clip(base_z - 0.5 - raw_heights, -1.0, 1.0)
        return (heights * 5.0).astype(np.float32, copy=False)

    def _unilab_base_height_from_scan(self, state: Any) -> np.ndarray:
        if self._height_scan_dim <= 0:
            return np.asarray(state.base_position, dtype=np.float32)[:, 2]
        base_z = np.asarray(state.base_position, dtype=np.float32)[:, 2:3]
        raw_heights = self._height_scan_world_height(state)
        return np.mean(base_z - raw_heights, axis=1).astype(np.float32, copy=False)

    def _height_scan_world_height(self, state: Any) -> np.ndarray:
        scan = np.asarray(state.height_scan, dtype=np.float32)
        points = np.asarray(getattr(state, "height_scan_point", np.empty((self.num_envs, 0, 3))), dtype=np.float32)
        if points.shape[:2] == scan.shape and points.shape[2:] == (3,):
            hit = np.asarray(getattr(state, "height_scan_hit", np.ones(scan.shape, dtype=bool)), dtype=bool)
            base_z = np.asarray(state.base_position, dtype=np.float32)[:, 2:3]
            return np.where(hit, points[:, :, 2], base_z - scan)
        base_z = np.asarray(state.base_position, dtype=np.float32)[:, 2:3]
        return base_z - scan

    def _feet_gait_reward(self, command_norm: np.ndarray, body_xy_vel: np.ndarray) -> np.ndarray:
        cfg = self.cfg_obj.unilab_rewards
        enabled = (command_norm > cfg.feet_gait_command_threshold) | (body_xy_vel > cfg.feet_gait_velocity_threshold)
        air = self._current_air_time
        contact = self._current_contact_time
        sync_fl_rr = _gait_sync_reward(air, contact, _GO1_FRONT_LEFT, _GO1_REAR_RIGHT, cfg.feet_gait_std, cfg.feet_gait_max_err)
        sync_fr_rl = _gait_sync_reward(air, contact, _GO1_FRONT_RIGHT, _GO1_REAR_LEFT, cfg.feet_gait_std, cfg.feet_gait_max_err)
        async_fl_fr = _gait_async_reward(air, contact, _GO1_FRONT_LEFT, _GO1_FRONT_RIGHT, cfg.feet_gait_std, cfg.feet_gait_max_err)
        async_rr_rl = _gait_async_reward(air, contact, _GO1_REAR_RIGHT, _GO1_REAR_LEFT, cfg.feet_gait_std, cfg.feet_gait_max_err)
        async_fl_rl = _gait_async_reward(air, contact, _GO1_FRONT_LEFT, _GO1_REAR_LEFT, cfg.feet_gait_std, cfg.feet_gait_max_err)
        async_fr_rr = _gait_async_reward(air, contact, _GO1_FRONT_RIGHT, _GO1_REAR_RIGHT, cfg.feet_gait_std, cfg.feet_gait_max_err)
        return (sync_fl_rr * sync_fr_rl * async_fl_fr * async_rr_rl * async_fl_rl * async_fr_rr * enabled).astype(np.float32)

    def _apply_pushes(self) -> None:
        if not self.cfg_obj.push_enabled:
            return
        self._push_step_left -= 1
        env_ids = np.flatnonzero(self._push_step_left <= 0).astype(np.int64, copy=False)
        if env_ids.size == 0:
            return
        push_force = self._sample_range_matrix(self.cfg_obj.push_force_ranges, ("x", "y", "z"), env_ids.size)
        self.backend.set_push_forces(env_ids.tolist(), push_force)
        self._push_count[env_ids] += 1
        self._push_step_left[env_ids] = max(1, int(self.cfg_obj.push_interval_steps))

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
            reset_lin_vel = np.zeros(3, dtype=np.float32)
            reset_ang_vel = np.zeros(3, dtype=np.float32)
            orientation = _quat_from_yaw(yaw)
            joint_pos_noise = (-0.05, 0.05)
            joint_vel_noise = (-0.05, 0.05)
            if self._task_profile == "unilab_flat":
                reset_lin_vel = self._rng.uniform(-0.5, 0.5, 3).astype(np.float32)
                reset_ang_vel = self._rng.uniform(-0.5, 0.5, 3).astype(np.float32)
            elif self._task_profile == "unilab_rough":
                base_z += float(self._rng.uniform(0.25, 0.5))
                roll = float(self._rng.uniform(-3.14, 3.14))
                pitch = float(self._rng.uniform(-3.14, 3.14))
                yaw = float(self._rng.uniform(-3.14, 3.14))
                orientation = _quat_from_euler_xyz(roll, pitch, yaw)
                reset_lin_vel = self._rng.uniform(-0.5, 0.5, 3).astype(np.float32)
                reset_ang_vel = self._rng.uniform(-0.5, 0.5, 3).astype(np.float32)
                joint_pos_noise = (0.0, 0.0)
                joint_vel_noise = (0.0, 0.0)
            elif self.cfg_obj.domain_randomization.enabled:
                reset_lin_vel = self._sample_range_vec(self.cfg_obj.domain_randomization.reset_lin_vel_ranges, ("x", "y", "z"))
                reset_ang_vel = self._sample_range_vec(self.cfg_obj.domain_randomization.reset_ang_vel_ranges, ("x", "y", "z"))
            if self.cfg_obj.domain_randomization.enabled:
                lo, hi = self.cfg_obj.domain_randomization.encoder_bias_range
                self._encoder_bias[env_id] = self._rng.uniform(lo, hi, self.num_actions).astype(np.float32)
            else:
                self._encoder_bias[env_id] = 0.0

            base_positions[row] = np.asarray([float(spawn[0]), float(spawn[1]), float(base_z)], dtype=np.float32)
            base_orientations[row] = orientation
            base_linear_velocities[row] = reset_lin_vel
            base_angular_velocities[row] = reset_ang_vel
            joint_positions[row] = self.default_joint_pos + self._rng.uniform(joint_pos_noise[0], joint_pos_noise[1], self.num_actions).astype(np.float32)
            joint_velocities[row] = self._rng.uniform(joint_vel_noise[0], joint_vel_noise[1], self.num_actions).astype(np.float32)

            self._spawn_indices[env_id] = spawn_index
            self._terrain_levels[env_id] = float(self._spawn_levels[spawn_index])
            self._episode_start_xy[env_id] = spawn[:2]
            self._episode_length_np[env_id] = 0
            self._episode_returns[env_id] = 0.0
            self._previous_actions[env_id] = 0.0
            self._last_actions[env_id] = 0.0
            self._gait_phase[env_id] = 0.0
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
        self._apply_reset_domain_randomization(env_ids)
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
        self._last_dof_vel_for_acc[rows] = joint_velocities
        self._reset_contact_timers(rows)
        self.backend.reset_commands(env_ids.tolist())

    def _apply_reset_domain_randomization(self, env_ids: np.ndarray) -> None:
        if not hasattr(self.backend, "reset_domain_randomization"):
            return
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if env_ids.size == 0:
            return
        dr = self.cfg_obj.domain_randomization
        count = int(env_ids.size)
        base_mass_delta = np.zeros((count,), dtype=np.float32)
        base_com_offset = np.zeros((count, 3), dtype=np.float32)
        joint_kp = np.zeros((count, self.num_actions), dtype=np.float32)
        joint_kd = np.zeros((count, self.num_actions), dtype=np.float32)
        if dr.enabled:
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
                joint_kp[:] = float(self.cfg_obj.kp) * self._rng.uniform(lo, hi, (count, 1)).astype(np.float32)
            if dr.randomize_kd:
                lo, hi = dr.kd_multiplier_range
                joint_kd[:] = float(self.cfg_obj.kd) * self._rng.uniform(lo, hi, (count, 1)).astype(np.float32)
        self.backend.reset_domain_randomization(
            env_ids.tolist(),
            base_mass_delta=base_mass_delta,
            base_com_offset=base_com_offset,
            joint_kp=joint_kp,
            joint_kd=joint_kd,
        )

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
        interval = max(1, int(self.cfg_obj.push_interval_steps))
        self._push_step_left[env_id] = int(self._rng.integers(1, interval + 1))
        self._push_time_left[env_id] = float(self._push_step_left[env_id]) * self.step_dt

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
