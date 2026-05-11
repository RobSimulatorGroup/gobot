"""Core reinforcement-learning environment helpers for Gobot.

This module deliberately stays above Gobot's public scene and simulation APIs.
It does not depend on ImGui, editor play mode, raw MuJoCo pointers, Gymnasium,
or MuJoCo Warp internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from . import _core


ArrayLike = Sequence[float] | np.ndarray


def _as_float_array(values: ArrayLike, *, shape: tuple[int, ...] | None = None, name: str = "array") -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _backend_type(value: str | _core.PhysicsBackendType) -> _core.PhysicsBackendType:
    if isinstance(value, _core.PhysicsBackendType):
        return value
    normalized = str(value).lower()
    if normalized == "null":
        return _core.PhysicsBackendType.Null
    if normalized in {"mujoco", "mujoco_cpu"}:
        return _core.PhysicsBackendType.MuJoCoCpu
    if normalized in {"mujoco_warp", "warp"}:
        return _core.PhysicsBackendType.MuJoCoWarp
    raise ValueError(f"unsupported Gobot RL backend: {value!r}")


def _find_robot(name_map: Mapping[str, Any], robot_name: str | None = None) -> Mapping[str, Any]:
    robots = list(name_map.get("robots", []))
    if robot_name is not None:
        for robot in robots:
            if robot.get("name") == robot_name:
                return robot
        raise ValueError(f"runtime name map has no robot named {robot_name!r}")
    if not robots:
        raise ValueError("runtime name map has no robots")
    return robots[0]


def _find_state_robot(state: Mapping[str, Any], robot_name: str) -> Mapping[str, Any]:
    for robot in state.get("robots", []):
        if robot.get("name") == robot_name:
            return robot
    raise ValueError(f"runtime state has no robot named {robot_name!r}")


def _joint_is_controllable(joint: Mapping[str, Any]) -> bool:
    joint_type = joint.get("type")
    return joint_type in {
        _core.JointType.Revolute,
        _core.JointType.Continuous,
        _core.JointType.Prismatic,
    }


@dataclass(frozen=True)
class VectorSpec:
    """Flat vector names, bounds, and units used by actions or observations."""

    names: tuple[str, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    units: tuple[str, ...] = field(default_factory=tuple)
    version: str = "gobot.vector.v1"

    def __post_init__(self) -> None:
        size = len(self.names)
        if len(self.lower_bounds) != size or len(self.upper_bounds) != size:
            raise ValueError("VectorSpec names and bounds must have the same length")
        if self.units and len(self.units) != size:
            raise ValueError("VectorSpec units must be empty or match names length")
        if not self.units:
            object.__setattr__(self, "units", tuple("" for _ in range(size)))

    @property
    def shape(self) -> tuple[int]:
        return (len(self.names),)

    @property
    def size(self) -> int:
        return len(self.names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "names": list(self.names),
            "lower_bounds": list(self.lower_bounds),
            "upper_bounds": list(self.upper_bounds),
            "units": list(self.units),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VectorSpec":
        return cls(
            names=tuple(str(item) for item in value.get("names", ())),
            lower_bounds=tuple(float(item) for item in value.get("lower_bounds", ())),
            upper_bounds=tuple(float(item) for item in value.get("upper_bounds", ())),
            units=tuple(str(item) for item in value.get("units", ())),
            version=str(value.get("version", "gobot.vector.v1")),
        )


class VectorSpace:
    """Small dependency-free Box-like space for wrappers and tests."""

    def __init__(
        self,
        low: Sequence[float],
        high: Sequence[float],
        *,
        names: Sequence[str] | None = None,
        units: Sequence[str] | None = None,
        seed: int | None = None,
    ) -> None:
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        if self.low.shape != self.high.shape:
            raise ValueError("VectorSpace low/high bounds must have the same shape")
        self.shape = self.low.shape
        self.dtype = np.float64
        self.names = list(names or [])
        self.units = list(units or [])
        self._rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        lower = np.where(np.isfinite(self.low), self.low, -1.0)
        upper = np.where(np.isfinite(self.high), self.high, 1.0)
        return self._rng.uniform(lower, upper).astype(self.dtype, copy=False)

    def contains(self, value: ArrayLike) -> bool:
        array = np.asarray(value, dtype=np.float64)
        return array.shape == self.shape and bool(np.all(array >= self.low) and np.all(array <= self.high))


def space_from_spec(spec: VectorSpec | Mapping[str, Any]) -> VectorSpace:
    if not isinstance(spec, VectorSpec):
        spec = VectorSpec.from_dict(spec)
    return VectorSpace(spec.lower_bounds, spec.upper_bounds, names=spec.names, units=spec.units)


@dataclass
class EnvConfig:
    scene: str = ""
    backend: str | _core.PhysicsBackendType = "mujoco"
    num_envs: int = 1
    physics_dt: float = 1.0 / 240.0
    decimation: int = 1
    episode_length_s: float = 5.0
    robot: str | None = None
    controlled_joints: tuple[str, ...] | None = None
    auto_reset: bool = True
    project_path: str | None = None
    simulation: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "EnvConfig":
        data = dict(value or {})
        accepted = set(cls.__dataclass_fields__)
        if "actions" in data and "controlled_joints" not in data:
            actions = data.get("actions") or {}
            joints = actions.get("controlled_joints") or actions.get("joints")
            if joints is not None:
                data["controlled_joints"] = tuple(joints)
        if data.get("controlled_joints") is not None:
            data["controlled_joints"] = tuple(str(item) for item in data["controlled_joints"])
        data = {key: data[key] for key in accepted if key in data}
        return cls(**data)


class ActionManager:
    """Clips batched normalized actions and applies them through Gobot controls."""

    def __init__(self, env: "ManagerBasedEnv", joint_names: Sequence[str]) -> None:
        self.env = env
        self.joint_names = tuple(joint_names)
        self.spec = VectorSpec(
            names=tuple(f"{name}/target_position_normalized" for name in self.joint_names),
            lower_bounds=tuple(-1.0 for _ in self.joint_names),
            upper_bounds=tuple(1.0 for _ in self.joint_names),
            units=tuple("normalized" for _ in self.joint_names),
            version="gobot.action.normalized_joint_position.v1",
        )
        self.last_action = np.zeros((env.num_envs, self.spec.size), dtype=np.float64)

    def process_action(self, action: ArrayLike) -> np.ndarray:
        expected_shape = (self.env.num_envs, self.spec.size)
        array = _as_float_array(action, shape=expected_shape, name="action")
        clipped = np.clip(array, -1.0, 1.0)
        self.last_action = clipped.copy()
        return clipped

    def apply_action(self, processed_action: np.ndarray) -> None:
        if self.spec.size == 0:
            return
        self.env._backend.set_normalized_joint_position_action(0, self.joint_names, processed_action[0])


class ObservationManager:
    """Builds flat observations from engine-facing runtime state."""

    def __init__(self, env: "ManagerBasedEnv", name_map: Mapping[str, Any], robot_name: str) -> None:
        self.env = env
        self.robot_name = robot_name
        robot = _find_robot(name_map, robot_name)
        joint_names = [str(joint["name"]) for joint in robot.get("joints", [])]

        names: list[str] = ["time", "episode_progress"]
        lower: list[float] = [0.0, 0.0]
        upper: list[float] = [math.inf, 1.0]
        units: list[str] = ["s", "ratio"]
        for joint in robot.get("joints", []):
            joint_name = str(joint["name"])
            names.extend([f"{joint_name}/position", f"{joint_name}/velocity"])
            lower_limit = float(joint.get("lower_limit", -math.inf))
            upper_limit = float(joint.get("upper_limit", math.inf))
            if joint.get("type") == _core.JointType.Continuous or lower_limit >= upper_limit:
                lower_limit = -math.inf
                upper_limit = math.inf
            lower.extend([lower_limit, -math.inf])
            upper.extend([upper_limit, math.inf])
            units.extend(["rad_or_m", "rad_or_m/s"])
        for joint_name in self.env.action_manager.joint_names:
            names.append(f"{joint_name}/previous_action")
            lower.append(-1.0)
            upper.append(1.0)
            units.append("normalized")

        self.joint_names = tuple(joint_names)
        self.spec = VectorSpec(
            names=tuple(names),
            lower_bounds=tuple(lower),
            upper_bounds=tuple(upper),
            units=tuple(units),
            version="gobot.observation.runtime_joint_state.v1",
        )

    def compute(self) -> np.ndarray:
        observations = np.zeros((self.env.num_envs, self.spec.size), dtype=np.float64)
        state = self.env._backend.get_runtime_state(0)
        robot = _find_state_robot(state, self.robot_name)
        joint_states = {str(joint["name"]): joint for joint in robot.get("joints", [])}

        values: list[float] = [
            float(self.env._backend.simulation_time(0)),
            min(self.env.episode_lengths[0] / max(self.env.max_episode_steps, 1), 1.0),
        ]
        for joint_name in self.joint_names:
            joint = joint_states.get(joint_name, {})
            values.append(float(joint.get("position", 0.0)))
            values.append(float(joint.get("velocity", 0.0)))
        values.extend(float(item) for item in self.env.action_manager.last_action[0])
        observations[0, :] = values
        return observations


class RewardManager:
    """Default alive reward scaled by env dt."""

    def __init__(self, env: "ManagerBasedEnv", terms: Sequence[Callable[["ManagerBasedEnv"], ArrayLike]] | None = None) -> None:
        self.env = env
        self.terms = list(terms or [])

    def compute(self) -> np.ndarray:
        reward = np.full((self.env.num_envs,), self.env.env_dt, dtype=np.float64)
        for term in self.terms:
            reward += _as_float_array(term(self.env), shape=(self.env.num_envs,), name="reward term")
        return reward


class TerminationManager:
    def __init__(
        self,
        env: "ManagerBasedEnv",
        conditions: Sequence[Callable[["ManagerBasedEnv"], ArrayLike]] | None = None,
    ) -> None:
        self.env = env
        self.conditions = list(conditions or [])

    def compute(self) -> tuple[np.ndarray, np.ndarray]:
        terminated = np.zeros((self.env.num_envs,), dtype=bool)
        for condition in self.conditions:
            terminated |= np.asarray(condition(self.env), dtype=bool)
        truncated = self.env.episode_lengths >= self.env.max_episode_steps
        return terminated, truncated


class EventManager:
    def __init__(self, env: "ManagerBasedEnv") -> None:
        self.env = env

    def reset(self, env_ids: np.ndarray) -> None:
        del env_ids


class CommandManager:
    def __init__(self, env: "ManagerBasedEnv") -> None:
        self.env = env

    def reset(self, env_ids: np.ndarray) -> None:
        del env_ids


class Recorder:
    def __init__(self) -> None:
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []

    def record_done(self, returns: np.ndarray, lengths: np.ndarray, done: np.ndarray) -> None:
        for index in np.flatnonzero(done):
            self.episode_returns.append(float(returns[index]))
            self.episode_lengths.append(int(lengths[index]))


class Metrics(dict):
    pass


class _SingleContextBackend:
    """Adapter for Gobot's current single EngineContext simulation API."""

    supports_vector_env = False

    def __init__(self, context: _core.AppContext, cfg: EnvConfig) -> None:
        self.context = context
        self.cfg = cfg

    def build(self) -> None:
        if self.cfg.project_path:
            self.context.set_project_path(self.cfg.project_path)
        if self.cfg.scene:
            self.context.load_scene(self.cfg.scene)
        elif not self.context.has_scene:
            raise RuntimeError(
                "ManagerBasedEnv requires cfg.scene or an already-loaded Gobot scene"
            )
        self.context.build_world(_backend_type(self.cfg.backend))
        self.context.reset_simulation()

    def reset(self, env_ids: Iterable[int], seed: int | None = None) -> None:
        del env_ids, seed
        self.context.reset_simulation()

    def set_normalized_joint_position_action(
        self,
        env_id: int,
        joint_names: Sequence[str],
        action: Sequence[float],
    ) -> None:
        if env_id != 0:
            raise IndexError("single Gobot context backend only supports env_id 0")
        robot_name = self.cfg.robot
        if robot_name is None:
            robot_name = _find_robot(self.context.get_runtime_name_map()).get("name")
        self.context.set_robot_named_action(str(robot_name), list(joint_names), list(float(item) for item in action))

    def step(self) -> None:
        self.context.step_once()

    def get_runtime_name_map(self) -> Mapping[str, Any]:
        return self.context.get_runtime_name_map()

    def get_runtime_state(self, env_id: int) -> Mapping[str, Any]:
        if env_id != 0:
            raise IndexError("single Gobot context backend only supports env_id 0")
        return self.context.get_runtime_state()

    def simulation_time(self, env_id: int) -> float:
        if env_id != 0:
            raise IndexError("single Gobot context backend only supports env_id 0")
        return float(self.context.simulation_time)


class ManagerBasedEnv:
    """Manager-structured Gobot RL environment.

    Step order:
    process action -> decimated physics substeps -> termination -> reward ->
    terminal observation -> auto reset done envs -> observation.
    """

    def __init__(self, cfg: Mapping[str, Any] | EnvConfig | None = None, *, context: _core.AppContext | None = None) -> None:
        self.cfg = cfg if isinstance(cfg, EnvConfig) else EnvConfig.from_mapping(cfg)
        if self.cfg.num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        if self.cfg.decimation < 1:
            raise ValueError("decimation must be at least 1")
        if self.cfg.physics_dt <= 0.0:
            raise ValueError("physics_dt must be greater than zero")

        self.context = context if context is not None else _core.app.context()
        self.num_envs = int(self.cfg.num_envs)
        self.physics_dt = float(self.cfg.physics_dt)
        self.decimation = int(self.cfg.decimation)
        self.env_dt = self.physics_dt * self.decimation
        self.max_episode_steps = max(1, int(math.ceil(float(self.cfg.episode_length_s) / self.env_dt)))
        self.auto_reset = bool(self.cfg.auto_reset)
        self._backend = _SingleContextBackend(self.context, self.cfg)
        if self.num_envs > 1 and not self._backend.supports_vector_env:
            raise NotImplementedError("current Gobot Python backend supports num_envs=1; CPU/Warp vector backends are not bound yet")

        self._backend.build()
        name_map = self._backend.get_runtime_name_map()
        robot = _find_robot(name_map, self.cfg.robot)
        self.robot_name = str(robot["name"])

        controlled_joints = self.cfg.controlled_joints
        if controlled_joints is None:
            controlled_joints = tuple(
                str(joint["name"]) for joint in robot.get("joints", []) if _joint_is_controllable(joint)
            )

        self.episode_lengths = np.zeros((self.num_envs,), dtype=np.int64)
        self.episode_returns = np.zeros((self.num_envs,), dtype=np.float64)
        self._last_observation: np.ndarray | None = None
        self._last_info: dict[str, Any] = {}
        self._rng = random.Random()

        self.action_manager = ActionManager(self, controlled_joints)
        self.observation_manager = ObservationManager(self, name_map, self.robot_name)
        self.reward_manager = RewardManager(self)
        self.termination_manager = TerminationManager(self)
        self.event_manager = EventManager(self)
        self.command_manager = CommandManager(self)
        self.recorder = Recorder()
        self.metrics = Metrics()

    @property
    def observation_spec(self) -> VectorSpec:
        return self.observation_manager.spec

    @property
    def action_spec(self) -> VectorSpec:
        return self.action_manager.spec

    @property
    def observation_space(self) -> VectorSpace:
        return space_from_spec(self.observation_spec)

    @property
    def action_space(self) -> VectorSpace:
        return space_from_spec(self.action_spec)

    def reset(
        self,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if seed is not None:
            self._rng.seed(int(seed))
        env_ids = np.arange(self.num_envs, dtype=np.int64)
        self._backend.reset(env_ids, seed=seed)
        self.episode_lengths[:] = 0
        self.episode_returns[:] = 0.0
        self.action_manager.last_action[:] = 0.0
        self.event_manager.reset(env_ids)
        self.command_manager.reset(env_ids)
        observation = self.observation_manager.compute()
        self._last_observation = observation.copy()
        info = {
            "seed": seed,
            "env_ids": env_ids.copy(),
            "runtime_name_map": self._backend.get_runtime_name_map(),
        }
        self._last_info = info
        return observation, info

    def step(self, action: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        processed_action = self.action_manager.process_action(action)

        for _ in range(self.decimation):
            self.action_manager.apply_action(processed_action)
            self._backend.step()

        self.episode_lengths += 1
        terminated, truncated = self.termination_manager.compute()
        reward = self.reward_manager.compute()
        self.episode_returns += reward
        done = terminated | truncated

        info: dict[str, Any] = {
            "episode_length": self.episode_lengths.copy(),
            "episode_return": self.episode_returns.copy(),
            "simulation_time": np.array([self._backend.simulation_time(0)], dtype=np.float64),
        }
        if np.any(done):
            terminal_observation = self.observation_manager.compute()
            info["terminal_observation"] = terminal_observation.copy()
            self.recorder.record_done(self.episode_returns, self.episode_lengths, done)
            if self.auto_reset:
                done_env_ids = np.flatnonzero(done).astype(np.int64)
                self._backend.reset(done_env_ids)
                self.episode_lengths[done_env_ids] = 0
                self.episode_returns[done_env_ids] = 0.0
                self.event_manager.reset(done_env_ids)
                self.command_manager.reset(done_env_ids)

        observation = self.observation_manager.compute()
        self._last_observation = observation.copy()
        self._last_info = info
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        pass


class VectorEnv(ManagerBasedEnv):
    """Alias for the batched manager environment API."""


class GymWrapper:
    """Gymnasium-style wrapper around a single-environment Gobot VectorEnv."""

    metadata = {"render_modes": []}

    def __init__(self, env: ManagerBasedEnv) -> None:
        if env.num_envs != 1:
            raise ValueError("GymWrapper requires env.num_envs == 1")
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        return observation[0].copy(), info

    def step(self, action: ArrayLike):
        action_array = _as_float_array(action, shape=(self.env.action_spec.size,), name="action")
        observation, reward, terminated, truncated, info = self.env.step(action_array.reshape(1, -1))
        if "terminal_observation" in info:
            info = dict(info)
            info["terminal_observation"] = info["terminal_observation"][0].copy()
        return observation[0].copy(), float(reward[0]), bool(terminated[0]), bool(truncated[0]), info

    def close(self) -> None:
        self.env.close()

    @property
    def unwrapped(self) -> ManagerBasedEnv:
        return self.env


class RslRlVecEnvWrapper:
    """Thin adapter exposing common rsl_rl vector-env attributes."""

    def __init__(self, env: ManagerBasedEnv) -> None:
        self.env = env
        self.num_envs = env.num_envs
        self.num_obs = env.observation_spec.size
        self.num_actions = env.action_spec.size

    def get_observations(self):
        if self.env._last_observation is None:
            observation, _ = self.env.reset()
            return observation
        return self.env._last_observation

    def reset(self):
        observation, _ = self.env.reset()
        return observation

    def step(self, actions):
        observation, reward, terminated, truncated, info = self.env.step(actions)
        done = terminated | truncated
        return observation, reward, done, info


__all__ = [
    "ActionManager",
    "CommandManager",
    "EnvConfig",
    "EventManager",
    "GymWrapper",
    "ManagerBasedEnv",
    "Metrics",
    "ObservationManager",
    "Recorder",
    "RewardManager",
    "RslRlVecEnvWrapper",
    "TerminationManager",
    "VectorEnv",
    "VectorSpace",
    "VectorSpec",
    "space_from_spec",
]
