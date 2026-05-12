"""Manager-based reinforcement-learning helpers for Gobot.

This module deliberately stays above Gobot's public scene and simulation APIs.
It does not depend on ImGui, editor play mode, raw MuJoCo pointers, Gymnasium,
or MuJoCo Warp internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import json
import math
from pathlib import Path
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


def _as_bool_array(values: ArrayLike, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=bool)
    if array.shape != shape:
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


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def _joint_limits(joint: Mapping[str, Any]) -> tuple[float, float]:
    lower_limit = float(joint.get("lower_limit", -math.inf))
    upper_limit = float(joint.get("upper_limit", math.inf))
    if joint.get("type") == _core.JointType.Continuous or lower_limit >= upper_limit:
        return -math.inf, math.inf
    return lower_limit, upper_limit


def _load_mapping_file(path: str | Path) -> Mapping[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("YAML RL task configs require PyYAML.") from error
        data = yaml.safe_load(text)
    elif suffix == ".py":
        spec = importlib.util.spec_from_file_location(f"_gobot_rl_task_{config_path.stem}", config_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load Python RL task config: {config_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "make_task"):
            data = module.make_task()
        elif hasattr(module, "TASK_CONFIG"):
            data = module.TASK_CONFIG
        elif hasattr(module, "CONFIG"):
            data = module.CONFIG
        else:
            raise ValueError("Python RL task config must define TASK_CONFIG, CONFIG, or make_task()")
    else:
        raise ValueError(f"unsupported RL task config format: {config_path.suffix}")
    if isinstance(data, TaskConfig):
        return data.to_dict()
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise TypeError("RL task config file must evaluate to a mapping")
    return data


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


@dataclass(frozen=True)
class ManagerTermConfig:
    """Serializable config for one action, observation, reward, termination, event, or command term."""

    name: str
    type: str
    weight: float = 1.0
    enabled: bool = True
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: str | Mapping[str, Any], *, default_name: str = "") -> "ManagerTermConfig":
        if isinstance(value, str):
            return cls(name=default_name or value, type=value)
        data = dict(value)
        params = dict(data.get("params", {}))
        known = {"name", "type", "weight", "enabled", "params"}
        for key in list(data):
            if key not in known:
                params[key] = data[key]
        term_type = str(data.get("type", default_name))
        name = str(data.get("name", default_name or term_type))
        return cls(
            name=name,
            type=term_type,
            weight=float(data.get("weight", 1.0)),
            enabled=bool(data.get("enabled", True)),
            params=params,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "weight": self.weight,
            "enabled": self.enabled,
            "params": dict(self.params),
        }


def _term_list(value: Any) -> tuple[ManagerTermConfig, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        if "terms" in value:
            value = value.get("terms")
        else:
            if any(key in value for key in ("controlled_joints", "joints", "version")):
                return ()
            return tuple(
                ManagerTermConfig.from_mapping(item, default_name=str(name))
                for name, item in value.items()
            )
    return tuple(
        ManagerTermConfig.from_mapping(item, default_name=str(index))
        for index, item in enumerate(value)
    )


def _mapping_value(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("expected a mapping")
    return value


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
    actions: Mapping[str, Any] = field(default_factory=dict)
    observations: Mapping[str, Any] = field(default_factory=dict)
    rewards: Mapping[str, Any] = field(default_factory=dict)
    terminations: Mapping[str, Any] = field(default_factory=dict)
    events: Mapping[str, Any] = field(default_factory=dict)
    commands: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "EnvConfig":
        data = dict(value or {})
        if "env" in data and isinstance(data.get("env"), Mapping):
            env_data = dict(data["env"])
            if "type" in env_data and "task" not in data:
                data["task"] = env_data["type"]
            if "project" in env_data and "project_path" not in env_data:
                env_data["project_path"] = env_data["project"]
            if "options" in env_data and isinstance(env_data["options"], Mapping):
                data.update(env_data["options"])
            data.update(env_data)

        if "project" in data and "project_path" not in data:
            data["project_path"] = data["project"]
        if "episode_length" in data and "episode_length_s" not in data:
            data["episode_length_s"] = data["episode_length"]
        if "max_episode_steps" in data and "episode_length_s" not in data:
            physics_dt = float(data.get("physics_dt", cls.physics_dt))
            decimation = int(data.get("decimation", cls.decimation))
            data["episode_length_s"] = float(data["max_episode_steps"]) * physics_dt * decimation

        if "actions" in data and "controlled_joints" not in data:
            actions = data.get("actions") or {}
            if isinstance(actions, Mapping):
                joints = actions.get("controlled_joints") or actions.get("joints")
                if joints is not None:
                    data["controlled_joints"] = tuple(joints)
        if data.get("controlled_joints") is not None:
            data["controlled_joints"] = tuple(str(item) for item in data["controlled_joints"])

        accepted = set(cls.__dataclass_fields__)
        filtered = {key: data[key] for key in accepted if key in data}
        for key in ("simulation", "actions", "observations", "rewards", "terminations", "events", "commands"):
            filtered[key] = _mapping_value(filtered.get(key, {}))
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "backend": str(self.backend),
            "num_envs": self.num_envs,
            "physics_dt": self.physics_dt,
            "decimation": self.decimation,
            "episode_length_s": self.episode_length_s,
            "robot": self.robot,
            "controlled_joints": None if self.controlled_joints is None else list(self.controlled_joints),
            "auto_reset": self.auto_reset,
            "project_path": self.project_path,
            "simulation": dict(self.simulation),
            "actions": dict(self.actions),
            "observations": dict(self.observations),
            "rewards": dict(self.rewards),
            "terminations": dict(self.terminations),
            "events": dict(self.events),
            "commands": dict(self.commands),
        }


@dataclass
class TaskConfig:
    """Top-level RL task config kept separate from the authored Gobot scene."""

    name: str = "gobot_task"
    scene: str = ""
    backend: str | _core.PhysicsBackendType = "mujoco"
    num_envs: int = 1
    physics_dt: float = 1.0 / 240.0
    decimation: int = 1
    episode_length_s: float = 5.0
    robot: str | None = None
    project_path: str | None = None
    auto_reset: bool = True
    actions: Mapping[str, Any] = field(default_factory=dict)
    observations: Mapping[str, Any] = field(default_factory=dict)
    rewards: Mapping[str, Any] = field(default_factory=dict)
    terminations: Mapping[str, Any] = field(default_factory=dict)
    events: Mapping[str, Any] = field(default_factory=dict)
    commands: Mapping[str, Any] = field(default_factory=dict)
    simulation: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "TaskConfig":
        data = dict(value or {})
        if "env" in data and isinstance(data["env"], Mapping):
            env_data = dict(data["env"])
            if "project" in env_data and "project_path" not in env_data:
                env_data["project_path"] = env_data["project"]
            if "type" in env_data and "name" not in data:
                data["name"] = env_data["type"]
            if "options" in env_data and isinstance(env_data["options"], Mapping):
                data.update(env_data["options"])
            data.update(env_data)
        if "project" in data and "project_path" not in data:
            data["project_path"] = data["project"]
        if "max_episode_steps" in data and "episode_length_s" not in data:
            physics_dt = float(data.get("physics_dt", cls.physics_dt))
            decimation = int(data.get("decimation", cls.decimation))
            data["episode_length_s"] = float(data["max_episode_steps"]) * physics_dt * decimation
        accepted = set(cls.__dataclass_fields__)
        filtered = {key: data[key] for key in accepted if key in data}
        for key in ("actions", "observations", "rewards", "terminations", "events", "commands", "simulation", "metadata"):
            filtered[key] = _mapping_value(filtered.get(key, {}))
        return cls(**filtered)

    @classmethod
    def from_file(cls, path: str | Path) -> "TaskConfig":
        return cls.from_mapping(_load_mapping_file(path))

    def env_config(self) -> EnvConfig:
        controlled_joints = None
        if isinstance(self.actions, Mapping):
            joints = self.actions.get("controlled_joints") or self.actions.get("joints")
            if joints is not None:
                controlled_joints = tuple(str(item) for item in joints)
        return EnvConfig(
            scene=self.scene,
            backend=self.backend,
            num_envs=self.num_envs,
            physics_dt=self.physics_dt,
            decimation=self.decimation,
            episode_length_s=self.episode_length_s,
            robot=self.robot,
            controlled_joints=controlled_joints,
            auto_reset=self.auto_reset,
            project_path=self.project_path,
            simulation=self.simulation,
            actions=self.actions,
            observations=self.observations,
            rewards=self.rewards,
            terminations=self.terminations,
            events=self.events,
            commands=self.commands,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scene": self.scene,
            "backend": str(self.backend),
            "num_envs": self.num_envs,
            "physics_dt": self.physics_dt,
            "decimation": self.decimation,
            "episode_length_s": self.episode_length_s,
            "robot": self.robot,
            "project_path": self.project_path,
            "auto_reset": self.auto_reset,
            "actions": dict(self.actions),
            "observations": dict(self.observations),
            "rewards": dict(self.rewards),
            "terminations": dict(self.terminations),
            "events": dict(self.events),
            "commands": dict(self.commands),
            "simulation": dict(self.simulation),
            "metadata": dict(self.metadata),
        }


def load_task_config(path: str | Path) -> TaskConfig:
    return TaskConfig.from_file(path)


def make_cartpole_target_task(
    *,
    scene: str = "res://cartpole.jscn",
    backend: str | _core.PhysicsBackendType = "mujoco",
    robot: str = "cartpole",
    project_path: str | None = None,
    slider_joint: str = "slider",
    hinge_joint: str = "hinge",
    target_cart_position: float = 1.0,
    randomize_target_position: bool = False,
    target_cart_position_range: tuple[float, float] = (-1.0, 1.0),
    max_episode_steps: int = 500,
    physics_dt: float = 1.0 / 240.0,
    decimation: int = 1,
    force_limit: float = 20.0,
    pole_angle_limit: float = 0.7,
    cart_position_limit: float = 2.4,
    target_tolerance: float = 0.05,
    target_near_tolerance: float = 0.1,
    target_velocity_tolerance: float = 0.2,
    target_overspeed_limit: float = 0.8,
    fast_reach_bonus: float = 20.0,
    initial_angle: float = 0.05,
    initial_cart_position: float = 0.0,
    randomize_initial_angle: bool = False,
    disturbance_force_std: float = 0.0,
    disturbance_impulse_probability: float = 0.0,
    disturbance_impulse_force: float = 0.0,
    disturbance_impulse_steps: int = 1,
) -> TaskConfig:
    """Create the standard CartPole target-reaching task config.

    The authored scene remains `.jscn`; this config describes the training
    task: action mapping, observation groups, reward terms, terminations,
    reset events, and target command generation.
    """

    episode_length_s = float(max_episode_steps) * float(physics_dt) * int(decimation)
    return TaskConfig(
        name="cartpole_target",
        scene=scene,
        backend=backend,
        num_envs=1,
        physics_dt=float(physics_dt),
        decimation=int(decimation),
        episode_length_s=episode_length_s,
        robot=robot,
        project_path=project_path,
        actions={
            "terms": [
                {
                    "name": "slider_effort",
                    "type": "joint_effort",
                    "joint": slider_joint,
                    "scale": float(force_limit),
                    "clip": [-1.0, 1.0],
                    "passive_joints": [hinge_joint],
                }
            ]
        },
        observations={
            "groups": {
                "policy": [
                    {"name": "cart_position", "type": "joint_position", "joint": slider_joint, "lower": -cart_position_limit, "upper": cart_position_limit, "unit": "m"},
                    {"name": "cart_velocity", "type": "joint_velocity", "joint": slider_joint, "lower": -math.inf, "upper": math.inf, "unit": "m/s"},
                    {"name": "pole_angle", "type": "joint_position", "joint": hinge_joint, "wrap": True, "lower": -math.pi, "upper": math.pi, "unit": "rad"},
                    {"name": "pole_angular_velocity", "type": "joint_velocity", "joint": hinge_joint, "lower": -math.inf, "upper": math.inf, "unit": "rad/s"},
                    {"name": "target_position_error", "type": "command_error", "command": "target_cart_position", "joint": slider_joint, "lower": -2.0 * cart_position_limit, "upper": 2.0 * cart_position_limit, "unit": "m"},
                ],
                "critic": [
                    {"type": "group", "group": "policy"},
                    {"name": "previous_action", "type": "previous_action", "action": "slider_effort", "lower": -1.0, "upper": 1.0, "unit": "normalized"},
                ],
            },
            "default_group": "policy",
        },
        rewards={
            "terms": [
                {"name": "alive", "type": "constant", "weight": 8.0, "scale_by_dt": False},
                {"name": "target_distance", "type": "squared_command_error", "weight": -8.0, "command": "target_cart_position", "joint": slider_joint},
                {"name": "cart_velocity", "type": "joint_velocity_l2", "weight": -0.5, "joint": slider_joint},
                {"name": "pole_angle", "type": "joint_position_l2", "weight": -300.0, "joint": hinge_joint, "wrap": True},
                {"name": "pole_angular_velocity", "type": "joint_velocity_l2", "weight": -8.0, "joint": hinge_joint},
                {"name": "overshoot", "type": "overshoot_l2", "weight": -20.0, "joint": slider_joint, "command": "target_cart_position"},
                {"name": "action_l2", "type": "action_l2", "weight": -0.02, "action": "slider_effort"},
                {"name": "target_progress", "type": "target_progress", "weight": 4.0, "joint": slider_joint, "command": "target_cart_position", "clip": [-0.05, 0.05]},
                {
                    "name": "settle_bonus",
                    "type": "cartpole_settle_bonus",
                    "weight": 8.0,
                    "slider_joint": slider_joint,
                    "hinge_joint": hinge_joint,
                    "command": "target_cart_position",
                    "target_tolerance": target_tolerance,
                    "velocity_tolerance": target_velocity_tolerance,
                    "angle_tolerance": 0.10,
                },
                {
                    "name": "fast_reach_bonus",
                    "type": "first_reach_bonus",
                    "weight": float(fast_reach_bonus),
                    "joint": slider_joint,
                    "hinge_joint": hinge_joint,
                    "command": "target_cart_position",
                    "target_tolerance": target_near_tolerance,
                    "angle_tolerance": 0.20,
                },
                {
                    "name": "overspeed_penalty",
                    "type": "target_overspeed",
                    "weight": -10.0,
                    "joint": slider_joint,
                    "command": "target_cart_position",
                    "target_tolerance": target_near_tolerance,
                    "velocity_limit": target_overspeed_limit,
                },
                {
                    "name": "fast_crossing_penalty",
                    "type": "target_fast_crossing",
                    "weight": -10.0,
                    "joint": slider_joint,
                    "command": "target_cart_position",
                    "velocity_tolerance": target_velocity_tolerance,
                },
                {"name": "failure_penalty", "type": "terminated", "weight": -100.0},
            ]
        },
        terminations={
            "terms": [
                {"name": "cart_position_limit", "type": "joint_position_abs_gt", "joint": slider_joint, "limit": cart_position_limit},
                {"name": "pole_angle_limit", "type": "joint_position_abs_gt", "joint": hinge_joint, "limit": pole_angle_limit, "wrap": True},
            ]
        },
        events={
            "terms": [
                {
                    "name": "reset_joint_state",
                    "type": "reset_joint_state",
                    "joints": {
                        slider_joint: {
                            "position": float(initial_cart_position),
                            "velocity": 0.0,
                        },
                        hinge_joint: {
                            "position": float(initial_angle),
                            "position_range": [-float(initial_angle), float(initial_angle)] if randomize_initial_angle else None,
                            "velocity": 0.0,
                        },
                    },
                },
                {
                    "name": "slider_disturbance",
                    "type": "joint_effort_disturbance",
                    "joint": slider_joint,
                    "std": disturbance_force_std,
                    "impulse_probability": disturbance_impulse_probability,
                    "impulse_force": disturbance_impulse_force,
                    "impulse_steps": disturbance_impulse_steps,
                },
            ]
        },
        commands={
            "terms": [
                {
                    "name": "target_cart_position",
                    "type": "uniform" if randomize_target_position else "constant",
                    "value": float(target_cart_position),
                    "range": [float(target_cart_position_range[0]), float(target_cart_position_range[1])],
                    "unit": "m",
                }
            ]
        },
        metadata={
            "task_kind": "cartpole_target",
            "slider_joint": slider_joint,
            "hinge_joint": hinge_joint,
            "force_limit": float(force_limit),
            "cart_position_limit": float(cart_position_limit),
            "pole_angle_limit": float(pole_angle_limit),
        },
    )


class RuntimeCache:
    def __init__(self, env: "ManagerBasedEnv") -> None:
        self.env = env
        self.state: Mapping[str, Any] | None = None
        self.robot: Mapping[str, Any] | None = None
        self.joints: dict[str, Mapping[str, Any]] = {}
        self.time = np.zeros((env.num_envs,), dtype=np.float64)

    def refresh(self) -> None:
        state = self.env._backend.get_runtime_state(0)
        robot = _find_state_robot(state, self.env.robot_name)
        self.state = state
        self.robot = robot
        self.joints = {str(joint["name"]): joint for joint in robot.get("joints", [])}
        self.time[0] = self.env._backend.simulation_time(0)

    def joint_position(self, name: str, *, wrap: bool = False) -> float:
        value = float(self.joints.get(name, {}).get("position", 0.0))
        return _wrap_angle(value) if wrap else value

    def joint_velocity(self, name: str) -> float:
        return float(self.joints.get(name, {}).get("velocity", 0.0))

    def joint_effort(self, name: str) -> float:
        return float(self.joints.get(name, {}).get("effort", 0.0))


@dataclass
class ActionTerm:
    name: str
    joint: str
    mode: str = "position"
    scale: float = 1.0
    offset: float = 0.0
    lower: float = -1.0
    upper: float = 1.0
    unit: str = "normalized"
    passive_joints: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, config: ManagerTermConfig) -> "ActionTerm":
        params = config.params
        clip = params.get("clip", (-1.0, 1.0))
        lower, upper = (float(clip[0]), float(clip[1])) if len(clip) == 2 else (-1.0, 1.0)
        term_type = str(config.type)
        mode = str(params.get("mode", term_type.removeprefix("joint_")))
        if mode in {"normalized_joint_position", "normalized_position"}:
            mode = "normalized_position"
        return cls(
            name=config.name,
            joint=str(params.get("joint", config.name)),
            mode=mode,
            scale=float(params.get("scale", 1.0)),
            offset=float(params.get("offset", 0.0)),
            lower=lower,
            upper=upper,
            unit=str(params.get("unit", "normalized")),
            passive_joints=tuple(str(item) for item in params.get("passive_joints", ())),
        )

    def apply(self, env: "ManagerBasedEnv", normalized_value: float) -> float:
        clipped = float(np.clip(normalized_value, self.lower, self.upper))
        target = self.offset + clipped * self.scale
        for joint in self.passive_joints:
            env._backend.set_joint_passive(0, joint)
        if self.mode in {"position", "target_position"}:
            env._backend.set_joint_position_target(0, self.joint, target)
        elif self.mode in {"normalized_position", "normalized"}:
            env._backend.set_normalized_joint_position_action(0, [self.joint], [clipped])
        elif self.mode in {"velocity", "target_velocity"}:
            env._backend.set_joint_velocity_target(0, self.joint, target)
        elif self.mode in {"effort", "force", "torque", "target_effort"}:
            disturbance = env.event_manager.sample_effort_disturbance(self.joint)
            env._backend.set_joint_effort_target(0, self.joint, target + disturbance)
        else:
            raise ValueError(f"unsupported action term mode: {self.mode!r}")
        return clipped


class ActionManager:
    """Clips batched normalized actions and applies them through Gobot controls."""

    def __init__(self, env: "ManagerBasedEnv", joint_names: Sequence[str], action_cfg: Mapping[str, Any] | None = None) -> None:
        self.env = env
        self.joint_names = tuple(joint_names)
        self._cfg = dict(action_cfg or {})
        terms = _term_list(self._cfg)
        if not terms and "terms" in self._cfg:
            terms = _term_list(self._cfg.get("terms"))

        self.terms: list[ActionTerm] = []
        for term_config in terms:
            if term_config.enabled:
                self.terms.append(ActionTerm.from_config(term_config))
        if not self.terms:
            self.terms = [
                ActionTerm(
                    name=f"{name}/target_position_normalized",
                    joint=name,
                    mode="normalized_position",
                    scale=1.0,
                    lower=-1.0,
                    upper=1.0,
                    unit="normalized",
                )
                for name in self.joint_names
            ]

        self.joint_names = tuple(term.joint for term in self.terms)
        self.spec = VectorSpec(
            names=tuple(term.name for term in self.terms),
            lower_bounds=tuple(term.lower for term in self.terms),
            upper_bounds=tuple(term.upper for term in self.terms),
            units=tuple(term.unit for term in self.terms),
            version=str(self._cfg.get("version", "gobot.action.manager.v1")),
        )
        self.last_action = np.zeros((env.num_envs, self.spec.size), dtype=np.float64)

    def process_action(self, action: ArrayLike) -> np.ndarray:
        expected_shape = (self.env.num_envs, self.spec.size)
        array = _as_float_array(action, shape=expected_shape, name="action")
        lower = np.asarray(self.spec.lower_bounds, dtype=np.float64).reshape(1, -1)
        upper = np.asarray(self.spec.upper_bounds, dtype=np.float64).reshape(1, -1)
        clipped = np.clip(array, lower, upper)
        self.last_action = clipped.copy()
        return clipped

    def apply_action(self, processed_action: np.ndarray) -> None:
        for index, term in enumerate(self.terms):
            term.apply(self.env, float(processed_action[0, index]))


@dataclass
class ObservationTerm:
    name: str
    type: str
    params: Mapping[str, Any]
    lower: float = -math.inf
    upper: float = math.inf
    unit: str = ""

    @classmethod
    def from_mapping(cls, value: str | Mapping[str, Any], *, default_name: str = "") -> "ObservationTerm":
        if isinstance(value, str):
            return cls(name=default_name or value, type=value, params={})
        data = dict(value)
        params = dict(data)
        name = str(params.pop("name", default_name or params.get("type", "")))
        term_type = str(params.pop("type", name))
        lower = float(params.pop("lower", params.pop("lower_bound", -math.inf)))
        upper = float(params.pop("upper", params.pop("upper_bound", math.inf)))
        unit = str(params.pop("unit", params.pop("units", "")))
        return cls(name=name, type=term_type, params=params, lower=lower, upper=upper, unit=unit)

    def compute(self, env: "ManagerBasedEnv", cache: RuntimeCache) -> float:
        term_type = self.type
        joint = str(self.params.get("joint", ""))
        wrap = bool(self.params.get("wrap", False))
        if term_type == "time":
            return float(cache.time[0])
        if term_type == "episode_progress":
            return min(float(env.episode_lengths[0]) / max(env.max_episode_steps, 1), 1.0)
        if term_type == "joint_position":
            return cache.joint_position(joint, wrap=wrap)
        if term_type == "joint_velocity":
            return cache.joint_velocity(joint)
        if term_type == "joint_effort":
            return cache.joint_effort(joint)
        if term_type == "command":
            return env.command_manager.value(str(self.params.get("command", self.name)))[0]
        if term_type == "command_error":
            command_name = str(self.params.get("command", self.name))
            return env.command_manager.value(command_name)[0] - cache.joint_position(joint, wrap=wrap)
        if term_type == "previous_action":
            action_name = str(self.params.get("action", self.name))
            return env.action_manager.last_action[0, env.action_manager.spec.names.index(action_name)]
        if term_type == "constant":
            return float(self.params.get("value", 0.0))
        raise ValueError(f"unsupported observation term type: {term_type!r}")


class ObservationManager:
    """Builds named observation groups from engine-facing runtime state."""

    def __init__(
        self,
        env: "ManagerBasedEnv",
        name_map: Mapping[str, Any],
        robot_name: str,
        observation_cfg: Mapping[str, Any] | None = None,
    ) -> None:
        self.env = env
        self.robot_name = robot_name
        self._cfg = dict(observation_cfg or {})
        robot = _find_robot(name_map, robot_name)
        self._cache = RuntimeCache(env)
        self.joint_names = tuple(str(joint["name"]) for joint in robot.get("joints", []))

        groups_cfg = self._cfg.get("groups")
        if not groups_cfg:
            groups_cfg = {"policy": self._default_terms(robot)}
        self.group_terms = self._build_groups(groups_cfg)
        self.default_group = str(self._cfg.get("default_group", "policy"))
        if self.default_group not in self.group_terms:
            self.default_group = next(iter(self.group_terms))

        self.group_specs = {
            name: self._spec_for_group(name, terms)
            for name, terms in self.group_terms.items()
        }
        self.spec = self.group_specs[self.default_group]

    def _default_terms(self, robot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        terms: list[Mapping[str, Any]] = [
            {"name": "time", "type": "time", "lower": 0.0, "upper": math.inf, "unit": "s"},
            {"name": "episode_progress", "type": "episode_progress", "lower": 0.0, "upper": 1.0, "unit": "ratio"},
        ]
        for joint in robot.get("joints", []):
            joint_name = str(joint["name"])
            lower_limit, upper_limit = _joint_limits(joint)
            terms.append({"name": f"{joint_name}/position", "type": "joint_position", "joint": joint_name, "lower": lower_limit, "upper": upper_limit, "unit": "rad_or_m"})
            terms.append({"name": f"{joint_name}/velocity", "type": "joint_velocity", "joint": joint_name, "lower": -math.inf, "upper": math.inf, "unit": "rad_or_m/s"})
        for action_name in self.env.action_manager.spec.names:
            terms.append({"name": f"{action_name}/previous_action", "type": "previous_action", "action": action_name, "lower": -1.0, "upper": 1.0, "unit": "normalized"})
        return terms

    def _build_groups(self, groups_cfg: Mapping[str, Any]) -> dict[str, tuple[ObservationTerm, ...]]:
        raw_groups: dict[str, list[Any]] = {
            str(name): list(items or [])
            for name, items in groups_cfg.items()
        }
        built: dict[str, tuple[ObservationTerm, ...]] = {}

        def expand(group_name: str, stack: tuple[str, ...] = ()) -> tuple[ObservationTerm, ...]:
            if group_name in built:
                return built[group_name]
            if group_name in stack:
                raise ValueError(f"recursive observation group reference: {' -> '.join(stack + (group_name,))}")
            terms: list[ObservationTerm] = []
            for index, item in enumerate(raw_groups.get(group_name, [])):
                if isinstance(item, Mapping) and item.get("type") == "group":
                    terms.extend(expand(str(item.get("group")), stack + (group_name,)))
                else:
                    terms.append(ObservationTerm.from_mapping(item, default_name=str(index)))
            built[group_name] = tuple(terms)
            return built[group_name]

        for group_name in raw_groups:
            expand(group_name)
        return built

    def _spec_for_group(self, group_name: str, terms: Sequence[ObservationTerm]) -> VectorSpec:
        return VectorSpec(
            names=tuple(term.name for term in terms),
            lower_bounds=tuple(term.lower for term in terms),
            upper_bounds=tuple(term.upper for term in terms),
            units=tuple(term.unit for term in terms),
            version=f"gobot.observation.{group_name}.v1",
        )

    def compute(self, group: str | None = None) -> np.ndarray:
        group_name = group or self.default_group
        terms = self.group_terms[group_name]
        observations = np.zeros((self.env.num_envs, len(terms)), dtype=np.float64)
        self._cache.refresh()
        observations[0, :] = [term.compute(self.env, self._cache) for term in terms]
        return observations

    def compute_groups(self) -> dict[str, np.ndarray]:
        self._cache.refresh()
        result: dict[str, np.ndarray] = {}
        for group_name, terms in self.group_terms.items():
            values = np.zeros((self.env.num_envs, len(terms)), dtype=np.float64)
            values[0, :] = [term.compute(self.env, self._cache) for term in terms]
            result[group_name] = values
        return result


@dataclass
class RewardTerm:
    config: ManagerTermConfig

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def weight(self) -> float:
        return self.config.weight

    @property
    def params(self) -> Mapping[str, Any]:
        return self.config.params

    def raw(self, env: "ManagerBasedEnv", cache: RuntimeCache) -> np.ndarray:
        params = self.params
        term_type = self.config.type
        joint = str(params.get("joint", ""))
        wrap = bool(params.get("wrap", False))

        if term_type == "constant":
            return np.full((env.num_envs,), float(params.get("value", 1.0)), dtype=np.float64)
        if term_type == "joint_position_l2":
            value = cache.joint_position(joint, wrap=wrap)
            target = float(params.get("target", 0.0))
            return np.array([(value - target) ** 2], dtype=np.float64)
        if term_type == "joint_velocity_l2":
            return np.array([cache.joint_velocity(joint) ** 2], dtype=np.float64)
        if term_type == "joint_effort_l2":
            return np.array([cache.joint_effort(joint) ** 2], dtype=np.float64)
        if term_type == "action_l2":
            action_name = str(params.get("action", ""))
            if action_name:
                index = env.action_manager.spec.names.index(action_name)
                value = env.action_manager.last_action[0, index]
                return np.array([value * value], dtype=np.float64)
            return np.sum(env.action_manager.last_action ** 2, axis=1)
        if term_type == "squared_command_error":
            command = env.command_manager.value(str(params.get("command", "")))[0]
            value = cache.joint_position(joint, wrap=wrap)
            return np.array([(command - value) ** 2], dtype=np.float64)
        if term_type == "overshoot_l2":
            command = abs(env.command_manager.value(str(params.get("command", "")))[0])
            value = abs(cache.joint_position(joint, wrap=wrap))
            return np.array([max(0.0, value - command) ** 2], dtype=np.float64)
        if term_type == "target_progress":
            command_name = str(params.get("command", ""))
            command = env.command_manager.value(command_name)[0]
            current_error = command - cache.joint_position(joint, wrap=wrap)
            previous_error = env.metrics.previous_target_errors.get((joint, command_name), current_error)
            progress = abs(previous_error) - abs(current_error)
            clip = params.get("clip")
            if clip is not None and len(clip) == 2:
                progress = float(np.clip(progress, float(clip[0]), float(clip[1])))
            env.metrics.next_target_errors[(joint, command_name)] = current_error
            return np.array([progress], dtype=np.float64)
        if term_type == "cartpole_settle_bonus":
            slider = str(params.get("slider_joint", params.get("joint", "")))
            hinge = str(params.get("hinge_joint", ""))
            command = env.command_manager.value(str(params.get("command", "")))[0]
            distance = abs(command - cache.joint_position(slider))
            ok = (
                distance <= float(params.get("target_tolerance", 0.05))
                and abs(cache.joint_velocity(slider)) <= float(params.get("velocity_tolerance", 0.2))
                and abs(cache.joint_position(hinge, wrap=True)) <= float(params.get("angle_tolerance", 0.1))
            )
            return np.array([1.0 if ok else 0.0], dtype=np.float64)
        if term_type == "first_reach_bonus":
            command_name = str(params.get("command", ""))
            command = env.command_manager.value(command_name)[0]
            distance = abs(command - cache.joint_position(joint, wrap=wrap))
            hinge = str(params.get("hinge_joint", ""))
            key = self.name
            reached = bool(env.metrics.flags.get(key, False))
            ok = (
                not reached
                and distance <= float(params.get("target_tolerance", 0.1))
                and (not hinge or abs(cache.joint_position(hinge, wrap=True)) <= float(params.get("angle_tolerance", math.inf)))
            )
            if ok:
                env.metrics.flags[key] = True
                time_left = 1.0 - min(env.episode_lengths[0] / max(env.max_episode_steps, 1), 1.0)
                return np.array([time_left], dtype=np.float64)
            return np.zeros((env.num_envs,), dtype=np.float64)
        if term_type == "target_overspeed":
            command = env.command_manager.value(str(params.get("command", "")))[0]
            distance = abs(command - cache.joint_position(joint, wrap=wrap))
            oversped = (
                distance <= float(params.get("target_tolerance", 0.1))
                and abs(cache.joint_velocity(joint)) > float(params.get("velocity_limit", math.inf))
            )
            return np.array([1.0 if oversped else 0.0], dtype=np.float64)
        if term_type == "target_fast_crossing":
            command_name = str(params.get("command", ""))
            command = env.command_manager.value(command_name)[0]
            current_error = command - cache.joint_position(joint, wrap=wrap)
            previous_error = env.metrics.previous_target_errors.get((joint, command_name), current_error)
            crossed = previous_error * current_error < 0.0 and abs(cache.joint_velocity(joint)) > float(params.get("velocity_tolerance", 0.2))
            return np.array([1.0 if crossed else 0.0], dtype=np.float64)
        if term_type == "terminated":
            return env.metrics.last_terminated.astype(np.float64)
        raise ValueError(f"unsupported reward term type: {term_type!r}")

    def compute(self, env: "ManagerBasedEnv", cache: RuntimeCache) -> np.ndarray:
        value = self.raw(env, cache) * self.weight
        if bool(self.params.get("scale_by_dt", False)):
            value = value * env.env_dt
        return value


class RewardManager:
    """Computes weighted reward terms and exposes per-term breakdowns."""

    def __init__(
        self,
        env: "ManagerBasedEnv",
        reward_cfg: Mapping[str, Any] | None = None,
        terms: Sequence[Callable[["ManagerBasedEnv"], ArrayLike]] | None = None,
    ) -> None:
        self.env = env
        self.callable_terms = list(terms or [])
        self.terms = [
            RewardTerm(config)
            for config in _term_list(reward_cfg)
            if config.enabled
        ]
        self.last_breakdown: dict[str, np.ndarray] = {}

    def compute(self) -> np.ndarray:
        cache = self.env.observation_manager._cache
        cache.refresh()
        reward = np.zeros((self.env.num_envs,), dtype=np.float64)
        breakdown: dict[str, np.ndarray] = {}

        if not self.terms and not self.callable_terms:
            breakdown["alive"] = np.full((self.env.num_envs,), self.env.env_dt, dtype=np.float64)

        for term in self.terms:
            value = term.compute(self.env, cache)
            breakdown[term.name] = value.copy()
            reward += value
        for index, term in enumerate(self.callable_terms):
            value = _as_float_array(term(self.env), shape=(self.env.num_envs,), name="reward term")
            breakdown[f"callable_{index}"] = value.copy()
            reward += value

        if not self.terms and not self.callable_terms:
            reward += breakdown["alive"]

        self.last_breakdown = breakdown
        self.env.metrics.previous_target_errors = dict(self.env.metrics.next_target_errors)
        self.env.metrics.next_target_errors = {}
        return reward


@dataclass
class TerminationTerm:
    config: ManagerTermConfig

    def compute(self, env: "ManagerBasedEnv", cache: RuntimeCache) -> np.ndarray:
        params = self.config.params
        term_type = self.config.type
        joint = str(params.get("joint", ""))
        wrap = bool(params.get("wrap", False))
        if term_type == "joint_position_abs_gt":
            value = abs(cache.joint_position(joint, wrap=wrap))
            return np.array([value > float(params.get("limit", math.inf))], dtype=bool)
        if term_type == "joint_position_gt":
            return np.array([cache.joint_position(joint, wrap=wrap) > float(params.get("limit", math.inf))], dtype=bool)
        if term_type == "joint_position_lt":
            return np.array([cache.joint_position(joint, wrap=wrap) < float(params.get("limit", -math.inf))], dtype=bool)
        if term_type == "non_finite_observation":
            observation = env.observation_manager.compute()
            return ~np.all(np.isfinite(observation), axis=1)
        raise ValueError(f"unsupported termination term type: {term_type!r}")


class TerminationManager:
    def __init__(
        self,
        env: "ManagerBasedEnv",
        termination_cfg: Mapping[str, Any] | None = None,
        conditions: Sequence[Callable[["ManagerBasedEnv"], ArrayLike]] | None = None,
    ) -> None:
        self.env = env
        self.conditions = list(conditions or [])
        self.terms = [
            TerminationTerm(config)
            for config in _term_list(termination_cfg)
            if config.enabled
        ]
        self.last_reasons: dict[str, np.ndarray] = {}

    def compute(self) -> tuple[np.ndarray, np.ndarray]:
        cache = self.env.observation_manager._cache
        cache.refresh()
        terminated = np.zeros((self.env.num_envs,), dtype=bool)
        reasons: dict[str, np.ndarray] = {}
        for term in self.terms:
            value = term.compute(self.env, cache)
            reasons[term.config.name] = value.copy()
            terminated |= value
        for index, condition in enumerate(self.conditions):
            value = _as_bool_array(condition(self.env), shape=(self.env.num_envs,), name="termination condition")
            reasons[f"callable_{index}"] = value.copy()
            terminated |= value
        truncated = self.env.episode_lengths >= self.env.max_episode_steps
        reasons["time_limit"] = truncated.copy()
        self.last_reasons = reasons
        self.env.metrics.last_terminated = terminated.copy()
        self.env.metrics.last_truncated = truncated.copy()
        return terminated, truncated


class EventManager:
    def __init__(self, env: "ManagerBasedEnv", event_cfg: Mapping[str, Any] | None = None) -> None:
        self.env = env
        self.terms = [
            config
            for config in _term_list(event_cfg)
            if config.enabled
        ]
        self._effort_disturbances = {
            config.name: {
                "joint": str(config.params.get("joint", "")),
                "std": float(config.params.get("std", 0.0)),
                "impulse_probability": float(config.params.get("impulse_probability", 0.0)),
                "impulse_force": float(config.params.get("impulse_force", 0.0)),
                "impulse_steps": max(1, int(config.params.get("impulse_steps", 1))),
                "remaining": 0,
                "value": 0.0,
            }
            for config in self.terms
            if config.type == "joint_effort_disturbance"
        }

    def reset(self, env_ids: np.ndarray) -> None:
        requires_rebuild = False
        for config in self.terms:
            if config.type == "reset_joint_state":
                joints = config.params.get("joints", {})
                if isinstance(joints, Mapping):
                    for joint_name, values in joints.items():
                        joint_values = dict(values or {})
                        position = joint_values.get("position", 0.0)
                        position_range = joint_values.get("position_range")
                        if position_range is not None:
                            lower, upper = list(position_range)
                            position = self.env._rng.uniform(float(lower), float(upper))
                        self.env._backend.set_authored_joint_position(str(joint_name), float(position))
                        requires_rebuild = True
            elif config.type == "callback":
                callback = config.params.get("callback")
                if callback is not None:
                    callback(self.env, env_ids)
        if requires_rebuild:
            self.env._backend.rebuild_world()
            for config in self.terms:
                if config.type != "reset_joint_state":
                    continue
                joints = config.params.get("joints", {})
                if isinstance(joints, Mapping):
                    for joint_name, values in joints.items():
                        joint_values = dict(values or {})
                        self.env._backend.set_joint_velocity_target(0, str(joint_name), float(joint_values.get("velocity", 0.0)))
        for disturbance in self._effort_disturbances.values():
            disturbance["remaining"] = 0
            disturbance["value"] = 0.0
        self.env.metrics.flags.clear()
        self.env.metrics.previous_target_errors.clear()
        self.env.metrics.next_target_errors.clear()

    def sample_effort_disturbance(self, joint: str) -> float:
        effort = 0.0
        for disturbance in self._effort_disturbances.values():
            if disturbance["joint"] != joint:
                continue
            std = float(disturbance["std"])
            if std > 0.0:
                effort += self.env._rng.gauss(0.0, std)

            remaining = int(disturbance["remaining"])
            if remaining > 0:
                disturbance["remaining"] = remaining - 1
                effort += float(disturbance["value"])
                continue

            probability = float(disturbance["impulse_probability"])
            impulse_force = float(disturbance["impulse_force"])
            if probability > 0.0 and impulse_force > 0.0 and self.env._rng.random() < probability:
                disturbance["remaining"] = int(disturbance["impulse_steps"]) - 1
                disturbance["value"] = self.env._rng.choice((-1.0, 1.0)) * impulse_force
                effort += float(disturbance["value"])
        return effort


class CommandManager:
    def __init__(self, env: "ManagerBasedEnv", command_cfg: Mapping[str, Any] | None = None) -> None:
        self.env = env
        self.terms = [
            config
            for config in _term_list(command_cfg)
            if config.enabled
        ]
        self.values: dict[str, np.ndarray] = {
            config.name: np.zeros((env.num_envs,), dtype=np.float64)
            for config in self.terms
        }

    def reset(self, env_ids: np.ndarray) -> None:
        del env_ids
        for config in self.terms:
            params = config.params
            if config.type == "constant":
                value = float(params.get("value", 0.0))
            elif config.type == "uniform":
                lower, upper = params.get("range", (-1.0, 1.0))
                value = self.env._rng.uniform(float(lower), float(upper))
            elif config.type == "choice":
                choices = list(params.get("choices", (0.0,)))
                value = float(self.env._rng.choice(choices))
            else:
                raise ValueError(f"unsupported command term type: {config.type!r}")
            self.values.setdefault(config.name, np.zeros((self.env.num_envs,), dtype=np.float64))
            self.values[config.name][:] = value

    def value(self, name: str) -> np.ndarray:
        if name in self.values:
            return self.values[name]
        return np.zeros((self.env.num_envs,), dtype=np.float64)

    def to_info(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self.values.items()}


class Recorder:
    def __init__(self) -> None:
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []

    def record_done(self, returns: np.ndarray, lengths: np.ndarray, done: np.ndarray) -> None:
        for index in np.flatnonzero(done):
            self.episode_returns.append(float(returns[index]))
            self.episode_lengths.append(int(lengths[index]))


class Metrics(dict):
    def __init__(self) -> None:
        super().__init__()
        self.flags: dict[str, bool] = {}
        self.previous_target_errors: dict[tuple[str, str], float] = {}
        self.next_target_errors: dict[tuple[str, str], float] = {}
        self.last_terminated = np.zeros((1,), dtype=bool)
        self.last_truncated = np.zeros((1,), dtype=bool)


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

    def rebuild_world(self) -> None:
        self.context.rebuild_world(False)

    def set_authored_joint_position(self, joint_name: str, position: float) -> None:
        root = self.context.root
        if root is None:
            raise RuntimeError("active Gobot context has no scene root")
        robot = root if root.name == self._robot_name() else root.find(self._robot_name())
        if robot is None:
            raise RuntimeError(f"scene has no robot node '{self._robot_name()}'")
        joint = robot.find(joint_name)
        if joint is None:
            joint = self._find_node_by_name(robot, joint_name)
        if joint is None:
            raise RuntimeError(f"robot '{self._robot_name()}' has no joint '{joint_name}'")
        joint.set("joint_position", float(position))

    def _find_node_by_name(self, node: Any, name: str) -> Any | None:
        if node.name == name:
            return node
        for child in node.children:
            found = self._find_node_by_name(child, name)
            if found is not None:
                return found
        return None

    def _robot_name(self) -> str:
        robot_name = self.cfg.robot
        if robot_name is None:
            robot_name = _find_robot(self.context.get_runtime_name_map()).get("name")
        return str(robot_name)

    def set_normalized_joint_position_action(
        self,
        env_id: int,
        joint_names: Sequence[str],
        action: Sequence[float],
    ) -> None:
        if env_id != 0:
            raise IndexError("single Gobot context backend only supports env_id 0")
        self.context.set_robot_named_action(self._robot_name(), list(joint_names), list(float(item) for item in action))

    def set_joint_position_target(self, env_id: int, joint_name: str, target_position: float) -> None:
        if env_id != 0:
            raise IndexError("single Gobot context backend only supports env_id 0")
        self.context.set_joint_position_target(self._robot_name(), joint_name, float(target_position))

    def set_joint_velocity_target(self, env_id: int, joint_name: str, target_velocity: float) -> None:
        if env_id != 0:
            raise IndexError("single Gobot context backend only supports env_id 0")
        self.context.set_joint_velocity_target(self._robot_name(), joint_name, float(target_velocity))

    def set_joint_effort_target(self, env_id: int, joint_name: str, target_effort: float) -> None:
        if env_id != 0:
            raise IndexError("single Gobot context backend only supports env_id 0")
        self.context.set_joint_effort_target(self._robot_name(), joint_name, float(target_effort))

    def set_joint_passive(self, env_id: int, joint_name: str) -> None:
        if env_id != 0:
            raise IndexError("single Gobot context backend only supports env_id 0")
        self.context.set_joint_passive(self._robot_name(), joint_name)

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

    def __init__(self, cfg: Mapping[str, Any] | EnvConfig | TaskConfig | None = None, *, context: _core.AppContext | None = None) -> None:
        if isinstance(cfg, TaskConfig):
            self.task_config = cfg
            self.cfg = cfg.env_config()
        elif isinstance(cfg, EnvConfig):
            self.task_config = None
            self.cfg = cfg
        elif isinstance(cfg, Mapping) and ("name" in cfg or "commands" in cfg or "metadata" in cfg):
            self.task_config = TaskConfig.from_mapping(cfg)
            self.cfg = self.task_config.env_config()
        else:
            self.task_config = None
            self.cfg = EnvConfig.from_mapping(cfg)
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

        self.metrics = Metrics()
        self.metrics.last_terminated = np.zeros((self.num_envs,), dtype=bool)
        self.metrics.last_truncated = np.zeros((self.num_envs,), dtype=bool)
        self.action_manager = ActionManager(self, controlled_joints, self.cfg.actions)
        self.observation_manager = ObservationManager(self, name_map, self.robot_name, self.cfg.observations)
        self.reward_manager = RewardManager(self, self.cfg.rewards)
        self.termination_manager = TerminationManager(self, self.cfg.terminations)
        self.event_manager = EventManager(self, self.cfg.events)
        self.command_manager = CommandManager(self, self.cfg.commands)
        self.recorder = Recorder()

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

    def observation_group_spec(self, group: str) -> VectorSpec:
        return self.observation_manager.group_specs[group]

    def reset(
        self,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        options = dict(options or {})
        if seed is not None:
            self._rng.seed(int(seed))
        env_ids = np.arange(self.num_envs, dtype=np.int64)
        self._backend.reset(env_ids, seed=seed)
        self.episode_lengths[:] = 0
        self.episode_returns[:] = 0.0
        self.action_manager.last_action[:] = 0.0
        self.metrics.flags.clear()
        self.metrics.previous_target_errors.clear()
        self.metrics.next_target_errors.clear()
        self.metrics.last_terminated[:] = False
        self.metrics.last_truncated[:] = False
        self.command_manager.reset(env_ids)
        for name, value in options.items():
            if name in self.command_manager.values:
                self.command_manager.values[name][:] = float(value)
        self.event_manager.reset(env_ids)
        observation = self.observation_manager.compute()
        self._last_observation = observation.copy()
        info = {
            "seed": seed,
            "env_ids": env_ids.copy(),
            "runtime_name_map": self._backend.get_runtime_name_map(),
            "commands": self.command_manager.to_info(),
            "observations": self.observation_manager.compute_groups(),
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

        observation_groups = self.observation_manager.compute_groups()
        info: dict[str, Any] = {
            "episode_length": self.episode_lengths.copy(),
            "episode_return": self.episode_returns.copy(),
            "simulation_time": np.array([self._backend.simulation_time(0)], dtype=np.float64),
            "commands": self.command_manager.to_info(),
            "observations": observation_groups,
            "reward_breakdown": {name: value.copy() for name, value in self.reward_manager.last_breakdown.items()},
            "termination_reasons": {name: value.copy() for name, value in self.termination_manager.last_reasons.items()},
        }
        if np.any(done):
            terminal_observation = observation_groups[self.observation_manager.default_group]
            info["terminal_observation"] = terminal_observation.copy()
            self.recorder.record_done(self.episode_returns, self.episode_lengths, done)
            if self.auto_reset:
                done_env_ids = np.flatnonzero(done).astype(np.int64)
                self._backend.reset(done_env_ids)
                self.episode_lengths[done_env_ids] = 0
                self.episode_returns[done_env_ids] = 0.0
                self.action_manager.last_action[done_env_ids] = 0.0
                self.command_manager.reset(done_env_ids)
                self.event_manager.reset(done_env_ids)

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
    "ManagerTermConfig",
    "Metrics",
    "ObservationManager",
    "Recorder",
    "RewardManager",
    "RslRlVecEnvWrapper",
    "TaskConfig",
    "TerminationManager",
    "VectorEnv",
    "VectorSpace",
    "VectorSpec",
    "load_task_config",
    "make_cartpole_target_task",
    "space_from_spec",
]
