"""Python facade over Gobot's existing batch simulation APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


class BatchSimulationRuntime:
    """Small facade used by CPU batch envs.

    This keeps task code from depending directly on the lower-level AppContext
    batch method names while the C++ backend contract continues to settle.
    """

    def __init__(
        self,
        context: Any,
        *,
        robot: str,
        base_link: str,
        joint_names: Sequence[str],
        link_names: Sequence[str] = (),
        sensor_names: Sequence[str] = (),
    ) -> None:
        self.context = context
        self.robot = str(robot)
        self.base_link = str(base_link)
        self.joint_names = tuple(str(name) for name in joint_names)
        self.link_names = tuple(str(name) for name in link_names)
        self.sensor_names = tuple(str(name) for name in sensor_names)

    @property
    def env_count(self) -> int:
        return int(getattr(self.context, "batch_env_count", 0))

    def configure(self, num_envs: int) -> None:
        self.context.configure_batch_world(int(num_envs))

    def resolved_workers(self, workers: int = 0) -> int:
        return int(self.context.resolved_batch_workers(int(workers)))

    def step(self, ticks: int, *, workers: int = 0) -> None:
        self.context.step_batch(int(ticks), workers=int(workers))

    def reset_env(self, env_id: int) -> None:
        self.context.reset_batch_env(int(env_id))

    def set_joint_position_target(self, env_id: int, joint: str, target_position: float) -> None:
        self.context.set_batch_joint_position_target(
            int(env_id),
            self.robot,
            str(joint),
            float(target_position),
        )

    def set_joint_position_targets(self, target_positions: Any) -> None:
        targets = np.asarray(target_positions, dtype=np.float64)
        self.context.set_batch_joint_position_targets(
            self.robot,
            list(self.joint_names),
            targets,
        )

    def reset_joint_state(self, env_id: int, joint: str, position: float, velocity: float = 0.0) -> None:
        self.context.reset_batch_joint_state(
            int(env_id),
            self.robot,
            str(joint),
            float(position),
            float(velocity),
        )

    def reset_link_state(
        self,
        env_id: int,
        link: str,
        position: Sequence[float],
        orientation: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        linear_velocity: Sequence[float] = (0.0, 0.0, 0.0),
        angular_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.context.reset_batch_link_state(
            int(env_id),
            self.robot,
            str(link),
            list(position),
            list(orientation),
            list(linear_velocity),
            list(angular_velocity),
        )

    def env_state(self, env_id: int) -> Any:
        return self.context.get_batch_runtime_state(int(env_id))

    def robot_state(self) -> Any:
        return self.context.get_batch_robot_state(
            self.robot,
            self.base_link,
            list(self.joint_names),
            list(self.link_names),
            list(self.sensor_names),
        )

    def step_and_robot_state(self, target_positions: Any, ticks: int, *, workers: int = 0) -> Any:
        targets = np.asarray(target_positions, dtype=np.float64)
        return self.context._step_batch_and_get_robot_state(
            self.robot,
            self.base_link,
            list(self.joint_names),
            list(self.link_names),
            list(self.sensor_names),
            targets,
            int(ticks),
            int(workers),
        )

    def reset_robot_states(
        self,
        env_ids: Sequence[int],
        *,
        base_positions: Any,
        base_orientations: Any,
        base_linear_velocities: Any,
        base_angular_velocities: Any,
        joint_positions: Any,
        joint_velocities: Any,
        joint_position_targets: Any,
    ) -> None:
        self.context._reset_batch_robot_states(
            [int(env_id) for env_id in env_ids],
            self.robot,
            self.base_link,
            np.asarray(base_positions, dtype=np.float64),
            np.asarray(base_orientations, dtype=np.float64),
            np.asarray(base_linear_velocities, dtype=np.float64),
            np.asarray(base_angular_velocities, dtype=np.float64),
            list(self.joint_names),
            np.asarray(joint_positions, dtype=np.float64),
            np.asarray(joint_velocities, dtype=np.float64),
            np.asarray(joint_position_targets, dtype=np.float64),
        )


@dataclass
class GobotSceneBatchState:
    """Cached batch state arrays returned by ``GobotSceneBatchBackend``."""

    raw: dict[str, Any]
    base_position: np.ndarray
    base_quaternion: np.ndarray
    base_linear_velocity: np.ndarray
    base_angular_velocity: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    joint_lower_limit: np.ndarray
    joint_upper_limit: np.ndarray
    link_names: tuple[str, ...]
    link_position: np.ndarray
    sensor_names: tuple[str, ...]
    sensor_position: np.ndarray
    sensor_values: np.ndarray
    sensor_value_count: np.ndarray
    sensor_hit: np.ndarray
    sensor_hit_point: np.ndarray
    sensor_hit_normal: np.ndarray
    sensor_hit_distance: np.ndarray
    contact_count: np.ndarray
    contact_link_index: np.ndarray
    contact_position: np.ndarray
    contact_force: np.ndarray
    contact_normal_force: np.ndarray


@dataclass
class GobotGo1FastBatchState:
    """Persistent NumPy views owned by Gobot's Go1 MuJoCo fast batch view."""

    base_position: np.ndarray
    base_quaternion: np.ndarray
    base_linear_velocity: np.ndarray
    base_angular_velocity: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    joint_lower_limit: np.ndarray
    joint_upper_limit: np.ndarray
    foot_position: np.ndarray
    foot_velocity: np.ndarray
    foot_height: np.ndarray
    foot_contact: np.ndarray
    foot_contact_force: np.ndarray
    height_scan: np.ndarray
    height_scan_hit: np.ndarray
    height_scan_point: np.ndarray
    height_scan_normal: np.ndarray
    illegal_contact_count: np.ndarray
    self_collision_count: np.ndarray
    shank_collision_count: np.ndarray
    trunk_head_collision_count: np.ndarray


class GobotSceneBatchBackend:
    """UniLab-style Python facade over Gobot's scene-authored batch runtime.

    The source of truth stays the loaded Gobot scene. This class only narrows
    the hot training path to batched control arrays, a single physics step call,
    and cached numpy state arrays.
    """

    def __init__(self, runtime: BatchSimulationRuntime) -> None:
        self.runtime = runtime
        self._state: GobotSceneBatchState | None = None

    @property
    def env_count(self) -> int:
        return self.runtime.env_count

    @property
    def state(self) -> GobotSceneBatchState:
        if self._state is None:
            return self.refresh()
        return self._state

    def configure(self, num_envs: int) -> None:
        self.runtime.configure(int(num_envs))
        self._state = None

    def resolved_workers(self, workers: int = 0) -> int:
        return self.runtime.resolved_workers(workers)

    def step(self, ctrl: Any, nsteps: int, *, workers: int = 0) -> dict[str, Any]:
        if hasattr(self.runtime.context, "_step_batch_and_get_robot_state"):
            raw = dict(self.runtime.step_and_robot_state(ctrl, int(nsteps), workers=int(workers)))
            self._state = self._state_from_raw(raw)
        else:
            self.set_position_targets(ctrl)
            self.runtime.step(int(nsteps), workers=int(workers))
            self._state = None
        return {}

    def set_position_targets(self, ctrl: Any) -> None:
        targets = np.asarray(ctrl, dtype=np.float64)
        self.runtime.set_joint_position_targets(targets)

    def set_joint_position_target(self, env_id: int, joint: str, target_position: float) -> None:
        self.runtime.set_joint_position_target(env_id, joint, target_position)
        self._state = None

    def reset_env(self, env_id: int) -> None:
        self.runtime.reset_env(int(env_id))
        self._state = None

    def reset_joint_state(self, env_id: int, joint: str, position: float, velocity: float = 0.0) -> None:
        self.runtime.reset_joint_state(env_id, joint, position, velocity)
        self._state = None

    def reset_link_state(
        self,
        env_id: int,
        link: str,
        position: Sequence[float],
        orientation: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        linear_velocity: Sequence[float] = (0.0, 0.0, 0.0),
        angular_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.runtime.reset_link_state(env_id, link, position, orientation, linear_velocity, angular_velocity)
        self._state = None

    def reset_robot_states(
        self,
        env_ids: Sequence[int],
        *,
        base_positions: Any,
        base_orientations: Any,
        base_linear_velocities: Any,
        base_angular_velocities: Any,
        joint_positions: Any,
        joint_velocities: Any,
        joint_position_targets: Any,
    ) -> None:
        if not hasattr(self.runtime.context, "_reset_batch_robot_states"):
            for row, env_id in enumerate(env_ids):
                self.reset_env(int(env_id))
                self.reset_link_state(
                    int(env_id),
                    self.runtime.base_link,
                    np.asarray(base_positions, dtype=np.float64)[row].tolist(),
                    np.asarray(base_orientations, dtype=np.float64)[row].tolist(),
                    np.asarray(base_linear_velocities, dtype=np.float64)[row].tolist(),
                    np.asarray(base_angular_velocities, dtype=np.float64)[row].tolist(),
                )
                for joint, position, velocity, target in zip(
                    self.runtime.joint_names,
                    np.asarray(joint_positions, dtype=np.float64)[row],
                    np.asarray(joint_velocities, dtype=np.float64)[row],
                    np.asarray(joint_position_targets, dtype=np.float64)[row],
                    strict=True,
                ):
                    self.reset_joint_state(int(env_id), joint, float(position), float(velocity))
                    self.set_joint_position_target(int(env_id), joint, float(target))
            return
        self.runtime.reset_robot_states(
            env_ids,
            base_positions=base_positions,
            base_orientations=base_orientations,
            base_linear_velocities=base_linear_velocities,
            base_angular_velocities=base_angular_velocities,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_position_targets=joint_position_targets,
        )
        self._state = None

    def env_state(self, env_id: int) -> Any:
        return self.runtime.env_state(env_id)

    def refresh(self) -> GobotSceneBatchState:
        raw = dict(self.runtime.robot_state())
        self._state = self._state_from_raw(raw)
        return self._state

    def _state_from_raw(self, raw: dict[str, Any]) -> GobotSceneBatchState:
        self._state = GobotSceneBatchState(
            raw=raw,
            base_position=np.asarray(raw["base_position"], dtype=np.float32),
            base_quaternion=np.asarray(raw["base_quaternion"], dtype=np.float32),
            base_linear_velocity=np.asarray(raw["base_linear_velocity"], dtype=np.float32),
            base_angular_velocity=np.asarray(raw["base_angular_velocity"], dtype=np.float32),
            joint_position=np.asarray(raw["joint_position"], dtype=np.float32),
            joint_velocity=np.asarray(raw["joint_velocity"], dtype=np.float32),
            joint_lower_limit=np.asarray(raw["joint_lower_limit"], dtype=np.float32),
            joint_upper_limit=np.asarray(raw["joint_upper_limit"], dtype=np.float32),
            link_names=tuple(str(name) for name in raw.get("link_names", self.runtime.link_names)),
            link_position=np.asarray(raw["link_position"], dtype=np.float32),
            sensor_names=tuple(str(name) for name in raw.get("sensor_names", self.runtime.sensor_names)),
            sensor_position=np.asarray(raw["sensor_position"], dtype=np.float32),
            sensor_values=np.asarray(raw["sensor_values"], dtype=np.float32),
            sensor_value_count=np.asarray(raw["sensor_value_count"], dtype=np.int64),
            sensor_hit=np.asarray(raw["sensor_hit"], dtype=bool),
            sensor_hit_point=np.asarray(raw["sensor_hit_point"], dtype=np.float32),
            sensor_hit_normal=np.asarray(raw["sensor_hit_normal"], dtype=np.float32),
            sensor_hit_distance=np.asarray(raw["sensor_hit_distance"], dtype=np.float32),
            contact_count=np.asarray(raw["contact_count"], dtype=np.int64),
            contact_link_index=np.asarray(raw["contact_link_index"], dtype=np.int64),
            contact_position=np.asarray(raw["contact_position"], dtype=np.float32),
            contact_force=np.asarray(raw["contact_force"], dtype=np.float32),
            contact_normal_force=np.asarray(raw["contact_normal_force"], dtype=np.float32),
        )
        return self._state

    def get_base_pos(self) -> np.ndarray:
        return self.state.base_position

    def get_base_quat(self) -> np.ndarray:
        return self.state.base_quaternion

    def get_base_lin_vel(self) -> np.ndarray:
        return self.state.base_linear_velocity

    def get_base_ang_vel(self) -> np.ndarray:
        return self.state.base_angular_velocity

    def get_dof_pos(self) -> np.ndarray:
        return self.state.joint_position

    def get_dof_vel(self) -> np.ndarray:
        return self.state.joint_velocity

    def get_link_pos(self) -> np.ndarray:
        return self.state.link_position

    def get_sensor_data(self, name: str) -> np.ndarray:
        sensor_names = self.state.sensor_names
        try:
            index = sensor_names.index(str(name))
        except ValueError as error:
            raise KeyError(f"Gobot batch state has no sensor {name!r}") from error
        return self.state.sensor_values[:, index, :]


class GobotGo1FastBatchBackend:
    """Go1-specific MuJoCo batch backend with persistent C++-owned buffers."""

    is_fast = True

    def __init__(
        self,
        runtime: BatchSimulationRuntime,
        *,
        foot_link_names: Sequence[str],
        foot_height_sensor_names: Sequence[str],
        foot_contact_sensor_names: Sequence[str],
        height_scan_sensor: str | None = None,
        thigh_link_patterns: Sequence[str] = (),
        shank_link_patterns: Sequence[str] = (),
        trunk_head_link_patterns: Sequence[str] = (),
        terminate_on_thigh_contact: bool = True,
        ground_force_threshold: float = 50.0,
        self_collision_force_threshold: float = 20.0,
    ) -> None:
        self.runtime = runtime
        self.foot_link_names = tuple(str(name) for name in foot_link_names)
        self.foot_height_sensor_names = tuple(str(name) for name in foot_height_sensor_names)
        self.foot_contact_sensor_names = tuple(str(name) for name in foot_contact_sensor_names)
        self.height_scan_sensor = "" if height_scan_sensor is None else str(height_scan_sensor)
        self.thigh_link_patterns = tuple(str(pattern) for pattern in thigh_link_patterns)
        self.shank_link_patterns = tuple(str(pattern) for pattern in shank_link_patterns)
        self.trunk_head_link_patterns = tuple(str(pattern) for pattern in trunk_head_link_patterns)
        self.terminate_on_thigh_contact = bool(terminate_on_thigh_contact)
        self.ground_force_threshold = float(ground_force_threshold)
        self.self_collision_force_threshold = float(self_collision_force_threshold)
        self._view: Any | None = None
        self._arrays: dict[str, np.ndarray] = {}
        self._state: GobotGo1FastBatchState | None = None

    @property
    def env_count(self) -> int:
        return self.runtime.env_count

    @property
    def state(self) -> GobotGo1FastBatchState:
        if self._state is None:
            return self.refresh()
        return self._state

    def configure(self, num_envs: int) -> None:
        if not hasattr(self.runtime.context, "create_go1_locomotion_batch_view"):
            raise RuntimeError("Gobot Python binding has no Go1 locomotion fast batch view")
        self.runtime.configure(int(num_envs))
        self._view = self.runtime.context.create_go1_locomotion_batch_view(
            self.runtime.robot,
            self.runtime.base_link,
            list(self.runtime.joint_names),
            list(self.foot_link_names),
            list(self.foot_height_sensor_names),
            list(self.foot_contact_sensor_names),
            self.height_scan_sensor,
            list(self.thigh_link_patterns),
            list(self.shank_link_patterns),
            list(self.trunk_head_link_patterns),
            self.terminate_on_thigh_contact,
            self.ground_force_threshold,
            self.self_collision_force_threshold,
        )
        self._arrays = {str(name): np.asarray(value) for name, value in dict(self._view.arrays()).items()}
        self._state = self._state_from_arrays()

    def resolved_workers(self, workers: int = 0) -> int:
        return self.runtime.resolved_workers(workers)

    def step(self, ctrl: Any, nsteps: int, *, workers: int = 0) -> dict[str, Any]:
        self._require_view()
        np.copyto(self._arrays["target_position"], np.asarray(ctrl, dtype=np.float32))
        self._view.step(int(nsteps), int(workers))
        return {}

    def set_position_targets(self, ctrl: Any) -> None:
        self._require_view()
        np.copyto(self._arrays["target_position"], np.asarray(ctrl, dtype=np.float32))

    def set_joint_position_target(self, env_id: int, joint: str, target_position: float) -> None:
        self._require_view()
        try:
            joint_index = self.runtime.joint_names.index(str(joint))
        except ValueError as error:
            raise KeyError(f"Gobot Go1 fast backend has no joint {joint!r}") from error
        self._arrays["target_position"][int(env_id), joint_index] = float(target_position)

    def reset_env(self, env_id: int) -> None:
        self.runtime.reset_env(int(env_id))
        if self._view is not None:
            self._view.refresh()

    def reset_joint_state(self, env_id: int, joint: str, position: float, velocity: float = 0.0) -> None:
        self.runtime.reset_joint_state(env_id, joint, position, velocity)
        if self._view is not None:
            self._view.refresh()

    def reset_link_state(
        self,
        env_id: int,
        link: str,
        position: Sequence[float],
        orientation: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        linear_velocity: Sequence[float] = (0.0, 0.0, 0.0),
        angular_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.runtime.reset_link_state(env_id, link, position, orientation, linear_velocity, angular_velocity)
        if self._view is not None:
            self._view.refresh()

    def reset_robot_states(
        self,
        env_ids: Sequence[int],
        *,
        base_positions: Any,
        base_orientations: Any,
        base_linear_velocities: Any,
        base_angular_velocities: Any,
        joint_positions: Any,
        joint_velocities: Any,
        joint_position_targets: Any,
    ) -> None:
        self._require_view()
        env_id_array = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if env_id_array.size == 0:
            return
        rows = env_id_array.astype(np.int64, copy=False)
        self._arrays["reset_base_position"][rows] = np.asarray(base_positions, dtype=np.float32)
        self._arrays["reset_base_quaternion"][rows] = np.asarray(base_orientations, dtype=np.float32)
        self._arrays["reset_base_linear_velocity"][rows] = np.asarray(base_linear_velocities, dtype=np.float32)
        self._arrays["reset_base_angular_velocity"][rows] = np.asarray(base_angular_velocities, dtype=np.float32)
        self._arrays["reset_joint_position"][rows] = np.asarray(joint_positions, dtype=np.float32)
        self._arrays["reset_joint_velocity"][rows] = np.asarray(joint_velocities, dtype=np.float32)
        self._arrays["target_position"][rows] = np.asarray(joint_position_targets, dtype=np.float32)
        self._view.reset([int(env_id) for env_id in env_id_array])

    def set_base_velocity(self, env_id: int, linear_velocity: Any, angular_velocity: Any) -> None:
        self._require_view()
        self._view.set_base_velocity(
            int(env_id),
            np.asarray(linear_velocity, dtype=np.float32),
            np.asarray(angular_velocity, dtype=np.float32),
        )

    def env_state(self, env_id: int) -> Any:
        return self.runtime.env_state(env_id)

    def refresh(self) -> GobotGo1FastBatchState:
        self._require_view()
        self._view.refresh()
        if self._state is None:
            self._state = self._state_from_arrays()
        return self._state

    def _require_view(self) -> None:
        if self._view is None:
            raise RuntimeError("Gobot Go1 fast batch backend has not been configured")

    def _state_from_arrays(self) -> GobotGo1FastBatchState:
        self._state = GobotGo1FastBatchState(
            base_position=self._arrays["base_position"],
            base_quaternion=self._arrays["base_quaternion"],
            base_linear_velocity=self._arrays["base_linear_velocity"],
            base_angular_velocity=self._arrays["base_angular_velocity"],
            joint_position=self._arrays["joint_position"],
            joint_velocity=self._arrays["joint_velocity"],
            joint_lower_limit=self._arrays["joint_lower_limit"],
            joint_upper_limit=self._arrays["joint_upper_limit"],
            foot_position=self._arrays["foot_position"],
            foot_velocity=self._arrays["foot_velocity"],
            foot_height=self._arrays["foot_height"],
            foot_contact=self._arrays["foot_contact"],
            foot_contact_force=self._arrays["foot_contact_force"],
            height_scan=self._arrays["height_scan"],
            height_scan_hit=self._arrays["height_scan_hit"],
            height_scan_point=self._arrays["height_scan_point"],
            height_scan_normal=self._arrays["height_scan_normal"],
            illegal_contact_count=self._arrays["illegal_contact_count"],
            self_collision_count=self._arrays["self_collision_count"],
            shank_collision_count=self._arrays["shank_collision_count"],
            trunk_head_collision_count=self._arrays["trunk_head_collision_count"],
        )
        return self._state

    def get_base_pos(self) -> np.ndarray:
        return self.state.base_position

    def get_base_quat(self) -> np.ndarray:
        return self.state.base_quaternion

    def get_base_lin_vel(self) -> np.ndarray:
        return self.state.base_linear_velocity

    def get_base_ang_vel(self) -> np.ndarray:
        return self.state.base_angular_velocity

    def get_dof_pos(self) -> np.ndarray:
        return self.state.joint_position

    def get_dof_vel(self) -> np.ndarray:
        return self.state.joint_velocity

    def get_link_pos(self) -> np.ndarray:
        return self.state.foot_position

    def get_sensor_data(self, name: str) -> np.ndarray:
        if name == self.height_scan_sensor:
            return self.state.height_scan
        if name in self.foot_height_sensor_names:
            index = self.foot_height_sensor_names.index(name)
            return self.state.foot_height[:, index : index + 1]
        if name in self.foot_contact_sensor_names:
            index = self.foot_contact_sensor_names.index(name)
            return self.state.foot_contact[:, index : index + 1]
        raise KeyError(f"Gobot Go1 fast backend has no sensor {name!r}")


__all__ = [
    "BatchSimulationRuntime",
    "GobotGo1FastBatchBackend",
    "GobotGo1FastBatchState",
    "GobotSceneBatchBackend",
    "GobotSceneBatchState",
]
