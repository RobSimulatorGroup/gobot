"""Python facade over Gobot's existing batch simulation APIs."""

from __future__ import annotations

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


__all__ = ["BatchSimulationRuntime"]
