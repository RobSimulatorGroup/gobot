"""Shared scaffolding for locomotion batch environments."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from typing import Any, Mapping, Sequence

import numpy as np

from gobot.rl.batch import CpuBatchEnv


@dataclass
class LocomotionControlCfg:
    """Common normalized action control settings."""

    action_scale: float | Mapping[str, float] = 0.25
    simulate_action_latency: bool = False


@dataclass
class LocomotionNoiseCfg:
    """Uniform observation noise settings."""

    level: float = 0.0
    ranges: Mapping[str, tuple[float, float] | float] = field(default_factory=dict)


class LocomotionBatchEnv(CpuBatchEnv):
    """Reusable base for CPU batch locomotion tasks."""

    profile_phase_order: Sequence[tuple[str, str]] = (
        ("action_prepare", "start"),
        ("action_apply", "action_prepare"),
        ("physics", "action_apply"),
        ("state", "physics"),
        ("command", "state"),
        ("contact", "command"),
        ("reward", "contact"),
        ("termination_reset", "reward"),
        ("obs_build", "termination_reset"),
        ("tensor_convert", "obs_build"),
        ("extras", "tensor_convert"),
    )

    def __init__(
        self,
        *,
        num_envs: int,
        seed: int = 0,
        control_cfg: LocomotionControlCfg | None = None,
        noise_cfg: LocomotionNoiseCfg | None = None,
        joint_names: Sequence[str] = (),
        default_joint_pos: Sequence[float] | np.ndarray = (),
        profile_step: bool = False,
        autoreset: bool = True,
    ) -> None:
        super().__init__(num_envs=num_envs, seed=seed, autoreset=autoreset)
        self.control_cfg = control_cfg if control_cfg is not None else LocomotionControlCfg()
        self.noise_cfg = noise_cfg if noise_cfg is not None else LocomotionNoiseCfg()
        self.profile_step = bool(profile_step)
        self._profile_step_count = 0
        self._profile_totals: dict[str, float] = {}

        self.joint_names = tuple(str(name) for name in joint_names)
        self.num_actions = len(self.joint_names)
        self.default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32)
        if self.default_joint_pos.size:
            self.default_joint_pos = self.default_joint_pos.reshape(-1)
            if self.num_actions == 0:
                self.num_actions = int(self.default_joint_pos.shape[0])
            elif self.default_joint_pos.shape != (self.num_actions,):
                raise ValueError(
                    "joint_names and default_joint_pos must have the same length "
                    f"({self.num_actions} != {self.default_joint_pos.shape[0]})"
                )
        self.action_scale = self._action_scale_array(self.control_cfg.action_scale)
        self._previous_actions = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)
        self._last_actions = np.zeros_like(self._previous_actions)
        self._submitted_actions = np.zeros_like(self._previous_actions)

    @property
    def submitted_actions(self) -> np.ndarray:
        return self._submitted_actions

    def _prepare_actions(self, actions: Any) -> np.ndarray:
        action_np = actions.detach().cpu().numpy() if hasattr(actions, "detach") else np.asarray(actions)
        action_np = np.asarray(action_np, dtype=np.float32).reshape(self.num_envs, self.num_actions)
        self._submitted_actions = np.clip(action_np, -1.0, 1.0)
        return self._submitted_actions

    def _actions_for_control(self, actions: np.ndarray) -> np.ndarray:
        control_cfg = getattr(self, "control_cfg", LocomotionControlCfg())
        if control_cfg.simulate_action_latency:
            return self._last_actions.copy()
        return np.asarray(actions, dtype=np.float32).reshape(self.num_envs, self.num_actions)

    def target_positions_from_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = self._actions_for_control(actions)
        return (
            self.default_joint_pos.reshape(1, -1)
            + self.action_scale.reshape(1, -1) * actions
        ).astype(np.float64)

    def _target_positions_from_actions(self, actions: np.ndarray) -> np.ndarray:
        return self.target_positions_from_actions(actions)

    def _action_scale_array(self, scale: float | Mapping[str, float]) -> np.ndarray:
        if self.num_actions <= 0:
            return np.zeros((0,), dtype=np.float32)
        if isinstance(scale, Mapping):
            values = []
            default = float(scale.get("__default__", 0.35))
            for joint_name in self.joint_names:
                matched = None
                for pattern, value in scale.items():
                    if pattern == "__default__":
                        continue
                    if pattern == joint_name or re.fullmatch(pattern, joint_name):
                        matched = float(value)
                        break
                values.append(default if matched is None else matched)
            return np.asarray(values, dtype=np.float32)
        return np.full((self.num_actions,), float(scale), dtype=np.float32)

    def _obs_noise(self, values: np.ndarray, scale_or_name: str | float, *, corrupt: bool = True) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if not corrupt:
            return values
        noise_cfg = getattr(self, "noise_cfg", LocomotionNoiseCfg())
        level = float(noise_cfg.level)
        if level <= 0.0:
            return values
        if isinstance(scale_or_name, str):
            noise_range = noise_cfg.ranges.get(scale_or_name)
            if noise_range is None:
                return values
            if isinstance(noise_range, tuple):
                lo, hi = noise_range
                return values + self._rng.uniform(float(lo), float(hi), size=values.shape).astype(np.float32) * level
            scale = float(noise_range)
        else:
            scale = float(scale_or_name)
        if scale <= 0.0:
            return values
        noise = self._rng.uniform(-1.0, 1.0, size=values.shape).astype(np.float32)
        return values + noise * level * scale

    def _locomotion_backend(self) -> Any:
        backend = getattr(self, "backend", None)
        if backend is None:
            backend = getattr(self, "_backend", None)
        if backend is None:
            raise RuntimeError("locomotion backend has not been configured")
        return backend

    def get_base_pos(self) -> np.ndarray:
        return self._locomotion_backend().get_base_pos()

    def get_base_quat(self) -> np.ndarray:
        return self._locomotion_backend().get_base_quat()

    def get_base_lin_vel(self) -> np.ndarray:
        return self._locomotion_backend().get_base_lin_vel()

    def get_base_ang_vel(self) -> np.ndarray:
        return self._locomotion_backend().get_base_ang_vel()

    def get_dof_pos(self) -> np.ndarray:
        return self._locomotion_backend().get_dof_pos()

    def get_dof_vel(self) -> np.ndarray:
        return self._locomotion_backend().get_dof_vel()

    def get_sensor_data(self, name: str) -> np.ndarray:
        return self._locomotion_backend().get_sensor_data(name)

    def profile_summary(self) -> dict[str, float]:
        if self._profile_step_count <= 0:
            return {}
        return {name: total / self._profile_step_count for name, total in self._profile_totals.items()}

    def _new_profile_marks(self) -> dict[str, float] | None:
        return {"start": time.perf_counter()} if self.profile_step else None

    @staticmethod
    def _mark_profile(marks: dict[str, float] | None, name: str) -> None:
        if marks is not None:
            marks[name] = time.perf_counter()

    def _consume_profile_marks(self, marks: dict[str, float]) -> dict[str, float]:
        values: dict[str, float] = {}
        for name, previous in self.profile_phase_order:
            if name in marks and previous in marks:
                values[name] = (marks[name] - marks[previous]) * 1000.0
        final_mark = self.profile_phase_order[-1][0] if self.profile_phase_order else None
        if final_mark is not None and final_mark in marks:
            values["total"] = (marks[final_mark] - marks["start"]) * 1000.0
        self._profile_step_count += 1
        for name, value in values.items():
            self._profile_totals[name] = self._profile_totals.get(name, 0.0) + value
        return self.profile_summary()


__all__ = ["LocomotionBatchEnv", "LocomotionControlCfg", "LocomotionNoiseCfg"]
