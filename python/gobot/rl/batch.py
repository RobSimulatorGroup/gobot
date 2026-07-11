"""Reusable NumPy batch environment scaffolding."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Any, Mapping

import numpy as np


@dataclass
class BatchEnvState:
    """Batched RL state returned by Gobot environments.

    The core Gobot RL contract is intentionally NumPy-only. Frameworks such as
    RSL-RL, Gymnasium, and Torch live behind adapters.
    """

    obs: dict[str, np.ndarray]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)
    final_observation: dict[str, np.ndarray] | None = None

    @property
    def done(self) -> np.ndarray:
        return np.logical_or(self.terminated, self.truncated)

    def replace(self, **updates: Any) -> "BatchEnvState":
        return replace(self, **updates)


class CpuBatchEnv:
    """Minimal NumPy batch-environment lifecycle base."""

    is_vector_env = True

    def __init__(self, *, num_envs: int, seed: int = 0, autoreset: bool = True) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._autoreset = bool(autoreset)
        self._state: BatchEnvState | None = None
        self.step_counter = 0
        self._final_observation_scratch: dict[str, np.ndarray] | None = None

    @property
    def state(self) -> BatchEnvState | None:
        return self._state

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        """Observation group dimensions, e.g. ``{"obs": 98, "critic": 101}``."""
        raise NotImplementedError

    @property
    def observation_space_shape(self) -> tuple[int]:
        return (sum(int(dim) for dim in self.obs_groups_spec.values()),)

    @property
    def autoreset(self) -> bool:
        return self._autoreset

    @autoreset.setter
    def autoreset(self, value: bool) -> None:
        self._autoreset = bool(value)

    def reset_seed(self, seed: int | None) -> None:
        if seed is None:
            return
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def make_empty_state(self, obs_shapes: dict[str, int], *, dtype: Any = np.float32) -> BatchEnvState:
        obs = {
            name: np.zeros((self.num_envs, int(dim)), dtype=dtype)
            for name, dim in obs_shapes.items()
        }
        return BatchEnvState(
            obs=obs,
            reward=np.zeros((self.num_envs,), dtype=dtype),
            terminated=np.ones((self.num_envs,), dtype=bool),
            truncated=np.zeros((self.num_envs,), dtype=bool),
            info={"steps": np.zeros((self.num_envs,), dtype=np.int64)},
        )

    def init_state(self) -> BatchEnvState:
        self._state = self.make_empty_state(self.obs_groups_spec)
        self.reset_done_envs(self.reset)
        self.clear_step_final_observation()
        return self._state

    def reset(self, env_ids: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        raise NotImplementedError

    def step(self, actions: Any) -> BatchEnvState:
        raise NotImplementedError

    def reset_done_envs(self, reset_fn, *, done: np.ndarray | None = None) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("batch env state has not been initialized")
        done_mask = np.asarray(self._state.done if done is None else done, dtype=bool)
        if done_mask.shape != (self.num_envs,):
            raise ValueError(f"done mask must have shape ({self.num_envs},), got {done_mask.shape}")
        env_ids = np.flatnonzero(done_mask).astype(np.int64)
        if env_ids.size == 0:
            return env_ids

        self.capture_final_observation(env_ids)
        result = reset_fn(env_ids)
        obs: Mapping[str, np.ndarray] | None = None
        reset_info: Mapping[str, Any] = {}
        if isinstance(result, tuple) and len(result) == 2:
            obs, reset_info = result
        elif isinstance(result, Mapping):
            obs = result
        if obs is not None:
            for key, values in obs.items():
                if key not in self._state.obs:
                    raise KeyError(f"reset returned unknown observation group {key!r}")
                self._state.obs[key][env_ids] = np.asarray(values, dtype=self._state.obs[key].dtype)
        if reset_info:
            self.merge_reset_info(env_ids, reset_info)
        steps = self._state.info.get("steps")
        if isinstance(steps, np.ndarray) and steps.shape == (self.num_envs,):
            steps[env_ids] = 0
        self._state.terminated[env_ids] = False
        self._state.truncated[env_ids] = False
        return env_ids

    def capture_final_observation(self, env_ids: np.ndarray) -> None:
        if self._state is None:
            return
        scratch = self._final_observation_scratch
        if scratch is None or set(scratch) != set(self._state.obs):
            scratch = {key: np.zeros_like(value) for key, value in self._state.obs.items()}
            self._final_observation_scratch = scratch
        for key, values in self._state.obs.items():
            scratch[key][env_ids] = values[env_ids]
        self._state.final_observation = scratch
        self._state.info["final_observation"] = scratch
        terminal_mask = self._state.info.get("_final_observation")
        if not isinstance(terminal_mask, np.ndarray) or terminal_mask.shape != (self.num_envs,):
            terminal_mask = np.zeros((self.num_envs,), dtype=bool)
            self._state.info["_final_observation"] = terminal_mask
        terminal_mask[:] = False
        terminal_mask[env_ids] = True

    def clear_step_final_observation(self) -> None:
        if self._state is None:
            return
        self._state.final_observation = None
        self._state.info.pop("final_observation", None)
        self._state.info.pop("_final_observation", None)

    def merge_reset_info(self, env_ids: np.ndarray, reset_info: Mapping[str, Any]) -> None:
        if self._state is None:
            return
        for key, value in reset_info.items():
            if isinstance(value, np.ndarray):
                array = np.asarray(value)
                if array.shape[:1] != (env_ids.size,):
                    self._state.info[key] = array
                    continue
                if key not in self._state.info or not isinstance(self._state.info[key], np.ndarray):
                    self._state.info[key] = np.zeros((self.num_envs, *array.shape[1:]), dtype=array.dtype)
                target = self._state.info[key]
                if isinstance(target, np.ndarray) and target.shape[:1] == (self.num_envs,):
                    target[env_ids] = array
                else:
                    self._state.info[key] = array
            else:
                self._state.info[key] = value

    def update_timing(self, **milliseconds: float) -> None:
        if self._state is None:
            return
        timing = self._state.info.setdefault("timing", {})
        for name, value in milliseconds.items():
            timing[name] = float(value)

    @staticmethod
    def perf_counter() -> float:
        return time.perf_counter()


__all__ = ["BatchEnvState", "CpuBatchEnv"]
