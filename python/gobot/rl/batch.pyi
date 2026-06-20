from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

class BatchEnvState:
    obs: dict[str, npt.NDArray[np.float32]]
    reward: npt.NDArray[np.float32]
    terminated: npt.NDArray[np.bool_]
    truncated: npt.NDArray[np.bool_]
    info: dict[str, Any]
    final_observation: dict[str, npt.NDArray[np.float32]] | None
    def __init__(
        self,
        obs: dict[str, npt.NDArray[np.float32]],
        reward: npt.NDArray[np.float32],
        terminated: npt.NDArray[np.bool_],
        truncated: npt.NDArray[np.bool_],
        info: dict[str, Any] = ...,
        final_observation: dict[str, npt.NDArray[np.float32]] | None = ...,
    ) -> None: ...
    @property
    def done(self) -> npt.NDArray[np.bool_]: ...
    def replace(self, **updates: Any) -> BatchEnvState: ...

class CpuBatchEnv:
    is_vector_env: bool
    num_envs: int
    seed: int
    step_counter: int
    def __init__(self, *, num_envs: int, seed: int = 0, autoreset: bool = True) -> None: ...
    @property
    def state(self) -> BatchEnvState | None: ...
    @property
    def autoreset(self) -> bool: ...
    @autoreset.setter
    def autoreset(self, value: bool) -> None: ...
    def reset_seed(self, seed: int | None) -> None: ...
    def make_empty_state(self, obs_shapes: dict[str, int], *, dtype: Any = np.float32) -> BatchEnvState: ...
    def reset_done_envs(self, reset_fn: Any, *, done: npt.NDArray[np.bool_] | None = None) -> npt.NDArray[np.int64]: ...
    def capture_final_observation(self, env_ids: npt.NDArray[np.int64]) -> None: ...
    def update_timing(self, **milliseconds: float) -> None: ...
