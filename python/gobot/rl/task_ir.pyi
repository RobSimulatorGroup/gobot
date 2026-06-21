from __future__ import annotations

from typing import Any, Mapping

from .spec import ActionSpec, ObservationSpec

class TaskBufferSpec:
    name: str
    shape: tuple[int | str, ...]
    dtype: str
    role: str
    optional: bool
    def __init__(
        self,
        name: str,
        shape: tuple[int | str, ...],
        dtype: str = "float32",
        role: str = "state",
        optional: bool = False,
    ) -> None: ...
    def metadata(self) -> dict[str, Any]: ...

class TaskExpression:
    op: str
    inputs: tuple[str, ...]
    params: Mapping[str, Any]
    def __init__(
        self,
        op: str,
        inputs: tuple[str, ...] = (),
        params: Mapping[str, Any] = {},
    ) -> None: ...
    def metadata(self) -> dict[str, Any]: ...

class RewardTermSpec:
    name: str
    weight: float
    expression: TaskExpression
    scale_by_dt: bool
    def __init__(
        self,
        name: str,
        weight: float,
        expression: TaskExpression,
        scale_by_dt: bool = True,
    ) -> None: ...
    def metadata(self) -> dict[str, Any]: ...

class TerminationSpec:
    name: str
    expression: TaskExpression
    def __init__(self, name: str, expression: TaskExpression) -> None: ...
    def metadata(self) -> dict[str, Any]: ...

class TaskLayout:
    name: str
    version: str
    action_spec: ActionSpec
    obs_groups: Mapping[str, ObservationSpec]
    buffers: tuple[TaskBufferSpec, ...]
    reward_terms: tuple[RewardTermSpec, ...]
    terminations: tuple[TerminationSpec, ...]
    backend: str
    def __init__(
        self,
        name: str,
        version: str,
        action_spec: ActionSpec,
        obs_groups: Mapping[str, ObservationSpec],
        buffers: tuple[TaskBufferSpec, ...],
        reward_terms: tuple[RewardTermSpec, ...],
        terminations: tuple[TerminationSpec, ...] = (),
        backend: str = "gobot_native_cpu",
    ) -> None: ...
    @property
    def obs_groups_spec(self) -> dict[str, int]: ...
    @property
    def reward_names(self) -> tuple[str, ...]: ...
    def metadata(self) -> dict[str, Any]: ...
    def validate_native_arrays(self, arrays: Any) -> None: ...

TaskIR = TaskLayout

def task_buffer(
    name: str,
    *shape: int | str,
    dtype: str = "float32",
    role: str = "state",
    optional: bool = False,
) -> TaskBufferSpec: ...
