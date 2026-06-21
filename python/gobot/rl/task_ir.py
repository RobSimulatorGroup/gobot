"""Restricted task IR metadata for compiled Gobot RL tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .spec import ActionSpec, ObservationSpec


@dataclass(frozen=True)
class TaskBufferSpec:
    """Named flat-array buffer required by a compiled task."""

    name: str
    shape: tuple[int | str, ...]
    dtype: str = "float32"
    role: str = "state"
    optional: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("task buffer name must be non-empty")
        object.__setattr__(self, "shape", tuple(self.shape))

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "role": self.role,
            "optional": self.optional,
        }


@dataclass(frozen=True)
class TaskExpression:
    """A restricted array expression node.

    This is metadata for code generation, not an executable Python callback.
    """

    op: str
    inputs: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.op:
            raise ValueError("task expression op must be non-empty")
        object.__setattr__(self, "inputs", tuple(str(value) for value in self.inputs))
        object.__setattr__(self, "params", dict(self.params))

    def metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = {"op": self.op}
        if self.inputs:
            data["inputs"] = list(self.inputs)
        if self.params:
            data["params"] = dict(self.params)
        return data


@dataclass(frozen=True)
class RewardTermSpec:
    """One reward term in a compiled task layout."""

    name: str
    weight: float
    expression: TaskExpression
    scale_by_dt: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("reward term name must be non-empty")

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weight": float(self.weight),
            "scale_by_dt": bool(self.scale_by_dt),
            "expression": self.expression.metadata(),
        }


@dataclass(frozen=True)
class TerminationSpec:
    """Termination condition metadata for a compiled task."""

    name: str
    expression: TaskExpression

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("termination name must be non-empty")

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression.metadata(),
        }


@dataclass(frozen=True)
class TaskLayout:
    """Task layout compiled from Gobot scene authoring and robot config."""

    name: str
    version: str
    action_spec: ActionSpec
    obs_groups: Mapping[str, ObservationSpec]
    buffers: tuple[TaskBufferSpec, ...]
    reward_terms: tuple[RewardTermSpec, ...]
    terminations: tuple[TerminationSpec, ...] = ()
    backend: str = "gobot_native_cpu"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("task layout name must be non-empty")
        if not self.version:
            raise ValueError("task layout version must be non-empty")
        object.__setattr__(self, "obs_groups", dict(self.obs_groups))
        object.__setattr__(self, "buffers", tuple(self.buffers))
        object.__setattr__(self, "reward_terms", tuple(self.reward_terms))
        object.__setattr__(self, "terminations", tuple(self.terminations))

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {name: int(spec.dim) for name, spec in self.obs_groups.items()}

    @property
    def reward_names(self) -> tuple[str, ...]:
        return tuple(term.name for term in self.reward_terms)

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "gobot_task_ir",
            "name": self.name,
            "version": self.version,
            "backend": self.backend,
            "action_spec": self.action_spec.metadata(),
            "obs_groups": {name: spec.metadata() for name, spec in self.obs_groups.items()},
            "obs_groups_spec": self.obs_groups_spec,
            "buffers": [buffer.metadata() for buffer in self.buffers],
            "reward_terms": [term.metadata() for term in self.reward_terms],
            "terminations": [termination.metadata() for termination in self.terminations],
        }

    def validate_native_arrays(self, arrays: Any) -> None:
        for buffer in self.buffers:
            if not hasattr(arrays, buffer.name):
                if buffer.optional:
                    continue
                raise RuntimeError(f"native task buffer {buffer.name!r} is missing")
            value = getattr(arrays, buffer.name)
            shape = tuple(getattr(value, "shape", ()))
            if len(shape) != len(buffer.shape):
                raise RuntimeError(f"native task buffer {buffer.name!r} rank mismatch: expected {buffer.shape}, got {shape}")
            for actual, expected in zip(shape, buffer.shape, strict=True):
                if isinstance(expected, int) and int(actual) != expected:
                    raise RuntimeError(
                        f"native task buffer {buffer.name!r} shape mismatch: expected {buffer.shape}, got {shape}"
                    )
            actual_dtype = getattr(value, "dtype", None)
            if actual_dtype is not None and str(np.dtype(actual_dtype)) != str(np.dtype(buffer.dtype)):
                raise RuntimeError(
                    f"native task buffer {buffer.name!r} dtype mismatch: expected {buffer.dtype}, got {actual_dtype}"
                )


TaskIR = TaskLayout


def task_buffer(name: str, *shape: int | str, dtype: str = "float32", role: str = "state", optional: bool = False) -> TaskBufferSpec:
    return TaskBufferSpec(name=name, shape=tuple(shape), dtype=dtype, role=role, optional=optional)


__all__ = [
    "RewardTermSpec",
    "TaskBufferSpec",
    "TaskExpression",
    "TaskIR",
    "TaskLayout",
    "TerminationSpec",
    "task_buffer",
]
