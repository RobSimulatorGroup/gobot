"""Small schema helpers for Gobot RL arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SpecField:
    """One named slice in a flat RL array."""

    name: str
    dim: int
    units: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("spec field name must be non-empty")
        if int(self.dim) < 0:
            raise ValueError(f"spec field {self.name!r} dim must be non-negative")
        object.__setattr__(self, "dim", int(self.dim))


@dataclass(frozen=True)
class ObservationSpec:
    """Named, versioned observation layout."""

    version: str
    fields: tuple[SpecField, ...]
    dtype: Any = np.float32

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("observation spec version must be non-empty")
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    @property
    def dim(self) -> int:
        return sum(field.dim for field in self.fields)

    @property
    def names(self) -> tuple[str, ...]:
        return _flatten_names(self.fields)

    def metadata(self) -> dict[str, Any]:
        return _metadata("observation", self.version, self.fields, self.dtype)

    def validate_array(self, values: Any, *, axis: int = -1, name: str = "observation") -> np.ndarray:
        return _validate_array(values, self.dim, self.dtype, axis=axis, name=name)


@dataclass(frozen=True)
class ActionSpec:
    """Named, versioned action layout."""

    version: str
    fields: tuple[SpecField, ...]
    lower: float | Sequence[float] = -1.0
    upper: float | Sequence[float] = 1.0
    dtype: Any = np.float32

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("action spec version must be non-empty")
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        lower = _bound_array(self.lower, self.dim, "lower")
        upper = _bound_array(self.upper, self.dim, "upper")
        if np.any(lower > upper):
            raise ValueError("action spec lower bounds must be <= upper bounds")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def dim(self) -> int:
        return sum(field.dim for field in self.fields)

    @property
    def names(self) -> tuple[str, ...]:
        return _flatten_names(self.fields)

    def metadata(self) -> dict[str, Any]:
        data = _metadata("action", self.version, self.fields, self.dtype)
        data["lower"] = np.asarray(self.lower, dtype=np.float64).tolist()
        data["upper"] = np.asarray(self.upper, dtype=np.float64).tolist()
        return data

    def validate_array(self, values: Any, *, axis: int = -1, name: str = "action") -> np.ndarray:
        return _validate_array(values, self.dim, self.dtype, axis=axis, name=name)

    def clip(self, values: Any) -> np.ndarray:
        array = self.validate_array(values, name="action")
        return np.clip(array, self.lower, self.upper).astype(self.dtype, copy=False)


def validate_spec_metadata(metadata: Mapping[str, Any], spec: ObservationSpec | ActionSpec, *, kind: str) -> None:
    """Validate stored policy metadata against a runtime spec."""

    version = metadata.get("version")
    if version is not None and str(version) != spec.version:
        raise RuntimeError(f"{kind} spec version mismatch: policy={version!r}, runtime={spec.version!r}")
    dim = metadata.get("dim")
    if dim is not None and int(dim) != spec.dim:
        raise RuntimeError(f"{kind} spec dimension mismatch: policy={dim}, runtime={spec.dim}")
    names = metadata.get("names")
    if names is not None and tuple(str(name) for name in names) != spec.names:
        raise RuntimeError(f"{kind} spec names mismatch")


def _flatten_names(fields: Sequence[SpecField]) -> tuple[str, ...]:
    names: list[str] = []
    for field in fields:
        if field.dim == 0:
            continue
        if field.dim == 1:
            names.append(field.name)
        else:
            names.extend(f"{field.name}.{index}" for index in range(field.dim))
    return tuple(names)


def _metadata(kind: str, version: str, fields: Sequence[SpecField], dtype: np.dtype) -> dict[str, Any]:
    return {
        "kind": kind,
        "version": version,
        "dim": sum(field.dim for field in fields),
        "dtype": str(np.dtype(dtype)),
        "names": _flatten_names(fields),
        "fields": [
            {"name": field.name, "dim": field.dim, "units": field.units}
            for field in fields
        ],
    }


def _validate_array(values: Any, dim: int, dtype: np.dtype, *, axis: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 0:
        raise ValueError(f"{name} must be an array with trailing dimension {dim}")
    resolved_axis = axis if axis >= 0 else array.ndim + axis
    if resolved_axis < 0 or resolved_axis >= array.ndim:
        raise ValueError(f"{name} validation axis {axis} is out of bounds for shape {array.shape}")
    if array.shape[resolved_axis] != dim:
        raise ValueError(f"{name} has dimension {array.shape[resolved_axis]}, expected {dim}")
    return array


def _bound_array(values: float | Sequence[float], dim: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0:
        return np.full((dim,), float(array), dtype=np.float32)
    if array.shape != (dim,):
        raise ValueError(f"action spec {name} bounds must have shape ({dim},), got {array.shape}")
    return array.astype(np.float32, copy=False)


__all__ = [
    "ActionSpec",
    "ObservationSpec",
    "SpecField",
    "validate_spec_metadata",
]
