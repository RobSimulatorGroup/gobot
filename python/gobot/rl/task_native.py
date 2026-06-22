"""Native task-kernel registration helpers for Gobot RL."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import numpy as np


ABI_VERSION = 2
DTYPE_FLOAT32 = 1
DTYPE_UINT8 = 2
MAX_RANK = 4


class TaskNativeError(RuntimeError):
    """Raised when a task kernel cannot be registered with Gobot native runtime."""


@dataclass(frozen=True)
class NativeTaskArraySpec:
    name: str
    dtype: int
    rank: int


class TaskArrayView(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("dtype", ctypes.c_uint32),
        ("rank", ctypes.c_uint32),
        ("shape", ctypes.c_size_t * MAX_RANK),
        ("strides", ctypes.c_size_t * MAX_RANK),
        ("data", ctypes.c_void_p),
    ]


class TaskKernelContext(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("array_count", ctypes.c_size_t),
        ("arrays", ctypes.POINTER(TaskArrayView)),
    ]


def dtype_code(array: np.ndarray) -> int:
    dtype = np.dtype(array.dtype)
    if dtype == np.dtype(np.float32):
        return DTYPE_FLOAT32
    if dtype == np.dtype(np.uint8):
        return DTYPE_UINT8
    raise TaskNativeError(f"task kernels only support float32/uint8 buffers, got {dtype}")


def array_specs(arrays: Any, array_names: tuple[str, ...]) -> tuple[NativeTaskArraySpec, ...]:
    specs: list[NativeTaskArraySpec] = []
    for name in array_names:
        array = np.asarray(getattr(arrays, name))
        if array.ndim > MAX_RANK:
            raise TaskNativeError(f"task kernel buffer {name!r} rank {array.ndim} exceeds max rank {MAX_RANK}")
        specs.append(NativeTaskArraySpec(name=name, dtype=dtype_code(array), rank=int(array.ndim)))
    return tuple(specs)


def make_context(arrays: Any, array_names: tuple[str, ...]) -> tuple[TaskKernelContext, dict[str, Any]]:
    keepalive_arrays: dict[str, np.ndarray] = {}
    keepalive_names: list[bytes] = []
    views = (TaskArrayView * len(array_names))()
    for index, name in enumerate(array_names):
        array = np.asarray(getattr(arrays, name))
        if not array.flags.c_contiguous:
            raise TaskNativeError(f"task kernel buffer {name!r} is not contiguous: shape={array.shape}")
        if array.ndim > MAX_RANK:
            raise TaskNativeError(f"task kernel buffer {name!r} rank {array.ndim} exceeds max rank {MAX_RANK}")
        name_bytes = str(name).encode("utf-8")
        keepalive_names.append(name_bytes)
        shape = [0] * MAX_RANK
        strides = [0] * MAX_RANK
        for axis in range(array.ndim):
            shape[axis] = int(array.shape[axis])
            strides[axis] = int(array.strides[axis] // array.itemsize)
        views[index] = TaskArrayView(
            ctypes.c_char_p(name_bytes),
            dtype_code(array),
            int(array.ndim),
            (ctypes.c_size_t * MAX_RANK)(*shape),
            (ctypes.c_size_t * MAX_RANK)(*strides),
            ctypes.c_void_p(int(array.ctypes.data)),
        )
        keepalive_arrays[name] = array
    context = TaskKernelContext(ABI_VERSION, len(array_names), views)
    return context, {"arrays": keepalive_arrays, "names": keepalive_names, "views": views}


__all__ = [
    "ABI_VERSION",
    "DTYPE_FLOAT32",
    "DTYPE_UINT8",
    "MAX_RANK",
    "NativeTaskArraySpec",
    "TaskArrayView",
    "TaskKernelContext",
    "TaskNativeError",
    "array_specs",
    "dtype_code",
    "make_context",
]
