from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

ABI_VERSION: int
DTYPE_FLOAT32: int
DTYPE_UINT8: int
MAX_RANK: int

class TaskNativeError(RuntimeError): ...

class NativeTaskArraySpec:
    name: str
    dtype: int
    rank: int

class TaskArrayView(ctypes.Structure): ...
class TaskKernelContext(ctypes.Structure): ...

def dtype_code(array: np.ndarray) -> int: ...
def array_specs(arrays: Any, array_names: tuple[str, ...]) -> tuple[NativeTaskArraySpec, ...]: ...
def make_context(arrays: Any, array_names: tuple[str, ...]) -> tuple[TaskKernelContext, dict[str, Any]]: ...
