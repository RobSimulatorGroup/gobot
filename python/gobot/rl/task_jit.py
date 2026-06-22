"""Warp-style LLVM compiler for restricted Gobot RL task layouts."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from gobot import _core

from .task_ir import TaskLayout
from .task_kernel import TaskKernel, TaskKernelCompileError, generate_cpp_from_task_kernel
from .task_native import (
    ABI_VERSION,
    NativeTaskArraySpec,
    TaskKernelContext,
    array_specs,
    make_context,
)


_COMPILER_VERSION = "gobot_task_llvm_v1"
_KERNEL_SYMBOL_NAME = "gobot_task_kernel_v1"


class TaskJitCompileError(RuntimeError):
    """Raised when a task JIT kernel cannot be generated or compiled."""


@dataclass(frozen=True)
class TaskJitBuildInfo:
    source_path: Path
    object_path: Path
    library_path: Path
    cache_key: str
    compiler: str
    backend: str
    compile_ms: float
    cache_hit: bool
    array_names: tuple[str, ...]
    array_specs: tuple[NativeTaskArraySpec, ...]


class CompiledTaskKernel:
    """Loaded native task kernel over persistent numpy arrays."""

    def __init__(self, function_address: int, build_info: TaskJitBuildInfo) -> None:
        self.library_path = Path(build_info.library_path)
        self.object_path = Path(build_info.object_path)
        self.build_info = build_info
        self._function_address = int(function_address)
        self._run = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(TaskKernelContext))(self._function_address)
        self._run.argtypes = [ctypes.POINTER(TaskKernelContext)]
        self._run.restype = ctypes.c_int

    @property
    def function_address(self) -> int:
        return self._function_address

    def run(self, arrays: Any) -> None:
        context, keepalive = make_context(arrays, self.build_info.array_names)
        _ = keepalive
        status = int(self._run(ctypes.byref(context)))
        if status != 0:
            raise RuntimeError(f"JIT task kernel failed with status {status}")


class TaskJitCompiler:
    """Compile a decorated Python task kernel to a cached LLVM-loaded object."""

    def __init__(
        self,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        compiler: str | os.PathLike[str] | None = None,
        cxxflags: tuple[str, ...] = (),
    ) -> None:
        self.cache_dir = Path(cache_dir or os.environ.get("GOBOT_RL_TASK_CACHE", "build/task_llvm_cache")).resolve()
        if compiler is not None:
            raise TaskJitCompileError("external task compilers are no longer supported; build Gobot with task LLVM")
        if cxxflags:
            raise TaskJitCompileError("custom task C++ flags are not supported by the LLVM task compiler yet")
        self.compiler = _compiler_identity()
        self.cxxflags = tuple(str(flag) for flag in cxxflags)

    def cache_key(self, task: TaskLayout, arrays: Any, kernel: TaskKernel, array_names: tuple[str, ...]) -> str:
        payload = {
            "compiler_version": _COMPILER_VERSION,
            "abi": ABI_VERSION,
            "task": {
                "name": task.name,
                "version": task.version,
                "backend": task.backend,
                "obs_groups_spec": task.obs_groups_spec,
                "reward_names": task.reward_names,
            },
            "kernel_name": kernel.name,
            "kernel_source": kernel.cache_key_source,
            "arrays": _array_metadata(arrays=arrays, array_names=array_names),
            "cxxflags": self.cxxflags,
            "compiler": self.compiler,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]

    def compile(
        self,
        task: TaskLayout,
        arrays: Any,
        *,
        kernel: TaskKernel,
        force: bool = False,
    ) -> CompiledTaskKernel:
        try:
            source, array_names = generate_cpp_from_task_kernel(kernel, arrays, abi_version=ABI_VERSION)
        except TaskKernelCompileError as error:
            raise TaskJitCompileError(str(error)) from error
        specs = array_specs(arrays, array_names)
        cache_key = self.cache_key(task, arrays, kernel, array_names)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        source_path = self.cache_dir / f"{task.name}_{kernel.name}_{cache_key}.cpp"
        object_path = self.cache_dir / f"{task.name}_{kernel.name}_{cache_key}.o"
        library_path = object_path
        metadata_path = self.cache_dir / f"{task.name}_{kernel.name}_{cache_key}.json"
        module_name = f"gobot_task_{task.name}_{kernel.name}_{cache_key}"
        cache_hit = object_path.exists() and source_path.exists() and not force
        compile_ms = 0.0
        if not cache_hit:
            source_path.write_text(source, encoding="utf-8")
            start = time.perf_counter()
            _compile_cpp_with_llvm(source, str(source_path), object_path)
            compile_ms = (time.perf_counter() - start) * 1000.0
            metadata_path.write_text(
                json.dumps(
                    {
                        "cache_key": cache_key,
                        "compiler": self.compiler,
                        "compile_ms": compile_ms,
                        "backend": "llvm",
                        "kernel": kernel.name,
                        "arrays": _array_metadata(arrays=arrays, array_names=array_names),
                        "task": {
                            "name": task.name,
                            "version": task.version,
                            "backend": task.backend,
                            "obs_groups_spec": task.obs_groups_spec,
                            "reward_names": task.reward_names,
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        build_info = TaskJitBuildInfo(
            source_path=source_path,
            object_path=object_path,
            library_path=library_path,
            cache_key=cache_key,
            compiler=self.compiler,
            backend="llvm",
            compile_ms=compile_ms,
            cache_hit=cache_hit,
            array_names=array_names,
            array_specs=specs,
        )
        function_address = _load_and_lookup(object_path, module_name)
        return CompiledTaskKernel(function_address, build_info)


def _array_metadata(*, arrays: Any, array_names: tuple[str, ...]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for name in array_names:
        array = np.asarray(getattr(arrays, name))
        metadata.append(
            {
                "name": name,
                "dtype": str(np.dtype(array.dtype)),
                "rank": int(array.ndim),
            }
        )
    return metadata


def _compiler_identity() -> str:
    if not bool(getattr(_core, "_task_llvm_available")()):
        raise TaskJitCompileError("Gobot was built without task LLVM support")
    return f"gobot-task-llvm-{getattr(_core, '_task_llvm_version')()}"


def _compile_cpp_with_llvm(source: str, virtual_path: str, object_path: Path) -> None:
    try:
        getattr(_core, "_task_llvm_compile_cpp")(
            source,
            str(virtual_path),
            str(object_path),
            3,
            True,
            False,
        )
    except Exception as error:  # noqa: BLE001 - preserve native diagnostic text.
        raise TaskJitCompileError(str(error)) from error


def _load_and_lookup(object_path: Path, module_name: str) -> int:
    try:
        getattr(_core, "_task_llvm_load_obj")(str(object_path), module_name)
        return int(getattr(_core, "_task_llvm_lookup")(module_name, _KERNEL_SYMBOL_NAME))
    except Exception as error:  # noqa: BLE001 - preserve native diagnostic text.
        raise TaskJitCompileError(str(error)) from error


__all__ = [
    "CompiledTaskKernel",
    "TaskJitBuildInfo",
    "TaskJitCompileError",
    "TaskJitCompiler",
]
