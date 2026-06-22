"""AOT compiler for restricted Gobot RL task layouts."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np

from .task_ir import TaskLayout
from .task_kernel import TaskKernel, TaskKernelCompileError, generate_cpp_from_task_kernel
from .task_native import (
    ABI_VERSION,
    NativeTaskArraySpec,
    TaskKernelContext,
    array_specs,
    make_context,
)


_COMPILER_VERSION = "gobot_task_aot_v3"


class TaskAotCompileError(RuntimeError):
    """Raised when a task AOT kernel cannot be generated or compiled."""


@dataclass(frozen=True)
class TaskAotBuildInfo:
    source_path: Path
    library_path: Path
    cache_key: str
    compiler: str
    compile_ms: float
    cache_hit: bool
    array_names: tuple[str, ...]
    array_specs: tuple[NativeTaskArraySpec, ...]


class CompiledTaskKernel:
    """Loaded native task kernel over persistent numpy arrays."""

    def __init__(self, library_path: Path, build_info: TaskAotBuildInfo) -> None:
        self.library_path = Path(library_path)
        self.build_info = build_info
        self._library = ctypes.CDLL(str(self.library_path))
        self._run = self._library.gobot_task_kernel_v1
        self._run.argtypes = [ctypes.POINTER(TaskKernelContext)]
        self._run.restype = ctypes.c_int

    @property
    def function_address(self) -> int:
        return int(ctypes.cast(self._run, ctypes.c_void_p).value or 0)

    def run(self, arrays: Any) -> None:
        context, keepalive = make_context(arrays, self.build_info.array_names)
        _ = keepalive
        status = int(self._run(ctypes.byref(context)))
        if status != 0:
            raise RuntimeError(f"AOT task kernel failed with status {status}")


class TaskAotCompiler:
    """Compile a decorated Python task kernel to a small C++ shared object."""

    def __init__(
        self,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        compiler: str | os.PathLike[str] | None = None,
        cxxflags: tuple[str, ...] = (),
    ) -> None:
        self.cache_dir = Path(cache_dir or os.environ.get("GOBOT_RL_TASK_CACHE", "build/task_aot_cache")).resolve()
        self.compiler = str(compiler or _discover_compiler())
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
            "compiler": Path(self.compiler).name,
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
            raise TaskAotCompileError(str(error)) from error
        specs = array_specs(arrays, array_names)
        cache_key = self.cache_key(task, arrays, kernel, array_names)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        source_path = self.cache_dir / f"{task.name}_{kernel.name}_{cache_key}.cpp"
        library_path = self.cache_dir / f"{task.name}_{kernel.name}_{cache_key}.so"
        metadata_path = self.cache_dir / f"{task.name}_{kernel.name}_{cache_key}.json"
        cache_hit = library_path.exists() and source_path.exists() and not force
        compile_ms = 0.0
        if not cache_hit:
            source_path.write_text(source, encoding="utf-8")
            start = time.perf_counter()
            command = [
                self.compiler,
                "-std=c++17",
                "-O3",
                "-fPIC",
                "-shared",
                "-ffast-math",
                str(source_path),
                "-o",
                str(library_path),
                *self.cxxflags,
            ]
            try:
                subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except FileNotFoundError as error:
                raise TaskAotCompileError(f"C++ compiler not found: {self.compiler}") from error
            except subprocess.CalledProcessError as error:
                detail = (error.stderr or error.stdout or "").strip()
                raise TaskAotCompileError(f"failed to compile TaskIR AOT kernel with {self.compiler}: {detail}") from error
            compile_ms = (time.perf_counter() - start) * 1000.0
            metadata_path.write_text(
                json.dumps(
                    {
                        "cache_key": cache_key,
                        "compiler": self.compiler,
                        "compile_ms": compile_ms,
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
        build_info = TaskAotBuildInfo(
            source_path=source_path,
            library_path=library_path,
            cache_key=cache_key,
            compiler=self.compiler,
            compile_ms=compile_ms,
            cache_hit=cache_hit,
            array_names=array_names,
            array_specs=specs,
        )
        return CompiledTaskKernel(library_path, build_info)


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


def _discover_compiler() -> str:
    env = os.environ.get("GOBOT_RL_TASK_CXX")
    if env:
        return env
    candidates = [
        Path("3rdparty/llvm/bin/clang++"),
        Path("warp/external/llvm-project/out/install/release-x86_64/bin/clang++"),
        Path("warp/_build/host-deps/llvm-project/release-x86_64/bin/clang++"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    for name in ("clang++", "g++", "c++"):
        resolved = shutil.which(name)
        if resolved:
            return resolved
    raise TaskAotCompileError(
        "no C++ compiler found for TaskIR AOT; set GOBOT_RL_TASK_CXX or install clang++/g++"
    )


__all__ = [
    "CompiledTaskKernel",
    "TaskAotBuildInfo",
    "TaskAotCompileError",
    "TaskAotCompiler",
]
