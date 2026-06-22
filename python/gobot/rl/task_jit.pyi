from __future__ import annotations

from pathlib import Path
from typing import Any

from .task_ir import TaskLayout
from .task_kernel import TaskKernel
from .task_native import NativeTaskArraySpec

class TaskJitCompileError(RuntimeError): ...

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
    library_path: Path
    object_path: Path
    build_info: TaskJitBuildInfo
    def __init__(self, function_address: int, build_info: TaskJitBuildInfo) -> None: ...
    @property
    def function_address(self) -> int: ...
    def run(self, arrays: Any) -> None: ...

class TaskJitCompiler:
    cache_dir: Path
    compiler: str
    cxxflags: tuple[str, ...]
    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        compiler: str | Path | None = None,
        cxxflags: tuple[str, ...] = (),
    ) -> None: ...
    def cache_key(self, task: TaskLayout, arrays: Any, kernel: TaskKernel, array_names: tuple[str, ...]) -> str: ...
    def compile(
        self,
        task: TaskLayout,
        arrays: Any,
        *,
        kernel: TaskKernel,
        force: bool = False,
    ) -> CompiledTaskKernel: ...
