"""Public reinforcement-learning API for Gobot."""

from __future__ import annotations

from .batch import BatchEnvState, CpuBatchEnv
from .runtime import (
    BatchSimulationRuntime,
    GobotSceneBatchBackend,
    GobotSceneBatchState,
    LocomotionBatchSpec,
    NativeLocomotionBatchBackend,
)
from .spec import ActionSpec, ObservationSpec, SpecField, validate_spec_metadata
from .task_jit import (
    CompiledTaskKernel,
    TaskJitBuildInfo,
    TaskJitCompileError,
    TaskJitCompiler,
)
from .task_ir import (
    RewardTermSpec,
    TaskBufferSpec,
    TaskExpression,
    TaskIR,
    TaskLayout,
    TerminationSpec,
    task_buffer,
)
from .task_kernel import (
    TaskKernel,
    TaskKernelCompileError,
    dim,
    flag,
    kernel,
    param,
    task_kernel,
    tid,
    weight,
    where,
)
from .task_native import (
    NativeTaskArraySpec,
    TaskNativeError,
)
from .task_llvm import (
    llvm_available,
    llvm_last_error,
    llvm_version,
)
from . import locomotion
from .rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
    rsl_rl_cfg_to_dataclass,
    rsl_rl_cfg_to_dict,
)

__all__ = [
    "ActionSpec",
    "BatchEnvState",
    "BatchSimulationRuntime",
    "CompiledTaskKernel",
    "CpuBatchEnv",
    "GobotSceneBatchBackend",
    "GobotSceneBatchState",
    "LocomotionBatchSpec",
    "locomotion",
    "llvm_available",
    "llvm_last_error",
    "llvm_version",
    "NativeLocomotionBatchBackend",
    "NativeTaskArraySpec",
    "ObservationSpec",
    "RewardTermSpec",
    "RslRlBaseRunnerCfg",
    "RslRlModelCfg",
    "RslRlOnPolicyRunnerCfg",
    "RslRlPpoAlgorithmCfg",
    "RslRlVecEnvWrapper",
    "SpecField",
    "TaskJitBuildInfo",
    "TaskJitCompileError",
    "TaskJitCompiler",
    "TaskBufferSpec",
    "TaskExpression",
    "TaskIR",
    "TaskKernel",
    "TaskKernelCompileError",
    "TaskLayout",
    "TaskNativeError",
    "TerminationSpec",
    "dim",
    "flag",
    "kernel",
    "param",
    "rsl_rl_cfg_to_dataclass",
    "rsl_rl_cfg_to_dict",
    "task_buffer",
    "task_kernel",
    "tid",
    "validate_spec_metadata",
    "weight",
    "where",
]
