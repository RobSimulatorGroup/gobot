"""Lazy entry point for the private Warp IPC numerical runtime."""

from __future__ import annotations

from typing import Any

def availability() -> tuple[bool, str]:
    try:
        import newton
        import warp as wp
    except Exception as error:
        return False, f"could not import Newton/Warp: {type(error).__name__}: {error}"
    solver_base = getattr(getattr(newton, "solvers", None), "SolverBase", None)
    if solver_base is None or not callable(getattr(newton, "eval_fk", None)):
        return False, "Newton 1.4 does not expose SolverBase and eval_fk"
    try:
        wp.init()
        if not wp.is_cuda_available():
            return False, "Warp 1.15 cannot access a CUDA device"
    except Exception as error:
        return False, f"Warp CUDA initialization failed: {type(error).__name__}: {error}"
    return True, ""


def create_session(artifact: Any, config: Any, num_envs: int) -> Any:
    from ._runtime_impl import create_session as create_runtime_session

    return create_runtime_session(artifact, config, num_envs)


__all__: list[str] = []
