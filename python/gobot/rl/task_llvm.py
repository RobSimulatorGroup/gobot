"""Diagnostics for Gobot RL task LLVM support."""

from __future__ import annotations

from gobot import _core


def llvm_available() -> bool:
    return bool(_core._task_llvm_available())


def llvm_version() -> str:
    return str(_core._task_llvm_version())


def llvm_last_error() -> str:
    return str(_core._task_llvm_last_error())


__all__ = [
    "llvm_available",
    "llvm_last_error",
    "llvm_version",
]
