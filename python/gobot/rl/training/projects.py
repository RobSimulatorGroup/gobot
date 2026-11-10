"""Locate shipped robot presets without changing Python's import search path."""
from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def examples_root() -> Path:
    import gobot

    package = Path(gobot.__file__).resolve().parent
    # Source/editable package, standalone CMake package, installed wheel.
    candidates = [package / "examples"]
    candidates.extend(parent / "examples" for parent in package.parents
                      if (parent / "pyproject.toml").is_file()
                      and (parent / "python/gobot/__init__.py").is_file())
    for root in candidates:
        if (root / "locomotion" / "presets.py").is_file():
            return root
    raise FileNotFoundError("Gobot example projects were not found; install a complete Gobot package or use a source checkout")


def example_module(name: str) -> ModuleType:
    root = examples_root()
    prefix = "gobot._examples"
    # Load from the resolved project path. A private namespace avoids editable
    # install hooks redirecting individual example modules to old wheel copies.
    if prefix not in sys.modules:
        spec = importlib.machinery.ModuleSpec(prefix, loader=None, is_package=True)
        spec.submodule_search_locations = [str(root)]
        sys.modules[prefix] = importlib.util.module_from_spec(spec)
    elif list(sys.modules[prefix].__path__) != [str(root)]:
        raise ImportError("Gobot examples were already loaded from another installation")
    module = importlib.import_module(f"{prefix}.{name}")
    source = getattr(module, "__file__", None)
    if source is None or not Path(source).resolve().is_relative_to(root):
        raise ImportError(f"{prefix}.{name} resolves outside Gobot's example projects: {source}")
    return module
