"""Local import helpers for running Go1 training scripts from a checkout."""

from __future__ import annotations

import sys
from pathlib import Path


def prefer_repo_gobot() -> None:
    """Let checkout/build Python files win over a stale editable install."""

    repo_root = Path(__file__).resolve().parents[3]
    # Prefer Python source files from the checkout, while keeping build/python
    # available for the native gobot._core extension.
    for path in (repo_root / "build/python", repo_root / "python"):
        path_string = str(path)
        while path_string in sys.path:
            sys.path.remove(path_string)
        sys.path.insert(0, path_string)
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not (
            finder.__class__.__name__ == "ScikitBuildRedirectingFinder"
            and "gobot" in getattr(finder, "known_wheel_files", {})
        )
    ]


__all__ = ["prefer_repo_gobot"]
