import sys
from pathlib import Path


def prefer_build_gobot():
    """Let PYTHONPATH/build artifacts win over scikit-build editable hooks."""
    repo_root = Path(__file__).resolve().parents[2]
    for path in (repo_root / "python", repo_root / "build/python"):
        path_string = str(path)
        while path_string in sys.path:
            sys.path.remove(path_string)
        sys.path.insert(0, path_string)
    sys.meta_path[:] = [
        finder for finder in sys.meta_path
        if not (
            finder.__class__.__name__ == "ScikitBuildRedirectingFinder"
            and "gobot" in getattr(finder, "known_wheel_files", {})
        )
    ]
