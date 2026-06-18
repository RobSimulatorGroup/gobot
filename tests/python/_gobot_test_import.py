import sys


def prefer_build_gobot():
    """Let PYTHONPATH/build artifacts win over scikit-build editable hooks."""
    sys.meta_path[:] = [
        finder for finder in sys.meta_path
        if not (
            finder.__class__.__name__ == "ScikitBuildRedirectingFinder"
            and "gobot" in getattr(finder, "known_wheel_files", {})
        )
    ]
