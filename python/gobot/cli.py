"""Console entry points for packaged Gobot executables."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
from importlib import resources
from pathlib import Path


def _runtime_library_dirs() -> list[str]:
    candidates = [
        sysconfig.get_config_var("LIBDIR"),
        os.path.join(sys.prefix, "lib"),
        os.path.join(sys.base_prefix, "lib"),
        os.path.abspath(os.path.join(os.path.dirname(sys.executable), os.pardir, "lib")),
    ]
    return [path for path in dict.fromkeys(candidates) if path and os.path.isdir(path)]


def _with_runtime_library_path() -> dict[str, str]:
    env = os.environ.copy()
    paths = _runtime_library_dirs()
    if not paths:
        return env

    if sys.platform == "win32":
        key = "PATH"
    elif sys.platform == "darwin":
        key = "DYLD_LIBRARY_PATH"
    else:
        key = "LD_LIBRARY_PATH"

    existing = env.get(key)
    env[key] = os.pathsep.join([*paths, existing] if existing else paths)
    return env


def _project_history_path() -> Path:
    home = Path.home()
    return home / ".gobot" / "projects.json"


def _register_packaged_examples() -> None:
    examples = resources.files(__package__).joinpath("examples")
    examples_path = Path(os.fspath(examples))
    if not examples_path.is_dir():
        return

    history_path = _project_history_path()
    data: dict[str, object] = {}
    if history_path.exists():
        try:
            loaded = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, json.JSONDecodeError):
            data = {}

    roots = data.get("example_roots")
    if not isinstance(roots, list):
        roots = []

    canonical_examples = str(examples_path.resolve())
    normalized_roots = [
        str(Path(root).expanduser().resolve())
        for root in roots
        if isinstance(root, str) and root and Path(root).expanduser().exists()
    ]
    normalized_roots = [root for root in normalized_roots if root != canonical_examples]
    normalized_roots.insert(0, canonical_examples)

    data["example_roots"] = normalized_roots
    projects = data.get("projects")
    if not isinstance(projects, list):
        data["projects"] = []

    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(data, indent=4), encoding="utf-8")
    except OSError:
        return


def _run_packaged_executable(name: str) -> int:
    if name == "gobot_editor":
        _register_packaged_examples()
    executable = resources.files(__package__).joinpath(name)
    return subprocess.call([os.fspath(executable), *sys.argv[1:]], env=_with_runtime_library_path())


def editor() -> int:
    return _run_packaged_executable("gobot_editor")
