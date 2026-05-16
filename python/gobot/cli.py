"""Console entry points for packaged Gobot executables."""

from __future__ import annotations

import json
import os
import sys
import sysconfig
from importlib import resources
from pathlib import Path

PYTHON_LIBRARY_ENV = "GOBOT_PYTHON_LIBRARY"
PYTHON_LIBRARY_PRINTED_ENV = "GOBOT_PYTHON_LIBRARY_PRINTED"
PYTHON_EXECUTABLE_ENV = "GOBOT_PYTHON_EXECUTABLE"
PYTHON_HOME_ENV = "GOBOT_PYTHON_HOME"


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            continue
        key = os.fspath(resolved)
        if key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    return result


def _libpython_names() -> list[str]:
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        sysconfig.get_config_var("INSTSONAME"),
        sysconfig.get_config_var("LDLIBRARY"),
        sysconfig.get_config_var("LIBRARY"),
        f"libpython{version}.so.1.0",
        f"libpython{version}.so",
        f"libpython{sys.version_info.major}.so",
    ]
    names: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate:
            continue
        name = Path(candidate).name
        if name.endswith(".a") or name in names:
            continue
        names.append(name)
    return names


def _current_environment_library_dirs() -> list[Path]:
    executable = Path(sys.executable).expanduser()
    roots = [
        executable.absolute().parent.parent,
        executable.resolve().parent.parent,
        Path(sys.prefix),
        Path(sys.exec_prefix),
    ]

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        roots.append(Path(conda_prefix))

    roots.extend([Path(sys.base_prefix), Path(sys.base_exec_prefix)])

    dirs = [root / "lib" for root in roots]
    for key in ("LIBDIR", "LIBPL"):
        value = sysconfig.get_config_var(key)
        if isinstance(value, str) and value:
            dirs.append(Path(value))
    return _dedupe_paths(dirs)


def _loaded_libpython_candidates() -> list[Path]:
    maps_path = Path("/proc/self/maps")
    if not maps_path.exists():
        return []

    prefix = f"libpython{sys.version_info.major}.{sys.version_info.minor}"
    candidates: list[Path] = []
    try:
        lines = maps_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []

    for line in lines:
        path_text = line.rsplit(maxsplit=1)[-1]
        path = Path(path_text)
        if path.name.startswith(prefix) and path.is_file():
            candidates.append(path)
    return _dedupe_paths(candidates)


def _find_current_python_library() -> str:
    override = os.environ.get(PYTHON_LIBRARY_ENV)
    if override:
        return override

    for path in _loaded_libpython_candidates():
        return os.fspath(path)

    names = _libpython_names()
    for directory in _current_environment_library_dirs():
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return os.fspath(candidate.resolve())

    search_dirs = ", ".join(os.fspath(path) for path in _current_environment_library_dirs())
    search_names = ", ".join(names)
    raise RuntimeError(
        "Could not find libpython for the current Python environment. "
        f"Searched names [{search_names}] in [{search_dirs}]. "
        f"Set {PYTHON_LIBRARY_ENV} to the exact libpython path to override discovery."
    )


def _runtime_library_dirs(python_library: str) -> list[str]:
    dirs = _current_environment_library_dirs()
    library_path = Path(python_library)
    if library_path.parent and os.fspath(library_path.parent) != ".":
        dirs.insert(0, library_path.expanduser().parent)
    return [os.fspath(path) for path in _dedupe_paths(dirs) if path.is_dir()]


def _with_editor_python_environment(python_library: str) -> dict[str, str]:
    env = os.environ.copy()
    env[PYTHON_LIBRARY_ENV] = python_library
    env[PYTHON_LIBRARY_PRINTED_ENV] = "1"
    env[PYTHON_EXECUTABLE_ENV] = os.fspath(Path(sys.executable).expanduser().absolute())

    if sys.prefix == sys.base_prefix:
        env[PYTHON_HOME_ENV] = os.fspath(Path(sys.prefix).resolve())
    else:
        env.pop(PYTHON_HOME_ENV, None)

    library_dirs = _runtime_library_dirs(python_library)
    existing = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = os.pathsep.join([*library_dirs, existing] if existing else library_dirs)
    return env


def _project_history_path() -> Path:
    home = Path.home()
    return home / ".gobot" / "projects.json"


def _register_packaged_examples() -> None:
    examples = resources.files(__package__).joinpath("examples")
    examples_path = Path(str(examples))
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


def editor() -> int:
    _register_packaged_examples()

    python_library = _find_current_python_library()
    print(f"[gobot] Python library: {python_library}", file=sys.stderr, flush=True)

    executable = resources.files(__package__).joinpath("gobot_editor")
    argv = [os.fspath(executable), *sys.argv[1:]]
    try:
        os.execvpe(argv[0], argv, _with_editor_python_environment(python_library))
    except OSError as error:
        print(f"[gobot] Failed to launch {argv[0]}: {error}", file=sys.stderr)
        return 127

    return 127
