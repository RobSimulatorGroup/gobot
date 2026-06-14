"""Console entry point for the packaged Gobot editor executable.

This module intentionally does not import :mod:`gobot`. Console scripts import
their target module before running the function, and importing ``gobot`` loads
the native ``_core`` extension before the editor launcher has prepared the
Python embedding environment.
"""

from __future__ import annotations

import json
import os
import sys
import sysconfig
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

PYTHON_LIBRARY_ENV = "GOBOT_PYTHON_LIBRARY"
PYTHON_LIBRARY_PRINTED_ENV = "GOBOT_PYTHON_LIBRARY_PRINTED"
PYTHON_EXECUTABLE_ENV = "GOBOT_PYTHON_EXECUTABLE"
PYTHON_HOME_ENV = "GOBOT_PYTHON_HOME"
EDITOR_EXECUTABLE_ENV = "GOBOT_EDITOR_EXECUTABLE"
DIST_NAME = "gobot"


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
    if library_path.is_dir():
        dirs.insert(0, library_path)
    elif library_path.parent and os.fspath(library_path.parent) != ".":
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
    return Path.home() / ".gobot" / "projects.json"


def _distribution() -> metadata.Distribution | None:
    try:
        return metadata.distribution(DIST_NAME)
    except metadata.PackageNotFoundError:
        return None


def _editable_source_root() -> Path | None:
    dist = _distribution()
    if dist is None:
        return None

    direct_url = dist.read_text("direct_url.json")
    if direct_url is None:
        return None

    try:
        data = json.loads(direct_url)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    dir_info = data.get("dir_info")
    if not isinstance(dir_info, dict) or not dir_info.get("editable"):
        return None

    url = data.get("url")
    if not isinstance(url, str):
        return None
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None

    root = Path(unquote(parsed.path))
    return root if root.is_dir() else None


def _editable_source_examples() -> Path | None:
    root = _editable_source_root()
    if root is None:
        return None

    examples = root / "examples"
    return examples if examples.is_dir() else None


def _package_file(relative_path: str) -> Path | None:
    dist = _distribution()
    if dist is None:
        return None

    files = dist.files
    if files is None:
        return None

    for file in files:
        if file.as_posix() == relative_path:
            path = Path(dist.locate_file(file))
            return path if path.exists() else None
    return None


def _packaged_gobot_dir() -> Path | None:
    editor_path = _package_file("gobot/gobot_editor")
    if editor_path is not None:
        return editor_path.parent

    marker = _package_file("gobot/__init__.py")
    if marker is not None:
        candidate = marker.parent / "gobot_editor"
        if candidate.exists():
            return marker.parent

    return None


def _packaged_examples() -> Path | None:
    gobot_dir = _packaged_gobot_dir()
    if gobot_dir is None:
        return None

    examples = gobot_dir / "examples"
    return examples if examples.is_dir() else None


def _find_editor_executable() -> Path:
    override = os.environ.get(EDITOR_EXECUTABLE_ENV)
    if override:
        executable = Path(override).expanduser()
        if executable.is_file():
            return executable.resolve()
        raise RuntimeError(
            f"{EDITOR_EXECUTABLE_ENV} is set to '{override}', but that file does not exist."
        )

    source_root = _editable_source_root()
    if source_root is not None:
        build_candidates = [
            source_root / "build" / "python" / "gobot" / "gobot_editor",
            source_root / "build" / "gobot_editor",
        ]
        build_root = source_root / "build"
        if build_root.is_dir():
            build_candidates.extend(sorted(build_root.glob("*/python/gobot/gobot_editor")))
        for executable in build_candidates:
            if executable.is_file():
                return executable

    candidate = _packaged_gobot_dir()
    if candidate is not None:
        executable = candidate / "gobot_editor"
        if executable.is_file():
            return executable

    raise RuntimeError(
        "Could not find the packaged gobot_editor executable. "
        "Reinstall gobot or build the project before running this command."
    )


def _register_packaged_examples() -> None:
    candidate_examples = [
        path for path in (_editable_source_examples(), _packaged_examples()) if path is not None
    ]
    existing_examples = [path for path in candidate_examples if path.is_dir()]
    if not existing_examples:
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

    canonical_examples = [str(path.resolve()) for path in existing_examples]
    normalized_roots = [
        str(Path(root).expanduser().resolve())
        for root in roots
        if isinstance(root, str) and root and Path(root).expanduser().exists()
    ]
    normalized_roots = [root for root in normalized_roots if root not in canonical_examples]
    normalized_roots = [*canonical_examples, *normalized_roots]

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

    executable = _find_editor_executable()
    argv = [os.fspath(executable), *sys.argv[1:]]
    try:
        os.execvpe(argv[0], argv, _with_editor_python_environment(python_library))
    except OSError as error:
        print(f"[gobot] Failed to launch {argv[0]}: {error}", file=sys.stderr)
        return 127

    return 127
