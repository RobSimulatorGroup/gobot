"""Console entry points for packaged Gobot executables."""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from importlib import resources


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


def _run_packaged_executable(name: str) -> int:
    executable = resources.files(__package__).joinpath(name)
    return subprocess.call([os.fspath(executable), *sys.argv[1:]], env=_with_runtime_library_path())


def editor() -> int:
    return _run_packaged_executable("gobot_editor")
