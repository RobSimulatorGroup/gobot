"""Console entry points for packaged Gobot executables."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib import resources


def _run_packaged_executable(name: str) -> int:
    executable = resources.files(__package__).joinpath(name)
    return subprocess.call([os.fspath(executable), *sys.argv[1:]])


def editor() -> int:
    return _run_packaged_executable("gobot_editor")
