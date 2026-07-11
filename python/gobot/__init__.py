"""Python package facade for the Gobot engine bindings."""

import ctypes as _ctypes
import importlib as _importlib
import os as _os
import sys as _sys
import sysconfig as _sysconfig


def _preload_current_libpython() -> None:
    if not _sys.platform.startswith("linux"):
        return
    libdir = _sysconfig.get_config_var("LIBDIR")
    soname = _sysconfig.get_config_var("INSTSONAME") or _sysconfig.get_config_var("LDLIBRARY")
    if not isinstance(libdir, str) or not isinstance(soname, str):
        return
    path = _os.path.join(libdir, soname)
    if not _os.path.isfile(path):
        return
    try:
        _ctypes.CDLL(path, mode=_ctypes.RTLD_GLOBAL)
    except OSError:
        return


_preload_current_libpython()

from . import _core
from ._core import *  # noqa: F401,F403
from .scene_helpers import create_cartpole_scene, save_cartpole_scene

__version__ = _core.__version__

_node_from_id = _core._node_from_id
NodeScript = _core.NodeScript


app = _importlib.import_module(__name__ + ".app")
physics = _importlib.import_module(__name__ + ".physics")
render = _importlib.import_module(__name__ + ".render")
scene = _importlib.import_module(__name__ + ".scene")
sim = _importlib.import_module(__name__ + ".sim")

_LAZY_SUBMODULES = {"rl", "terrain"}


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        module = _importlib.import_module(__name__ + "." + name)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

try:
    from ._core import __doc__ as __doc__
except ImportError:
    pass

__all__ = [
    name
    for name in globals()
    if not name.startswith("_") and name not in {"annotations"}
] + sorted(_LAZY_SUBMODULES)
