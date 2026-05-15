"""Python package facade for the Gobot engine bindings."""

import importlib as _importlib

from . import _core
from ._core import *  # noqa: F401,F403
from .scene_helpers import create_cartpole_scene, save_cartpole_scene

__version__ = "0.1.5"

_node_from_id = _core._node_from_id
NodeScript = _core.NodeScript


app = _importlib.import_module(__name__ + ".app")
physics = _importlib.import_module(__name__ + ".physics")
rl = _importlib.import_module(__name__ + ".rl")
scene = _importlib.import_module(__name__ + ".scene")
sim = _importlib.import_module(__name__ + ".sim")

try:
    from ._core import __doc__ as __doc__
except ImportError:
    pass

__all__ = [
    name
    for name in globals()
    if not name.startswith("_") and name not in {"annotations"}
]
