"""Use shared status helpers without loading Gobot's native physics library."""
import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[2] / 'python/gobot/rl/training/status.py'
_spec = importlib.util.spec_from_file_location('_gobot_training_status', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
ControlledStop = _module.ControlledStop
RunStatus = _module.RunStatus
write_json = _module.write_json
