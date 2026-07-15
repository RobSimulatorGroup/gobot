"""Shared trace format for Go1 task parity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


TRACE_SCHEMA_VERSION = 1
REFERENCE_TASK_ID = "Mjlab-Velocity-Rough-Unitree-Go1"
REFERENCE_REVISION = "9a43f2d508b963e8caf4af1e7c1d2533d4289287"
REFERENCE_COMPAT_PATCH_ID = "bc85014045526767f9bcf085133603a95ffbf256"
PARITY_SEED = 7
PARITY_COMMAND = (0.4, 0.0, 0.0)
PARITY_TERRAIN_SEED = 42
PARITY_TERRAIN_ROWS = 10
PARITY_TERRAIN_TYPES = (
    "flat",
    "pyramid_stairs",
    "pyramid_stairs_inverted",
    "pyramid_slope",
    "pyramid_slope_inverted",
    "random_rough",
    "wave",
)
PARITY_NUM_ENVS = PARITY_TERRAIN_ROWS * len(PARITY_TERRAIN_TYPES)

# Exercise zero action, action-rate cost, and asymmetric joint motion without
# driving the single-environment trace into an early termination.
PARITY_ACTIONS: tuple[tuple[float, ...], ...] = (
    (0.0,) * 12,
    (0.10, -0.05, 0.05, -0.10, 0.05, -0.05, 0.08, -0.04, 0.04, -0.08, 0.04, -0.04),
    (0.10, -0.05, 0.05, -0.10, 0.05, -0.05, 0.08, -0.04, 0.04, -0.08, 0.04, -0.04),
    (-0.05, 0.08, -0.10, 0.05, -0.08, 0.10, -0.04, 0.06, -0.08, 0.04, -0.06, 0.08),
    (-0.05, 0.08, -0.10, 0.05, -0.08, 0.10, -0.04, 0.06, -0.08, 0.04, -0.06, 0.08),
    (0.0,) * 12,
    (0.0,) * 12,
    (0.0,) * 12,
)


def to_json_value(value: Any) -> Any:
    """Convert a tensor, array, or scalar into stable JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return float(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [to_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_json_value(item) for key, item in value.items()}
    return value


def write_trace(path: str | Path, trace: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def read_trace(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mujoco_model_diagnostics(model: Any) -> dict[str, Any]:
    """Return stable physics diagnostics for a compiled mjModel."""
    dimensions: dict[str, int] = {}
    for name in sorted(dir(model)):
        if name.startswith("_"):
            continue
        try:
            value = getattr(model, name)
        except (AttributeError, RuntimeError, TypeError):
            continue
        if name.startswith("n") and isinstance(value, (int, np.integer)):
            dimensions[name] = int(value)

    options: dict[str, Any] = {}
    for name in sorted(dir(model.opt)):
        if name.startswith("_"):
            continue
        try:
            value = getattr(model.opt, name)
        except (AttributeError, RuntimeError, TypeError):
            continue
        if isinstance(value, (bool, int, np.integer)):
            options[name] = int(value)
        elif isinstance(value, (float, np.floating)):
            options[name] = float(value)
        elif isinstance(value, np.ndarray):
            options[name] = value.tolist()
    heightfields = []
    for index in range(int(model.nhfield)):
        address = int(model.hfield_adr[index])
        count = int(model.hfield_nrow[index] * model.hfield_ncol[index])
        data = np.ascontiguousarray(model.hfield_data[address : address + count])
        grid = data.reshape(int(model.hfield_nrow[index]), int(model.hfield_ncol[index]))
        physical_steps = np.rint(
            grid.astype(np.float64) * float(model.hfield_size[index, 2]) / 0.005
        ).astype(np.int16)
        heightfields.append(
            {
                "name": str(model.hfield(index).name),
                "rows": int(model.hfield_nrow[index]),
                "cols": int(model.hfield_ncol[index]),
                "size": model.hfield_size[index].tolist(),
                "minimum": float(data.min(initial=0.0)),
                "maximum": float(data.max(initial=0.0)),
                "sha256": hashlib.sha256(data.tobytes()).hexdigest(),
                "physical_steps_sha256": hashlib.sha256(
                    np.ascontiguousarray(physical_steps).tobytes()
                ).hexdigest(),
            }
        )

    return {
        "dimensions": dimensions,
        "options": options,
        "heightfields": heightfields,
    }


__all__ = [
    "PARITY_ACTIONS",
    "PARITY_COMMAND",
    "PARITY_SEED",
    "PARITY_NUM_ENVS",
    "PARITY_TERRAIN_ROWS",
    "PARITY_TERRAIN_SEED",
    "PARITY_TERRAIN_TYPES",
    "REFERENCE_REVISION",
    "REFERENCE_COMPAT_PATCH_ID",
    "REFERENCE_TASK_ID",
    "TRACE_SCHEMA_VERSION",
    "read_trace",
    "mujoco_model_diagnostics",
    "to_json_value",
    "write_trace",
]
