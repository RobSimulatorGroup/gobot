"""Algorithm-independent training launch contracts and configuration overrides.

A Python factory can return TrainingSpec directly; registering a task is optional.
The runner owns its algorithm's update clock, checkpoint format and environment.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
import math
import os
from pathlib import Path
from functools import lru_cache
from typing import Any, Callable, Mapping, get_args, get_type_hints


_COMMON_PATHS = {
    "backend": "backend", "device": "device", "num_envs": "num_envs",
    "seed": "seed", "output": "output", "resume": "resume",
}


@dataclass(frozen=True)
class TrainingSpec:
    config_factory: Callable[[], Any]
    run: Callable[[Any], Any]
    supported_backends: tuple[str, ...] = ("mujoco-cpu", "mujoco-warp")
    description: str = ""
    step_unit: str = "updates"
    supports_distributed: bool = False
    common_paths: Mapping[str, str] | None = None
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None
    configure: Callable[[Any, argparse.Namespace], Any] | None = None

    def paths(self) -> dict[str, str]:
        return dict(_COMMON_PATHS if self.common_paths is None else self.common_paths)


def config_dict(value: Any) -> Any:
    """JSON-safe resolved configuration; never stringifies unsupported objects."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: config_dict(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): config_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [config_dict(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"training configuration contains a non-serializable {type(value).__name__}")


@lru_cache(maxsize=None)
def _field_types(cls):
    return get_type_hints(cls)


def update_config(config: Any, values: Mapping[str, Any], *, path: str = "") -> Any:
    """Copy and merge known dataclass fields; reject misspelled recipe fields."""
    if not isinstance(values, Mapping):
        raise ValueError(f"{path or 'config'} must be an object")
    if is_dataclass(config) and not isinstance(config, type):
        known = {field.name for field in fields(config) if field.init}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown configuration field: {path}{sorted(unknown)[0]}")
        changes = {}
        for key, value in values.items():
            old = getattr(config, key)
            hint = _field_types(type(config)).get(key)
            changes[key] = _convert_value(old, value, f"{path}{key}",
                                          allow_none=hint is Any or type(None) in get_args(hint))
        return replace(config, **changes)
    if isinstance(config, Mapping):
        result = deepcopy(dict(config))
        for key, value in values.items():
            result[key] = _convert_value(config.get(key), value, f"{path}{key}")
        return result
    raise ValueError(f"{path or 'config'} is not a configurable object")


def _convert_value(old: Any, value: Any, path: str, *, allow_none: bool = False) -> Any:
    if value is None:
        if old is None or allow_none:
            return None
        raise ValueError(f"{path} cannot be null")
    if is_dataclass(old) or isinstance(old, Mapping):
        return update_config(old, value, path=path + ".")
    if isinstance(old, Path):
        if not isinstance(value, (str, Path)):
            raise ValueError(f"{path} must be a path")
        return Path(value)
    if old is not None:
        if isinstance(old, bool) and not isinstance(value, bool):
            raise ValueError(f"{path} must be a boolean")
        if isinstance(old, int) and not isinstance(old, bool) and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"{path} must be an integer")
        if isinstance(old, float) and (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)):
            raise ValueError(f"{path} must be a finite number")
        if isinstance(old, str) and not isinstance(value, str):
            raise ValueError(f"{path} must be a string")
        if isinstance(old, tuple):
            if not isinstance(value, (tuple, list)):
                raise ValueError(f"{path} must be an array")
            return tuple(value)
    return deepcopy(value)


def get_value(config: Any, path: str) -> Any:
    for part in path.split("."):
        try:
            config = config[part] if isinstance(config, Mapping) else getattr(config, part)
        except (KeyError, AttributeError) as error:
            raise ValueError(f"unknown configuration field: {path}") from error
    return config


def set_value(config: Any, path: str, value: Any) -> Any:
    parts = path.split(".")
    if any(not part for part in parts):
        raise ValueError(f"invalid configuration path: {path!r}")
    # Mapping fields are extension points (reward/algorithm overrides); typed
    # dataclass fields are still checked by update_config and task validation.
    current = config
    for part in parts:
        if isinstance(current, Mapping):
            current = current.get(part, {})
        else:
            try:
                current = getattr(current, part)
            except AttributeError as error:
                raise ValueError(f"unknown configuration field: {path}") from error
    nested = value
    for part in reversed(parts):
        nested = {part: nested}
    return update_config(config, nested)


def validate_config(spec: TrainingSpec, config: Any) -> None:
    paths = spec.paths()
    backend = get_value(config, paths["backend"])
    if backend not in spec.supported_backends:
        raise ValueError(f"unsupported backend {backend!r}; this training workflow supports: {', '.join(spec.supported_backends)}")
    device = get_value(config, paths["device"]) if "device" in paths else None
    if backend == "mujoco-warp" and (not isinstance(device, str) or not device.startswith("cuda")):
        raise ValueError("mujoco-warp requires a CUDA device; use --device cuda:0 or --backend mujoco-cpu")
    for key, minimum in (("num_envs", 1), ("seed", 0)):
        if key in paths:
            value = get_value(config, paths[key])
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not spec.supports_distributed:
        raise ValueError("this training workflow does not support synchronized multi-process training; launch independent runs instead")
    validator = getattr(config, "validate", None)
    if validator is not None:
        validator()


def launch(spec: TrainingSpec, config: Any | None = None) -> Any:
    """Launch a Python-configured job with the same checks as gobot-train."""
    cfg = deepcopy(spec.config_factory() if config is None else config)
    paths = spec.paths()
    if "device" in paths and get_value(cfg, paths["device"]) is None:
        backend = get_value(cfg, paths["backend"])
        cfg = set_value(cfg, paths["device"], "cuda:0" if backend == "mujoco-warp" else "cpu")
    validate_config(spec, cfg)
    return spec.run(cfg)
