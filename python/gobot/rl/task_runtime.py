"""Lightweight metadata for Gobot RL task runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TaskRuntimeMetadata:
    """Small debug/checkpoint metadata for a batch task runtime."""

    name: str
    version: str
    obs_groups_spec: Mapping[str, int] = field(default_factory=dict)
    reward_names: Sequence[str] = ()
    backend: str = "gobot_native_cpu"
    cache_info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("task runtime metadata name must be non-empty")
        if not self.version:
            raise ValueError("task runtime metadata version must be non-empty")
        object.__setattr__(self, "obs_groups_spec", {str(name): int(dim) for name, dim in self.obs_groups_spec.items()})
        object.__setattr__(self, "reward_names", tuple(str(name) for name in self.reward_names))
        object.__setattr__(self, "cache_info", dict(self.cache_info))

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "gobot_task_runtime_metadata",
            "name": self.name,
            "version": self.version,
            "backend": self.backend,
            "obs_groups_spec": dict(self.obs_groups_spec),
            "reward_names": list(self.reward_names),
            "cache_info": dict(self.cache_info),
        }


def normalize_task_runtime_metadata(
    metadata: TaskRuntimeMetadata | Mapping[str, Any] | None,
    *,
    default_name: str,
) -> TaskRuntimeMetadata:
    if metadata is None:
        return TaskRuntimeMetadata(name=default_name, version="v1")
    if isinstance(metadata, TaskRuntimeMetadata):
        return metadata
    return TaskRuntimeMetadata(
        name=str(metadata.get("name", default_name)),
        version=str(metadata.get("version", "v1")),
        obs_groups_spec=metadata.get("obs_groups_spec", {}),
        reward_names=metadata.get("reward_names", ()),
        backend=str(metadata.get("backend", "gobot_native_cpu")),
        cache_info=metadata.get("cache_info", {}),
    )


__all__ = [
    "TaskRuntimeMetadata",
    "normalize_task_runtime_metadata",
]
