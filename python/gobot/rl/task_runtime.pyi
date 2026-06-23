from __future__ import annotations

from typing import Any, Mapping, Sequence

class TaskRuntimeMetadata:
    name: str
    version: str
    obs_groups_spec: Mapping[str, int]
    reward_names: Sequence[str]
    backend: str
    cache_info: Mapping[str, Any]
    def __init__(
        self,
        name: str,
        version: str,
        obs_groups_spec: Mapping[str, int] = {},
        reward_names: Sequence[str] = (),
        backend: str = "gobot_native_cpu",
        cache_info: Mapping[str, Any] = {},
    ) -> None: ...
    def metadata(self) -> dict[str, Any]: ...

def normalize_task_runtime_metadata(
    metadata: TaskRuntimeMetadata | Mapping[str, Any] | None,
    *,
    default_name: str,
) -> TaskRuntimeMetadata: ...
