"""Backend provider contracts for device-native batched simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping


class ProviderUnavailableError(RuntimeError):
    """Raised when an explicitly requested simulation provider is unavailable."""


class GraphInvalidatedError(RuntimeError):
    """Raised when captured graph storage no longer matches the configured session."""


class SimulationCapacityError(RuntimeError):
    """Raised when a batched simulation exceeds a configured fixed capacity."""


def _artifact_digest(content: str) -> str:
    digest = 14695981039346656037
    for byte in content.encode("utf-8"):
        digest ^= byte
        digest = (digest * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{digest:016x}"


@dataclass(frozen=True)
class CompiledSceneArtifact:
    """Versioned backend artifact compiled from a Gobot scene."""

    schema_version: int
    backend: str
    format: str
    content: str
    digest: str
    backend_version: str
    dimensions: Mapping[str, int]
    robot_names: tuple[str, ...]
    robot_prefixes: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CompiledSceneArtifact":
        if not isinstance(value, Mapping):
            raise TypeError("compiled scene artifact must be a mapping")
        schema_version = int(value.get("schema_version", 0))
        if schema_version != 1:
            raise ValueError(f"unsupported compiled scene artifact schema {schema_version}")
        artifact_format = str(value.get("format", "")).lower()
        if artifact_format != "mjcf":
            raise ValueError(f"MuJoCo providers require an MJCF artifact, got {artifact_format!r}")
        content = str(value.get("content", ""))
        if not content.strip():
            raise ValueError("compiled scene artifact has no MJCF content")
        backend_value = value.get("backend", "")
        backend = str(getattr(backend_value, "name", backend_value)).rsplit(".", 1)[-1]
        if backend != "MuJoCoCpu":
            raise ValueError(f"MuJoCo Warp requires a MuJoCoCpu artifact, got {backend!r}")
        digest = str(value.get("content_digest", value.get("digest", "")))
        if not digest:
            raise ValueError("compiled scene artifact has no content digest")
        expected_digest = _artifact_digest(content)
        if digest != expected_digest:
            raise ValueError(
                f"compiled scene artifact digest mismatch: expected {expected_digest}, got {digest}"
            )
        dimensions_value = value.get("dimensions", {})
        if not isinstance(dimensions_value, Mapping):
            raise TypeError("compiled scene artifact dimensions must be a mapping")
        dimensions = {
            str(name): int(size)
            for name, size in dimensions_value.items()
        }
        for name in ("nq", "nv", "nu"):
            if name not in dimensions or dimensions[name] < 0:
                raise ValueError(f"compiled scene artifact has no valid {name!r} dimension")
        robot_names = tuple(str(name) for name in value.get("robot_names", ()))
        robot_prefixes = tuple(str(prefix) for prefix in value.get("robot_prefixes", ()))
        if len(robot_names) != len(robot_prefixes):
            raise ValueError("compiled scene artifact robot names and prefixes do not match")
        backend_version = str(value.get("backend_version", ""))
        if not backend_version:
            raise ValueError("compiled scene artifact has no backend version")
        return cls(
            schema_version=schema_version,
            backend=backend,
            format=artifact_format,
            content=content,
            digest=digest,
            backend_version=backend_version,
            dimensions=dimensions,
            robot_names=robot_names,
            robot_prefixes=robot_prefixes,
        )

    @property
    def content_digest(self) -> str:
        return self.digest

    def robot_prefix(self, robot_name: str) -> str:
        try:
            return self.robot_prefixes[self.robot_names.index(str(robot_name))]
        except ValueError as error:
            raise KeyError(f"compiled scene artifact has no robot {robot_name!r}") from error


@dataclass(frozen=True)
class BatchProviderCapabilities:
    name: str
    device: str
    device_native: bool
    graph_capture: bool
    masked_reset: bool
    fixed_capacity: bool


class BatchPhysicsProvider(ABC):
    """Stable lifecycle used by backend-specific batched physics providers."""

    accepts_device_actions = True

    def __enter__(self) -> "BatchPhysicsProvider":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.close()
        return False

    @property
    @abstractmethod
    def capabilities(self) -> BatchProviderCapabilities: ...

    @property
    @abstractmethod
    def num_envs(self) -> int: ...

    @property
    @abstractmethod
    def arrays(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def step(self, actions: Any | None = None, *, nsteps: int = 1) -> Mapping[str, Any]: ...

    @abstractmethod
    def reset(self, reset_mask: Any, **state: Any) -> Mapping[str, Any]: ...

    @abstractmethod
    def close(self) -> None: ...


__all__ = [
    "BatchPhysicsProvider",
    "BatchProviderCapabilities",
    "CompiledSceneArtifact",
    "GraphInvalidatedError",
    "ProviderUnavailableError",
    "SimulationCapacityError",
]
