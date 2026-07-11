"""Versioned policy manifests shared by training, export, and playback."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .spec import ActionSpec, ObservationSpec, validate_spec_metadata


POLICY_MANIFEST_KIND = "gobot_policy_manifest"
POLICY_MANIFEST_VERSION = 1
POLICY_MANIFEST_KEY = "gobot_policy_manifest"
ONNX_POLICY_MANIFEST_KEY = "gobot.policy_manifest"

_SCENE_DIGEST_IGNORED_KEYS = frozenset(
    {
        "debug_marker_radius",
        "script",
        "source_path",
        "visible",
        "visualize_debug",
    }
)


@dataclass(frozen=True)
class PolicyManifest:
    """The complete runtime contract required to execute a trained policy."""

    task_name: str
    task_version: str
    observation_spec: Mapping[str, Any]
    action_spec: Mapping[str, Any]
    joint_names: Sequence[str]
    physics_dt: float
    decimation: int
    control: Mapping[str, Any]
    model: Mapping[str, Any] = field(default_factory=dict)
    scene_path: str = ""
    scene_digest: str = ""
    extras: Mapping[str, Any] = field(default_factory=dict)
    format_version: int = POLICY_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if int(self.format_version) != POLICY_MANIFEST_VERSION:
            raise ValueError(
                f"unsupported policy manifest version {self.format_version}; "
                f"expected {POLICY_MANIFEST_VERSION}"
            )
        if not self.task_name:
            raise ValueError("policy manifest task_name must be non-empty")
        if not self.task_version:
            raise ValueError("policy manifest task_version must be non-empty")
        if float(self.physics_dt) <= 0.0:
            raise ValueError("policy manifest physics_dt must be greater than zero")
        if int(self.decimation) <= 0:
            raise ValueError("policy manifest decimation must be greater than zero")

        observation_spec = _normalize_mapping(self.observation_spec, "observation_spec")
        action_spec = _normalize_mapping(self.action_spec, "action_spec")
        joint_names = tuple(str(name) for name in self.joint_names)
        if not joint_names or any(not name for name in joint_names):
            raise ValueError("policy manifest joint_names must contain non-empty names")
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("policy manifest joint_names must be unique")
        action_dim = action_spec.get("dim")
        if action_dim is not None and int(action_dim) != len(joint_names):
            raise ValueError(
                "policy manifest action dimension must match joint_names: "
                f"{action_dim} != {len(joint_names)}"
            )

        object.__setattr__(self, "format_version", int(self.format_version))
        object.__setattr__(self, "observation_spec", observation_spec)
        object.__setattr__(self, "action_spec", action_spec)
        object.__setattr__(self, "joint_names", joint_names)
        object.__setattr__(self, "physics_dt", float(self.physics_dt))
        object.__setattr__(self, "decimation", int(self.decimation))
        object.__setattr__(self, "control", _normalize_mapping(self.control, "control"))
        object.__setattr__(self, "model", _normalize_mapping(self.model, "model"))
        object.__setattr__(self, "scene_path", str(self.scene_path))
        object.__setattr__(self, "scene_digest", str(self.scene_digest))
        object.__setattr__(self, "extras", _normalize_mapping(self.extras, "extras"))

    @property
    def policy_dt(self) -> float:
        return self.physics_dt * self.decimation

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": POLICY_MANIFEST_KIND,
            "format_version": self.format_version,
            "task": {"name": self.task_name, "version": self.task_version},
            "observation_spec": _jsonable(self.observation_spec),
            "action_spec": _jsonable(self.action_spec),
            "joint_names": list(self.joint_names),
            "simulation": {
                "physics_dt": self.physics_dt,
                "decimation": self.decimation,
                "policy_dt": self.policy_dt,
            },
            "control": _jsonable(self.control),
            "model": _jsonable(self.model),
            "scene": {
                "path": self.scene_path,
                "digest": self.scene_digest,
            },
            "extras": _jsonable(self.extras),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.metadata(),
            allow_nan=False,
            indent=indent,
            sort_keys=True,
            separators=None if indent is not None else (",", ":"),
        )

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "PolicyManifest":
        if str(metadata.get("kind", "")) != POLICY_MANIFEST_KIND:
            raise ValueError(f"metadata is not a {POLICY_MANIFEST_KIND!r}")
        task = _normalize_mapping(metadata.get("task", {}), "task")
        simulation = _normalize_mapping(metadata.get("simulation", {}), "simulation")
        scene = _normalize_mapping(metadata.get("scene", {}), "scene")
        return cls(
            format_version=int(metadata.get("format_version", 0)),
            task_name=str(task.get("name", "")),
            task_version=str(task.get("version", "")),
            observation_spec=_normalize_mapping(metadata.get("observation_spec", {}), "observation_spec"),
            action_spec=_normalize_mapping(metadata.get("action_spec", {}), "action_spec"),
            joint_names=tuple(str(name) for name in metadata.get("joint_names", ())),
            physics_dt=float(simulation.get("physics_dt", 0.0)),
            decimation=int(simulation.get("decimation", 0)),
            control=_normalize_mapping(metadata.get("control", {}), "control"),
            model=_normalize_mapping(metadata.get("model", {}), "model"),
            scene_path=str(scene.get("path", "")),
            scene_digest=str(scene.get("digest", "")),
            extras=_normalize_mapping(metadata.get("extras", {}), "extras"),
        )

    @classmethod
    def from_json(cls, value: str) -> "PolicyManifest":
        metadata = json.loads(value)
        if not isinstance(metadata, Mapping):
            raise ValueError("policy manifest JSON root must be an object")
        return cls.from_metadata(metadata)

    def validate_runtime(
        self,
        *,
        observation_spec: ObservationSpec,
        action_spec: ActionSpec,
        joint_names: Sequence[str],
        physics_dt: float,
        decimation: int,
        task_name: str | None = None,
        task_version: str | None = None,
        scene_digest: str | None = None,
    ) -> None:
        validate_spec_metadata(self.observation_spec, observation_spec, kind="observation")
        validate_spec_metadata(self.action_spec, action_spec, kind="action")
        runtime_joint_names = tuple(str(name) for name in joint_names)
        if runtime_joint_names != tuple(self.joint_names):
            raise RuntimeError(
                "policy joint order mismatch: "
                f"policy={tuple(self.joint_names)!r}, runtime={runtime_joint_names!r}"
            )
        if not math.isclose(float(physics_dt), self.physics_dt, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError(
                f"policy physics_dt mismatch: policy={self.physics_dt}, runtime={float(physics_dt)}"
            )
        if int(decimation) != self.decimation:
            raise RuntimeError(
                f"policy decimation mismatch: policy={self.decimation}, runtime={int(decimation)}"
            )
        if task_name is not None and str(task_name) != self.task_name:
            raise RuntimeError(
                f"policy task mismatch: policy={self.task_name!r}, runtime={str(task_name)!r}"
            )
        if task_version is not None and str(task_version) != self.task_version:
            raise RuntimeError(
                f"policy task version mismatch: policy={self.task_version!r}, runtime={str(task_version)!r}"
            )
        if scene_digest is not None and self.scene_digest and str(scene_digest) != self.scene_digest:
            raise RuntimeError(
                f"policy scene digest mismatch: policy={self.scene_digest}, runtime={str(scene_digest)}"
            )


def policy_manifest_from_checkpoint(checkpoint: Mapping[str, Any]) -> PolicyManifest | None:
    metadata = checkpoint.get(POLICY_MANIFEST_KEY)
    if metadata is None:
        infos = checkpoint.get("infos")
        if isinstance(infos, Mapping):
            metadata = infos.get(POLICY_MANIFEST_KEY)
    if metadata is None:
        return None
    if isinstance(metadata, PolicyManifest):
        return metadata
    if isinstance(metadata, str):
        return PolicyManifest.from_json(metadata)
    if isinstance(metadata, Mapping):
        return PolicyManifest.from_metadata(metadata)
    raise ValueError(f"checkpoint {POLICY_MANIFEST_KEY!r} must be a mapping or JSON string")


def policy_manifest_from_onnx_metadata(metadata: Mapping[str, str]) -> PolicyManifest | None:
    value = metadata.get(ONNX_POLICY_MANIFEST_KEY)
    return PolicyManifest.from_json(value) if value else None


def policy_manifest_sidecar_path(policy_path: str | Path) -> Path:
    path = Path(policy_path)
    return path.with_suffix(path.suffix + ".manifest.json")


def write_policy_manifest_sidecar(policy_path: str | Path, manifest: PolicyManifest) -> Path:
    sidecar = policy_manifest_sidecar_path(policy_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(manifest.to_json(indent=2) + "\n", encoding="utf-8")
    return sidecar


def read_policy_manifest_sidecar(policy_path: str | Path) -> PolicyManifest | None:
    sidecar = policy_manifest_sidecar_path(policy_path)
    return PolicyManifest.from_json(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else None


def scene_bundle_digest(project_path: str | Path, scene_path: str | Path) -> str:
    """Hash a scene and its recursively referenced simulation resources."""

    project_root = Path(project_path).resolve()
    root_path = _resolve_resource_path(project_root, project_root, str(scene_path))
    visited: set[Path] = set()
    resources: list[Path] = []

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in visited:
            return
        visited.add(path)
        if not path.is_file():
            raise FileNotFoundError(f"scene resource does not exist: {path}")
        resources.append(path)
        if path.suffix.lower() != ".jscn":
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError(f"scene resource root must be an object: {path}")
        external_resources = data.get("__EXT_RESOURCES__", ())
        if not isinstance(external_resources, Sequence) or isinstance(external_resources, (str, bytes)):
            raise ValueError(f"scene __EXT_RESOURCES__ must be an array: {path}")
        for resource in external_resources:
            if not isinstance(resource, Mapping) or "__PATH__" not in resource:
                continue
            if str(resource.get("__TYPE__", "")) == "PythonScript":
                continue
            visit(_resolve_resource_path(project_root, path.parent, str(resource["__PATH__"])))

    visit(root_path)
    digest = hashlib.sha256()
    for path in sorted(resources, key=lambda item: _stable_resource_name(project_root, item)):
        name = _stable_resource_name(project_root, path).encode("utf-8")
        data = _canonical_resource_bytes(path)
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return f"sha256:{digest.hexdigest()}"


def _canonical_resource_bytes(path: Path) -> bytes:
    if path.suffix.lower() != ".jscn":
        return path.read_bytes()
    data = json.loads(path.read_text(encoding="utf-8"))
    canonical = _canonical_scene_value(data)
    return json.dumps(canonical, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_scene_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key = str(key)
            if key in _SCENE_DIGEST_IGNORED_KEYS:
                continue
            if key == "__EXT_RESOURCES__" and isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                item = [
                    resource
                    for resource in item
                    if not isinstance(resource, Mapping) or str(resource.get("__TYPE__", "")) != "PythonScript"
                ]
            result[key] = _canonical_scene_value(item)
        return result
    if isinstance(value, list):
        return [_canonical_scene_value(item) for item in value]
    return value


def _resolve_resource_path(project_root: Path, owner_directory: Path, value: str) -> Path:
    if value.startswith("res://"):
        return project_root / value.removeprefix("res://")
    path = Path(value)
    return path if path.is_absolute() else owner_directory / path


def _stable_resource_name(project_root: Path, path: Path) -> str:
    try:
        return f"res://{path.relative_to(project_root).as_posix()}"
    except ValueError:
        return path.as_posix()


def _normalize_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"policy manifest {name} must be a mapping")
    return {str(key): _jsonable(item) for key, item in value.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("policy manifest cannot contain NaN")
        if math.isinf(value):
            return "-Infinity" if value < 0.0 else "Infinity"
        return value
    if value is None or isinstance(value, str | int | bool):
        return value
    raise TypeError(f"policy manifest value {value!r} is not JSON serializable")


__all__ = [
    "ONNX_POLICY_MANIFEST_KEY",
    "POLICY_MANIFEST_KEY",
    "POLICY_MANIFEST_KIND",
    "POLICY_MANIFEST_VERSION",
    "PolicyManifest",
    "policy_manifest_from_checkpoint",
    "policy_manifest_from_onnx_metadata",
    "policy_manifest_sidecar_path",
    "read_policy_manifest_sidecar",
    "scene_bundle_digest",
    "write_policy_manifest_sidecar",
]
