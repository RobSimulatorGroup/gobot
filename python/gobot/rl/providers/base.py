"""Backend provider contracts for device-native batched simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import operator
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

from ...sim import (
    ProviderCapabilities as BatchProviderCapabilities,
    ProviderUnavailableError,
)


class GraphInvalidatedError(RuntimeError):
    """Raised when captured graph storage no longer matches the configured session."""


class SimulationCapacityError(RuntimeError):
    """Raised when a batched simulation exceeds a configured fixed capacity."""


def _artifact_digest(content: str) -> str:
    """Return the digest used by Gobot's C++ scene artifact compiler."""

    digest = 14695981039346656037
    for byte in content.encode("utf-8"):
        digest ^= byte
        digest = (digest * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{digest:016x}"


def _normalized_json_value(value: Any) -> Any:
    """Convert provider configuration into deterministic JSON-compatible data."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _normalized_json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name in normalized:
                raise ValueError(f"provider config contains duplicate normalized key {name!r}")
            normalized[name] = _normalized_json_value(item)
        return {name: normalized[name] for name in sorted(normalized)}
    if isinstance(value, (tuple, list)):
        return [_normalized_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_normalized_json_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("provider config floats must be finite")
        return normalized
    enum_name = getattr(value, "name", None)
    if isinstance(enum_name, str):
        return enum_name
    raise TypeError(
        f"provider config value {value!r} has unsupported type {type(value).__name__}"
    )


@dataclass(frozen=True)
class CompiledControlTopology:
    """One stable control input in the compiled artifact's source order."""

    index: int
    name: str
    joint: str
    mode: str
    robot: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CompiledControlTopology":
        if not isinstance(value, Mapping):
            raise TypeError("compiled control topology must be a mapping")
        mode = str(value.get("mode", "")).lower()
        if mode not in ("position", "velocity", "direct"):
            raise ValueError(f"compiled control topology has invalid mode {mode!r}")
        name = str(value.get("name", ""))
        if not name:
            raise ValueError("compiled control topology has no runtime name")
        return cls(
            index=int(value.get("index", -1)),
            name=name,
            joint=str(value.get("joint", "")),
            mode=mode,
            robot=str(value.get("robot", "")),
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "joint": self.joint,
            "mode": self.mode,
            "robot": self.robot,
        }


@dataclass(frozen=True)
class CompiledRobotTopology:
    """Runtime names owned by one authored Gobot robot."""

    name: str
    runtime_prefix: str
    body_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    control_indices: tuple[int, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CompiledRobotTopology":
        if not isinstance(value, Mapping):
            raise TypeError("compiled robot topology must be a mapping")
        name = str(value.get("name", ""))
        if not name:
            raise ValueError("compiled robot topology has no robot name")
        runtime_prefix = str(value.get("runtime_prefix", ""))
        return cls(
            name=name,
            runtime_prefix=runtime_prefix,
            body_names=tuple(str(item) for item in value.get("body_names", ())),
            joint_names=tuple(str(item) for item in value.get("joint_names", ())),
            control_indices=tuple(int(item) for item in value.get("control_indices", ())),
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "runtime_prefix": self.runtime_prefix,
            "body_names": list(self.body_names),
            "joint_names": list(self.joint_names),
            "control_indices": list(self.control_indices),
        }


def _validate_unique_strings(values: Sequence[str], description: str) -> None:
    if any(not value for value in values):
        raise ValueError(f"compiled scene artifact has an empty {description}")
    if len(set(values)) != len(values):
        raise ValueError(f"compiled scene artifact has duplicate {description}s")


def _legacy_robot_and_control_topology(
    content: str,
    robot_names: tuple[str, ...],
    robot_prefixes: tuple[str, ...],
) -> tuple[tuple[CompiledRobotTopology, ...], tuple[CompiledControlTopology, ...]]:
    """Bridge legacy schema-v1 compiler output into the current v2 contract.

    This deliberately lives outside :meth:`CompiledSceneArtifact.from_mapping`:
    arbitrary v1 artifacts remain unsupported. The bridge derives only the
    topology fields that were absent from the legacy compiler response.
    """

    try:
        root = ET.fromstring(content)
    except ET.ParseError as error:
        raise ValueError(f"compiled scene artifact contains invalid MJCF: {error}") from error

    def owner(runtime_name: str) -> str:
        matches = [
            (len(prefix), robot_name)
            for robot_name, prefix in zip(robot_names, robot_prefixes, strict=True)
            if prefix and runtime_name.startswith(prefix)
        ]
        return max(matches, default=(0, ""))[1]

    body_names = tuple(
        element.attrib["name"]
        for element in root.findall("./worldbody//body")
        if element.attrib.get("name")
    )
    joint_names = tuple(
        element.attrib["name"]
        for tag in ("joint", "freejoint")
        for element in root.findall(f"./worldbody//{tag}")
        if element.attrib.get("name")
    )
    controls: list[CompiledControlTopology] = []
    for index, element in enumerate(
        child for section in root.findall("actuator") for child in section
    ):
        control_name = element.attrib.get("name", f"actuator_{index}")
        joint_name = element.attrib.get("joint", "")
        robot_name = owner(joint_name) or owner(control_name)
        if element.tag in ("position", "velocity") and joint_name:
            mode = element.tag
        elif element.tag == "general" and control_name.endswith("_position"):
            # The legacy compiler serialized MuJoCo's canonical affine form;
            # schema v2 carries this semantic explicitly and needs no inference.
            mode = "position"
        elif element.tag == "general" and control_name.endswith("_velocity"):
            mode = "velocity"
        else:
            mode = "direct"
        controls.append(
            CompiledControlTopology(
                index=index,
                name=control_name,
                joint=joint_name,
                mode=mode,
                robot=robot_name,
            )
        )

    robots = tuple(
        CompiledRobotTopology(
            name=robot_name,
            runtime_prefix=prefix,
            body_names=tuple(name for name in body_names if owner(name) == robot_name),
            joint_names=tuple(name for name in joint_names if owner(name) == robot_name),
            control_indices=tuple(
                control.index for control in controls if control.robot == robot_name
            ),
        )
        for robot_name, prefix in zip(robot_names, robot_prefixes, strict=True)
    )
    return robots, tuple(controls)


@dataclass(frozen=True)
class CompiledSceneArtifact:
    """Schema-v2 portable physics artifact compiled from a Gobot scene."""

    schema_version: int
    producer: str
    format: str
    content: str
    content_digest: str
    producer_version: str
    dimensions: Mapping[str, int]
    robots: tuple[CompiledRobotTopology, ...]
    controls: tuple[CompiledControlTopology, ...]
    terrain_geom_groups: tuple[int, ...]

    def __post_init__(self) -> None:
        schema_version = int(self.schema_version)
        if schema_version != 2:
            raise ValueError(
                f"unsupported compiled scene artifact schema {schema_version}; expected schema 2"
            )
        producer = str(self.producer).lower()
        if producer != "mujoco":
            raise ValueError(
                f"MuJoCo device providers require producer 'mujoco', got {self.producer!r}"
            )
        artifact_format = str(self.format).lower()
        if artifact_format != "mjcf":
            raise ValueError(
                f"MuJoCo providers require an MJCF artifact, got {artifact_format!r}"
            )
        content = str(self.content)
        if not content.strip():
            raise ValueError("compiled scene artifact has no MJCF content")
        expected_digest = _artifact_digest(content)
        content_digest = str(self.content_digest)
        if content_digest != expected_digest:
            raise ValueError(
                "compiled scene artifact digest mismatch: "
                f"expected {expected_digest}, got {content_digest}"
            )
        producer_version = str(self.producer_version)
        if not producer_version:
            raise ValueError("compiled scene artifact has no producer version")
        if not isinstance(self.dimensions, Mapping):
            raise TypeError("compiled scene artifact dimensions must be a mapping")
        dimensions = {str(name): int(size) for name, size in self.dimensions.items()}
        for name in ("nq", "nv", "nu"):
            if name not in dimensions or dimensions[name] < 0:
                raise ValueError(f"compiled scene artifact has no valid {name!r} dimension")

        robots = tuple(
            CompiledRobotTopology.from_mapping(robot.to_mapping())
            if isinstance(robot, CompiledRobotTopology)
            else CompiledRobotTopology.from_mapping(robot)
            for robot in self.robots
        )
        controls = tuple(
            CompiledControlTopology.from_mapping(control.to_mapping())
            if isinstance(control, CompiledControlTopology)
            else CompiledControlTopology.from_mapping(control)
            for control in self.controls
        )
        robot_names = tuple(robot.name for robot in robots)
        _validate_unique_strings(robot_names, "robot name")
        prefixes = tuple(robot.runtime_prefix for robot in robots)
        if len(set(prefixes)) != len(prefixes):
            raise ValueError("compiled scene artifact has duplicate robot runtime prefixes")
        if len(controls) != dimensions["nu"]:
            raise ValueError(
                "compiled scene artifact control topology does not match dimension nu: "
                f"{len(controls)}/{dimensions['nu']}"
            )
        if tuple(control.index for control in controls) != tuple(range(len(controls))):
            raise ValueError(
                "compiled scene artifact control indices must be contiguous source-order indices"
            )
        _validate_unique_strings(
            tuple(control.name for control in controls),
            "control runtime name",
        )
        robot_by_name = {robot.name: robot for robot in robots}
        for control in controls:
            if control.robot and control.robot not in robot_by_name:
                raise ValueError(
                    f"compiled control {control.name!r} references unknown robot {control.robot!r}"
                )
            if (
                control.robot
                and control.joint
                and control.joint not in robot_by_name[control.robot].joint_names
            ):
                raise ValueError(
                    f"compiled control {control.name!r} references joint {control.joint!r} "
                    f"outside robot {control.robot!r}"
                )
        for robot in robots:
            _validate_unique_strings(robot.body_names, f"body name for robot {robot.name!r}")
            _validate_unique_strings(robot.joint_names, f"joint name for robot {robot.name!r}")
            if len(set(robot.control_indices)) != len(robot.control_indices):
                raise ValueError(
                    f"compiled robot {robot.name!r} has duplicate control indices"
                )
            expected_controls = tuple(
                control.index for control in controls if control.robot == robot.name
            )
            if robot.control_indices != expected_controls:
                raise ValueError(
                    f"compiled robot {robot.name!r} control ownership does not match controls"
                )
            for control_index in robot.control_indices:
                if control_index < 0 or control_index >= len(controls):
                    raise ValueError(
                        f"compiled robot {robot.name!r} has invalid control index {control_index}"
                    )
                if controls[control_index].robot != robot.name:
                    raise ValueError(
                        f"compiled robot {robot.name!r} does not own control index {control_index}"
                    )
        terrain_geom_groups = tuple(int(group) for group in self.terrain_geom_groups)
        if len(set(terrain_geom_groups)) != len(terrain_geom_groups) or any(
            group < 0 or group > 5 for group in terrain_geom_groups
        ):
            raise ValueError(
                f"compiled scene artifact has invalid terrain geom groups {terrain_geom_groups}"
            )

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "format", artifact_format)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "content_digest", content_digest)
        object.__setattr__(self, "producer_version", producer_version)
        object.__setattr__(self, "dimensions", MappingProxyType(dimensions))
        object.__setattr__(self, "robots", robots)
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "terrain_geom_groups", terrain_geom_groups)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CompiledSceneArtifact":
        """Validate a public schema-v2 artifact mapping.

        Schema v1 is intentionally rejected here. Providers use the separate
        compiler bridge for the current in-process C++ binding during migration.
        """

        if not isinstance(value, Mapping):
            raise TypeError("compiled scene artifact must be a mapping")
        schema_version = int(value.get("schema_version", 0))
        if schema_version != 2:
            raise ValueError(
                f"unsupported compiled scene artifact schema {schema_version}; expected schema 2"
            )
        return cls(
            schema_version=schema_version,
            producer=str(value.get("producer", "")),
            format=str(value.get("format", "")),
            content=str(value.get("content", "")),
            content_digest=str(value.get("content_digest", "")),
            producer_version=str(value.get("producer_version", "")),
            dimensions=value.get("dimensions", {}),
            robots=tuple(value.get("robots", ())),
            controls=tuple(value.get("controls", ())),
            terrain_geom_groups=tuple(value.get("terrain_geom_groups", ())),
        )

    @classmethod
    def from_compiler_mapping(cls, value: Mapping[str, Any]) -> "CompiledSceneArtifact":
        """Validate compiler output, bridging only legacy Gobot schema v1."""

        if not isinstance(value, Mapping):
            raise TypeError("compiled scene artifact must be a mapping")
        schema_version = int(value.get("schema_version", 0))
        if schema_version == 2:
            return cls.from_mapping(value)
        if schema_version != 1:
            raise ValueError(
                f"unsupported compiled scene artifact schema {schema_version}; expected schema 2"
            )
        backend_value = value.get("backend", "")
        backend = str(getattr(backend_value, "name", backend_value)).rsplit(".", 1)[-1]
        if backend != "MuJoCoCpu":
            raise ValueError(
                "legacy compiler artifact bridge only accepts MuJoCoCpu output, "
                f"got {backend!r}"
            )
        content = str(value.get("content", ""))
        robot_names = tuple(str(name) for name in value.get("robot_names", ()))
        robot_prefixes = tuple(str(prefix) for prefix in value.get("robot_prefixes", ()))
        if len(robot_names) != len(robot_prefixes):
            raise ValueError("compiled scene artifact robot names and prefixes do not match")
        _validate_unique_strings(robot_names, "robot name")
        robots, controls = _legacy_robot_and_control_topology(
            content,
            robot_names,
            robot_prefixes,
        )
        return cls(
            schema_version=2,
            producer="mujoco",
            format=str(value.get("format", "")),
            content=content,
            content_digest=str(value.get("content_digest", value.get("digest", ""))),
            producer_version=str(value.get("backend_version", "")),
            dimensions=value.get("dimensions", {}),
            robots=robots,
            controls=controls,
            terrain_geom_groups=tuple(value.get("terrain_geom_groups", ())),
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "producer": self.producer,
            "format": self.format,
            "content": self.content,
            "content_digest": self.content_digest,
            "producer_version": self.producer_version,
            "dimensions": dict(self.dimensions),
            "robots": [robot.to_mapping() for robot in self.robots],
            "controls": [control.to_mapping() for control in self.controls],
            "terrain_geom_groups": list(self.terrain_geom_groups),
        }

    @property
    def digest(self) -> str:
        """Compatibility alias for code that stores artifact-bound layouts."""

        return self.content_digest

    @property
    def backend(self) -> str:
        return "MuJoCoCpu"

    @property
    def backend_version(self) -> str:
        return self.producer_version

    @property
    def robot_names(self) -> tuple[str, ...]:
        return tuple(robot.name for robot in self.robots)

    @property
    def robot_prefixes(self) -> tuple[str, ...]:
        return tuple(robot.runtime_prefix for robot in self.robots)

    def robot_prefix(self, robot_name: str) -> str:
        for robot in self.robots:
            if robot.name == str(robot_name):
                return robot.runtime_prefix
        raise KeyError(f"compiled scene artifact has no robot {robot_name!r}")

    def control_for_joint(self, runtime_joint_name: str) -> CompiledControlTopology:
        """Resolve one joint's preferred control without runtime-name inference."""

        candidates = [
            control for control in self.controls if control.joint == str(runtime_joint_name)
        ]
        if not candidates:
            raise KeyError(
                f"compiled scene artifact has no control for joint {runtime_joint_name!r}"
            )
        priority = {"position": 0, "direct": 1, "velocity": 2}
        best_priority = min(priority[control.mode] for control in candidates)
        preferred = [
            control for control in candidates if priority[control.mode] == best_priority
        ]
        if len(preferred) != 1:
            raise ValueError(
                f"compiled scene artifact has ambiguous controls for joint {runtime_joint_name!r}"
            )
        return preferred[0]

    def runtime_fingerprint(
        self,
        provider_name: str,
        provider_version: str,
        provider_config: Any,
    ) -> str:
        """Hash the artifact and normalized runtime contract for cache/session use."""

        name = str(provider_name).strip().lower()
        version = str(provider_version).strip()
        if not name or not version:
            raise ValueError("provider name and version are required for a runtime fingerprint")
        payload = {
            "artifact_schema": self.schema_version,
            "content_digest": self.content_digest,
            "producer": self.producer,
            "producer_version": self.producer_version,
            "artifact_contract": _normalized_json_value(
                {
                    "dimensions": self.dimensions,
                    "robots": [robot.to_mapping() for robot in self.robots],
                    "controls": [control.to_mapping() for control in self.controls],
                    "terrain_geom_groups": self.terrain_geom_groups,
                }
            ),
            "provider": name,
            "provider_version": version,
            "provider_config": _normalized_json_value(provider_config),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_compiled_artifact(
    artifact: Mapping[str, Any] | CompiledSceneArtifact,
    *,
    allow_current_compiler_bridge: bool = False,
) -> CompiledSceneArtifact:
    """Central validation entry point shared by all Python providers."""

    if isinstance(artifact, CompiledSceneArtifact):
        return CompiledSceneArtifact.from_mapping(artifact.to_mapping())
    if allow_current_compiler_bridge:
        return CompiledSceneArtifact.from_compiler_mapping(artifact)
    return CompiledSceneArtifact.from_mapping(artifact)


@dataclass(frozen=True)
class RobotBatchSpec:
    """Backend-neutral names selecting one robot from a compiled scene."""

    robot_name: str
    base_link: str
    joint_names: tuple[str, ...]
    link_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "robot_name", str(self.robot_name))
        object.__setattr__(self, "base_link", str(self.base_link))
        object.__setattr__(self, "joint_names", tuple(str(value) for value in self.joint_names))
        object.__setattr__(self, "link_names", tuple(str(value) for value in self.link_names))
        if not self.robot_name or not self.base_link:
            raise ValueError("robot_name and base_link must not be empty")
        _validate_unique_strings(self.joint_names, "robot joint name")
        _validate_unique_strings(self.link_names, "robot link name")


@dataclass(frozen=True)
class RobotBatchState:
    """Stable device tensors updated in place by :meth:`RobotBatchView.read_state`."""

    base_pose: Any
    base_velocity: Any
    joint_position: Any
    joint_velocity: Any
    joint_control: Any
    link_pose: Any


class RobotBatchView:
    """Provider-independent robot state, control, reset, and scene-sync view."""

    def __init__(self, provider: "BatchPhysicsProvider", spec: RobotBatchSpec, adapter: Any) -> None:
        self._provider = provider
        self.spec = spec
        self._adapter = adapter
        self._generation = getattr(provider, "generation", None)
        self._artifact_digest = getattr(getattr(provider, "artifact", None), "digest", None)
        self._state: RobotBatchState | None = None
        self._state_storage_signature: tuple[
            tuple[str, int, tuple[int, ...], str], ...
        ] | None = None
        self._scene_context: Any | None = None
        self._scene_links: tuple[Any, ...] | None = None

    def _validate(self) -> None:
        if getattr(self._provider, "generation", None) != self._generation:
            raise RuntimeError("robot batch view is stale because its provider was closed or rebuilt")
        digest = getattr(getattr(self._provider, "artifact", None), "digest", None)
        if digest != self._artifact_digest:
            raise RuntimeError("robot batch view cannot be used with a different compiled artifact")

    def read_state(self) -> RobotBatchState:
        self._validate()
        state = self._adapter.read_state(self._state)
        if not isinstance(state, RobotBatchState):
            raise RuntimeError("robot batch adapter returned an invalid state object")
        expected_shapes = {
            "base_pose": (int(self._provider.num_envs), 7),
            "base_velocity": (int(self._provider.num_envs), 6),
            "joint_position": (
                int(self._provider.num_envs),
                len(self.spec.joint_names),
            ),
            "joint_velocity": (
                int(self._provider.num_envs),
                len(self.spec.joint_names),
            ),
            "joint_control": (
                int(self._provider.num_envs),
                len(self.spec.joint_names),
            ),
            "link_pose": (
                int(self._provider.num_envs),
                len(self.spec.link_names),
                7,
            ),
        }
        signature = []
        for name, expected_shape in expected_shapes.items():
            value = getattr(state, name)
            shape = tuple(int(item) for item in getattr(value, "shape", ()))
            if shape != expected_shape:
                raise RuntimeError(
                    f"robot batch {name} tensor has shape {shape}, expected {expected_shape}"
                )
            data_ptr = getattr(value, "data_ptr", None)
            if callable(data_ptr):
                pointer = int(data_ptr())
            elif getattr(value, "ptr", None) is not None:
                pointer = int(value.ptr)
            else:
                pointer = id(value)
            array = getattr(value, "array", getattr(value, "_array", None))
            dtype = str(getattr(value, "dtype", getattr(array, "dtype", ""))).lower()
            signature.append((name, pointer, shape, dtype))
        resolved_signature = tuple(signature)
        if self._state_storage_signature is None:
            self._state_storage_signature = resolved_signature
        elif resolved_signature != self._state_storage_signature:
            raise GraphInvalidatedError(
                "robot batch state storage changed after its first read"
            )
        self._state = state
        return self._state

    def set_position_targets(self, targets: Any) -> None:
        self._validate()
        self._adapter.set_position_targets(targets)

    def set_base_pose_targets(self, targets: Any) -> None:
        """Set kinematic base targets as ``[x, y, z, qx, qy, qz, qw]``.

        Providers whose robot base is dynamic or fixed may decline this optional
        capability. Keeping it on the backend-neutral view lets manipulation
        examples move a kinematic hand without exposing solver storage.
        """

        self._validate()
        set_targets = getattr(self._adapter, "set_base_pose_targets", None)
        if not callable(set_targets):
            raise NotImplementedError(
                f"{type(self._provider).__name__} does not support kinematic base pose targets"
            )
        set_targets(targets)

    def set_controls(self, controls: Any) -> None:
        self._validate()
        self._adapter.set_controls(controls)

    def reset(self, reset_mask: Any, **state: Any) -> Mapping[str, Any]:
        self._validate()
        return self._adapter.reset(reset_mask, **state)

    def bind_scene(self, context: Any, links: Sequence[Any]) -> "RobotBatchView":
        if len(links) != len(self.spec.link_names):
            raise ValueError(
                f"scene sync requires {len(self.spec.link_names)} links, got {len(links)}"
            )
        self._scene_context = context
        self._scene_links = tuple(links)
        return self

    def sync_scene(
        self,
        context: Any | None = None,
        links: Sequence[Any] | None = None,
        *,
        env_index: int = 0,
    ) -> None:
        self._validate()
        context = self._scene_context if context is None else context
        links = self._scene_links if links is None else tuple(links)
        if context is None or links is None:
            raise RuntimeError("robot batch view has no bound scene context and links")
        if len(links) != len(self.spec.link_names):
            raise ValueError(
                f"scene sync requires {len(self.spec.link_names)} links, got {len(links)}"
            )
        try:
            resolved_env_index = operator.index(env_index)
        except TypeError as error:
            raise TypeError("scene sync environment index must be an integer") from error
        if isinstance(env_index, bool):
            raise TypeError("scene sync environment index must be an integer")
        if not 0 <= resolved_env_index < int(self._provider.num_envs):
            raise IndexError(
                f"scene sync environment index {env_index} is outside "
                f"[0, {self._provider.num_envs})"
            )
        state = self.read_state()
        poses = state.link_pose[resolved_env_index].detach().cpu().numpy()
        apply_poses = getattr(context, "apply_link_poses", None)
        if not callable(apply_poses):
            apply_poses = getattr(context, "_apply_link_pose_batch", None)
        if not callable(apply_poses):
            raise RuntimeError("this Gobot build does not expose AppContext.apply_link_poses")
        apply_poses(tuple(links), poses)


class BatchPhysicsProvider(ABC):
    """Stable lifecycle used by backend-specific batched physics providers."""

    accepts_device_actions = True

    def __enter__(self) -> "BatchPhysicsProvider":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.close()
        return False

    def create_robot_view(
        self,
        spec: RobotBatchSpec | None = None,
        **names: Any,
    ) -> RobotBatchView:
        """Resolve Gobot names once and return a backend-neutral batched view."""

        scene_context = names.pop("scene_context", None)
        scene_links = names.pop("scene_links", None)
        if spec is not None and names:
            raise TypeError("pass either RobotBatchSpec or keyword names, not both")
        if spec is None:
            spec = RobotBatchSpec(
                robot_name=str(names.pop("robot_name")),
                base_link=str(names.pop("base_link")),
                joint_names=tuple(str(value) for value in names.pop("joint_names")),
                link_names=tuple(str(value) for value in names.pop("link_names", ())),
            )
        if names:
            raise TypeError("unexpected robot view arguments: " + ", ".join(sorted(names)))
        self._last_robot_view_joint_count = len(spec.joint_names)
        view = RobotBatchView(self, spec, self._create_robot_view_adapter(spec))
        if scene_context is not None or scene_links is not None:
            if scene_context is None or scene_links is None:
                raise TypeError("scene_context and scene_links must be provided together")
            view.bind_scene(scene_context, scene_links)
        return view

    def _create_robot_view_adapter(self, spec: RobotBatchSpec) -> Any:
        raise NotImplementedError(f"{type(self).__name__} does not implement robot batch views")

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
    "CompiledControlTopology",
    "CompiledRobotTopology",
    "CompiledSceneArtifact",
    "GraphInvalidatedError",
    "ProviderUnavailableError",
    "RobotBatchSpec",
    "RobotBatchState",
    "RobotBatchView",
    "SimulationCapacityError",
    "validate_compiled_artifact",
]
