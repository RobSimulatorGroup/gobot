"""Batched MuJoCo Warp and libuipc co-simulation provider."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import operator
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ...ipc._artifact import (
    CompiledIpcSceneArtifact,
    validate_ipc_artifact,
)
from .base import (
    BatchPhysicsProvider,
    BatchProviderCapabilities,
    CompiledSceneArtifact,
    ProviderUnavailableError,
    RobotBatchSpec,
    validate_compiled_artifact,
)
from .mujoco_warp import MuJoCoWarpProvider


@contextmanager
def _nvtx_range(torch_runtime: Any, name: str):
    cuda = getattr(torch_runtime, "cuda", None)
    nvtx = getattr(cuda, "nvtx", None)
    pushed = False
    if nvtx is not None:
        try:
            if bool(cuda.is_available()):
                nvtx.range_push(name)
                pushed = True
        except Exception:
            pushed = False
    try:
        yield
    finally:
        if pushed:
            nvtx.range_pop()


_COLLISION_OWNERSHIP_V3 = MappingProxyType(
    {
        "rigid_rigid": "mujoco",
        "rigid_terrain": "mujoco",
        "deformable_deformable": "libuipc",
        "deformable_rigid": "libuipc",
        "deformable_terrain": "libuipc",
    }
)
_COLLISION_OWNERSHIP = MappingProxyType(
    {
        "rigid_rigid": "mujoco",
        "rigid_terrain": "mujoco",
        "deformable_deformable": "libuipc",
        "deformable_rigid": "libuipc",
        "deformable_static": "libuipc",
        "deformable_terrain": "unsupported",
    }
)


@dataclass(frozen=True)
class MuJoCoIpcBodyMapping:
    """One authored rigid link shared by MuJoCo and a libuipc proxy."""

    coupling_path: str
    ipc_path: str
    robot_name: str
    link_name: str
    mujoco_body_name: str
    ipc_body_index: int
    mode: str
    force_scale: float
    torque_scale: float

    def __post_init__(self) -> None:
        for name in (
            "coupling_path",
            "ipc_path",
            "robot_name",
            "link_name",
            "mujoco_body_name",
        ):
            value = str(getattr(self, name))
            if not value:
                raise ValueError(f"coupled body {name} must not be empty")
            object.__setattr__(self, name, value)
        index = operator.index(self.ipc_body_index)
        if isinstance(self.ipc_body_index, bool) or index < 0:
            raise ValueError("ipc_body_index must be non-negative")
        object.__setattr__(self, "ipc_body_index", index)
        mode = str(self.mode)
        if mode not in ("OneWay", "TwoWay"):
            raise ValueError("coupled body mode must be OneWay or TwoWay")
        object.__setattr__(self, "mode", mode)
        for name in ("force_scale", "torque_scale"):
            scale = float(getattr(self, name))
            if not math.isfinite(scale) or scale < 0.0:
                raise ValueError(f"coupled body {name} must be finite and non-negative")
            object.__setattr__(self, name, scale)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MuJoCoIpcBodyMapping":
        if not isinstance(value, Mapping):
            raise TypeError("coupled body mapping must be a mapping")
        return cls(
            coupling_path=str(value.get("coupling_path", "")),
            ipc_path=str(value.get("ipc_path", "")),
            robot_name=str(value.get("robot_name", "")),
            link_name=str(value.get("link_name", "")),
            mujoco_body_name=str(value.get("mujoco_body_name", "")),
            ipc_body_index=int(value.get("ipc_body_index", -1)),
            mode=str(value.get("mode", "")),
            force_scale=float(value.get("force_scale", float("nan"))),
            torque_scale=float(value.get("torque_scale", float("nan"))),
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "coupling_path": self.coupling_path,
            "ipc_path": self.ipc_path,
            "robot_name": self.robot_name,
            "link_name": self.link_name,
            "mujoco_body_name": self.mujoco_body_name,
            "ipc_body_index": self.ipc_body_index,
            "mode": self.mode,
            "force_scale": self.force_scale,
            "torque_scale": self.torque_scale,
        }


@dataclass(frozen=True)
class CompiledMuJoCoIpcArtifact:
    """Two runtime artifacts compiled from the same authored Gobot scene."""

    mujoco: CompiledSceneArtifact
    ipc: CompiledIpcSceneArtifact
    coupled_bodies: tuple[MuJoCoIpcBodyMapping, ...]
    schema_version: int = 4

    def __post_init__(self) -> None:
        schema_version = int(self.schema_version)
        if schema_version not in (3, 4):
            if schema_version == 1:
                raise ValueError(
                    "unsupported MuJoCo+IPC artifact schema 1; schema 3 requires "
                    "backend-neutral materials and explicit PhysicsCoupling entries"
                )
            raise ValueError(
                f"unsupported MuJoCo+IPC artifact schema {self.schema_version}; "
                "expected schema 3 or 4"
            )
        mujoco = validate_compiled_artifact(self.mujoco)
        ipc = validate_ipc_artifact(self.ipc)
        mappings = tuple(
            value
            if isinstance(value, MuJoCoIpcBodyMapping)
            else MuJoCoIpcBodyMapping.from_mapping(value)
            for value in self.coupled_bodies
        )
        if not mappings:
            raise ValueError(
                "MuJoCo+IPC artifact has no enabled PhysicsCoupling entries"
            )
        coupling_paths = tuple(mapping.coupling_path for mapping in mappings)
        paths = tuple(mapping.ipc_path for mapping in mappings)
        body_names = tuple(mapping.mujoco_body_name for mapping in mappings)
        indices = tuple(mapping.ipc_body_index for mapping in mappings)
        if len(set(coupling_paths)) != len(coupling_paths):
            raise ValueError("MuJoCo+IPC artifact has duplicate PhysicsCoupling paths")
        if len(set(paths)) != len(paths):
            raise ValueError("MuJoCo+IPC artifact has duplicate IPC body paths")
        if len(set(body_names)) != len(body_names):
            raise ValueError("MuJoCo+IPC artifact has duplicate MuJoCo body names")
        if indices != tuple(range(len(mappings))):
            raise ValueError(
                "MuJoCo+IPC body indices must be contiguous artifact-order indices"
            )
        expected = self._derive_mappings(mujoco, ipc)
        if mappings != expected:
            raise ValueError(
                "MuJoCo+IPC body mapping does not match the two compiled artifacts"
            )
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "mujoco", mujoco)
        object.__setattr__(self, "ipc", ipc)
        object.__setattr__(self, "coupled_bodies", mappings)

    @staticmethod
    def _derive_mappings(
        mujoco: CompiledSceneArtifact,
        ipc: CompiledIpcSceneArtifact,
    ) -> tuple[MuJoCoIpcBodyMapping, ...]:
        if not ipc.couplings:
            raise ValueError(
                "MuJoCo+IPC artifact requires at least one enabled PhysicsCoupling"
            )
        rigid_robots = {robot.name: robot for robot in mujoco.robots}
        result = []
        for coupling in ipc.couplings:
            robot_name = str(coupling["robot_name"])
            rigid_robot = rigid_robots.get(robot_name)
            if rigid_robot is None:
                raise ValueError(
                    f"PhysicsCoupling {coupling['coupling_path']!r} references "
                    f"rigid system {robot_name!r}, which has no matching "
                    "MuJoCo rigid system"
                )
            rigid_body_names = set(rigid_robot.body_names)
            link_name = str(coupling["link_name"])
            runtime_name = rigid_robot.runtime_prefix + link_name
            if runtime_name not in rigid_body_names:
                raise ValueError(
                    f"PhysicsCoupling {coupling['coupling_path']!r} maps to missing "
                    f"MuJoCo body {runtime_name!r}"
                )
            result.append(
                MuJoCoIpcBodyMapping(
                    coupling_path=str(coupling["coupling_path"]),
                    ipc_path=str(coupling["link_path"]),
                    robot_name=robot_name,
                    link_name=link_name,
                    mujoco_body_name=runtime_name,
                    ipc_body_index=int(coupling["proxy_index"]),
                    mode=str(coupling["mode"]),
                    force_scale=float(coupling["force_scale"]),
                    torque_scale=float(coupling["torque_scale"]),
                )
            )
        return tuple(result)

    @classmethod
    def from_artifacts(
        cls,
        mujoco: Mapping[str, Any] | CompiledSceneArtifact,
        ipc: Mapping[str, Any] | CompiledIpcSceneArtifact,
    ) -> "CompiledMuJoCoIpcArtifact":
        rigid_artifact = validate_compiled_artifact(
            mujoco, allow_current_compiler_bridge=True
        )
        ipc_artifact = validate_ipc_artifact(ipc)
        return cls(
            mujoco=rigid_artifact,
            ipc=ipc_artifact,
            coupled_bodies=cls._derive_mappings(rigid_artifact, ipc_artifact),
        )

    @classmethod
    def from_context(cls, context: Any) -> "CompiledMuJoCoIpcArtifact":
        compile_mujoco = getattr(context, "compile_scene_artifact", None)
        compile_ipc = getattr(context, "compile_ipc_scene_artifact", None)
        if not callable(compile_mujoco) or not callable(compile_ipc):
            raise RuntimeError(
                "Gobot AppContext must expose both MuJoCo and IPC artifact compilers"
            )
        return cls.from_artifacts(compile_mujoco(), compile_ipc())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CompiledMuJoCoIpcArtifact":
        if not isinstance(value, Mapping):
            raise TypeError("MuJoCo+IPC artifact must be a mapping")
        supplied_schema = int(value.get("schema_version", 0))
        if supplied_schema not in (3, 4):
            if supplied_schema < 3:
                raise ValueError(
                    f"unsupported MuJoCo+IPC artifact schema {supplied_schema}; "
                    "schema 3 requires backend-neutral materials and explicit PhysicsCoupling mapping"
                )
            raise ValueError(
                "unsupported MuJoCo+IPC artifact schema; expected schema 3 or 4"
            )
        artifact = cls(
            mujoco=validate_compiled_artifact(value.get("mujoco", {})),
            ipc=validate_ipc_artifact(value.get("ipc", {})),
            coupled_bodies=tuple(
                MuJoCoIpcBodyMapping.from_mapping(item)
                for item in value.get("coupled_bodies", ())
            ),
            schema_version=supplied_schema,
        )
        supplied_digest = str(value.get("digest", ""))
        if supplied_digest and supplied_digest != artifact.digest:
            raise ValueError("MuJoCo+IPC artifact digest mismatch")
        ownership = value.get("collision_ownership")
        expected_ownership = (
            _COLLISION_OWNERSHIP_V3
            if supplied_schema == 3
            else _COLLISION_OWNERSHIP
        )
        if ownership is not None and dict(ownership) != dict(expected_ownership):
            raise ValueError("MuJoCo+IPC artifact has invalid collision ownership")
        return artifact

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mujoco": self.mujoco.to_mapping(),
            "ipc": self.ipc.to_mapping(),
            "coupled_bodies": [
                mapping.to_mapping() for mapping in self.coupled_bodies
            ],
            "collision_ownership": dict(self.collision_ownership),
            "digest": self.digest,
        }

    @property
    def collision_ownership(self) -> Mapping[str, str]:
        return (
            _COLLISION_OWNERSHIP_V3
            if self.schema_version == 3
            else _COLLISION_OWNERSHIP
        )

    @property
    def digest(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "mujoco": self.mujoco.digest,
            "ipc": self.ipc.digest,
            "coupled_bodies": [
                mapping.to_mapping() for mapping in self.coupled_bodies
            ],
            "collision_ownership": dict(self.collision_ownership),
        }
        return "sha256:" + hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()


def validate_mujoco_ipc_artifact(
    artifact: Mapping[str, Any] | CompiledMuJoCoIpcArtifact,
) -> CompiledMuJoCoIpcArtifact:
    if isinstance(artifact, CompiledMuJoCoIpcArtifact):
        return CompiledMuJoCoIpcArtifact.from_mapping(artifact.to_mapping())
    return CompiledMuJoCoIpcArtifact.from_mapping(artifact)


@dataclass(frozen=True)
class MuJoCoIpcConfig:
    """Fixed topology and coupling parameters for MuJoCo+libuipc."""

    num_envs: int = 256
    device: str = "cuda:0"
    environments_per_shard: int = 64
    force_scale: float = 1.0
    torque_scale: float = 1.0
    rigid_substeps: int = 1
    ipc_substeps: int = 1
    coupling_iterations: int = 2
    relaxation_mode: str | None = None
    relaxation_factor: float = 1.0
    relaxation_min: float = 0.1
    relaxation_max: float = 1.0
    capture_mujoco_graphs: bool = True
    capture_coupler_graphs: bool = True

    def __post_init__(self) -> None:
        for name in (
            "num_envs",
            "environments_per_shard",
            "rigid_substeps",
            "ipc_substeps",
            "coupling_iterations",
        ):
            try:
                value = operator.index(getattr(self, name))
            except TypeError as error:
                raise TypeError(f"{name} must be an integer") from error
            if isinstance(getattr(self, name), bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.num_envs % self.environments_per_shard != 0:
            raise ValueError(
                "num_envs must be divisible by environments_per_shard"
            )
        for name in ("force_scale", "torque_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        relaxation_mode = self.relaxation_mode
        if relaxation_mode is None:
            relaxation_mode = "aitken" if self.coupling_iterations > 1 else "fixed"
        relaxation_mode = str(relaxation_mode)
        if relaxation_mode not in ("fixed", "aitken"):
            raise ValueError("relaxation_mode must be 'fixed' or 'aitken'")
        object.__setattr__(self, "relaxation_mode", relaxation_mode)
        for name in (
            "relaxation_factor",
            "relaxation_min",
            "relaxation_max",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.relaxation_min > self.relaxation_max:
            raise ValueError("relaxation_min must not exceed relaxation_max")
        if not self.relaxation_min <= self.relaxation_factor <= self.relaxation_max:
            raise ValueError(
                "relaxation_factor must be within relaxation_min and relaxation_max"
            )
        if self.rigid_substeps != self.ipc_substeps:
            raise ValueError(
                "SolverCoupledProxy requires rigid_substeps == ipc_substeps"
            )
        device = str(self.device)
        if not device:
            raise ValueError("device must not be empty")
        object.__setattr__(self, "device", device)
        object.__setattr__(
            self, "capture_mujoco_graphs", bool(self.capture_mujoco_graphs)
        )
        object.__setattr__(
            self, "capture_coupler_graphs", bool(self.capture_coupler_graphs)
        )

    @property
    def shard_count(self) -> int:
        return self.num_envs // self.environments_per_shard


class SolverCoupledProxy:
    """Staggered rollback coupling between MuJoCo and libuipc.

    MuJoCo owns rigid state, libuipc owns deformable state and contact, and
    authored mappings own the proxy exchange. OneWay mappings transfer only
    kinematics; TwoWay mappings also return the relaxed affine wrench.
    """

    def __init__(
        self,
        rigid_solver: Any,
        ipc_solver: Any,
        mappings: Sequence[MuJoCoIpcBodyMapping],
        *,
        force_scale: float = 1.0,
        torque_scale: float = 1.0,
        coupling_iterations: int = 2,
        relaxation_mode: str | None = None,
        relaxation_factor: float = 1.0,
        relaxation_min: float = 0.1,
        relaxation_max: float = 1.0,
        capture_graphs: bool = True,
    ) -> None:
        self.rigid_solver = rigid_solver
        self.ipc_solver = ipc_solver
        self.mappings = tuple(mappings)
        for name, value in (
            ("force_scale", force_scale),
            ("torque_scale", torque_scale),
        ):
            numeric_value = float(value)
            if not math.isfinite(numeric_value) or numeric_value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            if name == "force_scale":
                force_scale = numeric_value
            else:
                torque_scale = numeric_value
        for name, value in (("coupling_iterations", coupling_iterations),):
            try:
                count = operator.index(value)
            except TypeError as error:
                raise TypeError(f"{name} must be an integer") from error
            if isinstance(value, bool) or count <= 0:
                raise ValueError(f"{name} must be positive")
            coupling_iterations = count
        if relaxation_mode is None:
            relaxation_mode = "aitken" if coupling_iterations > 1 else "fixed"
        relaxation_mode = str(relaxation_mode)
        if relaxation_mode not in ("fixed", "aitken"):
            raise ValueError("relaxation_mode must be 'fixed' or 'aitken'")
        relaxation_values = {
            "relaxation_factor": float(relaxation_factor),
            "relaxation_min": float(relaxation_min),
            "relaxation_max": float(relaxation_max),
        }
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in relaxation_values.values()
        ):
            raise ValueError("relaxation parameters must be finite and positive")
        if not (
            relaxation_values["relaxation_min"]
            <= relaxation_values["relaxation_factor"]
            <= relaxation_values["relaxation_max"]
        ):
            raise ValueError(
                "relaxation_factor must be within relaxation_min and relaxation_max"
            )
        self._torch = getattr(rigid_solver, "_torch", None)
        if self._torch is None:
            raise TypeError("rigid solver must expose its Torch runtime")
        rigid_arrays = rigid_solver.arrays
        ipc_arrays = ipc_solver.arrays
        for name in ("xpos", "xmat", "xfrc_applied"):
            if name not in rigid_arrays:
                raise RuntimeError(f"rigid solver has no required array {name!r}")
        for name in ("affine_targets", "affine_contact_wrenches"):
            if name not in ipc_arrays:
                raise RuntimeError(f"IPC solver has no required array {name!r}")

        self._xpos = rigid_arrays["xpos"]
        self._xmat = rigid_arrays["xmat"]
        self._xfrc = rigid_arrays["xfrc_applied"]
        self._targets = ipc_arrays["affine_targets"]
        self._target_twists = ipc_arrays.get("affine_target_twists")
        self._wrench_source = str(
            getattr(ipc_solver, "wrench_source", "direct")
        )
        if self._wrench_source not in ("direct", "pose_error"):
            raise RuntimeError(
                f"unsupported IPC affine wrench source {self._wrench_source!r}"
            )
        self._proxy_transforms = ipc_arrays.get("affine_transforms")
        self._native_wrenches = ipc_arrays["affine_contact_wrenches"]
        self._source_wrenches = self._native_wrenches
        self._num_envs = int(rigid_solver.num_envs)
        self._body_count = len(self.mappings)
        expected_targets = (self._num_envs, self._body_count, 4, 4)
        expected_wrenches = (self._num_envs, self._body_count, 6)
        if tuple(self._targets.shape) != expected_targets:
            raise RuntimeError(
                f"IPC affine target storage has shape {tuple(self._targets.shape)}, "
                f"expected {expected_targets}"
            )
        if tuple(self._native_wrenches.shape) != expected_wrenches:
            raise RuntimeError(
                "IPC affine contact wrench storage has shape "
                f"{tuple(self._native_wrenches.shape)}, expected {expected_wrenches}"
            )
        if self._xpos.device != self._targets.device:
            raise RuntimeError(
                "MuJoCo and libuipc coupling tensors must be on the same device"
            )
        if (
            not self._targets.is_contiguous()
            or not self._native_wrenches.is_contiguous()
        ):
            raise RuntimeError("IPC coupling tensors must be contiguous")

        body_names = tuple(mapping.mujoco_body_name for mapping in self.mappings)
        body_ids = rigid_solver.resolve_object_ids("body", body_names)
        self._body_id_values = tuple(int(value) for value in body_ids)
        self._body_ids = self._torch.as_tensor(
            self._body_id_values,
            dtype=self._torch.long,
            device=self._xpos.device,
        )
        self._positions = self._torch.empty(
            (self._num_envs, self._body_count, 3),
            dtype=self._xpos.dtype,
            device=self._xpos.device,
        )
        rotation_tail = tuple(self._xmat.shape[2:])
        if rotation_tail not in ((9,), (3, 3)):
            raise RuntimeError(
                "MuJoCo xmat must have shape [env, body, 9] or "
                f"[env, body, 3, 3], got {tuple(self._xmat.shape)}"
            )
        self._rotations = self._torch.empty(
            (self._num_envs, self._body_count, *rotation_tail),
            dtype=self._xmat.dtype,
            device=self._xmat.device,
        )
        self._selected_wrenches = self._torch.empty(
            expected_wrenches, dtype=self._xfrc.dtype, device=self._xfrc.device
        )
        self._next_wrenches = self._torch.empty_like(self._selected_wrenches)
        self._applied_wrenches = self._torch.zeros_like(self._selected_wrenches)
        self._force_scale = float(force_scale)
        self._torque_scale = float(torque_scale)
        self._coupling_iterations = int(coupling_iterations)
        self._uses_rollback = self._coupling_iterations > 1
        self._relaxation_mode = relaxation_mode
        self._relaxation_factor = relaxation_values["relaxation_factor"]
        self._relaxation_min = relaxation_values["relaxation_min"]
        self._relaxation_max = relaxation_values["relaxation_max"]
        self._feedback_mask = self._torch.as_tensor(
            [mapping.mode == "TwoWay" for mapping in self.mappings],
            dtype=self._xfrc.dtype,
            device=self._xfrc.device,
        ).reshape(1, self._body_count, 1)
        self._force_scales = self._torch.as_tensor(
            [self._force_scale * mapping.force_scale for mapping in self.mappings],
            dtype=self._xfrc.dtype,
            device=self._xfrc.device,
        ).reshape(1, self._body_count, 1)
        self._torque_scales = self._torch.as_tensor(
            [self._torque_scale * mapping.torque_scale for mapping in self.mappings],
            dtype=self._xfrc.dtype,
            device=self._xfrc.device,
        ).reshape(1, self._body_count, 1)
        self._phase = "Idle"
        self._completed_steps = 0
        self._last_coupling_iterations = 0
        self._last_interface_residual_device = self._torch.zeros(
            (), dtype=self._xfrc.dtype, device=self._xfrc.device
        )
        self._last_relaxation_coefficient_device = self._torch.full(
            (),
            self._relaxation_factor,
            dtype=self._xfrc.dtype,
            device=self._xfrc.device,
        )
        self._capture_graphs_requested = bool(capture_graphs)
        self._cuda_graphs: dict[str, Any] = {}
        self._cuda_graph_capture_reason = "coupler graph capture not initialized"
        self._phase_latency_ms: dict[str, float] = {}
        if self._wrench_source == "pose_error":
            self._initialize_pose_error_feedback()
        self._initialize_proxy(rigid_arrays, ipc_arrays)
        self._storage_signature = self._capture_storage_signature()

        self._targets.zero_()
        self._targets[..., 3, 3] = 1.0

    def _capture_storage_signature(self) -> tuple[tuple[str, int], ...]:
        values = [
            ("xpos", self._xpos),
            ("xmat", self._xmat),
            ("xfrc_applied", self._xfrc),
            ("affine_targets", self._targets),
            ("affine_contact_wrenches", self._native_wrenches),
            ("body_ids", self._body_ids),
            ("positions", self._positions),
            ("rotations", self._rotations),
            ("selected_wrenches", self._selected_wrenches),
            ("next_wrenches", self._next_wrenches),
            ("applied_wrenches", self._applied_wrenches),
            ("feedback_mask", self._feedback_mask),
            ("force_scales", self._force_scales),
            ("torque_scales", self._torque_scales),
            ("last_interface_residual", self._last_interface_residual_device),
            (
                "last_relaxation_coefficient",
                self._last_relaxation_coefficient_device,
            ),
        ]
        if self._wrench_source == "pose_error":
            values.extend(
                (
                    ("affine_transforms", self._proxy_transforms),
                    ("mass_factor", self._mass_factor),
                    ("center_of_mass", self._center_of_mass),
                    ("gravity_force", self._gravity_force),
                    ("inertia", self._inertia),
                    ("computed_wrenches", self._computed_wrenches),
                    ("relative_rotation", self._relative_rotation),
                    ("rotation_error", self._rotation_error),
                    ("local_rotation_error", self._local_rotation_error),
                    ("local_torque", self._local_torque),
                    ("world_torque", self._world_torque),
                    ("world_center", self._world_center),
                    (
                        "center_acceleration_force",
                        self._center_acceleration_force,
                    ),
                )
            )
        values.extend(
            (
                ("subtree_com", self._subtree_com),
                ("root_body_ids", self._root_body_ids),
                ("cvel", self._cvel),
                ("affine_target_twists", self._target_twists),
                ("com_positions", self._com_positions),
                ("spatial_velocities", self._spatial_velocities),
                ("origin_twists", self._origin_twists),
                ("origin_minus_com", self._origin_minus_com),
                ("angular_cross_offset", self._angular_cross_offset),
                ("iteration_guess", self._iteration_guess),
                ("iteration_feedback", self._iteration_feedback),
                ("iteration_residual", self._iteration_residual),
                ("previous_residual", self._previous_residual),
                ("aitken_delta", self._aitken_delta),
                ("aitken_factors", self._aitken_factors),
                ("aitken_product", self._aitken_product),
                ("aitken_squared", self._aitken_squared),
                ("aitken_numerator", self._aitken_numerator),
                ("aitken_denominator", self._aitken_denominator),
                ("aitken_safe_denominator", self._aitken_safe_denominator),
                ("aitken_updated", self._aitken_updated),
                ("aitken_previous_factors", self._aitken_previous_factors),
                ("aitken_valid", self._aitken_valid),
                ("interface_residual_norms", self._interface_residual_norms),
                ("two_way_body_ids", self._two_way_body_ids),
                ("two_way_aitken_factors", self._two_way_aitken_factors),
            )
        )
        if self._uses_rollback:
            values.append(
                ("checkpoint_applied_wrenches", self._checkpoint_applied_wrenches)
            )
            values.extend(
                (f"rigid_checkpoint:{name}", value)
                for name, value in self._rigid_checkpoint_arrays.items()
            )
        return tuple((name, int(value.data_ptr())) for name, value in values)

    def _initialize_proxy(
        self, rigid_arrays: Mapping[str, Any], ipc_arrays: Mapping[str, Any]
    ) -> None:
        if self._target_twists is None:
            raise RuntimeError(
                "SolverCoupledProxy requires IPC affine_target_twists storage"
            )
        expected_twists = (self._num_envs, self._body_count, 6)
        if tuple(self._target_twists.shape) != expected_twists:
            raise RuntimeError(
                "IPC affine target twist storage has shape "
                f"{tuple(self._target_twists.shape)}, expected {expected_twists}"
            )
        if not self._target_twists.is_contiguous():
            raise RuntimeError("IPC affine target twists must be contiguous")
        for name in ("subtree_com", "cvel"):
            if name not in rigid_arrays:
                raise RuntimeError(
                    f"SolverCoupledProxy requires MuJoCo array {name!r}"
                )
        self._subtree_com = rigid_arrays["subtree_com"]
        self._cvel = rigid_arrays["cvel"]
        if tuple(self._subtree_com.shape[:2]) != tuple(
            self._xpos.shape[:2]
        ) or tuple(
            self._subtree_com.shape[2:]
        ) != (3,):
            raise RuntimeError("MuJoCo subtree_com must have shape [env, body, 3]")
        if tuple(self._cvel.shape[:2]) != tuple(self._xpos.shape[:2]) or tuple(
            self._cvel.shape[2:]
        ) != (6,):
            raise RuntimeError("MuJoCo cvel must have shape [env, body, 6]")
        resolve_root_ids = getattr(self.rigid_solver, "resolve_body_root_ids", None)
        if callable(resolve_root_ids):
            root_body_ids = tuple(resolve_root_ids(self._body_id_values))
        else:
            model = getattr(self.rigid_solver, "_mj_model", None)
            model_root_ids = getattr(model, "body_rootid", None)
            if model_root_ids is None:
                raise RuntimeError(
                    "SolverCoupledProxy requires MuJoCo body_rootid metadata"
                )
            root_body_ids = tuple(
                int(model_root_ids[body_id]) for body_id in self._body_id_values
            )
        if len(root_body_ids) != self._body_count:
            raise RuntimeError("MuJoCo body root-id lookup returned an invalid result")
        if any(
            isinstance(value, bool)
            or int(value) < 0
            or int(value) >= int(self._subtree_com.shape[1])
            for value in root_body_ids
        ):
            raise RuntimeError("MuJoCo body root-id lookup returned an invalid result")
        self._root_body_ids = self._torch.as_tensor(
            root_body_ids, dtype=self._torch.long, device=self._xpos.device
        )
        if self._uses_rollback:
            for name in (
                "capture_checkpoint",
                "rewind_checkpoint",
                "commit_checkpoint",
            ):
                if not callable(getattr(self.ipc_solver, name, None)):
                    raise RuntimeError(
                        f"SolverCoupledProxy x2+ requires IPC {name}() support"
                    )

        self._com_positions = self._torch.empty_like(self._positions)
        self._spatial_velocities = self._torch.empty(
            (self._num_envs, self._body_count, 6),
            dtype=self._cvel.dtype,
            device=self._cvel.device,
        )
        self._origin_twists = self._torch.empty(
            expected_twists,
            dtype=self._target_twists.dtype,
            device=self._target_twists.device,
        )
        self._origin_minus_com = self._torch.empty_like(self._positions)
        self._angular_cross_offset = self._torch.empty_like(self._positions)
        self._checkpoint_applied_wrenches = (
            self._torch.empty_like(self._applied_wrenches)
            if self._uses_rollback
            else None
        )
        self._iteration_guess = self._torch.empty_like(self._applied_wrenches)
        self._iteration_feedback = self._torch.empty_like(
            self._applied_wrenches
        )
        self._iteration_residual = self._torch.empty_like(
            self._applied_wrenches
        )
        self._previous_residual = self._torch.empty_like(
            self._applied_wrenches
        )
        self._aitken_delta = self._torch.empty_like(self._applied_wrenches)
        self._aitken_factors = self._torch.full(
            (self._num_envs, self._body_count, 1),
            self._relaxation_factor,
            dtype=self._applied_wrenches.dtype,
            device=self._applied_wrenches.device,
        )
        self._aitken_product = self._torch.empty_like(
            self._iteration_residual
        )
        self._aitken_squared = self._torch.empty_like(
            self._iteration_residual
        )
        self._aitken_numerator = self._torch.empty_like(
            self._aitken_factors
        )
        self._aitken_denominator = self._torch.empty_like(
            self._aitken_factors
        )
        self._aitken_safe_denominator = self._torch.empty_like(
            self._aitken_factors
        )
        self._aitken_updated = self._torch.empty_like(self._aitken_factors)
        self._aitken_previous_factors = self._torch.empty_like(
            self._aitken_factors
        )
        self._aitken_valid = self._torch.empty_like(
            self._aitken_factors, dtype=self._torch.bool
        )
        self._interface_residual_norms = self._torch.empty(
            (self._num_envs, self._body_count),
            dtype=self._iteration_residual.dtype,
            device=self._iteration_residual.device,
        )
        two_way_body_ids = tuple(
            index
            for index, mapping in enumerate(self.mappings)
            if mapping.mode == "TwoWay"
        )
        self._two_way_body_ids = self._torch.as_tensor(
            two_way_body_ids,
            dtype=self._torch.long,
            device=self._xfrc.device,
        )
        self._two_way_aitken_factors = self._torch.empty(
            (self._num_envs, len(two_way_body_ids), 1),
            dtype=self._aitken_factors.dtype,
            device=self._aitken_factors.device,
        )

        checkpoint_names = []
        names_method = getattr(self.rigid_solver, "_checkpoint_array_names", None)
        if self._uses_rollback and callable(names_method):
            checkpoint_names.extend(str(name) for name in names_method())
        if self._uses_rollback and not checkpoint_names:
            checkpoint_names.extend(
                (
                    "time",
                    "qpos",
                    "qvel",
                    "act",
                    "qacc_warmstart",
                    "ctrl",
                    "qfrc_applied",
                    "xfrc_applied",
                    "mocap_pos",
                    "mocap_quat",
                    "eq_active",
                    "userdata",
                    "plugin_state",
                )
            )
        unique_names = tuple(dict.fromkeys(checkpoint_names))
        self._rigid_checkpoint_arrays = {
            name: self._torch.empty_like(rigid_arrays[name])
            for name in unique_names
            if name in rigid_arrays
            and callable(getattr(rigid_arrays[name], "copy_", None))
        }
        if self._uses_rollback:
            for required in ("qpos", "qvel", "xfrc_applied"):
                if required not in self._rigid_checkpoint_arrays:
                    raise RuntimeError(
                        "SolverCoupledProxy cannot checkpoint MuJoCo array "
                        f"{required!r}"
                    )
        self._rigid_checkpoint_counters: dict[str, int] = {}
        self._rigid_counter_names = tuple(
            name
            for name in ("_step_count", "step_count")
            if isinstance(getattr(self.rigid_solver, name, None), int)
        )

    def _capture_rigid_checkpoint(self) -> None:
        self._replay_or_call(
            "checkpoint", self._capture_rigid_checkpoint_device
        )
        self._rigid_checkpoint_counters = {
            name: int(getattr(self.rigid_solver, name))
            for name in self._rigid_counter_names
        }

    def _capture_rigid_checkpoint_device(self) -> None:
        rigid_arrays = self.rigid_solver.arrays
        for name, destination in self._rigid_checkpoint_arrays.items():
            destination.copy_(rigid_arrays[name])
        self._checkpoint_applied_wrenches.copy_(self._applied_wrenches)

    def _rewind_rigid_checkpoint(self) -> None:
        self._replay_or_call("rewind", self._rewind_rigid_checkpoint_device)
        for name, value in self._rigid_checkpoint_counters.items():
            setattr(self.rigid_solver, name, value)
        forward = getattr(self.rigid_solver, "forward", None)
        if callable(forward):
            forward()

    def _rewind_rigid_checkpoint_device(self) -> None:
        rigid_arrays = self.rigid_solver.arrays
        for name, source in self._rigid_checkpoint_arrays.items():
            rigid_arrays[name].copy_(source)
        self._applied_wrenches.copy_(self._checkpoint_applied_wrenches)

    def _gather_rigid_twist(self) -> None:
        self._torch.index_select(
            self._subtree_com, 1, self._root_body_ids, out=self._com_positions
        )
        self._torch.index_select(
            self._cvel, 1, self._body_ids, out=self._spatial_velocities
        )
        angular = self._spatial_velocities[..., :3]
        linear_at_com = self._spatial_velocities[..., 3:]
        self._origin_minus_com.copy_(self._positions).sub_(self._com_positions)
        self._torch.cross(
            angular,
            self._origin_minus_com,
            dim=-1,
            out=self._angular_cross_offset,
        )
        self._origin_twists[..., :3].copy_(linear_at_com)
        self._origin_twists[..., :3].add_(self._angular_cross_offset)
        self._origin_twists[..., 3:].copy_(angular)
        self._target_twists.copy_(self._origin_twists)

    def _initialize_pose_error_feedback(self) -> None:
        if self._proxy_transforms is None or tuple(self._proxy_transforms.shape) != (
            self._num_envs,
            self._body_count,
            4,
            4,
        ):
            raise RuntimeError(
                "pose-error IPC feedback requires affine_transforms with the "
                "same shape as affine_targets"
            )
        properties = tuple(self.ipc_solver.affine_bodies)
        if len(properties) != self._body_count:
            raise RuntimeError(
                "pose-error IPC feedback requires one mass/inertia record per proxy"
            )
        dtype = self._targets.dtype
        device = self._targets.device
        masses = []
        centers = []
        inertias = []
        for body in properties:
            if "mass" not in body or "inertia_diagonal" not in body:
                raise RuntimeError(
                    "pose-error IPC feedback requires proxy mass and inertia"
                )
            masses.append(float(body["mass"]))
            center = tuple(
                float(value)
                for value in body.get("center_of_mass", (0.0, 0.0, 0.0))
            )
            if len(center) != 3:
                raise RuntimeError(
                    "IPC proxy center of mass must contain three components"
                )
            centers.append(center)
            diagonal = tuple(float(value) for value in body["inertia_diagonal"])
            off_diagonal = tuple(
                float(value)
                for value in body.get("inertia_off_diagonal", (0.0, 0.0, 0.0))
            )
            if len(diagonal) != 3 or len(off_diagonal) != 3:
                raise RuntimeError("IPC proxy inertia must contain three components")
            xy, xz, yz = off_diagonal
            inertias.append(
                (
                    (diagonal[0], xy, xz),
                    (xy, diagonal[1], yz),
                    (xz, yz, diagonal[2]),
                )
            )
        inverse_dt_squared = 1.0 / float(self.ipc_solver.fixed_time_step) ** 2
        self._mass_factor = self._torch.as_tensor(
            masses, dtype=dtype, device=device
        ).reshape(1, self._body_count, 1)
        self._mass_factor.mul_(inverse_dt_squared)
        self._center_of_mass = self._torch.as_tensor(
            centers, dtype=dtype, device=device
        ).reshape(1, self._body_count, 3, 1)
        gravity = tuple(
            float(value)
            for value in getattr(self.ipc_solver, "gravity", (0.0, 0.0, 0.0))
        )
        if len(gravity) != 3 or not all(math.isfinite(value) for value in gravity):
            raise RuntimeError("IPC solver gravity must contain three finite values")
        mass = self._torch.as_tensor(
            masses, dtype=dtype, device=device
        ).reshape(1, self._body_count, 1)
        gravity_vector = self._torch.as_tensor(
            gravity, dtype=dtype, device=device
        ).reshape(1, 1, 3)
        self._gravity_force = mass * gravity_vector
        inertia = self._torch.as_tensor(inertias, dtype=dtype, device=device)
        self._inertia = (
            inertia.unsqueeze(0)
            .expand(self._num_envs, -1, -1, -1)
            .contiguous()
        )
        self._inertia.mul_(inverse_dt_squared)
        self._computed_wrenches = self._torch.empty(
            (self._num_envs, self._body_count, 6),
            dtype=dtype,
            device=device,
        )
        self._relative_rotation = self._torch.empty(
            (self._num_envs, self._body_count, 3, 3),
            dtype=dtype,
            device=device,
        )
        self._rotation_error = self._torch.empty(
            (self._num_envs, self._body_count, 3, 1),
            dtype=dtype,
            device=device,
        )
        self._local_rotation_error = self._torch.empty_like(self._rotation_error)
        self._local_torque = self._torch.empty_like(self._rotation_error)
        self._world_torque = self._torch.empty_like(self._rotation_error)
        self._world_center = self._torch.empty_like(self._rotation_error)
        self._center_acceleration_force = self._torch.empty(
            (self._num_envs, self._body_count, 3),
            dtype=dtype,
            device=device,
        )
        self._source_wrenches = self._computed_wrenches

    def _compute_pose_error_wrenches(self) -> None:
        self._computed_wrenches[..., :3].copy_(
            self._proxy_transforms[..., :3, 3]
        )
        self._computed_wrenches[..., :3].sub_(self._targets[..., :3, 3])
        self._computed_wrenches[..., :3].mul_(self._mass_factor)

        proxy_rotation = self._proxy_transforms[..., :3, :3]
        target_rotation = self._targets[..., :3, :3]
        self._torch.matmul(
            proxy_rotation,
            target_rotation.transpose(-1, -2),
            out=self._relative_rotation,
        )
        rotation = self._relative_rotation
        error = self._rotation_error[..., 0]
        error[..., 0].copy_(rotation[..., 2, 1]).sub_(rotation[..., 1, 2])
        error[..., 1].copy_(rotation[..., 0, 2]).sub_(rotation[..., 2, 0])
        error[..., 2].copy_(rotation[..., 1, 0]).sub_(rotation[..., 0, 1])
        error.mul_(0.5)
        self._torch.matmul(
            target_rotation,
            self._center_of_mass,
            out=self._world_center,
        )
        self._torch.cross(
            error,
            self._world_center[..., 0],
            dim=-1,
            out=self._center_acceleration_force,
        )
        self._center_acceleration_force.mul_(self._mass_factor)
        self._computed_wrenches[..., :3].add_(
            self._center_acceleration_force
        )
        self._computed_wrenches[..., :3].sub_(self._gravity_force)
        self._torch.matmul(
            target_rotation.transpose(-1, -2),
            self._rotation_error,
            out=self._local_rotation_error,
        )
        self._torch.matmul(
            self._inertia,
            self._local_rotation_error,
            out=self._local_torque,
        )
        self._torch.matmul(
            target_rotation,
            self._local_torque,
            out=self._world_torque,
        )
        # MuJoCo applies xfrc_applied at xipos, the body center of mass.
        self._computed_wrenches[..., 3:].copy_(self._world_torque[..., 0])

    def validate_storage(self) -> None:
        if self._capture_storage_signature() != self._storage_signature:
            raise RuntimeError(
                "MuJoCo+IPC coupling storage changed after construction"
            )

    def _gather_rigid_pose_unchecked(self) -> None:
        self._torch.index_select(
            self._xpos, 1, self._body_ids, out=self._positions
        )
        self._torch.index_select(
            self._xmat, 1, self._body_ids, out=self._rotations
        )
        self._targets[..., :3, :3].copy_(
            self._rotations.reshape(self._num_envs, self._body_count, 3, 3)
        )
        self._targets[..., :3, 3].copy_(self._positions)

    def _gather_rigid_kinematics(self) -> None:
        self._replay_or_call(
            "gather_kinematics", self._gather_rigid_kinematics_unchecked
        )

    def _gather_rigid_kinematics_unchecked(self) -> None:
        self._gather_rigid_pose_unchecked()
        self._gather_rigid_twist()

    def _compute_scaled_feedback(self, destination: Any) -> None:
        if self._wrench_source == "pose_error":
            self._compute_pose_error_wrenches()
        destination.copy_(self._source_wrenches)
        destination[..., :3].mul_(self._force_scales)
        destination[..., 3:].mul_(self._torque_scales)
        destination.mul_(self._feedback_mask)

    def _replace_owned_wrenches(self, wrenches: Any) -> None:
        self._replace_owned_wrenches_unchecked(wrenches)

    def _replace_owned_wrenches_unchecked(self, wrenches: Any) -> None:
        self._torch.index_select(
            self._xfrc, 1, self._body_ids, out=self._selected_wrenches
        )
        self._selected_wrenches.sub_(self._applied_wrenches)
        self._selected_wrenches.add_(wrenches)
        self._xfrc.index_copy_(1, self._body_ids, self._selected_wrenches)
        self._applied_wrenches.copy_(wrenches)

    def _update_relaxed_wrench(self, iteration: int) -> None:
        self._iteration_residual.copy_(self._iteration_feedback)
        self._iteration_residual.sub_(self._iteration_guess)
        if self._relaxation_mode == "aitken" and iteration > 0:
            self._aitken_delta.copy_(self._iteration_residual)
            self._aitken_delta.sub_(self._previous_residual)
            self._torch.mul(
                self._previous_residual,
                self._aitken_delta,
                out=self._aitken_product,
            )
            self._torch.sum(
                self._aitken_product,
                dim=-1,
                keepdim=True,
                out=self._aitken_numerator,
            )
            self._torch.mul(
                self._aitken_delta,
                self._aitken_delta,
                out=self._aitken_squared,
            )
            self._torch.sum(
                self._aitken_squared,
                dim=-1,
                keepdim=True,
                out=self._aitken_denominator,
            )
            epsilon = self._torch.finfo(self._aitken_denominator.dtype).eps
            self._aitken_previous_factors.copy_(self._aitken_factors)
            self._aitken_safe_denominator.copy_(self._aitken_denominator)
            self._aitken_safe_denominator.clamp_min_(epsilon)
            self._aitken_updated.copy_(self._aitken_factors)
            self._aitken_updated.mul_(self._aitken_numerator).neg_()
            self._aitken_updated.div_(self._aitken_safe_denominator)
            self._torch.gt(
                self._aitken_denominator,
                epsilon,
                out=self._aitken_valid,
            )
            self._aitken_valid.logical_and_(
                self._torch.isfinite(self._aitken_updated)
            )
            self._torch.where(
                self._aitken_valid,
                self._aitken_updated,
                self._aitken_previous_factors,
                out=self._aitken_factors,
            )
            self._aitken_factors.clamp_(
                min=self._relaxation_min, max=self._relaxation_max
            )
        else:
            self._aitken_factors.fill_(self._relaxation_factor)
        self._next_wrenches.copy_(self._iteration_residual)
        self._next_wrenches.mul_(self._aitken_factors)
        self._next_wrenches.add_(self._iteration_guess)
        self._next_wrenches.mul_(self._feedback_mask)
        self._previous_residual.copy_(self._iteration_residual)

    def _relax_iteration(self, iteration: int) -> None:
        self._compute_scaled_feedback(self._iteration_feedback)
        self._update_relaxed_wrench(iteration)
        self._iteration_guess.copy_(self._next_wrenches)

    def _update_proxy_metrics(self) -> None:
        self._torch.linalg.vector_norm(
            self._iteration_residual,
            dim=-1,
            out=self._interface_residual_norms,
        )
        self._torch.amax(
            self._interface_residual_norms,
            out=self._last_interface_residual_device,
        )
        if self._two_way_body_ids.numel() != 0:
            self._torch.index_select(
                self._aitken_factors,
                1,
                self._two_way_body_ids,
                out=self._two_way_aitken_factors,
            )
            self._torch.mean(
                self._two_way_aitken_factors,
                out=self._last_relaxation_coefficient_device,
            )
        else:
            self._last_relaxation_coefficient_device.fill_(
                self._relaxation_factor
            )

    def _timed(
        self, phase: str, operation: Any, *args: Any, **kwargs: Any
    ) -> Any:
        start = time.perf_counter()
        with _nvtx_range(self._torch, f"MuJoCoIpc/{phase}"):
            result = operation(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000.0
        self._phase_latency_ms[phase] = (
            self._phase_latency_ms.get(phase, 0.0) + elapsed
        )
        return result

    def _replay_or_call(self, name: str, operation: Any) -> None:
        graph = self._cuda_graphs.get(name)
        if graph is None:
            operation()
            return
        with _nvtx_range(self._torch, f"MuJoCoIpcGraph/{name}"):
            graph.replay()

    def _graph_mutable_tensors(self) -> tuple[Any, ...]:
        names = (
            "_xfrc",
            "_targets",
            "_positions",
            "_rotations",
            "_selected_wrenches",
            "_next_wrenches",
            "_applied_wrenches",
            "_last_interface_residual_device",
            "_last_relaxation_coefficient_device",
            "_target_twists",
            "_com_positions",
            "_spatial_velocities",
            "_origin_twists",
            "_origin_minus_com",
            "_angular_cross_offset",
            "_checkpoint_applied_wrenches",
            "_iteration_guess",
            "_iteration_feedback",
            "_iteration_residual",
            "_previous_residual",
            "_aitken_delta",
            "_aitken_factors",
            "_aitken_product",
            "_aitken_squared",
            "_aitken_numerator",
            "_aitken_denominator",
            "_aitken_safe_denominator",
            "_aitken_updated",
            "_aitken_previous_factors",
            "_aitken_valid",
            "_interface_residual_norms",
            "_two_way_aitken_factors",
            "_computed_wrenches",
            "_relative_rotation",
            "_rotation_error",
            "_local_rotation_error",
            "_local_torque",
            "_world_torque",
            "_world_center",
            "_center_acceleration_force",
        )
        values = [
            getattr(self, name)
            for name in names
            if getattr(self, name, None) is not None
        ]
        if self._uses_rollback:
            values.extend(self._rigid_checkpoint_arrays.values())
            rigid_arrays = self.rigid_solver.arrays
            values.extend(
                rigid_arrays[name]
                for name in self._rigid_checkpoint_arrays
            )
        unique = []
        pointers = set()
        for value in values:
            data_ptr = getattr(value, "data_ptr", None)
            if not callable(data_ptr):
                continue
            pointer = int(data_ptr())
            if pointer in pointers:
                continue
            pointers.add(pointer)
            unique.append(value)
        return tuple(unique)

    def initialize_cuda_graphs(self) -> None:
        """Capture graph-safe tensor exchange segments, falling back to eager."""

        self.validate_storage()
        self._cuda_graphs.clear()
        if not self._capture_graphs_requested:
            self._cuda_graph_capture_reason = "coupler graph capture disabled"
            return
        if getattr(self._xfrc.device, "type", "cpu") != "cuda":
            self._cuda_graph_capture_reason = (
                "coupler graph capture requires CUDA tensors"
            )
            return
        cuda = getattr(self._torch, "cuda", None)
        if cuda is None or not bool(cuda.is_available()) or not hasattr(
            cuda, "CUDAGraph"
        ):
            self._cuda_graph_capture_reason = (
                "Torch CUDA graph support is unavailable"
            )
            return

        operations = {
            "checkpoint": self._capture_rigid_checkpoint_device,
            "rewind": self._rewind_rigid_checkpoint_device,
            "gather_kinematics": self._gather_rigid_kinematics_unchecked,
            "apply_guess": lambda: self._replace_owned_wrenches_unchecked(
                self._iteration_guess
            ),
            "relax_first": lambda: self._relax_iteration(0),
            "relax_next": lambda: self._relax_iteration(1),
            "metrics": self._update_proxy_metrics,
        }
        if not self._uses_rollback:
            del operations["checkpoint"]
            del operations["rewind"]
            del operations["relax_next"]

        mutable = self._graph_mutable_tensors()
        snapshots = tuple(value.clone() for value in mutable)
        device = self._xfrc.device
        try:
            capture_stream = cuda.Stream(device=device)
            current_stream = cuda.current_stream(device)
            capture_stream.wait_stream(current_stream)
            with cuda.stream(capture_stream):
                for operation in operations.values():
                    operation()
            capture_stream.synchronize()
            for name, operation in operations.items():
                graph = cuda.CUDAGraph()
                with cuda.graph(graph, stream=capture_stream):
                    operation()
                self._cuda_graphs[name] = graph
            capture_stream.synchronize()
            for value, snapshot in zip(mutable, snapshots, strict=True):
                value.copy_(snapshot)
            cuda.synchronize(device)
        except Exception as error:
            self._cuda_graphs.clear()
            try:
                for value, snapshot in zip(mutable, snapshots, strict=True):
                    value.copy_(snapshot)
                cuda.synchronize(device)
            except Exception:
                pass
            self._cuda_graph_capture_reason = (
                f"{type(error).__name__}: {error}"
            )
            return
        self._cuda_graph_capture_reason = ""

    def step(self, actions: Any | None = None) -> None:
        """Advance one shared MuJoCo/libuipc proxy-coupled microstep."""

        self._require_phase("Idle")
        self._phase_latency_ms = {}
        if self._uses_rollback:
            self._phase = "CaptureCheckpoint"
            self._timed("rigid_checkpoint", self._capture_rigid_checkpoint)
            self._timed("ipc_checkpoint", self.ipc_solver.capture_checkpoint)
        self._iteration_guess.copy_(self._applied_wrenches)
        self._iteration_guess.mul_(self._feedback_mask)
        self._aitken_factors.fill_(self._relaxation_factor)

        for iteration in range(self._coupling_iterations):
            if iteration > 0:
                self._phase = "RewindCheckpoint"
                self._timed("rigid_checkpoint", self._rewind_rigid_checkpoint)
                self._timed("ipc_checkpoint", self.ipc_solver.rewind_checkpoint)
            self._replay_or_call(
                "apply_guess",
                lambda: self._replace_owned_wrenches_unchecked(
                    self._iteration_guess
                ),
            )

            self._phase = "StepRigid"
            self._timed("rigid_advance", self.rigid_solver.step, actions, nsteps=1)
            self._phase = "PushRigidKinematics"
            self._timed("target_exchange", self._gather_rigid_kinematics)
            self._phase = "StepIpc"
            self._timed("ipc_advance", self.ipc_solver.step, nsteps=1)
            self._phase = "RelaxFeedback"
            graph_name = "relax_first" if iteration == 0 else "relax_next"
            self._timed(
                "reaction_exchange",
                self._replay_or_call,
                graph_name,
                lambda value=iteration: self._relax_iteration(value),
            )

        self._replay_or_call(
            "apply_guess",
            lambda: self._replace_owned_wrenches_unchecked(
                self._iteration_guess
            ),
        )
        if self._uses_rollback:
            self._phase = "CommitCheckpoint"
            self._timed("ipc_checkpoint", self.ipc_solver.commit_checkpoint)
        self._last_coupling_iterations = self._coupling_iterations
        self._replay_or_call("metrics", self._update_proxy_metrics)
        self._completed_steps += 1
        self._phase = "Idle"

    def _require_phase(self, expected: str) -> None:
        if self._phase != expected:
            raise RuntimeError(
                f"MuJoCo+IPC phase violation: expected {expected}, got {self._phase}"
            )

    def sync_rigid_pose(self) -> None:
        """Synchronize targets outside a step, for construction/reset/forward."""

        self._gather_rigid_kinematics()

    def release_wrenches(self) -> None:
        """Remove only forces previously contributed by this coupler."""

        self._torch.index_select(
            self._xfrc, 1, self._body_ids, out=self._selected_wrenches
        )
        self._selected_wrenches.sub_(self._applied_wrenches)
        self._xfrc.index_copy_(1, self._body_ids, self._selected_wrenches)
        self._applied_wrenches.zero_()

    def abort(self) -> None:
        self.release_wrenches()
        self._phase = "Faulted"

    def recover(self) -> None:
        self.release_wrenches()
        self._last_coupling_iterations = 0
        self._last_interface_residual_device.zero_()
        self._last_relaxation_coefficient_device.fill_(
            self._relaxation_factor
        )
        self._phase = "Idle"

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def feedback_source(self) -> str:
        return (
            "proxy_constraint"
            if self._wrench_source == "pose_error"
            else "native_contact_wrench"
        )

    @property
    def last_coupling_iterations(self) -> int:
        return self._last_coupling_iterations

    @property
    def last_interface_residual(self) -> float:
        return float(self._last_interface_residual_device.item())

    @property
    def last_relaxation_coefficient(self) -> float:
        return float(self._last_relaxation_coefficient_device.item())

    @property
    def cuda_graph_captured(self) -> bool:
        return bool(self._cuda_graphs)

    @property
    def cuda_graph_capture_reason(self) -> str:
        return self._cuda_graph_capture_reason

    @property
    def phase_latency_ms(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self._phase_latency_ms))

    @property
    def storage_signature(self) -> tuple[tuple[str, int], ...]:
        return self._storage_signature

class _MuJoCoIpcRobotViewAdapter:
    def __init__(
        self, provider: "MuJoCoIpcProvider", spec: RobotBatchSpec
    ) -> None:
        self.provider = provider
        self.inner = provider.rigid_solver._create_robot_view_adapter(spec)

    def read_state(self, state: Any) -> Any:
        return self.inner.read_state(state)

    def set_position_targets(self, targets: Any) -> None:
        self.inner.set_position_targets(targets)

    def set_controls(self, controls: Any) -> None:
        self.inner.set_controls(controls)

    def reset(self, reset_mask: Any, **state: Any) -> Mapping[str, Any]:
        mask = self.provider._require_full_reset(reset_mask)
        return self.provider._perform_full_reset(
            mask, lambda: self.inner.reset(mask, **state)
        )


class MuJoCoIpcProvider(BatchPhysicsProvider):
    """Composite simulator with MuJoCo rigid authority and libuipc FEM/contact.

    Each microstep runs a staggered SolverCoupledProxy iteration: apply the
    current interface wrench, advance MuJoCo, synchronize proxy kinematics,
    advance libuipc, then harvest and relax the new wrench. Iteration two and
    above rewind both solvers to the same microstep start.
    """

    accepts_device_actions = True

    def __init__(
        self,
        artifact: Mapping[str, Any] | CompiledMuJoCoIpcArtifact,
        *,
        config: MuJoCoIpcConfig | None = None,
        rigid_solver: Any | None = None,
        ipc_solver: Any | None = None,
        mujoco_options: Mapping[str, Any] | None = None,
        libuipc_config: Any | None = None,
    ) -> None:
        if config is not None and not isinstance(config, MuJoCoIpcConfig):
            raise TypeError("config must be a MuJoCoIpcConfig")
        self.artifact = validate_mujoco_ipc_artifact(artifact)
        self.config = config or MuJoCoIpcConfig()
        self._closed = False
        self._faulted = False
        self._fault_reason = ""
        self._generation = 1
        self._step_count = 0

        if rigid_solver is None:
            options = dict(mujoco_options or {})
            options.setdefault("num_envs", self.config.num_envs)
            options.setdefault("device", self.config.device)
            options.setdefault(
                "capture_graphs", self.config.capture_mujoco_graphs
            )
            rigid_solver = MuJoCoWarpProvider(self.artifact.mujoco, **options)
        elif mujoco_options:
            raise TypeError(
                "mujoco_options cannot be used with an injected rigid_solver"
            )
        self.rigid_solver = rigid_solver

        try:
            if ipc_solver is None:
                from ...ipc._batch_solver import (
                    LibuipcBatchConfig,
                    LibuipcBatchSolver,
                )
                from ...ipc._libuipc_provider import LibuipcConfig

                if libuipc_config is None:
                    solver_config = LibuipcConfig(
                        fixed_time_step=(
                            self._rigid_fixed_time_step()
                            * self.config.rigid_substeps
                            / self.config.ipc_substeps
                        ),
                        device_index=self._device_index(),
                    )
                    libuipc_config = LibuipcBatchConfig(
                        solver=solver_config,
                        environments_per_shard=self.config.environments_per_shard,
                    )
                ipc_solver = LibuipcBatchSolver(
                    self.artifact.ipc,
                    num_envs=self.config.num_envs,
                    config=libuipc_config,
                    device=self.config.device,
                )
            elif libuipc_config is not None:
                raise TypeError(
                    "libuipc_config cannot be used with an injected ipc_solver"
                )
            self.ipc_solver = ipc_solver
            self._validate_solvers()
            self.coupler = SolverCoupledProxy(
                self.rigid_solver,
                self.ipc_solver,
                self.artifact.coupled_bodies,
                force_scale=self.config.force_scale,
                torque_scale=self.config.torque_scale,
                coupling_iterations=self.config.coupling_iterations,
                relaxation_mode=self.config.relaxation_mode,
                relaxation_factor=self.config.relaxation_factor,
                relaxation_min=self.config.relaxation_min,
                relaxation_max=self.config.relaxation_max,
                capture_graphs=self.config.capture_coupler_graphs,
            )
            self.coupler.sync_rigid_pose()
            self.coupler.initialize_cuda_graphs()
            self._arrays = self._make_array_views()
        except Exception:
            coupler = getattr(self, "coupler", None)
            if coupler is not None:
                try:
                    coupler.release_wrenches()
                except Exception:
                    pass
            close = getattr(self.rigid_solver, "close", None)
            if callable(close):
                close()
            close = getattr(locals().get("ipc_solver"), "close", None)
            if callable(close):
                close()
            raise

        fingerprint = {
            "artifact": self.artifact.digest,
            "config": asdict(self.config),
            "rigid": str(getattr(self.rigid_solver, "runtime_fingerprint", "")),
            "ipc": str(getattr(self.ipc_solver, "runtime_fingerprint", "")),
            "provider": "mujoco-libuipc",
        }
        self._runtime_fingerprint = "sha256:" + hashlib.sha256(
            json.dumps(
                fingerprint, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()

    @classmethod
    def from_context(cls, context: Any, **kwargs: Any) -> "MuJoCoIpcProvider":
        return cls(CompiledMuJoCoIpcArtifact.from_context(context), **kwargs)

    @staticmethod
    def availability() -> Any:
        rigid = MuJoCoWarpProvider.availability()
        if not rigid.available:
            return rigid
        try:
            from ...ipc._batch_solver import LibuipcBatchSolver

            return LibuipcBatchSolver.availability()
        except Exception as error:
            from .mujoco_warp import MuJoCoWarpProviderAvailability

            return MuJoCoWarpProviderAvailability(False, str(error))

    def _device_index(self) -> int:
        device = str(self.config.device)
        if ":" not in device:
            return 0
        try:
            return int(device.rsplit(":", 1)[1])
        except ValueError as error:
            raise ValueError(f"invalid CUDA device {device!r}") from error

    def _rigid_fixed_time_step(self) -> float:
        value = getattr(self.rigid_solver, "fixed_time_step", None)
        if value is not None:
            return float(value)
        model = getattr(self.rigid_solver, "_mj_model", None)
        if model is not None and getattr(model, "opt", None) is not None:
            return float(model.opt.timestep)
        raise RuntimeError("rigid solver does not expose its fixed time step")

    def _validate_solvers(self) -> None:
        for name, solver in (
            ("MuJoCo", self.rigid_solver),
            ("libuipc", self.ipc_solver),
        ):
            if int(getattr(solver, "num_envs", -1)) != self.config.num_envs:
                raise ValueError(
                    f"{name} solver environment count does not match "
                    f"MuJoCoIpcConfig.num_envs"
                )
        rigid_dt = self._rigid_fixed_time_step()
        ipc_dt = float(getattr(self.ipc_solver, "fixed_time_step", float("nan")))
        rigid_macro_dt = rigid_dt * self.config.rigid_substeps
        ipc_macro_dt = ipc_dt * self.config.ipc_substeps
        if not math.isclose(
            rigid_macro_dt, ipc_macro_dt, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                "MuJoCo and libuipc must cover the same macro fixed time step: "
                f"{rigid_dt} * {self.config.rigid_substeps} != "
                f"{ipc_dt} * {self.config.ipc_substeps}"
            )
        if not math.isclose(rigid_dt, ipc_dt, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                "SolverCoupledProxy requires equal MuJoCo and libuipc microstep dt"
            )
        if self.config.coupling_iterations > 1 and not bool(
            getattr(self.ipc_solver.capabilities, "runtime_checkpoint", False)
        ):
            raise ValueError(
                "SolverCoupledProxy x2+ requires libuipc runtime checkpoint support"
            )
        ipc_shard_count = int(getattr(self.ipc_solver, "shard_count", -1))
        if ipc_shard_count != self.config.shard_count:
            raise ValueError(
                "libuipc shard layout does not match MuJoCoIpcConfig: "
                f"{ipc_shard_count} != {self.config.shard_count}"
            )
        affine_paths = tuple(
            str(value["path"]) for value in self.ipc_solver.affine_bodies
        )
        expected_paths = tuple(
            value.ipc_path for value in self.artifact.coupled_bodies
        )
        if affine_paths != expected_paths:
            raise RuntimeError(
                "libuipc affine proxy order does not match the composite artifact"
            )

    def _make_array_views(self) -> Mapping[str, Any]:
        values = dict(self.rigid_solver.arrays)
        ipc = self.ipc_solver.arrays
        values.update(
            {
                "ipc_positions": ipc["positions"],
                "ipc_velocities": ipc["velocities"],
                "ipc_contact_forces": ipc["contact_forces"],
                "ipc_affine_targets": ipc["affine_targets"],
                "ipc_affine_target_twists": ipc.get("affine_target_twists"),
                "ipc_affine_transforms": ipc["affine_transforms"],
                "ipc_affine_contact_wrenches": ipc[
                    "affine_contact_wrenches"
                ],
            }
        )
        return MappingProxyType(
            {name: value for name, value in values.items() if value is not None}
        )

    @property
    def capabilities(self) -> BatchProviderCapabilities:
        rigid_capabilities = self.rigid_solver.capabilities
        ipc_capabilities = self.ipc_solver.capabilities
        return BatchProviderCapabilities(
            name="MuJoCo+libuipc",
            device=self.config.device,
            device_native=(
                bool(rigid_capabilities.device_native)
                and bool(ipc_capabilities.device_native)
            ),
            graph_capture=False,
            masked_reset=False,
            fixed_capacity=True,
            runtime_checkpoint=False,
            exact_contact_wrench=(
                self.coupler.feedback_source == "native_contact_wrench"
                and bool(ipc_capabilities.exact_contact_wrench)
            ),
            sensor_batch=bool(rigid_capabilities.sensor_batch),
            solver_substeps=True,
            graph_capture_reason=(
                "the composite step spans MuJoCo Warp and libuipc; native "
                "affine-target and exact-contact-wrench exchange uses "
                "device-to-device kernels around the libuipc advance boundary"
            ),
            reset_scope="full_batch_only",
        )

    @property
    def num_envs(self) -> int:
        return self.config.num_envs

    @property
    def frame(self) -> int:
        self._require_open()
        return self._step_count

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def fixed_time_step(self) -> float:
        return self._rigid_fixed_time_step() * self.config.rigid_substeps

    @property
    def runtime_fingerprint(self) -> str:
        return self._runtime_fingerprint

    @property
    def graph_captured(self) -> bool:
        return False

    @property
    def arrays(self) -> Mapping[str, Any]:
        self._require_open()
        return self._arrays

    @property
    def contact_sensors(self) -> Mapping[str, Mapping[str, Any]]:
        self._require_open()
        return getattr(self.rigid_solver, "contact_sensors", MappingProxyType({}))

    @property
    def raycast_sensors(self) -> Mapping[str, Mapping[str, Any]]:
        self._require_open()
        return getattr(self.rigid_solver, "raycast_sensors", MappingProxyType({}))

    @property
    def capacities(self) -> Mapping[str, int]:
        self._require_open()
        rigid = dict(getattr(self.rigid_solver, "capacities", {}))
        ipc = dict(getattr(self.ipc_solver, "capacities", {}))
        values = {f"mujoco_{name}": int(value) for name, value in rigid.items()}
        values.update(
            {f"libuipc_{name}": int(value) for name, value in ipc.items()}
        )
        values.update(
            {
                "coupled_bodies": len(self.artifact.coupled_bodies),
                "shards": self.config.shard_count,
                "environments_per_shard": self.config.environments_per_shard,
            }
        )
        return MappingProxyType(values)

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        self._require_open()
        mujoco_diagnostics = dict(
            getattr(self.rigid_solver, "diagnostics", {})
        )
        ipc_diagnostics = dict(getattr(self.ipc_solver, "diagnostics", {}))
        phase_latency_ms = dict(self.coupler.phase_latency_ms)
        for source, target in (
            ("last_checkpoint_latency_ms", "ipc_checkpoint_native"),
            ("last_target_staging_latency_ms", "target_staging_native"),
            ("last_ipc_advance_latency_ms", "ipc_advance_native"),
            ("last_reaction_export_latency_ms", "reaction_export_native"),
            ("last_state_sync_latency_ms", "state_sync_native"),
        ):
            if source in ipc_diagnostics:
                phase_latency_ms[target] = float(ipc_diagnostics[source])
        return MappingProxyType(
            {
                "provider": "mujoco-libuipc",
                "device": self.config.device,
                "num_envs": self.num_envs,
                "frame": self._step_count,
                "shard_count": self.config.shard_count,
                "coupled_body_count": len(self.artifact.coupled_bodies),
                "proxy_count": len(self.artifact.coupled_bodies),
                "static_collider_count": sum(
                    not bool(collider["disabled"])
                    for collider in self.artifact.ipc.static_colliders
                ),
                "collision_ownership": dict(self.artifact.collision_ownership),
                "mujoco": mujoco_diagnostics,
                "libuipc": ipc_diagnostics,
                "graph_captured": False,
                "graph_capture_reason": (
                    self.capabilities.graph_capture_reason
                ),
                "coupler_graph_captured": self.coupler.cuda_graph_captured,
                "coupler_graph_capture_reason": (
                    self.coupler.cuda_graph_capture_reason
                ),
                "phase_latency_ms": phase_latency_ms,
                "feedback_source": self.coupler.feedback_source,
                "contact_pipeline": ipc_diagnostics.get(
                    "contact_constitution", "ipc"
                ),
                "faulted": self._faulted,
                "fault_reason": self._fault_reason,
                "coupler_phase": self.coupler.phase,
                "stable_storage": True,
                "reset_scope": "full_batch_only",
                "affine_target_staging": ipc_diagnostics.get(
                    "affine_target_staging", "unknown"
                ),
                "contact_wrench_staging": ipc_diagnostics.get(
                    "contact_wrench_staging", "unknown"
                ),
                "coupling_solver": "SolverCoupledProxy",
                "rollback_enabled": self.config.coupling_iterations > 1,
                "coupling_iterations": self.config.coupling_iterations,
                "actual_coupling_iterations": (
                    self.coupler.last_coupling_iterations
                ),
                "interface_residual": self.coupler.last_interface_residual,
                "relaxation_mode": self.config.relaxation_mode,
                "relaxation_factor": self.config.relaxation_factor,
                "aitken_coefficient": (
                    self.coupler.last_relaxation_coefficient
                ),
                "macro_fixed_time_step": self.fixed_time_step,
                "rigid_fixed_time_step": self._rigid_fixed_time_step(),
                "ipc_fixed_time_step": float(self.ipc_solver.fixed_time_step),
                "rigid_substeps": self.config.rigid_substeps,
                "ipc_substeps": self.config.ipc_substeps,
            }
        )

    def _create_robot_view_adapter(
        self, spec: RobotBatchSpec
    ) -> _MuJoCoIpcRobotViewAdapter:
        self._require_open()
        return _MuJoCoIpcRobotViewAdapter(self, spec)

    def step(
        self, actions: Any | None = None, *, nsteps: int = 1
    ) -> Mapping[str, Any]:
        self._require_operational()
        try:
            count = operator.index(nsteps)
        except TypeError as error:
            raise TypeError("MuJoCo+IPC step count must be an integer") from error
        if isinstance(nsteps, bool) or count <= 0:
            raise ValueError("MuJoCo+IPC step count must be positive")
        try:
            for index in range(count):
                step_actions = actions if index == 0 else None
                for _ in range(self.config.rigid_substeps):
                    self.coupler.step(step_actions)
        except Exception as error:
            self._faulted = True
            self._fault_reason = str(error)
            try:
                self.coupler.abort()
            except Exception:
                pass
            raise
        self._step_count += count
        return self._arrays

    def _require_full_reset(self, reset_mask: Any) -> Any:
        torch = getattr(self.rigid_solver, "_torch", None)
        if torch is None:
            raise TypeError("rigid solver must expose its Torch runtime")
        device = self.rigid_solver.arrays["qpos"].device
        mask = torch.as_tensor(reset_mask, dtype=torch.bool, device=device)
        if tuple(mask.shape) != (self.num_envs,):
            raise ValueError(
                f"reset mask must have shape ({self.num_envs},), "
                f"got {tuple(mask.shape)}"
            )
        if not bool(mask.all().item()):
            raise NotImplementedError(
                "partial reset is unsupported because libuipc cannot restore "
                "individual environments within a shard"
            )
        return mask

    def _perform_full_reset(self, mask: Any, reset_rigid: Any) -> Mapping[str, Any]:
        self._require_open()
        try:
            self.coupler.release_wrenches()
            result = reset_rigid()
            self.ipc_solver.reset(mask)
            self.coupler.recover()
            self.coupler.sync_rigid_pose()
        except Exception as error:
            self._faulted = True
            self._fault_reason = str(error)
            try:
                self.coupler.abort()
            except Exception:
                pass
            raise
        self._faulted = False
        self._fault_reason = ""
        self._step_count = 0
        return result

    def reset(self, reset_mask: Any, **state: Any) -> Mapping[str, Any]:
        self._require_open()
        mask = self._require_full_reset(reset_mask)
        self._perform_full_reset(
            mask, lambda: self.rigid_solver.reset(mask, **state)
        )
        return self._arrays

    def forward(self) -> Mapping[str, Any]:
        self._require_operational()
        forward = getattr(self.rigid_solver, "forward", None)
        if not callable(forward):
            raise NotImplementedError("rigid solver does not support forward()")
        forward()
        self.coupler.sync_rigid_pose()
        return self._arrays

    def sense(self) -> Mapping[str, Any]:
        self._require_operational()
        sense = getattr(self.rigid_solver, "sense", None)
        if not callable(sense):
            raise NotImplementedError("rigid solver does not support sense()")
        sense()
        return self._arrays

    def refresh_deformable_contact_forces(self) -> Any:
        self._require_operational()
        refresh = getattr(
            self.ipc_solver, "refresh_deformable_contact_forces", None
        )
        if not callable(refresh):
            raise NotImplementedError(
                "IPC solver does not support lazy contact-force refresh"
            )
        return refresh()

    def refresh_state(self) -> Mapping[str, Any]:
        self._require_operational()
        refresh = getattr(self.ipc_solver, "refresh_state", None)
        if not callable(refresh):
            raise NotImplementedError("IPC solver does not support lazy state refresh")
        refresh()
        return self._arrays

    def synchronize(self) -> None:
        self._require_operational()
        synchronize = getattr(self.ipc_solver, "synchronize", None)
        if callable(synchronize):
            synchronize()
        synchronize = getattr(self.rigid_solver, "synchronize", None)
        if callable(synchronize):
            synchronize()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.coupler.release_wrenches()
        finally:
            self._closed = True
            self._generation += 1
            self._arrays = MappingProxyType({})
            first_error = None
            for solver in (self.ipc_solver, self.rigid_solver):
                try:
                    solver.close()
                except Exception as error:
                    if first_error is None:
                        first_error = error
            if first_error is not None:
                raise first_error

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("MuJoCo+libuipc provider is closed")

    def _require_operational(self) -> None:
        self._require_open()
        if self._faulted:
            raise RuntimeError(
                "MuJoCo+libuipc provider is faulted; perform a successful full "
                "reset or close it"
            )


__all__ = [
    "CompiledMuJoCoIpcArtifact",
    "MuJoCoIpcBodyMapping",
    "MuJoCoIpcConfig",
    "MuJoCoIpcProvider",
    "SolverCoupledProxy",
    "validate_mujoco_ipc_artifact",
]
