"""Batched MuJoCo Warp and libuipc co-simulation provider."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import operator
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


_COLLISION_OWNERSHIP = MappingProxyType(
    {
        "rigid_rigid": "mujoco",
        "rigid_terrain": "mujoco",
        "deformable_deformable": "libuipc",
        "deformable_rigid": "libuipc",
        "deformable_terrain": "libuipc",
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
    schema_version: int = 2

    def __post_init__(self) -> None:
        if int(self.schema_version) != 2:
            if int(self.schema_version) == 1:
                raise ValueError(
                    "unsupported MuJoCo+IPC artifact schema 1; schema 2 requires "
                    "explicit PhysicsCoupling entries"
                )
            raise ValueError(
                f"unsupported MuJoCo+IPC artifact schema {self.schema_version}; "
                "expected schema 2"
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
        object.__setattr__(self, "schema_version", 2)
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
                    f"robot {robot_name!r}, which has no matching MuJoCo robot"
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
        if supplied_schema != 2:
            if supplied_schema == 1:
                raise ValueError(
                    "unsupported MuJoCo+IPC artifact schema 1; missing explicit "
                    "PhysicsCoupling mapping"
                )
            raise ValueError(
                "unsupported MuJoCo+IPC artifact schema; expected schema 2"
            )
        artifact = cls(
            mujoco=validate_compiled_artifact(value.get("mujoco", {})),
            ipc=validate_ipc_artifact(value.get("ipc", {})),
            coupled_bodies=tuple(
                MuJoCoIpcBodyMapping.from_mapping(item)
                for item in value.get("coupled_bodies", ())
            ),
        )
        supplied_digest = str(value.get("digest", ""))
        if supplied_digest and supplied_digest != artifact.digest:
            raise ValueError("MuJoCo+IPC artifact digest mismatch")
        ownership = value.get("collision_ownership")
        if ownership is not None and dict(ownership) != dict(_COLLISION_OWNERSHIP):
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
            "collision_ownership": dict(_COLLISION_OWNERSHIP),
            "digest": self.digest,
        }

    @property
    def collision_ownership(self) -> Mapping[str, str]:
        return _COLLISION_OWNERSHIP

    @property
    def digest(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "mujoco": self.mujoco.digest,
            "ipc": self.ipc.digest,
            "coupled_bodies": [
                mapping.to_mapping() for mapping in self.coupled_bodies
            ],
            "collision_ownership": dict(_COLLISION_OWNERSHIP),
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
    capture_mujoco_graphs: bool = True

    def __post_init__(self) -> None:
        for name in ("num_envs", "environments_per_shard"):
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
        device = str(self.device)
        if not device:
            raise ValueError("device must not be empty")
        object.__setattr__(self, "device", device)
        object.__setattr__(
            self, "capture_mujoco_graphs", bool(self.capture_mujoco_graphs)
        )

    @property
    def shard_count(self) -> int:
        return self.num_envs // self.environments_per_shard


class MuJoCoIpcCoupler:
    """CUDA tensor exchange between rigid and IPC solver state."""

    def __init__(
        self,
        rigid_solver: Any,
        ipc_solver: Any,
        mappings: Sequence[MuJoCoIpcBodyMapping],
        *,
        force_scale: float = 1.0,
        torque_scale: float = 1.0,
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
        if not self._targets.is_contiguous() or not self._native_wrenches.is_contiguous():
            raise RuntimeError("IPC coupling tensors must be contiguous")

        body_names = tuple(mapping.mujoco_body_name for mapping in self.mappings)
        body_ids = rigid_solver.resolve_object_ids("body", body_names)
        self._body_ids = self._torch.as_tensor(
            body_ids, dtype=self._torch.long, device=self._xpos.device
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
        if self._wrench_source == "pose_error":
            self._initialize_pose_error_feedback()
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
        return tuple((name, int(value.data_ptr())) for name, value in values)

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

    def _validate_storage(self) -> None:
        if self._capture_storage_signature() != self._storage_signature:
            raise RuntimeError(
                "MuJoCo+IPC coupling storage changed after construction"
            )

    def _gather_rigid_pose(self) -> None:
        self._validate_storage()
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

    def _require_phase(self, expected: str) -> None:
        if self._phase != expected:
            raise RuntimeError(
                f"MuJoCo+IPC phase violation: expected {expected}, got {self._phase}"
            )

    def sync_rigid_pose(self) -> None:
        """Synchronize targets outside a step, for construction/reset/forward."""

        self._gather_rigid_pose()

    def push_rigid_pose(self) -> None:
        """PushRigidPose: gather every mapped MuJoCo body into proxy targets."""

        self._require_phase("Idle")
        self._gather_rigid_pose()
        self._phase = "StepIpc"

    def step_ipc(self) -> None:
        """StepIpc: advance libuipc exactly one fixed tick."""

        self._require_phase("StepIpc")
        self.ipc_solver.step(nsteps=1)
        self._phase = "ApplyFeedback"

    def apply_feedback(self) -> None:
        """ApplyFeedback: replace only this coupler's TwoWay wrench contribution."""

        self._require_phase("ApplyFeedback")
        self._validate_storage()
        if self._wrench_source == "pose_error":
            self._compute_pose_error_wrenches()
        self._next_wrenches.copy_(self._source_wrenches)
        self._next_wrenches[..., :3].mul_(self._force_scales)
        self._next_wrenches[..., 3:].mul_(self._torque_scales)
        self._next_wrenches.mul_(self._feedback_mask)
        self._torch.index_select(
            self._xfrc, 1, self._body_ids, out=self._selected_wrenches
        )
        self._selected_wrenches.sub_(self._applied_wrenches)
        self._selected_wrenches.add_(self._next_wrenches)
        self._xfrc.index_copy_(1, self._body_ids, self._selected_wrenches)
        self._applied_wrenches.copy_(self._next_wrenches)
        self._phase = "StepRigid"

    def step_rigid(self, actions: Any | None = None) -> None:
        """StepRigid: advance MuJoCo exactly one fixed tick."""

        self._require_phase("StepRigid")
        self.rigid_solver.step(actions, nsteps=1)
        self._phase = "Finalize"

    def finalize(self) -> None:
        """Finalize: validate storage and complete the fixed tick."""

        self._require_phase("Finalize")
        self._validate_storage()
        self._completed_steps += 1
        self._phase = "Idle"

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
    def storage_signature(self) -> tuple[tuple[str, int], ...]:
        return self._storage_signature

    PushRigidPose = push_rigid_pose
    StepIpc = step_ipc
    ApplyFeedback = apply_feedback
    StepRigid = step_rigid
    Finalize = finalize


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

    Per fixed tick the coupler gathers current MuJoCo link poses, advances the
    isolated libuipc subscenes, scatters IPC proxy contact wrenches into
    ``xfrc_applied``, and finally advances MuJoCo. All hot-path exchange uses
    stable tensors on one device.
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
                        fixed_time_step=self._rigid_fixed_time_step(),
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
            self.coupler = MuJoCoIpcCoupler(
                self.rigid_solver,
                self.ipc_solver,
                self.artifact.coupled_bodies,
                force_scale=self.config.force_scale,
                torque_scale=self.config.torque_scale,
            )
            self.coupler.sync_rigid_pose()
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
        if not math.isclose(rigid_dt, ipc_dt, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                "MuJoCo and libuipc must use the same fixed time step: "
                f"{rigid_dt} != {ipc_dt}"
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
                "ipc_affine_transforms": ipc["affine_transforms"],
                "ipc_affine_contact_wrenches": ipc[
                    "affine_contact_wrenches"
                ],
            }
        )
        return MappingProxyType(values)

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
        )

    @property
    def num_envs(self) -> int:
        return self.config.num_envs

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def fixed_time_step(self) -> float:
        return self._rigid_fixed_time_step()

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
        return MappingProxyType(
            {
                "provider": "mujoco-libuipc",
                "device": self.config.device,
                "num_envs": self.num_envs,
                "frame": self._step_count,
                "shard_count": self.config.shard_count,
                "coupled_body_count": len(self.artifact.coupled_bodies),
                "collision_ownership": dict(self.artifact.collision_ownership),
                "mujoco": dict(getattr(self.rigid_solver, "diagnostics", {})),
                "libuipc": dict(getattr(self.ipc_solver, "diagnostics", {})),
                "graph_captured": False,
                "graph_capture_reason": (
                    "the composite step spans MuJoCo Warp and native libuipc shards"
                ),
                "feedback_source": self.coupler.feedback_source,
                "faulted": self._faulted,
                "fault_reason": self._fault_reason,
                "coupler_phase": self.coupler.phase,
                "stable_storage": True,
                "reset_scope": "full_batch_only",
                "affine_target_staging": "per_shard_device_host_device",
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
                self.coupler.push_rigid_pose()
                self.coupler.step_ipc()
                self.coupler.apply_feedback()
                self.coupler.step_rigid(actions if index == 0 else None)
                self.coupler.finalize()
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
    "MuJoCoIpcCoupler",
    "MuJoCoIpcProvider",
    "validate_mujoco_ipc_artifact",
]
