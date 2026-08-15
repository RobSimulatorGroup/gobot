"""Device-buffer contract for batched native libuipc simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import json
import math
import operator
from types import MappingProxyType
from typing import Any, Mapping

from gobot import _core
from gobot.sim import ProviderCapabilities, ProviderUnavailableError

from ._artifact import CompiledIpcSceneArtifact, validate_ipc_artifact
from ._libuipc_provider import (
    LibuipcConfig,
    LibuipcProviderAvailability,
    _preload_libuipc_cuda_libraries,
)


@dataclass(frozen=True)
class LibuipcBatchConfig:
    """Configuration for a fixed-capacity native libuipc batch.

    One native world owns ``environments_per_shard`` isolated libuipc
    subscenes. Multiple worlds are used when ``num_envs`` exceeds that shard
    size. The first implementation intentionally supports full-batch reset
    only; storage and subscene IDs therefore remain stable for its lifetime.
    """

    solver: LibuipcConfig = field(default_factory=LibuipcConfig)
    environments_per_shard: int = 64
    contact_constitution: str = "ipc"
    al_ipc_mu_scale_fem: float = 5.0e7
    al_ipc_mu_scale_abd: float = 1.0e5
    al_ipc_toi_threshold: float = 0.1
    al_ipc_alpha_lower_bound: float = 1.0e-6
    al_ipc_decay_factor: float = 0.3

    def __post_init__(self) -> None:
        if not isinstance(self.solver, LibuipcConfig):
            raise TypeError("solver must be a LibuipcConfig")
        try:
            shard_size = operator.index(self.environments_per_shard)
        except TypeError as error:
            raise TypeError("environments_per_shard must be an integer") from error
        if isinstance(self.environments_per_shard, bool) or shard_size <= 0:
            raise ValueError("environments_per_shard must be positive")
        object.__setattr__(self, "environments_per_shard", shard_size)
        constitution = str(self.contact_constitution).lower()
        if constitution not in ("ipc", "al-ipc"):
            raise ValueError("contact_constitution must be 'ipc' or 'al-ipc'")
        object.__setattr__(self, "contact_constitution", constitution)
        for name in ("al_ipc_mu_scale_fem", "al_ipc_mu_scale_abd"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in (
            "al_ipc_toi_threshold",
            "al_ipc_alpha_lower_bound",
            "al_ipc_decay_factor",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1]")
            object.__setattr__(self, name, value)

    def solver_mapping(self, num_envs: int) -> Mapping[str, Any]:
        values = dict(self.solver.solver_mapping())
        values.update(
            {
                "environment_count": int(num_envs),
                "environments_per_shard": self.environments_per_shard,
                "external_affine_proxies": True,
                "contact_constitution": self.contact_constitution,
                "al_ipc_mu_scale_fem": self.al_ipc_mu_scale_fem,
                "al_ipc_mu_scale_abd": self.al_ipc_mu_scale_abd,
                "al_ipc_toi_threshold": self.al_ipc_toi_threshold,
                "al_ipc_alpha_lower_bound": self.al_ipc_alpha_lower_bound,
                "al_ipc_decay_factor": self.al_ipc_decay_factor,
            }
        )
        return values


class LibuipcBatchSolver:
    """Fixed-capacity libuipc solver over externally-owned device tensors.

    The native session receives the tensors once during construction. Neither
    :meth:`step` nor :meth:`set_affine_targets` is allowed to replace their
    storage, which makes the contract suitable for a CUDA-only coupler.
    """

    accepts_device_targets = True
    wrench_source = "direct"

    def __enter__(self) -> "LibuipcBatchSolver":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.close()
        return False

    def __init__(
        self,
        artifact: Mapping[str, Any] | CompiledIpcSceneArtifact,
        *,
        num_envs: int,
        config: LibuipcBatchConfig | None = None,
        device: str | None = None,
        _session: Any | None = None,
        _torch: Any | None = None,
    ) -> None:
        try:
            environment_count = operator.index(num_envs)
        except TypeError as error:
            raise TypeError("num_envs must be an integer") from error
        if isinstance(num_envs, bool) or environment_count <= 0:
            raise ValueError("num_envs must be positive")
        if config is not None and not isinstance(config, LibuipcBatchConfig):
            raise TypeError("config must be a LibuipcBatchConfig")

        self.artifact = validate_ipc_artifact(artifact)
        self.config = config or LibuipcBatchConfig()
        if environment_count % self.config.environments_per_shard != 0:
            raise ValueError(
                "num_envs must be divisible by environments_per_shard so shard "
                "storage remains fixed"
            )
        if self.artifact.tactile_sensors:
            raise ValueError(
                "the libuipc batch solver does not yet render tactile images"
            )
        self._num_envs = environment_count
        self._device_name = str(
            device
            if device is not None
            else f"cuda:{self.config.solver.device_index}"
        )
        self._torch = _torch if _torch is not None else importlib.import_module("torch")
        self._device = self._torch.device(self._device_name)
        self._closed = False
        self._generation = 1
        self._frame = 0
        self._checkpoint_frame: int | None = None
        self.wrench_source = (
            "pose_error"
            if self.config.contact_constitution == "al-ipc"
            else "direct"
        )

        self._deformable_bodies = self._build_deformable_layout()
        self._affine_bodies = self._build_affine_layout()
        vertex_count = sum(
            int(body["element_count"]) for body in self._deformable_bodies
        )
        affine_count = len(self._affine_bodies)
        dtype = self._torch.float64
        native_affine_transforms = self._torch.empty(
            (self._num_envs, affine_count, 4, 4),
            dtype=dtype,
            device=self._device,
        )
        arrays = {
            "positions": self._torch.empty(
                (self._num_envs, vertex_count, 3), dtype=dtype, device=self._device
            ),
            "velocities": self._torch.empty(
                (self._num_envs, vertex_count, 3), dtype=dtype, device=self._device
            ),
            "contact_forces": self._torch.empty(
                (self._num_envs, vertex_count, 3), dtype=dtype, device=self._device
            ),
            "affine_targets": self._torch.zeros(
                (self._num_envs, affine_count, 4, 4),
                dtype=dtype,
                device=self._device,
            ),
            "affine_target_twists": self._torch.zeros(
                (self._num_envs, affine_count, 6),
                dtype=dtype,
                device=self._device,
            ),
            # Eigen matrices are written column-major by libuipc. Expose the
            # logical row-major transform without another device copy.
            "affine_transforms": native_affine_transforms.transpose(-1, -2),
            "affine_contact_wrenches": self._torch.zeros(
                (self._num_envs, affine_count, 6),
                dtype=dtype,
                device=self._device,
            ),
        }
        if affine_count:
            identity = self._torch.eye(4, dtype=dtype, device=self._device)
            arrays["affine_targets"].copy_(identity)
        self._arrays = MappingProxyType(arrays)
        native_buffers = dict(arrays)
        native_buffers["affine_transforms"] = native_affine_transforms
        self._native_buffers = MappingProxyType(native_buffers)

        self._session = _session if _session is not None else self._create_session()
        try:
            bind = getattr(self._session, "bind_device_buffers", None)
            if not callable(bind):
                raise RuntimeError(
                    "native libuipc batch session has no device-buffer binding API"
                )
            bind(dict(self._native_buffers))
            self._validate_native_layout()
        except Exception:
            self._close_session()
            raise

        fingerprint = {
            "artifact": self.artifact.digest,
            "config": {
                "solver": asdict(self.config.solver),
                "environments_per_shard": self.config.environments_per_shard,
                "contact_constitution": self.config.contact_constitution,
                "al_ipc_mu_scale_fem": self.config.al_ipc_mu_scale_fem,
                "al_ipc_mu_scale_abd": self.config.al_ipc_mu_scale_abd,
                "al_ipc_toi_threshold": self.config.al_ipc_toi_threshold,
                "al_ipc_alpha_lower_bound": self.config.al_ipc_alpha_lower_bound,
                "al_ipc_decay_factor": self.config.al_ipc_decay_factor,
            },
            "device": self._device_name,
            "num_envs": self._num_envs,
            "provider": "libuipc-batch",
            "schema_version": 4,
        }
        self._runtime_fingerprint = "sha256:" + hashlib.sha256(
            json.dumps(
                fingerprint, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()

    @classmethod
    def availability(
        cls, config: LibuipcBatchConfig | None = None
    ) -> LibuipcProviderAvailability:
        resolved = config or LibuipcBatchConfig()
        session_type = getattr(_core, "_IpcBatchSolverSession", None)
        if session_type is None:
            return LibuipcProviderAvailability(
                False, "this Gobot build has no native IPC batch bindings"
            )
        try:
            _preload_libuipc_cuda_libraries()
            available = bool(
                session_type.is_module_available(resolved.solver.module_path)
            )
        except Exception as error:
            return LibuipcProviderAvailability(False, str(error))
        if not available:
            module = (
                resolved.solver.module_path
                or "the bundled libuipc batch solver module"
            )
            return LibuipcProviderAvailability(
                False, f"{module} does not expose the IPC batch ABI"
            )
        return LibuipcProviderAvailability(True)

    @classmethod
    def from_context(cls, context: Any, **kwargs: Any) -> "LibuipcBatchSolver":
        compile_artifact = getattr(context, "compile_ipc_scene_artifact", None)
        if not callable(compile_artifact):
            raise RuntimeError(
                "this Gobot build does not expose IPC scene compilation"
            )
        return cls(compile_artifact(), **kwargs)

    def _create_session(self) -> Any:
        availability = self.availability(self.config)
        if not availability.available:
            raise ProviderUnavailableError(availability.reason)
        return _core._IpcBatchSolverSession(
            dict(self.artifact.to_mapping()),
            dict(self.config.solver_mapping(self._num_envs)),
            self.config.solver.module_path,
        )

    def _build_deformable_layout(self) -> tuple[Mapping[str, Any], ...]:
        offset = 0
        result = []
        for body in self.artifact.deformable_bodies:
            count = int(body["vertex_count"])
            result.append(
                MappingProxyType(
                    {
                        "path": str(body["path"]),
                        "element_offset": offset,
                        "element_count": count,
                    }
                )
            )
            offset += count
        return tuple(result)

    def _build_affine_layout(self) -> tuple[Mapping[str, Any], ...]:
        links_by_path = {
            str(link["path"]): link
            for robot in self.artifact.robots
            for link in robot["links"]
        }
        result = []
        for coupling in self.artifact.couplings:
            link = links_by_path[str(coupling["link_path"])]
            result.append(
                MappingProxyType(
                    {
                        "path": str(link["path"]),
                        "coupling_path": str(coupling["coupling_path"]),
                        "mode": str(coupling["mode"]),
                        "force_scale": float(coupling["force_scale"]),
                        "torque_scale": float(coupling["torque_scale"]),
                        "element_offset": int(coupling["proxy_index"]),
                        "element_count": 1,
                        "mass": max(float(link.get("mass", 0.0)), 1.0e-6),
                        "center_of_mass": tuple(
                            float(value)
                            for value in link.get(
                                "center_of_mass", (0.0, 0.0, 0.0)
                            )
                        ),
                        "inertia_diagonal": tuple(
                            float(value)
                            for value in link.get(
                                "inertia_diagonal", (1.0e-6,) * 3
                            )
                        ),
                        "inertia_off_diagonal": tuple(
                            float(value)
                            for value in link.get(
                                "inertia_off_diagonal", (0.0,) * 3
                            )
                        ),
                    }
                )
            )
        return tuple(result)

    def _validate_native_layout(self) -> None:
        for name, expected in (
            ("deformable_bodies", self._deformable_bodies),
            ("affine_bodies", self._affine_bodies),
        ):
            actual = getattr(self._session, name, None)
            if actual is None:
                continue
            normalized = tuple(
                (
                    str(value["path"]),
                    int(value["element_offset"]),
                    int(value["element_count"]),
                )
                for value in actual
            )
            expected_values = tuple(
                (
                    str(value["path"]),
                    int(value["element_offset"]),
                    int(value["element_count"]),
                )
                for value in expected
            )
            if normalized != expected_values:
                raise RuntimeError(
                    f"native libuipc batch {name} layout does not match the artifact"
                )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="libuipc-batch",
            device=self._device_name,
            device_native=self._device.type == "cuda",
            graph_capture=False,
            masked_reset=False,
            fixed_capacity=True,
            runtime_checkpoint=True,
            exact_contact_wrench=self.config.contact_constitution == "ipc",
            sensor_batch=False,
            solver_substeps=True,
            graph_capture_reason=(
                "libuipc shards stage affine targets and exact contact wrenches "
                "through host memory"
            ),
            reset_scope="full_batch_only",
        )

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def fixed_time_step(self) -> float:
        return self.config.solver.fixed_time_step

    @property
    def gravity(self) -> tuple[float, float, float]:
        return self.config.solver.gravity

    @property
    def runtime_fingerprint(self) -> str:
        return self._runtime_fingerprint

    @property
    def graph_captured(self) -> bool:
        return False

    @property
    def shard_count(self) -> int:
        return self._num_envs // self.config.environments_per_shard

    @property
    def arrays(self) -> Mapping[str, Any]:
        self._require_open()
        return self._arrays

    @property
    def deformable_bodies(self) -> tuple[Mapping[str, Any], ...]:
        return self._deformable_bodies

    @property
    def affine_bodies(self) -> tuple[Mapping[str, Any], ...]:
        return self._affine_bodies

    @property
    def capacities(self) -> Mapping[str, int]:
        static_collider_count = sum(
            not bool(collider["disabled"])
            for collider in self.artifact.static_colliders
        )
        return MappingProxyType(
            {
                "environments": self._num_envs,
                "shards": self.shard_count,
                "environments_per_shard": self.config.environments_per_shard,
                "deformable_bodies_per_env": len(self._deformable_bodies),
                "deformable_vertices_per_env": self._arrays["positions"].shape[1],
                "affine_bodies_per_env": len(self._affine_bodies),
                "static_colliders_per_env": static_collider_count,
            }
        )

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        self._require_open()
        native = getattr(self._session, "diagnostics", {})
        values = dict(native() if callable(native) else native)
        values.update(
            {
                "provider": "libuipc-batch",
                "device": self._device_name,
                "num_envs": self._num_envs,
                "shard_count": self.shard_count,
                "frame": int(values.get("frame", self._frame)),
                "graph_captured": False,
                "graph_capture_reason": self.capabilities.graph_capture_reason,
                "affine_target_staging": "per_shard_device_host_device",
                "contact_wrench_staging": "per_shard_device_host_device",
                "contact_constitution": self.config.contact_constitution,
                "feedback_source": (
                    "proxy_constraint"
                    if self.wrench_source == "pose_error"
                    else "native_contact_wrench"
                ),
                "exact_contact_wrench": self.capabilities.exact_contact_wrench,
                "checkpoint_active": self._checkpoint_frame is not None,
                "static_collider_count": sum(
                    not bool(collider["disabled"])
                    for collider in self.artifact.static_colliders
                ),
                "reset_scope": "full_batch_only",
                "stable_storage": True,
            }
        )
        return MappingProxyType(values)

    def set_affine_targets(self, targets: Any) -> None:
        self._require_open()
        target = self._torch.as_tensor(
            targets,
            dtype=self._arrays["affine_targets"].dtype,
            device=self._device,
        )
        expected = tuple(self._arrays["affine_targets"].shape)
        if tuple(target.shape) != expected:
            raise ValueError(
                f"libuipc affine targets must have shape {expected}, "
                f"got {tuple(target.shape)}"
            )
        storage = self._arrays["affine_targets"]
        if target is not storage:
            storage.copy_(target)

    def set_affine_target_twists(self, twists: Any) -> None:
        self._require_open()
        target = self._torch.as_tensor(
            twists,
            dtype=self._arrays["affine_target_twists"].dtype,
            device=self._device,
        )
        expected = tuple(self._arrays["affine_target_twists"].shape)
        if tuple(target.shape) != expected:
            raise ValueError(
                f"libuipc affine target twists must have shape {expected}, "
                f"got {tuple(target.shape)}"
            )
        storage = self._arrays["affine_target_twists"]
        if target is not storage:
            storage.copy_(target)

    def step(self, *, nsteps: int = 1) -> Mapping[str, Any]:
        self._require_open()
        try:
            count = operator.index(nsteps)
        except TypeError as error:
            raise TypeError("libuipc batch step count must be an integer") from error
        if isinstance(nsteps, bool) or count <= 0:
            raise ValueError("libuipc batch step count must be positive")
        self._session.step(count)
        self._frame += count
        return self._arrays

    def capture_checkpoint(self) -> None:
        self._require_open()
        if self._checkpoint_frame is not None:
            raise RuntimeError("libuipc batch checkpoint slot is already active")
        capture = getattr(self._session, "capture_checkpoint", None)
        if not callable(capture):
            raise RuntimeError("native libuipc batch session has no checkpoint API")
        capture()
        self._checkpoint_frame = self._frame

    def rewind_checkpoint(self) -> Mapping[str, Any]:
        self._require_open()
        if self._checkpoint_frame is None:
            raise RuntimeError("libuipc batch checkpoint slot is not active")
        rewind = getattr(self._session, "rewind_checkpoint", None)
        if not callable(rewind):
            raise RuntimeError("native libuipc batch session has no checkpoint API")
        rewind()
        self._frame = self._checkpoint_frame
        return self._arrays

    def commit_checkpoint(self) -> None:
        self._require_open()
        if self._checkpoint_frame is None:
            raise RuntimeError("libuipc batch checkpoint slot is not active")
        commit = getattr(self._session, "commit_checkpoint", None)
        if not callable(commit):
            raise RuntimeError("native libuipc batch session has no checkpoint API")
        commit()
        self._checkpoint_frame = None

    def reset(self, reset_mask: Any | None = None) -> Mapping[str, Any]:
        self._require_open()
        if reset_mask is not None:
            mask = self._torch.as_tensor(
                reset_mask, dtype=self._torch.bool, device=self._device
            )
            if tuple(mask.shape) != (self._num_envs,):
                raise ValueError(
                    f"reset mask must have shape ({self._num_envs},), "
                    f"got {tuple(mask.shape)}"
                )
            if not bool(mask.all().item()):
                raise NotImplementedError(
                    "partial reset is unsupported because libuipc cannot restore "
                    "individual environments within a shard"
                )
        self._session.reset()
        self._frame = 0
        self._checkpoint_frame = None
        self._arrays["affine_target_twists"].zero_()
        self._arrays["affine_contact_wrenches"].zero_()
        return self._arrays

    def synchronize(self) -> None:
        self._require_open()
        synchronize = getattr(self._session, "synchronize", None)
        if callable(synchronize):
            synchronize()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._generation += 1
        self._close_session()
        self._arrays = MappingProxyType({})
        self._native_buffers = MappingProxyType({})

    def _close_session(self) -> None:
        session = getattr(self, "_session", None)
        self._session = None
        if session is not None:
            session.close()

    def _require_open(self) -> None:
        if self._closed or self._session is None:
            raise RuntimeError("libuipc batch solver is closed")


__all__ = [
    "LibuipcBatchConfig",
    "LibuipcBatchSolver",
]
