"""MuJoCo Warp provider built from Gobot's compiled MJCF artifact."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from .base import (
    BatchPhysicsProvider,
    BatchProviderCapabilities,
    CompiledSceneArtifact,
    GraphInvalidatedError,
    ProviderUnavailableError,
    SimulationCapacityError,
)


@dataclass(frozen=True)
class MuJoCoWarpProviderAvailability:
    available: bool
    reason: str = ""


@dataclass(frozen=True)
class MuJoCoWarpRobotLayout:
    """Resolved Gobot names for one robot in a compiled MuJoCo model."""

    robot_name: str
    runtime_prefix: str
    base_link: str
    joint_names: tuple[str, ...]
    link_names: tuple[str, ...]
    sensor_names: tuple[str, ...]
    base_body_id: int
    joint_ids: tuple[int, ...]
    joint_qpos_addresses: tuple[int, ...]
    joint_dof_addresses: tuple[int, ...]
    actuator_ids: tuple[int, ...]
    actuator_modes: tuple[str, ...]
    link_body_ids: tuple[int, ...]
    sensor_ids: tuple[int, ...]
    sensor_addresses: tuple[int, ...]
    sensor_dimensions: tuple[int, ...]
    site_names: tuple[str, ...] = ()
    geom_names: tuple[str, ...] = ()
    site_ids: tuple[int, ...] = ()
    geom_ids: tuple[int, ...] = ()
    site_body_ids: tuple[int, ...] = ()
    geom_body_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class MuJoCoWarpContactSensorSpec:
    """Runtime contact sensor compiled below Gobot's scene boundary.

    Names are runtime MJCF names from :class:`CompiledSceneArtifact`. A spec is
    an observation resource; it does not become an authoring source or mutate
    the edited Gobot scene.
    """

    name: str
    primary_type: Literal["geom", "body", "subtree"]
    primary_names: tuple[str, ...]
    secondary_type: Literal["geom", "body", "subtree"] | None = None
    secondary_name: str | None = None
    fields: tuple[Literal["found", "force", "torque", "dist", "pos", "normal", "tangent"], ...] = (
        "found",
        "force",
    )
    reduce: Literal["none", "mindist", "maxforce", "netforce"] = "maxforce"
    num_slots: int = 1


@dataclass(frozen=True)
class MuJoCoWarpRaycastSensorSpec:
    """CUDA raycast description attached to compiled MuJoCo frames."""

    name: str
    frame_type: Literal["body", "site", "geom"]
    frame_names: tuple[str, ...]
    local_offsets: tuple[tuple[float, float, float], ...]
    local_directions: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, -1.0),)
    alignment: Literal["base", "yaw", "world"] = "yaw"
    max_distance: float = 10.0
    exclude_parent_body: bool = True
    include_geom_groups: tuple[int, ...] | None = (0,)


@dataclass
class _MuJoCoWarpRaycastRuntime:
    spec: MuJoCoWarpRaycastSensorSpec
    frame_ids: tuple[int, ...]
    frame_body_ids: tuple[int, ...]
    local_offsets: Any
    local_directions: Any
    ray_pnt: Any
    ray_vec: Any
    ray_dist: Any
    ray_geomid: Any
    ray_normal: Any
    bodyexclude: Any
    geomgroup: Any
    ray_pnt_tensor: Any
    ray_vec_tensor: Any
    distances: Any
    normals_w: Any
    world_origins: Any | None = None
    world_rays: Any | None = None
    frame_pos_w: Any | None = None
    frame_mat_w: Any | None = None
    hit_pos_w: Any | None = None


@dataclass(frozen=True)
class _MuJoCoWarpBindings:
    mujoco: Any
    mujoco_warp: Any
    warp: Any
    torch: Any


_OVERFLOW_NAMES = {
    1 << 0: "constraint rows (njmax)",
    1 << 1: "constraint Jacobian nonzeros (njmax_nnz)",
    1 << 2: "broadphase contacts",
    1 << 3: "narrowphase contacts (nconmax)",
    1 << 4: "CCD contacts",
    1 << 5: "height-field contacts",
    1 << 6: "contact-match sensors",
    1 << 7: "island dofs (nvmax)",
}

_CONTACT_FIELD_BITS = {
    "found": 0,
    "force": 1,
    "torque": 2,
    "dist": 3,
    "pos": 4,
    "normal": 5,
    "tangent": 6,
}
_CONTACT_FIELD_DIMS = {
    "found": 1,
    "force": 3,
    "torque": 3,
    "dist": 1,
    "pos": 3,
    "normal": 3,
    "tangent": 3,
}
_CONTACT_REDUCE = {
    "none": 0,
    "mindist": 1,
    "maxforce": 2,
    "netforce": 3,
}


def _major_minor(version: str) -> tuple[int, int] | None:
    try:
        parts = str(version).split(".")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None


class MuJoCoWarpProvider(BatchPhysicsProvider):
    """Persistent MuJoCo Warp session over CUDA-resident simulation arrays.

    The provider never reads MJCF from the project directly. It consumes the
    artifact compiled from the active Gobot scene, then keeps model, data,
    reset mask, Torch views, and captured graphs alive for the whole session.
    """

    def __init__(
        self,
        artifact: Mapping[str, Any] | CompiledSceneArtifact,
        *,
        num_envs: int,
        device: str = "cuda:0",
        nconmax: int | None = None,
        njmax: int | None = None,
        njmax_nnz: int | None = None,
        contact_sensor_maxmatch: int | None = None,
        ls_parallel: bool | None = True,
        capture_graphs: bool = True,
        overflow_check_interval: int = 256,
        strict_mujoco_version: bool = True,
        contact_sensors: Sequence[MuJoCoWarpContactSensorSpec] = (),
        raycast_sensors: Sequence[MuJoCoWarpRaycastSensorSpec] = (),
        _bindings: _MuJoCoWarpBindings | None = None,
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be positive")
        if nconmax is not None and int(nconmax) < 0:
            raise ValueError("nconmax must be non-negative")
        if njmax is not None and int(njmax) < 0:
            raise ValueError("njmax must be non-negative")
        if njmax_nnz is not None and int(njmax_nnz) < 0:
            raise ValueError("njmax_nnz must be non-negative")
        if contact_sensor_maxmatch is not None and int(contact_sensor_maxmatch) <= 0:
            raise ValueError("contact_sensor_maxmatch must be positive")
        if int(overflow_check_interval) < 0:
            raise ValueError("overflow_check_interval must be non-negative")

        self.artifact = (
            artifact
            if isinstance(artifact, CompiledSceneArtifact)
            else CompiledSceneArtifact.from_mapping(artifact)
        )
        self._bindings = _bindings if _bindings is not None else self._load_bindings()
        self._mujoco = self._bindings.mujoco
        self._mjw = self._bindings.mujoco_warp
        self._wp = self._bindings.warp
        self._torch = self._bindings.torch
        self._num_envs = int(num_envs)
        self._device_name = str(device)
        self._torch_device = self._torch.device(self._device_name)
        self._capture_enabled = bool(capture_graphs)
        self._overflow_check_interval = int(overflow_check_interval)
        self._step_count = 0
        self._generation = 1
        self._closed = False
        self._graphs: dict[str, Any] = {}
        self._actuator_index_cache: dict[tuple[int, ...], Any] = {}
        self._model_view_cache: dict[str, Any] = {}
        self._contact_specs = tuple(contact_sensors)
        self._raycast_specs = tuple(raycast_sensors)
        self._contact_sensor_ranges: dict[str, dict[str, tuple[int, int, int]]] = {}
        self._contact_sensor_views: Mapping[str, Mapping[str, Any]] = MappingProxyType({})
        self._raycast_runtimes: dict[str, _MuJoCoWarpRaycastRuntime] = {}
        self._raycast_views: Mapping[str, Mapping[str, Any]] = MappingProxyType({})
        self._render_context = None

        if self._torch_device.type != "cuda":
            raise ProviderUnavailableError(
                "MuJoCo Warp is a CUDA provider; use a cuda device or select MuJoCoCpu explicitly."
            )
        if not self._torch.cuda.is_available():
            raise ProviderUnavailableError("MuJoCo Warp requested but Torch cannot access a CUDA device.")

        self._wp.init()
        self._wp_device = self._wp.get_device(self._device_name)
        if not bool(getattr(self._wp_device, "is_cuda", False)):
            raise ProviderUnavailableError(f"Warp device {self._device_name!r} is not CUDA-capable.")

        runtime_version = str(self._mujoco.mj_versionString())
        artifact_version = self.artifact.backend_version
        if (
            strict_mujoco_version
            and artifact_version
            and _major_minor(runtime_version) != _major_minor(artifact_version)
        ):
            raise ProviderUnavailableError(
                "MuJoCo version mismatch: Gobot compiled the artifact with "
                f"{artifact_version}, but the Python Warp runtime uses {runtime_version}."
            )

        source_model = self._mujoco.MjModel.from_xml_string(self.artifact.content)
        self._validate_artifact_dimensions(source_model)
        self._mj_model = self._compile_runtime_model(source_model)
        self._mj_data = self._mujoco.MjData(self._mj_model)
        # Warp 1.14 cannot begin graph capture on a Torch-owned external stream.
        # Build, warm up, and capture on Warp's native stream; captured graphs
        # can still be replayed on the current Torch stream at runtime.
        with self._wp.ScopedDevice(self._wp_device):
            self._model = self._mjw.put_model(self._mj_model)
            if contact_sensor_maxmatch is not None:
                self._model.opt.contact_sensor_maxmatch = int(contact_sensor_maxmatch)
            if ls_parallel is not None:
                try:
                    self._model.opt.ls_parallel = bool(ls_parallel)
                except AttributeError:
                    if not ls_parallel:
                        raise ProviderUnavailableError(
                            "this MuJoCo Warp runtime no longer supports disabling parallel line search"
                        )
            self._data = self._mjw.put_data(
                self._mj_model,
                self._mj_data,
                nworld=self._num_envs,
                nconmax=nconmax,
                njmax=njmax,
                njmax_nnz=njmax_nnz,
            )
            self._reset_mask = self._wp.zeros(
                self._num_envs,
                dtype=self._wp.bool,
                device=self._wp_device,
            )
            self._reset_mask_tensor = self._wp.to_torch(self._reset_mask)
            self._arrays = MappingProxyType(self._make_torch_views())
            self._contact_sensor_views = self._make_contact_sensor_views()
            self._setup_raycast_sensors()
            self._warmup()
            if self._capture_enabled:
                self._capture_all_graphs()

        self._storage_signature = self._capture_storage_signature()

    @classmethod
    def from_context(cls, context: Any, **kwargs: Any) -> "MuJoCoWarpProvider":
        compile_artifact = getattr(context, "compile_scene_artifact", None)
        if compile_artifact is None:
            raise RuntimeError("Gobot AppContext has no compile-only scene artifact API")
        artifact = compile_artifact()
        return cls(artifact, **kwargs)

    @staticmethod
    def availability() -> MuJoCoWarpProviderAvailability:
        missing = [
            name
            for name in ("mujoco", "mujoco_warp", "warp", "torch")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            return MuJoCoWarpProviderAvailability(
                False,
                "missing Python package(s): " + ", ".join(missing),
            )
        return MuJoCoWarpProviderAvailability(True)

    @staticmethod
    def _load_bindings() -> _MuJoCoWarpBindings:
        availability = MuJoCoWarpProvider.availability()
        if not availability.available:
            raise ProviderUnavailableError(
                "MuJoCo Warp requested but its dependencies are unavailable: "
                f"{availability.reason}. Reinstall gobot in the active environment."
            )
        return _MuJoCoWarpBindings(
            mujoco=importlib.import_module("mujoco"),
            mujoco_warp=importlib.import_module("mujoco_warp"),
            warp=importlib.import_module("warp"),
            torch=importlib.import_module("torch"),
        )

    @property
    def capabilities(self) -> BatchProviderCapabilities:
        return BatchProviderCapabilities(
            name="MuJoCoWarp",
            device=self._device_name,
            device_native=True,
            graph_capture=self._capture_enabled,
            masked_reset=True,
            fixed_capacity=True,
        )

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def arrays(self) -> Mapping[str, Any]:
        self._require_open()
        return self._arrays

    @property
    def contact_sensors(self) -> Mapping[str, Mapping[str, Any]]:
        self._require_open()
        return self._contact_sensor_views

    @property
    def raycast_sensors(self) -> Mapping[str, Mapping[str, Any]]:
        self._require_open()
        return self._raycast_views

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def capacities(self) -> Mapping[str, int]:
        self._require_open()
        return MappingProxyType(
            {
                "nconmax": int(self._data.naconmax // self._num_envs),
                "njmax": int(self._data.njmax),
                "njmax_nnz": int(self._data.njmax_nnz),
            }
        )

    def step(self, actions: Any | None = None, *, nsteps: int = 1) -> Mapping[str, Any]:
        self._require_open()
        self._validate_storage()
        if int(nsteps) < 0:
            raise ValueError("nsteps must be non-negative")
        if actions is not None:
            self._copy_actions(actions)
        with self._stream_scope():
            for _ in range(int(nsteps)):
                self._launch_or_call("step", self._mjw.step)
        self._step_count += 1
        if self._overflow_check_interval and self._step_count % self._overflow_check_interval == 0:
            self.assert_no_overflow()
        return self._arrays

    def reset(
        self,
        reset_mask: Any,
        *,
        qpos: Any | None = None,
        qvel: Any | None = None,
        ctrl: Any | None = None,
        forward: bool = True,
    ) -> Mapping[str, Any]:
        self._require_open()
        self._validate_storage()
        mask = self._as_tensor(reset_mask, dtype=self._torch.bool)
        if tuple(mask.shape) != (self._num_envs,):
            raise ValueError(f"reset mask must have shape ({self._num_envs},), got {tuple(mask.shape)}")
        self._reset_mask_tensor.copy_(mask)
        with self._stream_scope():
            self._launch_or_call("reset", self._mjw.reset_data, self._reset_mask)
            self._copy_masked_state("qpos", qpos, mask)
            self._copy_masked_state("qvel", qvel, mask)
            self._copy_masked_state("ctrl", ctrl, mask)
            if forward:
                self._launch_or_call("forward", self._mjw.forward)
        self._reset_mask_tensor.zero_()
        return self._arrays

    def forward(self) -> Mapping[str, Any]:
        self._require_open()
        self._validate_storage()
        with self._stream_scope():
            self._launch_or_call("forward", self._mjw.forward)
        return self._arrays

    def sense(self) -> Mapping[str, Any]:
        self._require_open()
        self._validate_storage()
        self._prepare_raycasts()
        with self._stream_scope():
            if self._capture_enabled:
                self._wp.capture_launch(self._graphs["sense"])
            else:
                self._sense_kernel()
        self._postprocess_raycasts()
        return self._arrays

    def contact_sensor(self, name: str) -> Mapping[str, Any]:
        self._require_open()
        try:
            return self._contact_sensor_views[str(name)]
        except KeyError as error:
            raise KeyError(f"MuJoCo Warp provider has no contact sensor {name!r}") from error

    def raycast_sensor(self, name: str) -> Mapping[str, Any]:
        self._require_open()
        try:
            return self._raycast_views[str(name)]
        except KeyError as error:
            raise KeyError(f"MuJoCo Warp provider has no raycast sensor {name!r}") from error

    def resolve_object_ids(self, object_type: Literal["body", "joint", "geom", "site", "sensor"], names: Sequence[str]) -> tuple[int, ...]:
        self._require_open()
        enum = self._object_type_enum(object_type)
        return tuple(self._required_name_id(enum, str(name), object_type) for name in names)

    def model_constant(self, name: str) -> Any:
        """Copy one compiled MuJoCo model field to the provider device."""

        self._require_open()
        if not hasattr(self._mj_model, str(name)):
            raise KeyError(f"compiled MuJoCo model has no field {name!r}")
        value = getattr(self._mj_model, str(name))
        return self._torch.as_tensor(value, device=self._torch_device).clone()

    def expand_model_fields(self, names: Sequence[str]) -> None:
        """Expand selected Warp model arrays to one independently writable row per world."""

        self._require_open()
        changed = False
        with self._wp.ScopedDevice(self._wp_device):
            for name_value in names:
                name = str(name_value)
                if not hasattr(self._model, name):
                    raise KeyError(f"MuJoCo Warp model has no field {name!r}")
                value = getattr(self._model, name)
                shape = tuple(int(dim) for dim in value.shape)
                if not shape:
                    raise ValueError(f"MuJoCo Warp model field {name!r} is not an array")
                if shape[0] == self._num_envs:
                    self._model_view_cache[name] = self._wp.to_torch(value)
                    continue
                if shape[0] != 1:
                    raise ValueError(
                        f"MuJoCo Warp model field {name!r} has leading dimension {shape[0]}, "
                        f"expected 1 or {self._num_envs}"
                    )
                repeated = value.numpy().repeat(self._num_envs, axis=0)
                expanded = self._wp.array(repeated, dtype=value.dtype, device=self._wp_device)
                setattr(self._model, name, expanded)
                self._model_view_cache[name] = self._wp.to_torch(expanded)
                changed = True
        if changed and self._capture_enabled:
            self._capture_all_graphs()
        self._storage_signature = self._capture_storage_signature()

    def model_array(self, name: str) -> Any:
        self._require_open()
        key = str(name)
        value = self._model_view_cache.get(key)
        if value is None:
            if not hasattr(self._model, key):
                raise KeyError(f"MuJoCo Warp model has no field {name!r}")
            value = self._wp.to_torch(getattr(self._model, key))
            self._model_view_cache[key] = value
            self._storage_signature = self._capture_storage_signature()
        return value

    def recompute_constants(self, level: Literal["set_const", "set_const_0", "set_const_fixed"] = "set_const") -> None:
        self._require_open()
        operation = getattr(self._mjw, str(level), None)
        if operation is None:
            raise ValueError(f"unsupported MuJoCo Warp constant recomputation level {level!r}")
        with self._stream_scope():
            operation(self._model, self._data)

    def resolve_robot_layout(
        self,
        robot_name: str,
        *,
        base_link: str,
        joint_names: Sequence[str],
        link_names: Sequence[str] = (),
        sensor_names: Sequence[str] = (),
        site_names: Sequence[str] = (),
        geom_names: Sequence[str] = (),
    ) -> MuJoCoWarpRobotLayout:
        self._require_open()
        prefix = self.artifact.robot_prefix(robot_name)
        joint_names = tuple(str(name) for name in joint_names)
        link_names = tuple(str(name) for name in link_names)
        sensor_names = tuple(str(name) for name in sensor_names)
        site_names = tuple(str(name) for name in site_names)
        geom_names = tuple(str(name) for name in geom_names)
        joint_ids = tuple(
            self._required_name_id(self._mujoco.mjtObj.mjOBJ_JOINT, prefix + name, "joint")
            for name in joint_names
        )
        actuators = tuple(
            self._joint_actuator(prefix + name)
            for name in joint_names
        )
        link_body_ids = tuple(
            self._required_name_id(self._mujoco.mjtObj.mjOBJ_BODY, prefix + name, "link")
            for name in link_names
        )
        sensor_ids = tuple(
            self._required_name_id(self._mujoco.mjtObj.mjOBJ_SENSOR, prefix + name, "sensor")
            for name in sensor_names
        )
        site_ids = tuple(
            self._required_name_id(self._mujoco.mjtObj.mjOBJ_SITE, prefix + name, "site")
            for name in site_names
        )
        geom_ids = tuple(
            self._required_name_id(self._mujoco.mjtObj.mjOBJ_GEOM, prefix + name, "geom")
            for name in geom_names
        )
        return MuJoCoWarpRobotLayout(
            robot_name=str(robot_name),
            runtime_prefix=prefix,
            base_link=str(base_link),
            joint_names=joint_names,
            link_names=link_names,
            sensor_names=sensor_names,
            base_body_id=self._required_name_id(
                self._mujoco.mjtObj.mjOBJ_BODY,
                prefix + str(base_link),
                "base link",
            ),
            joint_ids=joint_ids,
            joint_qpos_addresses=tuple(int(self._mj_model.jnt_qposadr[index]) for index in joint_ids),
            joint_dof_addresses=tuple(int(self._mj_model.jnt_dofadr[index]) for index in joint_ids),
            actuator_ids=tuple(actuator_id for actuator_id, _ in actuators),
            actuator_modes=tuple(mode for _, mode in actuators),
            link_body_ids=link_body_ids,
            sensor_ids=sensor_ids,
            sensor_addresses=tuple(int(self._mj_model.sensor_adr[index]) for index in sensor_ids),
            sensor_dimensions=tuple(int(self._mj_model.sensor_dim[index]) for index in sensor_ids),
            site_names=site_names,
            geom_names=geom_names,
            site_ids=site_ids,
            geom_ids=geom_ids,
            site_body_ids=tuple(int(self._mj_model.site_bodyid[index]) for index in site_ids),
            geom_body_ids=tuple(int(self._mj_model.geom_bodyid[index]) for index in geom_ids),
        )

    def set_joint_position_targets(self, layout: MuJoCoWarpRobotLayout, targets: Any) -> None:
        invalid_modes = sorted({mode for mode in layout.actuator_modes if mode != "position"})
        if invalid_modes:
            raise ValueError(
                "joint position targets require position actuators; layout also contains "
                + ", ".join(invalid_modes)
            )
        self.set_joint_controls(layout, targets)

    def set_joint_controls(self, layout: MuJoCoWarpRobotLayout, controls: Any) -> None:
        self._require_open()
        target = self._as_tensor(controls, dtype=self._arrays["ctrl"].dtype)
        expected = (self._num_envs, len(layout.actuator_ids))
        if tuple(target.shape) != expected:
            raise ValueError(f"joint controls must have shape {expected}, got {tuple(target.shape)}")
        actuator_ids = self._actuator_index_cache.get(layout.actuator_ids)
        if actuator_ids is None:
            actuator_ids = self._torch.as_tensor(
                layout.actuator_ids,
                dtype=self._torch.long,
                device=self._torch_device,
            )
            self._actuator_index_cache[layout.actuator_ids] = actuator_ids
        self._arrays["ctrl"].index_copy_(1, actuator_ids, target)

    def assert_no_overflow(self) -> None:
        self._require_open()
        overflow = self._arrays.get("overflow")
        if overflow is None:
            return
        active = self._torch.nonzero(overflow != 0, as_tuple=False).flatten()
        if active.numel() == 0:
            return
        world_ids = active[:10].detach().cpu().tolist()
        values = overflow[active[:10]].detach().cpu().tolist()
        details = []
        for world_id, mask in zip(world_ids, values, strict=True):
            names = [name for flag, name in _OVERFLOW_NAMES.items() if int(mask) & flag]
            details.append(f"env {world_id}: {', '.join(names) if names else f'unknown mask {mask}'}")
        raise SimulationCapacityError(
            "MuJoCo Warp fixed capacity overflow; increase nconmax/njmax/njmax_nnz and rebuild the "
            f"session ({'; '.join(details)})."
        )

    def assert_finite(self, names: Sequence[str] = ("qpos", "qvel")) -> None:
        self._require_open()
        for name in names:
            value = self._arrays.get(str(name))
            if value is None:
                raise KeyError(f"MuJoCo Warp session has no array {name!r}")
            finite = self._torch.isfinite(value)
            invalid = ~(
                finite
                if value.ndim <= 1
                else finite.all(dim=tuple(range(1, value.ndim)))
            )
            if bool(invalid.any()):
                env_ids = self._torch.nonzero(invalid, as_tuple=False).flatten()[:10].cpu().tolist()
                raise FloatingPointError(f"MuJoCo Warp array {name!r} is non-finite in envs {env_ids}")

    def synchronize(self) -> None:
        self._require_open()
        self._wp.synchronize_device(self._wp_device)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._wp.synchronize_device(self._wp_device)
        finally:
            self._closed = True
            self._generation += 1
            self._graphs.clear()
            self._actuator_index_cache.clear()
            self._model_view_cache.clear()
            self._raycast_runtimes.clear()
            self._contact_sensor_views = MappingProxyType({})
            self._raycast_views = MappingProxyType({})
            self._render_context = None
            self._arrays = MappingProxyType({})
            self._reset_mask_tensor = None
            self._reset_mask = None
            self._data = None
            self._model = None
            self._mj_data = None
            self._mj_model = None

    def _stream_scope(self):
        if self._torch_device.type != "cuda" or not hasattr(self._wp, "stream_from_torch"):
            return self._wp.ScopedDevice(self._wp_device)
        torch_stream = self._torch.cuda.current_stream(self._torch_device)
        warp_stream = self._wp.stream_from_torch(torch_stream)
        return self._wp.ScopedStream(warp_stream, sync_enter=False)

    def _compile_runtime_model(self, source_model: Any) -> Any:
        if not self._contact_specs:
            return source_model

        spec_names: set[str] = set()
        model_spec = self._mujoco.MjSpec.from_string(self.artifact.content)
        pending_ranges: dict[str, dict[str, list[str]]] = {}
        for spec_index, sensor_spec in enumerate(self._contact_specs):
            self._validate_contact_sensor_spec(sensor_spec, spec_names, source_model)
            field_sensors: dict[str, list[str]] = {}
            for field in sensor_spec.fields:
                field_names: list[str] = []
                for primary_index, primary_name in enumerate(sensor_spec.primary_names):
                    runtime_name = f"__gobot_mw_contact_{spec_index}_{field}_{primary_index}"
                    kwargs: dict[str, Any] = {
                        "name": runtime_name,
                        "type": self._mujoco.mjtSensor.mjSENS_CONTACT,
                        "objtype": self._object_type_enum(sensor_spec.primary_type),
                        "objname": primary_name,
                        "intprm": [
                            1 << _CONTACT_FIELD_BITS[field],
                            _CONTACT_REDUCE[sensor_spec.reduce],
                            int(sensor_spec.num_slots),
                        ],
                    }
                    if sensor_spec.secondary_name is not None:
                        kwargs["reftype"] = self._object_type_enum(sensor_spec.secondary_type)
                        kwargs["refname"] = sensor_spec.secondary_name
                    model_spec.add_sensor(**kwargs)
                    field_names.append(runtime_name)
                field_sensors[field] = field_names
            pending_ranges[sensor_spec.name] = field_sensors

        model = model_spec.compile()
        for sensor_spec in self._contact_specs:
            field_ranges: dict[str, tuple[int, int, int]] = {}
            for field, sensor_names in pending_ranges[sensor_spec.name].items():
                starts: list[int] = []
                ends: list[int] = []
                field_dim = _CONTACT_FIELD_DIMS[field]
                expected_sensor_dim = int(sensor_spec.num_slots) * field_dim
                for sensor_name in sensor_names:
                    sensor = model.sensor(sensor_name)
                    start = int(sensor.adr[0])
                    dimension = int(sensor.dim[0])
                    if dimension != expected_sensor_dim:
                        raise RuntimeError(
                            f"contact sensor {sensor_spec.name!r} field {field!r} compiled to "
                            f"dimension {dimension}, expected {expected_sensor_dim}"
                        )
                    starts.append(start)
                    ends.append(start + dimension)
                for previous_end, next_start in zip(ends, starts[1:], strict=False):
                    if previous_end != next_start:
                        raise RuntimeError(
                            f"contact sensor {sensor_spec.name!r} field {field!r} is not contiguous"
                        )
                field_ranges[field] = (starts[0], ends[-1], field_dim)
            self._contact_sensor_ranges[sensor_spec.name] = field_ranges
        return model

    def _validate_contact_sensor_spec(
        self,
        sensor_spec: MuJoCoWarpContactSensorSpec,
        names: set[str],
        model: Any,
    ) -> None:
        if not sensor_spec.name or sensor_spec.name in names:
            raise ValueError(f"contact sensor names must be non-empty and unique, got {sensor_spec.name!r}")
        names.add(sensor_spec.name)
        if not sensor_spec.primary_names:
            raise ValueError(f"contact sensor {sensor_spec.name!r} has no primary names")
        if int(sensor_spec.num_slots) <= 0:
            raise ValueError(f"contact sensor {sensor_spec.name!r} num_slots must be positive")
        if sensor_spec.reduce == "netforce" and int(sensor_spec.num_slots) != 1:
            raise ValueError(f"contact sensor {sensor_spec.name!r} netforce reduction requires num_slots=1")
        if not sensor_spec.fields or len(set(sensor_spec.fields)) != len(sensor_spec.fields):
            raise ValueError(f"contact sensor {sensor_spec.name!r} fields must be non-empty and unique")
        if (sensor_spec.secondary_type is None) != (sensor_spec.secondary_name is None):
            raise ValueError(
                f"contact sensor {sensor_spec.name!r} must set both secondary_type and secondary_name"
            )
        primary_enum = self._object_type_enum(sensor_spec.primary_type)
        for name in sensor_spec.primary_names:
            if int(self._mujoco.mj_name2id(model, primary_enum, name)) < 0:
                raise KeyError(f"contact sensor {sensor_spec.name!r} has unknown primary {name!r}")
        if sensor_spec.secondary_name is not None:
            secondary_enum = self._object_type_enum(sensor_spec.secondary_type)
            if int(self._mujoco.mj_name2id(model, secondary_enum, sensor_spec.secondary_name)) < 0:
                raise KeyError(
                    f"contact sensor {sensor_spec.name!r} has unknown secondary {sensor_spec.secondary_name!r}"
                )

    def _make_contact_sensor_views(self) -> Mapping[str, Mapping[str, Any]]:
        if not self._contact_sensor_ranges:
            return MappingProxyType({})
        sensordata = self._arrays.get("sensordata")
        if sensordata is None:
            raise RuntimeError("runtime contact sensors require MuJoCo sensordata")
        result: dict[str, Mapping[str, Any]] = {}
        for sensor_spec in self._contact_specs:
            fields: dict[str, Any] = {}
            for field, (start, end, field_dim) in self._contact_sensor_ranges[sensor_spec.name].items():
                value = sensordata[:, start:end].view(
                    self._num_envs,
                    len(sensor_spec.primary_names) * int(sensor_spec.num_slots),
                    field_dim,
                )
                fields[field] = value.squeeze(-1) if field_dim == 1 else value
            result[sensor_spec.name] = MappingProxyType(fields)
        return MappingProxyType(result)

    def _setup_raycast_sensors(self) -> None:
        if not self._raycast_specs:
            self._raycast_views = MappingProxyType({})
            return
        names: set[str] = set()
        enabled_groups: set[int] = set()
        for sensor_spec in self._raycast_specs:
            self._validate_raycast_sensor_spec(sensor_spec, names)
            if sensor_spec.include_geom_groups is None:
                enabled_groups.update(range(6))
            else:
                enabled_groups.update(int(group) for group in sensor_spec.include_geom_groups)

        self._render_context = self._mjw.create_render_context(
            mjm=self._mj_model,
            nworld=self._num_envs,
            cam_res=None,
            render_rgb=None,
            render_depth=None,
            use_textures=False,
            use_shadows=False,
            enabled_geom_groups=sorted(enabled_groups),
            cam_active=[False] * int(self._mj_model.ncam),
            use_precomputed_rays=True,
            render_seg=None,
        )

        vec6 = self._wp.types.vector(length=6, dtype=float)
        raycast_views: dict[str, Mapping[str, Any]] = {}
        for sensor_spec in self._raycast_specs:
            object_enum = self._object_type_enum(sensor_spec.frame_type)
            frame_ids = tuple(
                self._required_name_id(object_enum, name, f"raycast {sensor_spec.frame_type}")
                for name in sensor_spec.frame_names
            )
            if sensor_spec.frame_type == "body":
                frame_body_ids = frame_ids
            elif sensor_spec.frame_type == "site":
                frame_body_ids = tuple(int(self._mj_model.site_bodyid[index]) for index in frame_ids)
            else:
                frame_body_ids = tuple(int(self._mj_model.geom_bodyid[index]) for index in frame_ids)

            local_offsets = self._torch.tensor(
                sensor_spec.local_offsets,
                dtype=self._torch.float32,
                device=self._torch_device,
            )
            local_directions = self._torch.tensor(
                sensor_spec.local_directions,
                dtype=self._torch.float32,
                device=self._torch_device,
            )
            if local_directions.shape[0] == 1:
                local_directions = local_directions.expand(local_offsets.shape[0], 3).clone()
            local_directions = local_directions / local_directions.norm(dim=1, keepdim=True).clamp(min=1.0e-8)

            rays_per_frame = int(local_offsets.shape[0])
            num_rays = len(frame_ids) * rays_per_frame
            ray_pnt = self._wp.zeros(
                (self._num_envs, num_rays), dtype=self._wp.vec3, device=self._wp_device
            )
            ray_vec = self._wp.zeros(
                (self._num_envs, num_rays), dtype=self._wp.vec3, device=self._wp_device
            )
            ray_dist = self._wp.zeros(
                (self._num_envs, num_rays), dtype=float, device=self._wp_device
            )
            ray_geomid = self._wp.zeros(
                (self._num_envs, num_rays), dtype=int, device=self._wp_device
            )
            ray_normal = self._wp.zeros(
                (self._num_envs, num_rays), dtype=self._wp.vec3, device=self._wp_device
            )
            body_excludes: list[int] = []
            for body_id in frame_body_ids:
                body_excludes.extend(
                    [body_id if sensor_spec.exclude_parent_body else -1] * rays_per_frame
                )
            bodyexclude = self._wp.array(
                body_excludes, dtype=int, device=self._wp_device
            )
            group_values = [0, 0, 0, 0, 0, 0]
            if sensor_spec.include_geom_groups is None:
                group_values = [-1] * 6
            else:
                for group in sensor_spec.include_geom_groups:
                    group_values[int(group)] = -1
            geomgroup = vec6(*group_values)
            ray_pnt_tensor = self._wp.to_torch(ray_pnt).view(self._num_envs, num_rays, 3)
            ray_vec_tensor = self._wp.to_torch(ray_vec).view(self._num_envs, num_rays, 3)
            distances = self._wp.to_torch(ray_dist)
            normals_w = self._wp.to_torch(ray_normal).view(self._num_envs, num_rays, 3)
            runtime = _MuJoCoWarpRaycastRuntime(
                spec=sensor_spec,
                frame_ids=frame_ids,
                frame_body_ids=frame_body_ids,
                local_offsets=local_offsets,
                local_directions=local_directions,
                ray_pnt=ray_pnt,
                ray_vec=ray_vec,
                ray_dist=ray_dist,
                ray_geomid=ray_geomid,
                ray_normal=ray_normal,
                bodyexclude=bodyexclude,
                geomgroup=geomgroup,
                ray_pnt_tensor=ray_pnt_tensor,
                ray_vec_tensor=ray_vec_tensor,
                distances=distances,
                normals_w=normals_w,
                frame_pos_w=self._torch.zeros(
                    (self._num_envs, len(frame_ids), 3),
                    dtype=self._torch.float32,
                    device=self._torch_device,
                ),
                hit_pos_w=self._torch.zeros(
                    (self._num_envs, num_rays, 3),
                    dtype=self._torch.float32,
                    device=self._torch_device,
                ),
            )
            self._raycast_runtimes[sensor_spec.name] = runtime
            raycast_views[sensor_spec.name] = MappingProxyType(
                {
                    "distances": runtime.distances,
                    "normals_w": runtime.normals_w,
                    "hit_pos_w": runtime.hit_pos_w,
                    "frame_pos_w": runtime.frame_pos_w,
                    "num_frames": len(frame_ids),
                    "num_rays_per_frame": rays_per_frame,
                }
            )
        self._raycast_views = MappingProxyType(raycast_views)

    def _validate_raycast_sensor_spec(
        self,
        sensor_spec: MuJoCoWarpRaycastSensorSpec,
        names: set[str],
    ) -> None:
        if not sensor_spec.name or sensor_spec.name in names:
            raise ValueError(f"raycast sensor names must be non-empty and unique, got {sensor_spec.name!r}")
        names.add(sensor_spec.name)
        if not sensor_spec.frame_names:
            raise ValueError(f"raycast sensor {sensor_spec.name!r} has no frames")
        if not sensor_spec.local_offsets:
            raise ValueError(f"raycast sensor {sensor_spec.name!r} has no rays")
        if len(sensor_spec.local_directions) not in (1, len(sensor_spec.local_offsets)):
            raise ValueError(
                f"raycast sensor {sensor_spec.name!r} directions must contain one value or one per offset"
            )
        if not math.isfinite(float(sensor_spec.max_distance)) or float(sensor_spec.max_distance) <= 0.0:
            raise ValueError(f"raycast sensor {sensor_spec.name!r} max_distance must be positive")
        if sensor_spec.include_geom_groups is not None:
            invalid = [group for group in sensor_spec.include_geom_groups if int(group) < 0 or int(group) > 5]
            if invalid:
                raise ValueError(f"raycast sensor {sensor_spec.name!r} has invalid geom groups {invalid}")

    def _prepare_raycasts(self) -> None:
        if not self._raycast_runtimes:
            return
        for runtime in self._raycast_runtimes.values():
            positions: list[Any] = []
            matrices: list[Any] = []
            for frame_id in runtime.frame_ids:
                if runtime.spec.frame_type == "body":
                    positions.append(self._arrays["xpos"][:, frame_id])
                    matrices.append(self._arrays["xmat"][:, frame_id].reshape(self._num_envs, 3, 3))
                elif runtime.spec.frame_type == "site":
                    positions.append(self._arrays["site_xpos"][:, frame_id])
                    matrices.append(self._arrays["site_xmat"][:, frame_id].reshape(self._num_envs, 3, 3))
                else:
                    positions.append(self._arrays["geom_xpos"][:, frame_id])
                    matrices.append(self._arrays["geom_xmat"][:, frame_id].reshape(self._num_envs, 3, 3))
            frame_pos = self._torch.stack(positions, dim=1)
            frame_mat = self._torch.stack(matrices, dim=1)
            rotation = self._ray_alignment_rotation(frame_mat, runtime.spec.alignment)
            world_offsets = self._torch.einsum("bfij,nj->bfni", rotation, runtime.local_offsets)
            world_origins = frame_pos[:, :, None, :] + world_offsets
            world_rays = self._torch.einsum("bfij,nj->bfni", rotation, runtime.local_directions)
            runtime.world_origins = world_origins.reshape(self._num_envs, -1, 3)
            runtime.world_rays = world_rays.reshape(self._num_envs, -1, 3)
            runtime.frame_mat_w = frame_mat
            assert runtime.frame_pos_w is not None
            runtime.frame_pos_w.copy_(frame_pos)
            runtime.ray_pnt_tensor.copy_(runtime.world_origins)
            runtime.ray_vec_tensor.copy_(runtime.world_rays)

    def _sense_kernel(self) -> None:
        self._mjw.sensor_pos(self._model, self._data)
        self._mjw.sensor_vel(self._model, self._data)
        self._mjw.sensor_acc(self._model, self._data)
        if self._render_context is None:
            return
        self._mjw.refit_bvh(self._model, self._data, self._render_context)
        for runtime in self._raycast_runtimes.values():
            self._mjw.rays(
                m=self._model,
                d=self._data,
                pnt=runtime.ray_pnt,
                vec=runtime.ray_vec,
                geomgroup=runtime.geomgroup,
                flg_static=True,
                bodyexclude=runtime.bodyexclude,
                dist=runtime.ray_dist,
                geomid=runtime.ray_geomid,
                normal=runtime.ray_normal,
                rc=self._render_context,
            )

    def _postprocess_raycasts(self) -> None:
        for runtime in self._raycast_runtimes.values():
            assert runtime.world_origins is not None and runtime.world_rays is not None
            assert runtime.hit_pos_w is not None
            runtime.distances.masked_fill_(runtime.distances > float(runtime.spec.max_distance), -1.0)
            hit = runtime.distances >= 0.0
            runtime.normals_w.masked_fill_(~hit.unsqueeze(-1), 0.0)
            self._torch.mul(
                runtime.world_rays,
                runtime.distances.clamp(min=0.0).unsqueeze(-1),
                out=runtime.hit_pos_w,
            )
            runtime.hit_pos_w.add_(runtime.world_origins)

    def _ray_alignment_rotation(self, frame_mat: Any, alignment: str) -> Any:
        if alignment == "base":
            return frame_mat
        if alignment == "world":
            return self._torch.eye(
                3, dtype=frame_mat.dtype, device=frame_mat.device
            ).view(1, 1, 3, 3).expand(frame_mat.shape[0], frame_mat.shape[1], 3, 3)
        x_axis = frame_mat[..., :, 0]
        x_projection = self._torch.stack(
            (x_axis[..., 0], x_axis[..., 1], self._torch.zeros_like(x_axis[..., 0])),
            dim=-1,
        )
        x_norm = x_projection.norm(dim=-1, keepdim=True)
        y_axis = frame_mat[..., :, 1]
        y_projection = self._torch.stack(
            (y_axis[..., 0], y_axis[..., 1], self._torch.zeros_like(y_axis[..., 0])),
            dim=-1,
        )
        y_projection = y_projection / y_projection.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
        x_from_y = self._torch.stack(
            (y_projection[..., 1], -y_projection[..., 0], self._torch.zeros_like(y_projection[..., 0])),
            dim=-1,
        )
        x_projection = self._torch.where(x_norm < 0.1, x_from_y, x_projection)
        x_projection = x_projection / x_projection.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
        result = self._torch.zeros_like(frame_mat)
        result[..., 0, 0] = x_projection[..., 0]
        result[..., 1, 0] = x_projection[..., 1]
        result[..., 0, 1] = -x_projection[..., 1]
        result[..., 1, 1] = x_projection[..., 0]
        result[..., 2, 2] = 1.0
        return result

    def _warmup(self) -> None:
        self._mjw.forward(self._model, self._data)
        self._mjw.step(self._model, self._data)
        self._mjw.reset_data(self._model, self._data)
        self._mjw.reset_data(self._model, self._data, reset=self._reset_mask)
        self._prepare_raycasts()
        self._sense_kernel()
        self._postprocess_raycasts()
        self._mjw.forward(self._model, self._data)
        self._wp.synchronize_device(self._wp_device)

    def _capture_all_graphs(self) -> None:
        self._graphs["step"] = self._capture(lambda: self._mjw.step(self._model, self._data))
        self._graphs["forward"] = self._capture(lambda: self._mjw.forward(self._model, self._data))
        self._graphs["reset"] = self._capture(
            lambda: self._mjw.reset_data(self._model, self._data, reset=self._reset_mask)
        )

        self._prepare_raycasts()
        self._graphs["sense"] = self._capture(self._sense_kernel)

    def _capture(self, operation) -> Any:
        with self._wp.ScopedCapture(device=self._wp_device) as capture:
            operation()
        return capture.graph

    def _launch_or_call(self, graph_name: str, operation, *args: Any) -> None:
        if self._capture_enabled:
            self._wp.capture_launch(self._graphs[graph_name])
        else:
            operation(self._model, self._data, *args)

    def _make_torch_views(self) -> dict[str, Any]:
        values: dict[str, Any] = {"reset_mask": self._reset_mask_tensor}
        for name in (
            "time",
            "qpos",
            "qvel",
            "qacc",
            "ctrl",
            "act",
            "xfrc_applied",
            "xpos",
            "xquat",
            "xmat",
            "xipos",
            "cvel",
            "subtree_com",
            "site_xpos",
            "site_xmat",
            "geom_xpos",
            "geom_xmat",
            "sensordata",
            "actuator_force",
            "qfrc_actuator",
            "ncon",
            "nefc",
            "overflow",
        ):
            value = getattr(self._data, name, None)
            if value is not None:
                values[name] = self._wp.to_torch(value)
        for required in ("qpos", "qvel", "ctrl"):
            if required not in values:
                raise RuntimeError(f"MuJoCo Warp data has no required {required!r} array")
        return values

    def _capture_storage_signature(self) -> tuple[Any, ...]:
        values: list[tuple[str, Any]] = list(self._arrays.items())
        values.extend((f"model:{name}", value) for name, value in self._model_view_cache.items())
        for name, runtime in self._raycast_runtimes.items():
            values.extend(
                (
                    (f"ray:{name}:pnt", runtime.ray_pnt_tensor),
                    (f"ray:{name}:vec", runtime.ray_vec_tensor),
                    (f"ray:{name}:dist", runtime.distances),
                    (f"ray:{name}:normal", runtime.normals_w),
                )
            )
        return tuple(
            (
                name,
                id(value),
                value.data_ptr(),
                tuple(value.shape),
                tuple(value.stride()),
                str(value.dtype),
                str(value.device),
            )
            for name, value in values
        )

    def _validate_storage(self) -> None:
        if self._capture_storage_signature() != self._storage_signature:
            raise GraphInvalidatedError(
                "MuJoCo Warp session storage changed after graph capture; rebuild the session."
            )

    def _copy_actions(self, actions: Any) -> None:
        action = self._as_tensor(actions, dtype=self._arrays["ctrl"].dtype)
        if tuple(action.shape) != tuple(self._arrays["ctrl"].shape):
            raise ValueError(
                f"control actions must have shape {tuple(self._arrays['ctrl'].shape)}, got {tuple(action.shape)}"
            )
        if action.data_ptr() != self._arrays["ctrl"].data_ptr():
            self._arrays["ctrl"].copy_(action)

    def _copy_masked_state(self, name: str, value: Any | None, mask: Any) -> None:
        if value is None:
            return
        target = self._arrays[name]
        source = self._as_tensor(value, dtype=target.dtype)
        if tuple(source.shape) != tuple(target.shape):
            raise ValueError(f"{name} reset values must have shape {tuple(target.shape)}")
        mask_shape = (self._num_envs,) + (1,) * (target.ndim - 1)
        self._torch.where(mask.reshape(mask_shape), source, target, out=target)

    def _as_tensor(self, value: Any, *, dtype: Any) -> Any:
        if type(value).__module__.partition(".")[0] == "torch":
            return value.detach().to(device=self._torch_device, dtype=dtype)
        return self._torch.as_tensor(value, dtype=dtype, device=self._torch_device)

    def _required_name_id(self, object_type: Any, runtime_name: str, label: str) -> int:
        object_id = int(self._mujoco.mj_name2id(self._mj_model, object_type, runtime_name))
        if object_id < 0:
            raise KeyError(f"compiled MuJoCo artifact has no {label} {runtime_name!r}")
        return object_id

    def _object_type_enum(self, object_type: str | None) -> Any:
        values = {
            "body": self._mujoco.mjtObj.mjOBJ_BODY,
            "subtree": self._mujoco.mjtObj.mjOBJ_XBODY,
            "joint": self._mujoco.mjtObj.mjOBJ_JOINT,
            "geom": self._mujoco.mjtObj.mjOBJ_GEOM,
            "site": self._mujoco.mjtObj.mjOBJ_SITE,
            "sensor": self._mujoco.mjtObj.mjOBJ_SENSOR,
        }
        try:
            return values[str(object_type)]
        except KeyError as error:
            raise ValueError(f"unsupported MuJoCo object type {object_type!r}") from error

    def _joint_actuator(self, prefixed_joint_name: str) -> tuple[int, str]:
        for suffix, mode in (
            ("_position", "position"),
            ("_motor", "motor"),
            ("_velocity", "velocity"),
        ):
            actuator_id = int(
                self._mujoco.mj_name2id(
                    self._mj_model,
                    self._mujoco.mjtObj.mjOBJ_ACTUATOR,
                    prefixed_joint_name + suffix,
                )
            )
            if actuator_id >= 0:
                return actuator_id, mode
        raise KeyError(f"compiled MuJoCo artifact has no actuator for joint {prefixed_joint_name!r}")

    def _validate_artifact_dimensions(self, model: Any) -> None:
        model_dimensions = {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "nbody": int(model.nbody),
            "njoint": int(model.njnt),
            "ngeom": int(model.ngeom),
            "nsensor": int(model.nsensor),
            "nhfield": int(model.nhfield),
        }
        for name, actual in model_dimensions.items():
            expected = self.artifact.dimensions.get(name)
            if expected is not None and int(expected) != actual:
                raise ValueError(
                    f"compiled scene artifact {name} mismatch: metadata says {expected}, "
                    f"MJCF contains {actual}"
                )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("MuJoCo Warp provider is closed")


__all__ = [
    "MuJoCoWarpContactSensorSpec",
    "MuJoCoWarpProvider",
    "MuJoCoWarpProviderAvailability",
    "MuJoCoWarpRaycastSensorSpec",
    "MuJoCoWarpRobotLayout",
]
