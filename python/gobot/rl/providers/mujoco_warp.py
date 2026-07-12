"""MuJoCo Warp provider built from Gobot's compiled MJCF artifact."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from types import MappingProxyType
from typing import Any, Mapping, Sequence

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
        capture_graphs: bool = True,
        overflow_check_interval: int = 256,
        strict_mujoco_version: bool = True,
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

        self._mj_model = self._mujoco.MjModel.from_xml_string(self.artifact.content)
        self._validate_artifact_dimensions()
        self._mj_data = self._mujoco.MjData(self._mj_model)
        # Warp 1.14 cannot begin graph capture on a Torch-owned external stream.
        # Build, warm up, and capture on Warp's native stream; captured graphs
        # can still be replayed on the current Torch stream at runtime.
        with self._wp.ScopedDevice(self._wp_device):
            self._model = self._mjw.put_model(self._mj_model)
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
            self._warmup()
            if self._capture_enabled:
                self._capture_all_graphs()

        self._reset_mask_tensor = self._wp.to_torch(self._reset_mask)
        self._arrays = MappingProxyType(self._make_torch_views())
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
                "MuJoCo Warp requested but its optional dependencies are unavailable: "
                f"{availability.reason}. Install gobot[mujoco-warp]."
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
            self._launch_or_call("forward", self._mjw.forward)
        self._reset_mask_tensor.zero_()
        return self._arrays

    def sense(self) -> Mapping[str, Any]:
        self._require_open()
        self._validate_storage()
        with self._stream_scope():
            if self._capture_enabled:
                self._wp.capture_launch(self._graphs["sense"])
            else:
                self._mjw.sensor_pos(self._model, self._data)
                self._mjw.sensor_vel(self._model, self._data)
                self._mjw.sensor_acc(self._model, self._data)
        return self._arrays

    def resolve_robot_layout(
        self,
        robot_name: str,
        *,
        base_link: str,
        joint_names: Sequence[str],
        link_names: Sequence[str] = (),
        sensor_names: Sequence[str] = (),
    ) -> MuJoCoWarpRobotLayout:
        self._require_open()
        prefix = self.artifact.robot_prefix(robot_name)
        joint_names = tuple(str(name) for name in joint_names)
        link_names = tuple(str(name) for name in link_names)
        sensor_names = tuple(str(name) for name in sensor_names)
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

    def _warmup(self) -> None:
        self._mjw.forward(self._model, self._data)
        self._mjw.step(self._model, self._data)
        self._mjw.reset_data(self._model, self._data)
        self._mjw.reset_data(self._model, self._data, reset=self._reset_mask)
        self._mjw.sensor_pos(self._model, self._data)
        self._mjw.sensor_vel(self._model, self._data)
        self._mjw.sensor_acc(self._model, self._data)
        self._mjw.forward(self._model, self._data)
        self._wp.synchronize_device(self._wp_device)

    def _capture_all_graphs(self) -> None:
        self._graphs["step"] = self._capture(lambda: self._mjw.step(self._model, self._data))
        self._graphs["forward"] = self._capture(lambda: self._mjw.forward(self._model, self._data))
        self._graphs["reset"] = self._capture(
            lambda: self._mjw.reset_data(self._model, self._data, reset=self._reset_mask)
        )

        def sense() -> None:
            self._mjw.sensor_pos(self._model, self._data)
            self._mjw.sensor_vel(self._model, self._data)
            self._mjw.sensor_acc(self._model, self._data)

        self._graphs["sense"] = self._capture(sense)

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
            "cvel",
            "sensordata",
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
            for name, value in self._arrays.items()
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

    def _validate_artifact_dimensions(self) -> None:
        model_dimensions = {
            "nq": int(self._mj_model.nq),
            "nv": int(self._mj_model.nv),
            "nu": int(self._mj_model.nu),
            "nbody": int(self._mj_model.nbody),
            "njoint": int(self._mj_model.njnt),
            "ngeom": int(self._mj_model.ngeom),
            "nsensor": int(self._mj_model.nsensor),
            "nhfield": int(self._mj_model.nhfield),
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
    "MuJoCoWarpProvider",
    "MuJoCoWarpProviderAvailability",
    "MuJoCoWarpRobotLayout",
]
