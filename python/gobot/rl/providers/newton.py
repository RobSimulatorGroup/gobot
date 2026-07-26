"""Newton GPU provider built from Gobot's compiled MJCF artifact."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import importlib.util
import math
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence
import xml.etree.ElementTree as ET

from .base import (
    BatchPhysicsProvider,
    BatchProviderCapabilities,
    CompiledSceneArtifact,
    ProviderUnavailableError,
    SimulationCapacityError,
)


@dataclass(frozen=True)
class NewtonProviderAvailability:
    """Result of checking whether the optional Newton runtime can be loaded."""

    available: bool
    reason: str = ""


@dataclass(frozen=True)
class _NewtonBindings:
    newton: Any
    warp: Any
    torch: Any
    mujoco: Any
    mujoco_warp: Any


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


def _major_minor(value: str) -> tuple[int, int] | None:
    try:
        parts = str(value).split(".")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None


def _validated_artifact(
    artifact: Mapping[str, Any] | CompiledSceneArtifact,
) -> CompiledSceneArtifact:
    if not isinstance(artifact, CompiledSceneArtifact):
        return CompiledSceneArtifact.from_mapping(artifact)
    return CompiledSceneArtifact.from_mapping(
        {
            "schema_version": artifact.schema_version,
            "backend": artifact.backend,
            "format": artifact.format,
            "content": artifact.content,
            "content_digest": artifact.digest,
            "backend_version": artifact.backend_version,
            "dimensions": artifact.dimensions,
            "robot_names": artifact.robot_names,
            "robot_prefixes": artifact.robot_prefixes,
            "terrain_geom_groups": artifact.terrain_geom_groups,
        }
    )


def _normalize_geom_solref(root: ET.Element) -> bool:
    """Expand one-value geom solref attributes using MuJoCo defaults."""

    changed = False
    class_second_value = {"main": "1"}

    def resolve_solref(element: ET.Element, inherited_second: str) -> str:
        nonlocal changed
        solref = element.attrib.get("solref")
        if solref is None:
            return inherited_second
        values = solref.split()
        if len(values) == 1:
            element.set("solref", f"{values[0]} {inherited_second}")
            changed = True
            return inherited_second
        if len(values) >= 2:
            return values[1]
        return inherited_second

    def visit_default(
        default: ET.Element,
        parent_class: str,
        inherited_second: str,
    ) -> None:
        class_name = default.attrib.get("class", parent_class)
        effective_second = inherited_second
        default_geom = next((child for child in default if child.tag == "geom"), None)
        if default_geom is not None:
            effective_second = resolve_solref(default_geom, inherited_second)
        class_second_value[class_name] = effective_second

        for child in default:
            if child.tag == "default":
                visit_default(child, class_name, effective_second)

    for default in root.findall("default"):
        visit_default(default, "main", class_second_value["main"])

    def visit_body_children(parent: ET.Element, inherited_class: str) -> None:
        child_class = parent.attrib.get("childclass", inherited_class)
        for child in parent:
            if child.tag == "geom":
                class_name = child.attrib.get("class", child_class)
                resolve_solref(
                    child,
                    class_second_value.get(class_name, class_second_value["main"]),
                )
            elif child.tag == "body":
                visit_body_children(child, child_class)

    for worldbody in root.findall("worldbody"):
        visit_body_children(worldbody, "main")

    return changed


@contextmanager
def _prepare_mjcf_for_newton(mjcf: str) -> Iterator[str]:
    """Normalize valid MuJoCo shorthand and materialize Newton 1.4 mesh assets."""

    try:
        root = ET.fromstring(mjcf)
    except ET.ParseError as error:
        raise ValueError(f"compiled scene artifact contains invalid MJCF: {error}") from error

    # Newton 1.4's raw custom-attribute path fills an omitted second geom
    # solref value with 0 instead of retaining the MuJoCo default, which can
    # make the MuJoCo-Warp constraint solve non-finite.
    normalized_solref = _normalize_geom_solref(root)

    inline_meshes = [
        mesh
        for asset in root.findall("asset")
        for mesh in asset.findall("mesh")
        if "file" not in mesh.attrib and "vertex" in mesh.attrib
    ]
    if not inline_meshes:
        yield ET.tostring(root, encoding="unicode") if normalized_solref else mjcf
        return

    with tempfile.TemporaryDirectory(prefix="gobot-newton-mesh-") as directory:
        directory_path = Path(directory)
        for mesh_index, mesh in enumerate(inline_meshes):
            vertex_tokens = mesh.attrib["vertex"].split()
            face_tokens = mesh.attrib.get("face", "").split()
            if len(vertex_tokens) % 3 != 0 or not vertex_tokens:
                raise ValueError("inline MJCF mesh vertex data must contain xyz triples")
            if len(face_tokens) % 3 != 0 or not face_tokens:
                raise ValueError("inline MJCF mesh face data must contain triangle triples")
            try:
                vertices = [float(value) for value in vertex_tokens]
                faces = [int(value) for value in face_tokens]
            except ValueError as error:
                raise ValueError("inline MJCF mesh contains a non-numeric vertex or face") from error

            vertex_count = len(vertices) // 3
            if any(index < 0 or index >= vertex_count for index in faces):
                raise ValueError("inline MJCF mesh face references an invalid vertex")

            mesh_path = directory_path / f"mesh_{mesh_index}.obj"
            lines = [
                f"v {vertices[index]} {vertices[index + 1]} {vertices[index + 2]}"
                for index in range(0, len(vertices), 3)
            ]
            lines.extend(
                f"f {faces[index] + 1} {faces[index + 1] + 1} {faces[index + 2] + 1}"
                for index in range(0, len(faces), 3)
            )
            mesh_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            mesh.set("file", str(mesh_path))
            for attribute in (
                "vertex",
                "normal",
                "texcoord",
                "face",
                "facenormal",
                "facetexcoord",
            ):
                mesh.attrib.pop(attribute, None)

        yield ET.tostring(root, encoding="unicode")


class NewtonProvider(BatchPhysicsProvider):
    """Persistent Newton/MuJoCo-Warp session over CUDA-resident arrays.

    Gobot remains the authoring source of truth. The provider consumes the
    versioned MJCF artifact compiled by Gobot and creates a private Newton model
    for batched simulation; Newton model and solver objects never enter the
    SceneTree or the public array mapping.
    """

    def __init__(
        self,
        artifact: Mapping[str, Any] | CompiledSceneArtifact,
        *,
        num_envs: int,
        device: str = "cuda:0",
        fixed_time_step: float = 0.002,
        nconmax: int | None = None,
        njmax: int | None = None,
        iterations: int | None = None,
        overflow_check_interval: int = 256,
        strict_mujoco_version: bool = True,
        _bindings: _NewtonBindings | None = None,
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be positive")
        if not math.isfinite(float(fixed_time_step)) or float(fixed_time_step) <= 0.0:
            raise ValueError("fixed_time_step must be finite and positive")
        if nconmax is not None and int(nconmax) < 0:
            raise ValueError("nconmax must be non-negative")
        if njmax is not None and int(njmax) < 0:
            raise ValueError("njmax must be non-negative")
        if iterations is not None and int(iterations) <= 0:
            raise ValueError("iterations must be positive")
        if int(overflow_check_interval) < 0:
            raise ValueError("overflow_check_interval must be non-negative")
        self.artifact = _validated_artifact(artifact)
        self._bindings = _bindings if _bindings is not None else self._load_bindings()
        self._newton = self._bindings.newton
        self._wp = self._bindings.warp
        self._torch = self._bindings.torch
        self._mujoco = self._bindings.mujoco
        self._num_envs = int(num_envs)
        self._device_name = str(device)
        self._fixed_time_step = float(fixed_time_step)
        self._overflow_check_interval = int(overflow_check_interval)
        self._step_count = 0
        self._torch_device = self._torch.device(self._device_name)
        self._closed = False
        self._empty_arrays: dict[str, Any] = {}

        if self._torch_device.type != "cuda":
            raise ProviderUnavailableError(
                "Newton is a CUDA provider; use a cuda device or select MuJoCoCpu explicitly."
            )
        if not self._torch.cuda.is_available():
            raise ProviderUnavailableError("Newton requested but Torch cannot access a CUDA device.")

        runtime_version = str(self._mujoco.mj_versionString())
        if (
            strict_mujoco_version
            and _major_minor(runtime_version) != _major_minor(self.artifact.backend_version)
        ):
            raise ProviderUnavailableError(
                "MuJoCo version mismatch: Gobot compiled the artifact with "
                f"{self.artifact.backend_version}, but Newton uses {runtime_version}."
            )

        solver_type = getattr(getattr(self._newton, "solvers", None), "SolverMuJoCo", None)
        if not callable(getattr(self._newton, "ModelBuilder", None)) or solver_type is None:
            raise ProviderUnavailableError(
                "The installed Newton package does not provide the required Newton 1.4 public API."
            )

        self._wp.init()
        self._wp_device = self._wp.get_device(self._device_name)
        if not bool(getattr(self._wp_device, "is_cuda", False)):
            raise ProviderUnavailableError(f"Warp device {self._device_name!r} is not CUDA-capable.")

        with self._wp.ScopedDevice(self._wp_device):
            blueprint = self._newton.ModelBuilder()
            with _prepare_mjcf_for_newton(self.artifact.content) as mjcf:
                blueprint.add_mjcf(
                    mjcf,
                    ctrl_direct=True,
                    parse_visuals=False,
                )

            builder = self._newton.ModelBuilder()
            solver_type.register_custom_attributes(builder)
            builder.replicate(blueprint, self._num_envs, spacing=(0.0, 0.0, 0.0))
            self._model = builder.finalize(device=self._wp_device)
            self._validate_model_dimensions()

            solver_kwargs: dict[str, Any] = {"use_mujoco_contacts": True}
            if nconmax is not None:
                solver_kwargs["nconmax"] = int(nconmax)
            if njmax is not None:
                solver_kwargs["njmax"] = int(njmax)
            if iterations is not None:
                solver_kwargs["iterations"] = int(iterations)
            self._solver = solver_type(self._model, **solver_kwargs)
            self._validate_solver_dimensions()
            self._solver_data = getattr(self._solver, "mjw_data", None)

            self._state = self._model.state()
            self._next_state = self._model.state()
            self._control = self._model.control()
            self._contacts = None
            self._newton.eval_fk(
                self._model,
                self._model.joint_q,
                self._model.joint_qd,
                self._state,
            )
            self._next_state.assign(self._state)

            # Model construction uses Warp's native stream. Complete it before
            # exporting zero-copy views to Torch's current stream.
            self._wp.synchronize_device(self._wp_device)

            self._ctrl_array = self._resolve_ctrl_array()
            self._reset_mask = self._wp.zeros(
                self._num_envs,
                dtype=self._wp.bool,
                device=self._wp_device,
            )
            self._reset_mask_tensor = self._wp.to_torch(self._reset_mask)
            self._arrays = MappingProxyType(self._make_torch_views())
            self._initial_ctrl = self._arrays["ctrl"].clone()
            self._initial_time = (
                self._arrays["time"].clone() if "time" in self._arrays else None
            )
            self._initial_overflow = (
                self._arrays["overflow"].clone() if "overflow" in self._arrays else None
            )

    @classmethod
    def from_context(cls, context: Any, **kwargs: Any) -> "NewtonProvider":
        compile_artifact = getattr(context, "compile_scene_artifact", None)
        if compile_artifact is None:
            raise RuntimeError("Gobot AppContext has no compile-only scene artifact API")
        return cls(compile_artifact(), **kwargs)

    @staticmethod
    def availability() -> NewtonProviderAvailability:
        missing = [
            name
            for name in ("newton", "mujoco", "mujoco_warp", "warp", "torch")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            return NewtonProviderAvailability(
                False,
                "missing Python package(s): " + ", ".join(missing),
            )
        return NewtonProviderAvailability(True)

    @staticmethod
    def _load_bindings() -> _NewtonBindings:
        availability = NewtonProvider.availability()
        if not availability.available:
            raise ProviderUnavailableError(
                "Newton requested but its dependencies are unavailable: "
                f"{availability.reason}. Install Gobot's default dependencies "
                "in the active environment."
            )
        try:
            return _NewtonBindings(
                newton=importlib.import_module("newton"),
                warp=importlib.import_module("warp"),
                torch=importlib.import_module("torch"),
                mujoco=importlib.import_module("mujoco"),
                mujoco_warp=importlib.import_module("mujoco_warp"),
            )
        except Exception as error:
            raise ProviderUnavailableError(
                "Newton dependencies are installed but could not be imported together: "
                f"{type(error).__name__}: {error}"
            ) from error

    @property
    def capabilities(self) -> BatchProviderCapabilities:
        return BatchProviderCapabilities(
            name="Newton",
            device=self._device_name,
            device_native=True,
            graph_capture=False,
            masked_reset=True,
            fixed_capacity=True,
        )

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def fixed_time_step(self) -> float:
        return self._fixed_time_step

    @property
    def arrays(self) -> Mapping[str, Any]:
        self._require_open()
        return self._arrays

    def step(self, actions: Any | None = None, *, nsteps: int = 1) -> Mapping[str, Any]:
        self._require_open()
        if int(nsteps) < 0:
            raise ValueError("nsteps must be non-negative")
        if actions is not None:
            self._copy_actions(actions)

        with self._stream_scope():
            state_in = self._state
            state_out = self._next_state
            for _ in range(int(nsteps)):
                state_in.clear_forces()
                self._solver.step(
                    state_in,
                    state_out,
                    self._control,
                    self._contacts,
                    self._fixed_time_step,
                )
                state_in, state_out = state_out, state_in
            # Keep public Torch views pinned to one allocation, with at most
            # one full state copy per public step call.
            if state_in is not self._state:
                self._state.assign(state_in)
        self._step_count += 1
        if self._overflow_check_interval and self._step_count % self._overflow_check_interval == 0:
            self.assert_no_overflow()
        return self._arrays

    def reset(
        self,
        reset_mask: Any,
        *,
        joint_q: Any | None = None,
        joint_qd: Any | None = None,
        ctrl: Any | None = None,
    ) -> Mapping[str, Any]:
        self._require_open()
        mask = self._as_tensor(reset_mask, dtype=self._torch.bool)
        if tuple(mask.shape) != (self._num_envs,):
            raise ValueError(f"reset mask must have shape ({self._num_envs},), got {tuple(mask.shape)}")
        self._reset_mask_tensor.copy_(mask)

        try:
            with self._stream_scope():
                self._solver.reset(self._state, self._reset_mask)
                if joint_q is not None:
                    self._copy_masked_state("joint_q", joint_q, mask)
                if joint_qd is not None:
                    self._copy_masked_state("joint_qd", joint_qd, mask)
                self._copy_masked_state("ctrl", self._initial_ctrl if ctrl is None else ctrl, mask)
                if self._initial_time is not None:
                    self._copy_masked_state("time", self._initial_time, mask)
                if self._initial_overflow is not None:
                    self._copy_masked_state("overflow", self._initial_overflow, mask)
                self._newton.eval_fk(
                    self._model,
                    self._state.joint_q,
                    self._state.joint_qd,
                    self._state,
                )
                self._next_state.assign(self._state)
        finally:
            self._reset_mask_tensor.zero_()
        return self._arrays

    def synchronize(self) -> None:
        self._require_open()
        self._wp.synchronize_device(self._wp_device)

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
            "Newton/MuJoCo-Warp fixed capacity overflow; increase nconmax/njmax and rebuild "
            f"the provider ({'; '.join(details)})."
        )

    def assert_finite(self, names: Sequence[str] = ("joint_q", "joint_qd")) -> None:
        self._require_open()
        for name in names:
            if name not in self._arrays:
                raise KeyError(f"Newton array {name!r} is unavailable")
            value = self._arrays[name]
            finite = self._torch.isfinite(value)
            finite_by_environment = (
                finite
                if value.ndim <= 1
                else finite.all(dim=tuple(range(1, value.ndim)))
            )
            invalid = self._torch.nonzero(~finite_by_environment, as_tuple=False).flatten()
            if invalid.numel():
                env_ids = invalid[:10].detach().cpu().tolist()
                raise FloatingPointError(
                    f"Newton array {name!r} is non-finite in envs {env_ids}"
                )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._wp.synchronize_device(self._wp_device)
        finally:
            self._closed = True
            self._arrays = MappingProxyType({})
            self._initial_ctrl = None
            self._initial_time = None
            self._initial_overflow = None
            self._reset_mask_tensor = None
            self._reset_mask = None
            self._ctrl_array = None
            self._contacts = None
            self._control = None
            self._next_state = None
            self._state = None
            self._solver = None
            self._solver_data = None
            self._model = None
            self._empty_arrays.clear()

    def _validate_model_dimensions(self) -> None:
        world_count = int(getattr(self._model, "world_count", 0))
        if world_count != self._num_envs:
            raise RuntimeError(
                f"Newton built {world_count} worlds, expected {self._num_envs}"
            )
        expected_q = self._num_envs * int(self.artifact.dimensions["nq"])
        expected_qd = self._num_envs * int(self.artifact.dimensions["nv"])
        actual_q = int(getattr(self._model, "joint_coord_count", -1))
        actual_qd = int(getattr(self._model, "joint_dof_count", -1))
        if actual_q != expected_q or actual_qd != expected_qd:
            raise RuntimeError(
                "Newton MJCF dimensions do not match Gobot's compiled artifact: "
                f"joint_q={actual_q}/{expected_q}, joint_qd={actual_qd}/{expected_qd}"
            )
        body_count = int(getattr(self._model, "body_count", 0))
        if body_count % self._num_envs != 0:
            raise RuntimeError(
                f"Newton body count {body_count} is not divisible by {self._num_envs} worlds"
            )

    def _validate_solver_dimensions(self) -> None:
        model = getattr(self._solver, "mjw_model", None)
        if model is None:
            return
        fields = {
            "nq": "nq",
            "nv": "nv",
            "nu": "nu",
            "nbody": "nbody",
            "njoint": "njnt",
            "ngeom": "ngeom",
            "nhfield": "nhfield",
        }
        for metadata_name, model_name in fields.items():
            expected = self.artifact.dimensions.get(metadata_name)
            actual = getattr(model, model_name, None)
            if expected is not None and actual is not None and int(expected) != int(actual):
                raise RuntimeError(
                    "Newton's MuJoCo model does not preserve the compiled scene artifact: "
                    f"{metadata_name}={int(actual)}/{int(expected)}"
                )

    def _resolve_ctrl_array(self) -> Any:
        namespace = getattr(self._control, "mujoco", None)
        ctrl = getattr(namespace, "ctrl", None) if namespace is not None else None
        expected = self._num_envs * int(self.artifact.dimensions["nu"])
        if ctrl is None:
            if expected:
                raise RuntimeError(
                    "Newton did not create control.mujoco.ctrl for Gobot's MJCF actuators"
                )
            ctrl = self._wp.zeros(0, dtype=float, device=self._wp_device)
            self._empty_arrays["ctrl"] = ctrl
        ctrl_tensor = self._wp.to_torch(ctrl)
        if int(ctrl_tensor.numel()) != expected:
            raise RuntimeError(
                "Newton MJCF actuator dimensions do not match Gobot's compiled artifact: "
                f"ctrl={int(ctrl_tensor.numel())}/{expected}"
            )
        return ctrl

    def _make_torch_views(self) -> dict[str, Any]:
        nq = int(self.artifact.dimensions["nq"])
        nv = int(self.artifact.dimensions["nv"])
        nu = int(self.artifact.dimensions["nu"])
        joint_q = self._state_view("joint_q", nq)
        joint_qd = self._state_view("joint_qd", nv)
        ctrl = self._wp.to_torch(self._ctrl_array).reshape(self._num_envs, nu)

        bodies_per_world = int(self._model.body_count) // self._num_envs
        body_q = self._body_view("body_q", bodies_per_world, 7)
        body_qd = self._body_view("body_qd", bodies_per_world, 6)
        values = {
            "joint_q": joint_q,
            "joint_qd": joint_qd,
            "body_q": body_q,
            "body_qd": body_qd,
            "ctrl": ctrl,
            "reset_mask": self._reset_mask_tensor,
        }
        for name in ("time", "overflow"):
            value = getattr(self._solver_data, name, None) if self._solver_data is not None else None
            if value is not None:
                tensor = self._wp.to_torch(value)
                if int(tensor.numel()) == self._num_envs:
                    tensor = tensor.reshape(self._num_envs)
                values[name] = tensor
        return values

    def _state_view(self, name: str, width: int) -> Any:
        value = getattr(self._state, name, None)
        if value is None:
            if width:
                raise RuntimeError(f"Newton state has no {name} array")
            value = self._wp.zeros(0, dtype=float, device=self._wp_device)
            self._empty_arrays[name] = value
        return self._wp.to_torch(value).reshape(self._num_envs, width)

    def _body_view(self, name: str, body_count: int, component_count: int) -> Any:
        value = getattr(self._state, name, None)
        if value is None:
            if body_count:
                raise RuntimeError(f"Newton state has no {name} array")
            value = self._wp.zeros((0, component_count), dtype=float, device=self._wp_device)
            self._empty_arrays[name] = value
        return self._wp.to_torch(value).reshape(
            self._num_envs,
            body_count,
            component_count,
        )

    def _copy_actions(self, actions: Any) -> None:
        target = self._arrays["ctrl"]
        value = self._as_tensor(actions, dtype=target.dtype)
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(
                f"actions must have shape {tuple(target.shape)}, got {tuple(value.shape)}"
            )
        target.copy_(value)

    def _copy_masked_state(self, name: str, value: Any, mask: Any) -> None:
        target = self._arrays[name]
        source = self._as_tensor(value, dtype=target.dtype)
        if tuple(source.shape) != tuple(target.shape):
            raise ValueError(
                f"{name} must have shape {tuple(target.shape)}, got {tuple(source.shape)}"
            )
        expanded_mask = mask.reshape(
            (self._num_envs,) + (1,) * (int(target.ndim) - 1)
        )
        target.copy_(self._torch.where(expanded_mask, source, target))

    def _as_tensor(self, value: Any, *, dtype: Any) -> Any:
        return self._torch.as_tensor(
            value,
            dtype=dtype,
            device=self._torch_device,
        )

    def _stream_scope(self) -> Any:
        if hasattr(self._wp, "stream_from_torch") and hasattr(self._wp, "ScopedStream"):
            torch_stream = self._torch.cuda.current_stream(self._torch_device)
            warp_stream = self._wp.stream_from_torch(torch_stream)
            return self._wp.ScopedStream(warp_stream, sync_enter=False)
        return self._wp.ScopedDevice(self._wp_device)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Newton provider is closed")


__all__ = [
    "NewtonProvider",
    "NewtonProviderAvailability",
]
