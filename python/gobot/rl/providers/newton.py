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
class NewtonModelConfig:
    """Optional Newton-side overrides for a compiled Gobot MJCF artifact.

    ``None`` preserves the parameter authored by the Gobot scene. Stiffness
    and damping overrides are paired because applying only half of a MuJoCo
    constraint response produces a different physical model. Contact
    overrides apply to geoms; explicit MJCF contact pairs remain authoritative.
    """

    joint_limit_stiffness: float | None = None
    joint_limit_damping: float | None = None
    contact_stiffness: float | None = None
    contact_damping: float | None = None
    contact_friction_stiffness: float | None = None
    default_contact_friction: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "joint_limit_stiffness",
            "joint_limit_damping",
            "contact_stiffness",
            "contact_damping",
            "contact_friction_stiffness",
            "default_contact_friction",
        ):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if (self.joint_limit_stiffness is None) != (
            self.joint_limit_damping is None
        ):
            raise ValueError(
                "joint_limit_stiffness and joint_limit_damping must be set together"
            )
        if (self.contact_stiffness is None) != (self.contact_damping is None):
            raise ValueError(
                "contact_stiffness and contact_damping must be set together"
            )
        for name in (
            "joint_limit_stiffness",
            "joint_limit_damping",
            "contact_stiffness",
            "contact_damping",
        ):
            value = getattr(self, name)
            if value is not None and float(value) <= 0.0:
                raise ValueError(f"{name} must be positive when set")


@dataclass(frozen=True)
class NewtonRobotLayout:
    """Gobot robot names resolved to columns in Newton's public arrays.

    All indices are local to one environment. They index the second dimension
    of :attr:`NewtonProvider.arrays`, rather than exposing Newton or MuJoCo
    model objects to callers.
    """

    artifact_digest: str
    robot_name: str
    runtime_prefix: str
    base_link: str
    joint_names: tuple[str, ...]
    link_names: tuple[str, ...]
    base_body_index: int
    base_joint_q_indices: tuple[int, ...]
    base_joint_qd_indices: tuple[int, ...]
    joint_q_indices: tuple[int, ...]
    joint_qd_indices: tuple[int, ...]
    actuator_indices: tuple[int, ...]
    actuator_modes: tuple[str, ...]
    link_body_indices: tuple[int, ...]


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


def _remove_overridden_mjcf_response(
    root: ET.Element,
    model_config: NewtonModelConfig,
) -> bool:
    """Remove raw MJCF response values superseded by Newton model settings."""

    changed = False
    if model_config.joint_limit_stiffness is not None:
        for joint in root.iter("joint"):
            changed = joint.attrib.pop("solreflimit", None) is not None or changed
    if model_config.contact_stiffness is not None:
        for geom in root.iter("geom"):
            changed = geom.attrib.pop("solref", None) is not None or changed
    return changed


def _configure_blueprint_defaults(
    blueprint: Any,
    model_config: NewtonModelConfig,
) -> None:
    if model_config.joint_limit_stiffness is not None:
        blueprint.default_joint_cfg.limit_ke = float(
            model_config.joint_limit_stiffness
        )
        blueprint.default_joint_cfg.limit_kd = float(model_config.joint_limit_damping)
    if model_config.contact_stiffness is not None:
        blueprint.default_shape_cfg.ke = float(model_config.contact_stiffness)
        blueprint.default_shape_cfg.kd = float(model_config.contact_damping)
    if model_config.contact_friction_stiffness is not None:
        blueprint.default_shape_cfg.kf = float(
            model_config.contact_friction_stiffness
        )
    if model_config.default_contact_friction is not None:
        blueprint.default_shape_cfg.mu = float(model_config.default_contact_friction)


def _apply_imported_joint_limit_response(
    blueprint: Any,
    model_config: NewtonModelConfig,
) -> None:
    """Replace MuJoCo's implicit limit response on articulated DOFs.

    Newton's MJCF importer resolves even an omitted ``solreflimit`` to the
    MuJoCo default. Apply the requested force-space gains after import while
    retaining zero response on free-joint coordinates.
    """

    if model_config.joint_limit_stiffness is None:
        return
    stiffness = float(model_config.joint_limit_stiffness)
    damping = float(model_config.joint_limit_damping)
    for index, (current_stiffness, current_damping) in enumerate(
        zip(blueprint.joint_limit_ke, blueprint.joint_limit_kd, strict=True)
    ):
        if float(current_stiffness) != 0.0 or float(current_damping) != 0.0:
            blueprint.joint_limit_ke[index] = stiffness
            blueprint.joint_limit_kd[index] = damping


def _normalize_position_actuators(root: ET.Element) -> bool:
    """Restore MuJoCo position shortcuts saved as equivalent general actuators.

    ``mj_saveLastXML`` canonicalizes ``<position>`` into an affine ``<general>``.
    Newton intentionally treats every general actuator as direct control, so
    the canonical form would bypass ``Control.joint_target_q``. Convert only
    the exact affine PD form produced by MuJoCo back into the shortcut.
    """

    def values(element: ET.Element, name: str) -> list[float] | None:
        text = element.attrib.get(name)
        if text is None:
            return None
        try:
            result = [float(value) for value in text.split()]
        except ValueError:
            return None
        return result if result and all(math.isfinite(value) for value in result) else None

    def is_zero(value: float) -> bool:
        return abs(value) <= 1.0e-12

    changed = False
    for actuator in root.findall("actuator"):
        for element in actuator:
            if element.tag != "general" or "joint" not in element.attrib:
                continue
            # Gobot preserves the authored USD/MJCF actuator alongside the
            # backend-neutral position drive as ``*_affine``. Only the
            # generated primary drive is safe to restore to Newton's target
            # path; an affine actuator may carry independent dynamics even
            # when MuJoCo resolved it to the same coefficient pattern.
            if not element.attrib.get("name", "").endswith("_position"):
                continue
            if element.attrib.get("biastype") != "affine":
                continue
            if element.attrib.get("gaintype", "fixed") != "fixed":
                continue
            if element.attrib.get("dyntype", "none") != "none":
                continue

            gain = values(element, "gainprm")
            bias = values(element, "biasprm")
            if gain is None or bias is None or len(bias) < 2:
                continue
            kp = gain[0]
            kv = -bias[2] if len(bias) >= 3 else 0.0
            if kp < 0.0 or kv < 0.0:
                continue
            if not all(is_zero(value) for value in gain[1:]):
                continue
            if not is_zero(bias[0]) or not math.isclose(
                bias[1], -kp, rel_tol=1.0e-12, abs_tol=1.0e-12
            ):
                continue
            if not all(is_zero(value) for value in bias[3:]):
                continue

            element.tag = "position"
            element.set("kp", format(kp, ".17g"))
            if kv > 0.0:
                element.set("kv", format(kv, ".17g"))
            else:
                element.attrib.pop("kv", None)
            for attribute in (
                "gainprm",
                "biasprm",
                "gaintype",
                "biastype",
                "dyntype",
                "dynprm",
            ):
                element.attrib.pop(attribute, None)
            changed = True
    return changed


def _mjcf_actuator_modes(root: ET.Element, expected_count: int) -> tuple[str, ...]:
    """Return the semantic input mode of each source MJCF actuator."""

    elements = [element for section in root.findall("actuator") for element in section]
    if len(elements) != expected_count:
        raise RuntimeError(
            "Newton cannot preserve the compiled MJCF actuator layout: "
            f"XML actuators={len(elements)}/{expected_count}"
        )
    return tuple(
        element.tag
        if element.tag in ("position", "velocity") and "joint" in element.attrib
        else "direct"
        for element in elements
    )


@contextmanager
def _prepare_mjcf_for_newton(
    mjcf: str,
    *,
    model_config: NewtonModelConfig | None = None,
) -> Iterator[str]:
    """Normalize valid MuJoCo shorthand and materialize Newton 1.4 mesh assets."""

    try:
        root = ET.fromstring(mjcf)
    except ET.ParseError as error:
        raise ValueError(f"compiled scene artifact contains invalid MJCF: {error}") from error

    model_config = model_config or NewtonModelConfig()
    normalized = _remove_overridden_mjcf_response(root, model_config)

    # Newton 1.4's raw custom-attribute path fills an omitted second geom
    # solref value with 0 instead of retaining the MuJoCo default, which can
    # make the MuJoCo-Warp constraint solve non-finite.
    normalized = _normalize_geom_solref(root) or normalized
    normalized = _normalize_position_actuators(root) or normalized

    inline_meshes = [
        mesh
        for asset in root.findall("asset")
        for mesh in asset.findall("mesh")
        if "file" not in mesh.attrib and "vertex" in mesh.attrib
    ]
    if not inline_meshes:
        yield ET.tostring(root, encoding="unicode") if normalized else mjcf
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
        use_mujoco_contacts: bool = True,
        overflow_check_interval: int = 256,
        strict_mujoco_version: bool = True,
        model_config: NewtonModelConfig | None = None,
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
        if not isinstance(use_mujoco_contacts, bool):
            raise TypeError("use_mujoco_contacts must be a bool")
        if int(overflow_check_interval) < 0:
            raise ValueError("overflow_check_interval must be non-negative")
        if model_config is not None and not isinstance(model_config, NewtonModelConfig):
            raise TypeError("model_config must be a NewtonModelConfig")
        self.artifact = _validated_artifact(artifact)
        self._model_config = model_config or NewtonModelConfig()
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
        self._use_mujoco_contacts = use_mujoco_contacts
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
            _configure_blueprint_defaults(blueprint, self._model_config)
            blueprint.rigid_gap = 0.0
            with _prepare_mjcf_for_newton(
                self.artifact.content,
                model_config=self._model_config,
            ) as mjcf:
                self._metadata_model = self._mujoco.MjModel.from_xml_string(mjcf)
                self._actuator_modes = _mjcf_actuator_modes(
                    ET.fromstring(mjcf),
                    int(self.artifact.dimensions["nu"]),
                )
                blueprint.add_mjcf(
                    mjcf,
                    ctrl_direct=False,
                    parse_visuals=False,
                    # Gobot's compiled MJCF already contains the scene-wide
                    # contact masks. Re-filtering the combined robot + world
                    # import would also disable foot-to-ground contacts.
                    enable_self_collisions=True,
                )
            _apply_imported_joint_limit_response(blueprint, self._model_config)
            blueprint.approximate_meshes("convex_hull")

            builder = self._newton.ModelBuilder()
            solver_type.register_custom_attributes(builder)
            builder.replicate(blueprint, self._num_envs, spacing=(0.0, 0.0, 0.0))
            self._model = builder.finalize(device=self._wp_device)
            self._validate_model_dimensions()

            solver_kwargs: dict[str, Any] = {
                "use_mujoco_cpu": False,
                "solver": "newton",
                "use_mujoco_contacts": self._use_mujoco_contacts,
            }
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
            self._joint_target_q_array = getattr(self._control, "joint_target_q", None)
            if self._joint_target_q_array is None:
                raise RuntimeError("Newton did not create control.joint_target_q")
            self._use_coord_layout_targets = bool(
                getattr(self._model, "use_coord_layout_targets", False)
            )
            self._joint_target_q_width = int(
                self.artifact.dimensions[
                    "nq" if self._use_coord_layout_targets else "nv"
                ]
            )
            self._joint_target_q_tensor = self._wp.to_torch(
                self._joint_target_q_array
            )
            expected_target_q = self._num_envs * self._joint_target_q_width
            if int(self._joint_target_q_tensor.numel()) != expected_target_q:
                raise RuntimeError(
                    "Newton joint position target layout does not match its model: "
                    f"joint_target_q={int(self._joint_target_q_tensor.numel())}/{expected_target_q}"
                )
            self._joint_target_q_tensor = self._joint_target_q_tensor.reshape(
                self._num_envs,
                self._joint_target_q_width,
            )
            self._joint_target_qd_array = getattr(self._control, "joint_target_qd", None)
            if self._joint_target_qd_array is None:
                raise RuntimeError("Newton did not create control.joint_target_qd")
            self._joint_target_qd_tensor = self._wp.to_torch(
                self._joint_target_qd_array
            )
            expected_target_qd = self._num_envs * int(self.artifact.dimensions["nv"])
            if int(self._joint_target_qd_tensor.numel()) != expected_target_qd:
                raise RuntimeError(
                    "Newton joint velocity target layout does not match its model: "
                    f"joint_target_qd={int(self._joint_target_qd_tensor.numel())}/{expected_target_qd}"
                )
            self._joint_target_qd_tensor = self._joint_target_qd_tensor.reshape(
                self._num_envs,
                int(self.artifact.dimensions["nv"]),
            )
            self._contacts = (
                None if self._use_mujoco_contacts else self._model.contacts()
            )
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

            self._direct_ctrl_array = self._resolve_ctrl_array()
            self._direct_ctrl_tensor = self._wp.to_torch(
                self._direct_ctrl_array
            ).reshape(
                self._num_envs,
                int(self.artifact.dimensions["nu"]),
            )
            self._ctrl_array = self._wp.zeros(
                self._num_envs * int(self.artifact.dimensions["nu"]),
                dtype=float,
                device=self._wp_device,
            )
            self._reset_mask = self._wp.zeros(
                self._num_envs,
                dtype=self._wp.bool,
                device=self._wp_device,
            )
            self._reset_mask_tensor = self._wp.to_torch(self._reset_mask)
            self._arrays = MappingProxyType(self._make_torch_views())
            self._index_cache: dict[tuple[int, ...], Any] = {}
            self._mapping_cache: dict[str, tuple[int, ...]] = {}
            self._model_int_cache: dict[str, tuple[int, ...]] = {}
            self._actuator_routes = self._build_actuator_routes()
            self._sync_semantic_controls_from_targets()
            self._initial_ctrl = self._arrays["ctrl"].clone()
            self._initial_joint_target_q = self._joint_target_q_tensor.clone()
            self._initial_joint_target_qd = self._joint_target_qd_tensor.clone()
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
    def use_mujoco_contacts(self) -> bool:
        """Whether SolverMuJoCo uses MuJoCo-Warp instead of Newton contacts."""

        return self._use_mujoco_contacts

    @property
    def model_config(self) -> NewtonModelConfig:
        """Newton-side model overrides used for this provider."""

        return self._model_config

    @property
    def arrays(self) -> Mapping[str, Any]:
        self._require_open()
        return self._arrays

    def resolve_robot_layout(
        self,
        robot_name: str,
        *,
        base_link: str,
        joint_names: Sequence[str],
        link_names: Sequence[str] = (),
    ) -> NewtonRobotLayout:
        """Resolve Gobot names to per-environment Newton array indices."""

        self._require_open()
        robot_name = str(robot_name)
        base_link = str(base_link)
        if not base_link:
            raise ValueError("base_link must not be empty")
        joint_names = self._unique_names(joint_names, "joint")
        link_names = self._unique_names(link_names, "link")
        prefix = self.artifact.robot_prefix(robot_name)
        mj_model = self._metadata_model
        if mj_model is None:
            raise RuntimeError("Newton SolverMuJoCo has no compiled model metadata")

        base_mj_body = self._required_name_id(
            self._mujoco.mjtObj.mjOBJ_BODY,
            prefix + base_link,
            "base link",
        )
        base_body_index = self._body_array_index(base_mj_body)
        base_joint = self._floating_joint_for_body(mj_model, base_mj_body)
        if base_joint is None:
            base_joint_q_indices: tuple[int, ...] = ()
            base_joint_qd_indices: tuple[int, ...] = ()
        else:
            base_joint_q_indices, base_joint_qd_indices = self._joint_array_ranges(
                mj_model,
                base_joint,
            )

        joint_ids = tuple(
            self._required_name_id(
                self._mujoco.mjtObj.mjOBJ_JOINT,
                prefix + name,
                "joint",
            )
            for name in joint_names
        )
        joint_ranges = tuple(
            self._joint_array_ranges(mj_model, joint_id) for joint_id in joint_ids
        )
        for name, (q_indices, qd_indices) in zip(
            joint_names,
            joint_ranges,
            strict=True,
        ):
            if len(q_indices) != 1 or len(qd_indices) != 1:
                raise ValueError(
                    f"Newton named joint controls require a scalar joint; {prefix + name!r} "
                    f"uses {len(q_indices)} position and {len(qd_indices)} velocity coordinates"
                )

        actuators = tuple(self._joint_actuator(prefix + name) for name in joint_names)
        return NewtonRobotLayout(
            artifact_digest=self.artifact.digest,
            robot_name=robot_name,
            runtime_prefix=prefix,
            base_link=base_link,
            joint_names=joint_names,
            link_names=link_names,
            base_body_index=base_body_index,
            base_joint_q_indices=base_joint_q_indices,
            base_joint_qd_indices=base_joint_qd_indices,
            joint_q_indices=tuple(indices[0][0] for indices in joint_ranges),
            joint_qd_indices=tuple(indices[1][0] for indices in joint_ranges),
            actuator_indices=tuple(actuator for actuator, _ in actuators),
            actuator_modes=tuple(mode for _, mode in actuators),
            link_body_indices=tuple(
                self._body_array_index(
                    self._required_name_id(
                        self._mujoco.mjtObj.mjOBJ_BODY,
                        prefix + name,
                        "link",
                    )
                )
                for name in link_names
            ),
        )

    def set_joint_position_targets(
        self,
        layout: NewtonRobotLayout,
        targets: Any,
    ) -> None:
        """Set position-actuator controls for the named joints in ``layout``."""

        self._validate_layout(layout)
        invalid_modes = sorted({mode for mode in layout.actuator_modes if mode != "position"})
        if invalid_modes:
            raise ValueError(
                "joint position targets require position actuators; layout also contains "
                + ", ".join(invalid_modes)
            )
        self.set_joint_controls(layout, targets)

    def set_joint_controls(self, layout: NewtonRobotLayout, controls: Any) -> None:
        """Write controls for the named joints without exposing solver objects."""

        self._validate_layout(layout)
        self._copy_columns(
            self._arrays["ctrl"],
            layout.actuator_indices,
            controls,
            "joint controls",
        )
        self._route_actuator_controls()

    def reset_robot_state(
        self,
        layout: NewtonRobotLayout,
        reset_mask: Any,
        *,
        base_pose: Any | None = None,
        base_velocity: Any | None = None,
        joint_position: Any | None = None,
        joint_velocity: Any | None = None,
        controls: Any | None = None,
    ) -> Mapping[str, Any]:
        """Reset named robot state while preserving unrelated state columns.

        ``base_pose`` uses Newton's public transform layout
        ``[x, y, z, qx, qy, qz, qw]``. Base values are accepted only when the
        selected base link is attached by a floating joint.
        """

        self._validate_layout(layout)
        joint_q = self._arrays["joint_q"].clone()
        joint_qd = self._arrays["joint_qd"].clone()
        ctrl = self._arrays["ctrl"].clone()
        if base_pose is not None:
            if not layout.base_joint_q_indices:
                raise ValueError(f"base link {layout.base_link!r} has no floating pose coordinates")
            self._copy_columns(
                joint_q,
                layout.base_joint_q_indices,
                base_pose,
                "base pose",
            )
        if base_velocity is not None:
            if not layout.base_joint_qd_indices:
                raise ValueError(f"base link {layout.base_link!r} has no floating velocity coordinates")
            self._copy_columns(
                joint_qd,
                layout.base_joint_qd_indices,
                base_velocity,
                "base velocity",
            )
        if joint_position is not None:
            self._copy_columns(
                joint_q,
                layout.joint_q_indices,
                joint_position,
                "joint position",
            )
        if joint_velocity is not None:
            self._copy_columns(
                joint_qd,
                layout.joint_qd_indices,
                joint_velocity,
                "joint velocity",
            )
        if controls is not None:
            self._copy_columns(
                ctrl,
                layout.actuator_indices,
                controls,
                "joint controls",
            )
        return self.reset(
            reset_mask,
            joint_q=joint_q,
            joint_qd=joint_qd,
            ctrl=ctrl,
        )

    def step(self, actions: Any | None = None, *, nsteps: int = 1) -> Mapping[str, Any]:
        self._require_open()
        if int(nsteps) < 0:
            raise ValueError("nsteps must be non-negative")
        if actions is not None:
            self._copy_actions(actions)

        with self._stream_scope():
            # ``arrays["ctrl"]`` uses source-MJCF actuator order. Newton's
            # solver consumes each actuator from one of three backend arrays,
            # so route even when callers mutated the public view directly.
            self._route_actuator_controls()
            state_in = self._state
            state_out = self._next_state
            for _ in range(int(nsteps)):
                state_in.clear_forces()
                if self._contacts is not None:
                    self._model.collide(state_in, self._contacts)
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
        joint_target_q: Any | None = None,
        joint_target_qd: Any | None = None,
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
                self._copy_masked_tensor(
                    self._joint_target_q_tensor,
                    self._initial_joint_target_q,
                    mask,
                    "joint_target_q",
                )
                self._copy_masked_tensor(
                    self._joint_target_qd_tensor,
                    self._initial_joint_target_qd,
                    mask,
                    "joint_target_qd",
                )
                self._route_actuator_controls()
                if joint_target_q is not None:
                    self._copy_masked_tensor(
                        self._joint_target_q_tensor,
                        joint_target_q,
                        mask,
                        "joint_target_q",
                    )
                if joint_target_qd is not None:
                    self._copy_masked_tensor(
                        self._joint_target_qd_tensor,
                        joint_target_qd,
                        mask,
                        "joint_target_qd",
                    )
                if joint_target_q is not None or joint_target_qd is not None:
                    self._sync_semantic_controls_from_targets()
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
            self._initial_joint_target_q = None
            self._initial_joint_target_qd = None
            self._initial_time = None
            self._initial_overflow = None
            self._index_cache.clear()
            self._mapping_cache.clear()
            self._model_int_cache.clear()
            self._metadata_model = None
            self._reset_mask_tensor = None
            self._reset_mask = None
            self._ctrl_array = None
            self._direct_ctrl_array = None
            self._direct_ctrl_tensor = None
            self._joint_target_q_tensor = None
            self._joint_target_q_array = None
            self._joint_target_qd_tensor = None
            self._joint_target_qd_array = None
            self._actuator_routes = None
            self._actuator_modes = ()
            self._use_coord_layout_targets = False
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

    def _build_actuator_routes(
        self,
    ) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
        """Map source-MJCF actuator columns to Newton's three control arrays."""

        route_lists: dict[str, tuple[list[int], list[int]]] = {
            "position": ([], []),
            "velocity": ([], []),
            "direct": ([], []),
        }
        actuator_count = int(self.artifact.dimensions["nu"])
        if actuator_count == 0:
            return {
                mode: (tuple(sources), tuple(destinations))
                for mode, (sources, destinations) in route_lists.items()
            }

        control_sources = self._solver_int_values("mjc_actuator_ctrl_source")
        mapped_indices = self._solver_int_values("mjc_actuator_to_newton_idx")
        target_q_indices = self._solver_int_values(
            "mjc_actuator_to_newton_target_q_idx"
        )
        target_q_axes = self._solver_int_values(
            "mjc_actuator_to_target_q_axis_idx"
        )
        if not (
            len(control_sources)
            == len(mapped_indices)
            == len(target_q_indices)
            == len(target_q_axes)
            == actuator_count
        ):
            raise RuntimeError(
                "Newton cannot preserve the compiled MJCF actuator layout: "
                f"solver routes={len(control_sources)}/{actuator_count}"
            )

        nv = int(self.artifact.dimensions["nv"])
        claimed_routes: set[int] = set()
        for actuator_id, mode in enumerate(self._actuator_modes):
            if mode == "direct":
                candidates = [
                    route
                    for route, (source, mapped) in enumerate(
                        zip(control_sources, mapped_indices, strict=True)
                    )
                    if source == 1 and mapped == actuator_id
                ]
            else:
                joint_id = self._metadata_actuator_joint(actuator_id)
                _, joint_dofs = self._joint_array_ranges(
                    self._metadata_model,
                    joint_id,
                )
                joint_dof_set = set(joint_dofs)
                candidates = []
                for route, (source, mapped) in enumerate(
                    zip(control_sources, mapped_indices, strict=True)
                ):
                    if source != 0:
                        continue
                    if mode == "position" and mapped >= 0:
                        mapped_dof = mapped % nv
                    elif mode == "velocity" and mapped <= -2:
                        mapped_dof = (-(mapped + 2)) % nv
                    else:
                        continue
                    if mapped_dof in joint_dof_set:
                        candidates.append(route)

            candidates = [route for route in candidates if route not in claimed_routes]
            if not candidates:
                raise RuntimeError(
                    "Newton has no compatible control route for source MJCF "
                    f"actuator {actuator_id} ({mode})"
                )
            if mode == "direct" and len(candidates) != 1:
                raise RuntimeError(
                    "Newton produced multiple direct routes for source MJCF "
                    f"actuator {actuator_id}"
                )

            sources, destinations = route_lists[mode]
            for route in candidates:
                claimed_routes.add(route)
                mapped = mapped_indices[route]
                if mode == "position":
                    destination = target_q_indices[route]
                    if destination < 0:
                        raise RuntimeError(
                            f"Newton position actuator route {route} has no target coordinate"
                        )
                    axis = target_q_axes[route]
                    if axis >= 0:
                        if self._use_coord_layout_targets:
                            raise RuntimeError(
                                "source MJCF scalar controls cannot represent a Newton "
                                "ball-joint quaternion target under coord layout"
                            )
                        destination += axis
                    destination %= self._joint_target_q_width
                elif mode == "velocity":
                    destination = (-(mapped + 2)) % nv
                else:
                    destination = mapped
                sources.append(actuator_id)
                destinations.append(destination)

        unclaimed = [
            route
            for route, mapped in enumerate(mapped_indices)
            if route not in claimed_routes and mapped != -1
        ]
        if unclaimed:
            raise RuntimeError(
                "Newton produced actuator routes with no source MJCF control: "
                + ", ".join(str(route) for route in unclaimed[:10])
            )

        routes = {
            mode: (tuple(sources), tuple(destinations))
            for mode, (sources, destinations) in route_lists.items()
        }
        for mode, (_, destinations) in routes.items():
            if len(set(destinations)) != len(destinations):
                raise RuntimeError(
                    "Newton cannot independently route multiple source MJCF "
                    f"{mode} actuators to the same target"
                )
        return routes

    def _route_actuator_controls(self) -> None:
        semantic = self._arrays["ctrl"]
        targets = {
            "position": self._joint_target_q_tensor,
            "velocity": self._joint_target_qd_tensor,
            "direct": self._direct_ctrl_tensor,
        }
        for mode, target in targets.items():
            sources, destinations = self._actuator_routes[mode]
            if destinations:
                self._copy_columns(
                    target,
                    destinations,
                    semantic[:, sources],
                    f"{mode} actuator controls",
                )

    def _sync_semantic_controls_from_targets(self) -> None:
        semantic = self._arrays["ctrl"]
        targets = {
            "position": self._joint_target_q_tensor,
            "velocity": self._joint_target_qd_tensor,
            "direct": self._direct_ctrl_tensor,
        }
        for mode, target in targets.items():
            sources, destinations = self._actuator_routes[mode]
            # A multi-axis joint can fan one source actuator out to several
            # target coordinates. Its first coordinate is the deterministic
            # scalar value exposed by the source-MJCF compatibility view.
            first_destination: dict[int, int] = {}
            for source, destination in zip(sources, destinations, strict=True):
                first_destination.setdefault(source, destination)
            if first_destination:
                semantic_sources = tuple(first_destination)
                target_sources = tuple(first_destination.values())
                self._copy_columns(
                    semantic,
                    semantic_sources,
                    target[:, target_sources],
                    f"{mode} semantic controls",
                )

    def _solver_int_values(self, name: str) -> list[int]:
        value = getattr(self._solver, name, None)
        if value is None:
            raise RuntimeError(f"Newton SolverMuJoCo has no {name} actuator mapping")
        tensor = self._wp.to_torch(value)
        if int(tensor.ndim) != 1:
            raise RuntimeError(f"Newton SolverMuJoCo has an invalid {name} actuator mapping")
        return [int(item) for item in tensor.detach().cpu().tolist()]

    def _metadata_actuator_joint(self, actuator_id: int) -> int:
        transmission = getattr(self._metadata_model, "actuator_trnid", None)
        if transmission is None:
            raise RuntimeError("compiled MuJoCo metadata has no actuator transmission mapping")
        try:
            joint_id = int(transmission[actuator_id, 0])
        except (IndexError, TypeError):
            joint_id = int(transmission[2 * actuator_id])
        if joint_id < 0:
            raise RuntimeError(
                f"source MJCF actuator {actuator_id} has no joint transmission"
            )
        return joint_id

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
        self._copy_masked_tensor(target, value, mask, name)

    def _copy_masked_tensor(self, target: Any, value: Any, mask: Any, name: str) -> None:
        source = self._as_tensor(value, dtype=target.dtype)
        if tuple(source.shape) != tuple(target.shape):
            raise ValueError(
                f"{name} must have shape {tuple(target.shape)}, got {tuple(source.shape)}"
            )
        expanded_mask = mask.reshape(
            (self._num_envs,) + (1,) * (int(target.ndim) - 1)
        )
        target.copy_(self._torch.where(expanded_mask, source, target))

    def _copy_columns(
        self,
        target: Any,
        indices: tuple[int, ...],
        value: Any,
        label: str,
    ) -> None:
        source = self._as_tensor(value, dtype=target.dtype)
        expected = (self._num_envs, len(indices))
        if tuple(source.shape) != expected:
            raise ValueError(f"{label} must have shape {expected}, got {tuple(source.shape)}")
        if not indices:
            return
        index = self._index_cache.get(indices)
        if index is None:
            index = self._torch.as_tensor(
                indices,
                dtype=self._torch.long,
                device=self._torch_device,
            )
            self._index_cache[indices] = index
        target.index_copy_(1, index, source)

    @staticmethod
    def _unique_names(names: Sequence[str], label: str) -> tuple[str, ...]:
        result = tuple(str(name) for name in names)
        if any(not name for name in result):
            raise ValueError(f"{label} names must not be empty")
        if len(set(result)) != len(result):
            raise ValueError(f"{label} names must be unique")
        return result

    def _validate_layout(self, layout: NewtonRobotLayout) -> None:
        self._require_open()
        if not isinstance(layout, NewtonRobotLayout):
            raise TypeError("layout must be a NewtonRobotLayout")
        if layout.artifact_digest != self.artifact.digest:
            raise ValueError("Newton robot layout belongs to a different compiled scene artifact")
        if not (
            len(layout.joint_names)
            == len(layout.joint_q_indices)
            == len(layout.joint_qd_indices)
            == len(layout.actuator_indices)
            == len(layout.actuator_modes)
        ):
            raise ValueError("Newton robot layout has inconsistent joint fields")
        if len(layout.link_names) != len(layout.link_body_indices):
            raise ValueError("Newton robot layout has inconsistent link fields")
        widths = {
            "base joint position": (layout.base_joint_q_indices, self._arrays["joint_q"].shape[1]),
            "base joint velocity": (layout.base_joint_qd_indices, self._arrays["joint_qd"].shape[1]),
            "joint position": (layout.joint_q_indices, self._arrays["joint_q"].shape[1]),
            "joint velocity": (layout.joint_qd_indices, self._arrays["joint_qd"].shape[1]),
            "actuator": (layout.actuator_indices, self._arrays["ctrl"].shape[1]),
            "link body": (layout.link_body_indices, self._arrays["body_q"].shape[1]),
        }
        for label, (indices, width) in widths.items():
            if len(set(indices)) != len(indices) or any(
                index < 0 or index >= int(width) for index in indices
            ):
                raise ValueError(f"Newton robot layout has invalid {label} indices")
        if layout.base_body_index < 0 or layout.base_body_index >= int(
            self._arrays["body_q"].shape[1]
        ):
            raise ValueError("Newton robot layout has an invalid base body index")

    def _required_name_id(self, object_type: Any, runtime_name: str, label: str) -> int:
        object_id = int(
            self._mujoco.mj_name2id(
                self._metadata_model,
                object_type,
                runtime_name,
            )
        )
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
                    self._metadata_model,
                    self._mujoco.mjtObj.mjOBJ_ACTUATOR,
                    prefixed_joint_name + suffix,
                )
            )
            if actuator_id >= 0:
                return actuator_id, mode
        raise KeyError(f"compiled MuJoCo artifact has no actuator for joint {prefixed_joint_name!r}")

    def _mapping_row(self, name: str) -> list[int]:
        cached = self._mapping_cache.get(name)
        if cached is not None:
            return list(cached)
        mapping = getattr(self._solver, name, None)
        if mapping is None:
            raise RuntimeError(f"Newton SolverMuJoCo has no {name} mapping")
        tensor = self._wp.to_torch(mapping)
        if int(tensor.ndim) != 2 or int(tensor.shape[0]) != self._num_envs:
            raise RuntimeError(f"Newton SolverMuJoCo has an invalid {name} mapping")
        values = tuple(int(value) for value in tensor[0].detach().cpu().tolist())
        self._mapping_cache[name] = values
        return list(values)

    def _model_int_values(self, name: str) -> list[int]:
        cached = self._model_int_cache.get(name)
        if cached is not None:
            return list(cached)
        value = getattr(self._model, name, None)
        if value is None:
            raise RuntimeError(f"Newton model has no {name} array")
        tensor = self._wp.to_torch(value)
        values = tuple(int(item) for item in tensor.detach().cpu().tolist())
        self._model_int_cache[name] = values
        return list(values)

    def _body_array_index(self, mj_body_id: int) -> int:
        mapping = self._mapping_row("mjc_body_to_newton")
        if mj_body_id < 0 or mj_body_id >= len(mapping):
            raise RuntimeError(f"MuJoCo body index {mj_body_id} is outside Newton's body mapping")
        index = mapping[mj_body_id]
        bodies_per_world = int(self._arrays["body_q"].shape[1])
        if index < 0 or index >= bodies_per_world:
            raise RuntimeError(f"MuJoCo body {mj_body_id} has no Newton body in environment 0")
        return index

    def _floating_joint_for_body(self, mj_model: Any, mj_body_id: int) -> int | None:
        first_joint = int(mj_model.body_jntadr[mj_body_id])
        joint_count = int(mj_model.body_jntnum[mj_body_id])
        free_type = int(self._mujoco.mjtJoint.mjJNT_FREE)
        floating = [
            joint_id
            for joint_id in range(first_joint, first_joint + joint_count)
            if int(mj_model.jnt_type[joint_id]) == free_type
        ]
        if len(floating) > 1:
            raise RuntimeError(f"MuJoCo body {mj_body_id} has multiple floating joints")
        return floating[0] if floating else None

    def _joint_array_ranges(
        self,
        mj_model: Any,
        mj_joint_id: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        joint_type = int(mj_model.jnt_type[mj_joint_id])
        joint_types = self._mujoco.mjtJoint
        if joint_type == int(joint_types.mjJNT_FREE):
            q_width, qd_width = 7, 6
        elif joint_type == int(joint_types.mjJNT_BALL):
            q_width, qd_width = 4, 3
        elif joint_type in (
            int(joint_types.mjJNT_SLIDE),
            int(joint_types.mjJNT_HINGE),
        ):
            q_width, qd_width = 1, 1
        else:
            raise RuntimeError(f"MuJoCo joint {mj_joint_id} has unsupported type {joint_type}")

        joint_mapping = self._mapping_row("mjc_jnt_to_newton_jnt")
        dof_mapping = self._mapping_row("mjc_jnt_to_newton_dof")
        if mj_joint_id >= len(joint_mapping) or mj_joint_id >= len(dof_mapping):
            raise RuntimeError(f"MuJoCo joint {mj_joint_id} is outside Newton's joint mapping")
        newton_joint = joint_mapping[mj_joint_id]
        newton_dof = dof_mapping[mj_joint_id]
        if newton_joint < 0 or newton_dof < 0:
            raise RuntimeError(f"MuJoCo joint {mj_joint_id} has no Newton coordinate mapping")

        q_starts = self._model_int_values("joint_q_start")
        qd_starts = self._model_int_values("joint_qd_start")
        if newton_joint >= len(q_starts) or newton_joint >= len(qd_starts):
            raise RuntimeError(f"Newton joint mapping {newton_joint} is out of range")
        q_start = q_starts[newton_joint]
        joint_qd_start = qd_starts[newton_joint]
        if newton_dof < joint_qd_start:
            raise RuntimeError(f"Newton DOF mapping {newton_dof} precedes its joint")
        if joint_type in (
            int(joint_types.mjJNT_FREE),
            int(joint_types.mjJNT_BALL),
        ):
            # SolverMuJoCo maps a multi-DOF MuJoCo joint to the last Newton
            # DOF it visited. The model's joint start is the authoritative
            # beginning of the free/ball velocity block.
            qd_start = joint_qd_start
        else:
            qd_start = newton_dof
            q_start += qd_start - joint_qd_start

        nq = int(self._arrays["joint_q"].shape[1])
        nv = int(self._arrays["joint_qd"].shape[1])
        q_start %= nq
        qd_start %= nv
        q_indices = tuple(range(q_start, q_start + q_width))
        qd_indices = tuple(range(qd_start, qd_start + qd_width))
        if q_indices[-1] >= nq or qd_indices[-1] >= nv:
            raise RuntimeError(f"MuJoCo joint {mj_joint_id} crosses a Newton world boundary")
        return q_indices, qd_indices

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
    "NewtonModelConfig",
    "NewtonProvider",
    "NewtonProviderAvailability",
    "NewtonRobotLayout",
]
