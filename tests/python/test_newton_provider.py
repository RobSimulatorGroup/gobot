from __future__ import annotations

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

import gobot
from gobot.rl.providers.newton import NewtonProvider, _NewtonBindings


REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeTensor:
    def __init__(self, value, *, dtype=None):
        self._value = np.asarray(value, dtype=dtype)

    @property
    def shape(self):
        return self._value.shape

    @property
    def ndim(self):
        return self._value.ndim

    @property
    def dtype(self):
        return self._value.dtype

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return _FakeTensor(self._value.reshape(*shape))

    def flatten(self):
        return _FakeTensor(self._value.flatten())

    def clone(self):
        return _FakeTensor(self._value.copy())

    def copy_(self, other):
        source = other._value if isinstance(other, _FakeTensor) else np.asarray(other)
        np.copyto(self._value, source)
        return self

    def zero_(self):
        self._value.fill(0)
        return self

    def numel(self):
        return self._value.size

    def data_ptr(self):
        return self._value.__array_interface__["data"][0]

    def numpy(self):
        return self._value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._value.tolist()

    def __getitem__(self, key):
        if isinstance(key, _FakeTensor):
            key = key._value
        value = self._value[key]
        return _FakeTensor(value) if isinstance(value, np.ndarray) else value

    def __setitem__(self, key, value):
        self._value[key] = value._value if isinstance(value, _FakeTensor) else value

    def __array__(self, dtype=None):
        return np.asarray(self._value, dtype=dtype)

    def __ne__(self, other):
        return _FakeTensor(self._value != other)


class _FakeTorchDevice:
    def __init__(self, value):
        self.type = str(value).split(":", 1)[0]


class _FakeCuda:
    @staticmethod
    def is_available():
        return True


class _FakeTorch:
    bool = np.bool_
    cuda = _FakeCuda()

    @staticmethod
    def device(value):
        return _FakeTorchDevice(value)

    @staticmethod
    def as_tensor(value, *, dtype, device):
        del device
        source = value._value if isinstance(value, _FakeTensor) else value
        return _FakeTensor(source, dtype=dtype)

    @staticmethod
    def where(mask, true_value, false_value):
        return _FakeTensor(
            np.where(mask._value, true_value._value, false_value._value)
        )

    @staticmethod
    def nonzero(value, *, as_tuple):
        assert not as_tuple
        return _FakeTensor(np.argwhere(value._value))


class _FakeWarpArray:
    def __init__(self, shape, *, dtype=np.float32, value=None):
        if value is None:
            value = np.zeros(shape, dtype=dtype)
        self.tensor = _FakeTensor(value, dtype=dtype)

    @property
    def shape(self):
        return self.tensor.shape


class _FakeScope:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _FakeWarpDevice:
    is_cuda = True


class _FakeWarp:
    bool = object()

    def __init__(self):
        self.initialized = False
        self.synchronize_count = 0

    def init(self):
        self.initialized = True

    @staticmethod
    def get_device(name):
        assert name == "cuda:0"
        return _FakeWarpDevice()

    def zeros(self, shape, *, dtype, device):
        assert device.is_cuda
        numpy_dtype = np.bool_ if dtype is self.bool else np.float32
        return _FakeWarpArray(shape, dtype=numpy_dtype)

    @staticmethod
    def to_torch(value):
        return value.tensor

    @staticmethod
    def ScopedDevice(device):
        assert device.is_cuda
        return _FakeScope()

    def synchronize_device(self, device):
        assert device.is_cuda
        self.synchronize_count += 1


class _FakeState:
    def __init__(self, model):
        self.joint_q = _FakeWarpArray(
            (model.joint_coord_count,),
            value=model.joint_q.tensor.numpy().copy(),
        )
        self.joint_qd = _FakeWarpArray(
            (model.joint_dof_count,),
            value=model.joint_qd.tensor.numpy().copy(),
        )
        self.body_q = _FakeWarpArray((model.body_count, 7))
        self.body_qd = _FakeWarpArray((model.body_count, 6))
        self.clear_forces_count = 0
        self.assign_count = 0

    def clear_forces(self):
        self.clear_forces_count += 1

    def assign(self, other):
        self.assign_count += 1
        self.joint_q.tensor.copy_(other.joint_q.tensor)
        self.joint_qd.tensor.copy_(other.joint_qd.tensor)
        self.body_q.tensor.copy_(other.body_q.tensor)
        self.body_qd.tensor.copy_(other.body_qd.tensor)


class _FakeControl:
    def __init__(self, world_count, nu):
        self.mujoco = SimpleNamespace(ctrl=_FakeWarpArray((world_count * nu,)))


class _FakeModel:
    def __init__(self, world_count, nq, nv, nu):
        self.world_count = world_count
        self.joint_coord_count = world_count * nq
        self.joint_dof_count = world_count * nv
        self.body_count = world_count * 2
        self.nu = nu
        self.joint_q = _FakeWarpArray((self.joint_coord_count,))
        self.joint_qd = _FakeWarpArray((self.joint_dof_count,))

    def state(self):
        return _FakeState(self)

    def control(self):
        return _FakeControl(self.world_count, self.nu)


class _FakeModelBuilder:
    def __init__(self):
        self.registered = False
        self.world_count = 0
        self.nq = 2
        self.nv = 2
        self.nu = 2
        self.mjcf_call = None
        self.loaded_meshes = []
        self.spacing = None

    def add_mjcf(self, content, **kwargs):
        self.registered = True
        self.mjcf_call = (content, kwargs)
        root = ET.fromstring(content)
        for mesh in root.findall("./asset/mesh"):
            mesh_file = mesh.attrib.get("file")
            if mesh_file:
                path = Path(mesh_file)
                assert path.is_file()
                self.loaded_meshes.append(path.read_text(encoding="utf-8"))

    def replicate(self, blueprint, world_count, spacing):
        assert self.registered
        assert blueprint.mjcf_call is not None
        self.world_count = world_count
        self.nq = blueprint.nq
        self.nv = blueprint.nv
        self.nu = blueprint.nu
        self.spacing = spacing

    def finalize(self, *, device):
        assert device.is_cuda
        return _FakeModel(self.world_count, self.nq, self.nv, self.nu)


class _FakeSolver:
    last_instance = None

    @staticmethod
    def register_custom_attributes(builder):
        builder.registered = True

    def __init__(self, model, **kwargs):
        self.model = model
        self.options = kwargs
        self.step_count = 0
        self.reset_count = 0
        self.mjw_model = SimpleNamespace(
            nq=model.joint_coord_count // model.world_count,
            nv=model.joint_dof_count // model.world_count,
            nu=model.nu,
        )
        self.mjw_data = SimpleNamespace(
            time=_FakeWarpArray((model.world_count,)),
            overflow=_FakeWarpArray((model.world_count,), dtype=np.int32),
        )
        type(self).last_instance = self

    def step(self, state_in, state_out, control, contacts, dt):
        assert contacts is None
        self.step_count += 1
        world_count = self.model.world_count
        nq = self.model.joint_coord_count // world_count
        nv = self.model.joint_dof_count // world_count
        ctrl = control.mujoco.ctrl.tensor.numpy().reshape(world_count, -1)
        q_in = state_in.joint_q.tensor.numpy().reshape(world_count, nq)
        qd_in = state_in.joint_qd.tensor.numpy().reshape(world_count, nv)
        q_out = state_out.joint_q.tensor.numpy().reshape(world_count, nq)
        qd_out = state_out.joint_qd.tensor.numpy().reshape(world_count, nv)
        qd_out[:] = qd_in + ctrl[:, :nv] * dt
        q_out[:] = q_in + qd_out[:, :nq] * dt
        state_out.body_q.tensor.copy_(state_in.body_q.tensor)
        state_out.body_qd.tensor.copy_(state_in.body_qd.tensor)
        state_out.body_q.tensor.numpy()[::2, 0] = q_out[:, 0]
        state_out.body_qd.tensor.numpy()[::2, 0] = qd_out[:, 0]
        self.mjw_data.time.tensor.numpy()[:] += dt

    def reset(self, state, world_mask):
        self.reset_count += 1
        mask = world_mask.tensor.numpy()
        nq = self.model.joint_coord_count // self.model.world_count
        nv = self.model.joint_dof_count // self.model.world_count
        state.joint_q.tensor.numpy().reshape(self.model.world_count, nq)[mask] = 0.0
        state.joint_qd.tensor.numpy().reshape(self.model.world_count, nv)[mask] = 0.0


class _FakeNewton:
    def __init__(self):
        self.builders = []
        self.solvers = SimpleNamespace(SolverMuJoCo=_FakeSolver)

    def ModelBuilder(self):
        builder = _FakeModelBuilder()
        self.builders.append(builder)
        return builder

    @staticmethod
    def eval_fk(model, joint_q, joint_qd, state):
        del joint_q, joint_qd
        nq = model.joint_coord_count // model.world_count
        nv = model.joint_dof_count // model.world_count
        q = state.joint_q.tensor.numpy().reshape(model.world_count, nq)
        qd = state.joint_qd.tensor.numpy().reshape(model.world_count, nv)
        state.body_q.tensor.numpy()[::2, 0] = q[:, 0]
        state.body_qd.tensor.numpy()[::2, 0] = qd[:, 0]


class _FakeMuJoCo:
    @staticmethod
    def mj_versionString():
        return "3.10.0"


def _digest(content):
    value = 14695981039346656037
    for byte in content.encode("utf-8"):
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{value:016x}"


def _artifact(*, nq=2, nv=2, nu=2):
    content = (
        "<mujoco><worldbody><body name='body'><joint name='joint' type='slide'/>"
        "<geom type='sphere' size='0.1'/></body></worldbody></mujoco>"
    )
    return gobot.rl.CompiledSceneArtifact(
        schema_version=1,
        backend="MuJoCoCpu",
        format="mjcf",
        content=content,
        digest=_digest(content),
        backend_version="3.10.0",
        dimensions={"nq": nq, "nv": nv, "nu": nu},
        robot_names=("robot",),
        robot_prefixes=("robot_",),
        terrain_geom_groups=(),
    )


def _bindings():
    return _NewtonBindings(
        newton=_FakeNewton(),
        warp=_FakeWarp(),
        torch=_FakeTorch(),
        mujoco=_FakeMuJoCo(),
        mujoco_warp=SimpleNamespace(__version__="3.10.0.2"),
    )


def test_fake_provider_lifecycle_and_masked_reset():
    assert gobot.rl.NewtonProvider is NewtonProvider
    bindings = _bindings()
    provider = NewtonProvider(
        _artifact(),
        num_envs=3,
        fixed_time_step=0.01,
        nconmax=32,
        njmax=64,
        iterations=8,
        _bindings=bindings,
    )
    try:
        arrays = provider.arrays
        assert "qpos" not in arrays
        assert "qvel" not in arrays
        assert arrays["joint_q"].shape == (3, 2)
        assert arrays["joint_qd"].shape == (3, 2)
        assert arrays["body_q"].shape == (3, 2, 7)
        assert arrays["body_qd"].shape == (3, 2, 6)
        assert arrays["ctrl"].shape == (3, 2)
        assert arrays["time"].shape == (3,)
        assert provider.fixed_time_step == 0.01
        assert provider.capabilities.name == "Newton"
        assert provider.capabilities.device_native
        assert provider.capabilities.masked_reset
        assert not provider.capabilities.graph_capture

        blueprint, runtime_builder = bindings.newton.builders
        assert blueprint.mjcf_call[1] == {
            "ctrl_direct": True,
            "parse_visuals": False,
        }
        assert runtime_builder.spacing == (0.0, 0.0, 0.0)
        assert _FakeSolver.last_instance.options == {
            "use_mujoco_contacts": True,
            "nconmax": 32,
            "njmax": 64,
            "iterations": 8,
        }

        q_pointer = arrays["joint_q"].data_ptr()
        qd_pointer = arrays["joint_qd"].data_ptr()
        actions = np.asarray(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=np.float32,
        )
        provider.step(actions, nsteps=2)
        assert provider.arrays["joint_q"].data_ptr() == q_pointer
        assert provider.arrays["joint_qd"].data_ptr() == qd_pointer
        assert np.all(arrays["joint_q"].numpy() > 0.0)
        np.testing.assert_allclose(arrays["time"].numpy(), 0.02)

        before_reset_q = arrays["joint_q"].clone()
        before_reset_qd = arrays["joint_qd"].clone()
        desired_q = before_reset_q.clone()
        desired_q[0, 0] = 0.75
        provider.reset([True, False, False], joint_q=desired_q)
        assert arrays["joint_q"][0, 0] == np.float32(0.75)
        np.testing.assert_allclose(
            arrays["joint_q"].numpy()[1:],
            before_reset_q.numpy()[1:],
        )
        np.testing.assert_allclose(
            arrays["joint_qd"].numpy()[1:],
            before_reset_qd.numpy()[1:],
        )
        np.testing.assert_allclose(arrays["joint_qd"].numpy()[0], 0.0)
        np.testing.assert_allclose(arrays["ctrl"].numpy()[0], 0.0)
        np.testing.assert_allclose(arrays["ctrl"].numpy()[1:], actions[1:])
        np.testing.assert_allclose(arrays["time"].numpy(), [0.0, 0.02, 0.02])
        assert not arrays["reset_mask"].numpy().any()

        provider.reset([False, True, False])
        np.testing.assert_allclose(arrays["joint_q"].numpy()[1], 0.0)
        assert _FakeSolver.last_instance.reset_count == 2

        arrays["overflow"][2] = 1 << 3
        try:
            provider.assert_no_overflow()
        except gobot.rl.SimulationCapacityError as error:
            assert "env 2" in str(error)
            assert "narrowphase contacts" in str(error)
        else:
            raise AssertionError("Newton capacity overflow was ignored")
        arrays["overflow"].zero_()

        try:
            provider.step(np.zeros((3, 1), dtype=np.float32))
        except ValueError as error:
            assert "actions must have shape" in str(error)
        else:
            raise AssertionError("invalid Newton action shape was accepted")
    finally:
        provider.close()

    provider.close()
    assert bindings.warp.synchronize_count == 2
    try:
        _ = provider.arrays
    except RuntimeError as error:
        assert "closed" in str(error)
    else:
        raise AssertionError("closed Newton provider exposed arrays")


def test_availability_reports_missing_optional_package():
    def find_spec(name):
        return None if name == "newton" else object()

    with patch("gobot.rl.providers.newton.importlib.util.find_spec", side_effect=find_spec):
        availability = NewtonProvider.availability()
    assert not availability.available
    assert "newton" in availability.reason


def test_import_version_failure_is_reported_as_unavailable():
    with (
        patch.object(
            NewtonProvider,
            "availability",
            return_value=gobot.rl.NewtonProviderAvailability(True),
        ),
        patch(
            "gobot.rl.providers.newton.importlib.import_module",
            side_effect=AttributeError("Warp is missing DeterministicMode"),
        ),
    ):
        try:
            NewtonProvider._load_bindings()
        except gobot.rl.ProviderUnavailableError as error:
            assert "could not be imported together" in str(error)
            assert "DeterministicMode" in str(error)
        else:
            raise AssertionError("incompatible Newton dependencies were accepted")


def test_provider_rejects_non_cuda_device_and_dimension_mismatch():
    try:
        NewtonProvider(_artifact(), num_envs=1, device="cpu", _bindings=_bindings())
    except gobot.rl.ProviderUnavailableError as error:
        assert "CUDA provider" in str(error)
    else:
        raise AssertionError("Newton accepted a CPU device")

    try:
        NewtonProvider(_artifact(nq=3), num_envs=2, _bindings=_bindings())
    except RuntimeError as error:
        assert "dimensions do not match" in str(error)
    else:
        raise AssertionError("Newton accepted mismatched artifact dimensions")

def test_inline_mjcf_mesh_is_materialized_for_newton_import():
    content = (
        "<mujoco><asset><mesh name='terrain' vertex='0 0 0 1 0 0 0 1 0' "
        "face='0 1 2'/></asset><worldbody><geom type='mesh' mesh='terrain'/>"
        "<body name='body'><joint name='joint' type='slide'/></body></worldbody></mujoco>"
    )
    base = _artifact()
    artifact = type(base)(
        **{
            **base.__dict__,
            "content": content,
            "digest": _digest(content),
        }
    )
    bindings = _bindings()
    provider = NewtonProvider(artifact, num_envs=1, _bindings=bindings)
    try:
        blueprint = bindings.newton.builders[0]
        assert len(blueprint.loaded_meshes) == 1
        assert "v 0.0 0.0 0.0" in blueprint.loaded_meshes[0]
        assert "f 1 2 3" in blueprint.loaded_meshes[0]
        imported = ET.fromstring(blueprint.mjcf_call[0]).find("./asset/mesh")
        assert imported is not None
        assert "vertex" not in imported.attrib
        assert "face" not in imported.attrib
        assert not Path(imported.attrib["file"]).exists()
    finally:
        provider.close()


def test_mujoco_solref_shorthand_is_normalized_for_newton_import():
    content = (
        "<mujoco><default><geom solref='0.02'/></default><worldbody>"
        "<body name='body'><joint name='joint' type='slide'/>"
        "<geom type='sphere' size='0.1' solref='-0.01'/></body>"
        "</worldbody><contact><pair geom1='a' geom2='b' solref='0.03'/>"
        "</contact></mujoco>"
    )
    base = _artifact()
    artifact = type(base)(
        **{
            **base.__dict__,
            "content": content,
            "digest": _digest(content),
        }
    )
    bindings = _bindings()
    provider = NewtonProvider(artifact, num_envs=1, _bindings=bindings)
    try:
        imported = ET.fromstring(bindings.newton.builders[0].mjcf_call[0])
        assert imported.find("./default/geom").attrib["solref"] == "0.02 1"
        assert imported.find("./worldbody/body/geom").attrib["solref"] == "-0.01 1"
        assert imported.find("./contact/pair").attrib["solref"] == "0.03"
    finally:
        provider.close()


def test_mujoco_solref_shorthand_preserves_default_class_inheritance():
    content = (
        "<mujoco><default><geom solref='0.1 3'/>"
        "<default class='nested'><geom solref='0.15'/></default>"
        "<default class='explicit'><geom solref='0.12 4'/></default>"
        "</default><worldbody><body name='parent' childclass='nested'>"
        "<geom name='inherited' type='sphere' size='0.1' solref='0.2'/>"
        "<geom name='explicit' class='explicit' type='sphere' size='0.1' solref='0.3'/>"
        "<body name='child'><joint name='joint' type='slide'/>"
        "<geom name='descendant' type='sphere' size='0.1' solref='0.4'/>"
        "</body></body></worldbody></mujoco>"
    )
    base = _artifact()
    artifact = type(base)(
        **{
            **base.__dict__,
            "content": content,
            "digest": _digest(content),
        }
    )
    bindings = _bindings()
    provider = NewtonProvider(artifact, num_envs=1, _bindings=bindings)
    try:
        imported = ET.fromstring(bindings.newton.builders[0].mjcf_call[0])
        assert imported.find("./default/default[@class='nested']/geom").attrib[
            "solref"
        ] == "0.15 3"
        geoms = {
            geom.attrib["name"]: geom.attrib["solref"]
            for geom in imported.findall("./worldbody//geom")
        }
        assert geoms == {
            "inherited": "0.2 3",
            "explicit": "0.3 4",
            "descendant": "0.4 3",
        }
    finally:
        provider.close()


def test_direct_artifact_is_revalidated():
    artifact = _artifact()
    corrupt = type(artifact)(**{**artifact.__dict__, "digest": "fnv1a64:bad"})
    try:
        NewtonProvider(corrupt, num_envs=1, _bindings=_bindings())
    except ValueError as error:
        assert "digest mismatch" in str(error)
    else:
        raise AssertionError("Newton accepted a corrupt direct artifact")


def test_optional_real_gpu_smoke():
    if os.environ.get("GOBOT_RUN_NEWTON_GPU_TEST") != "1":
        return
    availability = NewtonProvider.availability()
    if not availability.available:
        raise RuntimeError(f"Newton GPU smoke requested but unavailable: {availability.reason}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Newton GPU smoke requested but Torch cannot access CUDA")

    artifact = _artifact(nq=1, nv=1, nu=1)
    content = (
        "<mujoco><worldbody><body name='body'><joint name='joint' type='slide'/>"
        "<geom type='sphere' size='0.1' mass='1'/></body></worldbody>"
        "<actuator><motor name='motor' joint='joint'/></actuator></mujoco>"
    )
    artifact = type(artifact)(
        **{
            **artifact.__dict__,
            "content": content,
            "digest": _digest(content),
        }
    )
    provider = NewtonProvider(artifact, num_envs=2, device="cuda:0")
    try:
        provider.step(torch.zeros_like(provider.arrays["ctrl"]), nsteps=2)
        provider.synchronize()
        assert bool(torch.isfinite(provider.arrays["joint_q"]).all())
        assert bool(torch.isfinite(provider.arrays["joint_qd"]).all())
    finally:
        provider.close()

    context = gobot.app.create_context()
    temporary_directory = tempfile.TemporaryDirectory()
    try:
        context.set_project_path(temporary_directory.name)
        root = gobot.scene.create_cartpole_scene(name="newton_cartpole")
        slider = root.find("rail/slider")
        hinge = root.find("rail/slider/cart/hinge")
        assert slider is not None and hinge is not None
        slider.drive_mode = gobot.JointDriveMode.Position
        hinge.drive_mode = gobot.JointDriveMode.Position
        gobot.save_scene(root, "res://newton_cartpole.jscn")
        context.load_scene("res://newton_cartpole.jscn")
        compiled = context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
        for dimension in ("nq", "nv", "nu", "nbody", "njoint", "ngeom", "nhfield"):
            assert dimension in compiled["dimensions"]

        provider = NewtonProvider(compiled, num_envs=2, device="cuda:0")
        try:
            provider.step(torch.zeros_like(provider.arrays["ctrl"]), nsteps=2)
            provider.synchronize()
            assert bool(torch.isfinite(provider.arrays["joint_q"]).all())
            assert bool(torch.isfinite(provider.arrays["joint_qd"]).all())
        finally:
            provider.close()
    finally:
        context.clear_world()
        context.clear_scene()
        temporary_directory.cleanup()

    context = gobot.app.create_context()
    try:
        context.set_project_path(str(REPO_ROOT / "examples/go1"))
        context.load_scene("res://go1_scene.jscn")
        compiled = context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
        assert compiled["dimensions"]["nhfield"] == 40

        provider = NewtonProvider(compiled, num_envs=1, device="cuda:0")
        try:
            joint_q = provider.arrays["joint_q"].clone()
            joint_q[0, :7] = torch.tensor(
                [-4.0, 0.0, 1.198, 0.0, 0.0, 0.0, 1.0],
                device="cuda:0",
            )
            joint_q[0, 7:] = torch.tensor(
                [
                    0.1, 0.9, -1.8,
                    -0.1, 0.9, -1.8,
                    0.1, 0.9, -1.8,
                    -0.1, 0.9, -1.8,
                ],
                device="cuda:0",
            )
            provider.reset(
                torch.ones(1, dtype=torch.bool, device="cuda:0"),
                joint_q=joint_q,
                ctrl=joint_q[:, 7:].clone(),
            )
            provider.step(torch.zeros_like(provider.arrays["ctrl"]))
            provider.synchronize()
            provider.assert_no_overflow()
            provider.assert_finite()
        finally:
            provider.close()
    finally:
        context.clear_world()
        context.clear_scene()


def main():
    test_fake_provider_lifecycle_and_masked_reset()
    test_availability_reports_missing_optional_package()
    test_import_version_failure_is_reported_as_unavailable()
    test_provider_rejects_non_cuda_device_and_dimension_mismatch()
    test_inline_mjcf_mesh_is_materialized_for_newton_import()
    test_mujoco_solref_shorthand_is_normalized_for_newton_import()
    test_mujoco_solref_shorthand_preserves_default_class_inheritance()
    test_direct_artifact_is_revalidated()
    test_optional_real_gpu_smoke()


if __name__ == "__main__":
    main()
