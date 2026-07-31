from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

import gobot
from gobot.rl.providers.newton import (
    NewtonModelConfig,
    NewtonProvider,
    NewtonRobotLayout,
    _NewtonBindings,
)


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

    def index_copy_(self, dim, index, source):
        indices = index._value if isinstance(index, _FakeTensor) else np.asarray(index)
        values = source._value if isinstance(source, _FakeTensor) else np.asarray(source)
        if dim != 1:
            raise AssertionError(f"fake tensor only supports column index_copy_, got dim={dim}")
        self._value[:, indices] = values
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
    long = np.int64
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
    def __init__(self, world_count, target_q_width, nv, nu):
        self.joint_target_q = _FakeWarpArray((world_count * target_q_width,))
        self.joint_target_qd = _FakeWarpArray((world_count * nv,))
        self.mujoco = SimpleNamespace(ctrl=_FakeWarpArray((world_count * nu,)))


class _FakeModel:
    def __init__(
        self,
        world_count,
        nq,
        nv,
        nu,
        *,
        use_coord_layout_targets=False,
        actuator_routes=None,
        joint_mapping=None,
        dof_mapping=None,
    ):
        self.world_count = world_count
        self.joint_coord_count = world_count * nq
        self.joint_dof_count = world_count * nv
        self.body_count = world_count * 2
        self.nu = nu
        self.use_coord_layout_targets = use_coord_layout_targets
        self.actuator_routes = actuator_routes
        self.joint_mapping = joint_mapping or tuple(reversed(range(nv)))
        self.dof_mapping = dof_mapping or self.joint_mapping
        joint_count = len(self.joint_mapping)
        self.joint_q = _FakeWarpArray((self.joint_coord_count,))
        self.joint_qd = _FakeWarpArray((self.joint_dof_count,))
        self.joint_q_start = _FakeWarpArray(
            (world_count * joint_count,),
            dtype=np.int32,
            value=np.asarray(
                [
                    offset + index
                    for offset in range(0, world_count * nq, nq)
                    for index in range(joint_count)
                ],
                dtype=np.int32,
            ),
        )
        self.joint_qd_start = _FakeWarpArray(
            (world_count * joint_count,),
            dtype=np.int32,
            value=np.asarray(
                [
                    offset + index
                    for offset in range(0, world_count * nv, nv)
                    for index in range(joint_count)
                ],
                dtype=np.int32,
            ),
        )
        self.contacts_value = object()
        self.collide_count = 0

    def state(self):
        return _FakeState(self)

    def control(self):
        return _FakeControl(
            self.world_count,
            (
                self.joint_coord_count
                if self.use_coord_layout_targets
                else self.joint_dof_count
            )
            // self.world_count,
            self.joint_dof_count // self.world_count,
            self.nu,
        )

    def contacts(self):
        return self.contacts_value

    def collide(self, state, contacts):
        assert state is not None
        assert contacts is self.contacts_value
        self.collide_count += 1


class _FakeModelBuilder:
    def __init__(
        self,
        *,
        nq=2,
        nv=2,
        use_coord_layout_targets=False,
        actuator_routes=None,
        joint_mapping=None,
        dof_mapping=None,
    ):
        self.registered = False
        self.world_count = 0
        self.nq = nq
        self.nv = nv
        self.nu = 2
        self.use_coord_layout_targets = use_coord_layout_targets
        self.actuator_routes = actuator_routes
        self.joint_mapping = joint_mapping
        self.dof_mapping = dof_mapping
        self.mjcf_call = None
        self.loaded_meshes = []
        self.spacing = None
        self.rigid_gap = 0.1
        self.mesh_approximation = None
        self.default_joint_cfg = SimpleNamespace(limit_ke=1.0e4, limit_kd=1.0e1)
        self.default_shape_cfg = SimpleNamespace(
            ke=1.0e5,
            kd=1.0e3,
            kf=1.0e3,
            mu=0.5,
        )
        self.joint_limit_ke = []
        self.joint_limit_kd = []

    def add_mjcf(self, content, **kwargs):
        self.registered = True
        self.mjcf_call = (content, kwargs)
        root = ET.fromstring(content)
        joint_count = len(root.findall("./worldbody//joint"))
        self.joint_limit_ke = [2.5e3] * joint_count
        self.joint_limit_kd = [1.0e2] * joint_count
        self.nu = len([element for section in root.findall("actuator") for element in section])
        for mesh in root.findall("./asset/mesh"):
            mesh_file = mesh.attrib.get("file")
            if mesh_file:
                path = Path(mesh_file)
                assert path.is_file()
                self.loaded_meshes.append(path.read_text(encoding="utf-8"))

    def approximate_meshes(self, method):
        self.mesh_approximation = method
        return set()

    def replicate(self, blueprint, world_count, spacing):
        assert self.registered
        assert blueprint.mjcf_call is not None
        self.world_count = world_count
        self.nq = blueprint.nq
        self.nv = blueprint.nv
        self.nu = blueprint.nu
        self.use_coord_layout_targets = blueprint.use_coord_layout_targets
        self.actuator_routes = blueprint.actuator_routes
        self.joint_mapping = blueprint.joint_mapping
        self.dof_mapping = blueprint.dof_mapping
        self.spacing = spacing

    def finalize(self, *, device):
        assert device.is_cuda
        return _FakeModel(
            self.world_count,
            self.nq,
            self.nv,
            self.nu,
            use_coord_layout_targets=self.use_coord_layout_targets,
            actuator_routes=self.actuator_routes,
            joint_mapping=self.joint_mapping,
            dof_mapping=self.dof_mapping,
        )


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
        joint_count = len(model.joint_mapping)
        nv = model.joint_dof_count // model.world_count
        self.mj_model = SimpleNamespace(
            nq=model.joint_coord_count // model.world_count,
            nv=model.joint_dof_count // model.world_count,
            nu=model.nu,
            nbody=3,
            njnt=joint_count,
            body_jntadr=np.asarray([0, 0, 1], dtype=np.int32),
            body_jntnum=np.asarray([0, 1, 1], dtype=np.int32),
            jnt_type=np.asarray([3, 3], dtype=np.int32),
        )
        self.mjw_model = self.mj_model
        self.mjc_body_to_newton = _FakeWarpArray(
            (model.world_count, 3),
            dtype=np.int32,
            value=np.asarray(
                [
                    [-1, world * 2 + 1, world * 2]
                    for world in range(model.world_count)
                ],
                dtype=np.int32,
            ),
        )
        self.mjc_jnt_to_newton_jnt = _FakeWarpArray(
            (model.world_count, joint_count),
            dtype=np.int32,
            value=np.asarray(
                [
                    [world * joint_count + value for value in model.joint_mapping]
                    for world in range(model.world_count)
                ],
                dtype=np.int32,
            ),
        )
        self.mjc_jnt_to_newton_dof = _FakeWarpArray(
            (model.world_count, joint_count),
            dtype=np.int32,
            value=np.asarray(
                [
                    [world * nv + value for value in model.dof_mapping]
                    for world in range(model.world_count)
                ],
                dtype=np.int32,
            ),
        )
        routes = model.actuator_routes or {
            "sources": np.zeros(model.nu, dtype=np.int32),
            "indices": np.arange(model.nu, dtype=np.int32),
            "targets": np.arange(model.nu, dtype=np.int32),
            "axes": np.full(model.nu, -1, dtype=np.int32),
        }
        self.mjc_actuator_ctrl_source = _FakeWarpArray(
            (model.nu,),
            dtype=np.int32,
            value=routes["sources"],
        )
        self.mjc_actuator_to_newton_idx = _FakeWarpArray(
            (model.nu,),
            dtype=np.int32,
            value=routes["indices"],
        )
        self.mjc_actuator_to_newton_target_q_idx = _FakeWarpArray(
            (model.nu,),
            dtype=np.int32,
            value=routes["targets"],
        )
        self.mjc_actuator_to_target_q_axis_idx = _FakeWarpArray(
            (model.nu,),
            dtype=np.int32,
            value=routes.get("axes", np.full(model.nu, -1, dtype=np.int32)),
        )
        self.mjw_data = SimpleNamespace(
            time=_FakeWarpArray((model.world_count,)),
            overflow=_FakeWarpArray((model.world_count,), dtype=np.int32),
        )
        type(self).last_instance = self

    def step(self, state_in, state_out, control, contacts, dt):
        if self.options["use_mujoco_contacts"]:
            assert contacts is None
        else:
            assert contacts is self.model.contacts_value
        self.step_count += 1
        world_count = self.model.world_count
        nq = self.model.joint_coord_count // world_count
        nv = self.model.joint_dof_count // world_count
        ctrl = control.joint_target_q.tensor.numpy().reshape(world_count, -1)
        q_in = state_in.joint_q.tensor.numpy().reshape(world_count, nq)
        qd_in = state_in.joint_qd.tensor.numpy().reshape(world_count, nv)
        q_out = state_out.joint_q.tensor.numpy().reshape(world_count, nq)
        qd_out = state_out.joint_qd.tensor.numpy().reshape(world_count, nv)
        qd_out[:] = qd_in
        width = min(nv, ctrl.shape[1])
        qd_out[:, :width] += ctrl[:, :width] * dt
        q_out[:] = q_in
        width = min(nq, nv)
        q_out[:, :width] += qd_out[:, :width] * dt
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
    def __init__(self, **builder_options):
        self.builders = []
        self.solvers = SimpleNamespace(SolverMuJoCo=_FakeSolver)
        self.builder_options = builder_options

    def ModelBuilder(self):
        builder = _FakeModelBuilder(**self.builder_options)
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
    mjtObj = SimpleNamespace(
        mjOBJ_BODY=1,
        mjOBJ_JOINT=2,
        mjOBJ_ACTUATOR=3,
    )
    mjtJoint = SimpleNamespace(
        mjJNT_FREE=0,
        mjJNT_BALL=1,
        mjJNT_SLIDE=2,
        mjJNT_HINGE=3,
    )

    def __init__(self, *, jnt_type=None, actuator_trnid=None, name_ids=None):
        if jnt_type is None:
            jnt_type = [3, 3]
        if actuator_trnid is None:
            actuator_trnid = [[0, 0], [1, 0]]
        self.metadata_model = SimpleNamespace(
            body_jntadr=np.asarray([0, 0, 1], dtype=np.int32),
            body_jntnum=np.asarray([0, 1, 1], dtype=np.int32),
            jnt_type=np.asarray(jnt_type, dtype=np.int32),
            actuator_trnid=np.asarray(actuator_trnid, dtype=np.int32),
        )
        self.MjModel = SimpleNamespace(
            from_xml_string=lambda content: self._load_model(content)
        )
        self.name_ids = name_ids or {
            (self.mjtObj.mjOBJ_BODY, "robot_base"): 1,
            (self.mjtObj.mjOBJ_BODY, "robot_tip"): 2,
            (self.mjtObj.mjOBJ_JOINT, "robot_joint_a"): 0,
            (self.mjtObj.mjOBJ_JOINT, "robot_joint_b"): 1,
            (self.mjtObj.mjOBJ_ACTUATOR, "robot_joint_a_position"): 0,
            (self.mjtObj.mjOBJ_ACTUATOR, "robot_joint_b_position"): 1,
        }

    def _load_model(self, content):
        ET.fromstring(content)
        return self.metadata_model

    @staticmethod
    def mj_versionString():
        return "3.10.0"

    def mj_name2id(self, model, object_type, name):
        del model
        return self.name_ids.get((object_type, name), -1)


def _digest(content):
    value = 14695981039346656037
    for byte in content.encode("utf-8"):
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{value:016x}"


def _artifact(*, nq=2, nv=2, nu=2):
    content = (
        "<mujoco><worldbody><body name='body'><joint name='joint' type='slide'/>"
        "<geom type='sphere' size='0.1'/></body></worldbody><actuator>"
        "<position name='robot_joint_a_position' joint='joint_a'/>"
        "<position name='robot_joint_b_position' joint='joint_b'/>"
        "</actuator></mujoco>"
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


def _bindings(*, newton_options=None, mujoco_options=None):
    return _NewtonBindings(
        newton=_FakeNewton(**(newton_options or {})),
        warp=_FakeWarp(),
        torch=_FakeTorch(),
        mujoco=_FakeMuJoCo(**(mujoco_options or {})),
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
        assert provider.use_mujoco_contacts is True
        assert provider.capabilities.name == "Newton"
        assert provider.capabilities.device_native
        assert provider.capabilities.masked_reset
        assert not provider.capabilities.graph_capture

        blueprint, runtime_builder = bindings.newton.builders
        assert blueprint.mjcf_call[1] == {
            "ctrl_direct": False,
            "parse_visuals": False,
            "enable_self_collisions": True,
        }
        assert blueprint.rigid_gap == 0.0
        assert blueprint.mesh_approximation == "convex_hull"
        assert runtime_builder.spacing == (0.0, 0.0, 0.0)
        assert _FakeSolver.last_instance.options == {
            "use_mujoco_cpu": False,
            "solver": "newton",
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


def test_named_robot_layout_controls_and_reset():
    bindings = _bindings()
    provider = NewtonProvider(_artifact(), num_envs=2, _bindings=bindings)
    try:
        layout = provider.resolve_robot_layout(
            "robot",
            base_link="base",
            joint_names=("joint_a", "joint_b"),
            link_names=("base", "tip"),
        )
        assert isinstance(layout, NewtonRobotLayout)
        assert gobot.rl.NewtonRobotLayout is NewtonRobotLayout
        assert layout.robot_name == "robot"
        assert layout.runtime_prefix == "robot_"
        assert layout.base_body_index == 1
        assert layout.base_joint_q_indices == ()
        assert layout.base_joint_qd_indices == ()
        assert layout.joint_q_indices == (1, 0)
        assert layout.joint_qd_indices == (1, 0)
        assert layout.actuator_indices == (0, 1)
        assert layout.actuator_modes == ("position", "position")
        assert layout.link_body_indices == (1, 0)

        targets = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        provider.set_joint_position_targets(layout, targets)
        np.testing.assert_allclose(
            provider._joint_target_q_tensor.numpy(),
            np.asarray([[20.0, 10.0], [40.0, 30.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(provider.arrays["ctrl"].numpy(), targets)
        np.testing.assert_allclose(provider._direct_ctrl_tensor.numpy(), 0.0)

        provider.reset_robot_state(
            layout,
            [True, False],
            joint_position=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            joint_velocity=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            controls=np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(provider.arrays["joint_q"].numpy()[0], [0.2, 0.1])
        np.testing.assert_allclose(provider.arrays["joint_qd"].numpy()[0], [2.0, 1.0])
        np.testing.assert_allclose(provider.arrays["ctrl"].numpy()[0], [5.0, 6.0])
        np.testing.assert_allclose(provider._joint_target_q_tensor.numpy()[0], [6.0, 5.0])
        np.testing.assert_allclose(provider.arrays["joint_q"].numpy()[1], 0.0)
        np.testing.assert_allclose(provider.arrays["ctrl"].numpy()[1], [30.0, 40.0])
        np.testing.assert_allclose(provider._joint_target_q_tensor.numpy()[1], [40.0, 30.0])

        try:
            provider.reset_robot_state(
                layout,
                [True, False],
                base_pose=np.zeros((2, 7), dtype=np.float32),
            )
        except ValueError as error:
            assert "no floating pose coordinates" in str(error)
        else:
            raise AssertionError("fixed-base layout accepted a floating base pose")

        try:
            provider.set_joint_controls(layout, np.zeros((2, 1), dtype=np.float32))
        except ValueError as error:
            assert "joint controls must have shape (2, 2)" in str(error)
        else:
            raise AssertionError("named controls accepted an invalid shape")

        try:
            provider.set_joint_controls(
                replace(layout, artifact_digest="fnv1a64:other"),
                targets,
            )
        except ValueError as error:
            assert "different compiled scene artifact" in str(error)
        else:
            raise AssertionError("layout from a different artifact was accepted")
    finally:
        provider.close()


def test_named_robot_layout_reports_name_and_actuator_errors():
    bindings = _bindings()
    provider = NewtonProvider(_artifact(), num_envs=1, _bindings=bindings)
    try:
        try:
            provider.resolve_robot_layout(
                "robot",
                base_link="missing",
                joint_names=("joint_a",),
            )
        except KeyError as error:
            assert "robot_missing" in str(error)
        else:
            raise AssertionError("missing base link was accepted")

        try:
            provider.resolve_robot_layout(
                "robot",
                base_link="base",
                joint_names=("joint_a", "joint_a"),
            )
        except ValueError as error:
            assert "joint names must be unique" in str(error)
        else:
            raise AssertionError("duplicate joint names were accepted")

        del bindings.mujoco.name_ids[
            (bindings.mujoco.mjtObj.mjOBJ_ACTUATOR, "robot_joint_b_position")
        ]
        bindings.mujoco.name_ids[
            (bindings.mujoco.mjtObj.mjOBJ_ACTUATOR, "robot_joint_b_motor")
        ] = 1
        mixed_layout = provider.resolve_robot_layout(
            "robot",
            base_link="base",
            joint_names=("joint_a", "joint_b"),
        )
        assert mixed_layout.actuator_modes == ("position", "motor")
        try:
            provider.set_joint_position_targets(
                mixed_layout,
                np.zeros((1, 2), dtype=np.float32),
            )
        except ValueError as error:
            assert "position actuators" in str(error)
            assert "motor" in str(error)
        else:
            raise AssertionError("position targets were written to a motor actuator")
    finally:
        provider.close()


def test_multi_dof_joint_uses_newton_joint_start_not_last_mapped_dof():
    provider = object.__new__(NewtonProvider)
    provider._num_envs = 1
    provider._mujoco = _FakeMuJoCo()
    provider._solver = SimpleNamespace(
        mjc_jnt_to_newton_jnt=_FakeWarpArray(
            (1, 1), dtype=np.int32, value=np.asarray([[0]], dtype=np.int32)
        ),
        # SolverMuJoCo records the last DOF while exporting a free joint.
        mjc_jnt_to_newton_dof=_FakeWarpArray(
            (1, 1), dtype=np.int32, value=np.asarray([[5]], dtype=np.int32)
        ),
    )
    provider._model = SimpleNamespace(
        joint_q_start=_FakeWarpArray(
            (1,), dtype=np.int32, value=np.asarray([0], dtype=np.int32)
        ),
        joint_qd_start=_FakeWarpArray(
            (1,), dtype=np.int32, value=np.asarray([0], dtype=np.int32)
        ),
    )
    provider._wp = _FakeWarp()
    provider._arrays = {
        "joint_q": _FakeTensor(np.zeros((1, 7), dtype=np.float32)),
        "joint_qd": _FakeTensor(np.zeros((1, 6), dtype=np.float32)),
    }
    provider._mapping_cache = {}
    provider._model_int_cache = {}

    metadata = SimpleNamespace(
        jnt_type=np.asarray([provider._mujoco.mjtJoint.mjJNT_FREE], dtype=np.int32)
    )
    q_indices, qd_indices = provider._joint_array_ranges(metadata, 0)
    assert q_indices == (0, 1, 2, 3, 4, 5, 6)
    assert qd_indices == (0, 1, 2, 3, 4, 5)


def test_explicit_newton_contact_mode_generates_contacts_each_substep():
    bindings = _bindings()
    provider = NewtonProvider(
        _artifact(),
        num_envs=1,
        use_mujoco_contacts=False,
        _bindings=bindings,
    )
    model = provider._model
    try:
        assert provider.use_mujoco_contacts is False
        assert _FakeSolver.last_instance.options["use_mujoco_contacts"] is False
        provider.step(np.zeros((1, 2), dtype=np.float32), nsteps=3)
        assert model.collide_count == 3
    finally:
        provider.close()

    try:
        NewtonProvider(
            _artifact(),
            num_envs=1,
            use_mujoco_contacts="newton",
            _bindings=_bindings(),
        )
    except TypeError as error:
        assert "must be a bool" in str(error)
    else:
        raise AssertionError("non-boolean contact mode was accepted")


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
    base = _artifact(nu=0)
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
    base = _artifact(nu=0)
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
    base = _artifact(nu=0)
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


def test_explicit_model_config_overrides_mjcf_constraint_response():
    content = (
        "<mujoco><default><joint solreflimit='0.02 1'/>"
        "<geom solref='0.03 2'/></default><worldbody>"
        "<body name='body'><joint name='joint' type='slide' range='-1 1'/>"
        "<geom name='body_geom' type='sphere' size='0.1' solref='0.04 3'/>"
        "</body></worldbody><contact><pair geom1='a' geom2='b' "
        "solref='0.05 4' solreffriction='0.06 5'/></contact></mujoco>"
    )
    base = _artifact(nq=1, nv=1, nu=0)
    artifact = type(base)(
        **{
            **base.__dict__,
            "content": content,
            "digest": _digest(content),
        }
    )
    config = NewtonModelConfig(
        joint_limit_stiffness=100.0,
        joint_limit_damping=1.0,
        contact_stiffness=50_000.0,
        contact_damping=500.0,
        contact_friction_stiffness=1_000.0,
        default_contact_friction=0.75,
    )
    bindings = _bindings(newton_options={"nq": 1, "nv": 1})
    provider = NewtonProvider(
        artifact,
        num_envs=1,
        model_config=config,
        _bindings=bindings,
    )
    try:
        blueprint = bindings.newton.builders[0]
        assert provider.model_config == config
        assert blueprint.default_joint_cfg.limit_ke == 100.0
        assert blueprint.default_joint_cfg.limit_kd == 1.0
        assert blueprint.default_shape_cfg.ke == 50_000.0
        assert blueprint.default_shape_cfg.kd == 500.0
        assert blueprint.default_shape_cfg.kf == 1_000.0
        assert blueprint.default_shape_cfg.mu == 0.75
        assert blueprint.joint_limit_ke == [100.0]
        assert blueprint.joint_limit_kd == [1.0]

        imported = ET.fromstring(blueprint.mjcf_call[0])
        assert "solreflimit" not in imported.find("./default/joint").attrib
        assert "solref" not in imported.find("./default/geom").attrib
        assert "solref" not in imported.find("./worldbody/body/geom").attrib
        pair = imported.find("./contact/pair")
        assert pair.attrib["solref"] == "0.05 4"
        assert pair.attrib["solreffriction"] == "0.06 5"
    finally:
        provider.close()


def test_newton_model_config_validation_and_provider_type_check():
    for kwargs, expected in (
        ({"joint_limit_stiffness": 100.0}, "must be set together"),
        ({"contact_damping": 500.0}, "must be set together"),
        ({"contact_stiffness": -1.0, "contact_damping": 1.0}, "non-negative"),
        ({"joint_limit_stiffness": 0.0, "joint_limit_damping": 1.0}, "positive"),
        ({"contact_stiffness": 1.0, "contact_damping": 0.0}, "positive"),
        ({"default_contact_friction": float("nan")}, "finite"),
    ):
        try:
            NewtonModelConfig(**kwargs)
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError(f"invalid Newton model config was accepted: {kwargs}")

    try:
        NewtonProvider(
            _artifact(),
            num_envs=1,
            model_config={"contact_stiffness": 1.0},
            _bindings=_bindings(),
        )
    except TypeError as error:
        assert "NewtonModelConfig" in str(error)
    else:
        raise AssertionError("NewtonProvider accepted an untyped model config")


def test_mujoco_affine_position_general_is_normalized_for_joint_targets():
    content = (
        "<mujoco><worldbody><body name='body'><joint name='joint' type='slide'/>"
        "<geom type='sphere' size='0.1'/></body></worldbody><actuator>"
        "<general name='robot_joint_position' joint='joint' ctrllimited='false' "
        "forcerange='-12 12' forcelimited='true' biastype='affine' "
        "gainprm='150 0 0' biasprm='0 -150 -5 0'/>"
        "<general name='robot_joint_affine' joint='joint' biastype='affine' "
        "gainprm='150 0 0' biasprm='0 -150 -5 0'/></actuator></mujoco>"
    )
    base = _artifact()
    artifact = type(base)(
        **{
            **base.__dict__,
            "content": content,
            "digest": _digest(content),
        }
    )
    bindings = _bindings(
        newton_options={
            "actuator_routes": {
                "sources": np.asarray([0, 1], dtype=np.int32),
                "indices": np.asarray([1, 1], dtype=np.int32),
                "targets": np.asarray([1, -1], dtype=np.int32),
                "axes": np.asarray([-1, -1], dtype=np.int32),
            }
        },
        mujoco_options={"actuator_trnid": [[0, 0], [0, 0]]},
    )
    provider = NewtonProvider(artifact, num_envs=1, _bindings=bindings)
    try:
        imported = ET.fromstring(bindings.newton.builders[0].mjcf_call[0])
        position = imported.find("./actuator/position")
        assert position is not None
        assert position.attrib["name"] == "robot_joint_position"
        assert position.attrib["joint"] == "joint"
        assert position.attrib["kp"] == "150"
        assert position.attrib["kv"] == "5"
        assert position.attrib["forcerange"] == "-12 12"
        assert "gainprm" not in position.attrib
        assert "biasprm" not in position.attrib
        direct = imported.find("./actuator/general")
        assert direct is not None
        assert direct.attrib["name"] == "robot_joint_affine"
        assert direct.attrib["gainprm"] == "150 0 0"
        assert direct.attrib["biasprm"] == "0 -150 -5 0"
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
    provider = NewtonProvider(
        artifact,
        num_envs=2,
        device="cuda:0",
        use_mujoco_contacts=False,
    )
    try:
        assert provider.use_mujoco_contacts is False
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
            layout = provider.resolve_robot_layout(
                "newton_cartpole",
                base_link="cart",
                joint_names=("slider", "hinge"),
                link_names=("cart", "pole"),
            )
            assert layout.actuator_modes == ("position", "position")
            assert len(layout.joint_q_indices) == 2
            assert len(layout.joint_qd_indices) == 2
            assert len(layout.link_body_indices) == 2
            provider.set_joint_position_targets(
                layout,
                torch.zeros((2, 2), dtype=torch.float32, device="cuda:0"),
            )
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
            joint_names = (
                "FR_hip_joint",
                "FR_thigh_joint",
                "FR_calf_joint",
                "FL_hip_joint",
                "FL_thigh_joint",
                "FL_calf_joint",
                "RR_hip_joint",
                "RR_thigh_joint",
                "RR_calf_joint",
                "RL_hip_joint",
                "RL_thigh_joint",
                "RL_calf_joint",
            )
            layout = provider.resolve_robot_layout(
                "go1",
                base_link="trunk",
                joint_names=joint_names,
                link_names=("trunk", "FR_hip", "FR_thigh", "FR_calf"),
            )
            assert len(layout.base_joint_q_indices) == 7
            assert len(layout.base_joint_qd_indices) == 6
            assert len(layout.joint_q_indices) == 12
            initial_joint_position = torch.tensor(
                [[
                    0.1,
                    0.9,
                    -1.8,
                    -0.1,
                    0.9,
                    -1.8,
                    0.1,
                    0.9,
                    -1.8,
                    -0.1,
                    0.9,
                    -1.8,
                ]],
                device="cuda:0",
            )
            provider.reset_robot_state(
                layout,
                torch.ones(1, dtype=torch.bool, device="cuda:0"),
                base_pose=torch.tensor(
                    [[-4.0, 0.0, 1.198, 0.0, 0.0, 0.0, 1.0]],
                    device="cuda:0",
                ),
                base_velocity=torch.zeros((1, 6), device="cuda:0"),
                joint_position=initial_joint_position,
                joint_velocity=torch.zeros((1, 12), device="cuda:0"),
                controls=initial_joint_position,
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
    test_named_robot_layout_controls_and_reset()
    test_named_robot_layout_reports_name_and_actuator_errors()
    test_explicit_newton_contact_mode_generates_contacts_each_substep()
    test_availability_reports_missing_optional_package()
    test_import_version_failure_is_reported_as_unavailable()
    test_provider_rejects_non_cuda_device_and_dimension_mismatch()
    test_inline_mjcf_mesh_is_materialized_for_newton_import()
    test_mujoco_solref_shorthand_is_normalized_for_newton_import()
    test_mujoco_solref_shorthand_preserves_default_class_inheritance()
    test_explicit_model_config_overrides_mjcf_constraint_response()
    test_newton_model_config_validation_and_provider_type_check()
    test_mujoco_affine_position_general_is_normalized_for_joint_targets()
    test_direct_artifact_is_revalidated()
    test_optional_real_gpu_smoke()


if __name__ == "__main__":
    main()
