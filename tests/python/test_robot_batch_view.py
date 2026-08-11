from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from gobot.rl.providers import (
    BatchPhysicsProvider,
    BatchProviderCapabilities,
    GraphInvalidatedError,
    ProviderCheckpoint,
    RobotBatchSpec,
    RobotBatchState,
    SensorBatchSpec,
)


class _Tensor:
    """Small NumPy-backed stand-in for the device tensor view contract."""

    def __init__(self, values: Any):
        self._array = np.asarray(values)

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> "_Tensor":
        return cls(np.zeros(shape, dtype=np.float32))

    @classmethod
    def ones(cls, shape: tuple[int, ...]) -> "_Tensor":
        return cls(np.ones(shape, dtype=np.float32))

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    def clone(self) -> "_Tensor":
        return _Tensor(self._array.copy())

    def copy_(self, other: "_Tensor") -> "_Tensor":
        np.copyto(self._array, other._array)
        return self

    def data_ptr(self) -> int:
        return int(self._array.__array_interface__["data"][0])

    def detach(self) -> "_Tensor":
        return self

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array

    def __getitem__(self, key: Any) -> "_Tensor":
        return _Tensor(self._array[key])

    def __setitem__(self, key: Any, value: Any) -> None:
        self._array[key] = value._array if isinstance(value, _Tensor) else value

    def __mul__(self, value: float) -> "_Tensor":
        return _Tensor(self._array * value)


def _equal(left: _Tensor, right: _Tensor) -> bool:
    return bool(np.array_equal(left._array, right._array))


class _Adapter:
    def __init__(self, provider, spec):
        self.provider = provider
        self.spec = spec

    def read_state(self, state):
        values = (
            self.provider.base_pose,
            self.provider.base_velocity,
            self.provider.joint_position,
            self.provider.joint_velocity,
            self.provider.joint_control,
            self.provider.link_pose,
        )
        if state is None:
            return RobotBatchState(*(value.clone() for value in values))
        for target, value in zip(state.__dict__.values(), values, strict=True):
            target.copy_(value)
        return state

    def set_position_targets(self, targets):
        self.provider.position_targets.copy_(targets)

    def set_base_pose_targets(self, targets):
        self.provider.base_pose_targets.copy_(targets)

    def set_controls(self, controls):
        self.provider.joint_control.copy_(controls)

    def reset(self, reset_mask, **state):
        self.provider.last_reset = (reset_mask.clone(), state)
        return self.provider.arrays


class _Provider(BatchPhysicsProvider):
    def __init__(self):
        self.artifact = SimpleNamespace(digest="fixture")
        self.generation = 1
        self.base_pose = _Tensor.zeros((2, 7))
        self.base_velocity = _Tensor.zeros((2, 6))
        self.joint_position = _Tensor.zeros((2, 2))
        self.joint_velocity = _Tensor.zeros((2, 2))
        self.joint_control = _Tensor.zeros((2, 2))
        self.position_targets = _Tensor.zeros((2, 2))
        self.base_pose_targets = _Tensor.zeros((2, 7))
        self.link_pose = _Tensor.zeros((2, 1, 7))
        self.sensor_values = _Tensor.zeros((2, 3))
        self.contact_sensors = {
            "feet": {"force": self.sensor_values, "num_slots": 1}
        }
        self.last_reset = None

    @property
    def capabilities(self):
        return BatchProviderCapabilities(
            "Fake",
            "cpu",
            True,
            False,
            True,
            True,
            runtime_checkpoint=True,
            sensor_batch=True,
            reset_scope="masked",
        )

    @property
    def num_envs(self):
        return 2

    @property
    def arrays(self):
        return {"joint_position": self.joint_position}

    @property
    def fixed_time_step(self):
        return 0.01

    def _checkpoint_array_names(self):
        return ("joint_position",)

    def step(self, actions=None, *, nsteps=1):
        return self.arrays

    def reset(self, reset_mask, **state):
        self.last_reset = (reset_mask, state)
        return self.arrays

    def close(self):
        self.generation += 1

    def _create_robot_view_adapter(self, spec):
        return _Adapter(self, spec)


class _Context:
    def __init__(self):
        self.update = None

    def apply_link_poses(self, links, poses):
        self.update = (links, poses.copy())


def test_robot_batch_view_contract() -> None:
    provider = _Provider()
    view = provider.create_robot_view(
        RobotBatchSpec("robot", "base", ("joint_a", "joint_b"), ("link",))
    )
    state = view.read_state()
    pointers = tuple(value.data_ptr() for value in state.__dict__.values())
    provider.base_pose[:, 2] = 1.25
    updated = view.read_state()
    assert updated is state
    assert tuple(value.data_ptr() for value in updated.__dict__.values()) == pointers
    assert _equal(updated.base_pose[:, 2], _Tensor([1.25, 1.25]))

    targets = _Tensor.ones((2, 2))
    base_targets = _Tensor.ones((2, 7))
    view.set_position_targets(targets)
    view.set_base_pose_targets(base_targets)
    view.set_controls(targets * 2.0)
    assert _equal(provider.position_targets, targets)
    assert _equal(provider.base_pose_targets, base_targets)
    assert _equal(provider.joint_control, targets * 2.0)

    mask = _Tensor([True, False])
    view.reset(mask, joint_position=targets)
    assert _equal(provider.last_reset[0], mask)

    context = _Context()
    view.bind_scene(context, ("link-handle",))
    view.sync_scene(env_index=1)
    assert context.update[0] == ("link-handle",)
    assert context.update[1].shape == (1, 7)
    try:
        view.sync_scene(env_index=2)
    except IndexError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("out-of-range scene sync environment was accepted")
    try:
        view.sync_scene(env_index=0.5)
    except TypeError as error:
        assert "integer" in str(error)
    else:
        raise AssertionError("a fractional scene sync environment was accepted")

    provider.close()
    try:
        view.read_state()
    except RuntimeError as error:
        assert "stale" in str(error)
    else:
        raise AssertionError("closed-provider view access must fail")


def test_robot_batch_view_rejects_replaced_state_storage() -> None:
    provider = _Provider()
    view = provider.create_robot_view(
        RobotBatchSpec("robot", "base", ("joint_a", "joint_b"), ("link",))
    )
    view.read_state()
    adapter = view._adapter
    adapter.read_state = lambda state: RobotBatchState(
        provider.base_pose.clone(),
        provider.base_velocity.clone(),
        provider.joint_position.clone(),
        provider.joint_velocity.clone(),
        provider.joint_control.clone(),
        provider.link_pose.clone(),
    )
    try:
        view.read_state()
    except GraphInvalidatedError as error:
        assert "storage changed" in str(error)
    else:
        raise AssertionError("replaced robot batch state storage was accepted")


def test_robot_batch_spec_rejects_invalid_names() -> None:
    for spec in (
        ("", "base", ("joint",)),
        ("robot", "", ("joint",)),
        ("robot", "base", ("joint", "joint")),
    ):
        try:
            RobotBatchSpec(spec[0], spec[1], spec[2])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid RobotBatchSpec must fail")


def test_robot_batch_view_reports_unsupported_base_targets() -> None:
    provider = _Provider()
    view = provider.create_robot_view(
        RobotBatchSpec("robot", "base", ("joint_a", "joint_b"))
    )
    view._adapter.set_base_pose_targets = None
    try:
        view.set_base_pose_targets(_Tensor.ones((2, 7)))
    except NotImplementedError as error:
        assert "kinematic base pose targets" in str(error)
    else:
        raise AssertionError("unsupported kinematic base targets were accepted")


def test_sensor_batch_view_keeps_named_device_storage_stable() -> None:
    provider = _Provider()
    view = provider.create_sensor_view(
        SensorBatchSpec("feet", "contact", ("force",))
    )
    state = view.read_state()
    assert tuple(state.values) == ("force",)
    pointer = state.values["force"].data_ptr()
    provider.sensor_values[:, :] = 2.5
    updated = view.read_state()
    assert updated is state
    assert updated.values["force"].data_ptr() == pointer
    assert np.all(updated.values["force"].numpy() == 2.5)

    provider.close()
    try:
        view.read_state()
    except RuntimeError as error:
        assert "stale" in str(error)
    else:
        raise AssertionError("closed-provider sensor view access must fail")


def test_provider_checkpoint_restores_full_batch_without_replacing_storage() -> None:
    provider = _Provider()
    provider.joint_position[:, :] = 1.25
    pointer = provider.joint_position.data_ptr()
    checkpoint = provider.capture_checkpoint()
    assert isinstance(checkpoint, ProviderCheckpoint)
    provider.joint_position[:, :] = -3.0
    provider.restore_checkpoint(checkpoint)
    assert provider.joint_position.data_ptr() == pointer
    assert np.all(provider.joint_position.numpy() == 1.25)

    incompatible = ProviderCheckpoint(
        schema_version=checkpoint.schema_version,
        provider_name=checkpoint.provider_name,
        runtime_fingerprint="different",
        num_envs=checkpoint.num_envs,
        fixed_time_step=checkpoint.fixed_time_step,
        device=checkpoint.device,
        arrays=checkpoint.arrays,
        auxiliary=checkpoint.auxiliary,
    )
    try:
        provider.restore_checkpoint(incompatible)
    except ValueError as error:
        assert "incompatible" in str(error)
    else:
        raise AssertionError("an incompatible checkpoint was accepted")


if __name__ == "__main__":
    test_robot_batch_view_contract()
    test_robot_batch_view_rejects_replaced_state_storage()
    test_robot_batch_spec_rejects_invalid_names()
    test_robot_batch_view_reports_unsupported_base_targets()
    test_sensor_batch_view_keeps_named_device_storage_stable()
    test_provider_checkpoint_restores_full_batch_without_replacing_storage()
