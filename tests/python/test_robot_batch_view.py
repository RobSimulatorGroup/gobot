from __future__ import annotations

from types import SimpleNamespace

import torch

from gobot.rl.providers import (
    BatchPhysicsProvider,
    BatchProviderCapabilities,
    RobotBatchSpec,
    RobotBatchState,
)


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

    def set_controls(self, controls):
        self.provider.joint_control.copy_(controls)

    def reset(self, reset_mask, **state):
        self.provider.last_reset = (reset_mask.clone(), state)
        return self.provider.arrays


class _Provider(BatchPhysicsProvider):
    def __init__(self):
        self.artifact = SimpleNamespace(digest="fixture")
        self.generation = 1
        self.base_pose = torch.zeros((2, 7))
        self.base_velocity = torch.zeros((2, 6))
        self.joint_position = torch.zeros((2, 2))
        self.joint_velocity = torch.zeros((2, 2))
        self.joint_control = torch.zeros((2, 2))
        self.position_targets = torch.zeros((2, 2))
        self.link_pose = torch.zeros((2, 1, 7))
        self.last_reset = None

    @property
    def capabilities(self):
        return BatchProviderCapabilities("Fake", "cpu", True, False, True, True)

    @property
    def num_envs(self):
        return 2

    @property
    def arrays(self):
        return {"joint_position": self.joint_position}

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
    assert torch.equal(updated.base_pose[:, 2], torch.tensor([1.25, 1.25]))

    targets = torch.ones((2, 2))
    view.set_position_targets(targets)
    view.set_controls(targets * 2.0)
    assert torch.equal(provider.position_targets, targets)
    assert torch.equal(provider.joint_control, targets * 2.0)

    mask = torch.tensor([True, False])
    view.reset(mask, joint_position=targets)
    assert torch.equal(provider.last_reset[0], mask)

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

    provider.close()
    try:
        view.read_state()
    except RuntimeError as error:
        assert "stale" in str(error)
    else:
        raise AssertionError("closed-provider view access must fail")


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


if __name__ == "__main__":
    test_robot_batch_view_contract()
    test_robot_batch_spec_rejects_invalid_names()
