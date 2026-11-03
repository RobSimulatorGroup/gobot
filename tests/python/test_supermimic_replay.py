# Adapted from Motphys/Uni-SuperMimic, faf7818e8abac3ec6abb8f00e6e904d3577f800b.
# SPDX-License-Identifier: Apache-2.0
"""Sampling and boundary tests for SuperMimic replay."""

from __future__ import annotations

import pytest
import torch

from gobot.rl.supermimic.replay import ReplayBuffer


def _transition(
    time: int,
    num_envs: int,
    num_joints: int,
    context_dim: int,
    ep_step: int,
    done: bool = False,
) -> dict[str, torch.Tensor]:
    base = time * 1000 + torch.arange(num_envs) * 100

    def field(dim: int) -> torch.Tensor:
        return base.unsqueeze(-1) + torch.arange(dim)

    return {
        "s": field(13 + 2 * num_joints),
        "s1": field(13 + 2 * num_joints) + 0.5,
        "a": field(num_joints),
        "a_prev": field(num_joints) - 0.5,
        "xi": field(context_dim),
        "r": base.float(),
        "f": torch.full((num_envs,), time, dtype=torch.long),
        "ep_step": torch.full((num_envs,), ep_step, dtype=torch.int32),
        "done": torch.full((num_envs,), done, dtype=torch.bool),
    }


def test_replay_storage_shapes_and_dtypes() -> None:
    buffer = ReplayBuffer(4, 2, "cpu", num_joints=3, context_dim=4)

    assert set(buffer.data) == {"s", "s1", "a", "a_prev", "xi", "r", "f", "ep_step", "done"}
    assert buffer.data["s"].shape == (4, 2, 19)
    assert buffer.data["s1"].shape == (4, 2, 19)
    assert buffer.data["a"].shape == (4, 2, 3)
    assert buffer.data["a_prev"].shape == (4, 2, 3)
    assert buffer.data["xi"].shape == (4, 2, 4)
    for key in ("s", "s1", "a", "a_prev", "xi", "r"):
        assert buffer.data[key].dtype == torch.float32
    for key in ("r", "f", "ep_step", "done"):
        assert buffer.data[key].shape == (4, 2)
    assert buffer.data["f"].dtype == torch.long
    assert buffer.data["ep_step"].dtype == torch.int32
    assert buffer.data["done"].dtype == torch.bool


def test_eight_step_window_masks_transitions_after_done() -> None:
    buffer = ReplayBuffer(8, 1, "cpu", num_joints=2, context_dim=3)
    for time in range(8):
        buffer.add(_transition(time, 1, 2, 3, ep_step=time, done=time in (2, 5)))

    batch = buffer.sample_windows(4, 8)

    expected_mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert batch["s"].shape == (4, 8, 17)
    assert batch["s1"].shape == (4, 8, 17)
    assert batch["a"].shape == (4, 8, 2)
    assert batch["xi"].shape == (4, 8, 3)
    assert batch["done"].dtype == torch.bool
    assert batch["mask"].dtype == torch.float32
    torch.testing.assert_close(batch["mask"], expected_mask.expand(4, 8))
    torch.testing.assert_close(batch["s"][..., 0], torch.arange(8).expand(4, 8) * 1000.0)


def test_history_repeats_episode_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = ReplayBuffer(8, 1, "cpu", num_joints=2, context_dim=3)
    episode_steps = [0, 1, 2, 0, 1, 2, 3, 4]
    for time, ep_step in enumerate(episode_steps):
        buffer.add(_transition(time, 1, 2, 3, ep_step=ep_step))

    def latest(low: int, high: int, size: tuple[int, ...], device: torch.device) -> torch.Tensor:
        del low
        return torch.full(size, high - 1, dtype=torch.long, device=device)

    monkeypatch.setattr(torch, "randint", latest)
    start, history = buffer.sample_starts(2, 8)

    expected_frames = torch.tensor([3, 3, 3, 3, 4, 5, 6, 7])
    torch.testing.assert_close(start["f"], torch.full((2,), 7))
    torch.testing.assert_close(history["f"], expected_frames.expand(2, 8))
    assert history["s"].shape == (2, 8, 17)
    assert history["a_prev"].shape == (2, 8, 2)
    assert history["xi"].shape == (2, 8, 3)


def test_history_does_not_cross_oldest_buffered_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = ReplayBuffer(5, 1, "cpu", num_joints=2, context_dim=3)
    for time in range(7):
        buffer.add(_transition(time, 1, 2, 3, ep_step=time))

    def oldest(low: int, high: int, size: tuple[int, ...], device: torch.device) -> torch.Tensor:
        del low, high
        return torch.zeros(size, dtype=torch.long, device=device)

    monkeypatch.setattr(torch, "randint", oldest)
    start, history = buffer.sample_starts(3, 4)

    torch.testing.assert_close(start["f"], torch.full((3,), 2))
    torch.testing.assert_close(history["f"], torch.full((3, 4), 2))
