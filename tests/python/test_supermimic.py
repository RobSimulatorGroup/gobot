"""Official independent oracle: GenCoord math, BPTT and exact replay boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from gobot.rl.supermimic.models import Dynamics, MultiEncoderAE, build_mlp
from gobot.rl.supermimic.optim import _optimize_step
from gobot.rl.supermimic.policy_learner import (
    ActorCriticPolicyLearner,
    ActorCriticPolicyLearnerCfg,
)
from gobot.rl.supermimic.replay import FieldSpec, ReplayBuffer
from gobot.rl.supermimic.rewards import tracking_reward
from gobot.rl.supermimic.state import (
    GenCoordRepresentation,
    goal_window_indices,
    term_major_to_temporal,
)
from gobot.rl.supermimic.tasks import EncoderSpec
from gobot.rl.supermimic.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from examples.g1.supermimic._reference.probes import DifferentiableProbeEnv

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures/supermimic_official_gencoord_v1.pt"
)


@pytest.fixture(scope="module")
def reference():
    metadata = json.loads(FIXTURE.with_suffix(".json").read_text())
    assert metadata["gobot_imported"] is False
    assert (
        metadata["sources"]["supermimic"]["commit"]
        == "abc124c1bd5976341034668af3d12c087f2f1f55"
    )
    assert (
        hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == metadata["fixture_sha256"]
    )
    tools = Path(__file__).resolve().parents[2] / "examples/g1/supermimic"
    for name, key in (
        ("reference.lock.json", "lock_sha256"),
        ("_reference/export.py", "exporter_sha256"),
        ("_reference/probes.py", "probe_sha256"),
        ("reference.py", "verifier_sha256"),
    ):
        assert hashlib.sha256((tools / name).read_bytes()).hexdigest() == metadata[key]
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield torch.load(FIXTURE, weights_only=True)
    torch.set_num_threads(previous)


def assert_tree(actual, expected, *, parameters=False):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(
            actual, expected, atol=1e-6 if parameters else 1e-5, rtol=1e-4
        )
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert_tree(actual[key], value, parameters=parameters)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_tree(a, b, parameters=parameters)
    else:
        assert actual == expected


def make_world_model(reference):
    rep = GenCoordRepresentation(29, reference["default"])
    dynamics = Dynamics(
        rep, 29, 33, reference["action_scale"], reference["default"], (16, 12, 8)
    )
    dynamics.load_state_dict(reference["wm_initial"])
    dynamics.eval()
    return rep, dynamics, WorldModelLearner(dynamics, rep, WorldModelLearnerCfg())


@pytest.mark.parametrize("horizon", [1, 8, 32])
def test_state_rollout_loss_and_gradients_match_official(reference, horizon):
    rep, dynamics, learner = make_world_model(reference)
    states = reference["states"][:, : horizon + 1]
    actions = reference["actions"][:, : horizon + 1].clone().requires_grad_(True)
    rep_states = rep.represent(states)
    predicted, aux = learner._rollout(
        rep_states[:, 0], actions, states[:, 0, :7], reference["context"]
    )
    loss, groups = learner._compute_losses(
        predicted, aux, rep_states[:, 1:], states[:, 1:, :7]
    )
    loss.backward()
    actual = {
        "states": rep_states,
        "predicted": predicted,
        "aux": aux,
        "reconstructed": rep.reconstruct_full_state(aux, predicted),
        "loss": loss,
        "groups": groups,
        "grads": {n: p.grad for n, p in dynamics.named_parameters()},
        "action_grad": actions.grad,
    }
    assert_tree(actual, reference["wm"][str(horizon)])
    assert actions.grad[:, 0].abs().sum() > 0
    assert (
        torch.count_nonzero(actions.grad[:, -1]) == 0
    )  # Last stored action is not scored.
    torch.testing.assert_close(
        rep.reconstruct_full_state(states[:, 0, :7], rep_states[:, 0]),
        states[:, 0],
        atol=1e-6,
        rtol=1e-5,
    )


def test_world_model_adam_muon_update_and_optimizer_state_match_official(reference):
    rep, dynamics, learner = make_world_model(reference)
    source = reference["states"][:, :8]
    pred, aux = learner._rollout(
        rep.represent(source)[:, 0],
        reference["actions"][:, :8],
        source[:, 0, :7],
        reference["context"],
    )
    loss, _ = learner._compute_losses(
        pred, aux, rep.represent(source)[:, 1:], source[:, 1:, :7]
    )
    norm = _optimize_step(
        learner.optimizers, loss, list(dynamics.parameters()), max_norm=1.0
    )
    actual = {
        "loss": loss,
        "norm": norm,
        "parameters": dynamics.state_dict(),
        "grads": {n: p.grad for n, p in dynamics.named_parameters()},
        "optimizers": [o.state_dict() for o in learner.optimizers],
    }
    assert_tree(actual, reference["wm_update"], parameters=True)
    assert isinstance(learner.optimizers[0], torch.optim.Adam)
    assert isinstance(learner.optimizers[1], torch.optim.Muon)


def test_public_world_model_update_matches_official_row_convention(reference):
    _, dynamics, learner = make_world_model(reference)
    batch = TensorDict(
        {
            "state": reference["states"][:, :8],
            "action": reference["actions"][:, :8],
            "sim_ctx": reference["context"][:, None].expand(-1, 8, -1),
        },
        batch_size=[4, 8],
    )
    stats = learner.update([batch])
    assert_tree(stats["Loss/state"], reference["wm_update"]["loss"])
    assert_tree(
        dynamics.state_dict(), reference["wm_update"]["parameters"], parameters=True
    )


def test_normalization_uses_processed_action_and_keeps_captured_buffers(reference):
    rep = GenCoordRepresentation(29, reference["default"])
    dynamics = Dynamics(
        rep, 29, 33, reference["action_scale"], reference["default"], (16, 12, 8)
    )
    learner = WorldModelLearner(dynamics, rep, WorldModelLearnerCfg())
    addresses = [n._std.data_ptr() for n in dynamics.normalizers()]
    learner.update_normalization(reference["states"], reference["actions"])
    for name in ("state_normalizer", "action_normalizer"):
        normalizer = getattr(dynamics, name)
        expected = {
            k.removeprefix(name + "."): v
            for k, v in reference["wm_initial"].items()
            if k.startswith(name + ".")
        }
        assert_tree(normalizer.state_dict(), expected)
    assert addresses == [n._std.data_ptr() for n in dynamics.normalizers()]
    assert not any(n.training for n in dynamics.normalizers())


def test_official_goal_windows_do_not_cross_clip_boundaries(reference):
    rep = GenCoordRepresentation(29, reference["default"])
    goal = reference["goal"]
    motion = SimpleNamespace(**goal["motion"])
    indices = goal_window_indices(motion, goal["times"])
    assert torch.equal(indices, goal["indices"])
    assert torch.equal(indices[1], torch.full((10,), 3))
    assert torch.equal(indices[-1], torch.full((10,), 59))
    assert torch.equal(indices[2], torch.arange(4, 50, 5))
    wm, actor = rep.compute_wm_goal(
        motion, goal["times"], reference["states"][:, 0, :7]
    )
    assert wm.shape == (4, 670) and actor.shape == (4, 640)
    assert_tree(wm, goal["world_model"])
    assert_tree(actor, goal["actor"])


def test_proprio_and_temporal_history_layout(reference):
    rep = GenCoordRepresentation(29, reference["default"], actor_obs_mode="proprio")
    full = reference["states"][:, 0]
    state = rep.represent(full)
    obs = rep.compute_wm_obs(
        state, full[:, :7], rep.joint_state(state), reference["context"]
    )
    assert_tree(obs, reference["proprio"])
    assert rep.proprio_obs_dim(29) == 93
    widths, h = [3, 3, 29, 29, 29], 3
    frames = torch.arange(h * 93).reshape(1, h, 93)
    term_major = torch.cat([x.flatten(1) for x in frames.split(widths, -1)], -1)
    assert torch.equal(term_major_to_temporal(term_major, widths, h), frames)
    assert torch.equal(
        rep.build_actor_obs(frames, torch.ones(1, 640))[:, : h * 93], frames.flatten(1)
    )
    assert torch.count_nonzero(rep._proprio_noise_bounds[-29:]) == 0


@pytest.mark.parametrize("horizon", [1, 8, 32])
def test_actor_critic_update_matches_official_on_synthetic_dynamics(reference, horizon):
    expected = reference["policy"][str(horizon)]
    starts = TensorDict(reference["policy_starts"], batch_size=[4])
    specs = [EncoderSpec("motion", 640), EncoderSpec("velocity", 3)]
    actor = MultiEncoderAE(
        68, 29, specs, {"motion": 0.5, "velocity": 0.5}, 32, (16, 12, 8), 0.1, 1
    )
    critic = build_mlp(741, (12, 8), 1)
    actor.load_state_dict(expected["actor"])
    critic.load_state_dict(expected["critic"])
    env = DifferentiableProbeEnv(starts)
    env.dynamics.load_state_dict(expected["dynamics"])
    assert_tree(actor(env.reset(starts)), expected["action"])
    learner = ActorCriticPolicyLearner(
        actor, critic, env, ActorCriticPolicyLearnerCfg(horizon=horizon, batch_size=4)
    )
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(expected["rng"])
        actual = learner.update()
    assert_tree(actual, expected["metrics"])
    for name, module in (("actor", actor), ("critic", critic)):
        assert_tree(module.state_dict(), expected[name + "_after"], parameters=True)
        assert_tree(
            {n: p.grad for n, p in module.named_parameters()}, expected[name + "_grads"]
        )
        assert_tree(
            [o.state_dict() for o in getattr(learner, name + "_optimizers")],
            expected[name + "_optimizers"],
            parameters=True,
        )
    assert expected["dynamics_grad_none"]
    assert all(
        p.grad is None and not p.requires_grad for p in env.dynamics.parameters()
    )
    assert sum(p.grad.abs().sum() for p in actor.parameters() if p.grad is not None) > 0


@pytest.mark.parametrize("length", [1, 3, 8])
def test_replay_wrap_termination_and_history_padding_match_official(reference, length):
    schema = {
        "state": FieldSpec(2),
        "terminated": FieldSpec(dtype=torch.bool),
        "truncated": FieldSpec(dtype=torch.bool),
    }
    buffer = ReplayBuffer(schema, 12, 3)
    for row in reference["replay"]["rows"]:
        buffer.insert(TensorDict(row, batch_size=[3]))
    expected = reference["replay"]["samples"][str(length)]
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(expected["rng"])
        sample = buffer.sample(length, 12)
        causal, padded = buffer.sample_causal(length, 12)
    for key in schema:
        assert torch.equal(sample[key], expected["sample"][key])
        assert torch.equal(causal[key], expected["causal"][key])
    assert torch.equal(padded, expected["padded"])
    assert not (sample["terminated"][:, :-1] | sample["truncated"][:, :-1]).any()
    if length > 1:
        assert torch.equal(
            sample["state"][:, 1:, 1] - sample["state"][:, :-1, 1],
            torch.ones(12, length - 1),
        )


def test_tracking_reward_matches_official_pelvis_anchor(reference):
    assert_tree(
        tracking_reward(**reference["reward"]["inputs"]), reference["reward"]["value"]
    )


@pytest.mark.parametrize("decimation", [1, 2, 4, 8])
def test_zero_and_small_delta_gradients_remain_finite(decimation):
    rep = GenCoordRepresentation(2, torch.zeros(2), decimation=decimation).double()
    assert rep.substep_coeff == (decimation + 1) / (2 * decimation)
    full = torch.zeros(2, 17, dtype=torch.float64)
    full[:, 2] = 0.7
    full[:, 3] = 1.0
    full[:, 9] = 1.0
    for magnitude in (0.0, 1e-8, 1e-2, 1.0):
        delta = torch.full((2, 8), magnitude, dtype=torch.float64, requires_grad=True)
        aux, state = rep.apply_prediction(full[:, :7], rep.represent(full), delta, 0.02)
        (aux.square().sum() + state.square().sum()).backward()
        assert torch.isfinite(delta.grad).all()
        if magnitude == 0:
            torch.testing.assert_close(
                aux[:, 0], torch.full((2,), 0.02, dtype=torch.float64)
            )
    assert rep._default_joint_pos.dtype == torch.float64


@pytest.mark.parametrize(
    "field,value",
    [
        ("dt", 0),
        ("dt", float("nan")),
        ("decimation", 0),
        ("decimation", True),
        ("actor_obs_mode", "legacy"),
    ],
)
def test_invalid_representation_contract_fails_early(field, value):
    with pytest.raises(ValueError):
        GenCoordRepresentation(29, torch.zeros(29), **{field: value})
