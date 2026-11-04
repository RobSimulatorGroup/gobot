"""Independent reference fixtures: math, gradients and optimizer updates."""
from pathlib import Path
import hashlib
import json

import pytest
import torch

from gobot.rl.supermimic.heads import AnalyticHeads, ReferenceFrame, analytic_observation, flatten_history_term_major
from gobot.rl.supermimic.policy import Policy, PolicyCfg, Critic, CriticCfg
from gobot.rl.supermimic.rewards import TrackingRewardCfg, tracking_reward_terms
from gobot.rl.supermimic.state_spec import StateSpec, StateSpecCfg
from gobot.rl.supermimic.world_model import WorldModel, WorldModelCfg

FIXTURE = Path(__file__).resolve().parents[1]/'fixtures/supermimic_reproduced_math_v1.pt'


@pytest.fixture(scope='module')
def reference():
    metadata = json.loads(FIXTURE.with_suffix('.json').read_text())
    assert metadata['gobot_imported'] is False
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == metadata['fixture_sha256']
    torch.set_num_threads(2)
    return torch.load(FIXTURE, weights_only=True)


def test_observation_layout_and_fixed_scale_policy_match_reference(reference):
    data = reference
    obs = analytic_observation(data['history'], data['previous_actions'], data['context'], data['default'])
    torch.testing.assert_close(obs, data['observations'], atol=0, rtol=0)
    flat = flatten_history_term_major(obs)
    assert flat.shape == (6, 930)
    torch.testing.assert_close(flat, data['flat_history'], atol=0, rtol=0)
    cfg = PolicyCfg(encoder_hidden=(32, 16), decoder_hidden=(32, 16))
    policy = Policy(cfg)
    policy.load_state_dict(data['policy'])
    torch.testing.assert_close(policy(flat, data['goals']), data['policy_output'], atol=1e-6, rtol=1e-5)
    assert not any('norm' in name for name in policy.state_dict())


def test_state_features_and_critic_match_reference(reference):
    data = reference
    spec = StateSpec(StateSpecCfg(), 'cpu')
    canonical = spec.canonical(data['states'][0])
    xi = spec.normalize_xi(data['context'])
    torch.testing.assert_close(canonical, data['canonical'], atol=0, rtol=0)
    torch.testing.assert_close(xi, data['normalized_context'], atol=0, rtol=0)
    critic = Critic(CriticCfg(hidden=(16, 8)))
    critic.load_state_dict(data['critic'])
    torch.testing.assert_close(critic(canonical, data['goals'], xi, data['eroot']),
                               data['critic_output'], atol=1e-6, rtol=1e-5)


def test_all_reward_terms_match_independent_torso_anchor_reference(reference):
    terms = tracking_reward_terms(**reference['reward_input'], cfg=TrackingRewardCfg())
    assert len(terms) == 9  # Eight weighted terms plus their sum.
    for name, value in terms.items():
        torch.testing.assert_close(value, reference['reward_terms'][name], atol=1e-6, rtol=1e-5)


def test_autoregressive_loss_gradients_and_adam_update_match_reference(reference):
    data = reference
    model = WorldModel(WorldModelCfg(hidden=(32, 32), term_hidden=(8, 8)), AnalyticHeads, 'cpu')
    model.load_state_dict(data['wm_initial'])
    refs = [ReferenceFrame(None, None, None, None, p, q)
            for p, q in zip(data['ref_pos'], data['ref_quat'], strict=True)]
    actions = data['actions'].clone().requires_grad_(True)
    loss, metrics = model.window_loss(data['states'], actions, data['context'], data['done'], refs)
    torch.testing.assert_close(loss, data['wm_loss'], atol=1e-5, rtol=1e-5)
    for name, value in metrics.items():
        torch.testing.assert_close(value, data['wm_metrics'][name], atol=1e-5, rtol=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    loss.backward()
    assert torch.isfinite(actions.grad).all() and actions.grad.abs().sum() > 0
    torch.testing.assert_close(actions.grad, data['action_gradient'], atol=1e-6, rtol=1e-4)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, data['wm_grads'][name], atol=1e-5, rtol=1e-4)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, data['wm_after'][name], atol=1e-6, rtol=1e-4)


def test_zero_model_output_extrapolates_velocity_and_preserves_stationary_pose():
    spec = StateSpec(StateSpecCfg(), 'cpu', torch.float64)
    state = torch.zeros(2, 71, dtype=torch.float64)
    state[:, 3] = 1.
    state[0, 36] = 2.
    state[0, 42] = 3.
    after = spec.integrate(state, torch.zeros(2, 70, dtype=torch.float64))
    assert after[0, 0] == pytest.approx(.04)
    assert after[0, 7] == pytest.approx(.06)
    torch.testing.assert_close(after[1], state[1], atol=0, rtol=0)


def test_perfect_tracking_reaches_declared_step_reward(reference):
    inputs = dict(reference['reward_input'])
    for field in ('pos', 'quat', 'lin_vel', 'ang_vel'):
        inputs[f'ref_body_{field}_w'] = inputs[f'robot_body_{field}_w']
    inputs['prev_action'] = inputs['action']
    inputs['joint_pos'] = torch.zeros_like(inputs['joint_pos'])
    cfg = TrackingRewardCfg()
    terms = tracking_reward_terms(**inputs, cfg=cfg)
    torch.testing.assert_close(terms['total'], torch.full((6,), cfg.max_step_reward), atol=1e-6, rtol=1e-5)
