"""Real PPO updates: reject destructive steps and keep rollout statistics fixed."""
from copy import deepcopy
from dataclasses import asdict
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from gobot.rl.locomotion.guarded_ppo import GuardedPPO, PpoGuardCfg


def make_algorithm():
    torch.manual_seed(71)
    obs = TensorDict({'actor': torch.randn(8, 6), 'critic': torch.randn(8, 7)}, batch_size=[8])
    groups = {'actor': ['actor'], 'critic': ['critic']}
    actor = MLPModel(obs, groups, 'actor', 2, hidden_dims=[16], activation='elu',
                     obs_normalization=True, distribution_cfg={
                         'class_name': 'rsl_rl.modules.GaussianDistribution',
                         'init_std': .5, 'std_type': 'log', 'std_range': (.1, 1.)})
    critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=[16], activation='elu', obs_normalization=True)
    for model in (actor, critic):
        model.update_normalization(obs)
    alg = GuardedPPO(actor, critic, RolloutStorage('rl', 8, 4, obs, [2], 'cpu'),
                     guard_cfg=asdict(PpoGuardCfg(enabled=True)), num_learning_epochs=1,
                     num_mini_batches=2, learning_rate=3e-4)
    alg.train_mode()
    return alg, obs


def rollout(alg, obs):
    with torch.no_grad():
        for _ in range(4):
            alg.act(obs)
            alg.process_env_step(obs, torch.randn(8), torch.zeros(8, dtype=torch.bool), {})
        alg.compute_returns(obs)


def assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_state_equal(a, b)
    else:
        assert actual == expected


def test_rollout_normalizers_stay_fixed():
    alg, obs = make_algorithm()
    state = deepcopy(alg.actor.obs_normalizer.state_dict())
    rollout(alg, obs)
    assert_state_equal(alg.actor.obs_normalizer.state_dict(), state)
    result = alg.update()
    assert result['update_rejected'] == 0
    assert result['optimizer_steps'] > 0
    assert alg.actor.obs_normalizer.training is False
    assert all(torch.isfinite(torch.tensor(v)) for v in result.values())


def test_excessive_kl_restores_both_networks_and_optimizer():
    alg, obs = make_algorithm()
    rollout(alg, obs)
    alg.update()  # Populate real Adam momentum before testing rollback.
    rollout(alg, obs)
    before = deepcopy(alg.save())
    original_step = alg.optimizer.step

    def destructive_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        with torch.no_grad():
            alg.actor.mlp[-1].bias.add_(100.)
        return result

    alg.optimizer.step = destructive_step
    result = alg.update()
    assert result['update_rejected'] == 1
    assert result['guard_rollbacks'] == 1
    assert result['kl_max'] > alg.guard.rollback_kl
    for key in ('actor_state_dict', 'critic_state_dict'):
        assert_state_equal(alg.save()[key], before[key])
    # Only learning rate changes; all optimizer moments/counters must roll back.
    assert_state_equal(alg.optimizer.state_dict()['state'], before['optimizer_state_dict']['state'])
    assert alg.learning_rate < before['optimizer_guard']['learning_rate']
    assert alg.storage.step == 0


def test_normalization_distribution_shift_is_rejected():
    alg, obs = make_algorithm()
    before = [deepcopy(model.obs_normalizer.state_dict()) for model in (alg.actor, alg.critic)]
    shifted = TensorDict({'actor': torch.full((4096, 6), 1000.),
                          'critic': torch.full((4096, 7), 1000.)}, batch_size=[4096])
    divergence, rejected = alg._statistics_boundary(shifted)
    assert rejected == 1 and divergence > alg.guard.stop_kl
    for model, state in zip((alg.actor, alg.critic), before):
        assert_state_equal(model.obs_normalizer.state_dict(), state)
