"""Only time-limit final states participate in PPO value bootstrapping."""

from types import SimpleNamespace

import pytest
import torch
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from gobot.rl.locomotion.transition_ppo import TransitionPPO


def _collector():
    critic_inputs = []
    stored_rewards = []

    class Critic:
        def __call__(self, observations):
            critic_inputs.append(observations.clone())
            return observations['critic']

        def reset(self, dones):
            pass

    algorithm = TransitionPPO.__new__(TransitionPPO)
    algorithm.actor = algorithm.critic = Critic()
    algorithm.device = 'cpu'
    algorithm.gamma = .9
    algorithm.fall_penalty = .25
    algorithm.rnd = None
    algorithm.transition = RolloutStorage.Transition()
    algorithm.storage = SimpleNamespace(
        add_transition=lambda transition: stored_rewards.append(transition.rewards.clone()))
    return algorithm, critic_inputs, stored_rewards


def test_bootstrap_ignores_nonfinite_final_states_of_other_worlds():
    algorithm, critic_inputs, stored_rewards = _collector()
    reset_observations = TensorDict({'actor': torch.zeros(4, 1), 'critic': torch.zeros(4, 1)}, [4])
    rewards = torch.tensor([1., 2., 3., 4.])
    dones = torch.tensor([True, True, False, True])
    final = {'actor': torch.tensor([[float('nan')], [1.], [float('inf')], [2.]]),
             'critic': torch.tensor([[float('nan')], [4.], [float('inf')], [6.]])}
    algorithm.process_env_step(reset_observations, rewards, dones, {
        'time_outs': torch.tensor([False, True, False, True]),
        'final_observation': final,
    })

    assert len(critic_inputs) == len(stored_rewards) == 1
    torch.testing.assert_close(critic_inputs[0]['critic'], torch.tensor([[4.], [6.]]))
    torch.testing.assert_close(critic_inputs[0]['actor'], torch.tensor([[1.], [2.]]))
    torch.testing.assert_close(stored_rewards[0], torch.tensor([.75, 5.6, 3., 9.4]))
    torch.testing.assert_close(rewards, torch.tensor([1., 2., 3., 4.]))
    assert torch.isnan(final['critic'][0]).all()


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('group', ['actor', 'critic'])
def test_nonfinite_timeout_final_state_fails_before_critic_or_storage(value, group):
    algorithm, critic_inputs, stored_rewards = _collector()
    reset_observations = TensorDict({'actor': torch.zeros(2, 1), 'critic': torch.zeros(2, 1)}, [2])
    final = {'actor': torch.ones(2, 1), 'critic': torch.ones(2, 1)}
    final[group][1, 0] = value
    with pytest.raises(ValueError, match=f"observation group '{group}'.*(NaN|Inf)"):
        algorithm.process_env_step(reset_observations, torch.ones(2), torch.ones(2, dtype=torch.bool), {
            'time_outs': torch.tensor([False, True]),
            'final_observation': final,
        })

    assert not critic_inputs
    assert not stored_rewards
    assert algorithm.transition.rewards is None
    assert not torch.isfinite(final[group][1, 0])


def test_no_timeouts_never_reads_final_observations():
    algorithm, critic_inputs, stored_rewards = _collector()
    reset_observations = TensorDict({'critic': torch.zeros(2, 1)}, [2])
    algorithm.process_env_step(reset_observations, torch.ones(2), torch.tensor([True, False]), {
        'time_outs': torch.zeros(2, dtype=torch.bool),
        'final_observation': {'critic': torch.full((2, 1), float('nan'))},
    })
    assert not critic_inputs
    torch.testing.assert_close(stored_rewards[0], torch.tensor([.75, 1.]))
