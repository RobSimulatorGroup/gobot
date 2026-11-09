"""Episode boundaries, weighted rank reduction, and pre-reset logging data."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gobot.rl.locomotion.distributed import DistributedTraining
from examples.locomotion.presets import Experiment, task_config
from gobot.rl.locomotion.training_log import TrainingStatistics, format_training_log, step_statistics


def environment(cfg, levels=(0, 4), kinds=(0, 1)):
    return SimpleNamespace(cfg_obj=cfg, terrain_assignments=(levels, kinds), terrain_type_names=('flat', 'stairs'))


def stats(rates, reasons):
    count = len(rates)
    metrics = torch.zeros(count, 9)
    metrics[:, 6:8] = torch.tensor([.3, .1])
    metrics[:, 8] = 1
    return {'reward_terms': torch.tensor(rates, dtype=torch.float32).reshape(count, 1),
            'metrics': metrics, 'reset_reason': torch.tensor(reasons)}


def test_episode_sums_survive_update_and_reset_only_finished_worlds():
    cfg = task_config(Experiment(robot='g1', stage='flat'))
    log = TrainingStatistics(2, 1, ('upright',), cfg, 'cpu')
    action = torch.ones(2, 1)
    log.record(action, torch.tensor([1., 3.]), torch.tensor([True, False]), stats([2., 4.], [1, 0]))
    first = log.summarize(DistributedTraining('cpu'), environment(cfg))
    assert first['episodes_completed'] == 1 and first['episode_reward'] == 1.
    assert first['log']['Episode_Reward/upright'] == pytest.approx(2 * .02 / 20)
    assert first['log']['Episode_Termination/fell_over'] == 1
    assert first['log']['Metrics/landing_force_mean'] is None
    log.begin_update()
    log.record(action, torch.tensor([5., 7.]), torch.tensor([True, True]), stats([6., 8.], [2, 3]))
    second = log.summarize(DistributedTraining('cpu'), environment(cfg))
    assert second['episode_reward'] == 7.5  # New world 0: 5; world 1: 3+7.
    assert second['episode_steps'] == 1.5
    assert second['log']['Episode_Reward/upright'] == pytest.approx((6 + 4 + 8) * .02 / 2 / 20)
    assert second['log']['Episode_Metrics/mean_action_acc'] == 1.
    assert second['log']['Metrics/twist/error_vel_xy'] == pytest.approx(.3)
    assert second['log']['Episode_Termination/time_out'] == 1
    assert second['log']['Episode_Termination/out_of_terrain_bounds'] == 1
    log.begin_update()
    empty = log.summarize(DistributedTraining('cpu'), environment(cfg))
    assert empty['episode_reward'] is None and empty['episodes_completed'] == 0
    assert empty['log']['Episode_Reward/upright'] is None


def test_rank_reduction_weights_episodes_contacts_and_terrain_population():
    cfg = task_config(Experiment(robot='g1'))
    worker = TrainingStatistics(2, 1, ('upright',), cfg, 'cpu')
    worker.record(torch.zeros(2, 1), torch.tensor([2., 0.]), torch.tensor([True, False]), stats([1., 0.], [1, 0]))
    remote = deepcopy(worker)
    remote.completed *= 8  # Different completed reward total on the peer.
    remote.terminations *= 2  # Two completions, rather than one.

    class Workers:
        def reduce(self, value, maximum=False):
            if maximum:
                return [9.]
            other = torch.cat((remote.completed, remote.step_sums, remote.terminations,
                               torch.tensor([12., 0., 2., 0.])))
            return (value + other).tolist()

    result = worker.summarize(Workers(), environment(cfg))
    assert result['episodes_completed'] == 3
    assert result['episode_reward'] == 6.  # (2+16)/3, not mean(2,8).
    assert result['log']['Curriculum/terrain_levels/flat'] == 4.  # (0+12)/(1+2).
    assert result['log']['Curriculum/terrain_levels/stairs'] == 4.
    assert result['log']['Curriculum/terrain_levels/max'] == 9.


def test_numpy_and_torch_step_statistics_use_landing_and_contact_counts():
    values = dict(momentum=[[3., 4., 0.]], foot_velocity=[[[3., 4., 0.], [0., 2., 0.]]],
                  contact=[[True, False]], first_contact=[[False, True]], peak_height=[[.1, .3]],
                  foot_force=[[[3., 4., 0.], [0., 0., 20.]]],
                  linear_velocity=[[.8, .3, 1.]], angular_velocity=[[1., 2., .4]], command=[[.8, 0., .6]])
    expected = np.array([[5., .3, 5., 20., 1., 1., .3, .2, 1.]])
    for xp, array in ((np, np.asarray), (torch, torch.as_tensor)):
        result = step_statistics(xp, **{key: array(value) for key, value in values.items()})
        np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_cpu_training_statistics_keep_terminal_rewards_before_reset():
    from gobot.rl import RslRlVecEnvWrapper
    from gobot.rl.locomotion import make_velocity_env
    cfg = task_config(Experiment(robot='g1', stage='flat'))
    env = make_velocity_env(cfg, backend='mujoco-cpu', num_envs=2, seed=13, max_episode_length=2,
                            collect_step_extras=False, collect_training_stats=True)
    try:
        wrapper = RslRlVecEnvWrapper(env)
        for index in range(2):
            _, reward, done, extras = wrapper.step(torch.zeros(2, env.num_actions))
            training = extras['training_stats']
            torch.testing.assert_close(training['reward_terms'].sum(1) * env.step_dt, reward)
            assert torch.isfinite(training['metrics']).all()
            assert training['reset_reason'].tolist() == [2 if index else 0] * 2
        assert done.all()
    finally:
        env.close()


def test_console_shows_global_steps_and_missing_samples():
    metrics = dict(iteration=67, total_steps=13369344, steps_per_second=309618., collection_s=.535,
                   learning_s=.1, episode_reward=None, episode_steps=None, action_std=.87,
                   learning_rate=.001, world_size=8, total_num_envs=8192, episodes_completed=0,
                   log={'Episode_Reward/upright': None, 'Episode_Termination/fell_over': 0},
                   action_clip_fraction=0., joint_effort_saturation_fraction=.01, numerical_resets=0,
                   iteration_s=.64, elapsed_s=46.)
    text = format_training_log(metrics, target_iteration=18000, remaining_iterations=17932, average_iteration_s=.64)
    assert 'Learning iteration 67/18000' in text and '13,369,344' in text
    assert 'Mean reward: —' in text and 'Episode_Reward/upright: —' in text
    assert 'nan' not in text.lower()
