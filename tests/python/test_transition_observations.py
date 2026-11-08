"""Transition diagnostics must not change the trajectory used by PPO."""
import os
import unittest

import numpy as np
import torch

from gobot.rl.locomotion import make_velocity_env
from gobot.rl.locomotion.experiment import Experiment, task_config
from gobot.rl.locomotion.training_log import TrainingStatistics


def snapshot(value):
    return torch.as_tensor(np.array(value, copy=True)) if not isinstance(value, torch.Tensor) else value.detach().cpu().clone()


def check_diagnostics(backend):
    cfg = task_config(Experiment(robot='g1', stage='flat'))
    cfg.capture_transitions = True
    cfg.episode_length_s = .1  # Exercise autoreset, command resampling and pushes.
    cfg.command.resampling_time_range = (.04, .08)
    cfg.push_interval_range_s = (.04, .08)
    device = 'cuda:0' if backend == 'mujoco-warp' else 'cpu'
    env = make_velocity_env(cfg, backend=backend, num_envs=8, device=device, seed=71,
                            collect_step_extras=False, collect_training_stats=True)
    try:
        records = []
        for capture, statistics in ((False, False), (True, False), (True, True)):
            env.cfg_obj.capture_transitions = capture
            env.common_step_counter = env.profile_step_counter = 0
            env.reset(seed=71)
            rows = []
            log = TrainingStatistics(env.num_envs, env.num_actions, env._reward_term_names, cfg, device)
            for _ in range(12):
                action = torch.zeros(env.num_envs, env.num_actions, device=device)
                state = env.step(action)
                rows.append({**{key: snapshot(value) for key, value in state.obs.items()},
                             'reward': snapshot(state.reward), 'done': snapshot(state.done),
                             'command': snapshot(env.locomotion_state()['command'])})
                rows[-1]['observation_noise'] = rows[-1]['actor'] - rows[-1]['critic'][:, :env.num_obs]
                if backend == 'mujoco-warp':
                    rows[-1]['rng'] = env._generator.get_state().cpu().clone()
                if statistics:
                    log.record(action, torch.as_tensor(state.reward, device=device),
                               torch.as_tensor(state.done, device=device),
                               {key: torch.as_tensor(value, device=device) for key, value in state.info['training_stats'].items()})
                    for key in ('actor', 'critic'):
                        torch.testing.assert_close(snapshot(state.obs[key]), rows[-1][key], rtol=0, atol=0)
                assert ('transition_observation' in state.info) == capture
            records.append(rows)
        for other in records[1:]:
            assert any(row['done'].any() for row in other)
            for expected, actual in zip(records[0], other, strict=True):
                for key in expected:
                    # Parallel contact solves can round differently across replays.
                    # The RNG/done flags must match exactly; noise must match even
                    # when the underlying joint velocities differ slightly.
                    if key in {'rng', 'done'}:
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                    elif key == 'observation_noise':
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=2.e-6)
                    else:
                        torch.testing.assert_close(actual[key], expected[key], rtol=1.e-4, atol=1.e-3)

    finally:
        env.close()


class TransitionObservationTests(unittest.TestCase):
    def test_cpu_diagnostics_preserve_training_data(self):
        check_diagnostics('mujoco-cpu')

    @unittest.skipUnless(os.environ.get('GOBOT_RUN_WARP_IPC_GPU_TEST') == '1',
                         'set GOBOT_RUN_WARP_IPC_GPU_TEST=1 for the CUDA diagnostics regression')
    def test_warp_diagnostics_preserve_training_data(self):
        check_diagnostics('mujoco-warp')


if __name__ == '__main__':
    unittest.main()
