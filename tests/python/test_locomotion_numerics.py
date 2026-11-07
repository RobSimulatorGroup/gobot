"""Reject finite physics explosions and keep isolated bad worlds out of PPO."""
import os
from pathlib import Path

import pytest
import torch

from gobot.rl.locomotion.warp_env import WarpVelocityEnv


def test_physics_failures_include_finite_velocity_explosions():
    env = WarpVelocityEnv.__new__(WarpVelocityEnv)
    env._numerical_reset = torch.zeros(5, dtype=torch.bool)
    env._arrays = {name: torch.zeros(5, 8) for name in
                   ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'sensordata', 'cvel')}
    env._arrays['qvel'][0, 7] = -3268.07  # Finite hand-joint explosion observed on G1.
    env._arrays['qacc'][1, 3] = float('nan')
    env._arrays['qacc_warmstart'][2, 3] = float('inf')
    env._arrays['sensordata'][3, 1] = float('nan')
    env._arrays['qvel'][4] = 40.0
    assert env._physics_failure_mask().tolist() == [True, True, True, True, False]


@pytest.mark.skipif(os.environ.get('GOBOT_RUN_LOCOMOTION_NUMERICS_GPU_TEST') != '1', reason='requires CUDA')
@pytest.mark.parametrize('field,value', [('qpos', float('nan')), ('qvel', -3268.07),
                                        ('qacc_warmstart', float('inf')), ('sensordata', float('nan'))])
def test_bad_world_resets_without_poisoning_ppo(monkeypatch, field, value):
    import gobot
    from gobot.rl import RslRlVecEnvWrapper
    from gobot.rl.locomotion.experiment import Experiment, task_config
    from gobot.rl.locomotion.distributed import DistributedTraining
    from gobot.rl.locomotion.training import agent_config, build_algorithm, update_policy

    exp = Experiment(robot='g1', stage='rough', num_envs=4)
    cfg = task_config(exp)
    cfg.project_path = Path(__file__).resolve().parents[2] / 'examples/g1'
    cfg.capture_transitions = True
    cfg.domain_randomization.enabled = False
    cfg.observations.actor_noise = False
    cfg.push_enabled = False
    cfg.max_init_terrain_level = 0
    env = WarpVelocityEnv(cfg, num_envs=4, seed=49, context=gobot.app.create_context())
    wrapper = RslRlVecEnvWrapper(env, device='cuda:0')
    try:
        original_step = env.provider.step
        original_reset = env._reset_idx
        reset_ids = []
        substeps = 0
        levels = env.terrain_assignments[0].clone()

        def inject_failure(*args, **kwargs):
            nonlocal substeps
            result = original_step(*args, **kwargs)
            substeps += 1
            if substeps == cfg.decimation:
                env._arrays[field][1, -1] = value
            return result

        def capture_reset(ids, reasons):
            reset_ids.extend(ids.cpu().tolist())
            return original_reset(ids, reasons)

        monkeypatch.setattr(env.provider, 'step', inject_failure)
        monkeypatch.setattr(env, '_reset_idx', capture_reset)
        config = agent_config(exp)
        config['actor']['hidden_dims'] = config['critic']['hidden_dims'] = [16, 8]
        config['algorithm'].update(num_learning_epochs=1, num_mini_batches=1)
        config['num_steps_per_env'] = 4
        alg = build_algorithm(wrapper, config, 'cuda:0', DistributedTraining('cuda:0'))
        recorded = []

        def record(action, reward, done, extras):
            recorded.append((reward.clone(), done.clone(), extras))
            assert torch.isfinite(reward).all()
            for values in extras['transition_state'].values():
                assert torch.isfinite(values).all()
            for values in extras['final_observation'].values():
                assert torch.isfinite(values).all()

        obs, losses = update_policy(alg, wrapper, wrapper.get_observations(), 4, record)
        reward, done, info = recorded[0]
        assert info['numerical_reset'].tolist() == [False, True, False, False]
        assert done[1] and reward[1] == 0
        assert not info['time_outs'][1]
        assert reset_ids.count(1) == 1
        assert env.episode_length_buf[1] == 3
        assert env.terrain_assignments[0][1] == levels[1]
        diagnostic = info['numerical_failure']
        assert diagnostic['env_ids'].tolist() == [1]
        captured = diagnostic['state'][field][0, -1]
        assert captured == value if torch.isfinite(torch.tensor(value)) else not torch.isfinite(captured)
        assert all(not entry[2]['numerical_reset'].any() for entry in recorded[1:])
        assert all(torch.isfinite(values).all() for values in obs.values())
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses.values())
        assert all(torch.isfinite(p).all() for p in alg.actor.parameters())
        assert all(torch.isfinite(p).all() for p in alg.critic.parameters())
        env.provider.assert_finite()
    finally:
        wrapper.close()
