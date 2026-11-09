"""A failed physics rollout must stop before reaching the optimizer."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gobot.rl.locomotion.warp_env import WarpVelocityEnv


def test_physics_failures_include_finite_velocity_explosions():
    env = WarpVelocityEnv.__new__(WarpVelocityEnv)
    env.num_envs = 5
    env._torch_device = torch.device('cpu')
    env._arrays = {name: torch.zeros(5, 8) for name in
                   ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'sensordata', 'cvel')}
    env._arrays['qvel'][0, 7] = -3268.07
    env._arrays['qacc'][1, 3] = float('nan')
    env._arrays['qacc_warmstart'][2, 3] = float('inf')
    env._arrays['sensordata'][3, 1] = float('nan')
    env._arrays['qvel'][4] = 40.0
    assert env._physics_failure_mask().tolist() == [True, True, True, True, False]


def test_failed_rollout_never_calls_optimizer():
    from gobot.rl.locomotion.training import update_policy
    from gobot.rl.locomotion.distributed import DistributedTraining

    def fail(action):
        raise FloatingPointError('invalid locomotion physics')

    def forbidden(*args):
        pytest.fail('invalid rollout reached PPO')

    algorithm = SimpleNamespace(act=lambda obs: torch.zeros(2, 1),
                                update=forbidden, compute_returns=forbidden,
                                process_env_step=forbidden)
    with pytest.raises(FloatingPointError, match='invalid locomotion physics'):
        update_policy(algorithm, SimpleNamespace(step=fail), {}, 4,
                      distributed=DistributedTraining('cpu'))


@pytest.mark.skipif(os.environ.get('GOBOT_RUN_LOCOMOTION_NUMERICS_GPU_TEST') != '1', reason='requires CUDA')
@pytest.mark.parametrize('field,value', [('qpos', float('nan')), ('qvel', -3268.07),
                                        ('qacc_warmstart', float('inf')), ('sensordata', float('nan'))])
def test_bad_world_aborts_rollout_and_preserves_diagnostic(monkeypatch, field, value):
    import gobot
    from examples.locomotion.presets import Experiment, task_config

    cfg = task_config(Experiment(robot='g1', stage='flat', num_envs=2))
    cfg.project_path = Path(__file__).resolve().parents[2] / 'examples/g1'
    cfg.domain_randomization.enabled = cfg.observations.actor_noise = cfg.push_enabled = False
    env = WarpVelocityEnv(cfg, num_envs=2, seed=49, context=gobot.app.create_context())
    try:
        original_step = env.provider.step
        substeps = 0
        levels = env.terrain_assignments[0].clone()

        def inject_failure(*args, **kwargs):
            nonlocal substeps
            result = original_step(*args, **kwargs)
            substeps += 1
            if substeps == cfg.decimation:
                env._arrays[field][1, -1] = value
            return result

        def forbidden(*args):
            pytest.fail('numerical failure was silently reset or converted to reward')

        monkeypatch.setattr(env.provider, 'step', inject_failure)
        monkeypatch.setattr(env, '_reset_idx', forbidden)
        monkeypatch.setattr(env, '_compute_rewards', forbidden)
        with pytest.raises(FloatingPointError, match=r'envs \[1\]'):
            env.step(torch.zeros(2, env.num_actions, device=env.device))
        assert torch.equal(env.terrain_assignments[0], levels)
        diagnostic = env.numerical_failure
        assert diagnostic['env_ids'].tolist() == [1]
        captured = diagnostic['state'][field][0, -1]
        assert captured == value if torch.isfinite(torch.tensor(value)) else not torch.isfinite(captured)
        assert env.episode_length_buf.tolist() == [1, 1]
    finally:
        env.close()


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 3268.07])
def test_cpu_bad_state_aborts_before_reward_or_reset(monkeypatch, value):
    from gobot.rl.locomotion import make_velocity_env
    from examples.locomotion.presets import Experiment, task_config

    cfg = task_config(Experiment(robot='go1', stage='flat', num_envs=2))
    cfg.domain_randomization.enabled = cfg.observations.actor_noise = cfg.push_enabled = False
    env = make_velocity_env(cfg, backend='mujoco-cpu', num_envs=2, seed=49,
                            collect_step_extras=False)
    try:
        step = env.backend.step_task_inputs

        def inject_failure(*args, **kwargs):
            result = step(*args, **kwargs)
            velocity = env.backend.state.base_linear_velocity.copy()
            velocity[1, 0] = value
            monkeypatch.setitem(env.backend.state._arrays, 'base_linear_velocity', velocity)
            return result

        def forbidden(*args, **kwargs):
            pytest.fail('invalid CPU physics reached reward computation or reset')

        monkeypatch.setattr(env.backend, 'step_task_inputs', inject_failure)
        monkeypatch.setattr(env, '_run_task_runtime', forbidden)
        monkeypatch.setattr(env, '_reset_envs', forbidden)
        with pytest.raises(FloatingPointError, match='physics'):
            env.step(torch.zeros(2, env.num_actions))
    finally:
        env.close()
