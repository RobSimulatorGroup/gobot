"""Match mjlab velocity numerical handling and RSL-RL's output checks."""
from types import SimpleNamespace
import os

import numpy as np
import pytest
import torch

from gobot.rl.locomotion import make_velocity_env
from examples.locomotion.presets import Experiment, task_config


BACKENDS = ['mujoco-cpu', pytest.param('mujoco-warp', marks=pytest.mark.skipif(
    os.environ.get('GOBOT_RUN_LOCOMOTION_NUMERICS_GPU_TEST') != '1', reason='requires CUDA'))]


def _task():
    cfg = task_config(Experiment(robot='go1', stage='flat', num_envs=2))
    cfg.domain_randomization.enabled = cfg.observations.actor_noise = cfg.push_enabled = False
    return cfg


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('value', [-3268.07, float('nan'), float('inf')])
def test_physics_outputs_are_not_thresholded_or_numerically_reset(monkeypatch, backend, value):
    env = make_velocity_env(_task(), backend=backend, num_envs=2, seed=49,
                            collect_step_extras=False)
    try:
        def forbidden(*args, **kwargs):
            pytest.fail('numerical state caused an extra episode reset')

        if backend == 'mujoco-warp':
            original = env.provider.forward

            def inject(*args, **kwargs):
                result = original(*args, **kwargs)
                env._arrays['qvel'][1, env._layout.joint_dof_addresses[-1]] = value
                return result

            monkeypatch.setattr(env.provider, 'forward', inject)
            monkeypatch.setattr(env, '_reset_idx', forbidden)
        else:
            original = env.backend.refresh

            def inject(*args, **kwargs):
                result = original(*args, **kwargs)
                velocity = env.backend.state.joint_velocity.copy()
                velocity[1, -1] = value
                monkeypatch.setitem(env.backend.state._arrays, 'joint_velocity', velocity)
                return result

            monkeypatch.setattr(env.backend, 'refresh', inject)
            monkeypatch.setattr(env, '_reset_envs', forbidden)

        result = env.step(torch.zeros((2, env.num_actions), device=env.device))
        actor = torch.as_tensor(result.obs['actor'])
        observed = actor[1, 9 + 2 * env.num_actions - 1]
        if np.isnan(value):
            assert torch.isnan(observed)
        else:
            assert float(observed) == pytest.approx(value)
        assert not np.asarray(torch.as_tensor(result.terminated).cpu()).any()
        assert not np.asarray(torch.as_tensor(result.truncated).cpu()).any()
        assert torch.isfinite(torch.as_tensor(result.reward)).all()
        assert 'numerical_reset' not in result.info
    finally:
        env.close()


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
@pytest.mark.parametrize('output', ['actor', 'critic', 'reward'])
def test_rsl_rl_still_rejects_nonfinite_outputs_before_ppo(value, output):
    from gobot.rl.locomotion.training import update_policy
    from gobot.rl.locomotion.distributed import DistributedTraining

    observation = {'actor': torch.zeros(2, 1), 'critic': torch.zeros(2, 1)}
    reward = torch.ones(2)
    if output == 'reward':
        reward[1] = value
    else:
        observation[output][1, 0] = value

    def forbidden(*args):
        pytest.fail('nonfinite environment output reached PPO storage or optimizer')

    algorithm = SimpleNamespace(act=lambda obs: torch.zeros(2, 1),
                                update=forbidden, compute_returns=forbidden,
                                process_env_step=forbidden)
    environment = SimpleNamespace(step=lambda action: (
        observation, reward, torch.zeros(2, dtype=torch.bool), {}))
    with pytest.raises(ValueError, match='NaN|Inf'):
        update_policy(algorithm, environment, {}, 1, distributed=DistributedTraining('cpu'))


def test_environment_exceptions_still_stop_distributed_rollout():
    from gobot.rl.locomotion.training import update_policy
    from gobot.rl.locomotion.distributed import DistributedTraining

    def fail(action):
        raise RuntimeError('simulation failed')

    def forbidden(*args):
        pytest.fail('failed rollout reached PPO')

    algorithm = SimpleNamespace(act=lambda obs: torch.zeros(2, 1),
                                update=forbidden, compute_returns=forbidden,
                                process_env_step=forbidden)
    with pytest.raises(RuntimeError, match='simulation failed'):
        update_policy(algorithm, SimpleNamespace(step=fail), {}, 4,
                      distributed=DistributedTraining('cpu'))
