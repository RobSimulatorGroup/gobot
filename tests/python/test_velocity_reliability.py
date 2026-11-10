"""State and checkpoint contracts exercised with real velocity environments."""
from copy import deepcopy
import os

import numpy as np
import pytest
import torch

from gobot.rl.locomotion import make_velocity_env
from gobot.rl.locomotion.command import VelocityCommandStage
from examples.locomotion.presets import Experiment, task_config

BACKENDS = ['mujoco-cpu', pytest.param('mujoco-warp', marks=pytest.mark.skipif(
    os.environ.get('GOBOT_RUN_LOCOMOTION_NUMERICS_GPU_TEST') != '1', reason='requires CUDA'))]


def array(value):
    return value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor) else np.asarray(value).copy()


def config(stage='rough'):
    cfg = task_config(Experiment(robot='g1', stage=stage))
    cfg.domain_randomization.enabled = cfg.push_enabled = False
    cfg.command.source = 'external'
    cfg.spawn_jitter = 0.
    cfg.randomize_reset_yaw = False
    return cfg


@pytest.mark.parametrize('backend', BACKENDS)
def test_resume_preserves_curriculum_and_replays_fresh_episodes(backend):
    cfg = config()
    cfg.run_command_curriculum = (
        VelocityCommandStage(step=0, run_velocity_x=(1., 2.)),
        VelocityCommandStage(step=100, run_velocity_x=(2., 3.)),
    )
    env = make_velocity_env(cfg, backend=backend, num_envs=2, seed=11, collect_step_extras=False)
    try:
        env.set_command((.5, 0., 0.))
        env.profile_step_counter = 123
        env.set_training_progress(150000)
        env.set_terrain_assignments([3, 4], [0, 0], fixed=False)
        saved = deepcopy(env.training_state_dict())
        env.profile_step_counter = 0
        env.set_training_progress(0)
        assert cfg.command.run_velocity_x == (1., 2.)
        result = env.load_training_state_dict(saved)
        assert env.profile_step_counter == 123
        assert cfg.command.run_velocity_x == (2., 3.)
        np.testing.assert_array_equal(array(env.terrain_assignments[0]), [3, 4])
        assert result['terrain_curriculum'] == 'exact'
        assert env.common_step_counter == 150000
        np.testing.assert_array_equal(array(env.episode_length_buf), 0)
        observation = array(env.get_observations()['actor'])
        env.step(np.zeros((2, env.num_actions), dtype=np.float32))
        env.load_training_state_dict(saved)
        np.testing.assert_array_equal(array(env.terrain_assignments[0]), [3, 4])
        np.testing.assert_allclose(array(env.get_observations()['actor']), observation, atol=1e-6)
        for version in (0, 1, 2):
            with pytest.raises(RuntimeError, match='checkpoint version'):
                env.load_training_state_dict({**saved, 'version': version})
    finally:
        env.close()


@pytest.mark.parametrize('backend', BACKENDS)
def test_timeout_diagnostics_keep_transition_and_reset_clears_targets(backend):
    cfg = config('flat')
    cfg.capture_transitions = True
    cfg.observations.actor_noise = False
    env = make_velocity_env(cfg, backend=backend, num_envs=2, seed=11,
                            max_episode_length=2, collect_step_extras=False)
    try:
        env.set_command((.5, 0., 0.))
        for _ in range(2):
            result = env.step(np.full((2, env.num_actions), .1, dtype=np.float32))
        assert array(result.truncated).all() and not array(result.terminated).any()
        final = result.info['final_observation']
        physical = result.info['transition_state']
        expected = array(physical['base_linear_velocity_body']) + np.cross(
            array(physical['base_angular_velocity_body']), [.04525, 0., -.08339])
        np.testing.assert_allclose(array(final['actor'])[:, :3], expected, atol=2e-5)
        np.testing.assert_allclose(array(final['actor'])[:, 9:38],
                                   array(physical['joint_position']) - env.default_joint_pos, atol=1e-6)
        np.testing.assert_allclose(array(result.obs['actor'])[:, :6], 0., atol=1e-6)
        assert not np.allclose(array(final['critic']), array(result.obs['critic']))
        if backend == 'mujoco-warp':
            targets = array(env.provider.arrays['ctrl'])[:, env._layout.actuator_ids]
            np.testing.assert_allclose(targets, np.zeros_like(targets))
    finally:
        env.close()


@pytest.mark.skipif(os.environ.get('GOBOT_RUN_LOCOMOTION_NUMERICS_GPU_TEST') != '1', reason='requires CUDA')
def test_environment_reset_shares_policy_rng_like_mjlab():
    env = make_velocity_env(config('flat'), backend='mujoco-warp', num_envs=2,
                            seed=11, collect_step_extras=False)
    try:
        torch.cuda.manual_seed(791)
        before = torch.cuda.get_rng_state().clone()
        env.reset(seed=53)
        assert not torch.equal(torch.cuda.get_rng_state(), before)
        after = torch.cuda.get_rng_state().clone()
        env.reset(seed=53)
        assert torch.equal(torch.cuda.get_rng_state(), after)
    finally:
        env.close()


@pytest.mark.parametrize('backend', BACKENDS)
def test_partial_reset_preserves_unselected_world_observations_and_state(backend):
    cfg = config('flat')
    cfg.capture_transitions = True
    assert cfg.observations.actor_noise
    env = make_velocity_env(cfg, backend=backend, num_envs=2, seed=41, collect_step_extras=False)
    try:
        env.set_command((.5, 0., 0.))
        env.step(np.full((2, env.num_actions), .1, dtype=np.float32))
        observations = {key: array(value)[1] for key, value in env.get_observations().items()}
        state = {key: array(value)[1] for key, value in env.locomotion_state().items()}
        steps = array(env.episode_length_buf)
        returned, _ = env.reset([0])
        assert returned['actor'].shape[0] == 1
        assert array(env.episode_length_buf).tolist() == [0, int(steps[1])]
        for key, expected in observations.items():
            np.testing.assert_array_equal(array(env.get_observations()[key])[1], expected, err_msg=key)
        for key, expected in state.items():
            np.testing.assert_allclose(array(env.locomotion_state()[key])[1], expected, atol=1e-6, err_msg=key)
    finally:
        env.close()


@pytest.mark.parametrize('robot', ['go1', 'g1'])
def test_real_timeout_uses_unmodified_rsl_ppo_bootstrap(robot):
    from gobot.rl import RslRlVecEnvWrapper
    from gobot.rl.locomotion.distributed import DistributedTraining
    from gobot.rl.locomotion.training import agent_config, build_algorithm

    experiment = Experiment(robot=robot, stage='flat', num_envs=2)
    cfg = task_config(experiment)
    cfg.domain_randomization.enabled = cfg.observations.actor_noise = cfg.push_enabled = False
    cfg.command.source = 'external'
    cfg.spawn_jitter = 0.
    cfg.randomize_reset_yaw = False
    env = make_velocity_env(cfg, backend='mujoco-cpu', num_envs=2, seed=11,
                            max_episode_length=1, collect_step_extras=False)
    wrapper = RslRlVecEnvWrapper(env, device='cpu')
    try:
        env.set_command((.5, 0., 0.))
        config = agent_config(experiment)
        config['actor']['hidden_dims'] = config['critic']['hidden_dims'] = [16, 8]
        config['num_steps_per_env'] = 1
        torch.manual_seed(31)
        algorithm = build_algorithm(wrapper, config, 'cpu', DistributedTraining('cpu'))
        with torch.inference_mode():
            action = algorithm.act(wrapper.get_observations())
            previous_value = algorithm.transition.values.squeeze(-1).clone()
            observation, reward, done, extras = wrapper.step(action)
            assert done.all() and extras['time_outs'].all()
            reset_value = algorithm.critic(observation).squeeze(-1)
            expected = reward + algorithm.gamma * previous_value
            algorithm.process_env_step(observation, reward, done, extras)
        torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], expected)
        assert not torch.allclose(previous_value, reset_value)
    finally:
        wrapper.close()


@pytest.mark.skipif(os.environ.get('GOBOT_RUN_LOCOMOTION_NUMERICS_GPU_TEST') != '1', reason='requires CUDA')
def test_cpu_warp_agree_through_landing_with_current_contact_observations():
    traces = []
    for backend in ('mujoco-cpu', 'mujoco-warp'):
        cfg = config('flat')
        cfg.capture_transitions = True
        cfg.observations.actor_noise = False
        cfg.reset_z_range = (.03, .03)
        env = make_velocity_env(cfg, backend=backend, num_envs=2, seed=19, collect_step_extras=False)
        try:
            env.set_terrain_assignments([0, 0], [0, 0], fixed=True)
            env.reset(seed=19)
            env.set_command((.5, 0., 0.))
            trace = []
            for step in range(32):
                action = np.zeros((2, env.num_actions), dtype=np.float32)
                action[:, 0] = .1 * np.sin(step * .2)
                result = env.step(action)
                assert not array(result.terminated).any() and not array(result.truncated).any()
                state = result.info['transition_state']
                trace.append({**{key: array(state[key]) for key in (
                    'base_position', 'joint_position', 'joint_velocity', 'foot_contact', 'foot_force')},
                    'actor': array(result.info['final_observation']['actor']),
                    'critic': array(result.info['final_observation']['critic']),
                    'reward': array(result.reward)})
            traces.append(trace)
        finally:
            env.close()
    contact = np.stack([frame['foot_contact'] for frame in traces[0]])
    assert (contact == 0).any() and (contact > 0).any(), 'the regression must include the landing boundary'
    for cpu, gpu in zip(*traces, strict=True):
        np.testing.assert_array_equal(cpu['foot_contact'], gpu['foot_contact'])
        for key in ('base_position', 'joint_position', 'joint_velocity', 'actor'):
            np.testing.assert_allclose(cpu[key], gpu[key], atol=2e-3, rtol=5e-3, err_msg=key)
        # Contact impulses differ slightly between native and Warp solvers.
        np.testing.assert_allclose(cpu['foot_force'], gpu['foot_force'], atol=.03, rtol=5e-3)
        np.testing.assert_allclose(cpu['critic'], gpu['critic'], atol=.02, rtol=5e-3)
        np.testing.assert_allclose(cpu['reward'], gpu['reward'], atol=1e-4, rtol=5e-3)
