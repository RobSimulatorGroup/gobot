"""Evaluation counts first-episode outcomes and uses traversable spawn platforms."""
from pathlib import Path
from types import SimpleNamespace as NS
import os
import tempfile

import numpy as np
import pytest
import torch

from examples.locomotion.assessment import evaluation_scene, fixed_commands, rollout
from gobot.rl.locomotion import make_velocity_env
from gobot.rl.locomotion.experiment import Experiment, task_config


def test_route_outcomes_distinguish_falls_departures_and_numerical_failures():
    class Env:
        num_envs = 5
        step_dt = .02
        cfg_obj = NS(robot=NS(feet=('left', 'right')))
        terrain_grid_shape = (1, 1)
        terrain_type_names = ('flat',)
        terrain_origins = np.zeros((1, 3))
        terrain_assignments = (np.zeros(5, int), np.zeros(5, int))

        def reset(self, seed=None):
            self.steps = 0

        def set_command(self, command):
            pass

        def get_observations(self):
            return {'actor': np.zeros((5, 1))}

        def locomotion_state(self):
            return {'base_position': np.tile([-3.25, 0, .76], (5, 1))}

        def step(self, action):
            self.steps += 1
            position = np.array([[3.26, 0, .76], [0, 0, .1], [0, 3.6, .76], [0, 0, .76], [-3.25, 0, .76]])
            state = {'base_position': position, 'base_linear_velocity_body': np.tile([.8, 0, 0], (5, 1)),
                     'base_angular_velocity_body': np.zeros((5, 3)), 'foot_force': np.zeros((5, 2, 3)),
                     'foot_position': np.zeros((5, 2, 3)), 'foot_velocity': np.zeros((5, 2, 3)),
                     'base_clearance': np.full(5, .76)}
            return NS(terminated=np.array([False, True, False, True, False]), truncated=np.zeros(5, bool),
                      info={'transition_state': state, 'reset_reason': np.array([0, 1, 0, 4, 0])})

    result = rollout(Env(), lambda obs: np.zeros((5, 1)), command=(.8, 0, 0), steps=4, route=True)
    assert result['terrain_pass_rate'] == .2
    assert result['fall_rate'] == .2
    assert result['off_route_rate'] == .2
    assert result['numerical_failure_rate'] == .2
    assert result['tracking_episodes'] == 0
    assert result['velocity_error_after_warmup'] is None
    assert result['episode_metrics']['reached'] == [True, False, False, False, False]
    # Reset episodes must never add successes, failures or velocity samples.
    assert result['episode_metrics']['duration_s'] == [.02, .02, .02, .02, .08]


@pytest.mark.parametrize('backend', ['mujoco-cpu', 'mujoco-warp'])
def test_all_terrain_cells_have_valid_evaluation_spawns(backend):
    if backend == 'mujoco-warp' and os.environ.get('GOBOT_RUN_WARP_IPC_GPU_TEST') != '1':
        pytest.skip('set GOBOT_RUN_WARP_IPC_GPU_TEST=1 for CUDA spawn regression')
    cfg = task_config(Experiment(robot='g1'), play=True)
    fixed_commands(cfg)
    folder = Path(cfg.project_path)/'assets/evaluation'
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder, prefix='test-spawn-') as directory:
        cfg.scene_path = evaluation_scene(cfg, directory, 71)
        env = make_velocity_env(cfg, num_envs=70, backend=backend,
                                device='cuda:0' if backend.endswith('warp') else 'cpu')
        try:
            env.set_terrain_assignments(np.arange(70)//7, np.arange(70)%7, fixed=True)
            for offset in ((-3.25, 0), (3.25, 0), (0, -3.25), (0, 3.25)):
                cfg.spawn_offset_xy = offset
                observations, _ = env.reset(seed=71)
                assert bool(torch.isfinite(torch.as_tensor(observations['actor'])).all())
                state = env.locomotion_state()
                clearance = torch.tensor(np.array(state['base_clearance'])) if backend == 'mujoco-cpu' else state['base_clearance']
                assert bool(((clearance > .7) & (clearance < .85)).all())
        finally:
            env.close()


def test_model_transfer_is_explicit_and_keeps_control_checks():
    from dataclasses import replace
    from gobot.rl.locomotion.manifest import build_policy_manifest, validate_policy
    from gobot.rl.locomotion.training import agent_config
    from gobot.rl.rsl_rl import RslRlVecEnvWrapper
    experiment = Experiment(robot='g1', stage='flat')
    cfg = task_config(experiment, play=True)
    env = make_velocity_env(cfg, num_envs=1, backend='mujoco-cpu', device='cpu')
    try:
        wrapper = RslRlVecEnvWrapper(env)
        manifest = build_policy_manifest(wrapper, cfg, agent_config(experiment), Path(cfg.project_path))
        changed = replace(manifest, scene_digest='different_robot')
        with pytest.raises(RuntimeError, match='scene digest mismatch'):
            validate_policy(changed, env, cfg)
        validate_policy(changed, env, cfg, allow_model_change=True)
        wrong_control = replace(changed, control={**changed.control, 'kp': [1.]*29})
        with pytest.raises(ValueError, match='kp'):
            validate_policy(wrong_control, env, cfg, allow_model_change=True)
    finally:
        env.close()
