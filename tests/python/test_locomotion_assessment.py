"""Evaluation counts first-episode outcomes and checks spawn collisions."""
from pathlib import Path
from types import SimpleNamespace as NS
import os
import tempfile

import numpy as np
import pytest
import torch

from examples.locomotion.assessment import evaluation_scene, fixed_commands, rollout
from gobot.rl.locomotion import make_velocity_env
from examples.locomotion.presets import Experiment, task_config


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
            return {'base_position': np.tile([-3.25, 0, .76], (5, 1)),
                    'base_quaternion': np.tile([1., 0., 0., 0.], (5, 1))}

        def step(self, action):
            self.steps += 1
            position = np.array([[3.26, 0, .76], [0, 0, .1], [0, 3.6, .76], [0, 0, .76], [-3.25, 0, .76]])
            state = {'base_position': position, 'base_quaternion': np.tile([1.,0.,0.,0.], (5,1)), 'base_linear_velocity_body': np.tile([.8, 0, 0], (5, 1)),
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
                assert bool(((clearance > .7) & (clearance < 1.35)).all())
                # Base height alone misses toes intersecting an uphill surface.
                import mujoco
                model = mujoco.MjModel.from_xml_string(env.compiled_artifact.content)
                data = mujoco.MjData(model)
                prefix = env.compiled_artifact.robot_prefix(cfg.robot_name)
                joint_qpos = [model.jnt_qposadr[model.joint(prefix + name).id] for name in env.joint_names]
                def numpy(value):
                    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
                for position, quaternion, joints in zip(numpy(state['base_position']),
                        numpy(state['base_quaternion']), numpy(state['joint_position']), strict=True):
                    data.qpos[:] = model.qpos0
                    data.qpos[:3], data.qpos[3:7], data.qpos[joint_qpos] = position, quaternion, joints
                    mujoco.mj_fwdPosition(model, data)
                    ground = [c.dist for c in data.contact if 0 in model.geom_bodyid[c.geom]]
                    assert min(ground, default=0.) >= -1.e-5
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


def test_signed_turn_bias_and_unwrapped_heading_stop_at_first_episode_end():
    class Env:
        num_envs = 2
        step_dt = 1.
        cfg_obj = NS(robot=NS(feet=('left', 'right')))
        terrain_grid_shape = (1, 1)
        terrain_type_names = ('flat',)
        terrain_origins = np.zeros((1, 3))
        terrain_assignments = (np.zeros(2, int), np.zeros(2, int))

        @staticmethod
        def quaternion(degrees):
            half = np.deg2rad(degrees) / 2
            return np.stack((np.cos(half), np.zeros(2), np.zeros(2), np.sin(half)), 1)

        def reset(self, seed=None):
            self.steps = 0

        def set_command(self, command):
            pass

        def get_observations(self):
            return {'actor': np.zeros((2, 1))}

        def locomotion_state(self):
            return {'base_position': np.tile([-3.25, 0, .76], (2, 1)),
                    'base_quaternion': self.quaternion([179., -179.])}

        def step(self, action):
            self.steps += 1
            # World 0 ends after two steps, then autoresets to a different heading.
            angle = [-179., 179.] if self.steps == 1 else [-177., 177.] if self.steps == 2 else [30., 175.]
            state = {'base_position': np.tile([-3.25, 0, .76], (2, 1)),
                     'base_quaternion': self.quaternion(angle),
                     'base_linear_velocity_body': np.tile([.8,0,0], (2,1)),
                     'base_angular_velocity_body': np.array([[0,0,.1], [0,0,-.1]]),
                     'foot_force': np.zeros((2,2,3)), 'foot_position': np.zeros((2,2,3)),
                     'foot_velocity': np.zeros((2,2,3)), 'base_clearance': np.ones(2)*.76}
            ended = self.steps == 2
            return NS(terminated=np.array([ended,False]), truncated=np.zeros(2,bool),
                      info={'transition_state':state,'reset_reason':np.array([1 if ended else 0,0])})

    report = rollout(Env(), lambda obs: np.zeros((2,1)), command=(.8,0,0), steps=3)
    np.testing.assert_allclose(report['episode_metrics']['heading_change_deg'], [4.,-6.], atol=1e-6)
    np.testing.assert_allclose(report['episode_metrics']['heading_change_max_deg'], [4.,6.], atol=1e-6)
    assert report['heading_drift_deg'] == pytest.approx(-1.)
    assert report['heading_drift_abs_deg'] == pytest.approx(5.)
    assert report['heading_drift_max_deg'] == pytest.approx(6.)
    np.testing.assert_allclose(report['episode_metrics']['yaw_rate_bias_rad_s'], [.1,-.1])
    assert report['yaw_rate_bias_rad_s'] == pytest.approx(0.)
    assert report['yaw_error'] == pytest.approx(.1)
    # Deliberate turning is not classified as straight-line heading drift.
    turning = rollout(Env(), lambda obs: np.zeros((2,1)), command=(.8,0,.2), steps=3)
    assert turning['heading_drift_deg'] is None
    assert turning['heading_drift_max_deg'] is None


def test_fixed_turn_commands_are_passed_without_rewriting(monkeypatch, capsys):
    from contextlib import nullcontext
    from examples.locomotion import assessment
    calls=[]
    monkeypatch.setattr(assessment,'evaluation_budget',lambda *a,**k:nullcontext())
    monkeypatch.setattr(assessment,'evaluate',lambda *a,**kw:(calls.append(kw) or {'scenarios':[]}))
    monkeypatch.setattr('sys.argv',['assessment','--checkpoint','unused.pt','--output','unused.json',
        '--commands','.8','0','.5','--commands','.8','0','-.5','--commands','.8','0','0'])
    assessment.main()
    assert calls[0]['commands']==[[.8,0.,.5],[.8,0.,-.5],[.8,0.,0.]]
    capsys.readouterr()
