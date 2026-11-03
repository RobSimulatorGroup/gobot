"""Regression coverage for motion priors, history and pre-reset transition semantics."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from gobot.rl.locomotion.motion import motion_features, MotionDataset, MotionPriorCfg
from examples.g1.prepare_motion import resample
from examples.g1.config import g1_amp_cfg
from gobot.rl.locomotion import make_velocity_env


def test_resampling_quaternion_and_velocity():
    raw = np.zeros((31, 36)); raw[:, 0] = np.arange(31) / 30
    raw[:, 6] = np.where(np.arange(31) % 2, -1., 1.)
    p, r, q, v, w, qd = resample(raw)
    assert p.shape == (51, 3)
    np.testing.assert_allclose(v[:, 0], 1., atol=1e-6)
    np.testing.assert_allclose(r.as_matrix(), np.broadcast_to(np.eye(3), (51, 3, 3)), atol=1e-6)
    np.testing.assert_allclose(w, 0., atol=1e-6)
    raw[:, 3:7] = 0
    try: resample(raw)
    except ValueError: pass
    else: raise AssertionError('zero quaternion accepted')


def test_motion_features_heading_invariance():
    rng = np.random.default_rng(8)
    pos = rng.normal(size=(4, 3)).astype(np.float32)
    links = pos[:, None] + rng.normal(size=(4, 4, 3))
    quat = np.tile([1., 0., 0., 0.], (4, 1))
    state = dict(base_position=pos, base_quaternion=quat, link_position=links,
                 joint_position=rng.normal(size=(4, 29)), joint_velocity=rng.normal(size=(4, 29)),
                 base_linear_velocity_body=rng.normal(size=(4, 3)), base_angular_velocity_body=rng.normal(size=(4, 3)))
    expected = motion_features(state)
    rotation = Rotation.from_euler('z', 1.4)
    state['base_position'] = rotation.apply(pos) + [10., -7., 2.]
    state['link_position'] = rotation.apply(links.reshape(-1, 3)).reshape(4, 4, 3) + [10., -7., 2.]
    state['base_quaternion'] = np.tile(rotation.as_quat()[[3, 0, 1, 2]], (4, 1))
    torch.testing.assert_close(motion_features(state), expected, atol=2e-6, rtol=2e-6)


def test_dataset_split_and_rng():
    import tempfile, json
    with tempfile.TemporaryDirectory(prefix='gobot-motion-test-') as directory:
        path = Path(directory) / 'motion.npz'
        clips, arrays = [], {}
        for i in range(4):
            clips.append({'name': f'clip{i}', 'split': 'train' if i < 2 else 'holdout', 'gait': i % 2})
            arrays[f'features_{i}'] = np.broadcast_to(np.arange(4)[:, None] + i * 10., (4, 79)).astype(np.float32)
            arrays[f'valid_{i}'] = np.array([True, True, False, True])
        arrays['metadata'] = np.array(json.dumps({'schema': 'gobot_motion_v1', 'dt': .02, 'clips': clips}))
        np.savez(path, **arrays)
        a = MotionDataset(path, seed=12)
        b = MotionDataset(path, seed=12)
        holdout = MotionDataset(path, split='holdout')
        assert len(a.pairs) == len(holdout.pairs) == 2
        assert a.pairs.shape[1] == 158
        first, labels = a.sample(128)
        second, other_labels = b.sample(128)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(labels, other_labels)
        # Neither invalid-frame gaps nor clip boundaries may produce transitions.
        torch.testing.assert_close(first[:, 79:] - first[:, :79], torch.ones(128, 79))
        assert set(first[:, 0].tolist()) == {0., 10.}
        assert set(holdout.pairs[:, 0].tolist()) == {20., 30.}
        state = a.generator.get_state()
        expected = a.sample(64)
        a.generator.set_state(state)
        actual = a.sample(64)
        torch.testing.assert_close(expected[0], actual[0])


def test_timeout_uses_final_value():
    from types import SimpleNamespace
    from gobot.rl.locomotion.amp import MotionPriorPPO
    from rsl_rl.storage import RolloutStorage
    from tensordict import TensorDict
    class Model:
        def __call__(self, obs): return obs['critic'][:, :1]
        def update_normalization(self, obs): pass
        def reset(self, done): pass
    alg = MotionPriorPPO.__new__(MotionPriorPPO)
    alg.actor = alg.critic = Model()
    alg.device = 'cpu'; alg.gamma = .99; alg.rnd = None
    alg.prior_cfg = MotionPriorCfg(dataset='', weight=0., fall_penalty=0.)
    alg.reference = SimpleNamespace(mean=torch.zeros(158), std=torch.ones(158), generator=torch.Generator().manual_seed(1))
    alg.discriminator = lambda pair: torch.zeros(len(pair), 2)
    alg._remember = lambda pair, gait: None
    alg.transition = RolloutStorage.Transition(); alg.transition.values = torch.tensor([[2.], [2.]])
    recorded = []
    alg.storage = SimpleNamespace(add_transition=lambda transition: recorded.append(transition.rewards.clone()))
    obs = TensorDict({'critic': torch.zeros(2, 1)}, batch_size=[2])
    extras = {'motion_pair': torch.zeros(2, 158), 'transition_state': {'command': torch.zeros(2, 3), 'terrain_level': torch.zeros(2)},
              'time_outs': torch.tensor([True, False]), 'final_observation': {'critic': torch.tensor([[5.], [7.]])}}
    alg.process_env_step(obs, torch.ones(2), torch.ones(2, dtype=torch.bool), extras)
    torch.testing.assert_close(recorded[0], torch.tensor([5.95, 1.]))


def integration(backend):
    cfg = g1_amp_cfg(stage='flat')
    cfg.observations.actor_noise = False
    cfg.randomize_reset_yaw = False
    cfg.spawn_jitter = 0.
    env = make_velocity_env(cfg, backend=backend, num_envs=2, seed=17, max_episode_length=2)
    try:
        assert env.num_obs == 1150 and env.num_actions == 29
        env.reset(seed=17)
        zero = torch.zeros((2, 29), device=env.device)
        first = env.step(zero)
        initial_pair = first.info['motion_pair'].clone()
        second = env.step(zero)
        assert bool(torch.as_tensor(second.truncated).all())
        torch.testing.assert_close(initial_pair[:, 79:], second.info['motion_pair'][:, :79], atol=1e-5, rtol=1e-5)
        # A timeout ends in a physically different state than the reset observation.
        assert not torch.allclose(second.info['final_observation']['critic'], second.obs['critic'])
        history = second.obs['actor'][:, :960].reshape(2, 10, 96)
        torch.testing.assert_close(history, history[:, :1].expand_as(history))
        env.reset(seed=17)
        repeated = env.step(zero).info['motion_pair']
        torch.testing.assert_close(repeated, initial_pair, atol=1e-5, rtol=1e-5)
        # Partial reset must retain the other environment's history.
        before = env.state.obs['actor'][1].clone()
        env.reset(np.array([0]))
        torch.testing.assert_close(env.state.obs['actor'][1], before)
    finally:
        env.close()


def test_checkpoint_normalization():
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    from gobot.rl.checkpoint_policy import CheckpointPolicy
    from gobot.rl.policy import PolicyManifest
    from gobot.rl.spec import ObservationSpec, ActionSpec, SpecField
    obs = TensorDict({'actor': torch.randn(4, 5)}, batch_size=[4])
    actor = MLPModel(obs, {'actor': ['actor']}, 'actor', 2, hidden_dims=[8],
                     activation='elu', obs_normalization=True,
                     distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 1., 'std_type': 'scalar'})
    actor.obs_normalizer._mean.copy_(torch.arange(5).float()[None])
    actor.obs_normalizer._std.copy_(torch.tensor([[0., .1, 1., 2., 0.]]))
    manifest = PolicyManifest(task_name='normalizer_test', task_version='v1',
        observation_spec=ObservationSpec('test', (SpecField('state', 5),)).metadata(),
        action_spec=ActionSpec('test', (SpecField('left', 1), SpecField('right', 1))).metadata(),
        joint_names=('left', 'right'), physics_dt=.005, decimation=4, control={},
        model={'activation': 'elu', 'obs_normalizer_eps': .01})
    actor.eval()
    restored = CheckpointPolicy(actor.state_dict(), manifest).eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(obs['actor']), actor(obs, stochastic_output=False), atol=1e-6, rtol=1e-6)


def test_seated_and_nonfoot_support_fail():
    from gobot.rl.locomotion.terms import locomotion_failure_mask
    import math
    values = [np.array([-1., -1., -1., 0.]), np.array([.76, .13, .75, .76]),
              np.array([0, 0, 0, 0]), np.array([0, 0, 1, 0])]
    kwargs = dict(fall_angle=math.radians(70), min_clearance=.45, terminate_on_trunk=True)
    expected = [False, True, True, True]
    np.testing.assert_array_equal(locomotion_failure_mask(*values, **kwargs), expected)
    np.testing.assert_array_equal(locomotion_failure_mask(*(torch.as_tensor(v) for v in values), **kwargs), expected)


def test_biped_gait_rejects_hopping_and_standing():
    from gobot.rl.locomotion.gait_metrics import BipedGaitMetrics
    metrics = BipedGaitMetrics(3, warmup_s=0.)
    positions = np.zeros((3, 2, 3))
    for step in range(1000):
        phase = step % 40
        contact = [[phase < 16, 20 <= phase < 36], [phase < 16, phase < 16], [True, True]]
        positions[:, 0, 0] = (step // 40) * .8
        positions[:, 1, 0] = ((step + 20) // 40) * .8
        metrics.update(contact, positions, [True] * 3)
    values = metrics.arrays()
    assert values['alternation_fraction'][0] > .99
    assert abs(values['flight_fraction'][0] - .2) < .01
    assert 2.4 < values['cadence_hz'][0] < 2.6
    assert values['alternation_fraction'][1] == 0
    assert values['cadence_hz'][2] == 0 and values['flight_fraction'][2] == 0
    chatter = BipedGaitMetrics(1, warmup_s=0.)
    for step in range(100):
        chatter.update([[step % 2 == 0, True]], np.zeros((1,2,3)), [True])
    assert chatter.arrays()['landings'][0] == 0



def main():
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint-check', action='store_true'); p.add_argument('--backend', choices=('mujoco-cpu','mujoco-warp'))
    args=p.parse_args()
    if args.checkpoint_check:
        test_checkpoint_normalization(); print("checkpoint normalization passed")
    elif args.backend:
        integration(args.backend); print(args.backend, 'transition/history integration passed')
    else:
        for test in (test_resampling_quaternion_and_velocity, test_motion_features_heading_invariance,
                     test_dataset_split_and_rng, test_timeout_uses_final_value, test_checkpoint_normalization,
                     test_seated_and_nonfoot_support_fail, test_biped_gait_rejects_hopping_and_standing,
                     test_standing_is_not_locomotion_admission):
            test(); print(test.__name__, 'passed')



def test_standing_is_not_locomotion_admission():
    from types import SimpleNamespace
    from examples.g1.evaluate_amp import rollout
    class StandingEnv:
        num_envs = 2
        step_dt = .02
        terrain_grid_shape = (1, 1)
        terrain_type_names = ('flat',)
        terrain_origins = np.zeros((1, 3))
        terrain_assignments = (np.zeros(2, int), np.zeros(2, int))
        def reset(self, seed=None): pass
        def set_command(self, command): pass
        def get_observations(self): return {'actor': np.zeros((2, 1))}
        def locomotion_state(self): return {'base_position': np.tile([0, 0, .76], (2, 1))}
        def step(self, action):
            data = {**self.locomotion_state(), 'base_linear_velocity_body': np.zeros((2, 3)),
                    'base_angular_velocity_body': np.zeros((2, 3)), 'foot_force': np.tile([[[0,0,100],[0,0,100]]], (2,1,1)),
                    'foot_position': np.zeros((2,2,3)), 'foot_velocity': np.zeros((2,2,3)),
                    'base_clearance': np.full(2,.76)}
            return SimpleNamespace(terminated=np.zeros(2,bool), truncated=np.zeros(2,bool),
                info={'transition_state': data, 'reset_reason': np.zeros(2,int)})
    env = StandingEnv()
    result = rollout(env, lambda obs: np.zeros((2, 1)), steps=20, command=(.8,0,0))
    assert result['survival'] == 1. and result['admission'] == 0.
    result = rollout(env, lambda obs: np.zeros((2, 1)), steps=20, command=(0,0,0), cell_exit=False)
    assert result['admission'] == 1.

if __name__ == '__main__': main()
