"""Shared velocity contracts, G1 topology and backend-independent task semantics."""
from __future__ import annotations
from dataclasses import replace
import math
import argparse
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.g1.config import g1_velocity_cfg, HAND_JOINTS
from gobot.rl.locomotion import make_velocity_env
from gobot.rl.locomotion.curriculum import progress_masks
from gobot.rl.locomotion.terms import posture_std
from gobot.rl.runtime import BatchSimulationRuntime
from gobot.sim.providers.mujoco_warp import MuJoCoWarpProvider
from gobot.sim.providers.base import SimulationCapacityError


def test_progress_curriculum():
    # Standing drift cannot promote; completed commanded progress can; a short
    # failed episode still owes the uncompleted command distance.
    distance = np.array([4.5, 1.7, .4, 0.])
    required = np.array([0., 2., 8., 0.])
    up, down = progress_masks(distance, required, 8.)
    np.testing.assert_array_equal(up, [False, True, False, False])
    np.testing.assert_array_equal(down, [False, False, True, False])
    tup, tdown = progress_masks(torch.tensor(distance), torch.tensor(required), 8.)
    np.testing.assert_array_equal(tup.numpy(), up)
    np.testing.assert_array_equal(tdown.numpy(), down)


def test_solver_limits_are_not_capacity_loss():
    provider = MuJoCoWarpProvider.__new__(MuJoCoWarpProvider)
    provider._closed = False
    provider._torch = torch
    flags = torch.tensor([512, 1024, 1536], dtype=torch.int32)
    provider._arrays = {'overflow': flags}
    provider.assert_no_overflow()
    assert provider.solver_limit_counts() == {'iterations': 2, 'line_search_iterations': 2}
    flags[0] |= 8
    try:
        provider.assert_no_overflow()
    except SimulationCapacityError as error:
        assert 'narrowphase' in str(error)
    else:
        raise AssertionError('contact overflow must still fail')


def test_g1_profile():
    cfg = g1_velocity_cfg()
    assert len(cfg.joint_names) == 29
    assert len(cfg.robot.held_joint_targets) == 14
    assert not set(cfg.joint_names) & set(HAND_JOINTS)
    assert [len(foot.collision_shapes) for foot in cfg.robot.feet] == [7, 7]
    values = posture_std(cfg.joint_names, cfg.rewards.pose_std_walking, .3, .6)
    assert values.shape == (29,) and np.all(values > 0)
    assert cfg.robot.torso_link == 'torso_link'
    assert cfg.rewards.angular_momentum < 0
    try:
        replace(cfg.robot, held_joint_targets={cfg.joint_names[0]: 0.})
    except ValueError:
        pass
    else:
        raise AssertionError('overlapping policy and held joints must fail')


def test_g1_cpu_reset_and_terms():
    cfg = g1_velocity_cfg()
    cfg.observations.actor_noise = cfg.domain_randomization.enabled = cfg.push_enabled = False
    cfg.randomize_reset_yaw = False
    cfg.spawn_jitter = 0.
    env = make_velocity_env(cfg, num_envs=2, seed=23)
    try:
        assert (env.num_actions, env.num_obs, env.num_privileged_obs) == (29, 286, 298)
        env.set_terrain_assignments([0, 0], [0, 0])
        first, _ = env.reset(seed=23)
        trace = []
        for _ in range(4):
            trace.append(env.step(np.zeros((2, 29), dtype=np.float32)).obs['actor'].copy())
        second, _ = env.reset(seed=23)
        np.testing.assert_allclose(first['actor'], second['actor'], atol=1e-6)
        for expected in trace:
            np.testing.assert_allclose(env.step(np.zeros((2, 29), dtype=np.float32)).obs['actor'], expected, atol=1e-6)
        env.set_terrain_assignments([0, 0], [0, 0], fixed=True)
        cfg.terrain_curriculum = False
        env.reset(seed=23)
        assert env.terrain_assignments[0].tolist() == [0, 0]
        assert env.terrain_assignments[1].tolist() == [0, 0]
        before = env.locomotion_state()['joint_position'][1].copy()
        env.reset(np.array([0]))
        np.testing.assert_array_equal(env.locomotion_state()['joint_position'][1], before)
        hands = BatchSimulationRuntime(env.context, robot='g1', base_link='pelvis', joint_names=HAND_JOINTS)
        hand_state = hands.robot_state()
        np.testing.assert_allclose(np.asarray(hand_state['joint_position'])[0], 0., atol=1e-6)
        np.testing.assert_allclose(np.asarray(hand_state['joint_velocity'])[0], 0., atol=1e-6)
        # Real angular momentum from physical joint motion must contribute.
        env.backend.set_base_velocity(0, (0., 0., 0.), (1., .4, .2))
        state = env.step(np.zeros((2, 29), dtype=np.float32))
        assert np.isfinite(state.obs['actor']).all() and np.isfinite(state.reward).all()
        momentum = env.locomotion_state()['angular_momentum']
        assert np.linalg.norm(momentum[0]) > 0
        terms = env.backend.state.reward_terms
        index = env._reward_term_names.index('angular_momentum')
        assert terms[0, index] < 0
        # CUDA adapter descriptions follow authored scan geometry, not constants.
        from gobot.rl.locomotion.warp_env import _authored_raycast_specs
        from gobot.rl.locomotion.math import find_node_by_name
        scanner = find_node_by_name(env.robot, "terrain_scan")
        scanner.grid_size = (.4, .4)
        specs = _authored_raycast_specs("g1_", cfg.robot, (0,), env.robot, "terrain_scan")
        assert len(specs[0].local_offsets) == 25
        scanner.grid_size = (1.6, 1.)
        saved = env.training_state_dict()
        assert saved['curriculum_mode'] == 'command_progress'
        env.load_training_state_dict(saved)
        try:
            env.load_training_state_dict({**saved, 'curriculum_mode': 'legacy'})
        except ValueError:
            pass
        else:
            raise AssertionError('incompatible curriculum checkpoint must fail')
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration", action="store_true")
    args = parser.parse_args()
    if args.integration:
        if not (Path(__file__).resolve().parents[2] / "examples/g1/assets/meshes/pelvis.STL").exists():
            print("Prepare G1 assets with python -m examples.g1.prepare before the integration test")
            raise SystemExit(77)
        tests = (test_g1_cpu_reset_and_terms,)
    else:
        tests = (test_progress_curriculum, test_solver_limits_are_not_capacity_loss, test_g1_profile)
    for test in tests:
        test()
        print(test.__name__, 'passed')

if __name__ == '__main__':
    main()
