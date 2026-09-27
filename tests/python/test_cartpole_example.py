"""Cartpole scene/batch parity, reset isolation and learner adapter contracts."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

import numpy as np
import gobot

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.cartpole import cartpole_task as task
from examples.cartpole.train.env import CartPoleEnv, PROJECT_PATH


class CartpoleTest(unittest.TestCase):
    def make_env(self, **kwargs):
        env = CartPoleEnv(num_envs=3, sim_workers=1, **kwargs)
        self.addCleanup(env.close)
        return env

    def test_batch_efforts_match_single_scene_steps(self):
        env = self.make_env(disturbance_std=0)
        reference = gobot.app.create_context()
        self.addCleanup(reference.clear_scene)
        reference.set_project_path(str(PROJECT_PATH))
        robot = reference.load_scene("res://cartpole.jscn")
        joints = task.configure_robot(robot)
        reference.fixed_time_step = task.FIXED_DT
        reference.build_world(gobot.PhysicsBackendType.MuJoCoCpu)
        self.assertEqual(env.context.compiled_scene_artifact()["content"],
                         reference.compiled_scene_artifact()["content"])
        initial = env.runtime.robot_state()
        for column, joint in enumerate(joints):
            joint.reset_runtime_state(initial["joint_position"][1, column],
                                      initial["joint_velocity"][1, column])
        for efforts in ((1.0, 0.05), (-1.5, -0.02), (0.0, 0.0)) * 8:
            batch_efforts = np.array(((0.0, 0.0), efforts, (2.0, 0.0)))
            env.runtime.set_joint_effort_targets(batch_efforts)
            env.runtime.step(1, workers=2)
            for joint, effort in zip(joints, efforts):
                joint.set_effort_target(effort)
            reference.step_once()
            state = env.runtime.robot_state()
            for column, joint in enumerate(joints):
                actual = joint.get_runtime_state()
                self.assertAlmostEqual(state["joint_position"][1, column], actual["position"], places=6)
                self.assertAlmostEqual(state["joint_velocity"][1, column], actual["velocity"], places=6)
        self.assertEqual(env.context.root.find(task.JOINT_PATHS[0]).joint_position, 0.0)

    def test_seeded_replay_and_action_clamping(self):
        env = self.make_env(disturbance_start=0, disturbance_interval=3, disturbance_duration=2)
        def rollout(action):
            observations, _ = env.reset(seed=17)
            result = [observations["policy"]]
            for _ in range(12):
                state = env.step(np.full((3, 1), action))
                result.extend((state.obs["policy"].copy(), state.reward.copy(), state.done.copy()))
            return result
        for first, second in zip(rollout(100), rollout(task.FORCE_LIMIT), strict=True):
            np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(env.state.obs["policy"]).all())

    def test_autoreset_preserves_terminal_observation_and_other_environments(self):
        env = self.make_env(max_episode_length=4, disturbance_std=0)
        env.episode_length_buf[:] = (3, 0, 0)
        state = env.step(np.zeros((3, 1)))
        np.testing.assert_array_equal(state.done, (True, False, False))
        np.testing.assert_array_equal(state.info["time_outs"], (True, False, False))
        np.testing.assert_array_equal(state.info["_final_observation"], (True, False, False))
        np.testing.assert_array_equal(env.episode_length_buf, (0, 1, 1))
        terminal = state.final_observation["policy"][0].copy()
        self.assertFalse(np.array_equal(terminal, state.obs["policy"][0]))
        expected_reward = terminal[0] - abs(terminal[6]) - .01 * (terminal[3]**2 + terminal[4]**2)
        self.assertAlmostEqual(state.reward[0], expected_reward, places=6)
        state = env.step(np.zeros((3, 1)))
        self.assertNotIn("final_observation", state.info)
        env.runtime.reset_joint_state(1, "hinge", .6)
        state = env.step(np.zeros((3, 1)))
        self.assertTrue(state.terminated[1])
        self.assertFalse(state.truncated[1])

    def test_invalid_efforts_and_close(self):
        env = self.make_env()
        for efforts in (np.zeros(2), np.zeros((3, 1)), np.full((3, 2), np.nan)):
            with self.assertRaises(ValueError):
                env.runtime.set_joint_effort_targets(efforts)
        with self.assertRaises(RuntimeError):
            env.runtime.set_joint_effort_targets(np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            env.step(np.full((3, 1), np.inf))
        env.close()
        env.close()
        self.assertFalse(env.context.has_world)
        with self.assertRaises(RuntimeError):
            env.step(np.zeros((3, 1)))

    @unittest.skipUnless(find_spec("torch") and find_spec("tensordict"), "optional learner packages")
    def test_shared_rsl_adapter_and_playback_observation(self):
        import torch
        from gobot.rl.rsl_rl import RslRlVecEnvWrapper
        from examples.cartpole.scripts.cartpole import Script
        env = self.make_env(max_episode_length=2)
        wrapped = RslRlVecEnvWrapper(env, device="cpu")
        wrapped.reset(seed=8)
        wrapped.episode_length_buf = torch.tensor((1, 0, 0))
        obs, reward, done, extras = wrapped.step(torch.zeros((3, 1)))
        self.assertEqual(tuple(obs["policy"].shape), (3, 7))
        self.assertEqual(tuple(reward.shape), (3,))
        self.assertEqual(done.tolist(), [True, False, False])
        self.assertEqual(extras["time_outs"].tolist(), [True, False, False])
        # Feed a known runtime state into the actual Play observation entry point.
        from types import SimpleNamespace
        positions = (.2, .3)
        velocities = (-.4, .5)
        script = SimpleNamespace(target_cart_position=.7, _runtime_joint_states=lambda: {
            name: {"position": p, "velocity": v}
            for name, p, v in zip(task.JOINT_NAMES, positions, velocities)
        })
        np.testing.assert_array_equal(Script._observation(script, task.FIXED_DT),
                                      task.observation(positions, velocities, .7))


if __name__ == "__main__":
    unittest.main()
