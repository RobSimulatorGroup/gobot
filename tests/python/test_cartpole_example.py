"""Single-pole iLQR math, native-batch parity and actual Gobot playback."""
from pathlib import Path
import sys
import unittest
import threading
from unittest.mock import patch

import numpy as np
import mujoco
import gobot

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.cartpole import cartpole_task as task
from examples.cartpole.cartpole_swingup import Planner, PlanningCancelled, cost, expand, solve, verify_playback

PROJECT = Path(__file__).resolve().parents[2] / "examples/cartpole"


class CartpoleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.authored_scene = (PROJECT / "cartpole.jscn").read_bytes()
        cls.context = gobot.app.create_context()
        cls.context.set_project_path(str(PROJECT))
        cls.context.load_scene("res://cartpole.jscn")
        cls.joints = task.configure_context(cls.context)
        cls.artifact = cls.context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)

    @classmethod
    def tearDownClass(cls):
        cls.context.clear_scene()

    def planner(self, **kwargs):
        planner = Planner(self.artifact, **kwargs)
        self.addCleanup(planner.close)
        return planner

    def test_batch_transition_and_linearization_match_independent_mujoco(self):
        planner = self.planner(workers=2)
        rng = np.random.default_rng(3)
        x = rng.normal(scale=.1, size=(task.HORIZON * 6, 4))
        u = rng.uniform(-.5, .5, (len(x), 1))
        got = planner.transition(x, u)
        reference = mujoco.MjData(planner.model)
        for index in (0, 5, len(x) - 1):
            mujoco.mj_resetData(planner.model, reference)
            reference.qpos[:], reference.qvel[:] = x[index, :2], x[index, 2:]
            reference.ctrl[:] = task.FORCE_LIMIT * u[index]
            mujoco.mj_step(planner.model, reference, nstep=task.SUBSTEPS)
            np.testing.assert_allclose(got[index], np.r_[reference.qpos, reference.qvel], atol=1e-12, rtol=0)
        xs = rng.normal(scale=.1, size=(task.HORIZON + 1, 4))
        us = rng.uniform(-.5, .5, (task.HORIZON, 1))
        A, B = planner.linearize(xs, us)
        # Independent central differences check the complete six-substep map.
        for column in range(5):
            results = []
            for sign in (-1, 1):
                v, action = xs[0].copy(), us[0].copy()
                if column < 4:
                    v[column] += sign * 1e-5
                else:
                    action[0] += sign * 1e-5
                mujoco.mj_resetData(planner.model, reference)
                reference.qpos[:], reference.qvel[:] = v[:2], v[2:]
                reference.ctrl[:] = task.FORCE_LIMIT * action
                mujoco.mj_step(planner.model, reference, nstep=task.SUBSTEPS)
                results.append(np.r_[reference.qpos, reference.qvel])
            expected = (results[1] - results[0]) / 2e-5
            np.testing.assert_allclose(A[0, :, column] if column < 4 else B[0, :, 0], expected, atol=2e-6, rtol=2e-5)

    def test_cost_derivatives_include_terminal_weight(self):
        xs = np.tile([.3, 1.2, -.2, .7], (task.HORIZON + 1, 1))
        us = np.full((task.HORIZON, 1), .4)
        lx, lxx, lu, luu = expand(xs, us)
        eps = 1e-4
        for t in (0, task.HORIZON):
            for column in range(4):
                plus, minus = xs[t].copy(), xs[t].copy()
                plus[column] += eps
                minus[column] -= eps
                values = cost(t, plus, us[0]), cost(t, minus, us[0]), cost(t, xs[t], us[0])
                self.assertAlmostEqual((values[0] - values[1]) / (2 * eps), lx[t, column], places=5)
                self.assertAlmostEqual((values[0] + values[1] - 2 * values[2]) / eps**2, lxx[t, column, column], places=4)
        self.assertAlmostEqual(float(lu[0, 0]), .16)
        self.assertAlmostEqual(float(luu[0, 0, 0]), .4)

    def test_seeded_swingup_and_actual_scene_playback(self):
        plan = solve(self.artifact, workers=1)
        again = solve(self.artifact, workers=2)
        np.testing.assert_array_equal(plan.states, again.states)
        np.testing.assert_array_equal(plan.actions, again.actions)
        self.assertLessEqual(float(np.abs(plan.actions).max()), 1.0)
        self.assertLessEqual(abs(plan.force(0, np.array([100., -100., 100., -100.]))), task.FORCE_LIMIT)
        with self.assertRaises(ValueError):
            plan.force(0, np.full(4, np.nan))
        self.context.build_world(gobot.PhysicsBackendType.MuJoCoCpu)
        samples = verify_playback(self.context, self.joints, plan)
        self.assertEqual(samples.shape, (3360, 4))
        # A headless context owns its live scene; stepping never saves over the
        # authoring file. Editor clone isolation is covered by ScenePlaySession.
        self.assertEqual((PROJECT / "cartpole.jscn").read_bytes(), self.authored_scene)
        self.assertFalse(self.context.scene_dirty)

    def test_line_search_rejects_only_failed_candidate(self):
        planner = self.planner(workers=1)
        from mujoco.batch import SimulationError
        original = planner.line.step
        calls = 0

        def fail_one(*args, **kwargs):
            nonlocal calls
            original(*args, **kwargs)
            calls += 1
            if calls == 1:
                raise SimulationError(np.array([2], dtype=np.int32), "injected worker failure")

        with patch.object(planner.line, "step", side_effect=fail_one):
            _, _, costs = planner.rollout(task.INITIAL_STATE, np.zeros((task.HORIZON, 1)))
        self.assertTrue(np.isinf(costs[2]))
        self.assertTrue(np.isfinite(np.delete(costs, 2)).all())

    def test_editor_stop_cancels_solver_without_scene_access_on_worker(self):
        from examples.cartpole.scripts import cartpole as playback
        started = threading.Event()

        def waiting_solver(artifact, *, cancelled, progress):
            self.assertIsInstance(artifact["content"], str)
            started.set()
            while not cancelled():
                threading.Event().wait(.001)
            raise PlanningCancelled()

        script = playback.Script()
        script._attach(self.context.root, self.context.root, self.context)
        with patch.object(playback, "solve", side_effect=waiting_solver):
            script._ready()
            self.assertTrue(started.wait(5))
            script._exit_tree()
            script.worker.join(timeout=5)
        self.assertFalse(script.worker.is_alive())
        self.assertTrue(script.messages.empty())

    def test_cancellation_and_invalid_configuration(self):
        planner = self.planner(cancelled=lambda: True)
        with self.assertRaises(PlanningCancelled):
            planner.solve()
        with self.assertRaises(ValueError):
            Planner(self.artifact, workers=-1)
        with self.assertRaises(ValueError):
            planner.solve(iterations=0)


if __name__ == "__main__":
    unittest.main()
