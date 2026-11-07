"""Solver diagnostic summaries must preserve all capacity flags and solve calls."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import mujoco_warp as mjw
import warp as wp

spec = importlib.util.spec_from_file_location('mjlab_training_comparison',
    Path(__file__).resolve().parents[1] / 'support/train_mjlab_locomotion.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_solver_summary_clears_only_solver_bits_and_counts_each_call():
    capacity = int(mjw.OverflowType.NEFC)
    iterations = int(mjw.OverflowType.ITERATIONS)
    linesearch = int(mjw.OverflowType.LS_ITERATIONS)
    flags = wp.array([capacity, 0], dtype=int, device='cpu')
    values = wp.to_torch(flags)
    calls = []

    def step():
        calls.append('step')
        assert values.tolist() == [capacity, 0]
        values[0] |= iterations
        values[1] |= linesearch

    def forward():
        calls.append('forward')
        assert values.tolist() == [capacity, 0]

    sim = SimpleNamespace(wp_data=SimpleNamespace(overflow=flags), step=step, forward=forward)
    counter = module.SolverStatistics(sim)
    printed = []
    logger = SimpleNamespace(writer=object(), ep_extras=[])
    logger.log = lambda **kwargs: printed.append(dict(logger.ep_extras[-1]))
    counter.attach(SimpleNamespace(logger=logger))
    sim.step()
    sim.forward()
    logger.log(it=0)
    assert calls == ['step', 'forward']
    assert values.tolist() == [capacity, 0]
    assert float(printed[0]['Solver/iterations_limit_fraction']) == .25
    assert float(printed[0]['Solver/linesearch_limit_fraction']) == .25
    assert counter.counts.tolist() == [0., 0., 0.]
    sim.forward()
    logger.log(it=1)
    assert float(printed[1]['Solver/iterations_limit_fraction']) == 0.
    assert float(printed[0]['Solver/iterations_limit_fraction']) == .25
