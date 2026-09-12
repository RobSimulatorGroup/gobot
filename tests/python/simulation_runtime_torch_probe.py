"""Subprocess probe: third-party Torch bindings entered from the native worker."""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from gobot.sim import AsyncSimulationSession, SimulationRuntimeSpec


class Provider:
    num_envs = 4
    fixed_time_step = .002

    def __init__(self, device):
        import torch
        self.torch = torch
        self.values = torch.zeros((4, 12, 6), device=device)

    def step(self, *, nsteps=1):
        self.values.add_(nsteps)

    def reset(self, mask):
        self.values[self.torch.as_tensor(mask, device=self.values.device)] = 0

    def snapshot_arrays(self, fields, environments):
        # Match the first failing Torch operation in MuJoCo robot-state reads.
        spatial = self.values[:, 2]
        state = self.torch.empty((4, 6), device=self.values.device)
        state.copy_(spatial)
        return {"state": state[list(environments)].cpu().numpy()}

    def close(self):
        self.values = None


def create(device, fail=False):
    provider = Provider(device)
    if fail:
        provider.close()
        raise RuntimeError("injected failure after Torch import")
    return provider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--exit-mode", choices=("shutdown", "atexit", "factory-failure"), default="shutdown")
    args = parser.parse_args()
    assert "torch" not in sys.modules, "Torch must first load on the worker"
    runtime = AsyncSimulationSession(SimulationRuntimeSpec(
        "simulation_runtime_torch_probe:create", {"device": args.device,
            "fail": args.exit_mode == "factory-failure"}),
        fields=("state",), max_hz=1e9)

    def complete():
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            result = runtime.poll()
            if result is not None:
                if args.exit_mode == "factory-failure":
                    assert "injected failure after Torch import" in result.error
                else:
                    assert not result.error, result.error
                assert not result.presentation_error, result.presentation_error
                return result
            time.sleep(.001)
        raise AssertionError("worker completion timed out")

    try:
        complete()
        if args.exit_mode == "factory-failure":
            return
        for tick in range(1, 21):
            assert runtime.submit()
            result = complete()
            assert result.step.completed
            np.testing.assert_array_equal(np.asarray(result.snapshot.buffer("state")), np.full((1, 6), tick))
        assert runtime.reset()
        reset = complete()
        assert reset.clocks[0].tick == 0
        np.testing.assert_array_equal(np.asarray(reset.snapshot.buffer("state")), np.zeros((1, 6)))
    finally:
        if args.exit_mode != "atexit":
            runtime.shutdown()
    print(json.dumps({"device": args.device, "steps": 20, "reset": True,
        "exit_mode": args.exit_mode}))


if __name__ == "__main__":
    main()
