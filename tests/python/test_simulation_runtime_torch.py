"""Cold extension imports must work on native workers, including CPython 3.13."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


def run_probe(device, exit_mode):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("Torch is not installed")
    # A fatal CPython thread-state error must fail this test, not kill pytest.
    # A fresh process also prevents prior tests' Torch imports hiding the bug.
    result = subprocess.run([sys.executable,
        str(Path(__file__).with_name("simulation_runtime_torch_probe.py")),
        "--device", device, "--exit-mode", exit_mode],
        capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("exit_mode", ["shutdown", "atexit", "factory-failure"])
def test_torch_first_import_on_worker_cpu(exit_mode):
    run_probe("cpu", exit_mode)


@pytest.mark.skipif(os.environ.get("GOBOT_RUN_WARP_IPC_GPU_TEST") != "1",
                    reason="opt-in CUDA worker regression")
def test_torch_first_import_on_worker_cuda():
    run_probe("cuda:0", "shutdown")


@pytest.mark.skipif(os.environ.get("GOBOT_RUN_WARP_IPC_GPU_TEST") != "1",
                    reason="opt-in MuJoCo Warp + libuipc worker regression")
def test_press_factory_cold_start_full_gpu_cycle():
    result = subprocess.run([sys.executable,
        str(Path(__file__).with_name("simulation_runtime_press_probe.py"))],
        capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
