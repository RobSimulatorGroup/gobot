from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def benchmark():
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("conveyor_ipc_batch_test", root / "benchmark/conveyor_ipc_batch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_process_memory_matches_pid_and_normalizes_cuda_nvml_uuid(benchmark, monkeypatch):
    output = "42, 1024, GPU-abc\n99, 6000, GPU-abc\n42, 9000, GPU-other\n"
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=output))
    sampler = benchmark.ProcessMemorySampler(42, "abc")
    sampler.sample()
    result = sampler.result()
    assert result["sampled_process_peak_bytes"] == 1024 ** 3
    assert result["other_compute_pids"] == [99]
    assert result["error"] is None


def test_unavailable_memory_is_not_reported_as_zero(benchmark, monkeypatch):
    sampler = benchmark.ProcessMemorySampler(42, "abc")
    assert sampler.result()["sampled_process_peak_bytes"] is None
    assert sampler.result()["error"]
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="42, N/A, GPU-abc\n"))
    sampler.sample()
    assert sampler.result()["error"]
    assert sampler.result()["sampled_process_peak_bytes"] is None


def test_memory_sampler_always_stops_when_trial_fails(benchmark, monkeypatch):
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="42, 2, GPU-abc\n"))
    sampler = benchmark.ProcessMemorySampler(42, "GPU-abc", interval=.001)
    with pytest.raises(RuntimeError):
        with sampler:
            raise RuntimeError("trial failed")
    assert not sampler.thread.is_alive()
    assert sampler.result()["sampled_process_peak_bytes"] == 2 * 1024 ** 2


def test_latency_summary_keeps_sample_count_and_tail(benchmark):
    result = benchmark.timing([1., 2., 3., 4., 10.])
    assert result["samples"] == 5
    assert result["median"] == 3.
    assert result["p95"] > result["median"]
