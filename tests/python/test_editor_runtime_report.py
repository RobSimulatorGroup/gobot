"""Benchmark reporting must not turn missing telemetry or failed captures into passes."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "editor_runtime_report", Path(__file__).resolve().parents[2] / "benchmark/editor_runtime_report.py"
)
reporting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporting)


def capture():
    return {"schema": 1, "commit": "test", "project": "test", "play": True,
            "warmup_frames": 1, "requested_frames": 2, "faulted": False, "error": "",
            "scene_ready_ms": 300, "first_play_ms": 200, "fixed_dt": 0.002,
            "columns": ["frame_ms", "physics_dispatch_ms", "process_ms", "draw_ms", "simulation_time"],
            "samples": [[500, 400, 50, 50, 0], [10, 1, 2, 3, 0.01], [20, 2, 3, 4, 0.02]]}


def test_warmup_is_excluded_and_rtf_uses_matching_intervals():
    result = reporting.summarize(capture())
    assert result["metrics"]["frame_ms"]["p50"] == 15
    assert result["real_time_factor"] == 0.5
    assert result["scene_ready_ms"] == 300


def test_faulted_incomplete_and_empty_captures_are_rejected():
    for key, value in (("faulted", True), ("requested_frames", 3), ("samples", []), ("world_ready", False)):
        data = capture()
        data[key] = value
        with pytest.raises(ValueError):
            reporting.summarize(data)


def test_failed_or_unrecorded_exit_rejects_even_complete_frame_samples():
    for metadata in ({}, {"exit_code": -9}, {"exit_code": 0, "timed_out": True}):
        with pytest.raises(ValueError, match="exit successfully"):
            reporting.summarize(capture(), metadata)
    assert reporting.summarize(capture(), {"exit_code": 0})["frames"] == 2


def test_absent_process_memory_is_unavailable_not_zero():
    gpu = {"device": "4070", "device_used": "4 GiB", "processes": []}
    result = reporting.summarize_gpu({"gpu_samples": [{"gpus": [gpu]}]})["4070"]
    assert result["process_peak_mib"] is None
    assert result["telemetry"] == "unavailable"
    gpu["processes"] = [{"type": "C+G", "memory": "512 MiB"}]
    result = reporting.summarize_gpu({"gpu_samples": [{"gpus": [gpu]}]})["4070"]
    assert result["process_peak_mib"] == 512
    assert result["device_peak_mib"] == 4096


def test_counters_reset_does_not_report_negative_uploads():
    data = capture()
    data["columns"].append("geometry_uploads_total")
    for row, value in zip(data["samples"], (9, 10, 2)):
        row.append(value)
    result = reporting.summarize(data)["resources"]["geometry_uploads_total"]
    assert result == {"delta": None, "counter_reset": True}


def test_solver_timings_use_completed_ticks_not_repeated_display_frames():
    data = capture()
    data["physics_columns"] = ["render_frame", "tick", "dispatch_ticks", "step_ms", "solve_ms",
                               "newton_iterations", "line_search_iterations"]
    data["physics_samples"] = [[0, 1, 1, 900, 800, 16, 8], [1, 2, 1, 210, 200, 4, 2],
                               [2, 3, 1, 310, 300, 6, 4]]
    result = reporting.summarize(data)["physics"]
    assert result["step_ms"]["p50"] == 260
    assert result["samples"] == 2
    assert result["coverage"] == "all_ticks"
    data["physics_samples"][-1][2] = 8
    assert reporting.summarize(data)["physics"]["coverage"] == "last_tick_per_dispatch"


def test_reset_followed_by_catchup_is_not_misreported_as_monotonic():
    data = capture()
    data["requested_frames"] = 3
    data["samples"].insert(2, [10, 1, 2, 3, 0])
    with pytest.raises(ValueError, match="reset"):
        reporting.summarize(data)
