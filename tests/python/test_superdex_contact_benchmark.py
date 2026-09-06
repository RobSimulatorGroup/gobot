"""Contact captures must use actual ticks and retain contact loss and failures."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "superdex_contact_benchmark",
    Path(__file__).resolve().parents[2] / "benchmark/superdex_contact_benchmark.py",
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def sample(tick, hands=()):
    return {
        "tick": tick, "phase": "grip", "step_once_seconds": tick * 0.001,
        "simulation_time_seconds": tick * 0.002,
        "observed_tick_seconds": tick * 0.001, "command_seconds": 0.0, "state_view_seconds": 0.0,
        "max_penetration_meters": 0.0, "directed_contact_count": len(hands) * 2,
        "contacting_hands": {"blue_mailer": list(hands)},
        "solver": {
            "timings_available": True, "total_step_time_seconds": 0.001, "solve_time_seconds": 0.0009,
            "newton_iterations": 1, "line_search_iterations": 2, "linear_iterations": 3,
            "convergence": "converged", "residual_norm": 0.0,
            "stage_timings": [{"name": name, "time_seconds": 0.0001, "calls": 1,
                               "parallel_sum": name not in benchmark.TIMING_SEMANTICS["wall_stages"]}
                              for name in benchmark.STAGES],
        },
    }


def test_real_contact_triggers_consecutive_window_not_authored_phase():
    window = benchmark.ContactWindow("blue_mailer", 2, 2, 3)
    assert window.add(sample(1))
    assert window.add(sample(2, ["left"]))
    assert window.contact_tick is None
    assert window.add(sample(3, ["left", "right"]))
    assert window.add(sample(4, ["left", "right"]))
    assert window.add(sample(5))
    assert window.add(sample(6, ["left", "right"]))
    assert not window.add(sample(7, ["left"]))
    result = window.summary()
    assert result["window_complete"]
    assert result["contact_trigger_tick"] == 3
    assert result["precontact"]["ticks"] == 2
    assert result["contact_warmup"]["ticks"] == 2
    assert result["measurement"]["ticks"] == 3
    assert result["measurement"]["step_once_seconds"]["p50"] == 0.006
    assert result["measurement"]["step_once_seconds"]["p95"] == 0.007
    assert result["measurement"]["contact_fraction"] == pytest.approx(1 / 3)
    assert result["measured_contact_only"]["ticks"] == 1
    assert result["measured_contact_lost"]["ticks"] == 2


def test_unreached_contact_is_incomplete_not_a_startup_performance_pass():
    window = benchmark.ContactWindow("blue_mailer", 2, 100, 500)
    window.add(sample(1))
    result = window.summary()
    assert not result["window_complete"]
    assert result["contact_trigger_tick"] is None
    assert result["measurement"] == {"ticks": 0}


def test_contact_only_during_warmup_does_not_qualify_the_measured_window():
    window = benchmark.ContactWindow("blue_mailer", 2, 1, 1)
    window.add(sample(1, ["left", "right"]))
    window.add(sample(2))
    result = window.summary()
    assert result["window_complete"]
    assert not result["measurement_contains_contact"]


def test_zero_warmup_includes_trigger_tick_and_duplicate_hands_do_not_trigger():
    window = benchmark.ContactWindow("blue_mailer", 2, 0, 1)
    assert window.add(sample(1, ["left", "left"]))
    assert not window.add(sample(2, ["left", "right"]))
    assert window.summary()["measurement"]["first_tick"] == 2


@pytest.mark.parametrize("mode", ["unavailable", "missing", "duplicate", "nan", "negative", "diverged"])
def test_invalid_telemetry_is_not_zero_filled(mode):
    row = sample(1)
    if mode == "unavailable":
        row["solver"]["timings_available"] = False
    elif mode == "missing":
        row["solver"]["stage_timings"].pop()
    elif mode == "duplicate":
        row["solver"]["stage_timings"][1]["name"] = "apply_forces"
    elif mode == "nan":
        row["solver"]["residual_norm"] = float("nan")
    elif mode == "negative":
        row["solver"]["stage_timings"][0]["time_seconds"] = -1
    else:
        row["solver"]["convergence"] = "diverged"
    with pytest.raises(ValueError):
        benchmark.ContactWindow("blue_mailer", 2, 0, 1).add(row)


def test_skipped_or_reset_ticks_are_rejected():
    window = benchmark.ContactWindow("blue_mailer", 2, 100, 500)
    window.add(sample(1))
    for tick in (1, 3):
        with pytest.raises(ValueError, match="skipped, duplicated or reset"):
            window.add(sample(tick))


def test_fixed_step_does_not_change_or_reset_while_counter_keeps_advancing():
    window = benchmark.ContactWindow("blue_mailer", 2, 100, 500)
    window.add(sample(1))
    row = sample(2)
    row["simulation_time_seconds"] = 0.0
    with pytest.raises(ValueError, match="unchanged 2 ms"):
        window.add(row)


def test_parallel_inclusive_durations_are_retained_not_capped_to_wall_time():
    window = benchmark.ContactWindow("blue_mailer", 1, 0, 1)
    row = sample(1, ["left"])
    for stage in row["solver"]["stage_timings"]:
        if stage["name"] == "assembly":
            stage["time_seconds"] = 0.01
    window.add(row)
    assembly = window.summary()["measurement"]["stage_timings"]["assembly"]
    assert assembly["parallel_sum"]
    assert assembly["seconds"]["p50"] == 0.01


def test_failed_tick_preserves_nonfinite_diagnostic_evidence_as_json():
    import json
    original = {"tick": 942, "solver": {"residual_norm": float("nan"), "convergence": "diverged"}}
    record = benchmark.failure_record(original)
    assert record["solver"]["residual_norm"] == "nan"
    assert json.loads(json.dumps(record, allow_nan=False))["tick"] == 942
    assert original["solver"]["residual_norm"] != original["solver"]["residual_norm"]


@pytest.mark.parametrize("diagnostics_fail", [False, True])
def test_step_failure_is_observed_before_cleanup_without_masking_solver_error(monkeypatch, diagnostics_fail):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples/conveyor_packages"))
    import conveyor_packages_batch as conveyor

    class FailedContext:
        def step_once(self):
            raise RuntimeError("injected solver failure")

        def get_solver_diagnostics(self):
            if diagnostics_fail:
                raise RuntimeError("diagnostics unavailable")
            return {"convergence": "diverged"}

    failures = []
    with pytest.raises(RuntimeError, match="injected solver failure"):
        conveyor._step_with_diagnostics(FailedContext(), 942, 941, failures.append)
    assert len(failures) == 1
    assert failures[0]["tick"] == 942
    assert failures[0]["step_once_seconds"] >= 0
    if diagnostics_fail:
        assert failures[0]["diagnostic_error"] == "diagnostics unavailable"
    else:
        assert failures[0]["solver"]["convergence"] == "diverged"
