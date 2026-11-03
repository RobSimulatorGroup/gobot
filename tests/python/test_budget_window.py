"""A restarted or separately invoked job cannot replenish an attached allocation."""
import json
from unittest.mock import patch
import pytest
from gobot.rl.locomotion.budget import TrainingBudget, bind_budget_window, effective_cap_hours


def test_window_survives_restart_and_larger_cli_limit(tmp_path):
    ledger, window = tmp_path / 'ledger.json', tmp_path / 'window.json'
    ledger.write_text(json.dumps({'seconds': 2 * 3600, 'runs': []}))
    window.write_text(json.dumps({'baseline_seconds': 3600, 'additional_gpu_hours': 2, 'original_total_cap_hours': 24}))
    assert bind_budget_window(ledger, window) == 3
    assert effective_cap_hours(ledger, 24) == 3
    assert effective_cap_hours(ledger, 2.5) == 2.5
    with patch('gobot.rl.locomotion.budget.time.monotonic', return_value=0.) as clock:
        with TrainingBudget(ledger, total_hours=24, run_hours=8) as budget:
            assert budget.remaining == 3600
            clock.return_value = 600
        with TrainingBudget(ledger, total_hours=24, run_hours=8) as budget:
            assert budget.remaining == 3000
    assert bind_budget_window(ledger, window) == 3
    window.write_text(json.dumps({'baseline_seconds': 7200, 'additional_gpu_hours': 2, 'original_total_cap_hours': 24}))
    with pytest.raises(ValueError, match='different frozen'):
        bind_budget_window(ledger, window)


def test_exhausted_or_invalid_authorization_releases_lock(tmp_path):
    path = tmp_path / 'budget.json'
    data = {'seconds': 3600, 'runs': [], 'authorization': {
        'baseline_seconds': 0, 'additional_gpu_hours': 1, 'original_total_cap_hours': 24}}
    path.write_text(json.dumps(data))
    for _ in range(2):
        with pytest.raises(RuntimeError, match='exhausted'):
            TrainingBudget(path, total_hours=24, run_hours=8)
    data['authorization']['additional_gpu_hours'] = float('nan')
    path.write_text(json.dumps(data))
    for _ in range(2):
        with pytest.raises(ValueError, match='window'):
            TrainingBudget(path, total_hours=24, run_hours=8)
