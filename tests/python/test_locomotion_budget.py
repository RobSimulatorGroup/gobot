"""Reboot recovery must not charge offline time or double-spend a GPU budget."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from gobot.rl.locomotion.budget import TrainingBudget


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'budget.json'
        path.write_text(json.dumps({'seconds': 120., 'runs': [], 'active_run': {'elapsed_s': 30., 'started_unix': 1}}))
        with patch('gobot.rl.locomotion.budget.time.monotonic', return_value=100.) as clock:
            budget = TrainingBudget(path, total_hours=1, run_hours=.1, metadata={'stage': 'flat'})
            assert budget.limit_seconds == 360.
            try:
                TrainingBudget(path, total_hours=1, run_hours=.1)
            except BlockingIOError:
                pass
            else:
                raise AssertionError('two runs acquired the same budget')
            clock.return_value = 130.
            budget.update(last_iteration=7)
            assert json.loads(path.read_text())['seconds'] == 150.
            clock.return_value = 140.
            budget.close()
            budget.close()
        saved = json.loads(path.read_text())
        assert saved['seconds'] == 160. and 'active_run' not in saved
        assert saved['runs'][0]['interrupted']
        assert len(saved['runs']) == 2
    print('budget lock, offline recovery and idempotent close passed')

if __name__ == '__main__': main()
