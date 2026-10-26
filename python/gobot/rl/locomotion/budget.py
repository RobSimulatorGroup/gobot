"""Process-locked wall-time accounting for single-GPU locomotion experiments."""
from __future__ import annotations
import atexit
import fcntl
import json
from pathlib import Path
import time


class TrainingBudget:
    """Charge active process time, never the offline gap after a machine reboot."""
    def __init__(self, path, *, total_hours, run_hours, metadata=None):
        if not 0 < total_hours <= 24 or run_hours <= 0:
            raise ValueError('budget must be positive and at most 24 GPU-hours')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix('.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {'seconds': 0., 'runs': []}
        self.initial_seconds = float(self.data['seconds'])
        remaining = total_hours * 3600 - self.initial_seconds
        if remaining <= 0:
            self.lock.close()
            raise RuntimeError('GPU budget exhausted')
        self.limit_seconds = min(run_hours * 3600, remaining)
        self.start = time.monotonic()
        self.metadata = dict(metadata or {})
        self.closed = False
        if self.data.get('active_run'):
            self.data['runs'].append({**self.data.pop('active_run'), 'interrupted': True})
        self.data['active_run'] = {**self.metadata, 'started_unix': time.time(), 'elapsed_s': 0.}
        self._write()
        atexit.register(self.close)

    @property
    def elapsed(self):
        return time.monotonic() - self.start

    @property
    def remaining(self):
        return max(0., self.limit_seconds - self.elapsed)

    def _write(self):
        temp = self.path.with_suffix(self.path.suffix + '.tmp')
        temp.write_text(json.dumps(self.data, indent=2) + '\n')
        temp.replace(self.path)

    def update(self, **metadata):
        if self.closed:
            return
        self.metadata.update(metadata)
        self.data['seconds'] = self.initial_seconds + self.elapsed
        self.data['active_run'].update(self.metadata, elapsed_s=self.elapsed)
        self._write()

    def close(self, **metadata):
        if self.closed:
            return
        self.update(**metadata)
        self.data['runs'].append(self.data.pop('active_run'))
        self._write()
        self.closed = True
        self.lock.close()
        atexit.unregister(self.close)
