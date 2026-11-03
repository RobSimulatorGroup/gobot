"""Process-locked wall-time accounting for single-GPU locomotion experiments."""
from __future__ import annotations
import atexit
import fcntl
import json
from pathlib import Path
import time
import math
import os
import socket
import hashlib


def window_cap_hours(window):
    """Resolve a frozen additional-allocation window, never a fresh allocation."""
    baseline = float(window['baseline_seconds'])
    additional = float(window['additional_gpu_hours'])
    ceiling = float(window['original_total_cap_hours'])
    if not all(math.isfinite(v) for v in (baseline, additional, ceiling)) or baseline < 0 or min(additional, ceiling) <= 0:
        raise ValueError('invalid GPU budget window')
    return min(ceiling, baseline / 3600 + additional)


def bind_budget_window(path, window_path):
    """Attach an existing authorization to a ledger under its process lock.

    Persist the values, not just a path: moving a run or replacing a window file
    must not silently grant a second allocation. Rebinding requires the same window.
    """
    path, window_path = Path(path), Path(window_path)
    window = json.loads(window_path.read_text())
    cap = window_cap_hours(window)
    identity = {key: window[key] for key in ('baseline_seconds', 'additional_gpu_hours', 'original_total_cap_hours')}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    authorization = {**identity, 'sha256': digest, 'source': str(window_path.resolve())}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = json.loads(path.read_text()) if path.exists() else {'seconds': 0., 'runs': []}
        previous = data.get('authorization')
        if previous and previous.get('sha256') != digest:
            raise ValueError('ledger already belongs to a different frozen budget window')
        if not previous:
            data['authorization'] = authorization
            temp = path.with_suffix(path.suffix + '.tmp')
            temp.write_text(json.dumps(data, indent=2) + '\n')
            temp.replace(path)
    return cap


def effective_cap_hours(path, requested_hours):
    data = json.loads(Path(path).read_text()) if Path(path).exists() else {}
    window = data.get('authorization')
    return min(requested_hours, window_cap_hours(window)) if window else requested_hours


class TrainingBudget:
    """Charge active process time, never the offline gap after a machine reboot."""
    def __init__(self, path, *, total_hours, run_hours, metadata=None, allocated_gpus=1):
        values=(run_hours,allocated_gpus) if total_hours is None else (total_hours,run_hours,allocated_gpus)
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError('GPU-hour budgets and allocated GPU count must be positive and finite')
        self.allocated_gpus = allocated_gpus
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix('.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise
        try:
            self.data = json.loads(self.path.read_text()) if self.path.exists() else {'seconds': 0., 'runs': []}
            self.initial_seconds = float(self.data['seconds'])
            if not math.isfinite(self.initial_seconds) or self.initial_seconds < 0:
                raise ValueError('invalid GPU ledger consumption')
            window = self.data.get('authorization')
            requested = math.inf if total_hours is None else total_hours
            cap = min(requested,window_cap_hours(window)) if window else requested
            self.total_hours = None if math.isinf(cap) else cap
            remaining = cap * 3600 - self.initial_seconds
            if remaining <= 0:
                raise RuntimeError('GPU budget exhausted')
        except BaseException:
            self.lock.close()
            raise
        self.limit_seconds = min(run_hours * 3600, remaining) / self.allocated_gpus
        self.start = time.monotonic()
        self.metadata = {**dict(metadata or {}), "pid": os.getpid(), "host": socket.gethostname(),
                         "allocated_gpus": allocated_gpus, "effective_cap_hours": self.total_hours}
        self.closed = False
        if self.data.get('active_run'):
            self.data['runs'].append({**self.data.pop('active_run'), 'interrupted': True})
        self.data['active_run'] = {**self.metadata, 'started_unix': time.time(), 'elapsed_s': 0.}
        self._write()
        atexit.register(self.close)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

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
        self.data['seconds'] = self.initial_seconds + self.elapsed * self.allocated_gpus
        self.data['active_run'].update(self.metadata, elapsed_s=self.elapsed, heartbeat_unix=time.time())
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
