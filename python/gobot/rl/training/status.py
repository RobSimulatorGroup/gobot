"""Shared run status and atomic reports, usable without importing Gobot or Torch.

The isolated reference runner and scene-authored jobs use the same budget heartbeat.
Keep this module standard-library-only so reference physics remain isolated.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import threading
import time


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class ControlledStop(Exception):
    pass


class RunStatus:
    def __init__(self, run_dir: Path, budget):
        self.run_dir = run_dir
        self.budget = budget
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.cancelled = False
        self.state = {"status": "initializing", "pid": os.getpid(), "host": socket.gethostname(),
                      "started_unix": time.time(), "algorithm": "SuperMimic world-model/BPTT"}
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def update(self, **fields):
        with self.lock:
            self.state.update(fields)
            self.state.update(heartbeat_unix=time.time(), elapsed_s=self.budget.elapsed,
                              remaining_s=self.budget.remaining)
            self.budget.update(status=self.state["status"], iteration=self.state.get("iteration", 0))
            write_json(self.run_dir / "status.json", self.state)

    def _heartbeat(self):
        while not self.done.wait(15):
            self.update()
            if self.budget.remaining <= 0:
                self.update(status="budget_exhausted", stop_reason="hard_deadline", checkpoint_note="use most recent saved checkpoint")
                self.budget.close(status="budget_exhausted")
                os._exit(124)

    def close(self):
        self.done.set()
        self.thread.join(timeout=2)
        self.update()
        self.budget.close(status=self.state["status"], iteration=self.state.get("iteration", 0))
