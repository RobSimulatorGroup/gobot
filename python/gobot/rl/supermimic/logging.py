"""Live progress and metrics for the SuperMimic control-step clock."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Progress:
    """Heartbeat during data loading/compilation, without GPU or MPI thread work."""

    def __init__(self, output, rank, interval=10.0):
        self.output = Path(output)
        self.rank = rank
        self.path = self.output / (
            "status.json" if rank == 0 else f"status.rank{rank}.json"
        )
        self.interval = interval
        self.started = time.monotonic()
        self.phase_started = self.started
        self.row = {"state": "starting", "phase": "initializing", "rank": rank}
        self.lock = threading.Lock()
        self.finished = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.update()
        self.thread.start()

    def update(self, **values):
        with self.lock:
            if values.get("phase", self.row["phase"]) != self.row["phase"]:
                self.phase_started = time.monotonic()
            self.row.update(values)
            self._write()
        if self.error is not None:
            raise RuntimeError(f"progress writer failed: {self.error}")

    def _write(self):
        now = time.monotonic()
        self.row.update(
            heartbeat_unix_seconds=time.time(),
            invocation_seconds=now - self.started,
            phase_seconds=now - self.phase_started,
        )
        write_json(self.path, self.row)

    def _heartbeat(self):
        while not self.finished.wait(self.interval):
            try:
                with self.lock:
                    self._write()
                    if self.rank == 0:
                        print(
                            f"[SuperMimic] {self.row['phase']} "
                            f"step={self.row.get('control_steps', 0)} "
                            f"updates={self.row.get('learner_updates', 0)} "
                            f"phase={self.row['phase_seconds']:.0f}s",
                            flush=True,
                        )
            except Exception as error:
                self.error = error
                return

    def close(self):
        self.finished.set()
        self.thread.join()


class MetricsWriter:
    """Write control-step metrics to JSONL and optional TensorBoard events."""

    def __init__(self, output, *, tensorboard=True):
        self.output = Path(output)
        self.writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(str(self.output / "tensorboard"))

    def write(self, row):
        with (self.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
        if self.writer is not None:
            for key, value in row.items():
                if isinstance(value, (float, int)):
                    self.writer.add_scalar(key, value, row["control_steps"])
            for key in ("episode_return", "Policy/reward", "Loss/state"):
                if key in row:
                    self.writer.add_scalar(
                        "WallTime/" + key, row[key], int(row["elapsed_seconds"])
                    )
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
