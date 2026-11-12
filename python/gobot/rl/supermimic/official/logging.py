"""Live progress plus an RSL-RL log-writer adapter for the WM/BPTT clock."""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class Progress:
    """Heartbeat during data loading/compilation, without GPU or MPI thread work."""
    def __init__(self, output, rank, interval=10.):
        self.output = Path(output)
        self.rank = rank
        self.path = self.output / ('status.json' if rank == 0 else f'status.rank{rank}.json')
        self.interval = interval
        self.started = time.monotonic()
        self.phase_started = self.started
        self.row = {'state': 'starting', 'phase': 'initializing', 'rank': rank}
        self.lock = threading.Lock()
        self.finished = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.update()
        self.thread.start()

    def update(self, **values):
        with self.lock:
            if values.get('phase', self.row['phase']) != self.row['phase']:
                self.phase_started = time.monotonic()
            self.row.update(values)
            self._write()
        if self.error is not None:
            raise RuntimeError(f'progress writer failed: {self.error}')

    def _write(self):
        now = time.monotonic()
        self.row.update(heartbeat_unix_seconds=time.time(),
                        invocation_seconds=now - self.started,
                        phase_seconds=now - self.phase_started)
        write_json(self.path, self.row)

    def _heartbeat(self):
        while not self.finished.wait(self.interval):
            try:
                with self.lock:
                    self._write()
                    if self.rank == 0:
                        print(f"[SuperMimic] {self.row['phase']} "
                              f"step={self.row.get('control_steps', 0)} "
                              f"updates={self.row.get('learner_updates', 0)} "
                              f"phase={self.row['phase_seconds']:.0f}s", flush=True)
            except Exception as error:
                self.error = error
                return

    def close(self):
        self.finished.set()
        self.thread.join()


class MetricsWriter:
    """Use RSL-RL's writer lifecycle without its PPO iteration accounting.

    Only rank zero constructs this adapter. JSONL is always available; RSL's
    TensorBoard writer is optional. The official algorithm remains WM + BPTT.
    """
    def __init__(self, output, cfg, config, world_size):
        self.output = Path(output)
        self.logger = None
        if cfg.tensorboard:
            from rsl_rl.utils.logger import Logger
            self.logger = Logger(str(self.output / 'tensorboard'),
                                 {'logger': 'tensorboard', 'algorithm': {}, 'num_steps_per_env': 1},
                                 config, cfg.num_envs, world_size > 1, world_size, 0, cfg.device)
            self.logger.git_status_repos = [__file__]
            self.logger.init_logging_writer()

    def write(self, row):
        with (self.output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)
        writer = None if self.logger is None else self.logger.writer
        if writer is not None:
            for key, value in row.items():
                if isinstance(value, (float, int)):
                    writer.add_scalar(key, value, row['control_steps'])
            for key in ('episode_return', 'Policy/reward', 'Loss/state'):
                if key in row:
                    writer.add_scalar('WallTime/' + key, row[key], int(row['elapsed_seconds']))
            writer.flush()

    def close(self):
        if self.logger is not None:
            writer = self.logger.writer
            if writer is not None and hasattr(writer, 'close'):
                writer.close()
            self.logger.stop_logging_writer()
