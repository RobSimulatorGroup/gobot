"""Shared local allocation for training, assessment and recording entry points."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import threading

from gobot.rl.locomotion.budget import TrainingBudget, effective_cap_hours

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / 'examples/g1/logs/gpu_budget.json'


def add_budget_arguments(parser):
    parser.add_argument('--budget-file', type=Path, default=DEFAULT_LEDGER)
    parser.add_argument('--budget-hours', type=float, default=24.)
    parser.add_argument('--hours', type=float, default=.5)
    parser.add_argument('--budget-parent-pid', type=int)


@contextmanager
def evaluation_budget(args, *, stage):
    """A child shares its direct parent's allocation; standalone work owns one."""
    if args.budget_parent_pid is not None:
        data = json.loads(args.budget_file.read_text())
        active = data.get('active_run', {})
        if (args.budget_parent_pid != os.getppid() or active.get('pid') != args.budget_parent_pid
                or active.get('host') != socket.gethostname()):
            raise ValueError('evaluation must share the active direct parent budget')
        yield
        return
    with TrainingBudget(args.budget_file, total_hours=args.budget_hours, run_hours=args.hours,
                        metadata={'stage': stage}) as budget:
        finished = threading.Event()

        def watch():
            while not finished.wait(min(1., max(.01, budget.remaining))):
                budget.update()
                if budget.remaining <= 0:
                    os.kill(os.getpid(), signal.SIGINT)
                    # Native compilation may delay Python signal delivery.
                    if not finished.wait(15.):
                        os.kill(os.getpid(), signal.SIGTERM)
                    return

        worker = threading.Thread(target=watch, daemon=True)
        worker.start()
        try:
            yield
        finally:
            finished.set()
            worker.join(timeout=2.)
