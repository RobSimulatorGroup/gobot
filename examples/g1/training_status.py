"""Print recoverable G1 experiment status, including after a reboot."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log-dir', type=Path, default=Path('examples/g1/logs'))
    args = p.parse_args()
    budget_path = args.log_dir / 'gpu_budget.json'
    budget = json.loads(budget_path.read_text()) if budget_path.exists() else {'seconds': 0.}
    print(f'GPU budget used: {budget["seconds"] / 3600:.2f} / 24 h')
    for directory in sorted(args.log_dir.iterdir()):
        metrics = directory / 'metrics.jsonl'
        if not directory.is_dir() or not metrics.exists():
            continue
        records = metrics.read_text().splitlines()
        last = next((json.loads(line) for line in reversed(records) if line.strip().endswith('}')), {})
        evaluations = sorted(directory.glob('eval_*.json'))
        latest = json.loads(evaluations[-1].read_text()) if evaluations else {}
        print(json.dumps({'run': directory.name, 'last_logged_iteration': last.get('iteration'),
            'checkpoint': str(directory / 'latest.pt') if (directory / 'latest.pt').exists() else None,
            'eval_admission': latest.get('admission'), 'eval_velocity_error': latest.get('velocity_error'),
            'finished': (directory / 'summary.json').exists()}, ensure_ascii=False))

if __name__ == '__main__': main()
