"""Run bounded G1 training stages and admission without exceeding one GPU-day."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import time
from gobot.rl.locomotion.budget import TrainingBudget


@dataclass(frozen=True)
class TrainingStage:
    name: str
    hours: float


STAGES = (TrainingStage('flat', 5.), TrainingStage('rough', 10.), TrainingStage('mixed', 6.))


def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def run_logged(command, log, timeout):
    if timeout <= 0: raise TimeoutError('GPU budget exhausted')
    with log.open('a', buffering=1) as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try: process.wait(timeout=60)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise TimeoutError(f'job exceeded its budget; see {log}')
    if code:
        raise RuntimeError(f'job exited {code}; see {log}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('examples/g1/logs/campaign'))
    p.add_argument('--budget-file', type=Path, default=Path('examples/g1/logs/gpu_budget.json'))
    p.add_argument('--budget-hours', type=float, default=24.)
    p.add_argument('--num-envs', type=int, default=1024)
    p.add_argument('--resume', type=Path)
    p.add_argument('--start-stage', choices=('flat','rough','mixed'), default='flat')
    args = p.parse_args()
    if not 0 < args.budget_hours <= 24: raise ValueError('budget must be in (0, 24] GPU-hours')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'complete': False, 'budget_hours': args.budget_hours, 'stages': []}
    checkpoint = args.resume
    def remaining():
        return args.budget_hours * 3600 - read_json(args.budget_file, {'seconds': 0})['seconds']
    try:
        for stage in STAGES[[stage.name for stage in STAGES].index(args.start_stage):]:
            hours = min(stage.hours, max(0., remaining() / 3600 - 2.))
            if hours <= 0: raise TimeoutError('remaining budget reserved for final admission')
            directory = args.output / stage.name
            command = [sys.executable, '-B', '-m', 'examples.g1.train_amp', '--stage', stage.name,
                '--num-envs', str(args.num_envs), '--iterations', '1000000', '--hours', str(hours),
                '--log-dir', str(directory), '--budget-file', str(args.budget_file),
                '--budget-hours', str(args.budget_hours)]
            if checkpoint: command += ['--resume', str(checkpoint)]
            run_logged(command, args.output / f'{stage.name}.log', hours * 3600 + 120)
            result = read_json(directory / 'best_eval.json', {})
            report['stages'].append({'stage': stage.name, 'screening': result})
            write_json(args.output / 'campaign.json', report)
            if result.get('admission', 0) < .9:
                raise RuntimeError(f'{stage.name} has not passed the 90% screening gate; preserve it for diagnosis')
            checkpoint = directory / 'best.pt'
        # Admission owns its own bounded ledger entry. Each subprocess uses the exact candidate.
        evaluation_budget = TrainingBudget(args.budget_file, total_hours=args.budget_hours,
            run_hours=max(remaining() / 3600, .0001), metadata={'stage': 'admission', 'log_dir': str(args.output)})
        files = []
        try:
            for seed in (101, 202, 303):
                for label, command in (('walk05', (.5,0,0)), ('walk10', (1.,0,0)),
                        ('back', (-.3,0,0)), ('left', (0,.3,0)), ('right', (0,-.3,0)),
                        ('stand', (0,0,0)), ('yaw_left', (0,0,.6)), ('yaw_right', (0,0,-.6))):
                    output = args.output / f'{label}_{seed}.json'
                    speed = (command[0]**2 + command[1]**2)**.5
                    steps = max(1000, int((6 / speed + 4) / .02)) if speed else 1000
                    job = [sys.executable, '-B', '-m', 'examples.g1.evaluate_amp', '--checkpoint', str(checkpoint),
                           '--seed', str(seed), '--steps', str(steps), '--command', *(str(x) for x in command),
                           '--json-out', str(output)]
                    run_logged(job, args.output / 'admission.log', evaluation_budget.remaining)
                    files.append(output)
                for slope in (0., 5.):
                    for speed in ((.8, 1.2, 1.8, 2.) if slope == 0 else (1.8, 2.)):
                        output = args.output / f'run{speed:g}_slope{slope:g}_{seed}.json'
                        job = [sys.executable, '-B', '-m', 'examples.g1.evaluate_amp', '--checkpoint', str(checkpoint),
                               '--seed', str(seed), '--continuous', '--slope-degrees', str(slope),
                               '--command', str(speed), '0', '0', '--json-out', str(output)]
                        run_logged(job, args.output / 'admission.log', evaluation_budget.remaining)
                        files.append(output)
        finally:
            evaluation_budget.close()
        results = [read_json(path, {}) for path in files]
        report['checkpoint'] = str(checkpoint)
        report['admission_files'] = [str(path) for path in files]
        report['mobility_passed'] = all(result.get('admission', 0) >= .95 and
                                 min(cell['admission'] for cell in result['cells']) >= .9 for result in results)
        from .reference_gaits import build_reference, check_natural_gait
        reference = build_reference(Path('examples/g1/assets/motions/locomotion.npz'))
        report['natural_gaits'] = [check_natural_gait(result, reference) for result in results
            if result.get('cell_exit') is False and result['command'][0] > 0 and result.get('slope_degrees') == 0]
        report['gait_reference_passed'] = bool(report['natural_gaits']) and all(row['passed'] for row in report['natural_gaits'])
        report['complete'] = report['mobility_passed'] and report['gait_reference_passed']
        report['visual_review_required'] = True
        if report['complete']:
            run_logged([sys.executable, '-B', '-m', 'examples.g1.verify_export', '--checkpoint', str(checkpoint),
                        '--output', str(args.output / 'g1_velocity.onnx')], args.output / 'export.log', 300)
    except (RuntimeError, TimeoutError) as error:
        report['error'] = str(error)
    finally:
        report['gpu_seconds'] = read_json(args.budget_file, {'seconds': 0})['seconds']
        write_json(args.output / 'campaign.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'stages'}, indent=2))
    if not report['complete']: raise SystemExit(2)

if __name__ == '__main__': main()
