"""Compare native probe records; retain numerical differences without hiding them.

The exact gates cover stochastic/discrete behavior. Continuous physics is
reported separately because independent CUDA contact solves are not bitwise
deterministic, including repeated runs of mjlab itself. These checks do not
establish equal convergence or long-run policy quality.
"""
import argparse
import json
from pathlib import Path

import torch


def compare(candidate, reference):
    assert candidate['revision'] == reference['revision']
    assert candidate['terrain_sha256'] == reference['terrain_sha256']
    assert len(candidate['frames']) == len(reference['frames'])
    exact = ('rng', 'cpu_rng', 'episode_length', 'levels', 'types', 'bias',
             'heading', 'command_time', 'push_time', 'done', 'timeout')
    failures = []
    if not torch.equal(candidate['initialization_rng'], reference['initialization_rng']):
        failures.append('initialization_rng')
    errors = {}
    first_errors = {}
    for index, (actual, expected) in enumerate(zip(candidate['frames'], reference['frames'], strict=True)):
        assert actual.keys() == expected.keys()
        for name, value in actual.items():
            other = expected[name]
            assert value.shape == other.shape, name
            if not torch.isfinite(value).all() or not torch.isfinite(other).all():
                failures.append(f'nonfinite/{index}/{name}')
            delta = (value.double() - other.double()).abs()
            error = float(delta.max())
            errors[name] = max(errors.get(name, 0.), error)
            if index == 1:
                first_errors[name] = error
            if name in exact and not torch.equal(value, other):
                failures.append(f'{index}/{name}')
    return {'reference_revision': reference['revision'], 'frames': len(reference['frames']),
            'terrain_sha256': reference['terrain_sha256'],
            'stochastic_and_discrete_checks_passed': not failures,
            'failures': failures, 'max_absolute_errors': errors,
            'first_step_absolute_errors': first_errors,
            'continuous_trajectories_bitwise_equal': all(v == 0 for v in errors.values()),
            'training_quality_validated': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate', type=Path)
    p.add_argument('reference', type=Path)
    p.add_argument('--reference-repeat', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    load = lambda path: torch.load(path, map_location='cpu', weights_only=True)
    reference = load(args.reference)
    report = compare(load(args.candidate), reference)
    if args.reference_repeat:
        report['reference_repeat'] = compare(load(args.reference_repeat), reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['stochastic_and_discrete_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
