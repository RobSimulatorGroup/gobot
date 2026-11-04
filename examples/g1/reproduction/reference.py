"""Freeze and verify the independent SuperMimic result baseline.

All downloads live in an ignored experiment directory. This tool reads them;
it does not import either simulation framework or deserialize checkpoints.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

REFERENCE_COMMIT = 'ba6105349069bf040f1fdad60873d13138d505a2'
MJLAB_COMMIT = '8bb2bc9e187797689ec82c78c35019afd2fd0376'
LOCK = Path(__file__).with_name('reference.lock.json')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return 'sha256:' + value.hexdigest()


def verify_checkout(path, revision):
    path = Path(path).resolve()
    head = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if head != revision:
        raise ValueError(f'reference revision differs: {head} != {revision}')
    changed = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if changed:
        raise ValueError(f'reference tracked files are modified: {path}')
    names = subprocess.check_output(['git', '-C', str(path), 'ls-files', '-z']).decode().split('\0')
    return {name: digest(path/name) for name in names if name and (path/name).is_file()}


def verify_release(directory, records):
    verified = {}
    for row in records:
        path = Path(directory) / row['name']
        if path.stat().st_size != row['size'] or digest(path) != row['sha256']:
            raise ValueError(f'release asset differs from the pinned bytes: {path}')
        verified[row['name']] = row['sha256']
    return verified


def benchmark_summary(rows):
    if not rows or len({row['name'] for row in rows}) != len(rows):
        raise ValueError('benchmark must contain unique, nonempty clip results')
    if any(not isinstance(row['steps'], int) or isinstance(row['steps'], bool) or row['steps'] < 1
           or not isinstance(row['failed'], bool) or not math.isfinite(row['norm_return']) for row in rows):
        raise ValueError('benchmark has invalid step counts or failure flags')
    steps = sum(row['steps'] for row in rows)
    return {'clips': len(rows), 'failed': sum(row['failed'] for row in rows),
            'mean_norm_return': sum(row['norm_return'] for row in rows)/len(rows),
            'step_weighted_norm_return': sum(row['norm_return']*row['steps'] for row in rows)/steps}


def filtered_evaluation(rows, reference):
    """Recover the published evaluation subset, never infer training membership."""
    path = Path(reference)/'scripts/data/filter_bones_seed.py'
    groups = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('SONIC_KEYWORDS', 'SONIC_ACROBATICS', 'PAPER_EXTRA'):
                groups[name] = ast.literal_eval(node.value)
    if len(groups) != 3 or any(not isinstance(v, list) for v in groups.values()):
        raise ValueError('pinned evaluation filter layout changed')
    keywords = sorted({word for values in groups.values() for word in values})
    kept = [row for row in rows if not any(word in row['name'].lower() for word in keywords)]
    return kept, {'filter_sha256': digest(path), 'keywords': keywords,
                  'scope': 'name-filtered subset of published already-feasible evaluation clips'}


def freeze(reference, mjlab, release, output):
    lock = json.loads(LOCK.read_text())
    sources = {'supermimic': verify_checkout(reference, REFERENCE_COMMIT),
               'mjlab': verify_checkout(mjlab, MJLAB_COMMIT)}
    assets = verify_release(release, lock['release']['assets'])
    config = json.loads((Path(release)/'config.json').read_text())
    with gzip.open(Path(release)/'e8_boneseed_benchmark_full.json.gz', 'rt') as stream:
        benchmark = json.load(stream)
    summary = benchmark_summary(benchmark['clips'])
    filtered, filter_identity = filtered_evaluation(benchmark['clips'], reference)
    filtered_summary = benchmark_summary(filtered)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    archives = {}
    for name, source in (('supermimic', reference), ('mjlab', mjlab)):
        archive = output/f'{name}-source.tar.gz'
        with archive.open('wb') as stream:
            subprocess.run(['git', '-C', str(Path(source).resolve()), 'archive', '--format=tar.gz', 'HEAD'],
                           stdout=stream, check=True)
        archives[name] = digest(archive)
    # Published config describes a resumed segment and already-scaled learning rates.
    report = {'protocol': 'supermimic_reproduced_baseline_v1', 'source_files': sources,
              'source_archives': archives, 'release_assets': assets, 'published_config': config,
              'published_benchmark_recomputed': summary,
              'published_benchmark_summary': benchmark['summary'],
              'filtered_evaluation_summary': filtered_summary, 'filter_identity': filter_identity,
              'training_manifest_available': False,
              'notes': lock['experiment_notes'], 'physics_baselines': lock['physics_baselines']}
    (output/'baseline.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (output/'evaluation_clips.json').write_text(json.dumps(
        [row['name'] for row in benchmark['clips']], indent=2)+'\n')
    (output/'filtered_evaluation_clips.json').write_text(json.dumps(
        [row['name'] for row in filtered], indent=2)+'\n')
    return {'release_verified': True, 'summary': summary, 'filtered_summary': filtered_summary,
            'training_manifest_available': False, 'output': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--mjlab', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.reference, args.mjlab, args.release, args.output), indent=2))


if __name__ == '__main__':
    main()
