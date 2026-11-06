"""Export fixed reference inputs for prepare_bones without importing simulators.

Reads Git objects at pinned commits, not mutable working-tree files. The baseline
must come from the reference command's verified release. This is preparation only;
it never starts training, evaluation or a download.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .prepare_bones import (
    MJLAB_COMMIT, REFERENCE_COMMIT, REFERENCE_HASHES, TRAIN_FILTER_COMMIT,
)
from .evaluate_bones import atomic_json, digest

EVALUATION_HASHES = {
    'evaluation_clips.json': 'dab36886f96f21284a0756ff3c6f7fc6e10ea46511a6d94342bfd9cb0b51921b',
    'filtered_evaluation_clips.json': 'c66f6b9af9ac38592722cb749c8777df266c4beeab4b05ed679de1880143019b',
}


def git_bytes(repository, revision, name):
    return subprocess.check_output(['git', '-C', str(repository), 'show', f'{revision}:{name}'])


def write_verified(path, content, expected=None):
    actual = hashlib.sha256(content).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError(f'fixed reference hash differs: {path.name}')
    if path.exists():
        if digest(path) != actual:
            raise ValueError(f'refusing to replace changed reference input: {path}')
        return actual
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(content)
    temporary.replace(path)
    return actual


def export(reference, mjlab, baseline, output):
    reference, mjlab, baseline, output = map(Path, (reference, mjlab, baseline, output))
    # Verify the comparison lists before writing anything. They are not used to
    # select training clips; prepare_bones reconstructs selection independently.
    for name, expected in EVALUATION_HASHES.items():
        if digest(baseline / name) != expected:
            raise ValueError(f'published evaluation list differs: {name}')
    sources = (
        ('filter_train.py', TRAIN_FILTER_COMMIT, 'scripts/data/filter_bones_seed.py'),
        ('filter_eval.py', REFERENCE_COMMIT, 'scripts/data/filter_bones_seed.py'),
        ('motion_convert.py', REFERENCE_COMMIT, 'src/supermimic/data/motion_convert.py'),
        ('LICENSE', REFERENCE_COMMIT, 'LICENSE'),
    )
    manifest = {}
    for target, commit, source in sources:
        content = git_bytes(reference, commit, source)
        sha = write_verified(output / target, content, REFERENCE_HASHES.get(target))
        manifest[target] = {'commit': commit, 'source': source, 'sha256': sha}
    model_root = 'src/mjlab/asset_zoo/robots/unitree_g1/xmls/'
    tree = subprocess.check_output([
        'git', '-C', str(mjlab), 'ls-tree', '-r', '--name-only', '-z', MJLAB_COMMIT, '--', model_root,
    ]).decode().split('\0')
    model_files = {}
    for source in sorted(name for name in tree if name):
        relative = Path(source).relative_to(model_root)
        content = git_bytes(mjlab, MJLAB_COMMIT, source)
        model_files[relative.as_posix()] = write_verified(output / 'g1' / relative, content)
    if 'g1.xml' not in model_files:
        raise ValueError('pinned reference has no G1 model')
    manifest['model'] = {'mjlab_commit': MJLAB_COMMIT, 'files': model_files}
    manifest['evaluation_files'] = EVALUATION_HASHES
    for name, expected in EVALUATION_HASHES.items():
        write_verified(output / name, (baseline / name).read_bytes(), expected)
    atomic_json(output / 'source_manifest.json', manifest)
    return {'output': str(output), 'model_files': len(model_files),
            'reference_commit': REFERENCE_COMMIT, 'train_filter_commit': TRAIN_FILTER_COMMIT}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'mjlab', 'baseline', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.reference, args.mjlab, args.baseline, args.output), indent=2))


if __name__ == '__main__':
    main()
