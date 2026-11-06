"""Rebuild BONES training/evaluation manifests and a scene-owned compact library.

The historical e8 training filter and the later evaluation filter are independent
inputs with pinned hashes. Selection uses the untouched reference converter and
standalone reference G1 model. Training data uses Gobot's existing scene converter.
Published evaluation names are compared only after selection; they never select
training clips. Completed, content-checked shards can be resumed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import tempfile
import time

REFERENCE_COMMIT = 'ba6105349069bf040f1fdad60873d13138d505a2'
TRAIN_FILTER_COMMIT = 'd9be385b3eeb13a638fb33fb200d3bfd9be9839f'
MJLAB_COMMIT = '8bb2bc9e187797689ec82c78c35019afd2fd0376'
REFERENCE_HASHES = {
    'filter_train.py': '4a35a2ee49af18b0c0af91a93006eb0219f5f178c83570a5ccc5eb4d6e0c2715',
    'filter_eval.py': '862636774444ca2331c3b3e054e45398f7c74854172242111fdb4bc128c9d463',
    'motion_convert.py': '032047915b7dde7b348e72e5dca1529cee15d7259800d5d81144a5c9a82f5bc1',
}
PARITY_LIMITS = {'position_m': 2e-5, 'quaternion': 2e-5,
                 'linear_velocity_m_s': 1e-4, 'angular_velocity_rad_s': 1e-4,
                 'joint_position_rad': 1e-6, 'joint_velocity_rad_s': 1e-4}
PARITY_RTOL = 1e-5
_WORKER = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def inventory(root):
    files = sorted(Path(root).rglob('*.csv'), key=lambda p: p.stem)
    names = [p.stem for p in files]
    if not files or len(set(names)) != len(names):
        raise ValueError('BONES input must contain nonempty, globally unique clip names')
    return files


def load_reference_module(root, filename):
    path = Path(root) / filename
    if digest(path) != REFERENCE_HASHES[filename]:
        raise ValueError(f'reference source hash differs: {path}')
    name = '_bones_reference_' + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference_modules(root):
    train = load_reference_module(root, 'filter_train.py')
    evaluation = load_reference_module(root, 'filter_eval.py')
    motion = load_reference_module(root, 'motion_convert.py')
    if not set(train.KEYWORDS) <= set(evaluation.KEYWORDS):
        raise ValueError('evaluation name filter must extend the historical training filter')
    for name in ('feasibility_filter', '_foot_sole_height', '_longest_run'):
        if inspect.getsource(getattr(train, name)) != inspect.getsource(getattr(evaluation, name)):
            raise ValueError(f'reference feasibility implementations differ: {name}')
    return train, evaluation, motion


def initialize_worker(project, reference, plan_hash):
    global _WORKER
    import torch
    import mujoco
    torch.set_num_threads(1)
    train, evaluation, motion = reference_modules(reference)
    model = motion.load_g1_model(Path(reference) / 'g1/g1.xml')
    foot = [i for i in range(model.ngeom)
            if re.fullmatch(r'(left|right)_foot[1-7]_collision', model.geom(i).name or '')]
    if len(foot) != 14 or any(model.geom_type[i] != mujoco.mjtGeom.mjGEOM_CAPSULE for i in foot):
        raise ValueError('reference foot collision geometry differs from the fixed G1 model')
    # Supply the exact model that the reference cache normally loads via mjlab.
    # No filtering or motion-conversion functions are replaced.
    train._MODEL, train._FOOT_GEOMS = model, foot
    _WORKER = (Path(project), train, evaluation, motion, model, plan_hash)


def compact_reference(npz):
    import numpy as np
    with np.load(npz, allow_pickle=False) as data:
        root = list(data['body_names']).index('pelvis')
        result = np.concatenate((data['body_pos_w'][:, root], data['body_quat_w'][:, root],
            data['joint_pos'], data['body_lin_vel_w'][:, root], data['body_ang_vel_w'][:, root],
            data['joint_vel']), axis=1).astype(np.float16)
    if result.shape[1] != 71 or not np.isfinite(result).all():
        raise ValueError(f'nonfinite or invalid compact reference: {npz}')
    return result


def conversion_error(actual_path, reference_path):
    import numpy as np
    with np.load(actual_path, allow_pickle=False) as actual, np.load(reference_path, allow_pickle=False) as ref:
        names, other = list(actual['body_names']), list(ref['body_names'])
        if set(names) != set(other):
            raise ValueError('Gobot and reference body names differ')
        order = [other.index(name) for name in names]
        a, b = actual['body_quat_w'], ref['body_quat_w'][:, order]
        b = np.where((a * b).sum(-1, keepdims=True) < 0, -b, b)
        pairs = {'quaternion': (a, b)}
        for key, field in [('position_m', 'body_pos_w'), ('linear_velocity_m_s', 'body_lin_vel_w'),
                           ('angular_velocity_rad_s', 'body_ang_vel_w')]:
            pairs[key] = (actual[field], ref[field][:, order])
        for key, field in [('joint_position_rad', 'joint_pos'), ('joint_velocity_rad_s', 'joint_vel')]:
            pairs[key] = (actual[field], ref[field])
        return {
            'max_abs_error': {key: float(np.max(np.abs(x - y))) for key, (x, y) in pairs.items()},
            'max_tolerance_ratio': {key: float(np.max(np.abs(x - y) / (PARITY_LIMITS[key] + PARITY_RTOL * np.abs(y))))
                                    for key, (x, y) in pairs.items()},
        }


def validate_shard(path, sources, plan_hash):
    result_path = Path(path) / 'result.json'
    if not result_path.exists():
        return None
    result = json.loads(result_path.read_text())
    if result['plan_sha256'] != plan_hash or [r['name'] for r in result['rows']] != [p.stem for p in sources]:
        raise ValueError(f'shard identity differs: {path}')
    for row, source in zip(result['rows'], sources):
        if digest(source) != row['source_sha256'] or digest(Path(path) / (source.stem + '.npz')) != row['npz_sha256']:
            raise ValueError(f'shard content changed: {source.stem}')
    if digest(Path(path) / 'references.npy') != result['references_sha256']:
        raise ValueError(f'compact shard changed: {path}')
    return result


def process_shard(job):
    import numpy as np
    from gobot.rl.motion_tracking.convert import convert
    path, sources = Path(job[0]), [Path(p) for p in job[1]]
    project, train, evaluation, motion, model, plan_hash = _WORKER
    result = validate_shard(path, sources, plan_hash)
    if result is not None:
        return str(path)
    path.mkdir(parents=True, exist_ok=True)
    converted = convert(project, sources, path, input_fps=120., compact=False)
    rows, refs, comparisons, disagreements = [], [], [], []
    with tempfile.TemporaryDirectory(prefix='reference-', dir=path) as temporary:
        reference_path = Path(temporary) / 'motion.npz'
        for source, info in zip(sources, converted['clips']):
            candidate = path / (source.stem + '.npz')
            reason = train.name_filter(source.stem)
            if reason is None:
                independent = motion.convert_csv_to_npz(source, reference_path, 120., model=model)
                if independent['output_frames'] != info['frames']:
                    raise ValueError(f'reference frame count differs: {source}')
                reason = train.feasibility_filter(str(reference_path))
                candidate_reason = train.feasibility_filter(str(candidate))
                if candidate_reason != reason:
                    disagreements.append({'name': source.stem, 'reference_reason': reason,
                                          'gobot_reason': candidate_reason})
                if len(comparisons) < 2:
                    comparisons.append({'name': source.stem, **conversion_error(candidate, reference_path)})
            eval_reason = evaluation.name_filter(source.stem) or reason
            row = dict(info, keep=int(reason is None), reason=reason or '',
                       eval_keep=int(eval_reason is None), eval_reason=eval_reason or '')
            rows.append(row)
            if row['keep']:
                refs.append(compact_reference(candidate))
    packed = np.concatenate(refs, axis=0) if refs else np.empty((0, 71), dtype=np.float16)
    with (path / 'references.npy.tmp').open('wb') as stream:
        np.save(stream, packed, allow_pickle=False)
    (path / 'references.npy.tmp').replace(path / 'references.npy')
    result = {'plan_sha256': plan_hash, 'kinematics_digest': converted['kinematics_digest'],
              'rows': rows, 'reference_comparisons': comparisons, 'filter_disagreements': disagreements,
              'references_sha256': digest(path / 'references.npy')}
    atomic_json(path / 'result.json', result)
    return str(path)


def pack_library(shards, target):
    import numpy as np
    import torch
    names, lengths, part_frames = [], [], []
    for path in shards:
        rows = json.loads((Path(path) / 'result.json').read_text())['rows']
        kept = [row for row in rows if row['keep']]
        names.extend(row['name'] for row in kept)
        lengths.extend(row['frames'] for row in kept)
        part_frames.append(sum(row['frames'] for row in kept))
    if not names or names != sorted(names) or len(set(names)) != len(names):
        raise ValueError('packed library requires nonempty unique sorted clip names')
    ref = torch.empty((sum(lengths), 71), dtype=torch.float16)
    view, offset = ref.numpy(), 0
    for path, count in zip(shards, part_frames):
        data = np.load(Path(path) / 'references.npy', mmap_mode='r', allow_pickle=False)
        if data.shape != (count, 71) or data.dtype != np.float16 or not np.isfinite(data).all():
            raise ValueError(f'bad compact shard: {path}')
        view[offset:offset + count] = data
        offset += count
    starts = np.cumsum([0] + lengths[:-1], dtype=np.int64).tolist()
    temporary = Path(target).with_suffix('.pt.tmp')
    torch.save({'ref': ref, 'names': names, 'starts': starts, 'lengths': lengths}, temporary)
    temporary.replace(target)
    del view, ref
    loaded = torch.load(target, map_location='cpu', weights_only=True, mmap=True)
    if loaded['names'] != names or loaded['starts'] != starts or loaded['lengths'] != lengths:
        raise ValueError('packed metadata did not round trip')
    offset = 0
    for path, count in zip(shards, part_frames):
        data = np.load(Path(path) / 'references.npy', mmap_mode='r', allow_pickle=False)
        if not np.array_equal(loaded['ref'][offset:offset + count].numpy(), data):
            raise ValueError(f'packed reference differs from {path}')
        offset += count
    return {'clips': len(names), 'frames': offset, 'sha256': digest(target),
            'bytes': Path(target).stat().st_size, 'dtype': 'float16', 'state_dim': 71}


def write_manifests(shards, output, kinematics_digest):
    output = Path(output)
    dense = output / 'npz'
    dense.mkdir(exist_ok=True)
    train_names, eval_names, comparisons, disagreements = [], [], [], []
    reasons, eval_reasons = Counter(), Counter()
    raw_count = frames = 0
    with (output / 'manifest_all.jsonl.tmp').open('w') as ledger, \
         (output / 'train_manifest.csv.tmp').open('w', newline='') as training, \
         (output / 'eval_manifest.csv.tmp').open('w', newline='') as evaluation:
        tw, ew = csv.writer(training), csv.writer(evaluation)
        tw.writerow(['name', 'frames', 'keep', 'reason'])
        ew.writerow(['name', 'frames', 'keep', 'reason'])
        for path in shards:
            part = json.loads((Path(path) / 'result.json').read_text())
            if part['kinematics_digest'] != kinematics_digest:
                raise ValueError('scene kinematics changed between shards')
            comparisons.extend(part['reference_comparisons'])
            disagreements.extend(part['filter_disagreements'])
            for row in part['rows']:
                ledger.write(json.dumps(row, allow_nan=False) + '\n')
                tw.writerow([row['name'], row['frames'], row['keep'], row['reason']])
                ew.writerow([row['name'], row['frames'], row['eval_keep'], row['eval_reason']])
                raw_count += 1
                reasons[row['reason'] or 'keep'] += 1
                eval_reasons[row['eval_reason'] or 'keep'] += 1
                if row['keep']:
                    train_names.append(row['name'])
                    frames += row['frames']
                    source, dest = Path(path) / (row['name'] + '.npz'), dense / (row['name'] + '.npz')
                    if dest.exists():
                        if not os.path.samefile(source, dest):
                            raise ValueError(f'dense output is not the prepared shard: {dest}')
                    else:
                        os.link(source, dest)
                if row['eval_keep']:
                    eval_names.append(row['name'])
    for name in ['manifest_all.jsonl', 'train_manifest.csv', 'eval_manifest.csv']:
        (output / (name + '.tmp')).replace(output / name)
    if {p.stem for p in dense.glob('*.npz')} != set(train_names):
        raise ValueError('dense training directory contains missing or extra clips')
    atomic_json(output / 'train_clips.json', train_names)
    atomic_json(output / 'eval_clips.json', eval_names)
    return {'raw_clips': raw_count, 'train_clips': len(train_names), 'eval_clips': len(eval_names),
            'train_frames': frames, 'train_reasons': dict(reasons), 'eval_reasons': dict(eval_reasons),
            'reference_comparisons': comparisons, 'filter_disagreements': disagreements}


def compare_membership(actual, expected):
    a, b = set(actual), set(expected)
    return {'matches': a == b, 'actual_count': len(a), 'expected_count': len(b),
            'missing': sorted(b - a), 'extra': sorted(a - b)}


def run(args):
    import torch
    import gobot
    from gobot.rl.kinematics import SceneKinematics
    from gobot.rl.motion_tracking.identity import source_identity
    started = time.monotonic()
    project, root, output, reference = [p.resolve() for p in (args.project, args.input, args.output, args.reference)]
    if args.workers < 1 or args.shard_size < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError('workers, shard size and optional limit must be positive')
    files = inventory(root)
    total_available = len(files)
    if args.limit is None and total_available != 142220:
        raise ValueError(f'expected 142220 raw BONES G1 clips, found {total_available}')
    if args.limit is not None:
        files = files[:args.limit]
    train, evaluation, _ = reference_modules(reference)
    manifest = json.loads((reference / 'source_manifest.json').read_text())
    if manifest['model']['mjlab_commit'] != MJLAB_COMMIT:
        raise ValueError('reference G1 model revision differs')
    for filename, expected in manifest['model']['files'].items():
        if digest(reference / 'g1' / filename) != expected:
            raise ValueError(f'reference model file changed: {filename}')
    profile = json.loads((project / 'assets/supermimic/robot_profile.json').read_text())
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(project))
        context.load_scene('res://g1_29dof_flat_scene.jscn')
        kin = SceneKinematics(context.compile_physics_scene_snapshot(), robot_name=profile['robot_name'],
            base_link=profile['base_link'], joint_names=profile['joint_names'], dtype=torch.float64)
        kinematics_digest = kin.digest
    finally:
        context.clear_scene()
    inventory_hash = hashlib.sha256()
    for path in files:
        st = path.stat()
        inventory_hash.update(json.dumps([str(path.relative_to(root)), st.st_size, st.st_mtime_ns]).encode() + b'\n')
    plan = {'protocol': 'gobot_bones_e8_data_v1', 'input_root': str(root), 'project': str(project),
            'reference_root': str(reference), 'reference_commit': REFERENCE_COMMIT,
            'train_filter_commit': TRAIN_FILTER_COMMIT, 'mjlab_commit': MJLAB_COMMIT,
            'reference_hashes': REFERENCE_HASHES, 'model_files': manifest['model']['files'],
            'source_identity': source_identity(), 'preparer_sha256': digest(__file__),
            'kinematics_digest': kinematics_digest, 'input_fps': 120, 'output_fps': 50,
            'available_clips': total_available, 'selected_raw_clips': len(files),
            'pilot': args.limit is not None, 'shard_size': args.shard_size,
            'inventory_stat_sha256': inventory_hash.hexdigest(),
            'train_keywords': train.KEYWORDS, 'eval_keywords': evaluation.KEYWORDS,
            'parity_limits': PARITY_LIMITS, 'parity_rtol': PARITY_RTOL,
            'published_evaluation_hashes': {name: digest(reference / name) for name in
                ('evaluation_clips.json', 'filtered_evaluation_clips.json')},
            'software': {name: importlib.metadata.version(name) for name in
                         ('gobot', 'torch', 'mujoco', 'numpy', 'scipy', 'pandas')}}
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / 'preparation.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('existing preparation plan differs; use a new output directory')
    atomic_json(plan_path, plan)
    plan_hash = digest(plan_path)
    initialize_worker(str(project), str(reference), plan_hash)
    name_kept = sum(train.name_filter(p.stem) is None for p in files)
    if args.limit is None and name_kept != 130137:
        raise ValueError(f'historical name-filter count differs: {name_kept}, expected 130137')
    jobs = [(str(output / 'parts' / f'{i // args.shard_size:05d}'), [str(p) for p in files[i:i + args.shard_size]])
            for i in range(0, len(files), args.shard_size)]
    atomic_json(output / 'status.json', {'stage': 'converting', 'completed_shards': 0, 'total_shards': len(jobs)})
    print(json.dumps({'raw_clips': len(files), 'shards': len(jobs), 'workers': args.workers,
                      'name_filter_kept': name_kept}), flush=True)
    with mp.get_context('spawn').Pool(args.workers, initializer=initialize_worker,
            initargs=(str(project), str(reference), plan_hash)) as pool:
        for complete, path in enumerate(pool.imap_unordered(process_shard, jobs), 1):
            status = {'stage': 'converting', 'completed_shards': complete, 'total_shards': len(jobs),
                      'elapsed_seconds': time.monotonic() - started, 'last_shard': path}
            atomic_json(output / 'status.json', status)
            print(json.dumps(status), flush=True)
    shards = [Path(job[0]) for job in jobs]
    result = write_manifests(shards, output, kinematics_digest)
    maxima = {key: max((r['max_abs_error'][key] for r in result['reference_comparisons']), default=0.)
              for key in PARITY_LIMITS}
    ratios = {key: max((r['max_tolerance_ratio'][key] for r in result['reference_comparisons']), default=0.)
              for key in PARITY_LIMITS}
    result['conversion_parity'] = {'samples': len(result.pop('reference_comparisons')), 'max_abs_error': maxima,
        'limits': PARITY_LIMITS, 'rtol': PARITY_RTOL, 'max_tolerance_ratio': ratios,
        'passed': all(value <= 1. for value in ratios.values())}
    atomic_json(output / 'status.json', {'stage': 'packing', 'elapsed_seconds': time.monotonic() - started})
    result['library'] = pack_library(shards, output / 'library.pt')
    result['protocol'] = plan['protocol']
    result['plan_sha256'] = plan_hash
    result['artifacts'] = {name: digest(output / name) for name in
        ['manifest_all.jsonl', 'train_manifest.csv', 'eval_manifest.csv', 'train_clips.json', 'eval_clips.json']}
    result['dataset_identity'] = 'reconstructed from raw CSV and historical reference filter; original training library bytes unavailable'
    if args.limit is None:
        result['published_membership_comparison'] = {
            'full_evaluation': compare_membership(json.loads((output / 'train_clips.json').read_text()),
                                                  json.loads((reference / 'evaluation_clips.json').read_text())),
            'filtered_evaluation': compare_membership(json.loads((output / 'eval_clips.json').read_text()),
                                                  json.loads((reference / 'filtered_evaluation_clips.json').read_text()))}
    result['elapsed_seconds'] = time.monotonic() - started
    result['pilot'] = args.limit is not None
    result['complete'] = True
    result['ready_for_paired_training'] = (args.limit is None and result['conversion_parity']['passed']
        and all(r['matches'] for r in result.get('published_membership_comparison', {}).values()))
    atomic_json(output / 'manifest.json', result)
    atomic_json(output / 'status.json', {'stage': 'complete', 'raw_clips': result['raw_clips'],
        'train_clips': result['train_clips'], 'eval_clips': result['eval_clips'],
        'train_frames': result['train_frames'], 'elapsed_seconds': result['elapsed_seconds']})
    print(json.dumps({k: v for k, v in result.items() if k != 'filter_disagreements'}, indent=2), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'input', 'output', 'reference'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--shard-size', type=int, default=256)
    p.add_argument('--limit', type=int, help='pilot only; final report is explicitly marked as a pilot')
    args = p.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.output.exists():
            atomic_json(args.output / 'status.json', {'stage': 'failed', 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
