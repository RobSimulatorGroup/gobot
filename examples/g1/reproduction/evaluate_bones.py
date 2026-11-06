"""Bounded-memory, restartable complete-clip evaluation of a fixed BONES policy.

Uses the installed Gobot scene/environment without optimizer or training calls.
Length-bucketed shards keep dense motion libraries bounded. Merge rejects partial,
foreign, duplicate, nonfinite and prematurely truncated results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

PROTOCOL = 'gobot_bones_complete_clip_shards_v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def partition(rows, max_clips, max_frames):
    if max_clips < 1 or max_frames < 1:
        raise ValueError('shard capacities must be positive')
    if not rows or len({r['name'] for r in rows}) != len(rows):
        raise ValueError('expected unique nonempty clip set')
    parts, current, frames = [], [], 0
    for row in sorted(rows, key=lambda r: (r['frames'], r['name']), reverse=True):
        if not isinstance(row['frames'], int) or row['frames'] < 3:
            raise ValueError('clips must have at least three frames')
        if current and (len(current) >= max_clips or frames + row['frames'] > max_frames):
            parts.append(sorted(current, key=lambda r: r['name']))
            current, frames = [], 0
        current.append(row)
        frames += row['frames']
    if current:
        parts.append(sorted(current, key=lambda r: r['name']))
    return parts


def summarize(rows):
    if not rows:
        raise ValueError('empty benchmark')
    n, steps = len(rows), sum(r['steps'] for r in rows)
    return {'clips': n, 'failed': sum(r['failed'] for r in rows),
            'failure_rate': sum(r['failed'] for r in rows) / n,
            'mean_norm_return': sum(r['norm_return'] for r in rows) / n,
            'step_weighted_norm_return': sum(r['norm_return'] * r['steps'] for r in rows) / steps,
            'executed_steps': steps, 'full_clip_steps': sum(r['frames'] for r in rows),
            'frame_coverage': steps / sum(r['frames'] for r in rows),
            'expected_length_norm_return': sum(r['norm_return'] * r['steps'] for r in rows) / sum(r['frames'] for r in rows)}


def validate_rows(rows, expected):
    indexed = {}
    for row in rows:
        name = row['name']
        if name in indexed or name not in expected:
            raise ValueError(f'duplicate or foreign clip: {name}')
        frames, steps = row['frames'], row['steps']
        if not isinstance(frames, int) or frames != expected[name] or not isinstance(steps, int) or not 1 <= steps <= frames:
            raise ValueError(f'invalid frames/steps: {name}')
        if not isinstance(row['failed'], bool) or not math.isfinite(row['norm_return']):
            raise ValueError(f'invalid return/failure: {name}')
        if row['failed']:
            if row['fail_step'] != steps:
                raise ValueError(f'wrong failure step: {name}')
        elif steps != frames or row['fail_step'] != -1:
            raise ValueError(f'prematurely truncated successful clip: {name}')
        if not all(math.isfinite(v) for v in row['terms'].values()):
            raise ValueError(f'nonfinite reward terms: {name}')
        indexed[name] = row
    if indexed.keys() != expected.keys():
        raise ValueError('incomplete clip set')
    return indexed


def prepare(args):
    import torch
    from gobot.rl.motion_tracking.identity import source_identity
    if min(args.num_envs, args.nconmax, args.njmax) < 1:
        raise ValueError('environment count and physics capacities must be positive')
    root, output = args.data.resolve(), args.output.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if not manifest['complete'] or not manifest['ready_for_paired_training']:
        raise ValueError('data preparation has not passed')
    for name in ('manifest_all.jsonl', 'train_clips.json', 'eval_clips.json'):
        if digest(root / name) != manifest['artifacts'][name]:
            raise ValueError(f'data manifest hash differs: {name}')
    rows = [r for r in (json.loads(line) for line in (root / 'manifest_all.jsonl').open()) if r['keep']]
    if [r['name'] for r in rows] != json.loads((root / 'train_clips.json').read_text()):
        raise ValueError('data membership differs')
    subset = json.loads((root / 'eval_clips.json').read_text())
    if len(subset) != len(set(subset)) or not set(subset) <= {r['name'] for r in rows}:
        raise ValueError('invalid evaluation subset')
    for row in rows:
        if Path(row['name']).name != row['name'] or not (root / 'npz' / (row['name'] + '.npz')).is_file():
            raise ValueError(f'missing or invalid motion: {row["name"]}')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'gobot_supermimic_reproduced_v1':
        raise ValueError('expected a Gobot checkpoint')
    profile = args.project / 'assets/supermimic/robot_profile.json'
    sources = source_identity()
    if checkpoint['identity']['source'] != sources:
        raise ValueError('installed training/physics sources differ from checkpoint')
    shards = partition([{'name': r['name'], 'frames': r['frames'], 'npz_sha256': r['npz_sha256']} for r in rows],
                       args.max_clips, args.max_frames)
    plan = {'protocol': PROTOCOL, 'checkpoint': str(args.checkpoint.resolve()),
            'checkpoint_sha256': digest(args.checkpoint), 'checkpoint_iteration': checkpoint['iteration'],
            'checkpoint_updates': checkpoint['num_updates'], 'training_world_size': checkpoint['world_size'],
            'training_capacities': {k: checkpoint['config'][k] for k in ('nconmax', 'njmax')},
            'data': str(root), 'data_manifest_sha256': digest(root / 'manifest.json'),
            'project': str(args.project.resolve()), 'physics_digest': checkpoint['identity']['physics'],
            'kinematics_digest': checkpoint['identity']['kinematics'],
            'profile_sha256': digest(profile), 'source_sha256': sources,
            'evaluator_sha256': digest(__file__), 'max_envs': args.num_envs,
            'nconmax': args.nconmax, 'njmax': args.njmax, 'deterministic': args.deterministic,
            'precision': 'FP32 inputs/weights; same task TF32 setting as training',
            'start': 'first_frame', 'dense_motion': True, 'noise': False, 'domain_randomization': False,
            'foot_friction': 1.0, 'seed': 0, 'physics_overflow_check': 'every physics step, plus final state',
            'shards': shards, 'filtered_names': subset,
            'software': {p: importlib.metadata.version(p) for p in ('gobot', 'torch', 'mujoco', 'mujoco-warp', 'warp-lang')}}
    if (output / 'plan.json').exists():
        if json.loads((output / 'plan.json').read_text()) != plan:
            raise ValueError('existing evaluation plan differs')
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / 'plan.json', plan)
    print(json.dumps({'shards': len(shards), 'clips': len(rows), 'filtered_clips': len(subset),
                      'frames': sum(r['frames'] for r in rows), 'checkpoint_iteration': checkpoint['iteration'],
                      'max_shard_frames': max(sum(r['frames'] for r in shard) for shard in shards)}, indent=2))


def load_plan(output):
    path = Path(output) / 'plan.json'
    plan = json.loads(path.read_text())
    if plan['protocol'] != PROTOCOL or plan['evaluator_sha256'] != digest(__file__):
        raise ValueError('evaluation protocol/source changed')
    return plan, digest(path)


def validate_report(report, plan, plan_hash, index):
    if report['plan_sha256'] != plan_hash or report['shard_index'] != index or not report['complete']:
        raise ValueError('shard identity/completion differs')
    expected = {r['name']: r['frames'] for r in plan['shards'][index]}
    validate_rows(report['clips'], expected)
    if report['summary'] != summarize(report['clips']):
        raise ValueError('shard summary differs')


def evaluate_shard(cfg, policy, expected, progress):
    import torch
    from gobot.rl.motion_tracking.env import MotionTrackingEnv
    env = MotionTrackingEnv(cfg)
    try:
        provider = env.batch._provider
        if provider is None:
            raise ValueError('this cluster diagnostic requires the Warp provider')
        # Diagnostic guard only: no model or solver arrays are modified. Fail
        # before reset can clear a transient overflow flag on a terminated lane.
        provider._overflow_check_interval = 1
        provider.assert_no_overflow()
        if env.batch.artifact.digest != progress['physics_digest']:
            raise ValueError('physics scene changed from the trained checkpoint')
        if env.fk.scene.digest != progress['kinematics_digest']:
            raise ValueError('scene kinematics changed from the trained checkpoint')
        lib = env.library
        if {c.name: c.length for c in lib.clips} != expected:
            raise ValueError('loaded motion frames differ from manifest')
        current = env.ids.clone()
        nxt = current + env.num_envs
        obs, _ = env.reset(frames=lib.clip_start[current])
        provider.assert_no_overflow()
        env.forced_frame[:] = lib.clip_start[nxt.clamp_max(lib.num_clips - 1)]
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        returns = torch.zeros(env.num_envs, device=env.device)
        steps = torch.zeros_like(current)
        sums, result = {}, {}
        # A lane processes its entire queue by the sum of expected clip lengths.
        # Exceeding this bound indicates an evaluator/termination bug, not success.
        limit = max(sum(lib.clips[i].length for i in range(e, lib.num_clips, env.num_envs))
                    for e in range(env.num_envs))
        tick = 0
        with torch.no_grad():
            while len(result) < lib.num_clips:
                for value in obs.values():
                    if not torch.isfinite(value).all():
                        raise FloatingPointError('nonfinite policy observation')
                obs, reward, term, timeout, info = env.step(policy(obs['policy'], obs['goal']), record=False)
                provider.assert_no_overflow()
                provider.assert_finite()
                if not torch.isfinite(reward).all() or not torch.isfinite(info['terminal_state']).all():
                    raise FloatingPointError('nonfinite physics/reward state')
                tick += 1
                if tick > limit:
                    raise RuntimeError('complete-clip queue exceeded its expected step bound')
                returns += reward
                steps += 1
                for name, value in info['reward_terms'].items():
                    sums[name] = sums.get(name, torch.zeros_like(value)) + value
                ids = ((term | timeout) & active).nonzero().flatten()
                if len(ids):
                    indices = current[ids].tolist()
                    lengths = steps[ids].tolist()
                    scores = returns[ids].tolist()
                    failed = term[ids].tolist()
                    terms = {key: value[ids].tolist() for key, value in sums.items() if key != 'total'}
                    for j, ci in enumerate(indices):
                        st = lengths[j]
                        result[ci] = {'name': lib.clips[ci].name, 'frames': lib.clips[ci].length,
                            'steps': st, 'norm_return': scores[j] / (.1 * st), 'failed': failed[j],
                            'fail_step': st if failed[j] else -1,
                            'terms': {key: value[j] / st for key, value in terms.items()}}
                    current[ids] = nxt[ids]
                    nxt[ids] += env.num_envs
                    active[ids[current[ids] >= lib.num_clips]] = False
                    env.forced_frame[ids] = lib.clip_start[nxt[ids].clamp_max(lib.num_clips - 1)]
                    returns[ids] = 0
                    steps[ids] = 0
                    for value in sums.values():
                        value[ids] = 0
                if tick % 500 == 0:
                    print(json.dumps({'shard': progress['shard_index'], 'tick': tick,
                                      'completed_clips': len(result), 'total_clips': lib.num_clips}), flush=True)
        rows = [result[i] for i in range(lib.num_clips)]
        validate_rows(rows, expected)
        return rows
    finally:
        env.close()


def worker(args):
    import torch
    from gobot.rl.supermimic.config import TrainConfig
    from gobot.rl.motion_tracking.evaluate import load_policy
    from gobot.rl.motion_tracking.identity import source_identity
    plan, plan_hash = load_plan(args.output)
    if args.workers < 1 or not 0 <= args.index < args.workers:
        raise ValueError('invalid worker index')
    if digest(plan['checkpoint']) != plan['checkpoint_sha256']:
        raise ValueError('checkpoint changed')
    if source_identity() != plan['source_sha256']:
        raise ValueError('installed simulation/training source changed')
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    policy, raw = load_policy(plan['checkpoint'], args.device)
    policy.requires_grad_(False)
    cfg = TrainConfig.from_dict(raw)
    cfg.project = Path(plan['project'])
    cfg.data.npz_dir = Path(plan['data']) / 'npz'
    cfg.data.library = cfg.data.manifest = cfg.data.limit = None
    cfg.backend, cfg.device, cfg.deterministic = 'mujoco-warp', args.device, plan['deterministic']
    cfg.nconmax, cfg.njmax = plan['nconmax'], plan['njmax']
    cfg.env.replay_capacity = 32
    cfg.env.obs_noise = cfg.env.reset_noise = cfg.env.domain_randomization = False
    cfg.env.action_noise_std = 0.
    cfg.env.sampling_mode = 'start'
    cfg.env.compact_library = False
    cfg.env.episode_length_s = 1e5
    started = time.monotonic()
    for index in range(args.index, len(plan['shards']), args.workers):
        part = plan['shards'][index]
        target = args.output / 'shards' / f'{index:05d}.json'
        if target.exists():
            validate_report(json.loads(target.read_text()), plan, plan_hash, index)
            continue
        # Each dense file must match the data actually prepared for this run.
        for row in part:
            if digest(cfg.data.npz_dir / (row['name'] + '.npz')) != row['npz_sha256']:
                raise ValueError(f'motion content changed: {row["name"]}')
        cfg.data.clips = [r['name'] for r in part]
        cfg.env.num_envs = min(plan['max_envs'], len(part))
        torch.manual_seed(plan['seed'] + index)
        torch.cuda.reset_peak_memory_stats()
        print(json.dumps({'shard': index, 'clips': len(part), 'envs': cfg.env.num_envs,
                          'stage': 'initializing'}), flush=True)
        rows = evaluate_shard(cfg, policy, {r['name']: r['frames'] for r in part},
                              {**plan, 'shard_index': index})
        report = {'plan_sha256': plan_hash, 'shard_index': index, 'complete': True,
                  'checkpoint_sha256': plan['checkpoint_sha256'], 'summary': summarize(rows), 'clips': rows,
                  'peak_torch_allocated': torch.cuda.max_memory_allocated(),
                  'peak_torch_reserved': torch.cuda.max_memory_reserved()}
        atomic_json(target, report)
        status = {'worker': args.index, 'last_shard': index, 'elapsed_seconds': time.monotonic() - started,
                  'summary': report['summary'], 'stage': 'evaluating'}
        atomic_json(args.output / f'worker{args.index}.json', status)
        print(json.dumps(status), flush=True)
        torch.cuda.empty_cache()
    atomic_json(args.output / f'worker{args.index}.json', {'stage': 'complete', 'worker': args.index,
                                                          'elapsed_seconds': time.monotonic() - started})


def merge(args):
    plan, plan_hash = load_plan(args.output)
    all_rows, shard_hashes = [], {}
    for index in range(len(plan['shards'])):
        path = args.output / 'shards' / f'{index:05d}.json'
        if not path.exists():
            raise ValueError(f'missing shard: {index}')
        report = json.loads(path.read_text())
        validate_report(report, plan, plan_hash, index)
        all_rows.extend(report['clips'])
        shard_hashes[path.name] = digest(path)
    expected = {r['name']: r['frames'] for part in plan['shards'] for r in part}
    indexed = validate_rows(all_rows, expected)
    rows = [indexed[name] for name in sorted(indexed)]
    full = {'protocol': PROTOCOL, 'complete': True, 'plan_sha256': plan_hash,
            'checkpoint_sha256': plan['checkpoint_sha256'], 'checkpoint_iteration': plan['checkpoint_iteration'],
            'capacities': {'nconmax': plan['nconmax'], 'njmax': plan['njmax']},
            'summary': summarize(rows), 'clips': rows}
    atomic_json(args.output / 'full.json', full)
    filtered = [indexed[name] for name in plan['filtered_names']]
    atomic_json(args.output / 'filtered.json', {**full, 'summary': summarize(filtered), 'clips': filtered})
    failures = [r for r in rows if r['failed']]
    atomic_json(args.output / 'failures.json', sorted(failures, key=lambda r: (r['norm_return'], r['name'])))
    result = {'protocol': PROTOCOL, 'complete': True, 'full': full['summary'], 'filtered': summarize(filtered),
              'checkpoint_sha256': plan['checkpoint_sha256'], 'checkpoint_iteration': plan['checkpoint_iteration'],
              'capacities': full['capacities'], 'deterministic': plan['deterministic'],
              'training_world_size': plan['training_world_size'], 'data_manifest_sha256': plan['data_manifest_sha256'],
              'shard_hashes': shard_hashes, 'failed_clips': len(failures),
              'scope': 'Single Gobot trained seed, full prepared BONES training set and fixed filtered subset; not held-out generalization or matched-hardware reference retraining.'}
    atomic_json(args.output / 'summary.json', result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('checkpoint', 'project', 'data', 'output'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--max-clips', type=int, default=1024)
    prep.add_argument('--max-frames', type=int, default=1000000)
    prep.add_argument('--num-envs', type=int, default=1024)
    prep.add_argument('--nconmax', type=int, default=128)
    prep.add_argument('--njmax', type=int, default=500)
    prep.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=False,
                      help='extra Warp deterministic reductions; off in the pinned reference implementation')
    work = sub.add_parser('worker')
    work.add_argument('--output', type=Path, required=True)
    work.add_argument('--device', default='cuda:0')
    work.add_argument('--index', type=int, required=True)
    work.add_argument('--workers', type=int, default=8)
    merger = sub.add_parser('merge')
    merger.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        {'prepare': prepare, 'worker': worker, 'merge': merge}[args.command](args)
    except Exception as error:
        if args.command == 'worker':
            atomic_json(args.output / f'worker{args.index}.json', {'stage': 'failed', 'worker': args.index,
                                                                  'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
