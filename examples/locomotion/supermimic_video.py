"""Record comparable SuperMimic gait videos and periodically watch saved checkpoints."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

if __package__:
    from .supermimic_reference import DEFAULT_REFERENCE, DEFAULT_LEDGER, WINDOW, budget_class, write_json
else:
    from supermimic_reference import DEFAULT_REFERENCE, DEFAULT_LEDGER, WINDOW, budget_class, write_json

# Fixed active-motion windows; file preludes include standing/T poses.
CLIP_START_SECONDS = {'walk1_subject1': 4., 'run1_subject2': 32., 'sprint1_subject2': 8.}
CLIPS = tuple(CLIP_START_SECONDS)
ACTIVE = {'initializing', 'building_environment', 'training', 'trained', 'evaluating'}


def latest_checkpoint(run_dir: Path, status: dict):
    numbered = sorted(run_dir.glob('ckpt_*.pt'))
    if status.get('status') not in ACTIVE and (run_dir / 'last.pt').is_file():
        return run_dir / 'last.pt', int(status['iteration'])
    if numbered:
        return numbered[-1], int(numbered[-1].stem.removeprefix('ckpt_'))
    return None


def install_local_asset_resolver(reference: Path, manifest: dict):
    from unilab.assets import hub
    from unilab.base import backend_factory
    for row in manifest['assets']['files']:
        path = reference / row['local']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Pinned reference asset changed: {path}')
    original = hub.ensure_robot_assets_for_paths
    asset_root = reference / 'src/unisupermimic/assets'

    def resolve(paths):
        other = []
        for path in paths:
            if path and Path(path).resolve().is_relative_to(asset_root):
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
            else:
                other.append(path)
        original(other)

    hub.ensure_robot_assets_for_paths = resolve
    backend_factory.ensure_robot_assets_for_paths = resolve


def collect_clips(env, checkpoint, seconds):
    import torch
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
    from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
    from unisupermimic.tasks.motion_tracking.g1.supermimic.render import flattened_qpos, reference_qpos
    from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import load_policy

    mdp = env.supermimic_mdp
    motion = mdp.motion
    indices = [motion.names.index(name) for name in CLIPS]
    policy = load_policy(env, checkpoint)
    with torch.inference_mode():
        reset_frames = motion.clip_start.clone()
        for name, i in zip(CLIPS, indices):
            offset = round(CLIP_START_SECONDS[name] * 50)
            if offset + round(seconds * 50) >= int(motion.clip_len[i]):
                raise ValueError(f'Video window exceeds reference clip: {name}')
            reset_frames[i] += offset
        _, info = env.reset_to_frames(reset_frames)
        current = info[SUPERMIMIC_STATE_KEY]
        records = {name: {'states': [current['s'][i].cpu()], 'frames': [int(current['f'][i])],
                          'failed': False, 'done': False} for name, i in zip(CLIPS, indices)}
        history = None
        for _ in range(round(seconds * 50)):
            obs = mdp.obs(unflatten_state(current['s']), current['a_prev'], current['xi'], current['f'])
            history = (obs.unsqueeze(1).repeat(1, checkpoint['args']['hist'], 1) if history is None
                       else torch.cat((history[:, 1:], obs.unsqueeze(1)), dim=1))
            action = policy(history, mdp.goal(current['f'], checkpoint['args']['goal_frames']))
            state = env.step(action)
            transition = state.info[SUPERMIMIC_TRANSITION_KEY]
            for name, i in zip(CLIPS, indices):
                record = records[name]
                if record['done']:
                    continue
                # Keep the physical terminal frame; never show an automatic reset as recovery.
                record['states'].append(transition['s1'][i].cpu())
                end = int(motion.clip_start[i] + motion.clip_len[i] - 1)
                record['frames'].append(min(int(transition['f'][i]) + 1, end))
                record['failed'] = bool(transition['term'][i])
                record['done'] = bool(transition['done'][i])
            current = state.info[SUPERMIMIC_STATE_KEY]
            if all(record['done'] for record in records.values()):
                break
        for record in records.values():
            record['policy'] = flattened_qpos(torch.stack(record.pop('states'))).numpy()
            frames = torch.tensor(record.pop('frames'), device=motion.root_p.device, dtype=torch.long)
            record['reference'] = reference_qpos(motion, frames).cpu().numpy()
            record['simulated_seconds'] = (len(frames) - 1) / 50
    return records


def record(args):
    reference = args.reference.resolve()
    checkpoint_path = args.checkpoint.resolve()
    run_dir = args.run_dir.resolve()
    output = run_dir / 'videos/gait_windows' / f'iteration_{args.iteration:06d}'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').is_file():
        return
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.chdir(reference)
    owned_budget = None
    window = json.loads(WINDOW.read_text())
    total_hours = min(24., window['baseline_seconds'] / 3600 + window['additional_gpu_hours'])
    try:
        try:
            owned_budget = budget_class()(DEFAULT_LEDGER, total_hours=total_hours, run_hours=.17,
                metadata={'stage': 'supermimic_video', 'log_dir': str(run_dir), 'iteration': args.iteration})
        except BlockingIOError:
            # Rendering and training share the same single physical GPU. Its wall time
            # is already charged while the matching training process holds the ledger.
            active = json.loads(DEFAULT_LEDGER.read_text()).get('active_run', {})
            if active.get('log_dir') != str(run_dir):
                raise RuntimeError('Another experiment owns the GPU budget')
        started = time.time()
        import torch
        from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import make_eval_env
        from unisupermimic.tasks.motion_tracking.g1.supermimic.render import render_comparison, DEFAULT_SCENE
        torch.set_num_threads(2)
        torch.manual_seed(71)
        manifest = json.loads((run_dir / 'manifest.json').read_text())
        install_local_asset_resolver(reference, manifest)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if int(checkpoint['iteration']) != args.iteration:
            raise ValueError('Checkpoint iteration changed while selecting video')
        env = make_eval_env(run_dir, backend='mjwarp')
        try:
            records = collect_clips(env, checkpoint, args.seconds)
        finally:
            env.close()
            env.backend.close()
        result = {'iteration': args.iteration, 'checkpoint': str(checkpoint_path),
                  'created_unix': time.time(), 'requested_seconds': args.seconds,
                  'reference_alignment': 'world coordinates, no pelvis alignment',
                  'protocol': 'fixed active-motion windows with reference-state reset; failure stops clip; includes terminal state; not full-clip evaluation',
                  'clip_start_seconds': CLIP_START_SECONDS,
                  'clips': {}}
        for name, row in records.items():
            end = ('FAILED' if row['failed'] else 'CLIP END') if row['done'] else 'preview limit'
            label = f"iter {args.iteration} | {name} +{CLIP_START_SECONDS[name]:g}s | {end} {row['simulated_seconds']:.2f}s"
            video = output / f'{name}.mp4'
            temporary = output / f'{name}.partial.mp4'
            written = render_comparison(row['reference'], row['policy'], temporary,
                scene=DEFAULT_SCENE, clip_name=label, width=960, height=540, fps=25, global_ghost=True)
            temporary.replace(video)
            result['clips'][name] = {key: value for key, value in row.items() if key not in ('reference', 'policy')}
            result['clips'][name].update(video=str(video), frames=written, duration_s=written / 25)
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(min(2., written / 50)),
                '-i', str(video), '-frames:v', '1', str(video.with_suffix('.png'))], check=True)
        concat = output / 'concat.txt'
        concat.write_text(''.join(f"file '{name}.mp4'\n" for name in CLIPS))
        combined = output / 'comparison.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '1', '-i', str(concat),
                        '-c', 'copy', '-movflags', '+faststart', str(combined)], check=True)
        result.update(comparison=str(combined), elapsed_s=time.time() - started)
        from PIL import Image
        preview = Image.new('RGB', (960, 540 * len(CLIPS)))
        for i, name in enumerate(CLIPS):
            with Image.open(output / f'{name}.png') as frame:
                preview.paste(frame, (0, i * 540))
        preview.save(output / 'preview.jpg')
        write_json(output / 'manifest.json', result)
        write_json(run_dir / 'videos/latest.json', result)
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if owned_budget is not None:
            owned_budget.close()


def watch(args):
    run_dir = args.run_dir.resolve()
    output = run_dir / 'videos'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop = threading.Event()
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: stop.set())
        deadline = time.monotonic() + args.max_hours * 3600
        next_record = 0.
        failures = 0
        while not stop.is_set() and time.monotonic() < deadline:
            status = json.loads((run_dir / 'status.json').read_text())
            terminal = status.get('status') not in ACTIVE
            fresh = time.time() - status.get('heartbeat_unix', 0.) < 180
            if not terminal and not fresh:
                write_json(output / 'watch_status.json', {'state': 'stopped', 'reason': 'training heartbeat stale'})
                return
            chosen = latest_checkpoint(run_dir, status)
            if chosen is not None and (terminal or time.monotonic() >= next_record):
                checkpoint, iteration = chosen
                destination = output / 'gait_windows' / f'iteration_{iteration:06d}'
                if not (destination / 'manifest.json').is_file():
                    write_json(output / 'watch_status.json', {'state': 'recording', 'iteration': iteration,
                               'pid': os.getpid(), 'interval_minutes': args.interval_minutes, 'updated_unix': time.time()})
                    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--run-dir', str(run_dir),
                               '--reference', str(args.reference.resolve()), '--checkpoint', str(checkpoint),
                               '--iteration', str(iteration), '--seconds', str(args.seconds)]
                    with (output / f'record_{iteration:06d}.log').open('a') as stream:
                        try:
                            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)
                            failures = 0
                        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                            failures += 1
                            write_json(output / 'watch_status.json', {'state': 'recording_failed', 'iteration': iteration,
                                       'failures': failures, 'error': str(exc), 'updated_unix': time.time()})
                            if failures >= 3:
                                return
                            stop.wait(30)
                            continue
                next_record = time.monotonic() + args.interval_minutes * 60
                write_json(output / 'watch_status.json', {'state': 'finished' if terminal else 'waiting',
                           'last_iteration': iteration, 'interval_minutes': args.interval_minutes,
                           'next_record_unix': time.time() + args.interval_minutes * 60, 'updated_unix': time.time()})
            if terminal:
                return
            stop.wait(20)
        write_json(output / 'watch_status.json', {'state': 'stopped', 'reason': 'signal or watcher deadline'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--iteration', type=int)
    p.add_argument('--seconds', type=float, default=8.)
    p.add_argument('--watch', action='store_true')
    p.add_argument('--interval-minutes', type=float, default=15.)
    p.add_argument('--max-hours', type=float, default=4.)
    args = p.parse_args()
    if not all(math.isfinite(v) and v > 0 for v in (args.seconds, args.interval_minutes, args.max_hours)):
        p.error('time limits must be positive and finite')
    if args.watch:
        watch(args)
    elif args.checkpoint is not None and args.iteration is not None:
        record(args)
    else:
        p.error('use --watch or provide --checkpoint and --iteration')


if __name__ == '__main__':
    main()
