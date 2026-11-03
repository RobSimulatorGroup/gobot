"""Bounded, resumable G1 AMP/PPO training from Gobot-authored scenes."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import signal
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from torch.utils.tensorboard import SummaryWriter

from gobot.rl import RslRlVecEnvWrapper
from gobot.rl.locomotion import make_velocity_env
from gobot.rl.locomotion.motion import MotionPriorCfg
from gobot.rl.locomotion.budget import TrainingBudget
from gobot.rl.locomotion.ppo import rsl_rl_train_cfg
from gobot.rl.locomotion.manifest import build_policy_manifest, validate_policy
from gobot.rl.policy import POLICY_MANIFEST_KEY, policy_manifest_from_checkpoint, write_policy_manifest_sidecar
from .config import g1_amp_cfg
from .evaluate_amp import validate_training_env


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def snapshot_sources(log_dir):
    """Keep exact source/config inputs for experiments made on an uncommitted tree."""
    import hashlib
    import tarfile
    root = Path(__file__).resolve().parents[2]
    files = sorted(set(list((root / 'python/gobot/rl').rglob('*.py')) +
                       list((root / 'examples/g1').glob('*.py')) +
                       list((root / 'examples/g1').glob('*.jscn')) +
                       list((root / 'examples/g1/terrain').glob('*.jres'))))
    digest = hashlib.sha256()
    with tarfile.open(log_dir / 'sources.tar.gz', 'w:gz') as archive:
        for path in files:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
            archive.add(path, arcname=str(path.relative_to(root)))
    (log_dir / 'sources.sha256').write_text(digest.hexdigest() + '\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('flat', 'rough', 'mixed'), default='flat')
    p.add_argument('--num-envs', type=int, default=512)
    p.add_argument('--backend', choices=('mujoco-warp', 'mujoco-cpu'), default='mujoco-warp')
    p.add_argument('--device')
    p.add_argument('--iterations', type=int, default=30000)
    p.add_argument('--hours', type=float, default=5.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log-dir', type=Path, default=Path('examples/g1/logs/amp_flat'))
    p.add_argument('--resume', type=Path)
    p.add_argument('--dataset', type=Path, default=Path('examples/g1/assets/motions/locomotion.npz'))
    p.add_argument('--save-interval', type=int, default=250)
    p.add_argument('--eval-interval', type=int, default=500)
    p.add_argument('--eval-steps', type=int, default=1000)
    p.add_argument('--style-weight', type=float, default=2.)
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--schedule', choices=('adaptive', 'fixed'), default='adaptive')
    p.add_argument('--reset-action-std', type=float)
    p.add_argument('--log-stdout', type=Path)
    p.add_argument('--budget-file', type=Path, default=Path('examples/g1/logs/gpu_budget.json'))
    p.add_argument('--budget-hours', type=float, default=24.)
    p.add_argument('--no-eval', action='store_true')
    p.add_argument('--reference-reset-probability', type=float, default=.5)
    args = p.parse_args(argv)
    if args.log_stdout:
        import sys
        args.log_stdout.parent.mkdir(parents=True, exist_ok=True)
        stream = args.log_stdout.open('a', buffering=1)
        sys.stdout = stream
        sys.stderr = stream
    if args.hours <= 0 or not 0 < args.budget_hours <= 24:
        raise ValueError('training hours must be positive and cumulative budget at most 24 hours')
    device = args.device or ('cuda:0' if args.backend == 'mujoco-warp' else 'cpu')
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.budget_file.parent.mkdir(parents=True, exist_ok=True)
    budget = TrainingBudget(args.budget_file, total_hours=args.budget_hours, run_hours=args.hours,
                            metadata={'stage': args.stage, 'log_dir': str(args.log_dir)})
    start = budget.start
    run_seconds = budget.limit_seconds
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    cfg = g1_amp_cfg(stage=args.stage)
    env = make_velocity_env(cfg, backend=args.backend, device=device, num_envs=args.num_envs,
                            seed=args.seed, collect_step_extras=False)
    if args.stage == 'flat' and args.backend == 'mujoco-warp' and args.reference_reset_probability:
        from gobot.rl.locomotion.motion import ReferenceStateInitializer
        env.set_reference_initializer(ReferenceStateInitializer(args.dataset, device=device,
                                      probability=args.reference_reset_probability))
        env.reset(seed=args.seed)
    wrapper = RslRlVecEnvWrapper(env, device=device)
    writer = SummaryWriter(str(args.log_dir))
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    config = rsl_rl_train_cfg(experiment_name=cfg.name, obs_normalization=True)
    prior_cfg = MotionPriorCfg(dataset=str(args.dataset.resolve()), weight=args.style_weight)
    config['actor']['distribution_cfg']['init_std'] = .5
    config['algorithm']['learning_rate'] = args.learning_rate or 3.e-4
    config['algorithm']['schedule'] = args.schedule
    config['algorithm'].update(class_name='gobot.rl.locomotion.amp:MotionPriorPPO', motion_prior_cfg=asdict(prior_cfg))
    manifest = build_policy_manifest(wrapper, cfg, config, Path(cfg.project_path))
    snapshot_sources(args.log_dir)
    configuration = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                     'task': asdict(cfg), 'ppo': config}
    atomic_json(args.log_dir / 'config.json', json.loads(json.dumps(configuration, default=str)))
    runner = OnPolicyRunner(wrapper, config, log_dir=None, device=device)
    alg = runner.alg
    if alg.reference.metadata['robot_digest'] != manifest.scene_digest:
        raise ValueError('motion features were prepared for different robot physics; re-run prepare_motion')
    first = 0
    completed = 0
    ranking_path = args.log_dir / 'ranking.json'
    ranking = json.loads(ranking_path.read_text()) if ranking_path.exists() else []
    best = tuple(ranking[0]['score']) if ranking else None
    def save(path, iteration):
        payload = alg.save()
        payload.update(iter=iteration, infos={POLICY_MANIFEST_KEY: manifest.metadata(),
            'env_state': wrapper.training_state_dict(), 'stage': args.stage,
            'rng': {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}})
        temp = path.with_suffix('.tmp')
        torch.save(payload, temp); temp.replace(path)
        write_policy_manifest_sidecar(path, manifest)
    try:
        if args.resume:
            payload = torch.load(args.resume, map_location=device, weights_only=False)
            saved = policy_manifest_from_checkpoint(payload)
            validate_policy(saved, env, cfg)
            alg.load(payload, None, True)
            first = int(payload['iter']) + 1
            if payload['infos']['stage'] == args.stage:
                wrapper.load_training_state_dict(payload['infos']['env_state'])
            rng = payload['infos']['rng']
            random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'].cpu())
            if rng['cuda'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all([s.cpu() for s in rng['cuda']])
        if args.learning_rate is not None:
            alg.learning_rate = args.learning_rate
            for group in alg.optimizer.param_groups:
                group['lr'] = args.learning_rate
        if args.reset_action_std is not None:
            if args.reset_action_std <= 0:
                raise ValueError('action standard deviation must be positive')
            with torch.no_grad():
                alg.actor.distribution.std_param.fill_(args.reset_action_std)
        obs = wrapper.get_observations()
        alg.train_mode()
        totals = torch.zeros(args.num_envs, device=device)
        lengths = torch.zeros(args.num_envs, device=device)
        trace = (args.log_dir / 'metrics.jsonl').open('a', buffering=1)
        for iteration in range(first, first + args.iterations):
            if stopped or time.monotonic() - start >= run_seconds:
                break
            tick = time.monotonic()
            episode_rewards, episode_lengths = [], []
            with torch.inference_mode():
                for _ in range(config['num_steps_per_env']):
                    action = alg.act(obs)
                    obs, reward, done, extras = wrapper.step(action)
                    if not bool(torch.isfinite(obs['actor']).all() & torch.isfinite(reward).all()):
                        raise FloatingPointError('non-finite training transition')
                    alg.process_env_step(obs, reward, done, extras)
                    totals += reward; lengths += 1
                    if bool(done.any()):
                        episode_rewards.extend(totals[done].tolist()); episode_lengths.extend(lengths[done].tolist())
                        totals[done] = 0; lengths[done] = 0
                alg.compute_returns(obs)
            losses = alg.update()
            completed = iteration + 1
            duration = time.monotonic() - tick
            metrics = {'iteration': iteration, 'elapsed_s': time.monotonic() - start,
                       'steps_per_second': args.num_envs * config['num_steps_per_env'] / duration,
                       'episode_reward': float(np.mean(episode_rewards)) if episode_rewards else None,
                       'episode_steps': float(np.mean(episode_lengths)) if episode_lengths else None,
                       'terrain_level': float(torch.as_tensor(env.terrain_assignments[0]).float().mean()),
                       'learning_rate': alg.learning_rate,
                       'action_std': float(alg.actor.distribution.std_param.detach().mean()),
                       'command_speed': float(extras['transition_state']['command'][:, :2].norm(dim=1).mean()),
                       'measured_speed': float(extras['transition_state']['base_linear_velocity_body'][:, :2].norm(dim=1).mean()), **losses}
            trace.write(json.dumps(metrics) + '\n')
            for key, value in metrics.items():
                if value is not None: writer.add_scalar(key, value, iteration)
            if iteration % 25 == 0:
                print(json.dumps(metrics), flush=True)
            if (iteration + 1) % args.save_interval == 0:
                save(args.log_dir / 'latest.pt', iteration)
            if not args.no_eval and (iteration + 1) % args.eval_interval == 0:
                result = validate_training_env(env, alg.actor, steps=args.eval_steps)
                score = (min(cell['admission'] for cell in result['cells']), result['admission'], -result['velocity_error'])
                atomic_json(args.log_dir / f'eval_{iteration:06d}.json', result)
                print('EVALUATION', json.dumps(result), flush=True)
                if len(ranking) < 3 or score > tuple(ranking[-1]['score']):
                    candidate = args.log_dir / f'candidate_{iteration:06d}.pt'
                    save(candidate, iteration)
                    ranking.append({'score': score, 'checkpoint': candidate.name, 'iteration': iteration})
                    ranking.sort(key=lambda item: tuple(item['score']), reverse=True)
                    for discarded in ranking[3:]:
                        old = args.log_dir / discarded['checkpoint']
                        old.unlink(missing_ok=True)
                        old.with_suffix(old.suffix + '.manifest.json').unlink(missing_ok=True)
                    ranking = ranking[:3]
                    atomic_json(ranking_path, ranking)
                if best is None or score > best:
                    best = score
                    save(args.log_dir / 'best.pt', iteration)
                    atomic_json(args.log_dir / 'best_eval.json', result)
                wrapper._sync_steps_from_core()
                obs = wrapper.get_observations()
                totals.zero_(); lengths.zero_()
                alg.train_mode()
            budget.update(updates=max(0, completed - first), last_iteration=iteration)
        trace.close()
        save(args.log_dir / 'latest.pt', max(first - 1, completed - 1))
        save(args.log_dir / 'model_final.pt', max(first - 1, completed - 1))
        summary = {'stage': args.stage, 'updates_completed': max(0, completed - first),
                   'elapsed_s': time.monotonic() - start, 'best': best, 'stopped': stopped,
                   'checkpoint': str(args.log_dir / 'model_final.pt')}
        atomic_json(args.log_dir / 'summary.json', summary)
        print('FINISHED', json.dumps(summary), flush=True)
    finally:
        writer.close(); wrapper.close()
        budget.close(updates=max(0, completed - first))

if __name__ == '__main__':
    main()
