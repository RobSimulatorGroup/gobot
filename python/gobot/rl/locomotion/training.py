"""Bounded, resumable PPO/AMP experiments from Gobot-authored scenes."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import tarfile
import time
import traceback
import numpy as np
import torch
from gobot.rl import RslRlVecEnvWrapper
from gobot.rl.locomotion import make_velocity_env
from gobot.rl.locomotion.budget import TrainingBudget
from gobot.rl.locomotion.manifest import build_policy_manifest, validate_policy
from gobot.rl.locomotion.ppo import rsl_rl_train_cfg
from gobot.rl.locomotion.motion import MotionPriorCfg, ReferenceStateInitializer
from gobot.rl.policy import POLICY_MANIFEST_KEY, policy_manifest_from_checkpoint, write_policy_manifest_sidecar
from .experiment import Experiment, task_config
from gobot.rl.training.projects import example_module, examples_root


def atomic_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,default=str)+'\n');temp.replace(path)


def snapshot_sources(directory):
    """Archive the installed task sources, including optional checkout build inputs."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    package_root = Path(__file__).resolve().parents[2]
    example_root = examples_root()
    source_roots = [(package_root / 'rl', Path('python/gobot/rl'))]
    source_roots.extend((example_root / name, Path('examples') / name)
                        for name in ('g1', 'go1', 'locomotion'))
    files = {}
    for source_root, archive_root in source_roots:
        for path in source_root.rglob('*'):
            relative = path.relative_to(source_root)
            if (path.is_file() and path.suffix in {'.py', '.jscn', '.jres', '.json'}
                    and not any(part in {'logs', 'assets', 'policies', '__pycache__'}
                                for part in relative.parts)):
                files[str(archive_root / relative)] = path
    checkout_root = example_root.parent
    for name in ('pyproject.toml', 'uv.lock',
                 'include/gobot/simulation/locomotion_command_runtime.hpp',
                 'src/gobot/simulation/locomotion_command_runtime.cpp',
                 'src/gobot/python/manual_bindings/manual_binding_app_context.cpp'):
        path = checkout_root / name
        if path.is_file():
            files[name] = path
    digest = hashlib.sha256()
    with tarfile.open(directory / 'sources.tar.gz', 'w:gz') as archive:
        for name, path in sorted(files.items()):
            digest.update(name.encode())
            digest.update(path.read_bytes())
            archive.add(path, arcname=name)
    (directory / 'sources.sha256').write_text(digest.hexdigest() + '\n')
    return digest.hexdigest()


def agent_config(exp, *, experiment_name='velocity'):
    """Return independent PPO/AMP settings for the built-in experiment recipe."""
    config = rsl_rl_train_cfg(experiment_name=experiment_name, obs_normalization=True)
    for name in ('max_iterations', 'save_interval', 'clip_actions'):
        config.pop(name)
    config['actor']['distribution_cfg']['init_std'] = exp.initial_action_std
    config['algorithm'].update(
        class_name='PPO', learning_rate=exp.learning_rate,
        schedule='adaptive', entropy_coef=exp.entropy_coef)
    if exp.recipe == 'amp':
        dataset = Path(exp.motion_dataset)
        if not dataset.is_absolute() and dataset.parts[:1] == ('examples',):
            dataset = examples_root().joinpath(*dataset.parts[1:])
        config['algorithm'].update(
            class_name='gobot.rl.locomotion.amp:MotionPriorPPO',
            motion_prior_cfg=asdict(MotionPriorCfg(
                dataset=str(dataset.resolve()), weight=exp.style_weight, seed=exp.seed)))
    return config


def validate_agent_config(exp, config):
    """Check the custom runner's contracts before constructing a physics world.

    This loop owns iteration limits, checkpoint cadence and action clipping.
    Accepting RSL runner fields the loop never reads would silently ignore
    part of the requested recipe.
    """
    if not isinstance(config, dict):
        raise ValueError('agent configuration must be an object')
    configurable = {'actor', 'critic', 'algorithm', 'num_steps_per_env', 'obs_groups', 'experiment_name'}
    for name in config:
        if name not in configurable:
            target = {'max_iterations': 'experiment.iterations',
                      'save_interval': 'experiment.save_interval',
                      'clip_actions': 'env.action_clip'}.get(name)
            detail = f'; configure {target} instead' if target else '; use a custom TrainingSpec for another runner'
            raise ValueError(f'agent.{name} is not used by the locomotion training loop{detail}')

    def finite_values(value, path='agent'):
        if isinstance(value, dict):
            for key, item in value.items():
                finite_values(item, f'{path}.{key}')
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                finite_values(item, f'{path}[{index}]')
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'{path} must be finite')
    finite_values(config)

    def positive_integer(value, name):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')

    def positive_number(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')

    positive_integer(config.get('num_steps_per_env'), 'num_steps_per_env')
    distribution = config.get('actor', {}).get('distribution_cfg', {})
    if (distribution.get('class_name') not in {
            'rsl_rl.modules.GaussianDistribution', 'rsl_rl.modules:GaussianDistribution'}
            or distribution.get('std_type') not in {'scalar', 'log'}):
        raise ValueError('the locomotion trainer requires a scalar or log GaussianDistribution')
    positive_number(distribution.get('init_std'), 'agent.actor.distribution_cfg.init_std')
    algorithm = config.get('algorithm', {})
    for name in ('num_learning_epochs', 'num_mini_batches'):
        positive_integer(algorithm.get(name), f'agent.algorithm.{name}')
    for name in ('learning_rate', 'max_grad_norm'):
        positive_number(algorithm.get(name), f'agent.algorithm.{name}')
    for name in ('gamma', 'lam', 'clip_param'):
        value = algorithm.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f'agent.algorithm.{name} must be between zero and one')
    if int(os.environ.get('WORLD_SIZE', '1')) > 1 and algorithm.get('class_name') not in {
            'PPO', 'rsl_rl.algorithms.PPO', 'rsl_rl.algorithms:PPO'}:
        raise ValueError('synchronized locomotion training supports PPO; AMP discriminator synchronization is not enabled')
    if 'motion_prior_cfg' in algorithm:
        MotionPriorCfg(**algorithm['motion_prior_cfg'])


def validate_task_config(exp, cfg, agent=None):
    """Validate state and action contracts required by this PPO/AMP workflow."""
    import math
    if not math.isfinite(cfg.physics_dt) or cfg.physics_dt <= 0:
        raise ValueError('env.physics_dt must be finite and positive')
    if isinstance(cfg.decimation, bool) or not isinstance(cfg.decimation, int) or cfg.decimation < 1:
        raise ValueError('env.decimation must be a positive integer')
    if agent is not None and 'motion_prior_cfg' in agent['algorithm'] and not math.isclose(cfg.physics_dt * cfg.decimation, .02):
        raise ValueError('AMP motion transitions require a 50 Hz control period (physics_dt * decimation = 0.02)')
    if exp.recipe in {"amp", "gait"} and not cfg.capture_transitions:
        raise ValueError('PPO/AMP training requires env.capture_transitions=true for terminal bootstrap and diagnostics')


def _default_budget_file(directory, backend):
    if backend == 'mujoco-warp' and (examples_root().parent / 'pyproject.toml').is_file():
        return example_module('locomotion.budget').DEFAULT_LEDGER
    # Installed examples may be read-only. Keep a wheel user's ledger with the run.
    return Path(directory) / 'gpu_budget.json'


def restore_rng(rng, device):
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'].cpu())
    if rng['cuda'] is not None:
        torch.cuda.set_rng_state(rng['cuda'].cpu(), device)


def update_policy(algorithm, environment, observation, steps, record=None, distributed=None, timings=None):
    """Use RSL-RL's rollout, GAE and PPO update without modifying the optimizer."""
    from rsl_rl.utils import check_nan

    def clock():
        device = getattr(environment, "device", "cpu")
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        return time.monotonic()

    started = clock() if timings is not None else 0.
    error = None
    try:
        with torch.inference_mode():
            for _ in range(steps):
                action = algorithm.act(observation)
                observation, reward, done, extras = environment.step(action)
                check_nan(observation, reward, done)
                algorithm.process_env_step(observation, reward, done, extras)
                if record is not None:
                    record(action, reward, done, extras)
            collected = clock() if timings is not None else 0.
            algorithm.compute_returns(observation)
    except Exception as caught:
        error = caught
    if distributed is not None:
        distributed.check_error(error, phase='rollout')
    elif error is not None:
        raise error
    losses = algorithm.update()
    error = None
    if not all(np.isfinite(float(value)) for value in losses.values()):
        error = FloatingPointError('non-finite optimizer metric')
    if distributed is not None:
        distributed.check_error(error, phase='optimizer')
    elif error is not None:
        raise error
    if timings is not None:
        timings.update(collection_s=collected - started, learning_s=clock() - collected)
    return observation, losses


def build_algorithm(environment, config, device, distributed):
    """Construct RSL-RL directly: this module already owns the training loop."""
    from rsl_rl.utils import resolve_callable
    config = deepcopy(config)
    config['multi_gpu'] = distributed.algorithm_config
    algorithm_class = resolve_callable(config['algorithm']['class_name'])
    algorithm = algorithm_class.construct_algorithm(environment.get_observations(), environment, config, device)
    return algorithm


def _evaluate_checkpoint(exp, candidate, output, *, distributed, backend, budget,
                         budget_file, stop_requested, report, completed, iteration):
    """Evaluate on rank 0 while every worker participates in stop checks."""
    process = stream = None

    def start():
        nonlocal process, stream
        command = [sys.executable, '-B', '-c',
                   "from gobot.rl.training.projects import example_module; example_module('locomotion.assessment').main()",
                   '--checkpoint', str(candidate), '--output', str(output),
                   '--seed', str(exp.eval_seed), '--steps', str(exp.eval_steps),
                   '--num-envs', str(exp.eval_envs), '--device', distributed.device,
                   '--backend', backend, '--budget-file', str(budget_file.resolve()),
                   '--budget-parent-pid', str(os.getpid())]
        environ = {key: value for key, value in os.environ.items()
                   if key not in {'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
                                  'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'}}
        stream = (candidate.parent / 'evaluation.log').open('a', buffering=1)
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environ)

    def poll():
        try:
            result = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            result = None
        budget.update(updates=completed, last_iteration=iteration)
        report('evaluating', checkpoint=str(candidate))
        return result

    try:
        distributed.on_main(start)
        while True:
            reason = distributed.stop_reason(stop_requested(), budget)
            if reason:
                return None, reason
            result = distributed.on_main(poll)
            if result is not None:
                if result:
                    raise RuntimeError(f'evaluator exited {result}; see {candidate.parent / "evaluation.log"}')
                return distributed.on_main(lambda: json.loads(output.read_text())), None
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if stream is not None:
            stream.close()


def run(exp, directory, *, backend='mujoco-warp', device=None, budget_file=None,
        resume=None, initialize=None, no_eval=False, task_cfg=None, agent_cfg=None, terrain_reference=None):
    """Train velocity PPO, one environment batch per process under torchrun.

    RSL-RL synchronizes PPO gradients, adaptive learning rates and observation
    statistics. Checkpoints keep a shared algorithm and each worker's curriculum
    and RNG state. Resuming restarts physics episodes, as on a single GPU.
    """
    from contextlib import ExitStack
    from .distributed import DistributedTraining, cpu_state
    from .training_log import TrainingStatistics, format_training_log

    if backend not in {'mujoco-cpu', 'mujoco-warp'}:
        raise ValueError(f'unsupported locomotion training backend {backend!r}')
    if task_cfg is not None and not no_eval:
        raise ValueError('custom task_cfg requires no_eval=True; the built-in independent evaluator only supports built-in presets')
    if terrain_reference is not None and (exp.stage != 'rough' or exp.recipe != 'baseline'):
        raise ValueError('--terrain-reference requires rough baseline training')
    if resume and initialize:
        raise ValueError('choose resume or initialize, not both')
    cfg = deepcopy(task_cfg) if task_cfg is not None else task_config(exp)
    config = deepcopy(agent_cfg) if agent_cfg is not None else agent_config(exp, experiment_name=cfg.name)
    validate_agent_config(exp, config)
    validate_task_config(exp, cfg, config)
    effective_agent_config = deepcopy(config)
    distributed = DistributedTraining.from_environment(device or ('cuda:0' if backend == 'mujoco-warp' else 'cpu'))
    device = distributed.device
    seed = exp.seed + distributed.rank
    directory = Path(directory)
    budget_file = Path(budget_file or _default_budget_file(directory, backend))
    status = {'state': 'starting', 'pid': os.getpid(), 'host': socket.gethostname(),
              'started_unix': time.time(), 'pid_namespace': os.readlink('/proc/self/ns/pid'),
              'process_start_ticks': Path('/proc/self/stat').read_text().split()[21],
              'experiment': exp.metadata(), 'world_size': distributed.world_size,
              'num_envs_per_rank': exp.num_envs, 'total_num_envs': exp.num_envs * distributed.world_size}
    stop_requested = False

    def stop(*_):
        nonlocal stop_requested
        stop_requested = True

    old_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in old_signals:
        signal.signal(sig, stop)
    budget = env = wrapper = trace = None
    iteration = -1
    best = last_validation = None
    completed = 0
    reason = 'iterations'
    prepared = False
    resources = ExitStack()

    def report(state, **values):
        if distributed.main and prepared:
            status.update(state=state, heartbeat_unix=time.time(), iteration=iteration, **values)
            atomic_json(directory / 'status.json', status)

    def prepare():
        nonlocal budget, prepared
        if directory.is_dir() and any(directory.iterdir()) and resume is None:
            raise ValueError(f'output directory is not empty: {directory}; choose a new output or resume explicitly')
        directory.mkdir(parents=True, exist_ok=True)
        prepared = True
        report('starting')
        budget = TrainingBudget(
            budget_file, total_hours=exp.budget_hours, run_hours=exp.hours * distributed.world_size,
            allocated_gpus=distributed.world_size,
            metadata={'robot': exp.robot, 'stage': exp.stage, 'log_dir': str(directory)})
        source_digest = snapshot_sources(directory)
        atomic_json(directory / 'experiment.json', exp.metadata())
        atomic_json(directory / 'config.json', {
            'experiment': exp.metadata(), 'task': asdict(cfg), 'ppo': effective_agent_config,
            'source_digest': source_digest, 'world_size': distributed.world_size,
            'num_envs_per_rank': exp.num_envs, 'total_num_envs': exp.num_envs * distributed.world_size})
        return source_digest

    try:
        distributed.initialize()
        source_digest = distributed.on_main(prepare)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
        if backend == 'mujoco-warp':
            import warp as wp
            resources.enter_context(wp.ScopedDevice(device))
        env = make_velocity_env(cfg, backend=backend, num_envs=exp.num_envs, device=device,
                                seed=seed, collect_step_extras=False, collect_training_stats=True)
        terrain_info = None
        if terrain_reference is not None:
            from .terrain_reference import load, verify_environment, initial_assignments, assignment_digest
            terrain_data = distributed.on_main(lambda: load(terrain_reference))
            error = None
            try:
                verify_environment(env, terrain_data)
            except Exception as caught:
                error = caught
            distributed.check_error(error, phase='training terrain')
            levels, types = initial_assignments(terrain_data, exp.num_envs, seed)
            env.set_terrain_assignments(levels, types, fixed=False)
            env.reset(seed=seed)
            assignments = distributed.gather(assignment_digest(levels, types))
            terrain_info = {'sha256': terrain_data['sha256'], 'seed': terrain_data['seed'],
                            'initial_assignments': assignments}
            distributed.on_main(lambda: atomic_json(directory / 'terrain.json', terrain_info))
            distributed.on_main(lambda: atomic_json(directory / 'training_terrain.json', terrain_data))
        use_amp = 'motion_prior_cfg' in config['algorithm']
        wrapper = RslRlVecEnvWrapper(env, device=device)
        manifest = build_policy_manifest(wrapper, cfg, config, Path(cfg.project_path))
        alg = build_algorithm(wrapper, config, device, distributed)
        if exp.reference_reset_probability:
            if backend != 'mujoco-warp' or not exp.stage.startswith('flat') or exp.robot != 'g1':
                raise ValueError('reference reset is supported only for G1 flat Warp training')
            library = alg.reference if use_amp else exp.motion_dataset
            env.set_reference_initializer(ReferenceStateInitializer(
                library, device=device, probability=exp.reference_reset_probability))
            env.reset(seed=seed)
        task_identity = asdict(cfg)
        task_identity.pop('project_path', None)
        semantic = exp.metadata()
        for name in ('num_envs', 'iterations', 'hours', 'budget_hours', 'save_interval',
                     'eval_interval', 'eval_envs', 'eval_steps', 'eval_seed'):
            semantic.pop(name, None)
        signature_data = {'experiment': semantic, 'task': task_identity,
                          'backend': backend, 'agent': effective_agent_config, 'contact_timing': 'physics_dt',
                          'terrain_sha256': terrain_info['sha256'] if terrain_info else None}
        signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True, default=str).encode()).hexdigest()
        distributed.on_main(lambda: atomic_json(directory / 'training_signature.json', {'signature': signature}))
        if use_amp and alg.reference.metadata['robot_digest'] != manifest.scene_digest:
            raise ValueError('reference dataset physics differs from authored robot')
        first = 0
        if resume or initialize:
            payload = torch.load(resume or initialize, map_location='cpu', weights_only=False)
            validate_policy(policy_manifest_from_checkpoint(payload), env, cfg)
            if resume:
                info = payload['infos']
                if info.get('training_signature') != signature:
                    raise ValueError('resume configuration signature changed; use initialize to start a new experiment')
                if info.get('world_size') != distributed.world_size or info.get('num_envs_per_rank') != exp.num_envs:
                    raise ValueError('resume requires the same world size and num_envs per rank; use initialize to start a new experiment')
                if len(info['rank_states']) != distributed.world_size:
                    raise ValueError('checkpoint is missing worker states')
                alg.load(payload, None, True)
                local_state = info['rank_states'][distributed.rank]
                wrapper.load_training_state_dict(local_state['env_state'])
                restore_rng(local_state['rng'], device)
                first = int(payload['iter']) + 1
                best = info.get('best_score')
                last_validation = info.get('validation')
            else:
                distribution_state = deepcopy(alg.actor.distribution.state_dict())
                alg.actor.load_state_dict(payload['actor_state_dict'])
                alg.actor.distribution.load_state_dict(distribution_state)
        if not resume:
            wrapper.episode_length_buf = torch.randint_like(
                wrapper.episode_length_buf, high=wrapper.max_episode_length)
        alg.train_mode()
        if distributed.world_size > 1:
            alg.broadcast_parameters()
        iteration = first - 1
        if distributed.main:
            trace = (directory / 'metrics.jsonl').open('a', buffering=1)
        statistics = TrainingStatistics(exp.num_envs, env.num_actions, env._reward_term_names, cfg, device)
        iteration_seconds = 0.
        observation = wrapper.get_observations()

        def save(*paths):
            local_state = cpu_state({'env_state': wrapper.training_state_dict(),
                                     'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                                             'torch': torch.get_rng_state(),
                                             'cuda': torch.cuda.get_rng_state(device) if device.startswith('cuda') else None}})
            rank_states = distributed.gather(local_state)

            def write():
                payload = alg.save()
                payload.update(iter=iteration, infos={
                    POLICY_MANIFEST_KEY: manifest.metadata(), 'rank_states': rank_states,
                    'world_size': distributed.world_size, 'num_envs_per_rank': exp.num_envs,
                    'experiment': exp.metadata(), 'source_digest': source_digest, 'best_score': best,
                    'validation': last_validation, 'training_signature': signature,
                    'terrain_sha256': terrain_info['sha256'] if terrain_info else None})
                for path in paths:
                    temporary = path.with_suffix('.tmp')
                    torch.save(payload, temporary)
                    temporary.replace(path)
                    write_policy_manifest_sidecar(path, manifest)
            distributed.on_main(write)

        report('running', source_digest=source_digest)
        for iteration in range(first, first + exp.iterations):
            stopped = distributed.stop_reason(stop_requested, budget)
            if stopped:
                iteration -= 1
                reason = stopped
                break
            clip_rates, saturation, numerical_resets = [], [], []
            statistics.begin_update()

            def record(action, reward, done, extras):
                if cfg.action_clip:
                    clip_rates.append((action.abs() >= cfg.action_clip).float().mean())
                if 'transition_state' in extras:
                    effort = extras['transition_state']['joint_effort']
                    limits = effort.new_tensor(cfg.robot.effort_limit)
                    saturation.append((effort.abs() >= .95 * limits).float().mean())
                if 'numerical_reset' in extras:
                    numerical_resets.append(extras['numerical_reset'].sum())
                if 'numerical_failure' in extras:
                    diagnostic = directory / f'numerical_failure_rank{distributed.rank}.pt'
                    torch.save(extras['numerical_failure'], diagnostic)
                    print(f'Numerical physics failure on rank {distributed.rank}; reset affected worlds; state: {diagnostic}', flush=True)
                statistics.record(action, reward, done, extras['training_stats'])

            timings = {}
            observation, losses = update_policy(alg, wrapper, observation, config['num_steps_per_env'], record, distributed, timings)
            completed += 1
            effective_std = alg.actor.output_std.detach()
            averaged = {'action_std': float(effective_std.mean()),
                        'action_clip_fraction': float(torch.stack(clip_rates).mean()) if clip_rates else 0.,
                        'joint_effort_saturation_fraction': float(torch.stack(saturation).mean()) if saturation else 0.,
                        **losses}
            reset_count = int(torch.stack(numerical_resets).sum()) if numerical_resets else 0
            sums = distributed.reduce([reset_count, *averaged.values()])
            summary = statistics.summarize(distributed, env)
            duration, std_max, collection_s, learning_s = distributed.reduce(
                [timings['collection_s'] + timings['learning_s'], float(effective_std.max()),
                 timings['collection_s'], timings['learning_s']], maximum=True)
            iteration_seconds += duration
            steps_per_update = exp.num_envs * distributed.world_size * config['num_steps_per_env']
            metrics = {'iteration': iteration, 'total_steps': (iteration + 1) * steps_per_update,
                       'steps_per_second': steps_per_update / duration, 'iteration_s': duration,
                       'collection_s': collection_s, 'learning_s': learning_s,
                       'numerical_resets': int(sums[0]), **summary,
                       'action_std_max': std_max, 'learning_rate': alg.learning_rate,
                       'world_size': distributed.world_size, 'total_num_envs': exp.num_envs * distributed.world_size,
                       **{key: value / distributed.world_size for key, value in zip(averaged, sums[1:], strict=True)}}

            def log():
                metrics['elapsed_s'] = budget.elapsed
                trace.write(json.dumps(metrics) + '\n')
                print(format_training_log(metrics, target_iteration=first + exp.iterations,
                    remaining_iterations=first + exp.iterations - iteration - 1,
                    average_iteration_s=iteration_seconds / completed), flush=True)
                budget.update(updates=completed, last_iteration=iteration)
                report('running', updates_completed=completed, elapsed_s=budget.elapsed, latest_metrics=metrics)
            distributed.on_main(log)
            if (iteration + 1) % exp.save_interval == 0:
                save(directory / 'latest.pt')
            if not no_eval and (iteration + 1) % exp.eval_interval == 0:
                candidate = directory / f'candidate_{iteration:06d}.pt'
                save(candidate)
                validation, stopped = _evaluate_checkpoint(
                    exp, candidate, directory / f'eval_{iteration:06d}.json', distributed=distributed,
                    backend=backend, budget=budget, budget_file=budget_file,
                    stop_requested=lambda: stop_requested, report=report, completed=completed, iteration=iteration)
                if stopped:
                    reason = stopped
                    break
                last_validation = validation
                score = validation['score']
                if best is None or tuple(score) > tuple(best):
                    best = score
                    save(directory / 'best.pt')
                    distributed.on_main(lambda: atomic_json(directory / 'best_eval.json', validation))
                save(candidate)
                if distributed.main:
                    print('VALIDATION', json.dumps({'iteration': iteration, 'score': score,
                                                  'passed': validation['passed']}), flush=True)
        save(directory / 'latest.pt', directory / 'model_final.pt')

        def finish():
            summary = {'completed_updates': completed, 'last_iteration': iteration, 'stop_reason': reason,
                       'best_score': best, 'checkpoint': str(directory / 'model_final.pt'),
                       'elapsed_s': budget.elapsed, 'world_size': distributed.world_size,
                       'total_num_envs': exp.num_envs * distributed.world_size,
                       'admitted': bool(last_validation and last_validation.get('passed')),
                       'stage_passed': bool(last_validation and last_validation.get('stage_passed', last_validation.get('passed')))}
            atomic_json(directory / 'summary.json', summary)
            report('stopped' if reason == 'signal' else 'completed', **summary)
            print('FINISHED', json.dumps(summary), flush=True)
            return summary
        return distributed.on_main(finish)
    except BaseException as error:
        report('failed', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if trace is not None:
                trace.close()
            if wrapper is not None:
                wrapper.close()
            elif env is not None:
                env.close()
        finally:
            try:
                resources.close()
                if budget is not None:
                    budget.close(updates=completed, last_iteration=iteration)
            finally:
                for sig, handler in old_signals.items():
                    signal.signal(sig, handler)
                distributed.close()
