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
from rsl_rl.runners import OnPolicyRunner
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


def restore_rng(rng):
    random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'].cpu())
    if rng['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all([v.cpu() for v in rng['cuda']])


def update_policy(algorithm, environment, observation, steps, record=None):
    """Use RSL-RL's rollout, GAE and PPO update without modifying the optimizer."""
    from rsl_rl.utils import check_nan

    with torch.inference_mode():
        for _ in range(steps):
            action = algorithm.act(observation)
            observation, reward, done, extras = environment.step(action)
            check_nan(observation, reward, done)
            algorithm.process_env_step(observation, reward, done, extras)
            if record is not None:
                record(action, reward, done, extras)
        algorithm.compute_returns(observation)
    losses = algorithm.update()
    if not all(np.isfinite(float(value)) for value in losses.values()):
        raise FloatingPointError('non-finite optimizer metric')
    return observation, losses


def run(exp, directory, *, backend='mujoco-warp', device=None, budget_file=None, resume=None, initialize=None, no_eval=False, task_cfg=None, agent_cfg=None):
    """Train a built-in velocity recipe, optionally with explicit task/agent settings.

    Custom task settings require an independent evaluator; use ``no_eval=True``
    rather than evaluating a checkpoint against a different built-in preset.
    ``agent_cfg`` is a complete RSL-RL configuration and is never mutated.
    """
    if backend not in {'mujoco-cpu', 'mujoco-warp'}:
        raise ValueError(f'unsupported locomotion training backend {backend!r}')
    if task_cfg is not None and not no_eval:
        raise ValueError('custom task_cfg requires no_eval=True; the built-in independent evaluator only supports built-in presets')
    cfg = deepcopy(task_cfg) if task_cfg is not None else task_config(exp)
    config = deepcopy(agent_cfg) if agent_cfg is not None else agent_config(exp, experiment_name=cfg.name)
    validate_agent_config(exp, config)
    validate_task_config(exp, cfg, config)
    # RSL-RL consumes class names and modifies its dictionary during construction.
    # Checkpoint compatibility must retain the user's complete effective recipe.
    effective_agent_config = deepcopy(config)
    if int(os.environ.get('WORLD_SIZE','1'))>1:
        raise ValueError('Use one independent experiment per GPU; synchronized AMP is not enabled')
    if resume and initialize:raise ValueError('choose resume or initialize, not both')
    directory=Path(directory)
    if directory.is_dir() and any(directory.iterdir()) and resume is None:
        raise ValueError(f'output directory is not empty: {directory}; choose a new output or resume explicitly')
    directory.mkdir(parents=True,exist_ok=True)
    device=device or ('cuda:0' if backend=='mujoco-warp' else 'cpu')
    budget_file=Path(budget_file or _default_budget_file(directory, backend))
    status={'state':'starting','pid':os.getpid(),'host':socket.gethostname(),'started_unix':time.time(),
            'pid_namespace':os.readlink('/proc/self/ns/pid'),
            'process_start_ticks':Path('/proc/self/stat').read_text().split()[21], 'experiment':exp.metadata()}
    stop_requested=False
    def stop(*_):
        nonlocal stop_requested
        stop_requested=True
    old_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in old_signals:
        signal.signal(sig, stop)
    budget=None;env=None;wrapper=None;trace=None;alg=None
    iteration=-1;best=None;completed=0;reason='iterations'
    def report(state,**values):
        status.update(state=state,heartbeat_unix=time.time(),iteration=iteration,**values)
        atomic_json(directory/'status.json',status)
    try:
        report('starting')
        budget=TrainingBudget(budget_file,total_hours=exp.budget_hours,run_hours=exp.hours,
                              metadata={'robot':exp.robot,'stage':exp.stage,'log_dir':str(directory)})
        random.seed(exp.seed);np.random.seed(exp.seed);torch.manual_seed(exp.seed);torch.set_num_threads(4)
        env=make_velocity_env(cfg,backend=backend,num_envs=exp.num_envs,device=device,seed=exp.seed,collect_step_extras=False)
        use_amp='motion_prior_cfg' in config['algorithm']
        wrapper=RslRlVecEnvWrapper(env,device=device)
        manifest=build_policy_manifest(wrapper,cfg,config,Path(cfg.project_path))
        source_digest=snapshot_sources(directory)
        atomic_json(directory/'experiment.json',exp.metadata())
        atomic_json(directory/'config.json',{'experiment':exp.metadata(),'task':asdict(cfg),'ppo':config,'source_digest':source_digest})
        runner=OnPolicyRunner(wrapper,config,log_dir=None,device=device);alg=runner.alg
        if exp.reference_reset_probability:
            if backend!='mujoco-warp' or not exp.stage.startswith('flat') or exp.robot!='g1':
                raise ValueError('reference reset is supported only for G1 flat Warp training')
            library=alg.reference if use_amp else exp.motion_dataset
            env.set_reference_initializer(ReferenceStateInitializer(library,device=device,probability=exp.reference_reset_probability))
            env.reset(seed=exp.seed)
        task_identity=asdict(cfg);task_identity.pop('project_path',None)
        semantic=exp.metadata()
        for name in ('num_envs','iterations','hours','budget_hours','save_interval','eval_interval','eval_envs','eval_steps','eval_seed'):
            semantic.pop(name,None)
        signature_data={'experiment':semantic,'task':task_identity,'backend':backend}
        signature_data['agent']=effective_agent_config
        signature=hashlib.sha256(json.dumps(signature_data,sort_keys=True,default=str).encode()).hexdigest()
        atomic_json(directory/'training_signature.json',{'schema':2,'signature':signature})

        if use_amp and alg.reference.metadata['robot_digest']!=manifest.scene_digest:
            raise ValueError('reference dataset physics differs from authored robot')
        first=0
        if resume or initialize:
            payload=torch.load(resume or initialize,map_location=device,weights_only=False)
            validate_policy(policy_manifest_from_checkpoint(payload),env,cfg)
            previous=Experiment(**payload['infos']['experiment'])
            if resume:
                if payload['infos'].get('training_signature')!=signature:
                    raise ValueError('resume configuration signature changed; use an explicit fork')
                semantic=('robot','recipe','stage','reference_reset_probability','motion_dataset','style_weight','learning_rate','entropy_coef','seed')
                if any(getattr(previous,k)!=getattr(exp,k) for k in semantic):
                    raise ValueError('resume requires the same recipe; use initialize to start a new experiment')
                alg.load(payload,None,True)
                wrapper.load_training_state_dict(payload['infos']['env_state'])
                restore_rng(payload['infos']['rng'])
                first=int(payload['iter'])+1
                best=payload['infos'].get('best_score')
            else:
                # Load the actor and normalizer; optimizer and critic start fresh.
                distribution_state = {key: value.clone() for key, value in alg.actor.distribution.state_dict().items()}
                alg.actor.load_state_dict(payload['actor_state_dict'])
                alg.actor.distribution.load_state_dict(distribution_state)
        # Match the reference runner's initial episode-length randomization.
        wrapper.episode_length_buf = torch.randint_like(
            wrapper.episode_length_buf, high=wrapper.max_episode_length)
        iteration=first-1
        trace=(directory/'metrics.jsonl').open('a',buffering=1)
        total=torch.zeros(exp.num_envs,device=device);lengths=torch.zeros_like(total)
        observation=wrapper.get_observations();alg.train_mode();last_validation=None
        def save(path):
            payload=alg.save()
            payload.update(iter=iteration,infos={POLICY_MANIFEST_KEY:manifest.metadata(),'env_state':wrapper.training_state_dict(),
                'experiment':exp.metadata(),'source_digest':source_digest,'best_score':best,
                'validation':last_validation,'training_signature':signature,
                'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                       'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}})
            temporary=path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(path)
            write_policy_manifest_sidecar(path,manifest)
        report('running',source_digest=source_digest)
        for iteration in range(first,first+exp.iterations):
            if stop_requested or budget.remaining<=0:
                iteration-=1;reason='signal' if stop_requested else 'budget';break
            tick=time.monotonic();episode_rewards=[];episode_lengths=[];clip_rates=[];saturation=[]
            def record(action, reward, done, extras):
                if cfg.action_clip:
                    clip_rates.append((action.abs() >= cfg.action_clip).float().mean())
                if 'transition_state' in extras:
                    effort = extras['transition_state']['joint_effort']
                    limits = effort.new_tensor(cfg.robot.effort_limit)
                    saturation.append((effort.abs() >= .95 * limits).float().mean())
                total.add_(reward); lengths.add_(1)
                if bool(done.any()):
                    episode_rewards.extend(total[done].tolist())
                    episode_lengths.extend(lengths[done].tolist())
                    total[done] = 0; lengths[done] = 0
            observation, losses = update_policy(alg, wrapper, observation, config['num_steps_per_env'], record)
            completed += 1
            effective_std = alg.actor.output_std.detach()
            metrics={'iteration':iteration,'elapsed_s':budget.elapsed,'steps_per_second':exp.num_envs*config['num_steps_per_env']/(time.monotonic()-tick),
                     'episode_reward':float(np.mean(episode_rewards)) if episode_rewards else None,
                     'episode_steps':float(np.mean(episode_lengths)) if episode_lengths else None,
                     'action_std':float(effective_std.mean()),'action_std_max':float(effective_std.max()),
                     'action_clip_fraction':float(torch.stack(clip_rates).mean()) if clip_rates else 0.,
                     'joint_effort_saturation_fraction':float(torch.stack(saturation).mean()) if saturation else 0.,
                     'terrain_level':float(torch.as_tensor(env.terrain_assignments[0]).float().mean()),**losses}
            trace.write(json.dumps(metrics)+'\n')
            if iteration%25==0:print(json.dumps(metrics),flush=True)
            budget.update(updates=completed,last_iteration=iteration)
            report('running',updates_completed=completed,elapsed_s=budget.elapsed,latest_metrics=metrics)
            if (iteration+1)%exp.save_interval==0:save(directory/'latest.pt')
            if not no_eval and (iteration+1)%exp.eval_interval==0:
                candidate=directory/f'candidate_{iteration:06d}.pt';save(candidate)
                output=directory/f'eval_{iteration:06d}.json'
                report('evaluating',checkpoint=str(candidate))
                command=[sys.executable,'-B','-c',
                         "from gobot.rl.training.projects import example_module; example_module('locomotion.assessment').main()",
                         '--checkpoint',str(candidate),'--output',str(output),
                         '--seed',str(exp.eval_seed),'--steps',str(exp.eval_steps),'--num-envs',str(exp.eval_envs),'--device',device,'--backend',backend,
                         '--budget-file',str(budget_file.resolve()),'--budget-parent-pid',str(os.getpid())]
                with (directory/'evaluation.log').open('a',buffering=1) as stream:
                    process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                    try:
                        while process.poll() is None:
                            if stop_requested or budget.remaining<=0:
                                process.terminate()
                                try:process.wait(timeout=30)
                                except subprocess.TimeoutExpired:process.kill();process.wait()
                                break
                            try:process.wait(timeout=5)
                            except subprocess.TimeoutExpired:pass
                            budget.update(updates=completed,last_iteration=iteration)
                            report('evaluating',checkpoint=str(candidate))
                    finally:
                        if process.poll() is None:process.terminate();process.wait(timeout=30)
                if stop_requested or budget.remaining<=0:reason='signal' if stop_requested else 'budget';break
                if process.returncode:raise RuntimeError(f'evaluator exited {process.returncode}; see {directory/"evaluation.log"}')
                last_validation=json.loads(output.read_text());score=last_validation['score']
                if best is None or tuple(score)>tuple(best):
                    best=score;save(directory/'best.pt');atomic_json(directory/'best_eval.json',last_validation)
                # Keep every screening checkpoint: later audits must be able to recover a different candidate.
                save(candidate)
                print('VALIDATION',json.dumps({'iteration':iteration,'score':score,'passed':last_validation['passed']}),flush=True)
        save(directory/'latest.pt');save(directory/'model_final.pt')
        summary={'completed_updates':completed,'last_iteration':iteration,'stop_reason':reason,'best_score':best,
                 'checkpoint':str(directory/'model_final.pt'),'elapsed_s':budget.elapsed,'admitted':bool(last_validation and last_validation.get('passed')),
                 'stage_passed':bool(last_validation and last_validation.get('stage_passed',last_validation.get('passed')))}
        atomic_json(directory/'summary.json',summary)
        report('stopped' if reason=='signal' else 'completed',**summary)
        print('FINISHED',json.dumps(summary),flush=True)
        return summary
    except BaseException as error:
        report('failed',error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        if trace is not None:trace.close()
        try:
            if wrapper is not None:wrapper.close()
            elif env is not None:env.close()
        finally:
            try:
                if budget is not None:budget.close(updates=completed,last_iteration=iteration)
            finally:
                for sig, handler in old_signals.items():
                    signal.signal(sig, handler)
