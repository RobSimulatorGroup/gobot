"""Bounded, resumable PPO/AMP experiments from Gobot-authored scenes."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import json
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
from gobot.rl.locomotion.guarded_ppo import PpoGuardCfg
from gobot.rl.policy import POLICY_MANIFEST_KEY, policy_manifest_from_checkpoint, write_policy_manifest_sidecar
from .experiment import Experiment, task_config
from .budget import DEFAULT_LEDGER


def atomic_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,default=str)+'\n');temp.replace(path)


def snapshot_sources(directory):
    root=Path(__file__).resolve().parents[2]
    files=[]
    for folder in ('python/gobot/rl','examples/g1','examples/go1','examples/locomotion'):
        files.extend(p for p in (root/folder).rglob('*') if p.is_file() and p.suffix in {'.py','.jscn','.jres','.json'}
                     and not any(x in p.parts for x in ('logs','assets','policies','__pycache__')))
    files.extend(root/name for name in ('pyproject.toml','uv.lock','include/gobot/simulation/locomotion_command_runtime.hpp',
        'src/gobot/simulation/locomotion_command_runtime.cpp','src/gobot/python/manual_bindings/manual_binding_app_context.cpp'))
    digest=hashlib.sha256()
    with tarfile.open(directory/'sources.tar.gz','w:gz') as archive:
        for path in sorted(set(files)):
            digest.update(str(path.relative_to(root)).encode());digest.update(path.read_bytes())
            archive.add(path,arcname=str(path.relative_to(root)))
    (directory/'sources.sha256').write_text(digest.hexdigest()+'\n')
    return digest.hexdigest()


def restore_rng(rng):
    random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'].cpu())
    if rng['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all([v.cpu() for v in rng['cuda']])


def run(exp, directory, *, backend='mujoco-warp', device=None, budget_file=None, resume=None, initialize=None, no_eval=False, require_gate=True):
    if int(os.environ.get('WORLD_SIZE','1'))>1:
        raise ValueError('Use one independent experiment per GPU; synchronized AMP is not enabled')
    if resume and initialize:raise ValueError('choose resume or initialize, not both')
    if require_gate and exp.stage != 'flat_control' and not (resume or initialize):
        raise ValueError('advanced stages require an admitted initialization checkpoint')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    device=device or ('cuda:0' if backend=='mujoco-warp' else 'cpu')
    budget_file=Path(budget_file or (DEFAULT_LEDGER if backend=='mujoco-warp' else directory/'gpu_budget.json'))
    status={'state':'starting','pid':os.getpid(),'host':socket.gethostname(),'started_unix':time.time(),
            'pid_namespace':os.readlink('/proc/self/ns/pid'),
            'process_start_ticks':Path('/proc/self/stat').read_text().split()[21], 'experiment':exp.metadata()}
    stop_requested=False
    def stop(*_):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    budget=None;env=None;wrapper=None;trace=None;alg=None
    iteration=-1;best=None;completed=0;reason='iterations';stale=0;collapsed=0
    def report(state,**values):
        status.update(state=state,heartbeat_unix=time.time(),iteration=iteration,**values)
        atomic_json(directory/'status.json',status)
    try:
        report('starting')
        budget=TrainingBudget(budget_file,total_hours=exp.budget_hours,run_hours=exp.hours,
                              metadata={'robot':exp.robot,'stage':exp.stage,'log_dir':str(directory)})
        random.seed(exp.seed);np.random.seed(exp.seed);torch.manual_seed(exp.seed);torch.set_num_threads(4)
        cfg=task_config(exp)
        env=make_velocity_env(cfg,backend=backend,num_envs=exp.num_envs,device=device,seed=exp.seed,collect_step_extras=False)
        use_amp=exp.robot=='g1' and exp.recipe=='natural'
        wrapper=RslRlVecEnvWrapper(env,device=device)
        config=rsl_rl_train_cfg(experiment_name=cfg.name,obs_normalization=True)
        config['actor']['distribution_cfg'].update(init_std=exp.initial_action_std,std_type='log',std_range=(.1,1.))
        config['algorithm'].update(class_name='gobot.rl.locomotion.transition_ppo:TransitionPPO',
                                   learning_rate=exp.learning_rate,schedule='fixed',entropy_coef=exp.entropy_coef,
                                   guard_cfg=asdict(PpoGuardCfg(enabled=exp.schema_version==2 and exp.guarded_updates)))
        if use_amp:
            config['algorithm'].update(class_name='gobot.rl.locomotion.amp:MotionPriorPPO',
                motion_prior_cfg=asdict(MotionPriorCfg(dataset=str(Path(exp.motion_dataset).resolve()),weight=exp.style_weight,seed=exp.seed)))
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
        for name in ('num_envs','iterations','hours','budget_hours','save_interval','eval_interval','eval_envs','eval_steps','eval_seed','patience','stop_on_gate'):
            semantic.pop(name,None)
        signature=hashlib.sha256(json.dumps({'experiment':semantic,'task':task_identity,'backend':backend},sort_keys=True,default=str).encode()).hexdigest()
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
                    raise ValueError('resume requires the same recipe; use initialize for an admitted stage change')
                alg.load(payload,None,True)
                wrapper.load_training_state_dict(payload['infos']['env_state'])
                restore_rng(payload['infos']['rng'])
                first=int(payload['iter'])+1
                best=payload['infos'].get('best_score');stale=payload['infos'].get('stale_evaluations',0)
            else:
                if require_gate:
                    gate=payload['infos'].get('validation',{})
                    if not gate or not gate.get('stage_passed',gate.get('passed')) or gate.get('protocol')!='validation':
                        raise ValueError('stage initialization requires a checkpoint with a passed validation gate')
                alg.actor.load_state_dict(payload['actor_state_dict'])
                # A fork retains actor/normalizer only; value targets and optimizer state start fresh.
                with torch.no_grad():alg.actor.distribution.log_std_param.fill_(np.log(exp.initial_action_std))
        if not resume and exp.schema_version==2:
            if not initialize:
                linear=[module for module in alg.actor.mlp.modules() if isinstance(module,torch.nn.Linear)][-1]
                torch.nn.init.orthogonal_(linear.weight,gain=.01);torch.nn.init.zeros_(linear.bias)
            report('calibrating')
            probe=[]
            with torch.no_grad():
                for step in range(exp.normalization_warmup_steps):
                    if budget.remaining<=0:raise TimeoutError('budget exhausted during calibration')
                    if step%8==0:wrapper.reset()
                    observation=wrapper.get_observations()
                    for model in ((alg.critic,) if initialize else (alg.actor,alg.critic)):
                        model.obs_normalizer.train();model.update_normalization(observation)
                    current=torch.as_tensor(env.locomotion_state()['joint_position'],device=device)
                    target=((current-current.new_tensor(env.default_joint_pos))/current.new_tensor(env.action_scale)).clamp(-cfg.action_clip,cfg.action_clip)
                    wrapper.step(target)
                    if step%8==0:probe.append(observation['actor'].clone())
                alg.freeze_normalization()
                from tensordict import TensorDict
                batch=torch.cat(probe) if probe else wrapper.get_observations()['actor']
                mean=alg.actor(TensorDict({'actor':batch},batch_size=[len(batch)]),stochastic_output=False)
                rate=float((mean.abs()>cfg.action_clip).float().mean())
                atomic_json(directory/'calibration.json',{'observations':len(batch),'mean_clip_fraction':rate,'maximum_mean_action':float(mean.abs().max())})
                if rate>=.01:raise RuntimeError('calibrated policy exceeds 1% mean-action clipping; training not started')
            wrapper.reset(seed=exp.seed)
        iteration=first-1
        trace=(directory/'metrics.jsonl').open('a',buffering=1)
        total=torch.zeros(exp.num_envs,device=device);lengths=torch.zeros_like(total)
        observation=wrapper.get_observations();alg.train_mode();last_validation=None;passed_evaluations=0
        def save(path):
            payload=alg.save()
            payload.update(iter=iteration,infos={POLICY_MANIFEST_KEY:manifest.metadata(),'env_state':wrapper.training_state_dict(),
                'experiment':exp.metadata(),'source_digest':source_digest,'best_score':best,'stale_evaluations':stale,
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
            with torch.inference_mode():
                for _ in range(config['num_steps_per_env']):
                    action=alg.act(observation)
                    if cfg.action_clip:clip_rates.append((action.abs()>=cfg.action_clip).float().mean())
                    observation,reward,done,extras=wrapper.step(action)
                    if not bool(torch.isfinite(observation['actor']).all() & torch.isfinite(reward).all() & torch.isfinite(action).all()):
                        raise FloatingPointError('non-finite action, observation or reward')
                    effort=extras['transition_state']['joint_effort']
                    limits=effort.new_tensor(cfg.robot.effort_limit)
                    saturation.append((effort.abs() >= .95*limits).float().mean())
                    alg.process_env_step(observation,reward,done,extras)
                    total+=reward;lengths+=1
                    if bool(done.any()):
                        episode_rewards.extend(total[done].tolist());episode_lengths.extend(lengths[done].tolist())
                        total[done]=0;lengths[done]=0
                alg.compute_returns(observation)
            losses=alg.update();completed+=1
            if not all(np.isfinite(float(v)) for v in losses.values()):raise FloatingPointError('non-finite optimizer metric')
            with torch.no_grad():
                # Prevent an optimizer's unconstrained parameter drifting beyond the effective range.
                dist=alg.actor.distribution;dist.log_std_param.clamp_(*dist.log_std_range)
                effective_std=dist.log_std_param.exp()
            metrics={'iteration':iteration,'elapsed_s':budget.elapsed,'steps_per_second':exp.num_envs*24/(time.monotonic()-tick),
                     'episode_reward':float(np.mean(episode_rewards)) if episode_rewards else None,
                     'episode_steps':float(np.mean(episode_lengths)) if episode_lengths else None,
                     'action_std':float(effective_std.mean()),'action_std_max':float(effective_std.max()),
                     'action_clip_fraction':float(torch.stack(clip_rates).mean()) if clip_rates else 0.,
                     'joint_effort_saturation_fraction':float(torch.stack(saturation).mean()),
                     'terrain_level':float(torch.as_tensor(env.terrain_assignments[0]).float().mean()),**losses}
            trace.write(json.dumps(metrics)+'\n')
            if iteration%25==0:print(json.dumps(metrics),flush=True)
            budget.update(updates=completed,last_iteration=iteration)
            report('running',updates_completed=completed,elapsed_s=budget.elapsed,latest_metrics=metrics)
            if (iteration+1)%exp.save_interval==0:save(directory/'latest.pt')
            if use_amp and alg.update_count>500 and losses.get('amp_style_raw',1.)<.02:collapsed+=1
            else:collapsed=0
            if collapsed>=200:reason='amp_collapse';break
            if not no_eval and (iteration+1)%exp.eval_interval==0:
                candidate=directory/f'candidate_{iteration:06d}.pt';save(candidate)
                output=directory/f'eval_{iteration:06d}.json'
                report('evaluating',checkpoint=str(candidate))
                command=[sys.executable,'-B','-m','examples.locomotion.assessment','--checkpoint',str(candidate),'--output',str(output),
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
                    best=score;stale=0;save(directory/'best.pt');atomic_json(directory/'best_eval.json',last_validation)
                else:stale+=1
                # Keep every screening checkpoint: later audits must be able to recover a different candidate.
                save(candidate)
                print('VALIDATION',json.dumps({'iteration':iteration,'score':score,'passed':last_validation['passed']}),flush=True)
                passed_evaluations=passed_evaluations+1 if last_validation.get('stage_passed',last_validation['passed']) else 0
                if exp.stop_on_gate and passed_evaluations>=2:reason='gate_passed';break
                if stale>=exp.patience:reason='no_improvement';break
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
            if budget is not None:budget.close(updates=completed,last_iteration=iteration)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path);p.add_argument('--robot',choices=('g1','go1'))
    p.add_argument('--recipe',choices=('baseline','natural'));p.add_argument('--stage',choices=('flat_control','flat_gait','rough','mixed'))
    for name,kind in [('num-envs',int),('iterations',int),('seed',int),('hours',float),('budget-hours',float),('eval-interval',int),('eval-steps',int),('eval-envs',int),('save-interval',int)]:p.add_argument('--'+name,type=kind)
    p.add_argument('--backend',choices=('mujoco-cpu','mujoco-warp'),default='mujoco-warp');p.add_argument('--device')
    p.add_argument('--log-dir',type=Path,required=True);p.add_argument('--budget-file',type=Path)
    p.add_argument('--resume',type=Path);p.add_argument('--initialize','--fork',dest='initialize',type=Path);p.add_argument('--no-eval',action='store_true')
    p.add_argument('--log-stdout',type=Path)
    a=p.parse_args()
    if a.log_stdout:
        a.log_stdout.parent.mkdir(parents=True,exist_ok=True);stream=a.log_stdout.open('a',buffering=1);sys.stdout=sys.stderr=stream
    exp=Experiment.load(a.config) if a.config else Experiment()
    changes={k:getattr(a,k) for k in exp.__dataclass_fields__ if hasattr(a,k) and getattr(a,k) is not None}
    exp=replace(exp,**changes)
    run(exp,a.log_dir,backend=a.backend,device=a.device,budget_file=a.budget_file,resume=a.resume,initialize=a.initialize,no_eval=a.no_eval)

if __name__=='__main__':main()
