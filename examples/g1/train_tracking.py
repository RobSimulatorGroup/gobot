"""Train scene-authored SuperMimic recipes locally or as one process per cluster GPU."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import platform
import tarfile
import time
import subprocess
import sys
import torch
import gobot

from gobot.rl.motion_tracking import MotionTrackingCfg,MotionRandomizationCfg,make_motion_tracking_env
from gobot.rl.motion_tracking.training import TrackingTrainingSession
from gobot.rl.locomotion.contract import robot_physics_digest
from gobot.rl.locomotion.budget import TrainingBudget
from gobot.rl.experiment_identity import canonical_digest,file_digest
from gobot.sim.providers.base import CompiledSceneArtifact
from examples.locomotion.budget import DEFAULT_LEDGER
from examples.locomotion.supermimic_reference import RunStatus,write_json
from .tracking_experiment import TrackingExperiment,require_tracking_admission
from .tracking_video_queue import VideoSchedule

ROOT=Path(__file__).resolve().parents[2]


def snapshot_sources(output):
    from examples.locomotion.package import source_files
    suffixes={'.py','.pyi','.cpp','.hpp','.c','.h','.cmake','.in','.sh','.sbatch'}
    paths={path for path in source_files(ROOT) if path.suffix in suffixes or
           path.name in ('CMakeLists.txt','pyproject.toml','uv.lock')}
    records={str(path.relative_to(ROOT)):file_digest(path) for path in sorted(paths)}
    with tarfile.open(output/'source_overlay.tar.gz','w:gz',dereference=True) as archive:
        for name in records:archive.add(ROOT/name,arcname=name,recursive=False)
    with tarfile.open(output/'source_overlay.tar.gz') as archive:
        for name,digest in records.items():
            if 'sha256:'+hashlib.sha256(archive.extractfile(name).read()).hexdigest()!=digest:
                raise RuntimeError('source files changed during experiment preparation')
    write_json(output/'source_hashes.json',records)
    runtime={name:digest for name,digest in records.items() if not name.startswith('tests/') and
             not name.endswith(('.sh','.sbatch'))}
    return canonical_digest(runtime)


def preflight_signature(project,scene_path,solver):
    context=gobot.app.create_context()
    try:
        context.set_project_path(str(project));context.load_scene(scene_path)
        context.fixed_time_step=.005;context.set_mujoco_solver_settings(solver)
        artifact=CompiledSceneArtifact.from_mapping(context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu))
        return robot_physics_digest(artifact,'g1_29dof')
    finally:context.clear_scene()


def run(recipe,output,*,backend='mujoco-warp',ledger=DEFAULT_LEDGER,resume=None,fork=None,
        source_manifest=None,source_identity=None,admission=None,promotion=None):
    if resume and fork:raise ValueError('choose resume or fork')
    if (source_manifest is not None or source_identity is not None) and fork is None:
        raise ValueError('source metadata is only used with an explicit --fork')
    if promotion is not None and resume is None:
        raise ValueError('--promote requires --resume of the validated adaptation checkpoint')
    if int(os.environ.get('WORLD_SIZE','1'))>1:
        raise ValueError('use independent jobs per GPU; distributed synchronized training is not enabled')
    if recipe.purpose=='training' and admission is None:
        raise ValueError('formal training requires --admission with a passed full-library fidelity report')
    report=json.loads(Path(admission).read_text()) if admission else None
    if report is not None and report.get('passed') is not True:
        raise ValueError('migration fidelity has not passed; formal training is not admitted')
    project=ROOT/'examples/g1';output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'last.pt').exists() and resume is None:
        raise ValueError('existing run has a checkpoint; use --resume or a new output directory')
    source=json.loads((project/'assets/motions/tracking/manifest.json').read_text())
    records=[row for row in source['clips'] if recipe.split=='coverage40' or row['split']==recipe.split]
    files=tuple(str(project/'assets/motions/tracking'/row['file']) for row in records)
    for path,row in zip(files,records,strict=True):
        if file_digest(path)!='sha256:'+row['sha256']:raise ValueError(f'motion checksum changed: {path}')
    cfg=MotionTrackingCfg(project_path=project,motion_files=files,split=recipe.split,
        eval_mode=False,reset_noise=True,deterministic=recipe.deterministic,
        gamma=recipe.learner_config().gamma,goal_frames=recipe.learner_config().goal_frames,
        randomization=MotionRandomizationCfg(**recipe.randomization))
    profile=json.loads((project/cfg.robot_profile).read_text())
    if report is not None:
        signature=preflight_signature(project,cfg.scene_path,profile['solver_settings'])
        require_tracking_admission(report,robot_signature=signature,clip_names=[r['name'] for r in source['clips']])
    if recipe.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    gpu=backend=='mujoco-warp'
    budget=TrainingBudget(ledger,total_hours=recipe.total_gpu_hours,run_hours=recipe.hours,
        metadata={'stage':'gobot_tracking_training','output':str(output),'recipe':recipe.name}) if gpu else None
    status=RunStatus(output,budget) if budget else None
    started=time.monotonic();cancelled=False;env=None;session=None
    video=VideoSchedule(output,interval_seconds=recipe.video_interval_minutes*60,start=started)
    video_process=None;video_log=None;video_disabled=False
    last_save=started;last_progress=0.
    previous_handlers={}
    def cancel(*_):
        nonlocal cancelled
        cancelled=True
    for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGUSR1):
        previous_handlers[signum]=signal.signal(signum,cancel)
    def should_stop():
        remaining=budget.remaining if budget else recipe.hours*3600-(time.monotonic()-started)
        return cancelled or remaining<recipe.stop_reserve_seconds
    def progress(iteration,steps,phase):
        nonlocal last_progress,last_save,video_process,video_log,video_disabled
        now=time.monotonic()
        if status and now-last_progress>=2:
            status.update(status='training',iteration=iteration,sim_steps=steps,phase=phase)
            last_progress=now
        if session is not None and now-last_save>=recipe.checkpoint_minutes*60:
            session.save();last_save=now
        if session is not None and recipe.purpose=='training' and recipe.video_mode!='off':
            video.enqueue(session,now=now)
            if video_process is not None and not video_disabled and video_process.poll() not in (None,0):
                video_disabled=True
                write_json(output/'video_worker_failure.json',{'returncode':video_process.returncode,
                    'iteration':iteration,'requests_retained':True,'automatic_restart':False})
            if recipe.video_mode=='render' and gpu and not video_disabled and (video_process is None or video_process.poll() is not None):
                requests=list((output/'video_requests').glob('*.json'))
                pending=any(not (output/'video_results'/path.name).exists() for path in requests)
                if pending:
                    if video_log:video_log.close()
                    video_log=(output/'video_worker.log').open('a',buffering=1)
                    video_process=subprocess.Popen([sys.executable,'-u','-m','examples.g1.tracking_video_queue',
                        '--run-dir',str(output),'--budget-file',str(Path(ledger).resolve()),'--budget-parent-pid',str(os.getpid())],
                        cwd=ROOT,stdout=video_log,stderr=subprocess.STDOUT)
    try:
        versions={name:importlib.metadata.version(name) for name in ('torch','mujoco','mujoco-warp','warp-lang')}
        hardware={'backend':backend,'python':platform.python_version(),'platform':platform.platform(),
                  'scheduler_job':os.environ.get('SLURM_JOB_ID'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
        if gpu:
            properties=torch.cuda.get_device_properties(0)
            hardware.update(gpu=properties.name,memory_bytes=properties.total_memory,
                            compute_capability=[properties.major,properties.minor],cuda_runtime=torch.version.cuda)
        write_json(output/'hardware.json',hardware)
        software=canonical_digest({'sources':snapshot_sources(output),'versions':versions})
        write_json(output/'experiment.json',recipe.metadata());write_json(output/'versions.json',versions)
        write_json(output/'task_config.json',{**asdict(cfg),'project_path':str(project)})
        if report:write_json(output/'admission.json',report)
        env=make_motion_tracking_env(cfg,num_envs=recipe.num_envs,backend=backend,seed=recipe.seed)
        session=TrackingTrainingSession(env,recipe.learner_config(),output,software_digest=software,
            should_stop=should_stop,progress=progress)
        if resume:session.load(resume)
        elif fork:session.fork(fork,source_manifest=source_manifest,source_identity=source_identity)
        if promotion:
            session.promote(json.loads(Path(promotion).read_text()))
        write_json(output/'contract.json',{'digest':session.contract_digest,'contract':session.contract})
        result=session.train()
        if recipe.purpose=='training' and recipe.video_mode!='off':video.enqueue(session,force=True)
        result.update(phase=session.phase,purpose=recipe.purpose,elapsed_s=time.monotonic()-started,
                      checkpoint=str(output/'last.pt'),contract_digest=session.contract_digest)
        if gpu:
            result['peak_torch_allocated_bytes']=torch.cuda.max_memory_allocated(0)
            result['peak_torch_reserved_bytes']=torch.cuda.max_memory_reserved(0)
        write_json(output/'summary.json',result)
        if status:status.update(status='finished' if result.get('completed') else 'stopped',iteration=session.iteration)
        return result
    except BaseException as error:
        if status:status.update(status='failed',error=str(error))
        write_json(output/'failure.json',{'error':str(error),'elapsed_s':time.monotonic()-started})
        raise
    finally:
        if video_process is not None and video_process.poll() is None:
            video_process.terminate()
            try:video_process.wait(timeout=30)
            except subprocess.TimeoutExpired:video_process.kill();video_process.wait()
        if video_log:video_log.close()
        if env:env.close()
        if status:status.close()
        for signum,handler in previous_handlers.items():signal.signal(signum,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=('mujoco-cpu','mujoco-warp'),default='mujoco-warp')
    p.add_argument('--ledger',type=Path,default=DEFAULT_LEDGER)
    p.add_argument('--seed',type=int);p.add_argument('--num-envs',type=int);p.add_argument('--iterations',type=int)
    p.add_argument('--world-model-batch',type=int);p.add_argument('--policy-batch',type=int)
    p.add_argument('--hours',type=float);p.add_argument('--admission',type=Path)
    p.add_argument('--video-mode',choices=('queue','render','off'))
    modes=p.add_mutually_exclusive_group();modes.add_argument('--resume',type=Path);modes.add_argument('--fork',type=Path)
    p.add_argument('--source-manifest',type=Path);p.add_argument('--source-identity',type=Path)
    p.add_argument('--promote',type=Path,help='independent validation report for a resumed adaptation checkpoint')
    a=p.parse_args();recipe=TrackingExperiment.load(a.config)
    changes={key:getattr(a,key) for key in ('seed','num_envs','hours','video_mode') if getattr(a,key) is not None}
    algorithm_changes={key:value for key,value in (('iterations',a.iterations),('wm_batch',a.world_model_batch),('pi_batch',a.policy_batch)) if value is not None}
    if algorithm_changes:changes['algorithm']=recipe.algorithm|algorithm_changes
    recipe=replace(recipe,**changes)
    print(json.dumps(run(recipe,a.output,backend=a.backend,ledger=a.ledger,resume=a.resume,fork=a.fork,
        source_manifest=a.source_manifest,source_identity=a.source_identity,admission=a.admission,promotion=a.promote),indent=2))


if __name__=='__main__':main()
