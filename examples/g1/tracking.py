"""Evaluate frozen SuperMimic teachers or smoke-test scene-authored training.

Long training and stage promotion require physics/version parity first. The
smoke recipe explicitly uses fixed physics; it is not a randomized reproduction.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import time
import tarfile
import hashlib
import torch

from gobot.rl.motion_tracking import MotionTrackingCfg, make_motion_tracking_env
from gobot.rl.motion_tracking.evaluate import evaluate
from gobot.rl.motion_tracking.manifest import build_tracking_manifest
from gobot.rl.supermimic import SuperMimicConfig, SuperMimicLearner
from gobot.rl.policy import POLICY_MANIFEST_KEY,write_policy_manifest_sidecar
from gobot.rl.experiment_identity import file_digest, canonical_digest, freeze_protocol, make_run_identity
from examples.locomotion.supermimic_reference import RunStatus, ControlledStop, write_json
from examples.locomotion.budget import DEFAULT_LEDGER
from gobot.rl.locomotion.budget import TrainingBudget

ROOT = Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('evaluate','smoke-train'))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=('mujoco-cpu','mujoco-warp'),default='mujoco-cpu')
    p.add_argument('--device')
    p.add_argument('--deterministic',action='store_true')
    p.add_argument('--clips',nargs='*')
    p.add_argument('--split',choices=('coverage40','train','validation','holdout'),default='coverage40')
    p.add_argument('--num-envs',type=int,default=4)
    p.add_argument('--max-steps',type=int)
    p.add_argument('--start-offset',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--iterations',type=int,default=3)
    p.add_argument('--capture-graphs',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--hours',type=float,default=.5)
    p.add_argument('--ledger',type=Path,default=DEFAULT_LEDGER)
    args=p.parse_args()
    if args.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True)
    if args.operation=='evaluate' and args.checkpoint is None:p.error('evaluation requires --checkpoint')
    if args.operation=='smoke-train' and args.checkpoint is not None:p.error('smoke training starts fresh; checkpoint imports are evaluation-only')
    if not math.isfinite(args.hours) or args.hours<=0 or args.iterations<1 or args.num_envs<1:p.error('invalid job limit')
    if args.operation=='smoke-train' and args.split in ('validation','holdout'):p.error('training cannot consume validation or holdout splits')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'evaluation.json').exists():p.error('completed evaluation exists; choose a new output directory')
    project=ROOT/'examples/g1'
    manifest=json.loads((project/'assets/motions/tracking/manifest.json').read_text())
    records=[row for row in manifest['clips'] if args.split=='coverage40' or row['split']==args.split]
    if args.clips:
        records=[row for row in records if row['name'] in args.clips]
        if {row['name'] for row in records} != set(args.clips):p.error('unknown clip or clip outside requested split')
    files=tuple(str(project/'assets/motions/tracking'/row['file']) for row in records)
    for path,row in zip(files,records,strict=True):
        if file_digest(path) != 'sha256:'+row['sha256']:raise ValueError(f'motion checksum changed: {path}')
    cfg=MotionTrackingCfg(project_path=project,motion_files=files,split=args.split,
        eval_mode=args.operation=='evaluate',reset_noise=args.operation=='smoke-train',deterministic=args.deterministic)
    device=args.device or ('cuda:0' if args.backend=='mujoco-warp' else 'cpu')
    gpu=args.backend=='mujoco-warp' or device.startswith('cuda')
    allocation=TrainingBudget(args.ledger,total_hours=24,run_hours=args.hours,
        metadata={'stage':'gobot_tracking_'+args.operation,'output':str(args.output.resolve())}) if gpu else None
    status=RunStatus(args.output,allocation) if allocation else None
    start=time.monotonic();cancelled=False;env=None
    def cancel(*_):
        nonlocal cancelled
        cancelled=True
    signal.signal(signal.SIGINT,cancel);signal.signal(signal.SIGTERM,cancel)
    def check():
        if cancelled or (allocation and allocation.remaining<5) or time.monotonic()-start>args.hours*3600:
            raise ControlledStop('job cancelled or time limit reached')
    try:
        torch.set_num_threads(4)
        env=make_motion_tracking_env(cfg,num_envs=len(files) if args.operation=='evaluate' else args.num_envs,
            backend=args.backend,device=device,seed=args.seed,capture_graphs=args.capture_graphs)
        if args.operation=='evaluate':
            payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
            algorithm=payload['args']
        else:
            algorithm=SuperMimicConfig(iterations=args.iterations,buffer_steps=64,wm_batch=8,pi_batch=8,
                test_hidden_dims=(32,32),warmup_steps=12,wm_batches=1,wm_only_iters=1,
                critic_epochs=1,critic_minibatches=1,amp='none',compile=False,log_every=1,
                ckpt_every=1,seed=args.seed)
        policy=build_tracking_manifest(env,algorithm)
        versions={name:importlib.metadata.version(name) for name in ('torch','mujoco','mujoco-warp','warp-lang')}
        sources={str(path.relative_to(ROOT)):file_digest(path) for folder in ('python/gobot/rl','python/gobot/sim/providers',
                 'src/gobot/physics','src/gobot/core/io','src/gobot/python','src/gobot/scene','include/gobot/physics','include/gobot/scene')
                 for path in sorted((ROOT/folder).rglob('*')) if path.suffix in ('.py','.cpp','.hpp')}
        for source in (Path(__file__).resolve(),ROOT/'pyproject.toml',ROOT/'CMakeLists.txt'):
            sources[str(source.relative_to(ROOT))]=file_digest(source)
        archive_path=args.output/'source_overlay.tar.gz'
        with tarfile.open(archive_path,'w:gz') as archive:
            for name in sorted(sources):archive.add(ROOT/name,arcname=name,recursive=False)
        with tarfile.open(archive_path) as archive:
            for name,digest in sources.items():
                if 'sha256:'+hashlib.sha256(archive.extractfile(name).read()).hexdigest()!=digest:
                    raise RuntimeError('source changed while snapshotting the experiment')
        import gobot
        native_root=Path(gobot._core.__file__).parent
        native={p.name:file_digest(p) for p in (Path(gobot._core.__file__),native_root/'libgobot.so') if p.exists()}
        policy_metadata=policy.metadata()
        input_contract={key:policy_metadata[key] for key in ('observation_spec','action_spec','joint_names','control','model')}
        identity=make_run_identity(policy_contract=input_contract,physics={'artifact':env.batch.artifact.digest,'robot_signature':policy.extras['robot_signature'],'backend':args.backend,'deterministic':args.deterministic,'num_envs':env.num_envs},
            data={**policy.extras['data'],'task':{k:v for k,v in asdict(cfg).items() if k not in ('project_path','motion_files')}},algorithm=asdict(algorithm) if isinstance(algorithm,SuperMimicConfig) else algorithm,
            software={'versions':versions,'sources':canonical_digest(sources),'native_libraries':native},seed=args.seed,
            parent=None if args.checkpoint is None else {'checkpoint':str(args.checkpoint.resolve()),'sha256':file_digest(args.checkpoint)})
        write_json(args.output/'identity.json',identity)
        write_json(args.output/'task_config.json',{**asdict(cfg),'project_path':str(cfg.project_path)})
        write_json(args.output/'source_hashes.json',sources)
        write_json(args.output/'source_archive.json',{'file':archive_path.name,'sha256':file_digest(archive_path),'content_digest':canonical_digest(sources)})
        write_json(args.output/'policy.manifest.json',policy.metadata())
        protocol_file='g1_tracking_deterministic_v1.json' if args.deterministic else 'g1_tracking_v1.json'
        freeze_protocol(args.output/'evaluation_protocol.json',json.loads((ROOT/'examples/locomotion/protocols'/protocol_file).read_text()))
        if status:status.update(status='evaluating' if args.operation=='evaluate' else 'training',identity=identity['digest'])
        if args.operation=='evaluate':
            def progress(iteration,steps,failed):
                check()
                if status:status.update(status='evaluating',iteration=iteration,failed=int(failed.sum()))
            original=env.step
            def bounded(actions):check();return original(actions)
            env.step=bounded
            result=evaluate(env,payload,max_steps=args.max_steps,start_offset=args.start_offset,progress=progress)
            result.update(identity=identity['digest'],checkpoint_sha256=file_digest(args.checkpoint),elapsed_s=time.monotonic()-start)
            write_json(args.output/'evaluation.json',result)
        else:
            class SmokeLearner(SuperMimicLearner):
                def collect(self,n_steps):check();super().collect(n_steps)
                def _checkpoint(self,include_optimizers):
                    return {**super()._checkpoint(include_optimizers),'gobot_identity':identity,
                            POLICY_MANIFEST_KEY:policy.metadata(),'training_profile':'fixed_physics_smoke'}
                def save(self,path=None,include_optimizers=True):
                    destination=super().save(path,include_optimizers)
                    write_policy_manifest_sidecar(destination,policy)
                    return destination
            learner=SmokeLearner(env,algorithm,args.output)
            result=learner.train()
            result['training_profile']='fixed_physics_smoke'
            write_json(args.output/'training_summary.json',result)
        if status:status.update(status='finished')
        print(json.dumps({k:v for k,v in result.items() if k!='per_clip'},indent=2),flush=True)
    except BaseException as error:
        if status:status.update(status='failed',error=str(error))
        write_json(args.output/'failure.json',{'error':str(error),'elapsed_s':time.monotonic()-start})
        raise
    finally:
        if env:env.close()
        if status:status.close()


if __name__=='__main__':main()
