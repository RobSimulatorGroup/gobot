"""Bounded frozen-reference evaluation, isolated from Gobot's native libraries."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from .supermimic_reference import DEFAULT_REFERENCE, DEFAULT_LEDGER, RunStatus, budget_class, write_json, evaluation_summary, source_hash, ControlledStop


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=DEFAULT_REFERENCE)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ledger',type=Path,default=DEFAULT_LEDGER)
    parser.add_argument('--hours',type=float,default=.6)
    parser.add_argument('--expect-version',default='3.17.0')
    parser.add_argument('--deterministic',action='store_true')
    parser.add_argument('--max-steps',type=int)
    args=parser.parse_args()
    if args.deterministic:os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    if not math.isfinite(args.hours) or args.hours <= 0:
        parser.error('reference validation requires a finite positive job limit')
    reference=args.reference.resolve();checkpoint=args.checkpoint.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    budget=budget_class()(args.ledger,total_hours=24,run_hours=args.hours,metadata={'stage':'reference_version_parity','output':str(output)})
    status=RunStatus(output,budget);env=None
    def cancel(*_):status.cancelled=True
    signal.signal(signal.SIGINT,cancel);signal.signal(signal.SIGTERM,cancel)
    try:
        import torch,mujoco,mujoco_warp,warp
        if mujoco.__version__ != args.expect_version or mujoco_warp.__version__ != args.expect_version:
            raise ValueError('isolated reference process does not have the expected physics versions')
        torch.set_num_threads(4)
        if args.deterministic:
            torch.use_deterministic_algorithms(True)
            original_put_model=mujoco_warp.put_model
            def deterministic_model(*values,**options):
                model=original_put_model(*values,**options)
                model.opt.deterministic=True
                return model
            mujoco_warp.put_model=deterministic_model
        sys.path.insert(0,str(reference/'src'));os.chdir(reference)
        # Reuse only checksum-verified local assets; prevent UniLab's unrelated
        # robot-name fallback download from replacing this frozen model.
        assets=json.loads((reference/'reference-assets.json').read_text())
        for item in assets['files']:
            path=reference/item['local']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError(f'pinned reference asset changed: {path}')
        from unilab.assets import hub
        from unilab.base import backend_factory
        def local_assets(paths):
            for path in paths:
                if path and not Path(path).is_file():
                    raise FileNotFoundError(path)
        hub.ensure_robot_assets_for_paths=local_assets
        backend_factory.ensure_robot_assets_for_paths=local_assets
        from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import make_eval_env,evaluate
        compatibility_path=Path(sys.prefix)/'compatibility.json'
        compatibility=json.loads(compatibility_path.read_text()) if compatibility_path.exists() else None
        versions={name:importlib.metadata.version(name) for name in ('torch','mujoco','mujoco-warp','warp-lang','unilab','unisim-core')}
        write_json(output/'identity.json',{'kind':'frozen_reference_eval','versions':versions,'source_hashes':source_hash(reference),
            'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'device':torch.cuda.get_device_name(0),'source_revision':'faf7818e8abac3ec6abb8f00e6e904d3577f800b','compatibility':compatibility,'deterministic':args.deterministic})
        status.update(status='initializing',versions=versions)
        env=make_eval_env(checkpoint.parent)
        status.update(status='evaluating',iteration=0,initialization_seconds=budget.elapsed)
        original_step=env.step
        steps=0
        class ScreeningComplete(Exception):pass
        cumulative=torch.zeros(env.num_envs,device=env.device)
        counts=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
        finished=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        failed=finished.clone()
        def bounded_step(actions):
            nonlocal steps
            if status.cancelled or budget.remaining<5:raise ControlledStop('evaluation interrupted or budget exhausted')
            started=time.monotonic()
            result=original_step(actions);steps+=1
            t=result.info['supermimic'];alive=~finished
            cumulative.add_(torch.where(alive,t['r'],0.));counts.add_(alive)
            failed.logical_or_(alive&t['term']);finished.logical_or_(alive&t['done'])
            if steps<=5 or steps%100==0:
                torch.cuda.synchronize()
                status.update(status='evaluating',iteration=steps,last_step_seconds=time.monotonic()-started)
            if args.max_steps is not None and steps>=args.max_steps:raise ScreeningComplete()
            return result
        env.step=bounded_step
        payload=torch.load(checkpoint,map_location=env.device,weights_only=True)
        try:
            result=evaluation_summary(evaluate(env,payload),env.supermimic_mdp.motion)
        except ScreeningComplete:
            motion=env.supermimic_mdp.motion
            normalized=cumulative/((motion.clip_len-1)*env.supermimic_mdp.max_step_reward)
            result=evaluation_summary({'norm_return':float(normalized.mean()),'n_failed':int(failed.sum()),
                'per_clip':{name:{'norm_return':float(normalized[i]),'steps':int(counts[i]),'failed':bool(failed[i])}
                            for i,name in enumerate(motion.names)}},motion)
            result['screening']=True
        result['deterministic']=args.deterministic
        write_json(output/'evaluation.json',result)
        status.update(status='finished',result={k:v for k,v in result.items() if k!='per_clip'})
        print(json.dumps({k:v for k,v in result.items() if k!='per_clip'},indent=2),flush=True)
    except BaseException as error:
        status.update(status='failed',error=str(error));raise
    finally:
        if env is not None:env.close()
        status.close()


if __name__=='__main__':main()
