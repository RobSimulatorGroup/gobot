"""Measure fixed-global-batch 1/N-GPU throughput and verify shared replicas.

Runs short training jobs sequentially; does not launch a long training run.
The timing excludes data/environment initialization; cold loop/compilation and
steady intervals are reported separately. All artifacts are kept for inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def _digest(value):
    import torch
    hasher = hashlib.sha256()
    def add(item):
        if torch.is_tensor(item):
            t = item.detach().cpu().contiguous()
            hasher.update(str((t.dtype, tuple(t.shape))).encode())
            hasher.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                hasher.update(str(key).encode()); add(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item: add(child)
        else:
            hasher.update(repr(item).encode())
    add(value)
    return hasher.hexdigest()


def worker(argv):
    """Use the real trainer and check both parameters and optimizer states."""
    import torch.distributed as dist
    from gobot.rl.supermimic.official.training import Trainer, write_json
    from gobot.rl.train import main as train
    save = Trainer.save
    def checked_save(self, name='latest.pt'):
        if name == 'latest.pt':
            state = {key: getattr(self,key).state_dict() for key in ('actor','critic','dynamics')}
            for key, optimizers in (('wm_optimizers',self.wm.optimizers),
                                    ('actor_optimizers',self.policy.actor_optimizers),
                                    ('critic_optimizers',self.policy.critic_optimizers)):
                state[key] = [opt.state_dict() for opt in optimizers]
            hashes = {key:_digest(value) for key,value in state.items()}
            replicas = [hashes]
            if self.distributed.world_size > 1:
                replicas = [None] * self.distributed.world_size
                dist.all_gather_object(replicas, hashes)
            equal = all(value == replicas[0] for value in replicas)
            self.distributed.primary_call(lambda:write_json(self.output/'replica_check.json',
                {'equal':equal,'world_size':self.distributed.world_size,'hashes':replicas}))
            if not equal:raise RuntimeError('shared model or optimizer state diverged across ranks')
        return save(self,name)
    Trainer.save = checked_save
    return train(argv)


def summarize(run_dir, warmup_steps):
    status = json.loads((run_dir/'status.json').read_text())
    meta = json.loads((run_dir/'run.json').read_text())
    rows = [json.loads(line) for line in (run_dir/'metrics.jsonl').read_text().splitlines()]
    # A row describes the interval since the preceding log. Exclude any interval
    # whose beginning precedes the requested warmup boundary.
    steady = [row for previous,row in zip(rows,rows[1:]) if previous['control_steps'] >= warmup_steps]
    if not steady:raise ValueError('benchmark ended before any steady timing intervals')
    seconds = sum(row['Perf/interval_seconds'] for row in steady)
    transitions = sum(row['Perf/global_transitions_per_second']*row['Perf/interval_seconds'] for row in steady)
    return {'world_size':meta['world_size'],'global_num_envs':meta['global_num_envs'],
            'global_wm_batch_size':meta['global_wm_batch_size'],
            'global_policy_batch_size':meta['global_policy_batch_size'],
            'control_steps':status['control_steps'],'learner_updates':status['learner_updates'],
            'loop_seconds_including_compile':status['elapsed_seconds'],
            'steady_seconds':seconds,'steady_transitions_per_second':transitions/seconds,
            'steady_median_transitions_per_second':statistics.median(row['Perf/global_transitions_per_second'] for row in steady),
            'peak_cuda_gib':max(row.get('Perf/peak_cuda_gib',0.) for row in rows),
            'replicas_equal':json.loads((run_dir/'replica_check.json').read_text())['equal'],
            'run':str(run_dir)}


def main():
    if len(sys.argv)>1 and sys.argv[1]=='worker':
        return worker(sys.argv[2:])
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('examples/g1/train/official_bones_1024.json'))
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3])
    parser.add_argument('--workers',nargs='+',type=int,default=[1,2,4])
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--warmup-steps',type=int,default=50)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--set',action='append',default=[])
    args=parser.parse_args()
    if any(n<1 or n>len(args.gpus) for n in args.workers):parser.error('worker counts must fit the provided GPU list')
    if len(set(args.workers))!=len(args.workers) or len(set(args.gpus))!=len(args.gpus):parser.error('duplicate worker counts/GPUs')
    if args.steps<=args.warmup_steps+10:parser.error('allow at least two logged intervals after warmup')
    args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for world in args.workers:
        destination=args.output/f'gpu{world}'
        command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',
            '-m','examples.g1.official.benchmark','worker','g1-supermimic',
            '--config',str(args.config),'--project',str(args.project),'--library',str(args.library),
            '--backend','mujoco-warp','--device','cuda:0','--batch-scope','global',
            '--steps',str(args.steps),'--seed',str(args.seed),'--output',str(destination)]
        for override in args.set:command+=['--set',override]
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,args.gpus[:world])),
                         OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
        (args.output/f'command{world}.json').write_text(json.dumps(command,indent=2)+'\n')
        print(f'Starting {world} GPU benchmark: {destination}',flush=True)
        started=time.monotonic()
        with (args.output/f'gpu{world}.log').open('w') as log:
            subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=summarize(destination,args.warmup_steps)
        result['process_wall_seconds']=time.monotonic()-started
        if results:
            result['speedup_vs_first']=result['steady_transitions_per_second']/results[0]['steady_transitions_per_second']
        else:result['speedup_vs_first']=1.
        results.append(result)
        (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
