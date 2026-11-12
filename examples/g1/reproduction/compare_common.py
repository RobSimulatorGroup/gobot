"""Run two frozen checkpoints under exactly the same evaluation protocol.

Each policy is evaluated in a separate sequential wave on the requested GPUs.
Merge and comparison reject different common identities, missing clips and caps.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from gobot.rl.supermimic.common_evaluation import merge, summarize, hash_value
from gobot.rl.supermimic.official.motion import digest
from gobot.rl.supermimic.official.logging import write_json


def compare(official,reproduced,keep):
    if not official['complete'] or not reproduced['complete']:
        raise ValueError('both common evaluations must be complete')
    if official['common']!=reproduced['common'] or official['common_sha256']!=reproduced['common_sha256']:
        raise ValueError('common evaluation identities differ')
    if official['common_sha256']!=hash_value(official['common']):raise ValueError('common fingerprint differs')
    if official['policy_adapter']['kind']!='official' or reproduced['policy_adapter']['kind']!='reproduced':
        raise ValueError('wrong policy roles')
    a={r['name']:r for r in official['clips']};b={r['name']:r for r in reproduced['clips']}
    if len(a)!=len(official['clips']) or len(b)!=len(reproduced['clips']) or a.keys()!=b.keys():
        raise ValueError('clip members differ')
    if not keep or not keep<=set(a):raise ValueError('invalid fixed filtered subset')
    if any(a[n]['frames']!=b[n]['frames'] or a[n]['expected_steps']!=b[n]['expected_steps'] for n in a):
        raise ValueError('clip lengths or clocks differ')
    for result in (official,reproduced):
        if any(r['capped'] for r in result['clips']):raise ValueError('capped comparison')
        if summarize(result['clips'])!=result['summary']:raise ValueError('summary mismatch')
    af={n for n,r in a.items() if r['failed']};bf={n for n,r in b.items() if r['failed']}
    scopes={}
    for label,names in (('full',list(a)),('filtered',[n for n in a if n in keep])):
        scopes[label]={k:summarize([rows[n] for n in names]) for k,rows in (('official',a),('reproduced',b))}
    keywords={}
    for keyword in ('flip','handstand','cartwheel','stand_up','jump','dance','walk','run','safety_roll'):
        names=[n for n in a if keyword in n.lower()]
        if names:keywords[keyword]={'clips':len(names),'official_failures':sum(a[n]['failed'] for n in names),
                                    'reproduced_failures':sum(b[n]['failed'] for n in names)}
    return {'complete':True,'common_sha256':official['common_sha256'],'scopes':scopes,
        'checkpoints':{'official':official['checkpoint_sha256'],'reproduced':reproduced['checkpoint_sha256']},
        'same_clip_names_frames_clock':True,'both_failed':len(af&bf),
        'official_only_failed':sorted(af-bf),'reproduced_only_failed':sorted(bf-af),
        'keyword_diagnostics':keywords,'policy_adapters':{'official':official['policy_adapter'],'reproduced':reproduced['policy_adapter']}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('official','reproduced','project','reproduced-project','library','filter-manifest','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs='+',default=list(range(8)))
    p.add_argument('--num-envs',type=int,default=512)
    p.add_argument('--clips',type=Path,help='optional JSON name list for a validation subset')
    args=p.parse_args()
    if not args.gpus or len(args.gpus)!=len(set(args.gpus)) or min(args.gpus)<0:p.error('invalid GPU list')
    if args.num_envs<1:p.error('num-envs must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    names=json.loads((args.library/'motion_names.json').read_text())
    if args.clips:
        selected=json.loads(args.clips.read_text())
        if not selected or len(selected)!=len(set(selected)) or not set(selected)<=set(names):raise ValueError('invalid clip selection')
        names=selected
    with args.filter_manifest.open() as stream:keep={r['name'] for r in csv.DictReader(stream) if r['keep']=='1'}
    if args.clips:keep &= set(names)
    input_files={key:{'path':str(getattr(args,key).resolve()),'sha256':digest(getattr(args,key))}
                 for key in ('official','reproduced','filter_manifest')}
    write_json(args.output/'plan.json',{'started_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'inputs':input_files,'library':str(args.library.resolve()),'manifest_sha256':digest(args.library/'manifest.json'),
        'gpus':args.gpus,'num_envs':args.num_envs,'selection_count':len(names),'selection_sha256':hash_value(names)})
    results={}
    for kind in ('reproduced','official'):
        folder=args.output/kind;folder.mkdir()
        workers=[];children=[];count=len(args.gpus)
        try:
            for shard,gpu in enumerate(args.gpus):
                command=[sys.executable,'-u','-m','gobot.rl.supermimic.common_evaluation',
                    '--checkpoint',str(getattr(args,kind).resolve()),'--project',str(args.project.resolve()),
                    '--reproduced-project',str(args.reproduced_project.resolve()),'--library',str(args.library.resolve()),
                    '--output',str((folder/f'shard{shard}.json').resolve()),'--device','cuda:0',
                    '--num-envs',str(args.num_envs),'--shard-count',str(count),'--shard-index',str(shard)]
                if args.clips:command+=['--clips',*names]
                environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
                for key in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):environment.pop(key,None)
                with (folder/f'worker{shard}.log').open('w') as log:
                    child=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children.append(child);workers.append({'pid':child.pid,'gpu':gpu,'shard':shard,'command':command})
            write_json(folder/'workers.json',workers)
            last_log=0.
            while any(child.poll() is None for child in children):
                failed=[(i,c.returncode) for i,c in enumerate(children) if c.poll() not in (None,0)]
                if failed:raise RuntimeError(f'{kind} workers failed: {failed}')
                now=time.monotonic()
                if now-last_log>=30:
                    progress=[json.loads(path.read_text()) for path in folder.glob('shard*.progress.json')]
                    row={'state':'evaluating','policy':kind,'completed_clips':sum(r.get('completed_clips',0) for r in progress),
                         'failed_clips':sum(r.get('failed_clips',r.get('failures',0)) for r in progress),
                         'total_clips':len(names),'elapsed_seconds':now-started}
                    print(json.dumps(row),flush=True);write_json(args.output/'status.json',row);last_log=now
                time.sleep(2)
            if any(c.returncode!=0 for c in children):raise RuntimeError('worker failed on completion')
        except BaseException:
            for child in children:
                if child.poll() is None:child.terminate()
            for child in children:
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
        results[kind]=merge([folder/f'shard{i}.json' for i in range(count)],names,folder/'full.json')
        filtered=[r for r in results[kind]['clips'] if r['name'] in keep]
        write_json(folder/'filtered.json',{'clips':filtered,'summary':summarize(filtered),
            'checkpoint_sha256':results[kind]['checkpoint_sha256'],'common_sha256':results[kind]['common_sha256']})
        write_json(folder/'failures.json',[r for r in results[kind]['clips'] if r['failed']])
        print(json.dumps({'state':'policy_completed','policy':kind,'summary':results[kind]['summary']}),flush=True)
    result=compare(results['official'],results['reproduced'],keep)
    result['elapsed_seconds']=time.monotonic()-started
    write_json(args.output/'comparison.json',result)
    write_json(args.output/'status.json',{'state':'completed','elapsed_seconds':result['elapsed_seconds'],'scopes':result['scopes']})
    print(json.dumps({'state':'completed','scopes':result['scopes'],'elapsed_seconds':result['elapsed_seconds']},indent=2),flush=True)


if __name__=='__main__':main()
