"""Evaluate complete official-motion clips using a frozen Gobot checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import time
from pathlib import Path

import torch
from tensordict import TensorDict

from gobot.rl.training.workflow import update_config
from .config import Config
from .env import TrackingEnv
from .models import MultiEncoderAE
from .motion import digest
from .state import GenCoordRepresentation
from .tasks import EncoderSpec
from .training import FORMAT, write_json


@torch.no_grad()
def evaluate(checkpoint, output, *, project=None, library=None, device='cuda:0', clips=None, num_envs=64, max_steps=None, shard_index=0, shard_count=1):
    if isinstance(shard_count,bool) or not isinstance(shard_count,int) or shard_count<1 or not 0<=shard_index<shard_count:
        raise ValueError('invalid evaluation shard')
    if num_envs<1 or (max_steps is not None and max_steps<1):
        raise ValueError('evaluation counts must be positive')
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(f'evaluation output already exists: {output}')
    started=time.monotonic()
    progress_path=output.with_suffix('.progress.json')
    partial_path=output.with_suffix('.partial.jsonl')
    write_json(progress_path,{'state':'loading','shard_index':shard_index,'shard_count':shard_count})
    checkpoint_hash=digest(checkpoint)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ck.get('format')!=FORMAT:raise ValueError('expected an official Gobot training checkpoint')
    cfg=update_config(Config(),ck['config'])
    cfg=replace(cfg,project=Path(project or cfg.project),library=Path(library or cfg.library),
        backend='mujoco-warp' if device.startswith('cuda') else 'mujoco-cpu',device=device,
        num_envs=num_envs,domain_randomization=False,reset_noise=False,sampling_mode='start',
        episode_length_s=1.e9,compile=False)
    env=TrackingEnv(cfg)
    try:
        if env.motion.identity!=ck['identity']['motion'] or env.profile_digest!=ck['identity']['profile'] or env.batch.artifact.content_digest!=ck['identity']['scene']:
            raise ValueError('evaluation scene or motion identity differs from checkpoint')
        representation=GenCoordRepresentation(29,env.default_q,dt=env.step_dt,decimation=env.batch.decimation,
                                              actor_obs_mode=cfg.actor_obs).to(device)
        actor=MultiEncoderAE(68 if cfg.actor_obs=='rep_state' else 93,29,
            [EncoderSpec('motion',640),EncoderSpec('velocity',3)],{'motion':1.,'velocity':0.},
            cfg.latent_dim,cfg.actor_hidden_dims,cfg.latent_bc_weight,cfg.history_length).to(device)
        actor.load_state_dict(ck['actor']);actor.eval()
        selection=env.motion.names if clips is None else clips
        selected=selection
        if not selected or len(selected)!=len(set(selected)) or not set(selected)<=set(env.motion.names):
            raise ValueError('evaluation clips must be unique known motion names')
        selected=selected[shard_index::shard_count]
        if not selected:raise ValueError('evaluation shard is empty')
        name_to_index={name:i for i,name in enumerate(env.motion.names)}
        import hashlib
        selection_hash=hashlib.sha256(json.dumps(selection,separators=(',',':')).encode()).hexdigest()
        from . import env as env_module, models, state as state_module
        source_hashes={Path(m.__file__).name:digest(m.__file__) for m in (env_module,models,state_module)}
        source_hashes['evaluate.py']=digest(__file__)
        options={'num_envs':num_envs,'max_steps':max_steps,'nconmax':cfg.nconmax,'njmax':cfg.njmax,
                 'backend':cfg.backend,'capture_graphs':cfg.capture_graphs,'actor_deterministic':True,
                 'reset_noise':False,'domain_randomization':False,'seed':0,
                 'reset_reference_advance':1,'episode_timeout':False}
        torch.manual_seed(0)
        next_progress=0.
        partial_path.write_text('')
        ends=env.motion.motion_ends.cpu().tolist();starts=[0,*ends[:-1]]
        rows=[];queue=0;lane_clip=[None]*num_envs
        active=torch.zeros(num_envs,dtype=torch.bool,device=device)
        returns=torch.zeros(num_envs,device=device);steps=torch.zeros(num_envs,dtype=torch.long,device=device)
        state=env.reset()
        history=None
        while queue<len(selected) or active.any():
            free=(~active).nonzero().flatten().tolist()
            assigned=free[:min(len(free),len(selected)-queue)]
            if assigned:
                new_frames=[]
                for lane in assigned:
                    name=selected[queue];queue+=1
                    index=name_to_index[name];lane_clip[lane]=(name,index)
                    new_frames.append(starts[index])
                ids=torch.tensor(assigned,dtype=torch.long,device=device)
                env._reset(ids,torch.tensor(new_frames,dtype=torch.long,device=device))
                # Match explicit reset's one-frame reference advance, scoped to these lanes.
                env.frame[ids]+=1
                active[ids]=True;returns[ids]=0;steps[ids]=0
            if assigned:
                env.reference=env.motion.body_state(env.frame)
                from gobot.rl.supermimic.rewards import reanchor_reference
                state=env.full_state()
                env.relative_pose=reanchor_reference(env.reference[0],env.reference[1],env.reference[0][:,0],
                    env.reference[1][:,0],state[:,:3],state[:,3:7])
            rep=representation.represent(state)
            sensor=env.sensor(state) if cfg.actor_obs=='proprio' else rep
            if history is None:history=sensor[:,None].repeat(1,cfg.history_length,1)
            if assigned:history[assigned]=sensor[assigned,None]
            _,goal=representation.compute_wm_goal(env.motion,env.frame,state[:,:7])
            action=actor(TensorDict({'history':history,'goal_motion':goal},batch_size=[num_envs],device=device))
            state,reward,term,trunc,_=env.step(action)
            returns[active]+=reward[active];steps[active]+=1
            capped=(steps>=max_steps)&active if max_steps is not None else torch.zeros_like(active)
            done=active&(term|trunc|capped)
            new_rows=[]
            for lane in done.nonzero().flatten().tolist():
                name,index=lane_clip[lane];count=int(steps[lane]);expected=ends[index]-starts[index]-1
                failed=bool(term[lane]);finished=bool(env.motion_end[lane])
                new_rows.append({'name':name,'frames':ends[index]-starts[index],'executed_steps':count,
                    'expected_steps':expected,'return':float(returns[lane]/(.1*count)),
                    'coverage':min(count/expected,1.),'failed':failed,'finished':finished,
                    'completed':finished and not failed,'capped':bool(capped[lane]) and not(failed or finished)})
                active[lane]=False
            if new_rows:
                rows.extend(new_rows)
                with partial_path.open('a') as stream:
                    for row in new_rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
            now=time.monotonic()
            if now>=next_progress or len(rows)==len(selected):
                progress={'state':'evaluating','shard_index':shard_index,'shard_count':shard_count,
                    'completed_clips':len(rows),'total_clips':len(selected),'queued_clips':queue,
                    'failed_clips':sum(row['failed'] for row in rows),'elapsed_seconds':now-started}
                write_json(progress_path,progress);print(json.dumps(progress),flush=True)
                next_progress=now+30.
            sensor=env.sensor(state) if cfg.actor_obs=='proprio' else representation.represent(state)
            history=torch.cat((history[:,1:],sensor[:,None]),1)
        if {r['name'] for r in rows}!=set(selected) or len(rows)!=len(selected):
            raise RuntimeError('evaluation queue lost or duplicated clips')
        rows.sort(key=lambda row:name_to_index[row['name']])
        for row in rows:validate_row(row)
        complete=not any(r['capped'] for r in rows)
        if digest(checkpoint)!=checkpoint_hash:raise ValueError('checkpoint changed during evaluation')
        report={'protocol':'gobot_supermimic_official_evaluation_v1','checkpoint_sha256':checkpoint_hash,
            'selection_sha256':selection_hash,'selection_count':len(selection),
            'shard_index':shard_index,'shard_count':shard_count,'evaluation_options':options,
            'evaluation_sources':source_hashes,'elapsed_seconds':time.monotonic()-started,
            'data_manifest_sha256':env.motion.identity,'complete':complete,'clips':rows,
            'summary':summarize(rows)}
        write_json(output,report)
        write_json(progress_path,{'state':'completed' if complete else 'incomplete',
            'completed_clips':len(rows),'total_clips':len(selected),**report['summary'],
            'elapsed_seconds':report['elapsed_seconds']})
        return report
    finally:env.close()


def validate_row(row):
    frames,steps,expected=row['frames'],row['executed_steps'],row['expected_steps']
    if any(isinstance(v,bool) or not isinstance(v,int) for v in (frames,steps,expected)) or frames<3 or expected!=frames-1 or not 1<=steps<=expected:
        raise ValueError(f'invalid evaluation frame counts: {row["name"]}')
    if not all(isinstance(row[key],bool) for key in ('failed','finished','completed','capped')):
        raise ValueError('evaluation masks must be boolean')
    if not math.isfinite(row['return']) or not math.isfinite(row['coverage']):
        raise ValueError('nonfinite evaluation result')
    if not math.isclose(row['coverage'],steps/expected,abs_tol=1e-12):
        raise ValueError('coverage differs from executed clip length')
    if row['finished']!=(steps==expected) or row['completed']!=(row['finished'] and not row['failed']):
        raise ValueError('inconsistent finished/completed flags')
    if row['capped'] and (row['failed'] or row['finished']):
        raise ValueError('a capped clip cannot also be resolved')
    if not(row['failed'] or row['finished'] or row['capped']):
        raise ValueError('unresolved clip reported as evaluated')


def summarize(rows):
    if not rows:raise ValueError('empty evaluation')
    return {'clips':len(rows),'mean_return':sum(r['return'] for r in rows)/len(rows),
            'failures':sum(r['failed'] for r in rows),'completed':sum(r['completed'] for r in rows),
            'failure_rate':sum(r['failed'] for r in rows)/len(rows),
            'mean_coverage':sum(r['coverage'] for r in rows)/len(rows),
            'frame_coverage':sum(r['executed_steps'] for r in rows)/sum(r['expected_steps'] for r in rows),
            'mean_expected_length_return':sum(r['return']*r['coverage'] for r in rows)/len(rows)}


def merge_reports(paths, names, output):
    reports=[json.loads(Path(path).read_text()) for path in paths]
    if not reports:raise ValueError('no shard reports')
    first=reports[0];count=first['shard_count']
    if len(reports)!=count or {r['shard_index'] for r in reports}!=set(range(count)):
        raise ValueError('missing or duplicate evaluation shards')
    import hashlib
    selection_hash=hashlib.sha256(json.dumps(names,separators=(',',':')).encode()).hexdigest()
    if first['selection_sha256']!=selection_hash or first['selection_count']!=len(names):
        raise ValueError('evaluation selection differs from manifest')
    binding=('protocol','checkpoint_sha256','data_manifest_sha256','selection_sha256','selection_count',
             'shard_count','evaluation_options','evaluation_sources')
    rows=[]
    for report in reports:
        if not report['complete'] or any(report[key]!=first[key] for key in binding):
            raise ValueError('incomplete shard or mismatched evaluation identity')
        expected=set(names[report['shard_index']::count])
        actual=[row['name'] for row in report['clips']]
        if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('shard clip membership differs')
        for row in report['clips']:
            validate_row(row)
            if row['capped']:raise ValueError('cannot merge capped evaluations')
        if summarize(report['clips'])!=report['summary']:raise ValueError('shard summary differs from rows')
        rows.extend(report['clips'])
    order={name:i for i,name in enumerate(names)}
    if len(order)!=len(names) or len(rows)!=len(names):raise ValueError('invalid full evaluation membership')
    rows.sort(key=lambda row:order[row['name']])
    result={key:first[key] for key in binding}
    result.update(complete=True,clips=rows,summary=summarize(rows),
                  shards=[{'path':str(path),'sha256':digest(path)} for path in paths])
    write_json(output,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--project',type=Path);p.add_argument('--library',type=Path)
    p.add_argument('--device',default='cuda:0');p.add_argument('--num-envs',type=int,default=64)
    p.add_argument('--clips',nargs='+');p.add_argument('--max-steps',type=int)
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1)
    a=p.parse_args()
    if a.num_envs<1 or (a.max_steps is not None and a.max_steps<1):p.error('counts must be positive')
    result=evaluate(**vars(a));print(json.dumps(result['summary'],indent=2))
    if not result['complete']:raise SystemExit('evaluation stopped at --max-steps; report is incomplete')

if __name__=='__main__':main()
