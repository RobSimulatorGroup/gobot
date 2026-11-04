"""Complete-clip evaluation with the pinned reference's queue and return metric."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path, PosixPath
import torch
from gobot.rl.supermimic.policy import Policy, PolicyCfg
from .env import MotionTrackingEnv
from .training import FORMAT
from .convert import digest
from .identity import source_identity


def load_policy(checkpoint, device='cpu', *, release=False):
    if release:
        expected='0527bab674556f48eaa7d3a27d650d1d14986019fc982d8acecbf6da75bc5ea6'
        if digest(checkpoint)!=expected:
            raise ValueError('release evaluation requires the pinned e8 checkpoint SHA256')
        with torch.serialization.safe_globals([PosixPath,(PosixPath,'pathlib.PosixPath')]):
            ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    else:
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if ck.get('format')!=FORMAT:
            raise ValueError('expected Gobot SuperMimic-Reproduced checkpoint; use --release for the pinned public checkpoint')
    policy=Policy(PolicyCfg(**ck['config']['policy'])).to(device).eval()
    policy.load_state_dict(ck['policy'],strict=True)
    return policy,ck['config']


def summarize(rows):
    n,steps=len(rows),sum(r['steps'] for r in rows)
    return {'clips':n,'failed':sum(r['failed'] for r in rows),
            'mean_norm_return':sum(r['norm_return'] for r in rows)/max(n,1),
            'step_weighted_norm_return':sum(r['norm_return']*r['steps'] for r in rows)/max(steps,1)}


@torch.no_grad()
def benchmark(cfg,policy):
    cfg=deepcopy(cfg)
    cfg.env.obs_noise=cfg.env.reset_noise=cfg.env.domain_randomization=False
    cfg.env.action_noise_std=0.
    cfg.env.sampling_mode='start'
    cfg.env.compact_library=False
    cfg.data.library=None
    cfg.env.episode_length_s=1.e5
    source=source_identity()
    env=MotionTrackingEnv(cfg)
    try:
        lib=env.library
        if env.num_envs>lib.num_clips:
            raise ValueError('evaluation num_envs must not exceed the number of selected clips')
        current=env.ids.clone(); nxt=current+env.num_envs
        obs,_=env.reset(frames=lib.clip_start[current])
        env.forced_frame[:]=lib.clip_start[nxt.clamp_max(lib.num_clips-1)]
        active=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
        returns=torch.zeros(env.num_envs,device=env.device)
        steps=torch.zeros_like(current); sums={}; results={}
        control_steps=0
        while len(results)<lib.num_clips:
            obs,reward,term,timeout,info=env.step(policy(obs['policy'],obs['goal']),record=False)
            returns+=reward;steps+=1
            control_steps+=1
            if control_steps%1000==0:
                print(json.dumps({'evaluation_control_steps':control_steps,'completed_clips':len(results)}),flush=True)
            for key,value in info['reward_terms'].items():
                sums[key]=sums.get(key,torch.zeros_like(value))+value
            ids=((term|timeout)&active).nonzero().flatten()
            for e in ids.tolist():
                ci=int(current[e]);st=int(steps[e])
                results[ci]={'name':lib.clips[ci].name,'frames':lib.clips[ci].length,'steps':st,
                    'norm_return':float(returns[e])/(.1*st),'failed':bool(term[e]),'fail_step':st if term[e] else -1,
                    'terms':{k:float(v[e])/st for k,v in sums.items() if k!='total'}}
                print(json.dumps({'evaluated':len(results),'total':lib.num_clips,**results[ci]}),flush=True)
            current[ids]=nxt[ids];nxt[ids]+=env.num_envs
            active[ids[current[ids]>=lib.num_clips]]=False
            env.forced_frame[ids]=lib.clip_start[nxt[ids].clamp_max(lib.num_clips-1)]
            returns[ids]=0;steps[ids]=0
            for value in sums.values():value[ids]=0
        rows=[results[i] for i in range(lib.num_clips)]
        return {'protocol':'supermimic_reproduced_complete_clips_v1','summary':summarize(rows),'clips':rows,
                'physics_digest':env.batch.artifact.digest,'source_sha256':source,'data':[(p.name,digest(p)) for p in env.files]}
    finally:
        env.close()
