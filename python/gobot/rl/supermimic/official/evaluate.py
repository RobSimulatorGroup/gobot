"""Evaluate complete official-motion clips using a frozen Gobot checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
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
def evaluate(checkpoint, output, *, project=None, library=None, device='cuda:0', clips=None, num_envs=64, max_steps=None):
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
        selected=env.motion.names if clips is None else clips
        if not selected or len(selected)!=len(set(selected)) or not set(selected)<=set(env.motion.names):
            raise ValueError('evaluation clips must be unique known motion names')
        ends=env.motion.motion_ends.cpu().tolist();starts=[0,*ends[:-1]]
        rows=[];queue=0;lane_clip=[None]*num_envs
        active=torch.zeros(num_envs,dtype=torch.bool,device=device)
        returns=torch.zeros(num_envs,device=device);steps=torch.zeros(num_envs,dtype=torch.long,device=device)
        state=env.reset()
        history=None
        while queue<len(selected) or active.any():
            free=(~active).nonzero().flatten().tolist()
            assigned=[]
            for lane in free:
                if queue>=len(selected):break
                name=selected[queue];queue+=1
                index=env.motion.names.index(name);lane_clip[lane]=(name,index)
                ids=torch.tensor([lane],device=device)
                env._reset(ids,torch.tensor([starts[index]],device=device))
                # Match explicit env.reset's command advance; do not move other lanes.
                env.frame[lane]+=1
                active[lane]=True;returns[lane]=0;steps[lane]=0;assigned.append(lane)
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
            for lane in done.nonzero().flatten().tolist():
                name,index=lane_clip[lane];count=int(steps[lane]);expected=ends[index]-starts[index]-1
                failed=bool(term[lane]);finished=bool(env.motion_end[lane])
                rows.append({'name':name,'frames':ends[index]-starts[index],'executed_steps':count,
                    'expected_steps':expected,'return':float(returns[lane]/(.1*count)),
                    'coverage':min(count/expected,1.),'failed':failed,'finished':finished,
                    'completed':finished and not failed,'capped':bool(capped[lane]) and not(failed or finished)})
                active[lane]=False
            sensor=env.sensor(state) if cfg.actor_obs=='proprio' else representation.represent(state)
            history=torch.cat((history[:,1:],sensor[:,None]),1)
        complete=not any(r['capped'] for r in rows)
        report={'protocol':'gobot_supermimic_official_evaluation_v1','checkpoint_sha256':digest(checkpoint),
            'data_manifest_sha256':env.motion.identity,'complete':complete,'clips':rows,
            'summary':{'clips':len(rows),'mean_return':sum(r['return'] for r in rows)/len(rows),
                       'failures':sum(r['failed'] for r in rows),'completed':sum(r['completed'] for r in rows)}}
        Path(output).parent.mkdir(parents=True,exist_ok=True);write_json(output,report)
        return report
    finally:env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--project',type=Path);p.add_argument('--library',type=Path)
    p.add_argument('--device',default='cuda:0');p.add_argument('--num-envs',type=int,default=64)
    p.add_argument('--clips',nargs='+');p.add_argument('--max-steps',type=int)
    a=p.parse_args()
    if a.num_envs<1 or (a.max_steps is not None and a.max_steps<1):p.error('counts must be positive')
    result=evaluate(**vars(a));print(json.dumps(result['summary'],indent=2))
    if not result['complete']:raise SystemExit('evaluation stopped at --max-steps; report is incomplete')

if __name__=='__main__':main()
