"""Compare the reference and Gobot on identical actions in one 3.17 process.

Diagnostic only: the reference imports live here, never in gobot.rl runtime.
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import site
import sys

from .supermimic_reference import DEFAULT_REFERENCE,DEFAULT_LEDGER,budget_class,RunStatus,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=32)
    parser.add_argument('--hours',type=float,default=.15)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];reference=DEFAULT_REFERENCE.resolve()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    checkpoint=args.checkpoint.resolve()
    budget=budget_class()(DEFAULT_LEDGER,total_hours=24,run_hours=args.hours,metadata={'stage':'reference_gobot_identical_actions'})
    status=RunStatus(output,budget);a=b=None
    try:
        site.addsitedir(str(root/'.venv/lib/python3.13/site-packages'))
        import torch,numpy as np,mujoco,mujoco_warp
        if mujoco.__version__!='3.17.0' or mujoco_warp.__version__!='3.17.0':raise ValueError('requires isolated 3.17 reference environment')
        torch.set_num_threads(4);sys.path.insert(0,str(reference/'src'));os.chdir(reference)
        from unilab.assets import hub
        from unilab.base import backend_factory
        def local_assets(paths):
            for path in paths:
                if path and not Path(path).is_file():raise FileNotFoundError(path)
        hub.ensure_robot_assets_for_paths=local_assets;backend_factory.ensure_robot_assets_for_paths=local_assets
        from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import make_eval_env,load_policy
        from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
        from gobot.rl.motion_tracking import MotionTrackingCfg,make_motion_tracking_env
        a=make_eval_env(checkpoint.parent)
        names=a.supermimic_mdp.motion.names
        project=root/'examples/g1'
        files=tuple(str(project/f'assets/motions/tracking/{name}.csv') for name in names)
        b=make_motion_tracking_env(MotionTrackingCfg(project_path=project,motion_files=files),num_envs=len(names),backend='mujoco-warp')
        payload=torch.load(checkpoint,map_location='cuda',weights_only=True)
        policy=load_policy(a,payload)
        _,ai=a.reset_to_frames(a.supermimic_mdp.motion.clip_start.clone())
        _,bi=b.reset_to_frames(b.supermimic_mdp.motion.clip_start.clone())
        history=None;records=[];trace=[]
        def errors(x,y):
            diff=(x-y).abs()
            values,index=diff.flatten(1).max(1)
            return {'max_abs':float(values.max()),'worst_clip':names[int(values.argmax())],
                    'per_clip':values.cpu().tolist()}
        with torch.inference_mode():
            for step in range(args.steps):
                if budget.remaining<5:raise TimeoutError('trace budget exhausted')
                ac,bc=ai['supermimic_state'],bi['supermimic_state']
                obs=a.supermimic_mdp.obs(unflatten_state(ac['s']),ac['a_prev'],ac['xi'],ac['f'])
                other=b.supermimic_mdp.obs(unflatten_state(bc['s']),bc['a_prev'],bc['xi'],bc['f'])
                goal=a.supermimic_mdp.goal(ac['f'],int(payload['args']['goal_frames']))
                other_goal=b.supermimic_mdp.goal(bc['f'],int(payload['args']['goal_frames']))
                history=obs[:,None].repeat(1,int(payload['args']['hist']),1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
                action=policy(history,goal)
                ar=a.step(action);br=b.step(action)
                at,bt=ar.info['supermimic'],br.info['supermimic']
                row={'step':step,'state_before':errors(ac['s'],bc['s']), 'observation':errors(obs,other),
                     'goal':errors(goal,other_goal),'state_after':errors(at['s1'],bt['s1']),
                     'reward':errors(at['r'][:,None],bt['r'][:,None])}
                records.append(row)
                trace.append({'a':action.cpu(),'reference':at['s1'].cpu(),'gobot':bt['s1'].cpu()})
                ai,bi=ar.info,br.info
                if step%8==0:status.update(status='checking',iteration=step+1)
        write_json(output/'trace.json',{'names':names,'records':records})
        torch.save(trace,output/'states.pt')
        status.update(status='finished')
        print(json.dumps(records[-1],indent=2))
    except BaseException as error:
        status.update(status='failed',error=str(error));raise
    finally:
        if b:b.close()
        if a:a.close()
        status.close()


if __name__=='__main__':main()
