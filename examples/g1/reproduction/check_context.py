"""Independent G1 context/action-noise trace with identical injected stimuli.

Reference mode runs untouched mjlab/SuperMimic code. Candidate mode consumes its
recorded actions, executed noise and context; it never imports the reference.
This tests adaptation semantics, not equality of the frameworks' RNG consumption.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch


def reference(args):
    import torch
    import mjlab
    from .check_release_policy import load_release
    package=types.ModuleType('mjlab.tasks')
    package.__path__=[str(Path(mjlab.__file__).parent/'tasks')]
    sys.modules['mjlab.tasks']=package
    from supermimic.envs.tracking_env_cfg import make_env,SuperMimicEnvCfgArgs
    from supermimic.policy.nets import Policy,PolicyCfg
    from supermimic.envs.state_io import StateIO
    ck,digest=load_release(args.checkpoint)
    torch.set_num_threads(2)
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(args.data)],num_envs=4,
        episode_length_s=10.,action_noise_std=.1,obs_noise=False,reset_noise=False,
        domain_randomization=True,sampling_mode='start',replay_capacity=96,seed=193),device='cuda:0')
    try:
        pi=Policy(PolicyCfg(**ck['config']['policy'])).to('cuda:0').eval()
        pi.load_state_dict(ck['policy'])
        recorder=env.recorder_manager.get_term('replay')
        action_term=env.action_manager.get_term('joint_pos')
        cmd=env.command_manager.get_term('motion')
        io=StateIO(env)
        rows=[]
        with torch.no_grad():
            obs,_=env.reset()
            initial=io.read();xi=recorder.ctx.read()
            for _ in range(args.steps):
                action=pi(obs['policy'],obs['goal'])
                row={'state':io.read(),'frame':cmd.frame.clone(),'obs':obs['policy'].clone(),
                     'goal':obs['goal'].clone(),'action':action.clone()}
                obs,reward,term,tout,_=env.step(action)
                row.update(executed=action_term.executed_raw.clone(),reward=reward.clone(),
                           terminated=term.clone(),timeout=tout.clone())
                rows.append({k:v.cpu() for k,v in row.items()})
                if (term|tout).any():
                    raise RuntimeError('context trace reached a reset; shorten the pre-reset stimulus')
        result={k:torch.stack([r[k] for r in rows]) for k in rows[0]}
        result['xi']=xi.cpu();result['initial_state']=initial.cpu()
        torch.save(result,args.output)
        info={'protocol':'g1_random_context_reference_v1','checkpoint_sha256':digest,
              'steps':len(rows),'gobot_imported':'gobot' in sys.modules,
              'scope':'per-lane COM, foot friction, bias and executed noise; no observation/reset noise',
              'data_sha256':sha256(args.data)}
        args.output.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
        return info
    finally:env.close()


def candidate(args):
    import torch
    from gobot.rl.supermimic.config import TrainConfig
    from gobot.rl.motion_tracking.env import MotionTrackingEnv
    from gobot.rl.motion_tracking.identity import source_identity
    data=torch.load(args.fixture,weights_only=True,map_location='cuda:0')
    metadata=json.loads(args.fixture.with_suffix('.json').read_text())
    if metadata['data_sha256']!=sha256(args.data):raise ValueError('motion data differs from reference')
    torch.set_num_threads(2)
    cfg=TrainConfig()
    cfg.project=args.project;cfg.data.npz_dir=args.data.parent;cfg.data.clips=[args.data.stem]
    cfg.env.num_envs=data['xi'].shape[0];cfg.env.replay_capacity=96;cfg.env.action_noise_std=.1
    cfg.env.obs_noise=cfg.env.reset_noise=cfg.env.domain_randomization=False
    cfg.env.sampling_mode='start'
    env=MotionTrackingEnv(cfg)
    try:
        env.xi.copy_(data['xi'])
        env._apply_physics_context(include_com=True)
        metrics={key:0. for key in ('state','obs','goal','frame','executed','reward')}
        flags=0
        with torch.no_grad():
            obs,_=env.reset(frames=torch.zeros(env.num_envs,dtype=torch.long,device=env.device))
            for i in range(len(data['action'])):
                actual={'state':env.learning_state(),'obs':obs['policy'],'goal':obs['goal'],'frame':env.frame}
                for key,value in actual.items():metrics[key]=max(metrics[key],float((value-data[key][i]).abs().max()))
                noise=(data['executed'][i]-data['action'][i])/.1
                # Only random execution noise is replaced; production process_actions
                # still applies scale, offset, encoder bias and physics stepping.
                with patch('torch.randn_like',side_effect=lambda value:noise.clone()):
                    obs,reward,term,tout,_=env.step(data['action'][i])
                metrics['reward']=max(metrics['reward'],float((reward-data['reward'][i]).abs().max()))
                metrics['executed']=max(metrics['executed'],float((env.buffer.a_exec[:,i]-data['executed'][i]).abs().max()))
                flags+=int((term!=data['terminated'][i]).sum()+(tout!=data['timeout'][i]).sum())
        limits={'state':.02,'obs':.02,'goal':1.e-6,'frame':0.,'executed':1.e-6,'reward':1.e-4}
        result={'protocol':'g1_random_context_candidate_v1','steps':len(data['action']),
                'max_abs_error':metrics,'limits':limits,'flag_mismatches':flags,
                'passed':flags==0 and all(metrics[k]<=limits[k] for k in limits),
                'source_sha256':source_identity(),'data_sha256':sha256(args.data),
                'reference_fixture_sha256':sha256(args.fixture),
                'scope':'same injected context and executed actions; not equality of RNG streams or fresh training'}
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        if not result['passed']:raise RuntimeError(f'context comparison failed: {args.output}')
        return result
    finally:env.close()


def sha256(path):
    import hashlib
    with open(path,'rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('reference','candidate'))
    p.add_argument('--project',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--fixture',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=24)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.mode=='reference' and not a.checkpoint:p.error('reference mode requires --checkpoint')
    if a.mode=='candidate' and not a.fixture:p.error('candidate mode requires --fixture')
    print(json.dumps(reference(a) if a.mode=='reference' else candidate(a),indent=2))

if __name__=='__main__':main()
