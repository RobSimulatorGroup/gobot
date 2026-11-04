"""Compare a full-network update using real G1 replay and independent G1 FK.

The reference process imports only pinned numerical sources and MuJoCo FK;
the candidate uses Gobot's authored scene FK. Both consume the same recorded
transitions, initialized tensors and sampling/noise seed.
"""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import sys


def execute(args):
    import torch
    from .check_context import sha256
    torch.set_num_threads(2)
    if args.mode=='reference':
        from .math_fixture import numerical_packages
        numerical_packages(args.reference,args.mjlab)
        prefix='supermimic'
        paths={'heads':'world_model.heads','world_model':'world_model.world_model','replay':'data.replay_buffer',
               'policy':'policy.nets','wm_trainer':'world_model.wm_trainer','policy_trainer':'imagination.policy_trainer',
               'motion_library':'envs.mdp.motion_library'}
    else:
        prefix='gobot.rl.supermimic';paths={}
    def module(name):return importlib.import_module(prefix+'.'+paths.get(name,name))
    wm_mod,head_mod,pi_mod,wm_train,pi_train,lib_mod,replay_mod=map(module,
        ('world_model','heads','policy','wm_trainer','policy_trainer','motion_library','replay'))
    profile=json.loads((args.project/'assets/supermimic/robot_profile.json').read_text())
    context=None
    if args.mode=='reference':
        from supermimic.data.motion_convert import load_g1_model
        from supermimic.kinematics.g1_fk import G1Kinematics
        model=load_g1_model(args.mjlab/'src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml')
        fk=G1Kinematics(model,device='cpu')
        limits=torch.tensor(model.jnt_range[1:],dtype=torch.float32)
        middle=limits.mean(-1);half=(limits[:,1]-limits[:,0])*.45
    else:
        import gobot
        from gobot.rl.kinematics import SceneKinematics
        from gobot.rl.motion_tracking.kinematics import SceneStateKinematics
        context=gobot.app.create_context();context.set_project_path(str(args.project.resolve()))
        context.load_scene('res://g1_29dof_flat_scene.jscn')
        scene=SceneKinematics(context.compile_physics_scene_snapshot(),robot_name=profile['robot_name'],
            base_link=profile['base_link'],joint_names=profile['joint_names'],device='cpu')
        fk=SceneStateKinematics(scene)
        middle=(scene.q_lo+scene.q_hi)/2;half=(scene.q_hi-scene.q_lo)*.45
    try:
        q0=torch.tensor(profile['default_joint_position'])
        names=tuple(profile['tracked_links'])
        heads=head_mod.AnalyticHeads(fk,names,profile['anchor_link'],profile['base_link'],torch.stack((middle-half,middle+half),-1),q0)
        lib=lib_mod.MotionLibrary([args.data],names,q0,device='cpu')
        data=torch.load(args.transitions,weights_only=True,map_location='cpu')
        n=data['state'].shape[1]
        buffer=replay_mod.ReplayBuffer(n,96,71,29,33,'cpu')
        for i in range(len(data['action'])):
            buffer.add(torch.arange(n),data['state'][i],data['action'][i],data['executed'][i],data['xi'],data['reward'][i],
                data['terminated'][i].float(),data['frame'][i],lib.frame_clip[data['frame'][i]],
                torch.zeros(n,dtype=torch.bool),torch.full((n,),i==0,dtype=torch.bool))
        if data['terminated'].any() or data['timeout'].any():raise ValueError('fixture must contain one uninterrupted episode')
        torch.manual_seed(8281)
        wm=wm_mod.WorldModel(wm_mod.WorldModelCfg(),heads,'cpu')
        pi=pi_mod.Policy(pi_mod.PolicyCfg());critic=pi_mod.Critic(pi_mod.CriticCfg())
        wm_trainer=wm_train.WorldModelTrainer(wm,lib,wm_train.WMTrainerCfg(batch_size=4,batches_per_update=1,compile=None))
        pi_trainer=pi_train.PolicyTrainer(wm,pi,critic,lib,pi_train.PolicyTrainerCfg(batch_size=4,horizon=32,compile=None))
        torch.manual_seed(3347)
        wm_metrics=wm_trainer.update(buffer)
        pi_metrics=pi_trainer.update(buffer)
        actual={'wm_metrics':wm_metrics,'policy_metrics':pi_metrics,
                'wm':wm.state_dict(),'policy':pi.state_dict(),'critic':critic.state_dict(),
                'wm_grad':{n:p.grad for n,p in wm.named_parameters()},
                'policy_grad':{n:p.grad for n,p in pi.named_parameters()},
                'critic_grad':{n:p.grad for n,p in critic.named_parameters()}}
        provenance={'protocol':'real_g1_full_network_update_v1','mode':args.mode,'gobot_imported':'gobot' in sys.modules,
                    'transitions_sha256':sha256(args.transitions),'data_sha256':sha256(args.data),
                    'horizon':32,'wm_batch':4,'policy_batch':4,'wm_batches':1,'critic_epochs':5,'critic_minibatches':4,
                    'parameter_counts':{k:sum(p.numel() for p in m.parameters()) for k,m in (('wm',wm),('policy',pi),('critic',critic))}}
        if args.mode=='reference':
            torch.save(actual,args.output)
            provenance['fixture_sha256']=sha256(args.output)
            args.output.with_suffix('.json').write_text(json.dumps(provenance,indent=2)+'\n')
            return provenance
        expected=torch.load(args.fixture,weights_only=True,map_location='cpu')
        meta=json.loads(args.fixture.with_suffix('.json').read_text())
        for key in ('transitions_sha256','data_sha256'):
            if meta[key]!=provenance[key]:raise ValueError(f'reference input identity differs: {key}')
        report={};passed=True
        for group in actual:
            maximum=0.;failed=[]
            for key,value in actual[group].items():
                target=expected[group][key]
                if value is None:
                    ok=target is None;difference=0.
                elif isinstance(value,torch.Tensor):
                    difference=float((value-target).abs().max())
                    ok=bool(torch.allclose(value,target,atol=1e-6,rtol=1e-4))
                else:
                    difference=abs(value-target);ok=difference<=1e-5+1e-4*abs(target)
                maximum=max(maximum,difference)
                if not ok:failed.append(key)
            report[group]={'max_abs_error':maximum,'failed_keys':failed};passed &= not failed
        provenance.update(comparisons=report,passed=bool(passed),scope='full networks, real replay and G1 FK; one update, not convergence')
        args.output.write_text(json.dumps(provenance,indent=2)+'\n')
        if not passed:raise RuntimeError(f'G1 full update comparison failed: {args.output}')
        return provenance
    finally:
        if context:context.clear_scene()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('reference','candidate'))
    for key in ('project','data','transitions','output'):
        p.add_argument('--'+key,type=Path,required=True)
    for key in ('reference','mjlab','fixture'):p.add_argument('--'+key,type=Path)
    args=p.parse_args();print(json.dumps(execute(args),indent=2))

if __name__=='__main__':main()
