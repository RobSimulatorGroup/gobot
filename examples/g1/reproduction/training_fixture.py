"""Independent pinned-reference replay, BPTT and critic-update fixture."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from examples.g1.reproduction.math_fixture import numerical_packages


def experiment(namespace, output=None):
    import importlib
    import torch
    def mod(name):
        names = {'replay':'data.replay_buffer','policy':'policy.nets','world_model':'world_model.world_model',
                 'heads':'world_model.heads','policy_trainer':'imagination.policy_trainer',
                 'wm_trainer':'world_model.wm_trainer','rotations':'kinematics.rotations'}
        return importlib.import_module(namespace+'.'+(names.get(name,name) if namespace=='supermimic' else name))
    replay, policy, model, heads, trainer, rotations = [mod(s) for s in ('replay','policy','world_model','heads','policy_trainer','rotations')]
    torch.set_num_threads(2)
    torch.manual_seed(2017)

    class FK:
        # A differentiable 29-joint analytic test rig, identical input for both implementations.
        def body_indices(self,names): return list(range(len(names)))
        def body_index(self,name): return 0
        def __call__(self,q,v):
            n = q.shape[0]
            offsets = torch.zeros(n,14,3); offsets[:,:,0]=torch.linspace(-.2,.2,14)
            offsets[:,:,2]=q[:,7:21]*.03
            orientation = q[:,None,3:7].expand(n,14,4)
            return {'pos':q[:,None,:3]+rotations.quat_apply(orientation,offsets), 'quat':orientation,
                    'lin_vel':v[:,None,:3].expand(n,14,3)+v[:,None,6:9]*.02,
                    'ang_vel':rotations.quat_apply(q[:,3:7],v[:,3:6])[:,None].expand(n,14,3)}

    class Library:
        default_joint_pos = torch.zeros(29)
        def last_frame_of(self,frame): return torch.full_like(frame,1000)
        def ref(self,frame):
            from types import SimpleNamespace
            n = frame.numel()
            pos=torch.zeros(n,3);pos[:,2]=.8;pos[:,0]=frame.reshape(-1)*.001
            quat=torch.zeros(n,4);quat[:,0]=1
            return SimpleNamespace(root_pos_w=pos,root_quat_w=quat)
        def bodies(self,frame):
            from types import SimpleNamespace
            r=self.ref(frame);q=torch.cat((r.root_pos_w,r.root_quat_w,torch.zeros(len(frame),29)),-1)
            b=FK()(q,torch.zeros(len(frame),35))
            return SimpleNamespace(pos_w=b['pos'],quat_w=b['quat'],lin_vel_w=b['lin_vel'],ang_vel_w=b['ang_vel'])
        def goal_window(self,frame,k):
            return (frame[:,None]*.0001).expand(-1,k*68).clone()

    buf=replay.ReplayBuffer(2,24,71,29,33,'cpu')
    for t in range(38):
        s=torch.randn(2,71)*.01;s[:,2]+=.8;s[:,3:7]=torch.tensor([1.,0.,0.,0.])
        a=torch.randn(2,29)*.03;xi=torch.zeros(2,33);xi[:,0]=.8
        terminal=torch.tensor([t in (12,31),t in (14,30)])
        start=torch.tensor([t in (0,13,32),t in (0,15,31)])
        buf.add(torch.arange(2),s,a,a+.01,xi,torch.ones(2)*.09,terminal.float(),torch.full((2,),t),torch.zeros(2,dtype=torch.long),terminal,start)
        buf.end_episode(terminal.nonzero().flatten())
    torch.manual_seed(983)
    windows=buf.sample_windows(9,8)
    starts=buf.sample_starts(11,10)
    ah=heads.AnalyticHeads(FK(),tuple(str(i) for i in range(14)),'7','0',torch.tensor([[-1.,1.]]*29),torch.zeros(29))
    wm=model.WorldModel(model.WorldModelCfg(hidden=(24,16),term_hidden=(8,8)),ah,'cpu')
    with torch.no_grad(): wm.dynamics.mlp.net[-1].weight.normal_(std=.001)
    pcfg=policy.PolicyCfg(encoder_hidden=(24,16),decoder_hidden=(24,16))
    pi=policy.Policy(pcfg);critic=policy.Critic(policy.CriticCfg(hidden=(16,8)),pcfg=pcfg)
    cfg=trainer.PolicyTrainerCfg(horizon=4,batch_size=4,critic_epochs=2,critic_minibatches=2,compile=None)
    pt=trainer.PolicyTrainer(wm,pi,critic,Library(),cfg)
    wm_trainer=mod('wm_trainer')
    wt=wm_trainer.WorldModelTrainer(wm,Library(),wm_trainer.WMTrainerCfg(batch_size=4,batches_per_update=2,compile=None))
    # Freeze RNG immediately before sampling/noise/critic permutations.
    torch.manual_seed(3901)
    wm_metrics=wt.update(buf)
    metrics=pt.update(buf)
    result={'window_mask':buf.window_mask(8),'windows':vars(windows),'starts':vars(starts),
            'wm_metrics':wm_metrics,'policy_metrics':metrics,
            'wm':wm.state_dict(),'policy':pi.state_dict(),'critic':critic.state_dict(),
            'wm_grads':{n:p.grad for n,p in wm.named_parameters()},'policy_grads':{n:p.grad for n,p in pi.named_parameters()},
            'critic_grads':{n:p.grad for n,p in critic.named_parameters()},
            'wm_opt':wt.opt.state_dict(),'policy_opt':pt.opt.state_dict(),'critic_opt':pt.critic_opt.state_dict()}
    if output:
        torch.save(result,output)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--mjlab',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();numerical_packages(a.reference,a.mjlab)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    experiment('supermimic',a.output)
    sources={name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for name,m in sys.modules.items()
             if name.startswith('supermimic.') and getattr(m,'__file__',None) and m.__file__.endswith('.py')}
    info={'protocol':'supermimic_reproduced_training_v1','gobot_imported':'gobot' in sys.modules,
          'fixture_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'source_hashes':sources,
          'scope':'independent replay plus full WM/actor/critic update; analytic test rig, not simulator rollout'}
    a.output.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))

if __name__=='__main__': main()
