"""Behavioral regressions for shared natural locomotion training and admission."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from examples.locomotion.experiment import Experiment, task_config
from examples.locomotion.assessment import rank_report, rollout
from gobot.rl.locomotion.config import NaturalGaitCfg
from gobot.rl.locomotion.curriculum import directed_increment, progress_masks
from gobot.rl.locomotion.gait_metrics import QuadrupedGaitMetrics
from gobot.rl.locomotion.natural_gait import PHASES, desired_contact, gait_weights
from gobot.rl.locomotion.motion import MotionDataset, ReferenceStateInitializer
from gobot.rl.locomotion.budget import TrainingBudget


def test_directional_curriculum():
    commands=np.array([[1.,0.],[-1.,0.],[1.,0.],[0.,0.]])
    delta=np.array([[1.,0.],[-1.,0.],[0.,3.],[3.,0.]])
    expected=np.array([1.,1.,0.,0.])
    np.testing.assert_allclose(directed_increment(delta,commands),expected)
    torch.testing.assert_close(directed_increment(torch.tensor(delta),torch.tensor(commands)),torch.tensor(expected))
    up,down=progress_masks(np.array([4.,0.,2.,0.]),np.array([4.,4.,0.,.1]),8.)
    np.testing.assert_array_equal(up,[True,False,False,False]);np.testing.assert_array_equal(down,[False,True,False,False])


def test_clock_continuity_and_gait_classification():
    cfg=NaturalGaitCfg(enabled=True)
    speed=torch.tensor([.79999,.80001,1.99999,2.00001])
    weights=gait_weights(speed,cfg)
    torch.testing.assert_close(weights.sum(-1),torch.ones(4))
    assert float((weights[0]-weights[1]).abs().max())<.001
    assert float((weights[2]-weights[3]).abs().max())<.001
    for kind,offset in enumerate(PHASES):
        metrics=QuadrupedGaitMetrics(1,dt=.01,warmup_s=0.)
        for step in range(1200):
            phase=step*.01*2.
            contact=((phase-np.array(offset))%1 < .35)[None]
            position=np.zeros((1,4,3));position[:,:,0]=step*.01
            metrics.update(contact,position,[True])
        result=metrics.arrays()
        key=('walk_fraction','trot_fraction','gallop_fraction')[kind]
        assert result['cycles'][0]>10 and result[key][0]>.8,(kind,result)
    metrics=QuadrupedGaitMetrics(1,dt=.01,warmup_s=0.)
    for step in range(1000):
        metrics.update(np.full((1,4),step%50<20),np.zeros((1,4,3)),[True])
    result=metrics.arrays()
    assert result['simultaneous_landings'][0]>10
    assert max(result[k][0] for k in ('walk_fraction','trot_fraction','gallop_fraction'))==0.


def test_conditional_reference_sampling():
    with tempfile.TemporaryDirectory() as directory:
        arrays={};clips=[]
        for i,(name,gait,speed) in enumerate([('walk_subject1',0,.6),('run_subject2',1,2.),('walk_subject4',0,.6),('run_subject5',1,2.)]):
            features=np.zeros((10,79),np.float32);features[:,58]=speed;features[:,0]=i
            arrays[f'features_{i}']=features;arrays[f'valid_{i}']=np.ones(10,bool)
            roots=np.zeros((10,13),np.float32);roots[:,3]=1;roots[:,7]=speed
            arrays[f'root_{i}']=roots;arrays[f'joint_{i}']=np.zeros((10,58),np.float32)
            clips.append({'name':name,'split':'holdout' if 'subject5' in name else 'train','gait':gait})
        arrays['metadata']=np.array(json.dumps({'schema':'gobot_motion_v1','dt':.02,'clips':clips}))
        path=Path(directory)/'motions.npz';np.savez(path,**arrays)
        dataset=MotionDataset(path)
        pairs,supported=dataset.sample_matching(torch.tensor([0,1,0]),torch.tensor([.6,2.,3.]))
        assert supported.tolist()==[True,True,False]
        assert pairs[0,0]==0 and pairs[1,0]==1
        initializer=ReferenceStateInitializer(path,device='cpu',probability=1.)
        selected,roots,_=initializer.sample(3,torch.Generator().manual_seed(4),torch.tensor([[.6,0.,0.],[2.,0.,0.],[-.6,0.,0.]]))
        assert selected.tolist()==[True,True,False]
        torch.testing.assert_close(roots[:2,7],torch.tensor([.6,2.]))


def test_categorical_commands_and_low_speed_rewards():
    from gobot.rl.locomotion.command import sample_categorical_commands
    from gobot.rl.locomotion.terms import adaptive_tracking_terms
    cfg=task_config(Experiment(robot='g1',stage='flat_control'))
    rng=torch.Generator().manual_seed(5)
    uniform=lambda shape,lo,hi: torch.rand(shape,generator=rng)*(hi-lo)+lo
    command,modes=sample_categorical_commands(cfg.command,100000,uniform,'cpu')
    frequencies=torch.bincount(modes,minlength=6)/len(modes)
    torch.testing.assert_close(frequencies,torch.tensor(cfg.command.mode_probabilities),rtol=0,atol=.005)
    assert (command[modes==0,1:]==0).all() and (command[modes==0,0]>=.4).all()
    assert (command[modes==1,:2]==0).all() and (command[modes==1,2].abs()>=.1).all()
    assert (command[modes==2]==0).all()
    assert (command[modes==3,0]<=-.1).all() and (command[modes==3,1:]==0).all()
    assert (command[modes==4][:,[0,2]]==0).all() and (command[modes==4,1].abs()>=.1).all()
    commands=np.array([[-.1,0,0],[0,.15,0],[0,0,0]],dtype=np.float32)
    zero=np.zeros_like(commands)
    still=adaptive_tracking_terms(commands,zero,zero,cfg.rewards,np)
    matching=adaptive_tracking_terms(commands,commands,zero,cfg.rewards,np)
    opposite=adaptive_tracking_terms(commands,-commands,zero,cfg.rewards,np)
    assert max(still[0][:2])<.4
    np.testing.assert_allclose(matching[0],1.)
    np.testing.assert_allclose(matching[2],[1.,1.,0.])
    np.testing.assert_allclose(opposite[2],[-1.,-1.,0.])
    tensor=adaptive_tracking_terms(torch.tensor(commands),torch.tensor(commands),torch.tensor(zero),cfg.rewards,torch)
    for a,b in zip(matching,tensor):np.testing.assert_allclose(a,b.numpy(),atol=1e-6)


def test_yaw_participates_in_ranking():
    def report(velocity,yaw):
        return {'passed':False,'scenarios':[{'command':[.8,0,0],'cells':[{'admission':0.}],
          'admission':0.,'natural_admission':0.,'velocity_error':velocity,'yaw_error':yaw,'cross_track_max_m':.1,'foot_slip_m2_s2':.1}]}
    assert rank_report(report(.08,.22))>rank_report(report(.05,.7))


def test_gpu_hour_accounting():
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'ledger.json'
        with patch('gobot.rl.locomotion.budget.time.monotonic',return_value=100.) as clock:
            budget=TrainingBudget(path,total_hours=100,run_hours=2,allocated_gpus=4)
            assert budget.limit_seconds==1800
            clock.return_value=130.;budget.close()
        assert json.loads(path.read_text())['seconds']==120.


def integration(backend):
    from gobot.rl.locomotion import make_velocity_env
    from gobot.rl import RslRlVecEnvWrapper
    from gobot.rl.locomotion.ppo import rsl_rl_train_cfg
    from gobot.rl.locomotion.manifest import build_policy_manifest,validate_policy
    for robot in ('go1','g1'):
        exp=Experiment(robot=robot,recipe='natural' if robot=='go1' else 'baseline',num_envs=2)
        cfg=task_config(exp);cfg.observations.actor_noise=False
        cfg.command.source='external';cfg.command.resampling_time_range=(.02,.02)
        env=make_velocity_env(cfg,num_envs=2,backend=backend,device='cuda:0' if backend=='mujoco-warp' else 'cpu',max_episode_length=3,collect_step_extras=False)
        try:
            wrapper=RslRlVecEnvWrapper(env,device=env.device)
            manifest=build_policy_manifest(wrapper,cfg,rsl_rl_train_cfg(obs_normalization=True),Path(cfg.project_path))
            validate_policy(manifest,env,cfg)
            actions=torch.zeros(2,env.num_actions,device=env.device)
            env.set_command((-.1,0,0))
            for _ in range(4):
                _,_,_,reverse_extras=wrapper.step(actions)
                expected=torch.tensor([[-.1,0,0],[-.1,0,0]],device=env.device)
                torch.testing.assert_close(reverse_extras['transition_state']['command'],expected)
            if robot=='go1':assert (env.phase>0).all()
            wrapper.reset(seed=17)
            env.set_command((.8,0,0))
            previous=wrapper.get_observations()['actor'].clone()
            obs,reward,done,extras=wrapper.step(actions)
            assert torch.isfinite(obs['actor']).all() and not done.any()
            saved=obs['actor'].clone()
            for _ in range(2):obs,reward,done,extras=wrapper.step(actions)
            assert done.all() and 'final_observation' in extras
            assert torch.isfinite(extras['transition_state']['joint_effort']).all()
            if robot=='go1':
                assert not torch.equal(obs['actor'][:,-5:-3],extras['final_observation']['actor'][:,-5:-3])
                env.step(actions)
                clock=env.phase.clone();env.reset([0])
                assert env.phase[0]==0 and env.phase[1]==clock[1]
                changed=replace(cfg.gait,run_speed=2.5);cfg.gait=changed
                try:validate_policy(manifest,env,cfg)
                except ValueError:pass
                else:raise AssertionError('mismatched gait clock accepted')
            assert saved.shape==previous.shape
        finally:env.close()
    print(backend,'natural gait observations/reset/manifest integration passed')


def test_status_respects_pid_namespace():
    import os, socket, time
    from examples.locomotion.status import inspect
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'status.json'
        value={'state':'running','pid':99999999,'host':socket.gethostname(),'heartbeat_unix':time.time(),
               'pid_namespace':'different','process_start_ticks':'0'}
        path.write_text(json.dumps(value))
        assert inspect(directory)['state']=='running' # A sandbox cannot disprove a fresh host heartbeat.
        value['pid_namespace']=os.readlink('/proc/self/ns/pid');path.write_text(json.dumps(value))
        assert inspect(directory)['state']=='lost'


def test_campaign_resumes_graceful_stop():
    from examples.locomotion import campaign
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);run_dir=root/'g1_control_s42';run_dir.mkdir()
        (run_dir/'latest.pt').write_bytes(b'checkpoint')
        (root/'campaign.json').write_text(json.dumps({'state':'stopped','initial_gpu_hours':0.,'jobs':[
            {'name':'g1_control_s42','finished':True,'admitted':False,'summary':{'stop_reason':'signal'}}]}))
        calls=[]
        class Process:
            pid=123
            def __init__(self,command):self.command=command;calls.append(command)
            def wait(self):
                out=Path(self.command[self.command.index('--log-dir')+1]);out.mkdir(exist_ok=True)
                (out/'summary.json').write_text(json.dumps({'admitted':False,'stop_reason':'budget'}))
                return 0
        with patch.object(campaign.subprocess,'Popen',Process), patch.object(campaign.signal,'signal'), patch.object(sys,'argv',[
            'campaign','--output',str(root),'--budget-file',str(root/'budget.json'),'--seeds','42','--num-envs','2']):
            campaign.main()
        assert '--resume' in calls[0] and len(calls)==3


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('mujoco-cpu','mujoco-warp'));a=p.parse_args()
    if a.backend:integration(a.backend)
    else:
        for test in (test_directional_curriculum,test_clock_continuity_and_gait_classification,
                     test_conditional_reference_sampling,test_categorical_commands_and_low_speed_rewards,
                     test_yaw_participates_in_ranking,test_gpu_hour_accounting,
                     test_status_respects_pid_namespace,test_campaign_resumes_graceful_stop):
            test();print(test.__name__,'passed')

if __name__=='__main__':main()
