"""Behavioral contracts for the retained aligned AMP workflow."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from gobot.rl.motion_amp.features import Features
from gobot.rl.motion_amp.motion import MotionLibrary

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT/'examples/g1/assets/mimickit/gobot'


def features(**kwargs):
    return Features(dict(rotation_dof_indices=[0,-1,1],rotation_axes=[[0,1,0],[1,0,0],[0,0,1]],
        body_names=['base','left','fixed','right'],key_bodies=['left','right'],joint_names=['a','b'],
        action_low=[-2,-3],action_high=[2,3]),'cpu',**kwargs)


def test_root_velocity_uses_rotated_body_com_and_keeps_origin_state():
    f=features(base_center_of_mass=(0.,0.,-.1))
    # Rotate the local -Z offset to world -X, then spin about world +Z.
    quaternion=torch.tensor([[0.,2**-.5,0.,2**-.5]])
    angular=torch.tensor([[0.,0.,2.]])
    origin=torch.tensor([[1.,0.,0.]])
    com=f.root_com_velocity(quaternion,origin,angular)
    torch.testing.assert_close(com,torch.tensor([[1.,-.2,0.]]))
    torch.testing.assert_close(f.root_origin_velocity(quaternion,com,angular),origin)
    state=SimpleNamespace(base_pose=torch.cat((torch.zeros(1,3),quaternion),-1),
        base_velocity=torch.cat((origin,angular),-1),joint_position=torch.zeros(1,2),
        joint_velocity=torch.zeros(1,2),link_pose=torch.zeros(1,4,7))
    before=state.base_velocity.clone()
    actor,frame=f.live(state)
    torch.testing.assert_close(actor[:,7:10],com)
    torch.testing.assert_close(frame[:,-8:-5],com)
    torch.testing.assert_close(state.base_velocity,before,rtol=0,atol=0)
    with pytest.raises(ValueError,match='center of mass'):
        features(base_center_of_mass=(0.,float('nan'),0.))


def test_actor_and_history_translation_and_time_order():
    f=features()
    p=torch.tensor([[4.,2.,.8],[4.1,2.1,.7]])
    r=torch.tensor([[0.,0.,0.,1.]]).expand(2,-1)
    q=torch.tensor([[.1,-.2],[.3,-.4]])
    j=f.joint_rotations(q)
    v=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
    keys=p[:,None]+torch.tensor([[[0.,.1,-.5],[0.,-.1,-.5]]])
    args=(p,r,v,v,j,q,keys)
    actor=f.actor(*args)
    frames=f.frame(*args)
    assert actor.shape==(2,39)
    torch.testing.assert_close(actor[:,0],p[:,2])
    torch.testing.assert_close(actor[:,1:7],torch.tensor([[1.,0,0,0,0,1.]]).expand(2,-1))
    shift=torch.tensor([100.,-30.,0.])
    torch.testing.assert_close(f.actor(p+shift,r,v,v,j,q,keys+shift),actor,atol=1.e-5,rtol=1.e-5)
    disc=f.discriminator(frames[None]).reshape(1,2,-1)
    torch.testing.assert_close(disc[0,:,0],torch.tensor([-.1,0.]))
    torch.testing.assert_close(disc[0,:,2],p[:,2])
    # Each frame retains its own velocity and temporal position.
    torch.testing.assert_close(disc[0,:,-8:-5],v)
    torch.testing.assert_close(f.joint_coordinates(j),q)
    assert f.action_spec.names==('a','b')


def test_motion_wrap_and_held_velocity(tmp_path):
    f=features();count=3
    quats=torch.tensor([[0.,0.,0.,1.]]).expand(count,-1)
    arrays=dict(root_pos=torch.tensor([[0.,0.,1.],[1.,0.,1.],[2.,0.,1.]]),root_rot=quats,
        root_vel=torch.tensor([[1.,0.,0.],[2.,0.,0.],[2.,0.,0.]]),root_ang_vel=torch.zeros(count,3),
        joint_rot=f.joint_rotations(torch.zeros(count,2)),dof_vel=torch.zeros(count,2))
    path=tmp_path/'motions.npz'
    np.savez(path,**{'walk_'+k:v.numpy() for k,v in arrays.items()})
    profile={'files':{'motions.npz':hashlib.sha256(path.read_bytes()).hexdigest()},
        'motions':[{'name':'walk','length':1.,'wrap_delta':[2.,0.,0.]}], 'key_bodies':['left','right']}
    library=MotionLibrary(path,profile,'walk',f,SimpleNamespace(body_names=['left','right']),'cpu')
    times=torch.tensor([-.25,0.,.25,.75,1.,1.25])
    p,r,v,w,j,qd=library.sample(times)
    torch.testing.assert_close(p[:,0],2*times)
    torch.testing.assert_close(v[:,0],torch.tensor([2.,1.,1.,2.,1.,1.]))
    assert torch.isfinite(j).all()
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='digest'):
        MotionLibrary(path,profile,'walk',f,SimpleNamespace(body_names=['left','right']),'cpu')


@pytest.fixture
def env():
    if not (PROJECT/'world.jscn').exists():
        pytest.skip('prepare optional Gobot AMP project for integration tests')
    from gobot.rl.motion_amp import MotionAMPEnv
    torch.set_num_threads(4)
    value=MotionAMPEnv(PROJECT,num_envs=2)
    try:yield value
    finally:value.close()


def test_masked_reset_terminal_history_and_finite_actions(env):
    env.reset(seed=3)
    before=env.history.clone()
    other={k:v[1].clone() for k,v in env.batch.read_state().__dict__.items()}
    env.reset(mask=[True,False],phase=[.2])
    assert not torch.equal(env.history[0],before[0])
    torch.testing.assert_close(env.history[1],before[1],atol=0,rtol=0)
    for k,v in env.batch.read_state().__dict__.items():
        torch.testing.assert_close(v[1],other[k],atol=0,rtol=0)
    with pytest.raises(ValueError,match='finite'):
        env.step(torch.full((2,29),float('nan')))
    history=env.history.clone()
    env.steps[0]=env.max_steps-1
    state=env.step(env.batch.read_state().joint_position.clone())
    assert state.truncated[0] and state.final_observation is not None
    torch.testing.assert_close(env.history[:,:-1],history[:,1:],atol=0,rtol=0)
    terminal=state.obs['disc'].clone()
    with pytest.raises(RuntimeError,match='reset'):
        env.step(torch.zeros(2,29))
    env.reset(mask=state.done)
    torch.testing.assert_close(state.obs['disc'],terminal,atol=0,rtol=0)
    assert not torch.equal(env.history[0],history[0])
    a=env.reset(seed=44).obs['actor'].clone()
    b=env.reset(seed=44).obs['actor']
    torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_rsl_wrapper_autoreset_keeps_terminal_history_and_previous_observation(env):
    from gobot.rl.rsl_rl import RslRlVecEnvWrapper
    env.autoreset=True;env.reset(seed=13)
    wrapper=RslRlVecEnvWrapper(env)
    before=wrapper.get_observations();owned=before['actor'].clone()
    wrapper.episode_length_buf=torch.full((2,),env.max_steps-1,dtype=torch.long)
    after,reward,done,extras=wrapper.step(env.batch.read_state().joint_position.clone())
    assert done.all()
    torch.testing.assert_close(before['actor'],owned,atol=0,rtol=0)
    assert env.steps.max()==0
    assert extras['transition_observation']['disc'].shape==(2,2390)
    assert not torch.equal(extras['final_observation']['actor'],after['actor'])
    assert set(after.keys())=={'actor','policy'}
    # Persistent observations must stay within the common wrapper's cache bound.
    for _ in range(12):wrapper.step(env.batch.read_state().joint_position.clone())
    assert wrapper.memory_profile()['native_obs_tensordicts']<=3


def test_amp_keeps_rsl_ppo_and_bootstraps_only_terminal_timeout(monkeypatch):
    from rsl_rl.algorithms import PPO
    from gobot.rl.motion_amp.algorithm import MotionAMP, AMPConfig
    assert MotionAMP.act is PPO.act
    assert MotionAMP.compute_returns is PPO.compute_returns
    captured={}
    monkeypatch.setattr(PPO,'process_env_step',lambda self,obs,rewards,dones,extras:
        captured.update(rewards=rewards.clone(),extras=extras))
    alg=MotionAMP.__new__(MotionAMP)
    alg.amp_cfg=AMPConfig(observation_dim=3)
    alg.discriminator=torch.nn.Linear(3,1)
    torch.nn.init.zeros_(alg.discriminator.weight);torch.nn.init.zeros_(alg.discriminator.bias)
    alg._normalize=lambda x:x
    alg._frames=[];alg._reward_sum=0.;alg.device='cpu';alg.gamma=.9
    alg.critic=lambda obs:obs['actor'][:,:1]
    extras={'transition_observation':{'disc':torch.zeros(2,3)},
            'time_outs':torch.ones(2,dtype=torch.bool),'terminated':torch.tensor([True,False]),
            'final_observation':{'actor':torch.tensor([[100.],[7.]])}}
    rewards=torch.zeros(2)
    alg.process_env_step(None,rewards,torch.ones(2,dtype=torch.bool),extras)
    torch.testing.assert_close(captured['rewards']-rewards,torch.tensor([0.,6.3]))
    assert 'time_outs' not in captured['extras']
    assert rewards.min()>0


def test_manifest_rejects_changed_control(env):
    from gobot.rl.motion_amp.training import manifest_for,validate_manifest
    from gobot.rl.policy import POLICY_MANIFEST_KEY
    payload={POLICY_MANIFEST_KEY:manifest_for(env).metadata()}
    validate_manifest(payload,env)
    payload[POLICY_MANIFEST_KEY]['control']['kp'][0]+=1
    with pytest.raises(ValueError,match='manifest'):validate_manifest(payload,env)


def test_reset_matches_reference_com_velocity_and_rejects_old_manifest(env):
    from gobot.rl.motion_amp.training import manifest_for,validate_manifest
    from gobot.rl.policy import POLICY_MANIFEST_KEY
    phases=torch.tensor([.1,.7])
    _,r,v,w,_,_=env.motion.sample(phases)
    state=env.reset(phase=phases)
    physical=env.batch.read_state()
    expected=env.features.root_origin_velocity(r,v,w)
    torch.testing.assert_close(physical.base_velocity[:,:3],expected,atol=2.e-6,rtol=2.e-6)
    torch.testing.assert_close(state.obs['actor'][:,7:10],v,atol=2.e-6,rtol=2.e-6)
    assert (physical.base_velocity[:,:3]-v).abs().max()>.001
    manifest=manifest_for(env).metadata()
    manifest['observation_spec']['version']='motion_amp_global_v1'
    with pytest.raises(ValueError,match='manifest'):
        validate_manifest({POLICY_MANIFEST_KEY:manifest},env)


def test_joint_range_gaussian_preserves_normalized_likelihood_ratios():
    from gobot.rl.motion_amp.models import JointRangeGaussian
    scale=torch.tensor([.4,3.]);offset=torch.tensor([.2,-.3])
    dist=JointRangeGaussian(2,action_scale=scale,action_offset=offset)
    old=torch.tensor([[.1,.2],[-.3,.4]])
    new=old+.01
    normalized_action=old+torch.tensor([[.02,-.03],[.01,.04]])
    physical=normalized_action*scale+offset
    dist.update(old);old_log=dist.log_prob(physical)
    torch.testing.assert_close(dist.std, (.05*scale).expand_as(old))
    assert not dist.std_param.requires_grad
    dist.update(new);new_log=dist.log_prob(physical)
    ref_old=torch.distributions.Normal(old,.05)
    ref_new=torch.distributions.Normal(new,.05)
    torch.testing.assert_close(new_log-old_log,
        (ref_new.log_prob(normalized_action)-ref_old.log_prob(normalized_action)).sum(-1),atol=2.e-6,rtol=2.e-6)
    torch.testing.assert_close(dist.deterministic_output(new),new*scale+offset)
    torch.testing.assert_close(dist.as_deterministic_output_module()(new),new*scale+offset)


def test_reference_model_checkpoint_inference_matches_training_actor(env):
    from tensordict import TensorDict
    from gobot.rl.motion_amp.models import ReferenceMLP
    from gobot.rl.motion_amp.training import manifest_for
    from gobot.rl.checkpoint_policy import CheckpointPolicy
    low=torch.tensor(env.profile['action_low']);high=torch.tensor(env.profile['action_high'])
    dist={'class_name':'gobot.rl.motion_amp.models:JointRangeGaussian',
          'action_scale':((high-low)/2).tolist(),'action_offset':((high+low)/2).tolist()}
    obs=TensorDict({'actor':env.reset(seed=1).obs['actor'].clone()},[2])
    actor=ReferenceMLP(obs,{'actor':['actor']},'actor',29,hidden_dims=(16,8),activation='relu',
        obs_normalization=True,distribution_cfg=dist)
    actor.update_normalization(obs)
    manifest=manifest_for(env,{'activation':'relu','obs_normalizer_eps':0.,'obs_normalizer_clip':10.,
        'distribution_cfg':dist})
    restored=CheckpointPolicy(actor.state_dict(),manifest)
    # Include inputs well outside the normalizer clip range.
    query=TensorDict({'actor':torch.cat((obs['actor'],obs['actor']*100))},[4])
    with torch.no_grad():
        torch.testing.assert_close(restored(query['actor']),actor(query),atol=1.e-6,rtol=1.e-6)
    bad=deepcopy(actor.state_dict());bad['distribution.output_transform.scale'][0]*=2
    with pytest.raises(ValueError,match='transform'):
        CheckpointPolicy(bad,manifest)


def test_clipped_normalization_floor_and_sample_limit():
    from gobot.rl.motion_amp.models import ClippedNormalization
    norm=ClippedNormalization(2,until=4)
    values=torch.tensor([[1.,2.],[1.,4.],[1.,6.],[1.,8.]])
    norm.update(values)
    torch.testing.assert_close(norm._std,torch.tensor([[1.e-4,5**.5]]))
    before=deepcopy(norm.state_dict())
    norm.update(values+100)
    for key,value in norm.state_dict().items():torch.testing.assert_close(value,before[key])
    assert norm(torch.tensor([[2.,100.]])).tolist()==[[10.,10.]]


def test_aligned_amp_keeps_normalizer_fixed_during_discriminator_updates(monkeypatch):
    from dataclasses import asdict
    from tensordict import TensorDict
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    from gobot.rl.motion_amp.algorithm import MotionAMP,AMPConfig
    obs=TensorDict({'actor':torch.zeros(2,3)},[2])
    groups={'actor':['actor'],'critic':['actor']}
    actor=MLPModel(obs,groups,'actor',1,hidden_dims=(4,),distribution_cfg={'class_name':'GaussianDistribution'})
    critic=MLPModel(obs,groups,'critic',1,hidden_dims=(4,))
    storage=RolloutStorage('rl',2,2,obs,[1],'cpu')
    cfg=AMPConfig(observation_dim=3,hidden_dims=(4,3),batch_size=2,updates_per_iteration=3,
        replay_capacity=16,optimizer='sgd',momentum=.9,normalization_update='rollout',initialization='reference')
    alg=MotionAMP(actor,critic,storage,amp_cfg=asdict(cfg),optimizer='sgd',optimizer_momentum=.9,advantage_clip=4.)
    with torch.inference_mode():storage.advantages=torch.full_like(storage.advantages,9.)
    frames=torch.arange(12,dtype=torch.float32).reshape(4,3)
    demos=frames+20
    calls=[];counts=[]
    alg.demo_sampler=lambda n:(calls.append(n) or demos.clone())
    alg._frames=[frames];alg._reward_sum=0.
    monkeypatch.setattr(PPO,'update',lambda self:{})
    hook=alg.discriminator.register_forward_pre_hook(lambda module,args:counts.append(int(alg.disc_normalizer.count)))
    metrics=alg.update();hook.remove()
    assert calls==[4] and counts==[0]*6
    assert int(alg.disc_normalizer.count)==8
    torch.testing.assert_close(alg.disc_normalizer.mean,torch.cat((frames,demos)).mean(0))
    assert storage.advantages.max()==4.
    assert isinstance(alg.optimizer,torch.optim.SGD) and alg.optimizer.param_groups[0]['momentum']==.9
    assert isinstance(alg.disc_optimizer,torch.optim.SGD) and alg.disc_optimizer.param_groups[0]['momentum']==.9
    assert all(np.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize('field', ['base_pose','base_velocity','joint_position','joint_velocity','link_pose','contact_force'])
def test_numerical_failure_terminates_only_bad_lane_and_autoresets(env,monkeypatch,tmp_path,field):
    env.autoreset=True
    env.diagnostics_path=tmp_path/'diagnostics'
    env.reset(seed=42)
    before=env.state.obs['actor'].clone()
    env.steps[1]=env.max_steps-1
    original=env.batch.step_control
    def corrupt():
        physical=original()
        if field=='contact_force':env.batch.read_contacts()['force'][0,0,0]=float('nan')
        else:getattr(physical,field)[0].flatten()[0]=float('nan')
        return physical
    monkeypatch.setattr(env.batch,'step_control',corrupt)
    with pytest.warns(RuntimeWarning,match='numerical failure'):
        state=env.step(env.batch.read_state().joint_position.clone())
    assert state.terminated[0] and state.truncated[1]
    assert state.info['transition_observation']['valid'].tolist()==[False,True]
    assert torch.isfinite(state.obs['actor']).all() and torch.isfinite(state.final_observation['actor']).all()
    assert torch.isfinite(state.info['transition_observation']['disc']).all()
    assert torch.count_nonzero(state.final_observation['actor'][0])==0
    assert not env.pending_reset.any() and env.numerical_failure_count==1
    for value in env.batch.read_state().__dict__.values():assert torch.isfinite(value).all()
    snapshot=torch.load(tmp_path/'diagnostics/first.pt',weights_only=True)
    assert snapshot['record']['env_ids']==[0]
    torch.testing.assert_close(snapshot['previous_actor_observation'],before[:1])
    monkeypatch.setattr(env.batch,'step_control',original)
    next_state=env.step(env.batch.read_state().joint_position.clone())
    assert next_state.info['transition_observation']['valid'].all()


def test_amp_excludes_invalid_transition_from_reward_replay_and_bootstrap(monkeypatch):
    from rsl_rl.algorithms import PPO
    from gobot.rl.motion_amp.algorithm import MotionAMP,AMPConfig
    captured={}
    monkeypatch.setattr(PPO,'process_env_step',lambda self,obs,rewards,dones,extras:
        captured.update(rewards=rewards.clone()))
    alg=MotionAMP.__new__(MotionAMP)
    alg.amp_cfg=AMPConfig(observation_dim=3)
    alg.discriminator=torch.nn.Linear(3,1)
    torch.nn.init.zeros_(alg.discriminator.weight);torch.nn.init.zeros_(alg.discriminator.bias)
    alg._normalize=lambda x:x
    alg._frames=[];alg._reward_sum=0.;alg._invalid_count=0;alg.device='cpu';alg.gamma=.9
    alg.critic=lambda obs:obs['actor'][:,:1]
    frames=torch.tensor([[float('nan')]*3,[1.,2.,3.]])
    extras={'transition_observation':{'disc':frames,'valid':torch.tensor([False,True])},
        'time_outs':torch.ones(2,dtype=torch.bool),'terminated':torch.tensor([True,False]),
        'final_observation':{'actor':torch.tensor([[float('nan')],[7.]])}}
    rewards=torch.zeros(2)
    alg.process_env_step(None,rewards,torch.ones(2,dtype=torch.bool),extras)
    assert captured['rewards'][0]==0 and rewards[0]==0 and alg._invalid_count==1
    torch.testing.assert_close(captured['rewards'][1]-rewards[1],torch.tensor(6.3))
    torch.testing.assert_close(alg._frames[0],frames[1:])
    alg.replay=torch.empty(4,3);alg.replay_size=alg.replay_cursor=0
    alg._remember(torch.cat(alg._frames))
    assert alg.replay_size==1
    torch.testing.assert_close(alg.replay[:1],frames[1:])
    extras['terminated'][0]=False
    with pytest.raises(ValueError,match='must terminate'):
        alg.process_env_step(None,rewards,torch.ones(2,dtype=torch.bool),extras)


@pytest.mark.parametrize('backend',['mujoco-cpu','mujoco-warp'])
def test_training_recovers_from_numerical_failure_without_poisoning_checkpoint(tmp_path,monkeypatch,backend):
    import os
    if backend=='mujoco-warp' and os.environ.get('GOBOT_RUN_MOTION_AMP_GPU_TEST')!='1':
        pytest.skip('set GOBOT_RUN_MOTION_AMP_GPU_TEST=1 for real CUDA reset recovery')
    if not (PROJECT/'world.jscn').exists():pytest.skip('optional motion project')
    from gobot.rl.articulation import ArticulationBatch
    from gobot.rl.motion_amp.training import Config,run
    from gobot.rl.training.workflow import update_config
    config=update_config(Config(),json.loads((ROOT/'examples/g1/mimickit/rsl_reference_aligned.json').read_text()))
    config=update_config(config,{'project_path':str(PROJECT),'backend':backend,
        'device':'cuda:0' if backend=='mujoco-warp' else 'cpu','num_envs':4,'updates':2,
        'output':str(tmp_path/'training'),'amp':{'batch_size':16,'updates_per_iteration':2,'replay_capacity':1024}})
    original=ArticulationBatch.step_control
    calls=0
    def inject(batch,**kwargs):
        nonlocal calls
        physical=original(batch,**kwargs);calls+=1
        if calls==3:
            if backend=='mujoco-warp':
                for name in ('qpos','qvel','qacc_warmstart'):batch._provider.arrays[name][0]=float('nan')
                batch.refresh_kinematics();physical=batch.read_state()
            else:physical.base_pose[0,0]=float('nan')
            # Reproduce the original missed contact-termination case.
            for value in batch.read_contacts().values():value[0]=0
        elif calls>3 and backend=='mujoco-warp':
            for name in ('qpos','qvel','qacc_warmstart'):assert torch.isfinite(batch._provider.arrays[name]).all()
        return physical
    monkeypatch.setattr(ArticulationBatch,'step_control',inject)
    with pytest.warns(RuntimeWarning,match='numerical failure'):result=run(config)
    assert result['state']=='trained' and calls==64
    payload=torch.load(result['checkpoint'],map_location='cpu',weights_only=True)
    assert len(payload['motion_amp']['replay'])==255
    def check(value):
        if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
        elif isinstance(value,dict):
            for item in value.values():check(item)
        elif isinstance(value,(tuple,list)):
            for item in value:check(item)
    check(payload)
    events=(tmp_path/'training/numerical_failures/events.jsonl').read_text().splitlines()
    assert len(events)==1 and json.loads(events[0])['env_ids']==[0]
