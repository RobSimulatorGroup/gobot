"""Real native tracking resets, pre-reset transitions and deterministic replay."""
from pathlib import Path
import pytest
import torch
from gobot.rl.motion_tracking import MotionTrackingCfg, make_motion_tracking_env
from gobot.rl.supermimic.geometry import flatten_state
from gobot.rl.supermimic import SuperMimicConfig, SuperMimicLearner

PROJECT = Path(__file__).resolve().parents[2] / 'examples/g1'
MOTION = PROJECT / 'assets/motions/tracking/walk1_subject1.csv'


@pytest.fixture
def env():
    if not MOTION.exists():
        pytest.skip('prepare the pinned tracking motion assets first')
    torch.set_num_threads(2)
    value = make_motion_tracking_env(MotionTrackingCfg(project_path=PROJECT, motion_files=(str(MOTION),)),num_envs=2)
    try:
        yield value
    finally:
        value.close()


def reference_action(env):
    return (env.batch.read_state().joint_position-env.default_q)/env.action_scale


def test_partial_reset_preserves_other_environment(env):
    env.reset_to_frames(torch.tensor([200, 300]))
    assert env.state.obs['actor'].shape == (2, 99)
    assert env.state.obs['goal'].shape == (2, 730)
    assert env.state.obs['critic'].shape == (2, env.critic_observation_spec.dim)
    env.step(reference_action(env))
    state = env.state.info['supermimic_state']
    kept = {key:value[1].clone() for key,value in state.items()}
    env.reset_to_frames(torch.tensor([500]), [0])
    for key,value in env.state.info['supermimic_state'].items():
        torch.testing.assert_close(value[1],kept[key],atol=2.e-7,rtol=1.e-6)
    assert env.f[0] == 500 and env.ep_step[0] == 0
    assert torch.equal(env.a_prev[0],torch.zeros(29))


def test_clip_end_retains_transition_before_autoreset(env):
    end = int(env.supermimic_mdp.motion.clip_end[0])
    env.reset_to_frames(torch.tensor([end-1, 100]))
    initial = env.state.info['supermimic_state']['s'].clone()
    result = env.step(reference_action(env))
    transition = result.info['supermimic']
    assert result.truncated[0] and transition['done'][0]
    assert transition['f'][0] == end-1
    assert env.f[0] == 0 and env.ep_step[0] == 0
    torch.testing.assert_close(transition['s'],initial)
    assert result.info['terminal_frames'][0] == end
    saved = transition['s1'].clone()
    assert not torch.allclose(saved[0],result.info['supermimic_state']['s'][0])
    env.step(reference_action(env))
    torch.testing.assert_close(transition['s1'],saved,rtol=0,atol=0)
    assert torch.isfinite(saved).all()


def test_native_control_and_reference_reset_replay(env):
    trajectories = []
    for _ in range(2):
        env.reset_to_frames(torch.tensor([200, 300]))
        values=[]
        action=reference_action(env).clone()
        for _ in range(10):
            values.append(env.step(action).info['supermimic']['s1'])
        trajectories.append(torch.stack(values))
    torch.testing.assert_close(trajectories[0],trajectories[1],rtol=0,atol=0)
    assert torch.isfinite(trajectories[0]).all()
    # A fixed PD target must not explode the small-inertia wrists.
    assert trajectories[0][...,13:42].abs().max() < 4


def test_world_model_and_bptt_backprop_on_real_scene(env,tmp_path):
    cfg = SuperMimicConfig(hist=2,goal_frames=2,horizon=4,wm_window=2,buffer_steps=24,
        wm_batch=4,pi_batch=4,test_hidden_dims=(16,16),amp='none',compile=False,
        log_backend='no_print',critic_epochs=1,critic_minibatches=1)
    learner = SuperMimicLearner(env,cfg,tmp_path)
    env.reset_to_frames(torch.tensor([200,300]))
    learner.collect(8)
    loss,error = learner.wm_update()
    result = learner.policy_update()
    assert all(torch.isfinite(v).all() for v in (loss,error,*result))


def test_g1_fk_matches_compiled_random_poses_and_velocities(env):
    from gobot.rl.supermimic.geometry import quat_to_mat
    generator = torch.Generator().manual_seed(43)
    kin=env.supermimic_mdp.kin
    for _ in range(4):
        pose=env.batch.read_state().base_pose.clone()
        quat=torch.randn(2,4,generator=generator);quat/=quat.norm(dim=-1,keepdim=True)
        pose[:,3:]=quat
        pose[:,:3]=torch.randn(2,3,generator=generator);pose[:,2]+=2
        q=kin.q_lo+(kin.q_hi-kin.q_lo)*torch.rand(2,29,generator=generator)
        qd=torch.randn(2,29,generator=generator)
        velocity=torch.randn(2,6,generator=generator)
        env.batch.reset([True,True],base_pose=pose,base_velocity=velocity,joint_position=q,joint_velocity=qd,targets=q)
        state=env.batch.read_state()
        expected=kin(state.base_pose[:,:3],quat_to_mat(state.base_pose[:,(6,3,4,5)]),
                     state.base_velocity[:,:3],state.base_velocity[:,3:],state.joint_position,state.joint_velocity)
        torch.testing.assert_close(expected[0],state.link_pose[:,:,:3],atol=1.e-5,rtol=0)
        actual_rotation=quat_to_mat(state.link_pose[:,:,(6,3,4,5)])
        torch.testing.assert_close(expected[1],actual_rotation,atol=1.e-5,rtol=0)
        delta=expected[1].transpose(-1,-2)@actual_rotation
        skew=torch.stack((delta[...,2,1]-delta[...,1,2],delta[...,0,2]-delta[...,2,0],delta[...,1,0]-delta[...,0,1]),-1)/2
        cosine=(delta.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
        assert torch.atan2(skew.norm(dim=-1),cosine).abs().max()<=1.e-5
        raw=env.batch._raw_cpu
        torch.testing.assert_close(expected[2],torch.as_tensor(raw['link_linear_velocity'])[:,env.batch._link_indices],atol=1.e-4,rtol=0)
        torch.testing.assert_close(expected[3],torch.as_tensor(raw['link_angular_velocity'])[:,env.batch._link_indices],atol=1.e-4,rtol=0)


def test_empty_reset_is_noop_and_holdout_is_not_training(env):
    previous=env.state
    before=env.generator.get_state().clone()
    env.reset(torch.zeros(env.num_envs,dtype=torch.bool))
    assert env.state is previous
    assert torch.equal(before,env.generator.get_state())
    with pytest.raises(ValueError,match='violates split'):
        make_motion_tracking_env(MotionTrackingCfg(project_path=PROJECT,motion_files=(str(MOTION),),split='holdout'))
