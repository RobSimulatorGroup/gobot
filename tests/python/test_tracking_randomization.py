"""Randomization changes the physical world, survives reset, and resumes explicitly."""
from copy import deepcopy
from pathlib import Path
import pytest
import torch

from gobot.rl.motion_tracking import MotionTrackingCfg, MotionRandomizationCfg, make_motion_tracking_env
from test_articulation_batch import make_batch

PROJECT=Path(__file__).resolve().parents[2]/'examples/g1'
MOTION=PROJECT/'assets/motions/tracking/walk1_subject1.csv'


def test_named_com_changes_dynamics_without_accumulation(tmp_path):
    batch=make_batch(tmp_path)
    try:
        initial={k:v.clone() for k,v in batch.read_state().__dict__.items()}
        offsets=torch.tensor([[.12,0,0],[-.12,0,0]])
        robot=batch.snapshot['robots'][0]
        shape=next(s['name'] for link in robot['links'] if link['name']=='foot' for s in link['shapes'])
        mu=torch.tensor([[.4,.005,.0001],[1.1,.005,.0001]])
        trajectories=[]
        for _ in range(2):
            batch.configure_physics(link_center_of_mass_offsets={'foot':offsets},shape_friction={shape:mu})
            batch.reset([True,True],base_pose=initial['base_pose'],base_velocity=initial['base_velocity'],
                        joint_position=initial['joint_position'],joint_velocity=initial['joint_velocity'],
                        targets=initial['joint_position'])
            batch.set_joint_targets(torch.full((2,1),.5))
            values=[]
            for _ in range(6):values.append(batch.step_control().joint_position.clone())
            trajectories.append(torch.stack(values))
        torch.testing.assert_close(trajectories[0],trajectories[1],atol=1.e-7,rtol=1.e-6)
        assert (trajectories[0][:,0]-trajectories[0][:,1]).abs().max()>1.e-4
        saved=batch.physics_parameter_state()
        torch.testing.assert_close(saved['link_center_of_mass_offsets']['foot'],offsets)
        saved['link_center_of_mass_offsets']['foot'].fill_(100)
        torch.testing.assert_close(batch.physics_parameter_state()['link_center_of_mass_offsets']['foot'],offsets)
        with pytest.raises(ValueError,match='unknown'):
            batch.configure_physics(link_center_of_mass_offsets={'missing':offsets})
        with pytest.raises(ValueError,match='finite'):
            batch.configure_physics(link_center_of_mass_offsets={'foot':torch.full((2,3),float('nan'))})
    finally:batch.close()


@pytest.fixture
def randomized_env():
    if not MOTION.exists():pytest.skip('prepared G1 tracking assets required')
    torch.set_num_threads(2)
    cfg=MotionTrackingCfg(project_path=PROJECT,motion_files=(str(MOTION),),eval_mode=False,
                          randomization=MotionRandomizationCfg(enabled=True),reset_noise=True)
    env=make_motion_tracking_env(cfg,num_envs=2,seed=43)
    try:yield env
    finally:env.close()


def test_context_matches_applied_parameters_and_masked_reset(randomized_env):
    env=randomized_env
    before=env.xi.clone()
    physical=env.batch.physics_parameter_state()
    torch.testing.assert_close(physical['link_center_of_mass_offsets']['torso_link'],env.xi[:,30:33])
    assert len(physical['shape_friction'])==36
    for coefficients in physical['shape_friction'].values():
        torch.testing.assert_close(coefficients[:,0],env.xi[:,0])
    env.reset_to_frames(torch.tensor([200]),[0])
    torch.testing.assert_close(env.xi[:,0],before[:,0],rtol=0,atol=0)
    torch.testing.assert_close(env.xi[:,30:33],before[:,30:33],rtol=0,atol=0)
    torch.testing.assert_close(env.xi[1],before[1],rtol=0,atol=0)
    assert not torch.equal(env.xi[0,1:30],before[0,1:30])
    result=env.step(torch.zeros(2,29))
    torch.testing.assert_close(result.info['supermimic']['xi'],env.xi)
    assert torch.isfinite(result.reward).all()


def test_resume_restores_sampling_context_and_starts_fresh_episodes(randomized_env):
    env=randomized_env
    env.step(torch.zeros(2,29))
    saved=env.training_state_dict()
    env.load_training_state_dict(saved)
    first={k:v.clone() for k,v in env.state.info['supermimic_state'].items()}
    env.step(torch.ones(2,29))
    env.load_training_state_dict(saved)
    for key,value in env.state.info['supermimic_state'].items():
        torch.testing.assert_close(value,first[key],rtol=0,atol=0)
    assert torch.equal(env.ep_step,torch.zeros_like(env.ep_step))
    invalid=deepcopy(saved);invalid['context'][0,0]=-1
    with pytest.raises(ValueError,match='outside configured'):
        env.load_training_state_dict(invalid)
    wrong=deepcopy(saved);wrong['num_envs']=3
    with pytest.raises(ValueError,match='same batch'):
        env.load_training_state_dict(wrong)


def test_eval_rejects_training_randomization():
    with pytest.raises(ValueError,match='disable physics randomization'):
        MotionTrackingCfg(project_path='.',motion_files=('clip.csv',),randomization=MotionRandomizationCfg(enabled=True))
