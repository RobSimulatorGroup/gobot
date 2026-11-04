"""Real scene integration checks; opt in with the prepared G1 project and NPZ data."""
import os
from pathlib import Path
import pytest
import torch
from gobot.rl.supermimic.config import TrainConfig
from gobot.rl.motion_tracking.env import MotionTrackingEnv


@pytest.fixture
def config():
    project=os.environ.get('GOBOT_TRACKING_TEST_PROJECT')
    data=os.environ.get('GOBOT_TRACKING_TEST_DATA')
    if not project or not data:
        pytest.skip('requires prepared reference G1 scene and converted motion data')
    cfg=TrainConfig()
    cfg.project=Path(project);cfg.data.npz_dir=Path(data)
    cfg.env.num_envs=2;cfg.env.replay_capacity=32;cfg.env.sampling_mode='start'
    cfg.env.obs_noise=cfg.env.reset_noise=False
    cfg.device='cpu';cfg.backend='mujoco-cpu'
    return cfg


def test_context_reset_and_clean_executed_action(config):
    torch.manual_seed(179)
    env=MotionTrackingEnv(config)
    try:
        obs,_=env.reset()
        initial_xi=env.xi.clone()
        assert obs['policy'].shape==(2,930) and obs['goal'].shape==(2,680)
        assert env.observation_spec.dim==930 and env.goal_observation_spec.dim==680
        assert env.action_spec.dim==29
        assert torch.equal(env.frame,env.library.clip_start[:1].expand(2)+1)
        physics=env.batch.physics_parameter_state()
        assert set(physics['shape_friction'])==set(env.profile['foot_shapes'])
        action=torch.full((2,29),.1)
        env.step(action)
        torch.testing.assert_close(env.buffer.a[:,0],action,atol=0,rtol=0)
        assert not torch.equal(env.buffer.a[:,0],env.buffer.a_exec[:,0])
        assert not env.buffer.terminal[:,0].any()
        env.reset()
        torch.testing.assert_close(env.xi[:,30:],initial_xi[:,30:],atol=0,rtol=0)
        assert not torch.equal(env.xi[:,:30],initial_xi[:,:30])
    finally: env.close()


def test_compact_reset_and_timeout_terminal_rows(config):
    config.env.compact_library=True
    config.env.domain_randomization=False
    config.env.action_noise_std=0.
    config.env.episode_length_s=.02
    env=MotionTrackingEnv(config)
    try:
        env.reset()
        raw=env.library.ref(env.library.clip_start[:1].expand(2)).root_quat_w
        torch.testing.assert_close(env.learning_state()[:,3:7],raw,atol=0,rtol=0)
        torch.testing.assert_close(env.physical_state()[:,3:7].norm(dim=-1),torch.ones(2),atol=1e-6,rtol=0)
        _,_,term,timeout,info=env.step(torch.zeros(2,29))
        assert timeout.all() and not term.any()
        assert env.buffer.ptr.tolist()==[2,2]
        assert env.buffer.terminal[:,1].all()
        assert not env.buffer.done[:,:2].any()  # timeout is not a learned failure
        torch.testing.assert_close(env.buffer.s[:,1],info['terminal_state'])
        assert env.ep_start.all()
    finally: env.close()


def test_packed_library_matches_the_converted_npz(config):
    from gobot.rl.supermimic.motion_library import MotionLibrary
    packed=Path(config.data.npz_dir)/'library.pt'
    if not packed.exists():
        pytest.skip('conversion was not requested with --compact')
    config.env.compact_library=True
    config.env.domain_randomization=False
    env=MotionTrackingEnv(config)
    try:
        other=MotionLibrary([packed],env.body_names,env.default_q,device='cpu',compact=True,fk=env.fk)
        assert [c.name for c in other.clips]==[c.name for c in env.library.clips]
        frame=torch.arange(other.num_frames)
        torch.testing.assert_close(other.ref71(frame),env.library.ref71(frame),atol=0,rtol=0)
        torch.testing.assert_close(other.goal_window(frame[:10],10),env.library.goal_window(frame[:10],10),atol=0,rtol=0)
    finally:env.close()


def test_training_resume_restores_startup_context_and_counters(config,tmp_path):
    from gobot.rl.motion_tracking.training import Trainer
    config.env.domain_randomization=True
    config.log_root=tmp_path;config.run_name='first';config.stop_iteration=12
    config.schedule.warmup_iterations=10;config.schedule.log_interval=20
    config.world_model.hidden=(24,16);config.world_model.term_hidden=(8,8)
    config.policy.encoder_hidden=config.policy.decoder_hidden=(24,16)
    config.critic.hidden=(16,8)
    config.wm_trainer.batch_size=4;config.wm_trainer.batches_per_update=1;config.wm_trainer.compile=None
    config.policy_trainer.batch_size=4;config.policy_trainer.horizon=4
    config.policy_trainer.critic_epochs=1;config.policy_trainer.critic_minibatches=2;config.policy_trainer.compile=None
    env=MotionTrackingEnv(config);trainer=Trainer(env,config)
    try:
        trainer.run()
        saved_com=env.xi[:,30:33].clone()
        saved_counters=trainer.counters()
    finally:trainer.close();env.close()
    config.resume=tmp_path/'first/ckpt_latest.pt';config.run_name='resume';config.stop_iteration=24
    # Deliberately advance the generator before rebuilding. Startup context must
    # come from the checkpoint, independently of construction RNG consumption.
    torch.rand(100)
    env=MotionTrackingEnv(config);trainer=Trainer(env,config)
    try:
        torch.testing.assert_close(env.xi[:,30:33],saved_com,atol=0,rtol=0)
        assert trainer.counters()==saved_counters
        trainer.run()
        assert trainer.counters()['learner_updates']==2
        assert trainer.counters()['control_steps_per_rank']==24
    finally:trainer.close();env.close()
