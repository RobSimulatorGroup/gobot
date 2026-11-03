"""Real task/optimizer restoration and frozen-actor physics-transfer contracts."""
from dataclasses import replace
from pathlib import Path
import torch
import pytest

from gobot.rl.motion_tracking import MotionTrackingCfg, MotionRandomizationCfg, make_motion_tracking_env
from gobot.rl.motion_tracking.training import TrackingTrainingSession
from gobot.rl.supermimic import SuperMimicConfig

PROJECT=Path(__file__).resolve().parents[2]/'examples/g1'
MOTION=PROJECT/'assets/motions/tracking/walk1_subject1.csv'


@pytest.fixture
def env():
    if not MOTION.exists():pytest.skip('prepared tracking assets required')
    torch.set_num_threads(2)
    cfg=MotionTrackingCfg(project_path=PROJECT,motion_files=(str(MOTION),),eval_mode=False,
        reset_noise=True,randomization=MotionRandomizationCfg(enabled=True))
    value=make_motion_tracking_env(cfg,num_envs=2,seed=42)
    try:yield value
    finally:value.close()


def config(iterations=1):
    return SuperMimicConfig(iterations=iterations,buffer_steps=32,wm_batch=4,pi_batch=4,
        hist=2,horizon=2,goal_frames=10,wm_window=2,wm_batches=1,warmup_steps=4,wm_only_iters=0,
        critic_epochs=1,critic_minibatches=1,test_hidden_dims=(8,),amp='none',compile=False,
        log_backend='no_print',log_every=1,ckpt_every=1,seed=42)


def test_full_checkpoint_resumes_with_new_total_iteration_target(env,tmp_path):
    first=TrackingTrainingSession(env,config(),tmp_path/'first',software_digest='test_source')
    assert first.train()['completed']
    saved=torch.load(tmp_path/'first/last.pt',weights_only=True)
    second=TrackingTrainingSession(env,config(2),tmp_path/'second',software_digest='test_source')
    second.load(tmp_path/'first/last.pt')
    assert second.iteration==1 and second.buffer.size==0
    assert torch.equal(second.env.ep_step,torch.zeros(2,dtype=torch.long))
    assert second.opt_pi.state_dict()['state']
    assert second.train()['iteration']==2
    restored=torch.load(tmp_path/'second/last.pt',weights_only=True)
    assert restored['gobot_training']['contract_digest']==saved['gobot_training']['contract_digest']
    assert (tmp_path/'second/last.pt.manifest.json').exists()


def test_fork_keeps_actor_and_normalization_frozen_and_refills_replay(env,tmp_path):
    original=TrackingTrainingSession(env,config(),tmp_path/'original',software_digest='source')
    original.train();checkpoint=tmp_path/'original/last.pt'
    before={key:value.clone() for key,value in original.pi.state_dict().items()}
    forked=TrackingTrainingSession(env,config(2),tmp_path/'fork',software_digest='new_source')
    forked.fork(checkpoint)
    assert forked.actor_frozen and forked.phase=='adaptation'
    assert not forked.opt_pi.state and not forked.opt_wm.state and forked.buffer.size==0
    forked.train()
    for name,value in forked.pi.state_dict().items():
        torch.testing.assert_close(value,before[name],atol=0,rtol=0)
    assert forked.opt_v.state
    assert forked.lineage[-1]['data_split']=='coverage40'
    with pytest.raises(ValueError,match='resumed adaptation'):
        forked.promote({'passed':True})


def test_resume_rejects_changed_semantics_and_controlled_stop_saves(env,tmp_path):
    session=TrackingTrainingSession(env,config(),tmp_path/'source',software_digest='source')
    session.save()
    changed=TrackingTrainingSession(env,replace(config(),gamma=.95),tmp_path/'changed',software_digest='source')
    with pytest.raises(ValueError,match='changed'):
        changed.load(tmp_path/'source/last.pt')
    stop=TrackingTrainingSession(env,config(),tmp_path/'stopped',software_digest='source',should_stop=lambda:True)
    result=stop.train()
    assert result['stop_reason']=='controlled_stop' and not result['completed']
    saved=torch.load(tmp_path/'stopped/last.pt',weights_only=True)
    assert saved['iteration']==0 and saved['gobot_training']['task_state']['resume_episodes']=='fresh'


def test_adaptation_measurements_do_not_authorize_unfreezing_without_thresholds(env,tmp_path):
    from gobot.rl.motion_tracking.validation import validate_adaptation
    original=TrackingTrainingSession(env,config(),tmp_path/'source',software_digest='source')
    baseline=original.save()
    candidate=TrackingTrainingSession(env,config(),tmp_path/'candidate',software_digest='source')
    candidate.fork(baseline);checkpoint=candidate.save()
    report=validate_adaptation(env,checkpoint,baseline,steps=32)
    assert report['metrics_only'] and not report['passed']
    assert set(report['checks']['world_model_1_8_32']['normalized_rmse']['candidate'])=={'1','8','32'}
    assert report['checks']['physical_rollout']['actor_and_normalizers_unchanged']
    resumed=TrackingTrainingSession(env,config(),tmp_path/'resume',software_digest='source')
    resumed.load(checkpoint)
    with pytest.raises(ValueError,match='has not passed'):
        resumed.promote(report)


def test_promotion_requires_matching_evidence_and_survives_resume(env,tmp_path):
    from copy import deepcopy
    from gobot.rl.experiment_identity import file_digest,canonical_digest
    source=TrackingTrainingSession(env,config(),tmp_path/'source',software_digest='source')
    baseline=source.save()
    forked=TrackingTrainingSession(env,config(),tmp_path/'fork',software_digest='source')
    forked.fork(baseline);checkpoint=forked.save()
    resumed=TrackingTrainingSession(env,config(),tmp_path/'resume',software_digest='source')
    resumed.load(checkpoint)
    thresholds={'world_model_rmse':{'1':.1,'8':.5,'32':1.},'critic_relative_rmse':.05,
                'minimum_survival_fraction':.9,'minimum_step_return':.8}
    report={'protocol':'tracking_adaptation_validation_v1','passed':True,'metrics_only':False,
        'thresholds':thresholds,'thresholds_digest':canonical_digest(thresholds),
        'checkpoint_sha256':file_digest(checkpoint),'contract_digest':resumed.contract_digest,
        'checks':{key:{'passed':True} for key in ('world_model_1_8_32','critic','physical_rollout')}}
    wrong=deepcopy(report);wrong['checkpoint_sha256']='wrong'
    with pytest.raises(ValueError,match='different checkpoint'):resumed.promote(wrong)
    assert resumed.actor_frozen
    resumed.promote(report)
    assert resumed.phase=='joint' and not resumed.actor_frozen
    promoted=resumed.save()
    again=TrackingTrainingSession(env,config(),tmp_path/'again',software_digest='source')
    again.load(promoted)
    assert again.phase=='joint' and not again.actor_frozen and len(again.validation_history)==1


def test_legacy_import_binds_manifest_to_exact_weights(env,tmp_path):
    from gobot.rl.experiment_identity import file_digest
    source=TrackingTrainingSession(env,config(),tmp_path/'source',software_digest='source')
    payload=source._checkpoint(False)
    legacy=tmp_path/'legacy.pt'
    torch.save({key:payload[key] for key in ('pi','wm','args')},legacy)
    target=TrackingTrainingSession(env,config(),tmp_path/'target',software_digest='source')
    with pytest.raises(ValueError,match='matching checkpoint hash'):
        target.fork(legacy,source_manifest=source.manifest)
    target.fork(legacy,source_manifest=source.manifest,source_identity={'checkpoint_sha256':file_digest(legacy)})
    assert target.actor_frozen and target.phase=='adaptation'
