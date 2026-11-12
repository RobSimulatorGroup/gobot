"""Validate adapters against native observations, common scoring and shard identity."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.g1.reproduction.prepare_common_reference import prepare
from gobot.rl.supermimic.common_evaluation import (
    ANCHORS, PROTOCOL, CommonEnv, ReproducedAdapter, OfficialAdapter, configure_precision,
    evaluate, hash_value, merge, summarize,
)
from gobot.rl.supermimic.official.config import Config
from gobot.rl.supermimic.official.env import TrackingEnv
from gobot.rl.supermimic.official.motion import digest
from gobot.rl.supermimic.official.models import MultiEncoderAE
from gobot.rl.supermimic.official.tasks import EncoderSpec
from gobot.rl.supermimic.official.state import GenCoordRepresentation
from gobot.rl.supermimic.config import TrainConfig
from gobot.rl.supermimic.policy import Policy, PolicyCfg
from gobot.rl.motion_tracking.env import MotionTrackingEnv
from gobot.rl.training.workflow import config_dict


@pytest.fixture
def scenes(tmp_path):
    original=os.environ.get('GOBOT_TRACKING_TEST_PROJECT')
    official=os.environ.get('GOBOT_OFFICIAL_TEST_PROJECT')
    data=os.environ.get('GOBOT_TRACKING_TEST_DATA')
    if not all((original,official,data)):pytest.skip('requires prepared G1 scenes and dense FP32 motion')
    source=next(Path(data).glob('*.npz'))
    npz=tmp_path/'dense/npz';npz.mkdir(parents=True)
    with np.load(source) as src:
        # Two short but valid motions make actual queue/end/reset tests inexpensive.
        values={k:src[k] for k in src.files}
        for k,v in values.items():
            if v.ndim and k not in ('body_names','joint_names') and v.shape[0]>32:values[k]=v[:32]
    for name in ('clip_a','clip_b'):np.savez(npz/f'{name}.npz',**values)
    names=['clip_a','clip_b'];rows=[{'name':n,'frames':32,'keep':True,'npz_sha256':digest(npz/f'{n}.npz')} for n in names]
    root=npz.parent
    (root/'manifest_all.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for n in ('train_clips.json','eval_clips.json'):(root/n).write_text(json.dumps(names))
    (root/'manifest.json').write_text(json.dumps({'complete':True,'ready_for_paired_training':True,
        'artifacts':{n:digest(root/n) for n in ('manifest_all.jsonl','train_clips.json','eval_clips.json')}}))
    packed=tmp_path/'packed'
    prepare(root,Path(official)/'assets/supermimic/robot_profile.json',packed)
    cfg=Config(project=Path(official),library=packed,num_envs=2,backend='mujoco-cpu',device='cpu',
               sampling_mode='start',domain_randomization=False,reset_noise=False,episode_length_s=1e9,
               compile=False,capture_graphs=False,nconmax=128,njmax=500)
    native=TrainConfig(project=Path(original),backend='mujoco-cpu',device='cpu')
    native.data.npz_dir=npz;native.env.num_envs=2;native.env.replay_capacity=32
    native.env.domain_randomization=native.env.reset_noise=native.env.obs_noise=False
    native.env.action_noise_std=0.;native.env.sampling_mode='start';native.env.episode_length_s=1e9
    native.policy.encoder_hidden=native.policy.decoder_hidden=(32,16)
    return cfg,native


def reproduced_ck(native):
    policy=Policy(native.policy)
    return {'format':'gobot_supermimic_reproduced_v1','config':config_dict(native),'policy':policy.state_dict()}


def official_ck(cfg):
    cfg=replace(cfg,actor_hidden_dims=(32,16))
    actor=MultiEncoderAE(68,29,[EncoderSpec('motion',640),EncoderSpec('velocity',3)],
                         {'motion':1.,'velocity':0.},32,cfg.actor_hidden_dims,.1,1)
    return {'format':'gobot_supermimic_official_v1','config':config_dict(cfg),'actor':actor.state_dict()}


def test_reproduced_adapter_matches_native_history_goals_actions_and_torso_score(scenes):
    cfg,native_cfg=scenes
    original=MotionTrackingEnv(native_cfg);common=CommonEnv(cfg)
    configure_precision()
    try:
        ck=reproduced_ck(native_cfg);adapter=ReproducedAdapter(ck,common)
        obs,_=original.reset(frames=torch.tensor([0,32]));state=common.reset(torch.tensor([0,32]))
        for tick in range(14):
            inputs=adapter.observation(common,state,common.ids if tick==0 else common.ids[:0])
            # Temporal/term-major history and exact goal window stride/clamping.
            torch.testing.assert_close(inputs[0],obs['policy'],atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(inputs[1],obs['goal'],atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(adapter.actor(*inputs),adapter.actor(obs['policy'],obs['goal']),atol=2e-6,rtol=2e-5)
            action=torch.full((2,29),.01)
            obs,reward,term,trunc,info=original.step(action,record=False)
            state,_,ct,cu,ci=common.step(action)
            torch.testing.assert_close(ci['anchor_rewards']['torso_link'],reward,atol=2e-5,rtol=2e-5)
            assert torch.equal(ct,term) and not cu.any() and not trunc.any()
        # Native goal window clamps to the last frame without crossing clip_b.
        common.frame=torch.tensor([31,63]);original.frame=common.frame.clone()
        assert torch.equal(adapter.observation(common,state,common.ids[:0])[1],original.library.goal_window(original.frame,10))
    finally:original.close();common.close()


def test_official_adapter_and_shared_pelvis_scoring_match_native(scenes):
    cfg,_=scenes
    common=CommonEnv(cfg);native=TrackingEnv(cfg)
    try:
        ck=official_ck(cfg);adapter=OfficialAdapter(ck,common)
        state=common.reset(torch.tensor([0,32]));native_state=native.reset(torch.tensor([0,32]))
        for tick in range(8):
            obs=adapter.observation(common,state,common.ids[:0])
            representation=adapter.representation
            _,goal=representation.compute_wm_goal(native.motion,native.frame,native_state[:,:7])
            torch.testing.assert_close(obs['goal_motion'],goal,atol=1e-5,rtol=1e-5)
            torch.testing.assert_close(obs['history'][:,0],representation.represent(native_state),atol=1e-5,rtol=1e-5)
            action=torch.zeros(2,29)
            state,reward,term,trunc,_=common.step(action)
            native_state,nreward,nt,nu,_=native.step(action)
            torch.testing.assert_close(reward,nreward,atol=1e-6,rtol=1e-5)
            assert torch.equal(term,nt) and torch.equal(trunc,nu)
        # Both policies' shared clock stops on the last reference frame, N-1.
        common.reset(common.motion.motion_ends-2)
        _,_,_,trunc,info=common.step(torch.zeros(2,29))
        assert common.motion_end.all() and trunc.all()
        assert set(info['anchor_rewards'])==set(ANCHORS)
    finally:common.close();native.close()


@pytest.mark.parametrize('kind',['official','reproduced'])
def test_real_frozen_checkpoint_evaluation_and_capped_status(scenes,tmp_path,kind):
    cfg,native=scenes
    checkpoint=tmp_path/'checkpoint.pt'
    torch.save(official_ck(cfg) if kind=='official' else reproduced_ck(native),checkpoint)
    result=evaluate(checkpoint,cfg.project,cfg.library,tmp_path/'full.json',reproduced_project=native.project,
                    device='cpu',num_envs=2)
    assert result['complete'] and result['summary']['clips']==2
    assert all(row['expected_steps']==31 for row in result['clips'])
    assert all(row['failed'] or row['executed_steps']==31 for row in result['clips'])
    assert result['policy_adapter']['kind']==kind and result['common']['tf32'] is False
    assert 'optimizer' not in result
    capped=evaluate(checkpoint,cfg.project,cfg.library,tmp_path/'capped.json',device='cpu',num_envs=2,max_steps=1)
    assert not capped['complete'] and all(row['capped'] for row in capped['clips'])
    with pytest.raises(FileExistsError):evaluate(checkpoint,cfg.project,cfg.library,tmp_path/'full.json',device='cpu')


def reports(tmp_path):
    names=['a','b'];common={'reference':'FP32','clock':'N-1'};paths=[]
    for i,name in enumerate(names):
        rows=[{'name':name,'frames':4,'expected_steps':3,'executed_steps':3,'return':.8,
            'returns':{'pelvis':.8,'torso_link':.7},'coverage':1.,'failed':False,'finished':True,'completed':True,'capped':False}]
        r={'protocol':PROTOCOL,'checkpoint_sha256':'ck','policy_adapter':{'kind':'official'},'common':common,
           'common_sha256':hash_value(common),'selection_sha256':hash_value(names),'selection_count':2,
           'shard_index':i,'shard_count':2,'complete':True,'clips':rows,'summary':summarize(rows)}
        p=tmp_path/f'{i}.json';p.write_text(json.dumps(r));paths.append(p)
    return paths,names


def test_merge_preserves_both_rewards_and_rejects_foreign_protocol(tmp_path):
    paths,names=reports(tmp_path);result=merge(paths[::-1],names,tmp_path/'full.json')
    assert result['summary']['anchors']['torso_link']['mean_return']==.7
    r=json.loads(paths[0].read_text());r['protocol']='historical';paths[0].write_text(json.dumps(r))
    with pytest.raises(ValueError):merge(paths,names,tmp_path/'bad.json')


@pytest.mark.parametrize('damage',['common','checkpoint','missing','score','capped'])
def test_merge_rejects_incompatible_or_incomplete_results(tmp_path,damage):
    paths,names=reports(tmp_path)
    if damage=='missing':paths=paths[:1]
    else:
        r=json.loads(paths[0].read_text())
        if damage=='common':r['common']['clock']='N'
        if damage=='checkpoint':r['checkpoint_sha256']='other'
        if damage=='score':r['clips'][0]['returns']['torso_link']=float('nan')
        if damage=='capped':r['clips'][0]['capped']=True
        paths[0].write_text(json.dumps(r))
    with pytest.raises(ValueError):merge(paths,names,tmp_path/'bad.json')
    assert not (tmp_path/'bad.json').exists()


def test_pair_comparison_checks_protocol_precision_clock_and_roles(tmp_path):
    from examples.g1.reproduction.compare_common import compare
    paths,names=reports(tmp_path);official=merge(paths,names,tmp_path/'full.json')
    reproduced=deepcopy(official);reproduced['policy_adapter']={'kind':'reproduced'}
    result=compare(official,reproduced,{'a'})
    assert result['same_clip_names_frames_clock'] and result['scopes']['filtered']['official']['clips']==1
    for mutate in (lambda r:r['common'].update(reference='fp16'),
                   lambda r:r['clips'][0].update(expected_steps=4),
                   lambda r:r['policy_adapter'].update(kind='official'),
                   lambda r:r['clips'][0].update(capped=True)):
        broken=deepcopy(reproduced);mutate(broken)
        with pytest.raises(ValueError):compare(official,broken,{'a'})
