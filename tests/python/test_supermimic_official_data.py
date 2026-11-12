"""Data and scene boundary tests for official BONES-SEED integration."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import os

import numpy as np
import pytest
import torch

from examples.g1.official.prepare_bones import prepare
from gobot.rl.supermimic.official.motion import MotionLibrary, inspect_library, FIELDS, SLICES
from gobot.rl.supermimic.official.training import Ratio
from gobot.rl.supermimic.official.config import Config
from gobot.rl.supermimic.official.wm_learner import WorldModelLearnerCfg
from gobot.rl.supermimic.official.policy_learner import ActorCriticPolicyLearnerCfg


@pytest.fixture
def packed(tmp_path):
    torch.manual_seed(11)
    state=torch.randn(15,71).half();state[:,3:7]=torch.tensor([1.,0,0,0]).half()
    source=tmp_path/'source.pt'
    torch.save({'ref':state,'names':['a','b','c'],'starts':[0,4,10],'lengths':[4,6,5]},source)
    profile=tmp_path/'profile.json'
    profile.write_text(json.dumps({'joint_names':[f'joint{i}' for i in range(29)],'reference_commit':'test'}))
    return source,profile,state


def test_split_preserves_every_value_and_clip_order(packed,tmp_path):
    source,profile,state=packed
    output=tmp_path/'packed'
    manifest=prepare(source,profile,output,clips=['c','a'],chunk_frames=2)
    assert manifest['frames']==9 and manifest['clips']==2
    assert json.loads((output/'motion_names.json').read_text())==['a','c']
    assert np.array_equal(np.load(output/'motion_ends.npy'),[4,9])
    selected=torch.cat((state[:4],state[10:]))
    for name,(lo,hi) in SLICES.items():
        assert np.array_equal(np.load(output/f'{name}.npy'),selected[:,lo:hi].numpy())
    motion=MotionLibrary(output,lambda *args:args,manifest['joint_names'])
    torch.testing.assert_close(motion.full_state(torch.arange(9)),selected.float(),rtol=0,atol=0)
    inspect_library(output,verify_hashes=True)


def test_hash_check_rejects_corruption_even_when_size_is_unchanged(packed,tmp_path):
    source,profile,_=packed;output=tmp_path/'packed';prepare(source,profile,output)
    path=output/'joint_pos.npy'
    data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
    with pytest.raises(ValueError,match='hash changed'):inspect_library(output,verify_hashes=True)


def test_failed_conversion_is_not_published_and_source_is_preserved(packed,tmp_path):
    source,profile,state=packed
    state[5,0]=float('nan')
    torch.save({'ref':state,'names':['a','b','c'],'starts':[0,4,10],'lengths':[4,6,5]},source)
    output=tmp_path/'packed'
    with pytest.raises(ValueError,match='nonfinite'):prepare(source,profile,output)
    assert source.is_file() and not output.exists()
    assert not list(tmp_path.glob('packed.partial-*'))


def test_duplicate_unknown_and_existing_outputs_are_rejected(packed,tmp_path):
    source,profile,_=packed
    for clips in (['a','a'],['unknown'],[]):
        with pytest.raises(ValueError):prepare(source,profile,tmp_path/'rejected',clips=clips)
    output=tmp_path/'packed';prepare(source,profile,output)
    with pytest.raises(FileExistsError):prepare(source,profile,output)
    with pytest.raises(ValueError,match='joint order'):
        MotionLibrary(output,None,[str(i) for i in range(29)])


def test_ratio_matches_official_control_step_clock():
    ratio=Ratio(.4)
    assert [ratio(t) for t in range(11)]==[1,0,0,1,0,1,0,0,1,0,1]
    # With default 4096 envs, 10*2048*8 WM batch and replay ratio 16.
    cfg=Config();r=Ratio(cfg.num_envs/cfg.wm.num_batches/cfg.wm.batch_size/cfg.wm.horizon*cfg.replay_ratio)
    assert r.ratio==.4


def test_task_cli_resolves_official_recipe_without_creating_environment(tmp_path):
    from gobot.rl.train import main
    result=main(['g1-supermimic','--backend','mujoco-cpu','--num-envs','2',
                 '--project',str(tmp_path/'scene'),'--library',str(tmp_path/'data'),
                 '--max-seconds','60','--output',str(tmp_path/'run'),'--dry-run'])
    assert result['config']['max_seconds']==60 and result['config']['device']=='cpu'
    assert result['step_unit']=='control_steps' and not (tmp_path/'run').exists()


@pytest.fixture
def scene_config(tmp_path):
    project=os.environ.get('GOBOT_OFFICIAL_TEST_PROJECT')
    library=os.environ.get('GOBOT_OFFICIAL_TEST_LIBRARY')
    if not project or not library:pytest.skip('requires audited official G1 scene and packed motions')
    return Config(project=Path(project),library=Path(library),backend='mujoco-cpu',device='cpu',
                  num_envs=2,steps=12,output=tmp_path/'run',buffer_size=32,
                  dynamics_hidden_dims=(24,16,8),actor_hidden_dims=(24,16,8),critic_hidden_dims=(16,8),
                  sampling_mode='start',reset_noise=False,domain_randomization=True,compile=False,
                  capture_graphs=False,capture_updates=False,
                  wm=WorldModelLearnerCfg(num_batches=1,batch_size=4),
                  policy=ActorCriticPolicyLearnerCfg(batch_size=4,horizon=4))


def test_real_scene_startup_context_and_episode_boundary(scene_config):
    from gobot.rl.supermimic.official.env import TrackingEnv
    cfg=replace(scene_config,episode_length_s=.02)
    env=TrackingEnv(cfg)
    try:
        context=env.context.clone();env.reset()
        state,reward,term,timeout,info=env.step(torch.zeros(2,29))
        assert timeout.all() and env.frame.tolist()==[1,1]
        assert torch.equal(env.context,context)
        assert info['terminal_state'].shape==(2,71) and torch.isfinite(reward).all()
        env.reset();assert torch.equal(env.context,context)
        frames=env.motion.motion_ends[:1].expand(2)-2
        env.reset(frames)
        env.step(torch.zeros(2,29))
        assert env.motion_end.all() and env.frame.tolist()==[1,1]
    finally:env.close()


def test_real_training_checkpoint_resume_and_incompatible_data(scene_config):
    from gobot.rl.supermimic.official.training import run
    first=run(scene_config)
    ck=torch.load(first['checkpoint'],weights_only=True)
    assert first['learner_updates']>0 and first['critic_optimizer_steps']==20*first['learner_updates']
    cfg=replace(scene_config,steps=24,resume=Path(first['checkpoint']),output=scene_config.output.parent/'resumed')
    second=run(cfg)
    assert second['control_steps']==24 and second['learner_updates']>first['learner_updates']
    loaded=torch.load(second['checkpoint'],weights_only=True)
    assert torch.equal(ck['context'],loaded['context'])
    changed=replace(cfg,seed=cfg.seed+1,output=cfg.output.parent/'incompatible')
    with pytest.raises(ValueError,match='resume requires'):run(changed)


def test_time_budget_saves_and_capped_evaluation_is_not_complete(scene_config):
    from gobot.rl.supermimic.official.training import run
    from gobot.rl.supermimic.official.evaluate import evaluate
    result=run(replace(scene_config,max_seconds=1e-9))
    assert result['stop_reason']=='time_budget' and result['control_steps']==1
    report=evaluate(result['checkpoint'],scene_config.output/'evaluation.json',device='cpu',num_envs=2,max_steps=1)
    assert not report['complete'] and len(report['clips'])>=1
    assert any(row['capped'] for row in report['clips'])


def test_articulation_capacity_error_is_exposed_before_reset():
    from gobot.rl.articulation import ArticulationBatch
    from types import SimpleNamespace
    batch=object.__new__(ArticulationBatch)
    batch._closed=False
    def fail():raise RuntimeError('capacity overflow')
    batch._provider=SimpleNamespace(assert_no_overflow=fail)
    with pytest.raises(RuntimeError,match='capacity overflow'):batch.assert_no_overflow()
    batch._provider=None
    batch.assert_no_overflow()
