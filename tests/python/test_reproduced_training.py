"""Numerical regression against an independent untouched-reference process."""
import hashlib
import json
from pathlib import Path
import pytest
import torch
from examples.g1.reproduction.training_fixture import experiment
from gobot.rl.supermimic.schedule import cosine_lr
from gobot.rl.supermimic.config import TrainConfig

FIXTURE=Path(__file__).resolve().parents[1]/'fixtures/supermimic_reproduced_training_v1.pt'


def test_full_update_matches_reference():
    metadata=json.loads(FIXTURE.with_suffix('.json').read_text())
    assert not metadata['gobot_imported']
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest()==metadata['fixture_sha256']
    expected=torch.load(FIXTURE,weights_only=True)
    actual=experiment('gobot.rl.supermimic')
    def compare(a,b):
        if isinstance(a,torch.Tensor):
            torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-5)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for key in a: compare(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):compare(x,y)
        elif isinstance(a,float):
            assert a==pytest.approx(b,abs=2e-7,rel=2e-5)
        else: assert a==b
    compare(actual,expected)


def test_schedule_retains_320k_when_stopping_at_100k():
    cfg=TrainConfig.from_dict({'stop_iteration':100000,'world_model':{'state_spec':{}}})
    assert cfg.schedule.total_iterations==320000
    assert cfg.wm_trainer.lr==2e-4
    assert cosine_lr(250000,320000,2e-4)==pytest.approx(6e-6)
    assert cosine_lr(100000,320000,2e-4)>6e-6
    assert cfg.world_model.state_spec is not None


def test_episode_sampler_uses_return_to_go_and_ignores_truncation_tail():
    from types import SimpleNamespace
    from gobot.rl.supermimic.sampler import ReturnSampler
    lib=SimpleNamespace(device='cpu',num_bins=2,bin_of_frame=lambda frame:frame)
    sampler=ReturnSampler(lib,2,120)
    # A short true failure contributes; a short timeout contributes no near-tail data.
    for _ in range(2):sampler.log_step(torch.tensor([.1,.2]),torch.tensor([0,1]))
    sampler.finalize(torch.tensor([0,1]),torch.tensor([True,False]),torch.tensor([False,False]))
    assert sampler.bin_return[0]==pytest.approx(.95*10+.05*(.199+.1)/2)
    assert sampler.bin_return[1]==10


def test_short_motion_rejected_before_frame_sampling(tmp_path):
    from gobot.rl.supermimic.motion_library import MotionLibrary
    path=tmp_path/'short.pt'
    torch.save({'ref':torch.zeros(1,71),'names':['short'],'starts':[0],'lengths':[1]},path)
    with pytest.raises(ValueError,match='at least three'):
        MotionLibrary([path],('pelvis',),torch.zeros(29),device='cpu',compact=True)


@pytest.mark.parametrize('change,message', [
    (lambda c:setattr(c.schedule,'log_interval',0),'log_interval'),
    (lambda c:setattr(c,'stop_iteration',-1),'stop_iteration'),
    (lambda c:setattr(c.env,'action_noise_std',float('nan')),'action_noise_std'),
    (lambda c:setattr(c.policy_trainer,'critic_minibatches',10**9),'critic minibatches'),
])
def test_bad_training_recipe_fails_before_simulation(change,message):
    cfg=TrainConfig()
    change(cfg)
    with pytest.raises(ValueError,match=message):cfg.validate()
