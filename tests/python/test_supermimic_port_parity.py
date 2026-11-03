"""Fixed replay, gradients and one optimizer update match the pinned reference.

Only type annotations and logging are substituted to load the pure reference
algorithm without installing its simulator. No learning equations are replaced.
"""
from pathlib import Path
from types import ModuleType
import importlib
import sys
import pytest
import torch
from gobot.rl.supermimic import SuperMimicLearner
from gobot.rl.supermimic.contracts import EnvProtocol
from gobot.rl.supermimic import logging as gobot_logging
from test_supermimic_learner import FakeEnv, _config


def test_fixed_replay_losses_gradients_and_update_match_reference(tmp_path,monkeypatch):
    source=Path(__file__).resolve().parents[2]/'examples/locomotion/logs/supermimic_reference/Uni-SuperMimic/src'
    if not source.is_dir():pytest.skip('pinned reference checkout is not installed')
    packages={'unisupermimic':source/'unisupermimic',
              'unisupermimic.rl':source/'unisupermimic/rl',
              'unisupermimic.rl.algos':source/'unisupermimic/rl/algos',
              'unisupermimic.rl.contracts':source/'unisupermimic/rl/contracts'}
    before=set(sys.modules)
    for name,path in packages.items():
        module=ModuleType(name);module.__path__=[str(path)]
        monkeypatch.setitem(sys.modules,name,module)
    annotation=ModuleType('unilab.rl.contracts.env');annotation.EnvProtocol=EnvProtocol
    monkeypatch.setitem(sys.modules,'unilab.rl.contracts.env',annotation)
    monkeypatch.setitem(sys.modules,'unisupermimic.rl.logging',gobot_logging)
    try:
        upstream=importlib.import_module('unisupermimic.rl.algos.supermimic.learner')
        old=upstream.SuperMimicLearner(FakeEnv(),_config(),tmp_path/'reference')
        new=SuperMimicLearner(FakeEnv(),_config(),tmp_path/'gobot')
        for learner in (old,new):
            torch.manual_seed(71);learner.collect(4)
        for key in old.buffer.data:
            torch.testing.assert_close(old.buffer.data[key],new.buffer.data[key],rtol=0,atol=0)
        torch.manual_seed(72);wm_old=old.wm_update()
        torch.manual_seed(72);wm_new=new.wm_update()
        torch.manual_seed(73);pi_old=old.policy_update()
        torch.manual_seed(73);pi_new=new.policy_update()
        for actual,expected in zip((*wm_new,*pi_new),(*wm_old,*pi_old),strict=True):
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for name in ('wm','pi','critic'):
            a,b=getattr(new,name),getattr(old,name)
            for key,value in b.state_dict().items():
                torch.testing.assert_close(a.state_dict()[key],value,rtol=0,atol=0)
            for p,q in zip(a.parameters(),b.parameters(),strict=True):
                assert (p.grad is None)==(q.grad is None)
                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    finally:
        for name in set(sys.modules)-before:
            if name.startswith('unisupermimic.'):
                sys.modules.pop(name,None)
