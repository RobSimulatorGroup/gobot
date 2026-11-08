"""Turning reward semantics and the matching independent mjlab experiment."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from gobot.rl.locomotion.config import VelocityRewardCfg
from gobot.rl.locomotion.terms import angular_tracking_error
from gobot.rl.training.locomotion import training_spec
from gobot.rl.training.workflow import update_config

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('turning_reference_support',ROOT/'tests/support/mjlab_training.py')
support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)


def test_yaw_reward_does_not_exchange_turning_for_reduced_body_wobble():
    command=np.array([[.8,0.,0.],[.8,0.,0.]],np.float32)
    angular=np.array([[.4,.3,0.],[0.,0.,.1]],np.float32)
    # A straight but wobbling gait loses to smooth circling in the old term.
    full=angular_tracking_error(command,angular,'xyz')
    yaw=angular_tracking_error(command,angular,'z')
    assert full[0]>full[1]
    assert yaw[0]<yaw[1]
    reflected=angular.copy();reflected[:,2]*=-1
    np.testing.assert_array_equal(angular_tracking_error(command,reflected,'z'),yaw)


def test_turn_commands_have_symmetric_rewards_and_correct_sign():
    command=torch.tensor([[.8,0.,.5],[.8,0.,-.5],[.8,0.,.5],[.8,0.,-.5]])
    angular=torch.tensor([[.3,.4,.5],[-.3,.4,-.5],[.3,.4,-.5],[-.3,.4,.5]])
    torch.testing.assert_close(angular_tracking_error(command,angular,'z'),torch.tensor([0.,0.,1.,1.]))


@pytest.mark.parametrize('axes',['xyz','z'])
def test_mjlab_adapter_matches_numpy_and_torch_without_mutating_inputs(axes):
    generator=torch.Generator().manual_seed(19)
    command=torch.randn(128,3,generator=generator)
    angular=torch.randn(128,3,generator=generator)
    before=(command.clone(),angular.clone())
    env=NS(command_manager=NS(get_command=lambda name:command),
           scene={'robot':NS(data=NS(root_link_ang_vel_b=angular))})
    actual=support.angular_velocity_reward(env,.5,'twist',axes)
    expected=np.exp(-angular_tracking_error(command.numpy(),angular.numpy(),axes)/.25)
    np.testing.assert_allclose(actual.numpy(),expected,rtol=2e-6,atol=1e-7)
    torch.testing.assert_close(command,before[0],rtol=0,atol=0)
    torch.testing.assert_close(angular,before[1],rtol=0,atol=0)


def test_experiment_matches_reference_options_and_keeps_other_settings():
    workflow=training_spec(robot='g1')
    base=workflow.config_factory()
    variant=update_config(base,json.loads((ROOT/'examples/g1/turning.json').read_text()))
    variant.validate()
    a,b=base.resolve_env(),variant.resolve_env()
    assert b.rewards.ang_vel_axes=='z' and b.rewards.ang_vel_std==.5
    assert a.rewards.track_angular_velocity==b.rewards.track_angular_velocity==2.
    assert a.command==b.command
    assert base.resolve_agent()==variant.resolve_agent()
    marker=lambda:None
    term=NS(func=marker,params={'std':a.rewards.ang_vel_std,'command_name':'twist'},weight=2.)
    cfg=NS(rewards={'track_angular_velocity':term})
    support.configure_angular_tracking(cfg,'xyz',a.rewards.ang_vel_std)
    assert term.func is marker  # Default still uses the original reference reward.
    settings=support.configure_angular_tracking(cfg,'z',.5)
    assert settings=={'axes':b.rewards.ang_vel_axes,'std':b.rewards.ang_vel_std,'weight':2.}
    assert term.func is support.angular_velocity_reward


@pytest.mark.parametrize('kwargs',[{'ang_vel_axes':'x'},{'ang_vel_std':0},{'ang_vel_std':float('nan')},{'ang_vel_std':-1}])
def test_invalid_angular_tracking_is_rejected(kwargs):
    with pytest.raises(ValueError):VelocityRewardCfg(**kwargs)


def test_fixed_turn_commands_are_passed_without_rewriting(monkeypatch, capsys):
    from contextlib import nullcontext
    from examples.locomotion import assessment
    calls=[]
    monkeypatch.setattr(assessment,'evaluation_budget',lambda *a,**k:nullcontext())
    monkeypatch.setattr(assessment,'evaluate',lambda *a,**kw:(calls.append(kw) or {'scenarios':[]}))
    monkeypatch.setattr('sys.argv',['assessment','--checkpoint','unused.pt','--output','unused.json',
        '--commands','.8','0','.5','--commands','.8','0','-.5','--commands','.8','0','0'])
    assessment.main()
    assert calls[0]['commands']==[[.8,0.,.5],[.8,0.,-.5],[.8,0.,0.]]
    capsys.readouterr()
