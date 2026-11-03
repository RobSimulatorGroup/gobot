import torch
from gobot.rl.motion_tracking.validation import real_return_targets


def test_zero_critic_does_not_look_calibrated_from_tiny_one_step_rewards():
    reward=torch.full((32,2),.1)
    flags=torch.zeros((32,2),dtype=torch.bool)
    target=real_return_targets(reward,flags,flags,torch.zeros_like(reward),gamma=.99)
    expected=.1*(1-.99**32)/(1-.99)
    torch.testing.assert_close(target,torch.full((1,2),expected))
    assert float(target.mean()/10)>.25


def test_timeouts_bootstrap_final_state_and_failures_do_not_cross_reset():
    reward=torch.tensor([[1.,1.],[1.,1.],[100.,100.]])
    done=torch.tensor([[False,False],[True,True],[False,False]])
    term=torch.tensor([[False,False],[True,False],[False,False]])
    next_values=torch.tensor([[0.,0.],[10.,10.],[1000.,1000.]])
    target=real_return_targets(reward,term,done,next_values,gamma=.9,horizon=3)
    torch.testing.assert_close(target,torch.tensor([[1.9,10.]]))
