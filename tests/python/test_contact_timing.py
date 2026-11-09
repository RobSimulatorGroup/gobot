"""Contact timers follow physics ticks, independent of the simulator's float clock."""
import pytest
import torch

from gobot.rl.locomotion.warp_env import WarpVelocityEnv


def contact_env(dt):
    env = WarpVelocityEnv.__new__(WarpVelocityEnv)
    env.physics_dt = dt
    env.step_dt = dt * 4
    env._refresh_foot_contacts = lambda: None
    env._arrays = {'time': torch.tensor([0., 32., 1000., 100000.])}
    env._feet_contact = {'found': torch.ones(4, 2)}
    env._current_air_time = torch.full((4, 2), .1)
    env._last_air_time = torch.zeros(4, 2)
    env._current_contact_time = torch.zeros(4, 2)
    env._last_contact_time = torch.zeros(4, 2)
    env._contact_histories = {}
    env._collision_contacts = {}
    env._contact_history_cursor = 0
    return env


@pytest.mark.parametrize('dt', [.002, .005, .01])
def test_contact_window_is_one_control_step_at_any_clock_magnitude(dt):
    env = contact_env(dt)
    for _ in range(4):
        env._arrays['time'].add_(dt)
        env._update_contact_state_substep()
    torch.testing.assert_close(env._current_contact_time, torch.full((4, 2), 4 * dt))
    assert ((env._current_contact_time > 0) & (env._current_contact_time < env.step_dt + 1.e-6)).all()
    torch.testing.assert_close(env._last_air_time, torch.full((4, 2), .1 + dt))
    env._update_contact_state_substep()
    assert not (env._current_contact_time < env.step_dt + 1.e-6).any()


def test_detach_and_partial_reset_do_not_change_other_contact_timers():
    env = contact_env(.005)
    for _ in range(4):
        env._update_contact_state_substep()
    env._feet_contact['found'][:, 1] = 0
    env._update_contact_state_substep()
    torch.testing.assert_close(env._last_contact_time[:, 1], torch.full((4,), .025))
    torch.testing.assert_close(env._current_air_time[:, 1], torch.full((4,), .005))
    before = env._current_contact_time[1:].clone()
    env._reset_contact_state(torch.tensor([0]))
    assert env._current_contact_time[0].count_nonzero() == 0
    assert env._current_air_time[0].count_nonzero() == 0
    assert env._last_contact_time[0].count_nonzero() == 0
    assert env._last_air_time[0].count_nonzero() == 0
    torch.testing.assert_close(env._current_contact_time[1:], before)
    assert not hasattr(env, '_contact_last_time')
