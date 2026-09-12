from __future__ import annotations

import threading

import pytest

from gobot.sim import SimulationSession, SimulationStepError
from gobot._core import SimulationMicrostepResult


class Provider:
    num_envs = 2
    fixed_time_step = .004
    session_substeps = 2

    def __init__(self):
        self.calls = 0
        self.fail_at = 0
        self.close_calls = 0
        self.commands = []
        self.resets = []

    def advance_microstep(self, dt, actions):
        self.calls += 1
        self.commands.append(actions)
        values = []
        for _ in range(self.num_envs):
            value = SimulationMicrostepResult()
            if self.calls == self.fail_at:
                value.error = "injected Newton failure; microstep rolled back"
                value.failure_stage = "contact"
            else:
                value.completed = True
                value.advanced_time = dt
            values.append(value)
        return values

    def reset(self, mask):
        self.resets.append(mask)

    def close(self):
        self.close_calls += 1


def test_partial_progress_and_exception_share_the_cpp_clock():
    provider = Provider()
    provider.fail_at = 4
    with SimulationSession(provider) as session:
        with pytest.raises(SimulationStepError) as error:
            session.step("command", nsteps=3)
        result = error.value.result
        assert result is session.last_result
        assert result.environments[0].completed_microsteps == 3
        assert result.environments[0].completed_ticks == 1
        assert result.environments[0].advanced_time == pytest.approx(.006)
        assert session.clocks[0].time == pytest.approx(.006)
        assert provider.commands == ["command", None, None, None]
        with pytest.raises(RuntimeError, match="Reset faulted"):
            session.step()
        session.reset()
        assert session.step().completed
        assert session.clocks[0].time == pytest.approx(.004)
    assert provider.close_calls == 1


def test_reset_selection_noop_and_validation():
    provider = Provider()
    with SimulationSession(provider) as session:
        session.step(nsteps=2)
        session.reset([])
        assert provider.resets == []
        session.reset([1])
        assert provider.resets == [[False, True]]
        assert session.clocks[0].tick == 2
        assert session.clocks[1].tick == 0
        for invalid in ([0, 0], [2], [-1], [True]):
            with pytest.raises((ValueError, TypeError)):
                session.reset(invalid)
        assert len(provider.resets) == 1
        for invalid in (0, -1, True, 1.5):
            with pytest.raises((ValueError, TypeError, OverflowError)):
                session.step(nsteps=invalid)


def test_session_rejects_cross_thread_access():
    provider = Provider()
    errors = []
    with SimulationSession(provider) as session:
        def run():
            try:
                session.step()
            except RuntimeError as error:
                errors.append(str(error))
        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert errors == ["SimulationSession must be used on its owner thread"]
        assert provider.calls == 0


@pytest.mark.parametrize("substeps", [0, -1, True, 1.5])
def test_substeps_reject_invalid_configuration(substeps):
    provider = Provider()
    provider.session_substeps = substeps
    with pytest.raises((ValueError, TypeError)):
        SimulationSession(provider)
    assert provider.calls == 0
