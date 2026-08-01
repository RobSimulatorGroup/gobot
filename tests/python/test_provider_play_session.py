from __future__ import annotations

import gobot


class FakeProvider:
    def __init__(self) -> None:
        self.steps = 0
        self.closed = False

    def step(self, *, nsteps: int = 1) -> None:
        self.steps += nsteps

    def close(self) -> None:
        self.closed = True


class FakeContext:
    def __init__(self) -> None:
        self.token = 41
        self.callbacks = None
        self.ended = []

    def _begin_external_simulation(self, *callbacks):
        self.callbacks = callbacks
        return self.token

    def _reset_external_simulation(self, token: int) -> bool:
        assert token == self.token
        self.callbacks[1]()
        self.callbacks[2]()
        return True

    def _end_external_simulation(self, token: int) -> bool:
        self.ended.append(token)
        self.callbacks[3]()
        return True

    def _sync_external_simulation(self, token: int) -> bool:
        assert token == self.token
        self.callbacks[2]()
        return True


class FailingEndContext(FakeContext):
    def _end_external_simulation(self, token: int) -> bool:
        assert token == self.token
        raise RuntimeError("external end failed")


def assert_raises(expected: type[BaseException], callback) -> BaseException:
    try:
        callback()
    except expected as error:
        return error
    raise AssertionError(f"expected {expected.__name__}")


def test_parameter_validation() -> None:
    provider = FakeProvider()
    context = FakeContext()
    for fixed_dt in (float("nan"), float("inf"), float("-inf"), 0.0, -0.1):
        assert_raises(
            ValueError,
            lambda value=fixed_dt: gobot.sim.ProviderPlaySession(
                context, provider, fixed_dt=value
            ),
        )
    assert_raises(
        TypeError,
        lambda: gobot.sim.ProviderPlaySession(
            context, provider, fixed_dt=0.005, max_sub_steps=1.5
        ),
    )
    for max_sub_steps in (True, 0, -1):
        assert_raises(
            ValueError,
            lambda value=max_sub_steps: gobot.sim.ProviderPlaySession(
                context, provider, fixed_dt=0.005, max_sub_steps=value
            ),
        )


def test_close_failure_still_closes_provider() -> None:
    provider = FakeProvider()
    session = gobot.sim.ProviderPlaySession(
        FailingEndContext(), provider, fixed_dt=0.005
    ).start()

    error = assert_raises(RuntimeError, session.close)
    assert "external end failed" in str(error)
    assert provider.closed
    assert not session.running
    session.close()


def main() -> int:
    test_parameter_validation()
    test_close_failure_still_closes_provider()
    provider = FakeProvider()
    context = FakeContext()
    events: list[object] = []
    session = gobot.sim.ProviderPlaySession(
        context,
        provider,
        fixed_dt=0.005,
        max_sub_steps=4,
        before_step=lambda dt: events.append(("step", dt)),
        reset=lambda: events.append("reset"),
        sync_scene=lambda: events.append("sync"),
    ).start()

    assert session.running
    step, reset, sync, close, fixed_dt, max_sub_steps = context.callbacks
    assert fixed_dt == 0.005
    assert max_sub_steps == 4
    step(fixed_dt)
    assert provider.steps == 1
    assert events == [("step", 0.005)]
    session.reset()
    assert events[-2:] == ["reset", "sync"]
    session.sync_scene()
    assert events[-1] == "sync"

    session.close()
    session.close()
    assert context.ended == [context.token]
    assert provider.closed
    assert not session.running
    close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
