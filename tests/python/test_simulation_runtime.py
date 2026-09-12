from __future__ import annotations

import gc
import threading
import time

import numpy as np
import pytest

from gobot.sim import AsyncSimulationSession, SimulationRuntimeSpec


class Probe:
    def __init__(self):
        self.events = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.resets = []
        self.calls = 0


_probe = None


class Provider:
    num_envs = 2
    fixed_time_step = .002

    def __init__(self, fail_at=0):
        self.probe = _probe
        self.fail_at = fail_at
        self.probe.events.append(("build", threading.get_ident()))
        self.arrays = {"positions": np.zeros((2, 3))}

    def step(self, actions=None, *, nsteps=1):
        self.probe.events.append(("step", threading.get_ident()))
        self.probe.entered.set()
        assert self.probe.release.wait(timeout=5)
        self.probe.calls += 1
        if self.probe.calls == self.fail_at:
            raise RuntimeError("injected solver failure")
        self.arrays["positions"] += 1

    def reset(self, mask):
        self.probe.events.append(("reset", threading.get_ident()))
        self.probe.resets.append(mask)
        self.arrays["positions"][mask] = 0

    def close(self):
        self.probe.events.append(("close", threading.get_ident()))

    def __del__(self):
        self.probe.events.append(("destroy", threading.get_ident()))


def make_provider(fail_at=0):
    return Provider(fail_at)


def fail_factory():
    raise RuntimeError("injected build failure")


def binary_factory(payload):
    assert payload == {"blob": b"\x00\xffmesh", "tag": {"kind": "bytes", "value": "authored"}}
    return Provider()


@pytest.fixture
def probe():
    global _probe
    _probe = Probe()
    yield _probe
    _probe.release.set()


def await_result(session):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = session.poll()
        if result is not None:
            return result
        time.sleep(.001)
    raise AssertionError("simulation completion timeout")


def spec(**arguments):
    return SimulationRuntimeSpec(__name__ + ":make_provider", arguments)


def test_provider_lifecycle_uses_one_worker_and_held_views_survive_close(probe):
    session = AsyncSimulationSession(spec(), fields=("positions",), max_hz=1e9)
    try:
        installed = await_result(session)
        assert installed.error == ""
        assert session.info.provider_name == "Provider"
        assert session.info.environment_count == 2
        assert session.info.fixed_time_step == pytest.approx(.002)
        snapshot = installed.snapshot
        original = np.asarray(snapshot.buffer("positions"))
        assert original.shape == (1, 3)
        assert not original.flags.writeable
        assert session.submit()
        stepped = await_result(session)
        assert stepped.step.completed
        assert stepped.clocks[0].time == pytest.approx(.002)
        assert np.asarray(stepped.snapshot.buffer("positions")).tolist() == [[1, 1, 1]]
        assert original.tolist() == [[0, 0, 0]]
        assert session.reset([1])
        reset = await_result(session)
        assert reset.clocks[0].tick == 1
        assert reset.clocks[1].tick == 0
        assert probe.resets == [[False, True]]
    finally:
        session.shutdown()
    assert original.tolist() == [[0, 0, 0]]
    assert {name for name, _ in probe.events} == {"build", "step", "reset", "close", "destroy"}
    threads = {thread for _, thread in probe.events}
    assert len(threads) == 1
    assert threading.get_ident() not in threads


def test_reset_during_solve_discards_old_epoch_and_stop_does_not_join(probe):
    session = AsyncSimulationSession(spec(), fields=("positions",))
    try:
        await_result(session)
        probe.release.clear()
        assert session.submit()
        assert probe.entered.wait(timeout=2)
        assert not session.submit()
        assert session.reset()
        assert session.epoch == 2
        probe.release.set()
        result = await_result(session)
        assert result.operation == "reset"
        assert result.epoch == 2
        assert result.clocks[0].time == 0
        probe.entered.clear()
        probe.release.clear()
        assert session.submit()
        assert probe.entered.wait(timeout=2)
        start = time.monotonic()
        session.close()
        assert time.monotonic() - start < .1
        assert session.poll() is None
    finally:
        probe.release.set()
        session.shutdown()


def test_partial_failure_preserves_progress_and_requires_reset(probe):
    session = AsyncSimulationSession(spec(fail_at=3), fields=("positions",))
    try:
        await_result(session)
        assert session.submit(nsteps=4)
        failed = await_result(session)
        assert not failed.step.completed
        assert failed.step.environments[0].advanced_time == pytest.approx(.004)
        assert failed.clocks[0].tick == 2
        assert failed.clocks[0].faulted
        assert failed.snapshot is None
        assert "injected solver failure" in failed.step.environments[0].error
        assert session.reset()
        await_result(session)
        assert session.submit()
        assert await_result(session).step.completed
    finally:
        session.shutdown()


def test_full_snapshot_pool_does_not_stop_physics(probe):
    session = AsyncSimulationSession(spec(), fields=("positions",), max_hz=1e9)
    try:
        # Retain only NumPy views: their buffer owners must lease the frames.
        views = [np.asarray(await_result(session).snapshot.buffer("positions"))]
        for _ in range(2):
            assert session.submit()
            views.append(np.asarray(await_result(session).snapshot.buffer("positions")))
        assert session.submit()
        skipped = await_result(session)
        assert skipped.snapshot_skipped and skipped.snapshot is None
        assert skipped.clocks[0].tick == 3
        assert [view[0, 0] for view in views] == [0, 1, 2]
        views.pop()
        gc.collect()
        assert session.submit()
        assert await_result(session).snapshot is not None
    finally:
        session.shutdown()


def test_factory_failure_and_recipe_reject_live_objects(probe):
    with pytest.raises(TypeError):
        SimulationRuntimeSpec(__name__ + ":make_provider", {"context": object()})
    with pytest.raises(ValueError):
        SimulationRuntimeSpec("not a callable", {})
    session = AsyncSimulationSession(SimulationRuntimeSpec(__name__ + ":fail_factory", {}))
    try:
        result = await_result(session)
        assert "injected build failure" in result.error
        assert not session.ready
        assert not session.submit()
    finally:
        session.shutdown()


def test_binary_artifact_transport_preserves_bytes_and_authored_tag_keys(probe):
    recipe = SimulationRuntimeSpec(__name__ + ":binary_factory", {
        "payload": {"blob": b"\x00\xffmesh", "tag": {"kind": "bytes", "value": "authored"}}})
    session = AsyncSimulationSession(recipe)
    try:
        assert await_result(session).error == ""
    finally:
        session.shutdown()


def test_subscription_during_solve_preserves_completion_and_coalesces_selection(probe):
    session = AsyncSimulationSession(spec(), fields=("positions",), max_hz=1e9)
    try:
        await_result(session)
        probe.release.clear()
        assert session.submit()
        assert probe.entered.wait(timeout=2)
        assert session.subscribe([], environments=[0])
        assert session.subscribe(["positions"], environments=[1])
        assert not session.ready
        probe.release.set()
        stepped = await_result(session)
        assert stepped.operation == "step" and stepped.clocks[0].tick == 1
        changed = await_result(session)
        assert changed.operation == "subscribe"
        assert changed.clocks[0].tick == 1
        assert changed.snapshot.environments == [1]
        assert np.asarray(changed.snapshot.buffer("positions")).tolist() == [[1, 1, 1]]
        assert session.ready
    finally:
        probe.release.set()
        session.shutdown()


def test_play_rejects_runtime_timestep_mismatch_before_first_physics_step(probe):
    from gobot.sim import ProviderPlaySession
    from test_provider_play_session import FakeContext

    class AsyncContext(FakeContext):
        def _begin_external_simulation(self, *callbacks, **asynchronous):
            self.asynchronous = asynchronous
            return super()._begin_external_simulation(*callbacks)

    context = AsyncContext()
    session = AsyncSimulationSession(spec())
    play = ProviderPlaySession(context, session, fixed_dt=.004).start()
    try:
        deadline = time.monotonic() + 5
        with pytest.raises(ValueError, match="timestep differs"):
            while time.monotonic() < deadline:
                context.asynchronous["poll"]()
                time.sleep(.001)
        assert probe.calls == 0
    finally:
        play.close()
        session.shutdown()
