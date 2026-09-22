"""Real worker rope factory: reset, selected snapshots and contact subscriptions."""
from dataclasses import asdict, replace
import os
from pathlib import Path
import time

import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get("GOBOT_RUN_WARP_IPC_GPU_TEST") != "1", reason="requires CUDA/libuipc")
def test_rope_worker_reset_and_contact_subscription(monkeypatch, tmp_path):
    import gobot
    from gobot.sim import AsyncSimulationSession, SimulationRuntimeSpec
    from gobot.sim.providers import CompiledMuJoCoIpcArtifact

    project = Path(__file__).resolve().parents[2] / "examples/dual_arm_rope_twist"
    monkeypatch.syspath_prepend(str(project))
    import rope_twist_config as config
    import controllers

    context = gobot.app.create_context()
    context.set_project_path(str(project))
    context.load_scene("res://dual_arm_rope_twist.jscn")
    settings = context.get_mujoco_solver_settings()
    settings.update(integrator=gobot.PhysicsIntegratorType.ImplicitFast,
                    cone=gobot.PhysicsFrictionConeType.Elliptic,
                    impedance_ratio=controllers.GRIP_IMPEDANCE_RATIO)
    context.set_mujoco_solver_settings(settings)
    artifact = CompiledMuJoCoIpcArtifact.from_context(context)
    profile = config.QUALITY_PROFILES["interactive"]
    solver = config._batch_config(context, profile)
    solver = replace(solver, solver=replace(solver.solver, workspace=str(tmp_path)))
    recipe = SimulationRuntimeSpec("rope_twist_runtime:create", {
        "artifact": artifact.to_mapping(), "solver_config": asdict(solver),
        "quality": asdict(profile), "coupling_iterations": profile.coupling_iterations,
        "drive_mode": controllers.SHOWCASE_DRIVE_MODE,
        "wrist_torque_limit": controllers.wrist_drive_torque_limit(controllers.SHOWCASE_DRIVE_MODE)})
    runtime = AsyncSimulationSession(recipe, fields=config.BASE_FIELDS, max_hz=1e9)

    def complete():
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            value = runtime.poll()
            if value is not None:
                assert not value.error, value.error
                assert not value.presentation_error, value.presentation_error
                if value.snapshot is not None:
                    for field in value.snapshot.fields:
                        assert np.isfinite(np.asarray(value.snapshot.buffer(field))).all(), field
                return value
            time.sleep(.005)
        raise AssertionError("rope worker completion timed out")

    try:
        installed = complete()
        assert installed.info.provider_name == "MuJoCo+libuipc"
        initial = {name: np.asarray(installed.snapshot.buffer(name)).copy()
                   for name in (*config.SCENE_FIELDS, "deformable.local_vertices")}
        del installed
        for tick in range(1, 25):
            assert runtime.submit()
            result = complete()
            assert result.step.completed
            assert result.clocks[0].tick == tick
            del result
        assert runtime.subscribe(config.BASE_FIELDS + config.CONTACT_FIELDS, max_hz=1e9)
        complete()  # subscription acknowledgement
        assert runtime.submit()
        result = complete()
        assert set(config.CONTACT_FIELDS) <= set(result.snapshot.fields)
        del result
        assert runtime.reset()
        result = complete()
        assert result.clocks[0].tick == 0
        for name, expected in initial.items():
            np.testing.assert_allclose(result.snapshot.buffer(name), expected, rtol=0, atol=1e-7)
        assert np.asarray(result.snapshot.buffer("rope.metrics"))[0, -1] == 0
        del result
        assert runtime.subscribe(config.BASE_FIELDS, max_hz=1e9)
        complete()
        assert runtime.submit()
        result = complete()
        assert result.clocks[0].tick == 1
        assert not set(config.CONTACT_FIELDS) & set(result.snapshot.fields)
    finally:
        runtime.shutdown()
        context.clear_scene()
