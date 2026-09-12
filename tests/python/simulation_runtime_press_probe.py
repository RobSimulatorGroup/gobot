"""Cold-start GPU regression for the editor's real four-environment factory."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import gobot
from gobot.sim import AsyncSimulationSession, SimulationRuntimeSpec
from gobot.sim.providers import CompiledMuJoCoIpcArtifact


def main():
    project = Path(__file__).resolve().parents[2] / "examples" / "mujoco_libuipc"
    sys.path.insert(0, str(project))
    import mujoco_libuipc_play as demo

    context = gobot.app.create_context()
    context.set_project_path(str(project))
    context.load_scene("res://soft_press_batch.jscn")
    artifact = CompiledMuJoCoIpcArtifact.from_context(context)
    config = asdict(demo._batch_config(context))
    with tempfile.TemporaryDirectory(prefix="gobot-async-press-") as workspace:
        config["solver"]["workspace"] = workspace
        assert "torch" not in sys.modules, "Torch must first load on the worker"
        runtime = AsyncSimulationSession(SimulationRuntimeSpec("mujoco_libuipc_runtime:create", {
            "artifact": artifact.to_mapping(), "solver_config": config,
            "num_envs": demo.NUM_ENVS, "environments_per_shard": demo.ENVIRONMENTS_PER_SHARD,
            "trajectory": {"depth_scales": demo.DEPTH_SCALES, "press_depth": demo.PRESS_DEPTH,
                "settle_ticks": demo.SETTLE_TICKS, "press_ticks": demo.PRESS_TICKS,
                "hold_ticks": demo.HOLD_TICKS, "release_ticks": demo.RELEASE_TICKS}}),
            fields=("robot.press.link_pose", "deformable.local_vertices"),
            environments=range(demo.NUM_ENVS), max_hz=1e9)

        def complete():
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                result = runtime.poll()
                if result is not None:
                    assert not result.error, result.error
                    assert not result.presentation_error, result.presentation_error
                    assert result.snapshot is not None
                    for field in result.snapshot.fields:
                        values = np.asarray(result.snapshot.buffer(field))
                        assert values.shape[0] == demo.NUM_ENVS
                        assert np.isfinite(values).all()
                    return result
                time.sleep(.001)
            raise AssertionError("GPU worker timed out")

        try:
            complete()
            for tick in range(1, demo.CYCLE_TICKS + 1):
                assert runtime.submit()
                result = complete()
                assert result.step.completed
                assert all(clock.tick == tick for clock in result.clocks)
                if tick % 64 == 0:
                    print(f"GPU press: {tick}/{demo.CYCLE_TICKS} ticks", flush=True)
            assert runtime.reset()
            assert all(clock.tick == 0 for clock in complete().clocks)
            assert runtime.submit()
            assert all(clock.tick == 1 for clock in complete().clocks)
        finally:
            runtime.shutdown()
    print(json.dumps({"environments": demo.NUM_ENVS, "cycle_ticks": demo.CYCLE_TICKS,
        "finite_snapshots": True, "reset_and_restep": True, "shutdown": True}))


if __name__ == "__main__":
    main()
