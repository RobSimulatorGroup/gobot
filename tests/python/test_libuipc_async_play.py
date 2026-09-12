"""Exercise the real authored-artifact/factory/snapshot path with a CPU solver double."""
from dataclasses import asdict
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np

import gobot
from gobot.ipc import CompiledIpcSceneArtifact, LibuipcConfig, LibuipcProvider
from gobot.sim import AsyncSimulationSession, SceneSnapshotSync, SimulationRuntimeSpec
from test_libuipc_provider import _FakeSession


def test_native_factory_keeps_solver_and_controller_off_the_scene_thread(monkeypatch):
    project = Path(__file__).resolve().parents[2] / "examples" / "libuipc"
    monkeypatch.syspath_prepend(str(project))
    threads = []

    class Solver(_FakeSession):
        def __init__(self, artifact):
            threads.append(("build", threading.get_ident()))
            super().__init__(artifact)

        def step(self, steps=1):
            threads.append(("step", threading.get_ident()))
            super().step(steps)

        def set_joint_target(self, path, position):
            threads.append(("control", threading.get_ident()))
            super().set_joint_target(path, position)

        def close(self):
            threads.append(("close", threading.get_ident()))
            super().close()

    monkeypatch.setattr(LibuipcProvider, "_create_session", lambda self: Solver(self.artifact.to_mapping()))
    context = gobot.app.create_context()
    context.set_project_path(str(project))
    root = context.load_scene("res://fr3_brick_grasp.jscn")
    artifact = CompiledIpcSceneArtifact.from_mapping(context.compile_ipc_scene_artifact())
    nodes = {}
    pending = [root]
    while pending:
        node = pending.pop()
        nodes[node.name] = node
        pending.extend(node.children)
    affine = [link for robot in artifact.robots for link in robot["links"]
              if any(not shape.get("disabled", False) for shape in link["collision_shapes"])]
    sync = SceneSnapshotSync(context,
        links={"affine.link_pose": [[nodes[entry["path"].rsplit("/", 1)[-1]] for entry in affine]]},
        deformables=[[nodes[entry["path"].rsplit("/", 1)[-1]] for entry in artifact.deformable_bodies]],
        vertex_counts=[entry["vertex_count"] for entry in artifact.deformable_bodies])
    runtime = AsyncSimulationSession(SimulationRuntimeSpec("libuipc_runtime:create", {
        "artifact": artifact.to_mapping(), "config": asdict(LibuipcConfig()),
        "affine_paths": [entry["path"] for entry in affine]}),
        fields=("affine.link_pose", "deformable.local_vertices"), max_hz=1e9)

    def complete():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = runtime.poll()
            if result is not None:
                assert not result.error, result.error
                assert not result.presentation_error, result.presentation_error
                return result
            time.sleep(.001)
        raise AssertionError("native runtime factory timeout")

    try:
        installed = complete()
        assert installed.info.provider_name == "libuipc"
        sync(installed.snapshot)
        assert runtime.submit()
        stepped = complete()
        assert stepped.step.completed
        assert stepped.clocks[0].tick == 1
        sync(stepped.snapshot)
        assert runtime.reset()
        assert complete().clocks[0].tick == 0
    finally:
        runtime.shutdown()
    assert {name for name, _ in threads} == {"build", "step", "control", "close"}
    owners = {thread for _, thread in threads}
    assert len(owners) == 1 and threading.get_ident() not in owners


def test_composite_press_factory_drives_all_environments_from_binary_artifact(monkeypatch):
    import torch
    import gobot.sim.providers as providers
    from gobot.ipc import LibuipcBatchConfig

    project = Path(__file__).resolve().parents[2] / "examples" / "mujoco_libuipc"
    monkeypatch.syspath_prepend(str(project))
    context = gobot.app.create_context()
    context.set_project_path(str(project))
    context.load_scene("res://soft_press_batch.jscn")
    artifact = providers.CompiledMuJoCoIpcArtifact.from_context(context)
    commands = []
    owners = []

    class Composite:
        def __init__(self, compiled, *, config, libuipc_config, mujoco_options):
            owners.append(threading.get_ident())
            self.num_envs = config.num_envs
            self.fixed_time_step = libuipc_config.solver.fixed_time_step
            self.bodies = []
            offset = 0
            for entry in compiled.ipc.deformable_bodies:
                self.bodies.append({"path": entry["path"], "element_offset": offset,
                                    "element_count": entry["vertex_count"]})
                offset += entry["vertex_count"]
            self.ipc_solver = SimpleNamespace(deformable_bodies=self.bodies)
            self.arrays = {"ctrl": torch.zeros((self.num_envs, 1)),
                           "ipc_positions": torch.zeros((self.num_envs, offset, 3))}
            poses = torch.zeros((self.num_envs, 1, 7))
            poses[..., 6] = 1
            self.view = SimpleNamespace(read_state=lambda: SimpleNamespace(link_pose=poses),
                                        set_position_targets=self.set_targets)

        def create_robot_view(self, **kwargs):
            assert "scene_context" not in kwargs
            return self.view

        def set_targets(self, values):
            owners.append(threading.get_ident())
            commands.append(values.clone().numpy())

        def reset(self, mask):
            owners.append(threading.get_ident())
            assert all(mask)

        def step(self, *, nsteps=1):
            owners.append(threading.get_ident())

        def refresh_state(self):
            owners.append(threading.get_ident())

        def close(self):
            owners.append(threading.get_ident())

    monkeypatch.setattr(providers, "MuJoCoIpcProvider", Composite)
    runtime = AsyncSimulationSession(SimulationRuntimeSpec("mujoco_libuipc_runtime:create", {
        "artifact": artifact.to_mapping(),
        "solver_config": asdict(LibuipcBatchConfig(solver=LibuipcConfig(fixed_time_step=.002))),
        "num_envs": 4, "environments_per_shard": 4,
        "trajectory": {"depth_scales": [1., .8, .6, .4], "press_depth": .01,
                       "settle_ticks": 0, "press_ticks": 1, "hold_ticks": 1, "release_ticks": 1}}),
        fields=("robot.press.link_pose", "deformable.local_vertices"), environments=range(4), max_hz=1e9)

    def complete():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = runtime.poll()
            if result is not None:
                assert not result.error, result.error
                assert not result.presentation_error, result.presentation_error
                return result
            time.sleep(.001)
        raise AssertionError("composite runtime factory timeout")

    try:
        installed = complete()
        assert installed.info.environment_count == 4
        assert runtime.submit()
        stepped = complete()
        assert stepped.step.completed
        assert [clock.tick for clock in stepped.clocks] == [1] * 4
        assert np.asarray(stepped.snapshot.buffer("robot.press.link_pose")).shape == (4, 1, 7)
        np.testing.assert_allclose(commands[-1][:, 0], [-.01, -.008, -.006, -.004])
        assert runtime.reset()
        assert [clock.tick for clock in complete().clocks] == [0] * 4
        np.testing.assert_array_equal(commands[-1], np.zeros((4, 1)))
    finally:
        runtime.shutdown()
    assert len(set(owners)) == 1 and threading.get_ident() not in owners
