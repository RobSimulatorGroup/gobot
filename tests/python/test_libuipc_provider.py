from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import gobot
from gobot.ipc import LibuipcConfig, LibuipcProvider


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "examples" / "libuipc" / "fr3_brick_grasp.jscn"


def _artifact():
    context = gobot.app.create_context()
    context.set_project_path(str(SCENE.parent))
    context.load_scene("res://" + SCENE.name)
    return context.compile_ipc_scene_artifact()


class _FakeSession:
    def __init__(self, artifact) -> None:
        manifest = json.loads(artifact["manifest"])
        deformables = manifest["deformable_bodies"]
        affine_links = [
            link
            for robot in manifest["robots"]
            for link in robot["links"]
            if any(
                not shape.get("disabled", False)
                for shape in link["collision_shapes"]
            )
        ]
        offset = 0
        self.deformable_bodies = []
        for body in deformables:
            count = int(body["vertex_count"])
            self.deformable_bodies.append(
                {
                    "path": body["path"],
                    "element_offset": offset,
                    "element_count": count,
                }
            )
            offset += count
        self.affine_bodies = [
            {"path": link["path"], "element_offset": index, "element_count": 1}
            for index, link in enumerate(affine_links)
        ]
        self.positions = np.zeros((offset, 3), dtype=np.float64)
        self.velocities = np.zeros_like(self.positions)
        self.deformable_contact_forces = np.zeros_like(self.positions)
        self.affine_transforms = np.repeat(
            np.eye(4, dtype=np.float64)[None], len(affine_links), axis=0
        )
        self.frame = 0
        self.closed = False
        self.target = None
        self.joint_target = None

    @property
    def diagnostics(self):
        return {
            "provider_name": "fake-libuipc",
            "frame": self.frame,
            "deformable_body_count": len(self.deformable_bodies),
            "deformable_vertex_count": len(self.positions),
            "affine_body_count": len(self.affine_bodies),
            "last_step_latency_ms": 0.25,
            "valid": True,
        }

    def step(self, steps=1) -> None:
        self.frame += steps
        self.positions[:, 2] += 0.01 * steps
        self.deformable_contact_forces[:, 2] = float(self.frame)

    def reset(self) -> None:
        self.frame = 0
        self.positions.fill(0.0)
        self.deformable_contact_forces.fill(0.0)

    def set_affine_target(self, path, transform) -> None:
        self.target = (path, np.asarray(transform).copy())

    def set_joint_target(self, path, position) -> None:
        self.joint_target = (path, float(position))

    def close(self) -> None:
        self.closed = True


class _SceneContext:
    def __init__(self) -> None:
        self.update = None
        self.link_update = None

    def apply_deformable_vertices(self, bodies, positions, vertex_counts) -> None:
        self.update = (bodies, positions.copy(), tuple(vertex_counts))

    def apply_link_poses(self, links, poses) -> None:
        self.link_update = (links, poses.copy())


def _raises(expected, callback):
    try:
        callback()
    except expected as error:
        return error
    raise AssertionError(f"expected {expected.__name__}")


def test_libuipc_config_validation() -> None:
    config = LibuipcConfig(device_index=2, contact_activation_distance=0.0005)
    assert config.device_index == 2
    assert config.contact_activation_distance == 0.0005
    assert config.solver_mapping()["contact_activation_distance"] == 0.0005
    assert LibuipcConfig().contact_activation_distance == 0.01
    for value in (0.0, -1.0, float("nan"), float("inf")):
        _raises(ValueError, lambda item=value: LibuipcConfig(fixed_time_step=item))
        _raises(
            ValueError,
            lambda item=value: LibuipcConfig(contact_activation_distance=item),
        )
    _raises(ValueError, lambda: LibuipcConfig(friction_coefficient=-0.1))
    _raises(TypeError, lambda: LibuipcConfig(device_index=1.5))
    _raises(ValueError, lambda: LibuipcConfig(device_index=True))


def test_libuipc_public_api_is_native_only() -> None:
    assert gobot.ipc.__all__ == [
        "CompiledIpcSceneArtifact",
        "LibuipcBatchConfig",
        "LibuipcBatchSolver",
        "LibuipcConfig",
        "LibuipcProvider",
        "LibuipcProviderAvailability",
    ]
    for removed_name in (
        "DeformableBatchView",
        "TactileBatchView",
        "WarpIpcProvider",
    ):
        assert not hasattr(gobot.ipc, removed_name)
        assert not hasattr(gobot.rl, removed_name)
    assert gobot.rl.BatchProviderCapabilities is gobot.sim.ProviderCapabilities
    assert gobot.rl.ProviderUnavailableError is gobot.sim.ProviderUnavailableError
    assert not issubclass(LibuipcProvider, gobot.rl.BatchPhysicsProvider)
    assert hasattr(gobot.rl, "MuJoCoIpcProvider")


def test_libuipc_provider_lifecycle_and_scene_sync() -> None:
    artifact = _artifact()
    manifest = json.loads(artifact["manifest"])
    deformable_vertex_count = sum(
        int(body["vertex_count"]) for body in manifest["deformable_bodies"]
    )
    affine_body_count = sum(
        1
        for robot in manifest["robots"]
        for link in robot["links"]
        if any(
            not shape.get("disabled", False)
            for shape in link["collision_shapes"]
        )
    )
    joint_count = sum(len(robot["joints"]) for robot in manifest["robots"])
    session = _FakeSession(artifact)
    provider = LibuipcProvider(artifact, _session=session)
    assert provider.capabilities.name == "libuipc"
    assert provider.capabilities.device == "cuda:0"
    assert provider.num_envs == 1
    assert provider.capacities == {
        "deformable_bodies": len(manifest["deformable_bodies"]),
        "deformable_vertices": deformable_vertex_count,
        "affine_bodies": affine_body_count,
        "joints": joint_count,
    }
    assert provider.diagnostics["provider_name"] == "fake-libuipc"
    storage = provider.arrays["positions"]
    contact_force_storage = provider.arrays["contact_forces"]
    assert contact_force_storage.shape == (deformable_vertex_count, 3)

    provider.step(nsteps=2)
    assert provider.diagnostics["frame"] == 2
    assert np.allclose(storage[:, 2], 0.02)
    assert np.allclose(contact_force_storage[:, 2], 2.0)
    _raises(TypeError, lambda: provider.step(object()))
    _raises(TypeError, lambda: provider.step(nsteps=1.5))
    _raises(ValueError, lambda: provider.step(nsteps=0))

    context = _SceneContext()
    body = SimpleNamespace(name="soft_workpiece")
    links = tuple(
        SimpleNamespace(name=str(entry["path"]).rsplit("/", 1)[-1])
        for entry in provider.affine_bodies
    )
    provider.bind_scene(context, (body,), links)
    provider.sync_scene()
    assert context.update[0] == (body,)
    assert context.update[1].shape == (
        len(manifest["deformable_bodies"]),
        deformable_vertex_count,
        3,
    )
    assert context.update[1].dtype == np.float32
    assert context.update[2] == tuple(
        int(body["vertex_count"]) for body in manifest["deformable_bodies"]
    )
    assert context.link_update[0] == links
    assert context.link_update[1].shape == (affine_body_count, 7)
    assert context.link_update[1].dtype == np.float32
    np.testing.assert_allclose(
        context.link_update[1][:, 3:],
        np.repeat(
            np.asarray([[0, 0, 0, 1]], dtype=np.float32),
            affine_body_count,
            axis=0,
        ),
    )

    path = str(provider.affine_bodies[0]["path"])
    target = np.eye(4, dtype=np.float64)
    target[2, 3] = 0.5
    provider.set_affine_target(path, target)
    assert session.target[0] == path
    np.testing.assert_allclose(session.target[1], target)
    _raises(KeyError, lambda: provider.set_affine_target("/missing", target))

    joint_path = str(provider.joints[0]["path"])
    provider.set_joint_target(joint_path, 0.25)
    assert session.joint_target == (joint_path, 0.25)
    _raises(KeyError, lambda: provider.set_joint_target("/missing", 0.0))
    _raises(ValueError, lambda: provider.set_joint_target(joint_path, float("nan")))

    provider.reset(np.asarray([True]))
    assert provider.diagnostics["frame"] == 0
    assert np.all(provider.arrays["positions"] == 0.0)
    assert np.all(provider.arrays["contact_forces"] == 0.0)
    assert provider.arrays["contact_forces"] is contact_force_storage
    _raises(ValueError, lambda: provider.reset(np.asarray([False])))
    _raises(TypeError, lambda: provider.reset(custom=np.zeros(1)))

    provider.close()
    assert session.closed
    _raises(RuntimeError, lambda: provider.step())
    _raises(RuntimeError, lambda: provider.arrays)


def test_libuipc_provider_rejects_runtime_layout_mismatch() -> None:
    artifact = _artifact()
    session = _FakeSession(artifact)
    session.deformable_bodies[0]["element_count"] -= 1
    error = _raises(
        RuntimeError, lambda: LibuipcProvider(artifact, _session=session)
    )
    assert "layout" in str(error)
    assert session.closed


def test_libuipc_provider_rejects_contact_force_shape_mismatch() -> None:
    artifact = _artifact()
    session = _FakeSession(artifact)
    session.deformable_contact_forces = np.zeros(
        (len(session.positions) - 1, 3), dtype=np.float64
    )
    error = _raises(
        RuntimeError, lambda: LibuipcProvider(artifact, _session=session)
    )
    assert "contact_forces" in str(error)
    assert session.closed


def main() -> int:
    test_libuipc_config_validation()
    test_libuipc_public_api_is_native_only()
    test_libuipc_provider_lifecycle_and_scene_sync()
    test_libuipc_provider_rejects_runtime_layout_mismatch()
    test_libuipc_provider_rejects_contact_force_shape_mismatch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
