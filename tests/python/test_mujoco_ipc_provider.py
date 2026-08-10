from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import torch

import gobot
from gobot.ipc import LibuipcBatchConfig, LibuipcBatchSolver, LibuipcConfig
from gobot.rl import (
    CompiledMuJoCoIpcArtifact,
    MuJoCoIpcConfig,
    MuJoCoIpcProvider,
)


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "examples" / "libuipc" / "fr3_brick_grasp.jscn"
_ARTIFACT = None


def _artifact() -> CompiledMuJoCoIpcArtifact:
    global _ARTIFACT
    if _ARTIFACT is None:
        context = gobot.app.create_context()
        context.set_project_path(str(SCENE.parent))
        context.load_scene("res://" + SCENE.name)
        _ARTIFACT = CompiledMuJoCoIpcArtifact.from_context(context)
    return _ARTIFACT


def _raises(expected, callback):
    try:
        callback()
    except expected as error:
        return error
    raise AssertionError(f"expected {expected.__name__}")


class _FakeRigidSolver:
    def __init__(self, artifact, num_envs: int) -> None:
        self.artifact = artifact.mujoco
        self.num_envs = num_envs
        self.fixed_time_step = 0.01
        self._torch = torch
        names = tuple(mapping.mujoco_body_name for mapping in artifact.coupled_bodies)
        self._body_ids = {name: index for index, name in enumerate(names)}
        count = len(names)
        xmat = torch.zeros((num_envs, count, 3, 3), dtype=torch.float32)
        xmat[..., 0, 0] = 1.0
        xmat[..., 1, 1] = 1.0
        xmat[..., 2, 2] = 1.0
        self._arrays = MappingProxyType(
            {
                "qpos": torch.zeros((num_envs, 1), dtype=torch.float32),
                "qvel": torch.zeros((num_envs, 1), dtype=torch.float32),
                "ctrl": torch.zeros((num_envs, 1), dtype=torch.float32),
                "xpos": torch.zeros((num_envs, count, 3), dtype=torch.float32),
                "xmat": xmat,
                "xfrc_applied": torch.zeros(
                    (num_envs, count, 6), dtype=torch.float32
                ),
            }
        )
        self.step_count = 0
        self.reset_count = 0
        self.last_actions = None
        self.closed = False
        self.runtime_fingerprint = "fake-rigid"
        self.capabilities = gobot.sim.ProviderCapabilities(
            "fake-mujoco", "cpu", False, True, True, True
        )
        self.capacities = {"bodies": count}

    @property
    def arrays(self):
        return self._arrays

    @property
    def diagnostics(self):
        return {"frame": self.step_count}

    def resolve_object_ids(self, object_type, names):
        assert object_type == "body"
        return tuple(self._body_ids[name] for name in names)

    def step(self, actions=None, *, nsteps=1):
        self.step_count += nsteps
        if actions is not None:
            self.last_actions = actions.clone()
        return self._arrays

    def reset(self, reset_mask, **state):
        self.reset_count += 1
        self._arrays["qpos"].zero_()
        self._arrays["qvel"].zero_()
        self._arrays["xfrc_applied"].zero_()
        return self._arrays

    def close(self):
        self.closed = True


class _FakeIpcSolver:
    def __init__(self, artifact, num_envs: int) -> None:
        count = len(artifact.coupled_bodies)
        vertex_count = sum(
            int(body["vertex_count"]) for body in artifact.ipc.deformable_bodies
        )
        self.num_envs = num_envs
        self.fixed_time_step = 0.01
        self.shard_count = num_envs // 2
        self.affine_bodies = tuple(
            {
                "path": mapping.ipc_path,
                "element_offset": mapping.ipc_body_index,
                "element_count": 1,
            }
            for mapping in artifact.coupled_bodies
        )
        self._arrays = MappingProxyType(
            {
                "positions": torch.zeros(
                    (num_envs, vertex_count, 3), dtype=torch.float64
                ),
                "velocities": torch.zeros(
                    (num_envs, vertex_count, 3), dtype=torch.float64
                ),
                "contact_forces": torch.zeros(
                    (num_envs, vertex_count, 3), dtype=torch.float64
                ),
                "affine_targets": torch.zeros(
                    (num_envs, count, 4, 4), dtype=torch.float64
                ),
                "affine_transforms": torch.zeros(
                    (num_envs, count, 4, 4), dtype=torch.float64
                ),
                "affine_contact_wrenches": torch.zeros(
                    (num_envs, count, 6), dtype=torch.float64
                ),
            }
        )
        self.step_count = 0
        self.reset_count = 0
        self.closed = False
        self.runtime_fingerprint = "fake-ipc"
        self.capabilities = gobot.sim.ProviderCapabilities(
            "fake-libuipc", "cpu", False, False, False, True
        )
        self.capacities = {"affine_bodies_per_env": count}

    @property
    def arrays(self):
        return self._arrays

    @property
    def diagnostics(self):
        return {"frame": self.step_count}

    def set_affine_targets(self, targets):
        assert targets.data_ptr() == self._arrays["affine_targets"].data_ptr()

    def step(self, *, nsteps=1):
        self.step_count += nsteps
        base = torch.arange(1, 7, dtype=torch.float64)
        self._arrays["affine_contact_wrenches"].copy_(
            base * float(self.step_count)
        )
        return self._arrays

    def reset(self, reset_mask):
        self.reset_count += 1
        self._arrays["affine_contact_wrenches"].zero_()
        return self._arrays

    def close(self):
        self.closed = True


class _FakePoseErrorIpcSolver(_FakeIpcSolver):
    wrench_source = "pose_error"
    gravity = (0.0, 0.0, -9.81)

    def __init__(self, artifact, num_envs: int) -> None:
        super().__init__(artifact, num_envs)
        self.affine_bodies = tuple(
            {
                **body,
                "mass": 2.0,
                "center_of_mass": (0.0, 1.0, 0.0),
                "inertia_diagonal": (1.0, 1.0, 1.0),
                "inertia_off_diagonal": (0.0, 0.0, 0.0),
            }
            for body in self.affine_bodies
        )

    def step(self, *, nsteps=1):
        self.step_count += nsteps
        self._arrays["affine_transforms"].copy_(
            self._arrays["affine_targets"]
        )
        self._arrays["affine_transforms"][..., 0, 3].add_(0.001)
        self._arrays["affine_transforms"][..., 2, 3].add_(
            self.gravity[2] * self.fixed_time_step**2
        )
        return self._arrays


class _FakeNativeBatchSession:
    def __init__(self) -> None:
        self.buffers = None
        self.frame = 0
        self.closed = False

    def bind_device_buffers(self, buffers):
        self.buffers = buffers
        self.buffers["positions"].zero_()
        self.buffers["velocities"].zero_()
        self.buffers["contact_forces"].zero_()
        self.buffers["affine_transforms"].zero_()

    @property
    def diagnostics(self):
        return {"frame": self.frame, "valid": True}

    def step(self, count):
        self.frame += count
        self.buffers["positions"].add_(0.01 * count)

    def reset(self):
        self.frame = 0
        self.buffers["positions"].zero_()

    def close(self):
        self.closed = True


def test_composite_artifact_has_explicit_mapping_and_ownership() -> None:
    artifact = _artifact()
    assert artifact.schema_version == 1
    assert artifact.coupled_bodies
    assert tuple(
        mapping.ipc_body_index for mapping in artifact.coupled_bodies
    ) == tuple(range(len(artifact.coupled_bodies)))
    assert all(
        mapping.mujoco_body_name.endswith(mapping.link_name)
        for mapping in artifact.coupled_bodies
    )
    assert artifact.collision_ownership == {
        "rigid_rigid": "mujoco",
        "rigid_terrain": "mujoco",
        "deformable_deformable": "libuipc",
        "deformable_rigid": "libuipc",
        "deformable_terrain": "libuipc",
    }
    restored = CompiledMuJoCoIpcArtifact.from_mapping(artifact.to_mapping())
    assert restored.digest == artifact.digest

    changed = dict(artifact.to_mapping())
    changed["digest"] = "sha256:" + "0" * 64
    _raises(ValueError, lambda: CompiledMuJoCoIpcArtifact.from_mapping(changed))


def test_libuipc_batch_solver_owns_stable_tensor_storage() -> None:
    artifact = _artifact()
    session = _FakeNativeBatchSession()
    config = LibuipcBatchConfig(
        solver=LibuipcConfig(fixed_time_step=0.01),
        environments_per_shard=2,
    )
    solver = LibuipcBatchSolver(
        artifact.ipc,
        num_envs=4,
        config=config,
        device="cpu",
        _session=session,
        _torch=torch,
    )
    assert solver.shard_count == 2
    assert solver.arrays["positions"].shape[0] == 4
    assert solver.arrays["affine_targets"].shape == (
        4,
        len(artifact.coupled_bodies),
        4,
        4,
    )
    storage = solver.arrays["positions"]
    solver.step(nsteps=2)
    assert solver.arrays["positions"].data_ptr() == storage.data_ptr()
    assert torch.allclose(storage, torch.full_like(storage, 0.02))
    _raises(
        NotImplementedError,
        lambda: solver.reset(torch.tensor([True, False, True, True])),
    )
    solver.reset(torch.ones(4, dtype=torch.bool))
    assert torch.count_nonzero(storage) == 0
    solver.close()
    assert session.closed


def test_mujoco_ipc_step_order_wrench_ownership_and_full_reset() -> None:
    artifact = _artifact()
    rigid = _FakeRigidSolver(artifact, 4)
    ipc = _FakeIpcSolver(artifact, 4)
    config = MuJoCoIpcConfig(
        num_envs=4,
        device="cpu",
        environments_per_shard=2,
        force_scale=2.0,
        torque_scale=0.5,
        capture_mujoco_graphs=False,
    )
    provider = MuJoCoIpcProvider(
        artifact, config=config, rigid_solver=rigid, ipc_solver=ipc
    )
    assert isinstance(provider, gobot.rl.BatchPhysicsProvider)
    assert provider.capabilities.graph_capture is False
    assert provider.capabilities.masked_reset is False
    assert provider.capacities["shards"] == 2

    rigid.arrays["xpos"][..., 0] = 0.25
    rigid.arrays["xfrc_applied"][..., 0] = 10.0
    actions = torch.full((4, 1), 0.5)
    provider.step(actions)
    assert ipc.step_count == 1
    assert rigid.step_count == 1
    assert torch.equal(rigid.last_actions, actions)
    assert torch.allclose(
        ipc.arrays["affine_targets"][..., 0, 3],
        torch.full_like(ipc.arrays["affine_targets"][..., 0, 3], 0.25),
    )
    expected = torch.tensor([12.0, 4.0, 6.0, 2.0, 2.5, 3.0])
    assert torch.allclose(rigid.arrays["xfrc_applied"], expected)

    # A caller may edit the same MuJoCo force slot. The coupler removes only
    # its previous contribution before applying the next IPC wrench.
    rigid.arrays["xfrc_applied"][..., 0].add_(5.0)
    provider.step()
    expected = torch.tensor([19.0, 8.0, 12.0, 4.0, 5.0, 6.0])
    assert torch.allclose(rigid.arrays["xfrc_applied"], expected)

    _raises(
        NotImplementedError,
        lambda: provider.reset(torch.tensor([True, False, True, True])),
    )
    assert rigid.reset_count == 0
    assert ipc.reset_count == 0
    provider.reset(torch.ones(4, dtype=torch.bool))
    assert rigid.reset_count == 1
    assert ipc.reset_count == 1
    assert provider.diagnostics["frame"] == 0
    provider.close()
    assert rigid.closed and ipc.closed
    _raises(RuntimeError, lambda: provider.step())


def test_mujoco_ipc_pose_error_feedback_uses_proxy_displacement() -> None:
    artifact = _artifact()
    rigid = _FakeRigidSolver(artifact, 2)
    ipc = _FakePoseErrorIpcSolver(artifact, 2)
    provider = MuJoCoIpcProvider(
        artifact,
        config=MuJoCoIpcConfig(
            num_envs=2,
            device="cpu",
            environments_per_shard=2,
            capture_mujoco_graphs=False,
        ),
        rigid_solver=rigid,
        ipc_solver=ipc,
    )

    provider.step()

    # mass / dt^2 * displacement = 2 / 0.01^2 * 0.001 = 20 N.
    assert torch.allclose(
        rigid.arrays["xfrc_applied"][..., 0],
        torch.full_like(rigid.arrays["xfrc_applied"][..., 0], 20.0),
    )
    assert torch.count_nonzero(rigid.arrays["xfrc_applied"][..., 1:]) == 0
    provider.close()


def test_mujoco_ipc_rejects_time_diverging_substeps() -> None:
    _raises(ValueError, lambda: MuJoCoIpcConfig(ipc_substeps=2))
    _raises(ValueError, lambda: MuJoCoIpcConfig(require_full_reset=False))


def test_composite_rejects_time_step_and_layout_mismatch() -> None:
    artifact = _artifact()
    rigid = _FakeRigidSolver(artifact, 4)
    ipc = _FakeIpcSolver(artifact, 4)
    ipc.fixed_time_step = 0.02
    config = MuJoCoIpcConfig(
        num_envs=4, device="cpu", environments_per_shard=2
    )
    error = _raises(
        ValueError,
        lambda: MuJoCoIpcProvider(
            artifact, config=config, rigid_solver=rigid, ipc_solver=ipc
        ),
    )
    assert "same fixed time step" in str(error)
    assert rigid.closed and ipc.closed

    rigid = _FakeRigidSolver(artifact, 4)
    ipc = _FakeIpcSolver(artifact, 4)
    ipc.shard_count = 1
    error = _raises(
        ValueError,
        lambda: MuJoCoIpcProvider(
            artifact, config=config, rigid_solver=rigid, ipc_solver=ipc
        ),
    )
    assert "shard layout" in str(error)
    assert rigid.closed and ipc.closed


def main() -> int:
    test_composite_artifact_has_explicit_mapping_and_ownership()
    test_libuipc_batch_solver_owns_stable_tensor_storage()
    test_mujoco_ipc_step_order_wrench_ownership_and_full_reset()
    test_mujoco_ipc_pose_error_feedback_uses_proxy_displacement()
    test_mujoco_ipc_rejects_time_diverging_substeps()
    test_composite_rejects_time_step_and_layout_mismatch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
