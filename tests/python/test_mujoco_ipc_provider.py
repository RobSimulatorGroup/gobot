from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

import gobot
from gobot.ipc import (
    CompiledIpcSceneArtifact,
    LibuipcBatchConfig,
    LibuipcBatchSolver,
    LibuipcConfig,
)
from gobot.rl import (
    CompiledMuJoCoIpcArtifact,
    MuJoCoIpcConfig,
    MuJoCoIpcCoupler,
    MuJoCoIpcProvider,
)


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "examples" / "mujoco_libuipc" / "soft_press_batch.jscn"
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
        self.step_nsteps = []
        self.reset_count = 0
        self.last_actions = None
        self.fail_next_step = False
        self.closed = False
        self.runtime_fingerprint = "fake-rigid"
        self.capabilities = gobot.sim.ProviderCapabilities(
            "fake-mujoco",
            "cpu",
            False,
            True,
            True,
            True,
            runtime_checkpoint=True,
            exact_contact_wrench=True,
            sensor_batch=True,
            solver_substeps=True,
            reset_scope="masked",
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
        if self.fail_next_step:
            self.fail_next_step = False
            raise RuntimeError("injected rigid step failure")
        self.step_count += nsteps
        self.step_nsteps.append(nsteps)
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
        self.step_nsteps = []
        self.reset_count = 0
        self.closed = False
        self.target_set_calls = 0
        self.fail_next_reset = False
        self.runtime_fingerprint = "fake-ipc"
        self.capabilities = gobot.sim.ProviderCapabilities(
            "fake-libuipc",
            "cpu",
            False,
            False,
            False,
            True,
            exact_contact_wrench=True,
            solver_substeps=True,
            graph_capture_reason="fake staged exchange",
            reset_scope="full_batch_only",
        )
        self.capacities = {"affine_bodies_per_env": count}

    @property
    def arrays(self):
        return self._arrays

    @property
    def diagnostics(self):
        return {"frame": self.step_count}

    def set_affine_targets(self, targets):
        self.target_set_calls += 1

    def step(self, *, nsteps=1):
        self.step_count += nsteps
        self.step_nsteps.append(nsteps)
        base = torch.arange(1, 7, dtype=torch.float64)
        self._arrays["affine_contact_wrenches"].copy_(
            base * float(self.step_count)
        )
        return self._arrays

    def reset(self, reset_mask):
        if self.fail_next_reset:
            self.fail_next_reset = False
            raise RuntimeError("injected IPC reset failure")
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
        self.step_nsteps.append(nsteps)
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
    assert artifact.schema_version == 3
    assert artifact.coupled_bodies
    assert tuple(mapping.mode for mapping in artifact.coupled_bodies) == (
        "OneWay",
        "TwoWay",
    )
    assert tuple(
        mapping.coupling_path for mapping in artifact.coupled_bodies
    ) == tuple(sorted(mapping.coupling_path for mapping in artifact.coupled_bodies))
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


def test_composite_rejects_v1_and_missing_physics_coupling() -> None:
    artifact = _artifact()
    legacy = dict(artifact.to_mapping())
    legacy["schema_version"] = 1
    error = _raises(
        ValueError, lambda: CompiledMuJoCoIpcArtifact.from_mapping(legacy)
    )
    assert "PhysicsCoupling" in str(error)

    ipc_mapping = dict(artifact.ipc.to_mapping())
    manifest = json.loads(str(ipc_mapping["manifest"]))
    manifest["couplings"] = []
    manifest_text = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    ipc_mapping["manifest"] = manifest_text
    ipc_mapping["manifest_sha256"] = "sha256:" + hashlib.sha256(
        manifest_text.encode("utf-8")
    ).hexdigest()
    uncoupled_ipc = CompiledIpcSceneArtifact.from_mapping(ipc_mapping)
    error = _raises(
        ValueError,
        lambda: CompiledMuJoCoIpcArtifact.from_artifacts(
            artifact.mujoco, uncoupled_ipc
        ),
    )
    assert "PhysicsCoupling" in str(error)


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
    affine_paths = tuple(body["path"] for body in solver.affine_bodies)
    assert affine_paths == tuple(
        mapping.ipc_path for mapping in artifact.coupled_bodies
    )
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
    assert ipc.target_set_calls == 0
    expected = torch.tensor(
        [[10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [12.0, 4.0, 6.0, 2.0, 2.5, 3.0]]
    )
    assert torch.allclose(rigid.arrays["xfrc_applied"], expected)

    # A caller may edit the same MuJoCo force slot. The coupler removes only
    # its previous contribution before applying the next IPC wrench.
    rigid.arrays["xfrc_applied"][..., 0].add_(5.0)
    provider.step()
    expected = torch.tensor(
        [[15.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [19.0, 8.0, 12.0, 4.0, 5.0, 6.0]]
    )
    assert torch.allclose(rigid.arrays["xfrc_applied"], expected)

    partial_error = _raises(
        NotImplementedError,
        lambda: provider.reset(torch.tensor([True, False, True, True])),
    )
    assert "within a shard" in str(partial_error)
    assert rigid.reset_count == 0
    assert ipc.reset_count == 0
    provider.reset(torch.ones(4, dtype=torch.bool))
    assert rigid.reset_count == 1
    assert ipc.reset_count == 1
    assert provider.diagnostics["frame"] == 0
    provider.close()
    assert rigid.closed and ipc.closed
    _raises(RuntimeError, lambda: provider.step())


def test_coupler_five_phase_protocol_binding_scales_and_storage() -> None:
    artifact = _artifact()
    mappings = tuple(
        replace(mapping, force_scale=0.5, torque_scale=0.25)
        if mapping.mode == "TwoWay"
        else mapping
        for mapping in artifact.coupled_bodies
    )
    rigid = _FakeRigidSolver(artifact, 2)
    ipc = _FakeIpcSolver(artifact, 2)
    _raises(
        ValueError,
        lambda: MuJoCoIpcCoupler(rigid, ipc, mappings, force_scale=-1.0),
    )
    _raises(
        ValueError,
        lambda: MuJoCoIpcCoupler(rigid, ipc, mappings, torque_scale=float("nan")),
    )
    coupler = MuJoCoIpcCoupler(
        rigid,
        ipc,
        mappings,
        force_scale=2.0,
        torque_scale=4.0,
    )
    storage = coupler.storage_signature

    coupler.PushRigidPose()
    assert coupler.phase == "StepIpc"
    coupler.StepIpc()
    assert coupler.phase == "ApplyFeedback"
    coupler.ApplyFeedback()
    assert coupler.phase == "StepRigid"
    expected = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    )
    assert torch.allclose(rigid.arrays["xfrc_applied"], expected)
    coupler.StepRigid()
    assert coupler.phase == "Finalize"
    coupler.Finalize()
    assert coupler.phase == "Idle"
    assert coupler.storage_signature == storage
    assert ipc.target_set_calls == 0
    coupler.release_wrenches()
    assert torch.count_nonzero(rigid.arrays["xfrc_applied"]) == 0


def test_step_failure_releases_owned_wrench_and_full_reset_recovers() -> None:
    artifact = _artifact()
    rigid = _FakeRigidSolver(artifact, 2)
    ipc = _FakeIpcSolver(artifact, 2)
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
    rigid.arrays["xfrc_applied"][..., 0] = 7.0
    rigid.fail_next_step = True
    _raises(RuntimeError, lambda: provider.step())
    assert torch.allclose(
        rigid.arrays["xfrc_applied"][..., 0],
        torch.full_like(rigid.arrays["xfrc_applied"][..., 0], 7.0),
    )
    assert torch.count_nonzero(rigid.arrays["xfrc_applied"][..., 1:]) == 0
    assert provider.diagnostics["faulted"] is True
    assert provider.diagnostics["coupler_phase"] == "Faulted"
    _raises(RuntimeError, lambda: provider.step())

    ipc.fail_next_reset = True
    _raises(RuntimeError, lambda: provider.reset(torch.ones(2, dtype=torch.bool)))
    assert provider.diagnostics["faulted"] is True
    provider.reset(torch.ones(2, dtype=torch.bool))
    assert provider.diagnostics["faulted"] is False
    assert provider.diagnostics["coupler_phase"] == "Idle"
    provider.step()
    provider.close()


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
        rigid.arrays["xfrc_applied"][:, 1, 0],
        torch.full_like(rigid.arrays["xfrc_applied"][:, 1, 0], 20.0),
    )
    assert torch.count_nonzero(rigid.arrays["xfrc_applied"][:, 0]) == 0
    assert torch.count_nonzero(rigid.arrays["xfrc_applied"][:, 1, 1:]) == 0
    assert provider.diagnostics["feedback_source"] == "proxy_constraint"
    provider.close()


def test_mujoco_ipc_config_validates_solver_substeps() -> None:
    config = MuJoCoIpcConfig(rigid_substeps=2, ipc_substeps=3)
    assert config.rigid_substeps == 2
    assert config.ipc_substeps == 3
    _raises(ValueError, lambda: MuJoCoIpcConfig(rigid_substeps=0))
    _raises(ValueError, lambda: MuJoCoIpcConfig(ipc_substeps=True))
    _raises(TypeError, lambda: MuJoCoIpcConfig(ipc_substeps=1.5))
    _raises(TypeError, lambda: MuJoCoIpcConfig(require_full_reset=False))


def test_composite_supports_explicit_solver_subcycling() -> None:
    artifact = _artifact()
    rigid = _FakeRigidSolver(artifact, 4)
    ipc = _FakeIpcSolver(artifact, 4)
    ipc.fixed_time_step = 0.02
    provider = MuJoCoIpcProvider(
        artifact,
        config=MuJoCoIpcConfig(
            num_envs=4,
            device="cpu",
            environments_per_shard=2,
            rigid_substeps=2,
            ipc_substeps=1,
        ),
        rigid_solver=rigid,
        ipc_solver=ipc,
    )

    provider.step()

    assert provider.fixed_time_step == 0.02
    assert rigid.step_nsteps == [2]
    assert ipc.step_nsteps == [1]
    assert provider.capabilities.exact_contact_wrench is True
    assert provider.capabilities.sensor_batch is True
    assert provider.capabilities.solver_substeps is True
    assert provider.capabilities.runtime_checkpoint is False
    assert provider.capabilities.reset_scope == "full_batch_only"
    assert provider.diagnostics["integration_scheme"] == "sequential_split"
    assert provider.diagnostics["rigid_substeps"] == 2
    assert provider.diagnostics["ipc_substeps"] == 1
    provider.close()


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
    assert "same macro fixed time step" in str(error)
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
    if torch is None:
        print("MuJoCo+IPC provider test skipped: Torch is not installed")
        return 77
    test_composite_artifact_has_explicit_mapping_and_ownership()
    test_composite_rejects_v1_and_missing_physics_coupling()
    test_libuipc_batch_solver_owns_stable_tensor_storage()
    test_mujoco_ipc_step_order_wrench_ownership_and_full_reset()
    test_coupler_five_phase_protocol_binding_scales_and_storage()
    test_step_failure_releases_owned_wrench_and_full_reset_recovers()
    test_mujoco_ipc_pose_error_feedback_uses_proxy_displacement()
    test_mujoco_ipc_config_validates_solver_substeps()
    test_composite_supports_explicit_solver_subcycling()
    test_composite_rejects_time_step_and_layout_mismatch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
