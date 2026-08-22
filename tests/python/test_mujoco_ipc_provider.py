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
    MuJoCoIpcConvergencePolicy,
    MuJoCoIpcConfig,
    MuJoCoIpcProvider,
    SolverCoupledProxy,
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
                "time": torch.zeros((num_envs,), dtype=torch.float32),
                "qpos": torch.zeros((num_envs, 1), dtype=torch.float32),
                "qvel": torch.zeros((num_envs, 1), dtype=torch.float32),
                "ctrl": torch.zeros((num_envs, 1), dtype=torch.float32),
                "xpos": torch.zeros((num_envs, count, 3), dtype=torch.float32),
                "xipos": torch.zeros((num_envs, count, 3), dtype=torch.float32),
                "subtree_com": torch.zeros(
                    (num_envs, count, 3), dtype=torch.float32
                ),
                "xmat": xmat,
                "cvel": torch.zeros((num_envs, count, 6), dtype=torch.float32),
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

    def resolve_body_root_ids(self, body_ids):
        return tuple(body_ids)

    def step(self, actions=None, *, nsteps=1):
        if self.fail_next_step:
            self.fail_next_step = False
            raise RuntimeError("injected rigid step failure")
        self.step_count += nsteps
        self.step_nsteps.append(nsteps)
        self._arrays["time"].add_(self.fixed_time_step * nsteps)
        if actions is not None:
            self.last_actions = actions.clone()
        return self._arrays

    def reset(self, reset_mask, **state):
        self.reset_count += 1
        self._arrays["qpos"].zero_()
        self._arrays["qvel"].zero_()
        self._arrays["time"].zero_()
        self._arrays["xfrc_applied"].zero_()
        return self._arrays

    def forward(self):
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
                "external_forces": torch.zeros(
                    (num_envs, vertex_count, 3), dtype=torch.float64
                ),
                "contact_forces": torch.zeros(
                    (num_envs, vertex_count, 3), dtype=torch.float64
                ),
                "affine_targets": torch.zeros(
                    (num_envs, count, 4, 4), dtype=torch.float64
                ),
                "affine_target_twists": torch.zeros(
                    (num_envs, count, 6), dtype=torch.float64
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
        self.fail_next_rewind = False
        self.capture_count = 0
        self.rewind_count = 0
        self.commit_count = 0
        self._checkpoint = None
        self.runtime_fingerprint = "fake-ipc"
        self.capabilities = gobot.sim.ProviderCapabilities(
            "fake-libuipc",
            "cpu",
            False,
            False,
            False,
            True,
            runtime_checkpoint=True,
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
        self.step_count = 0
        self._checkpoint = None
        self._arrays["affine_contact_wrenches"].zero_()
        return self._arrays

    def capture_checkpoint(self):
        self.capture_count += 1
        if self._checkpoint is not None:
            raise RuntimeError("fake IPC checkpoint already active")
        self._checkpoint = {
            "step_count": self.step_count,
            "positions": self._arrays["positions"].clone(),
            "velocities": self._arrays["velocities"].clone(),
            "contact_forces": self._arrays["contact_forces"].clone(),
            "affine_transforms": self._arrays["affine_transforms"].clone(),
            "affine_contact_wrenches": self._arrays[
                "affine_contact_wrenches"
            ].clone(),
        }

    def rewind_checkpoint(self):
        self.rewind_count += 1
        if self.fail_next_rewind:
            self.fail_next_rewind = False
            raise RuntimeError("injected IPC checkpoint rewind failure")
        if self._checkpoint is None:
            raise RuntimeError("fake IPC checkpoint is not active")
        self.step_count = self._checkpoint["step_count"]
        for name, value in self._checkpoint.items():
            if name != "step_count":
                self._arrays[name].copy_(value)

    def commit_checkpoint(self):
        self.commit_count += 1
        if self._checkpoint is None:
            raise RuntimeError("fake IPC checkpoint is not active")
        self._checkpoint = None

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


class _FakeNewtonRigidSolver(_FakeRigidSolver):
    def _update_derived_state(self) -> None:
        position = self._arrays["qpos"][:, 0]
        self._arrays["xpos"][..., 0].copy_(position[:, None])
        self._arrays["xipos"].copy_(self._arrays["xpos"])
        self._arrays["subtree_com"].copy_(self._arrays["xpos"])
        self._arrays["subtree_com"][..., 1].add_(1.0)
        self._arrays["cvel"].zero_()
        self._arrays["cvel"][..., 2] = 2.0
        self._arrays["cvel"][..., 3] = 1.0

    def step(self, actions=None, *, nsteps=1):
        super().step(actions, nsteps=nsteps)
        force = self._arrays["xfrc_applied"][:, 1, 0]
        self._arrays["qpos"][:, 0].add_(force * self.fixed_time_step)
        self._arrays["qvel"][:, 0].copy_(force)
        self._update_derived_state()
        return self._arrays

    def forward(self):
        self._update_derived_state()
        return self._arrays


class _FakeNewtonIpcSolver(_FakeIpcSolver):
    def step(self, *, nsteps=1):
        self.step_count += nsteps
        self.step_nsteps.append(nsteps)
        self._arrays["affine_transforms"].copy_(
            self._arrays["affine_targets"]
        )
        self._arrays["affine_contact_wrenches"].zero_()
        reaction = 1.0 - 50.0 * self._arrays["affine_targets"][..., 0, 3]
        self._arrays["affine_contact_wrenches"][..., 0].copy_(reaction)
        return self._arrays


class _FakeAdaptiveIpcSolver(_FakeNewtonIpcSolver):
    def __init__(self, artifact, num_envs: int) -> None:
        super().__init__(artifact, num_envs)
        self.failures_remaining = 0
        self.runtime_options_calls = []
        self.report_stressed = False
        self._solver_diagnostics = {
            "solver_failure": 0,
            "newton_iterations": 4,
            "line_search_iterations_max": 1,
            "minimum_step_length": 1.0,
            "linear_system_converged": True,
        }

    @property
    def diagnostics(self):
        return {"frame": self.step_count, **self._solver_diagnostics}

    def set_runtime_solver_options(
        self,
        *,
        newton_max_iterations,
        line_search_max_iterations,
        linear_system_tolerance_rate,
        strict_convergence,
    ):
        self.runtime_options_calls.append(
            {
                "newton_max_iterations": newton_max_iterations,
                "line_search_max_iterations": line_search_max_iterations,
                "linear_system_tolerance_rate": linear_system_tolerance_rate,
                "strict_convergence": strict_convergence,
            }
        )

    def step(self, *, nsteps=1):
        result = super().step(nsteps=nsteps)
        if self.failures_remaining:
            self.failures_remaining -= 1
            self._solver_diagnostics.update(
                {
                    "solver_failure": 2,
                    "newton_iterations": 16,
                    "line_search_iterations_max": 8,
                    "minimum_step_length": 1.0e-5,
                }
            )
            raise RuntimeError("injected strict Newton failure")
        self._solver_diagnostics.update(
            {
                "solver_failure": 0,
                "newton_iterations": 12 if self.report_stressed else 4,
                "line_search_iterations_max": 6 if self.report_stressed else 1,
                "minimum_step_length": 1.0e-5 if self.report_stressed else 1.0,
                "linear_system_converged": True,
            }
        )
        return result


class _FakeImpulseRigidSolver(_FakeNewtonRigidSolver):
    def step(self, actions=None, *, nsteps=1):
        _FakeRigidSolver.step(self, actions, nsteps=nsteps)
        force = self._arrays["xfrc_applied"][:, 1, 0]
        self._arrays["qvel"][:, 0].add_(
            force * self.fixed_time_step * nsteps
        )
        self._arrays["qpos"][:, 0].add_(
            self._arrays["qvel"][:, 0] * self.fixed_time_step * nsteps
        )
        self._update_derived_state()
        return self._arrays


class _FakeImpulseIpcSolver(_FakeNewtonIpcSolver):
    gravity = (0.0, 0.0, 0.0)

    def step(self, *, nsteps=1):
        super().step(nsteps=nsteps)
        self._arrays["velocities"].zero_()
        reaction = self._arrays["affine_contact_wrenches"][:, 1, 0]
        self._arrays["velocities"][:, 0, 0].copy_(reaction).mul_(
            -self.fixed_time_step * nsteps
        )
        return self._arrays


class _FakeNativeBatchSession:
    def __init__(self) -> None:
        self.buffers = None
        self.frame = 0
        self.closed = False
        self.checkpoint = None
        self.refresh_output_flags = []
        self.runtime_options_calls = []

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

    def refresh_outputs(self, output_flags):
        self.refresh_output_flags.append(output_flags)
        if output_flags & (1 << 2):
            self.buffers["contact_forces"].fill_(float(self.frame))

    def set_runtime_options(
        self,
        newton_max_iterations,
        line_search_max_iterations,
        linear_system_tolerance_rate,
        strict_convergence,
    ):
        self.runtime_options_calls.append(
            (
                newton_max_iterations,
                line_search_max_iterations,
                linear_system_tolerance_rate,
                strict_convergence,
            )
        )

    def reset(self):
        self.frame = 0
        self.checkpoint = None
        self.buffers["positions"].zero_()

    def capture_checkpoint(self):
        self.checkpoint = (self.frame, self.buffers["positions"].clone())

    def rewind_checkpoint(self):
        self.frame = self.checkpoint[0]
        self.buffers["positions"].copy_(self.checkpoint[1])

    def commit_checkpoint(self):
        self.checkpoint = None

    def close(self):
        self.closed = True


def test_composite_artifact_has_explicit_mapping_and_ownership() -> None:
    artifact = _artifact()
    assert artifact.schema_version == 4
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
        "deformable_static": "libuipc",
        "deformable_terrain": "unsupported",
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


def test_ipc_schema_v3_normalizes_missing_static_colliders() -> None:
    artifact = _artifact().ipc
    mapping = dict(artifact.to_mapping())
    manifest = json.loads(mapping["manifest"])
    manifest["schema_version"] = 3
    manifest.pop("static_colliders", None)
    manifest_text = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    mapping["schema_version"] = 3
    mapping["manifest"] = manifest_text
    mapping["manifest_sha256"] = "sha256:" + hashlib.sha256(
        manifest_text.encode("utf-8")
    ).hexdigest()

    restored = CompiledIpcSceneArtifact.from_mapping(mapping)

    assert restored.schema_version == 3
    assert restored.static_colliders == ()
    assert restored.manifest_data["static_colliders"] == ()


def test_ipc_schema_v4_remains_readable() -> None:
    artifact = _artifact().ipc
    mapping = dict(artifact.to_mapping())
    manifest = json.loads(mapping["manifest"])
    manifest["schema_version"] = 4
    manifest_text = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    mapping["schema_version"] = 4
    mapping["manifest"] = manifest_text
    mapping["manifest_sha256"] = "sha256:" + hashlib.sha256(
        manifest_text.encode("utf-8")
    ).hexdigest()

    restored = CompiledIpcSceneArtifact.from_mapping(mapping)

    assert restored.schema_version == 4
    assert restored.static_colliders == artifact.static_colliders


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
    assert solver.frame == 0
    assert solver.arrays["positions"].shape[0] == 4
    assert solver.arrays["affine_targets"].shape == (
        4,
        len(artifact.coupled_bodies),
        4,
        4,
    )
    assert solver.arrays["affine_target_twists"].shape == (
        4,
        len(artifact.coupled_bodies),
        6,
    )
    storage = solver.arrays["positions"]
    affine_paths = tuple(body["path"] for body in solver.affine_bodies)
    assert affine_paths == tuple(
        mapping.ipc_path for mapping in artifact.coupled_bodies
    )
    solver.step(nsteps=2)
    assert solver.frame == 2
    assert solver.arrays["positions"].data_ptr() == storage.data_ptr()
    assert torch.allclose(storage, torch.full_like(storage, 0.02))
    solver.capture_checkpoint()
    checkpoint_positions = storage.clone()
    solver.step()
    solver.rewind_checkpoint()
    assert torch.equal(storage, checkpoint_positions)
    solver.step()
    solver.commit_checkpoint()
    assert solver.diagnostics["checkpoint_active"] is False
    _raises(
        NotImplementedError,
        lambda: solver.reset(torch.tensor([True, False, True, True])),
    )
    solver.reset(torch.ones(4, dtype=torch.bool))
    assert torch.count_nonzero(storage) == 0
    solver.close()
    assert session.closed


def test_libuipc_batch_lazy_contact_force_refresh() -> None:
    artifact = _artifact()
    session = _FakeNativeBatchSession()
    config = LibuipcBatchConfig(
        solver=LibuipcConfig(fixed_time_step=0.01),
        environments_per_shard=2,
        newton_max_iterations=12,
        line_search_max_iterations=6,
        linear_system_tolerance_rate=2.0e-3,
        export_deformable_state=False,
        export_affine_state=False,
        export_deformable_contact_forces=False,
    )
    mapping = config.solver_mapping(2)
    assert mapping["newton_max_iterations"] == 12
    assert mapping["line_search_max_iterations"] == 6
    assert mapping["linear_system_tolerance_rate"] == 2.0e-3
    assert mapping["output_flags"] & (1 << 0) == 0
    assert mapping["output_flags"] & (1 << 1) == 0
    assert mapping["output_flags"] & (1 << 2) == 0
    solver = LibuipcBatchSolver(
        artifact.ipc,
        num_envs=2,
        config=config,
        device="cpu",
        _session=session,
        _torch=torch,
    )
    solver.step(nsteps=3)
    assert torch.count_nonzero(solver.arrays["contact_forces"]) == 0
    assert solver.diagnostics["affine_target_staging"] == (
        "per_shard_device_host_device"
    )
    assert solver.diagnostics["contact_wrench_staging"] == (
        "per_shard_device_host_device"
    )
    state = solver.refresh_state()
    assert state["positions"] is solver.arrays["positions"]
    refreshed = solver.refresh_deformable_contact_forces()
    assert torch.all(refreshed == 3.0)
    assert solver.diagnostics["deformable_contact_force_frame"] == 3
    assert session.refresh_output_flags == [(1 << 0) | (1 << 1), 1 << 2]
    solver.set_runtime_solver_options(
        newton_max_iterations=32,
        line_search_max_iterations=16,
        linear_system_tolerance_rate=2.5e-4,
        strict_convergence=True,
    )
    assert session.runtime_options_calls == [(32, 16, 2.5e-4, True)]
    _raises(
        ValueError,
        lambda: solver.set_runtime_solver_options(
            newton_max_iterations=0,
            line_search_max_iterations=16,
            linear_system_tolerance_rate=2.5e-4,
            strict_convergence=True,
        ),
    )
    solver.close()


def test_libuipc_al_ipc_config_reports_approximate_proxy_feedback() -> None:
    artifact = _artifact()
    config = LibuipcBatchConfig(
        solver=LibuipcConfig(fixed_time_step=0.01),
        environments_per_shard=2,
        contact_constitution="al-ipc",
        al_ipc_mu_scale_fem=4.0e7,
        al_ipc_mu_scale_abd=2.0e5,
        al_ipc_toi_threshold=0.2,
        al_ipc_alpha_lower_bound=2.0e-6,
        al_ipc_decay_factor=0.4,
    )
    mapping = config.solver_mapping(2)
    assert mapping["contact_constitution"] == "al-ipc"
    assert mapping["al_ipc_mu_scale_fem"] == 4.0e7
    solver = LibuipcBatchSolver(
        artifact.ipc,
        num_envs=2,
        config=config,
        device="cpu",
        _session=_FakeNativeBatchSession(),
        _torch=torch,
    )
    assert solver.wrench_source == "pose_error"
    assert solver.capabilities.exact_contact_wrench is False
    assert solver.diagnostics["contact_constitution"] == "al-ipc"
    assert solver.diagnostics["feedback_source"] == "proxy_constraint"
    solver.close()
    _raises(
        ValueError,
        lambda: LibuipcBatchConfig(contact_constitution="unknown"),
    )
    _raises(
        ValueError,
        lambda: LibuipcBatchConfig(al_ipc_decay_factor=0.0),
    )
    _raises(
        ValueError,
        lambda: LibuipcBatchConfig(newton_max_iterations=0),
    )
    _raises(
        TypeError,
        lambda: LibuipcBatchConfig(export_deformable_contact_forces=1),
    )
    _raises(
        TypeError,
        lambda: LibuipcBatchConfig(export_deformable_state=1),
    )


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
        coupling_iterations=1,
        capture_mujoco_graphs=False,
    )
    provider = MuJoCoIpcProvider(
        artifact, config=config, rigid_solver=rigid, ipc_solver=ipc
    )
    assert isinstance(provider, gobot.rl.BatchPhysicsProvider)
    assert provider.capabilities.graph_capture is False
    assert provider.capabilities.masked_reset is False
    assert provider.capacities["shards"] == 2
    assert provider.frame == 0
    assert not provider.diagnostics["coupler_graph_captured"]
    assert "requires CUDA" in provider.diagnostics[
        "coupler_graph_capture_reason"
    ]

    rigid.arrays["xpos"][..., 0] = 0.25
    rigid.arrays["xfrc_applied"][..., 0] = 10.0
    actions = torch.full((4, 1), 0.5)
    provider.step(actions)
    assert provider.frame == 1
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
    assert provider.frame == 2
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
    assert provider.frame == 0
    provider.close()
    assert rigid.closed and ipc.closed
    _raises(RuntimeError, lambda: provider.step())


def test_solver_coupled_proxy_x1_scales_storage_and_skips_checkpoints() -> None:
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
        lambda: SolverCoupledProxy(rigid, ipc, mappings, force_scale=-1.0),
    )
    _raises(
        ValueError,
        lambda: SolverCoupledProxy(
            rigid, ipc, mappings, torque_scale=float("nan")
        ),
    )
    coupler = SolverCoupledProxy(
        rigid,
        ipc,
        mappings,
        force_scale=2.0,
        torque_scale=4.0,
        coupling_iterations=1,
        relaxation_mode="fixed",
    )
    storage = coupler.storage_signature

    coupler.step()
    assert coupler.phase == "Idle"
    expected = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    )
    assert torch.allclose(rigid.arrays["xfrc_applied"], expected)
    assert coupler.storage_signature == storage
    assert ipc.target_set_calls == 0
    assert ipc.capture_count == 0
    assert ipc.rewind_count == 0
    assert ipc.commit_count == 0
    assert "rigid_checkpoint" not in coupler.phase_latency_ms
    assert "ipc_checkpoint" not in coupler.phase_latency_ms
    coupler.release_wrenches()
    assert torch.count_nonzero(rigid.arrays["xfrc_applied"]) == 0


def test_solver_coupled_proxy_overrides_one_proxy_target_twist() -> None:
    artifact = _artifact()
    rigid = _FakeNewtonRigidSolver(artifact, 2)
    ipc = _FakeIpcSolver(artifact, 2)
    coupler = SolverCoupledProxy(
        rigid,
        ipc,
        artifact.coupled_bodies,
        coupling_iterations=1,
        relaxation_mode="fixed",
        capture_graphs=False,
    )
    storage = coupler.storage_signature
    override = torch.tensor(
        [0.75, -0.25, 0.0, 0.0, 0.0, 0.5], dtype=torch.float64
    )
    transform_override = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    transform_override[:, 0, 3] = torch.tensor(
        [0.4, 0.8], dtype=torch.float64
    )

    coupler.set_proxy_transform_override(0, transform_override)
    coupler.set_proxy_twist_override(0, override)
    coupler.step()

    assert torch.allclose(
        ipc.arrays["affine_targets"][:, 0], transform_override
    )
    expected = override.expand(2, 6)
    assert torch.allclose(
        ipc.arrays["affine_target_twists"][:, 0], expected
    )
    assert torch.allclose(
        ipc.arrays["affine_target_twists"][:, 1, 0],
        torch.full((2,), 3.0, dtype=torch.float64),
    )
    assert torch.allclose(
        ipc.arrays["affine_target_twists"][:, 1, 5],
        torch.full((2,), 2.0, dtype=torch.float64),
    )
    assert coupler.storage_signature == storage

    coupler.clear_proxy_transform_override(0)
    coupler.clear_proxy_twist_override(0)
    coupler.sync_rigid_pose()
    assert torch.allclose(
        ipc.arrays["affine_targets"][:, 0],
        torch.eye(4, dtype=torch.float64).expand(2, 4, 4),
    )
    assert torch.allclose(
        ipc.arrays["affine_target_twists"][:, 0],
        ipc.arrays["affine_target_twists"][:, 1],
    )
    _raises(IndexError, lambda: coupler.set_proxy_twist_override(-1, override))
    _raises(IndexError, lambda: coupler.clear_proxy_twist_override(99))
    _raises(
        IndexError,
        lambda: coupler.set_proxy_transform_override(99, transform_override),
    )
    _raises(IndexError, lambda: coupler.clear_proxy_transform_override(-1))
    _raises(
        ValueError,
        lambda: coupler.set_proxy_twist_override(0, torch.zeros(2, 5)),
    )
    _raises(
        ValueError,
        lambda: coupler.set_proxy_transform_override(0, torch.zeros(2, 3, 4)),
    )


def test_solver_coupled_proxy_stress_guard_promotes_x1_to_x2() -> None:
    artifact = _artifact()
    rigid = _FakeNewtonRigidSolver(artifact, 2)
    ipc = _FakeAdaptiveIpcSolver(artifact, 2)
    ipc.report_stressed = True
    coupler = SolverCoupledProxy(
        rigid,
        ipc,
        artifact.coupled_bodies,
        coupling_iterations=1,
        relaxation_mode="fixed",
        capture_graphs=False,
        convergence_policy=MuJoCoIpcConvergencePolicy(enabled=True),
    )

    coupler.step()
    assert coupler.convergence_guard_diagnostics["guarded"] is True
    assert coupler.rollback_enabled is True
    assert coupler.last_coupling_iterations == 1

    coupler.step()
    guard = coupler.convergence_guard_diagnostics
    assert guard["guarded_step_count"] == 1
    assert coupler.last_coupling_iterations == 2
    assert ipc.capture_count == 2
    assert ipc.rewind_count == 1
    assert ipc.commit_count == 2
    assert ipc.runtime_options_calls[0]["strict_convergence"] is True
    assert ipc.runtime_options_calls[-2] == {
        "newton_max_iterations": 64,
        "line_search_max_iterations": 16,
        "linear_system_tolerance_rate": 2.5e-4,
        "strict_convergence": True,
    }
    assert ipc.runtime_options_calls[-1] == ipc.runtime_options_calls[0]


def test_solver_coupled_proxy_strict_failure_rewinds_and_retries_once() -> None:
    artifact = _artifact()
    rigid = _FakeNewtonRigidSolver(artifact, 2)
    ipc = _FakeAdaptiveIpcSolver(artifact, 2)
    policy = MuJoCoIpcConvergencePolicy(enabled=True)
    coupler = SolverCoupledProxy(
        rigid,
        ipc,
        artifact.coupled_bodies,
        coupling_iterations=1,
        relaxation_mode="fixed",
        capture_graphs=False,
        convergence_policy=policy,
    )
    ipc.failures_remaining = 1

    coupler.step()

    assert coupler.phase == "Idle"
    assert rigid.step_count == 1
    assert ipc.step_count == 1
    assert ipc.capture_count == 1
    assert ipc.rewind_count == 2
    assert ipc.commit_count == 1
    assert coupler.last_coupling_iterations == 2
    guard = coupler.convergence_guard_diagnostics
    assert guard["solver_retry_count"] == 1
    assert guard["solver_restore_count"] == 1
    assert guard["strict_failure_count"] == 1
    assert "strict Newton failure" in guard["last_retry_reason"]
    assert ipc.runtime_options_calls[-2] == {
        "newton_max_iterations": policy.fallback_newton_max_iterations,
        "line_search_max_iterations": policy.fallback_line_search_max_iterations,
        "linear_system_tolerance_rate": (
            policy.fallback_linear_system_tolerance_rate
        ),
        "strict_convergence": True,
    }
    assert ipc.runtime_options_calls[-1] == ipc.runtime_options_calls[0]


def test_solver_coupled_proxy_x2_failure_rewinds_without_adaptive_policy() -> None:
    artifact = _artifact()
    rigid = _FakeNewtonRigidSolver(artifact, 2)
    ipc = _FakeAdaptiveIpcSolver(artifact, 2)
    coupler = SolverCoupledProxy(
        rigid,
        ipc,
        artifact.coupled_bodies,
        coupling_iterations=2,
        relaxation_mode="aitken",
        capture_graphs=False,
    )
    ipc.failures_remaining = 1

    error = _raises(RuntimeError, coupler.step)

    assert "strict Newton failure" in str(error)
    assert coupler.phase == "Idle"
    assert rigid.step_count == 0
    assert ipc.step_count == 0
    assert ipc.capture_count == 1
    assert ipc.rewind_count == 1
    assert ipc.commit_count == 1


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
            coupling_iterations=1,
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
            coupling_iterations=1,
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


def test_mujoco_ipc_config_requires_matching_solver_substeps() -> None:
    config = MuJoCoIpcConfig(rigid_substeps=2, ipc_substeps=2)
    assert config.rigid_substeps == 2
    assert config.ipc_substeps == 2
    _raises(
        ValueError,
        lambda: MuJoCoIpcConfig(rigid_substeps=2, ipc_substeps=3),
    )
    _raises(ValueError, lambda: MuJoCoIpcConfig(rigid_substeps=0))
    _raises(ValueError, lambda: MuJoCoIpcConfig(ipc_substeps=True))
    _raises(TypeError, lambda: MuJoCoIpcConfig(ipc_substeps=1.5))
    _raises(TypeError, lambda: MuJoCoIpcConfig(require_full_reset=False))


def test_solver_coupled_proxy_config_defaults() -> None:
    config = MuJoCoIpcConfig()
    assert config.coupling_iterations == 2
    assert config.relaxation_mode == "aitken"
    assert config.relaxation_factor == 1.0
    assert config.relaxation_min == 0.1
    assert config.relaxation_max == 1.0
    assert config.capture_coupler_graphs
    assert not MuJoCoIpcConfig(capture_coupler_graphs=False).capture_coupler_graphs
    assert MuJoCoIpcConfig(coupling_iterations=1).relaxation_mode == "fixed"
    _raises(
        ValueError,
        lambda: MuJoCoIpcConfig(relaxation_factor=0.05),
    )
    _raises(
        ValueError,
        lambda: MuJoCoIpcConfig(relaxation_mode="unknown"),
    )


def test_solver_coupled_proxy_rewinds_each_iteration_and_transfers_origin_twist() -> None:
    artifact = _artifact()
    rigid = _FakeNewtonRigidSolver(artifact, 2)
    ipc = _FakeNewtonIpcSolver(artifact, 2)
    actions = torch.full((2, 1), 0.25)
    provider = MuJoCoIpcProvider(
        artifact,
        config=MuJoCoIpcConfig(
            num_envs=2,
            device="cpu",
            environments_per_shard=2,
            coupling_iterations=3,
            relaxation_mode="aitken",
            capture_mujoco_graphs=False,
        ),
        rigid_solver=rigid,
        ipc_solver=ipc,
    )
    rigid.arrays["xfrc_applied"][:, 1, 1] = 7.0

    provider.step(actions)

    assert rigid.step_nsteps == [1, 1, 1]
    assert ipc.step_nsteps == [1, 1, 1]
    assert rigid.step_count == 1
    assert ipc.step_count == 1
    assert ipc.capture_count == 1
    assert ipc.rewind_count == 2
    assert ipc.commit_count == 1
    assert torch.equal(rigid.last_actions, actions)
    expected_twist = torch.tensor(
        [3.0, 0.0, 0.0, 0.0, 0.0, 2.0], dtype=torch.float64
    )
    assert torch.allclose(
        ipc.arrays["affine_target_twists"],
        expected_twist.expand_as(ipc.arrays["affine_target_twists"]),
    )
    assert torch.count_nonzero(rigid.arrays["xfrc_applied"][:, 0]) == 0
    assert torch.allclose(
        rigid.arrays["xfrc_applied"][:, 1, 0],
        torch.full((2,), 2.0 / 3.0),
        atol=1.0e-5,
        rtol=0.0,
    )
    assert torch.allclose(
        rigid.arrays["xfrc_applied"][:, 1, 1],
        torch.full((2,), 7.0),
    )
    assert provider.diagnostics["actual_coupling_iterations"] == 3
    assert provider.diagnostics["interface_residual"] < 1.0e-5
    assert (
        provider.diagnostics["interface_residual_l2"]
        >= provider.diagnostics["interface_residual"]
    )
    assert abs(provider.diagnostics["aitken_coefficient"] - 2.0 / 3.0) < 1.0e-5
    assert provider.diagnostics["proxy_count"] == 2
    provider.close()


def test_solver_coupled_proxy_residual_does_not_increase_with_more_iterations() -> None:
    artifact = _artifact()
    residuals = []
    for coupling_iterations in (1, 2, 4):
        rigid = _FakeNewtonRigidSolver(artifact, 2)
        ipc = _FakeNewtonIpcSolver(artifact, 2)
        provider = MuJoCoIpcProvider(
            artifact,
            config=MuJoCoIpcConfig(
                num_envs=2,
                device="cpu",
                environments_per_shard=2,
                coupling_iterations=coupling_iterations,
                relaxation_mode="aitken",
                capture_mujoco_graphs=False,
            ),
            rigid_solver=rigid,
            ipc_solver=ipc,
        )
        provider.step()
        residuals.append(provider.diagnostics["interface_residual"])
        assert rigid.step_count == 1
        assert ipc.step_count == 1
        provider.close()

    assert residuals[1] <= residuals[0]
    assert residuals[2] <= residuals[1]


def test_solver_coupled_proxy_no_gravity_impulse_balance_is_below_one_percent() -> None:
    artifact = _artifact()
    rigid = _FakeImpulseRigidSolver(artifact, 2)
    ipc = _FakeImpulseIpcSolver(artifact, 2)
    provider = MuJoCoIpcProvider(
        artifact,
        config=MuJoCoIpcConfig(
            num_envs=2,
            device="cpu",
            environments_per_shard=2,
            coupling_iterations=4,
            relaxation_mode="aitken",
            capture_mujoco_graphs=False,
        ),
        rigid_solver=rigid,
        ipc_solver=ipc,
    )

    provider.step()

    rigid_impulse = rigid.arrays["qvel"][:, 0]
    deformable_impulse = ipc.arrays["velocities"][..., 0].sum(dim=1)
    imbalance = (rigid_impulse + deformable_impulse).abs() / torch.maximum(
        rigid_impulse.abs(), deformable_impulse.abs()
    ).clamp_min(torch.finfo(rigid_impulse.dtype).eps)
    assert imbalance.max().item() < 0.01
    assert rigid.step_count == 1
    assert ipc.step_count == 1
    provider.close()


def test_aitken_relaxation_is_per_body_bounded_and_masks_one_way() -> None:
    artifact = _artifact()
    rigid = _FakeNewtonRigidSolver(artifact, 2)
    ipc = _FakeNewtonIpcSolver(artifact, 2)
    coupler = SolverCoupledProxy(
        rigid,
        ipc,
        artifact.coupled_bodies,
        coupling_iterations=2,
        relaxation_mode="aitken",
        relaxation_factor=0.5,
        relaxation_min=0.1,
        relaxation_max=0.8,
    )
    coupler._previous_residual.zero_()
    coupler._previous_residual[..., 0] = 1.0
    coupler._aitken_factors.fill_(0.5)
    coupler._iteration_guess.zero_()
    coupler._iteration_feedback.zero_()
    coupler._iteration_feedback[..., 0] = 0.9
    coupler._update_relaxed_wrench(1)
    assert torch.allclose(
        coupler._aitken_factors,
        torch.full_like(coupler._aitken_factors, 0.8),
    )
    assert torch.count_nonzero(coupler._next_wrenches[:, 0]) == 0

    coupler._previous_residual.zero_()
    coupler._previous_residual[..., 0] = 1.0
    coupler._aitken_factors.fill_(0.5)
    coupler._iteration_feedback.zero_()
    coupler._iteration_feedback[..., 0] = -100.0
    coupler._update_relaxed_wrench(1)
    assert torch.allclose(
        coupler._aitken_factors,
        torch.full_like(coupler._aitken_factors, 0.1),
    )

    coupler._relaxation_mode = "fixed"
    coupler._relaxation_factor = 0.25
    coupler._iteration_guess.zero_()
    coupler._iteration_feedback.zero_()
    coupler._iteration_feedback[..., 0] = 4.0
    coupler._update_relaxed_wrench(3)
    assert torch.allclose(
        coupler._next_wrenches[:, 1, 0],
        torch.ones_like(coupler._next_wrenches[:, 1, 0]),
    )
    assert torch.count_nonzero(coupler._next_wrenches[:, 0]) == 0


def test_newton_checkpoint_failure_faults_until_full_reset() -> None:
    artifact = _artifact()
    rigid = _FakeNewtonRigidSolver(artifact, 2)
    ipc = _FakeNewtonIpcSolver(artifact, 2)
    provider = MuJoCoIpcProvider(
        artifact,
        config=MuJoCoIpcConfig(
            num_envs=2,
            device="cpu",
            environments_per_shard=2,
            coupling_iterations=2,
            capture_mujoco_graphs=False,
        ),
        rigid_solver=rigid,
        ipc_solver=ipc,
    )
    ipc.fail_next_rewind = True
    _raises(RuntimeError, lambda: provider.step())
    assert provider.diagnostics["faulted"] is True
    assert provider.diagnostics["coupler_phase"] == "Faulted"
    _raises(RuntimeError, lambda: provider.step())
    provider.reset(torch.ones(2, dtype=torch.bool))
    assert provider.diagnostics["faulted"] is False
    provider.step()
    provider.close()


def test_composite_supports_explicit_solver_subcycling() -> None:
    artifact = _artifact()
    rigid = _FakeRigidSolver(artifact, 4)
    ipc = _FakeIpcSolver(artifact, 4)
    provider = MuJoCoIpcProvider(
        artifact,
        config=MuJoCoIpcConfig(
            num_envs=4,
            device="cpu",
            environments_per_shard=2,
            rigid_substeps=2,
            ipc_substeps=2,
            coupling_iterations=1,
        ),
        rigid_solver=rigid,
        ipc_solver=ipc,
    )

    provider.step()

    assert provider.fixed_time_step == 0.02
    assert rigid.step_nsteps == [1, 1]
    assert ipc.step_nsteps == [1, 1]
    assert provider.capabilities.exact_contact_wrench is True
    assert provider.capabilities.sensor_batch is True
    assert provider.capabilities.solver_substeps is True
    assert provider.capabilities.runtime_checkpoint is False
    assert provider.capabilities.reset_scope == "full_batch_only"
    assert provider.diagnostics["coupling_solver"] == "SolverCoupledProxy"
    assert provider.diagnostics["rollback_enabled"] is False
    assert provider.diagnostics["rigid_substeps"] == 2
    assert provider.diagnostics["ipc_substeps"] == 2
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
    test_ipc_schema_v3_normalizes_missing_static_colliders()
    test_ipc_schema_v4_remains_readable()
    test_libuipc_batch_solver_owns_stable_tensor_storage()
    test_libuipc_batch_lazy_contact_force_refresh()
    test_libuipc_al_ipc_config_reports_approximate_proxy_feedback()
    test_mujoco_ipc_step_order_wrench_ownership_and_full_reset()
    test_solver_coupled_proxy_x1_scales_storage_and_skips_checkpoints()
    test_solver_coupled_proxy_overrides_one_proxy_target_twist()
    test_step_failure_releases_owned_wrench_and_full_reset_recovers()
    test_mujoco_ipc_pose_error_feedback_uses_proxy_displacement()
    test_mujoco_ipc_config_requires_matching_solver_substeps()
    test_solver_coupled_proxy_config_defaults()
    test_solver_coupled_proxy_rewinds_each_iteration_and_transfers_origin_twist()
    test_solver_coupled_proxy_residual_does_not_increase_with_more_iterations()
    test_solver_coupled_proxy_no_gravity_impulse_balance_is_below_one_percent()
    test_aitken_relaxation_is_per_body_bounded_and_masks_one_way()
    test_newton_checkpoint_failure_faults_until_full_reset()
    test_composite_supports_explicit_solver_subcycling()
    test_composite_rejects_time_step_and_layout_mismatch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
