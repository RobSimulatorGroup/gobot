from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile

import pytest


if os.environ.get("GOBOT_RUN_LIBUIPC_BATCH_GPU_TEST") != "1":
    raise SystemExit(77)

import torch

import gobot
from gobot.ipc import LibuipcBatchConfig, LibuipcBatchSolver, LibuipcConfig
from gobot.rl import (
    CompiledMuJoCoIpcArtifact,
    MuJoCoIpcConfig,
    MuJoCoIpcProvider,
)

from libuipc_test_scenes import build_libuipc_test_scene


MODULE_PATH = os.environ.get("GOBOT_LIBUIPC_TEST_MODULE_PATH", "")
ROOT = Path(__file__).resolve().parents[2]
MUJOCO_LIBUIPC_EXAMPLE = ROOT / "examples" / "mujoco_libuipc"


def _load_mujoco_libuipc_play_script():
    spec = importlib.util.spec_from_file_location(
        "gobot_mujoco_libuipc_play_gpu_test",
        MUJOCO_LIBUIPC_EXAMPLE / "mujoco_libuipc_play.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_libuipc_batch_device_buffers() -> None:
    with tempfile.TemporaryDirectory(
        prefix="gobot-libuipc-batch-gpu-"
    ) as temporary_directory:
        project_root = Path(temporary_directory)
        scene_path = build_libuipc_test_scene(
            project_root, "soft_cube_press.jscn"
        )
        context = gobot.app.create_context()
        context.set_project_path(str(project_root))
        context.load_scene("res://" + scene_path.name)
        artifact = context.compile_ipc_scene_artifact()
        solver = LibuipcBatchSolver(
            artifact,
            num_envs=2,
            device="cuda:0",
            config=LibuipcBatchConfig(
                solver=LibuipcConfig(
                    fixed_time_step=0.01,
                    module_path=MODULE_PATH,
                    workspace=str(project_root / "workspace"),
                ),
                environments_per_shard=2,
                export_deformable_contact_forces=False,
            ),
        )
        try:
            assert solver.capabilities.device_native
            assert solver.shard_count == 1
            assert solver.diagnostics["valid"]
            device_native_coupling = os.environ.get(
                "GOBOT_LIBUIPC_DEVICE_NATIVE_COUPLING", "1"
            ).strip().lower() not in ("0", "false", "off")
            assert solver.diagnostics["device_native_coupling"] == (
                device_native_coupling
            )
            assert solver.diagnostics["cuda_stream_interop"]
            if device_native_coupling:
                assert (
                    solver.diagnostics["device_workspace_allocation_count"]
                    >= 1
                )
            else:
                assert (
                    solver.diagnostics["device_workspace_allocation_count"]
                    == 0
                )
            assert solver.diagnostics["affine_target_staging"] == (
                "per_shard_device_to_device"
                if device_native_coupling
                else "per_shard_device_host_device"
            )
            assert not solver.diagnostics[
                "export_deformable_contact_forces"
            ]
            positions = solver.arrays["positions"]
            transforms = solver.arrays["affine_transforms"]
            target_twists = solver.arrays["affine_target_twists"]
            assert positions.is_cuda and transforms.is_cuda
            assert target_twists.is_cuda
            assert positions.dtype == torch.float64
            assert target_twists.shape == (2, len(solver.affine_bodies), 6)
            assert torch.count_nonzero(target_twists) == 0
            assert torch.isfinite(positions).all().item()
            assert torch.isfinite(transforms).all().item()
            torch.testing.assert_close(
                transforms[..., 3, :],
                torch.tensor(
                    [0.0, 0.0, 0.0, 1.0],
                    dtype=transforms.dtype,
                    device=transforms.device,
                ).expand_as(transforms[..., 3, :]),
            )

            target_storage = solver.arrays["affine_targets"]
            initial_positions = positions.clone()
            initial_transforms = transforms.clone()
            press_index = next(
                index
                for index, body in enumerate(solver.affine_bodies)
                if str(body["path"]).endswith("/press_head")
            )
            position_pointer = positions.data_ptr()
            transform_pointer = transforms.data_ptr()
            twist_pointer = target_twists.data_ptr()

            caller_stream = torch.cuda.Stream(device="cuda:0")
            with torch.cuda.stream(caller_stream):
                target_storage.copy_(transforms)
                target_storage[0, press_index, 2, 3] -= 0.01
                target_twists[0, press_index, 2] = -1.0
                target_twists[0, press_index, 5] = 25.0
                solver.capture_checkpoint()
            assert solver.diagnostics["checkpoint_active"]

            solver.step()

            assert positions.data_ptr() == position_pointer
            assert transforms.data_ptr() == transform_pointer
            assert target_twists.data_ptr() == twist_pointer
            assert torch.isfinite(positions).all().item()
            assert torch.isfinite(transforms).all().item()
            assert (
                transforms[1, press_index, 2, 3]
                - transforms[0, press_index, 2, 3]
            ).item() > 0.005
            assert solver.diagnostics["frame"] == 1
            assert solver.diagnostics[
                "deformable_contact_force_frame"
            ] == 0
            contact_force_pointer = solver.arrays[
                "contact_forces"
            ].data_ptr()
            solver.refresh_deformable_contact_forces()
            assert solver.arrays["contact_forces"].data_ptr() == contact_force_pointer
            assert solver.diagnostics[
                "deformable_contact_force_frame"
            ] == 1
            first_step_contact_forces = solver.arrays[
                "contact_forces"
            ].clone()
            first_step_positions = positions.clone()
            first_step_transforms = transforms.clone()
            first_step_wrenches = solver.arrays[
                "affine_contact_wrenches"
            ].clone()

            solver.rewind_checkpoint()
            assert solver.diagnostics["frame"] == 0
            assert solver.diagnostics["checkpoint_active"]
            torch.testing.assert_close(positions, initial_positions)
            torch.testing.assert_close(transforms, initial_transforms)

            solver.step()
            assert solver.diagnostics[
                "deformable_contact_force_frame"
            ] == 0
            solver.refresh_deformable_contact_forces()
            torch.testing.assert_close(positions, first_step_positions)
            torch.testing.assert_close(transforms, first_step_transforms)
            torch.testing.assert_close(
                solver.arrays["affine_contact_wrenches"],
                first_step_wrenches,
            )
            torch.testing.assert_close(
                solver.arrays["contact_forces"],
                first_step_contact_forces,
            )
            solver.commit_checkpoint()
            assert not solver.diagnostics["checkpoint_active"]
            assert solver.diagnostics["frame"] == 1

            solver.reset(torch.ones(2, dtype=torch.bool, device="cuda:0"))
            assert solver.diagnostics["frame"] == 0
            assert positions.data_ptr() == position_pointer
            torch.testing.assert_close(positions, initial_positions)
            torch.testing.assert_close(transforms, initial_transforms)

            solver.step()
            torch.testing.assert_close(positions, first_step_positions)
            torch.testing.assert_close(transforms, first_step_transforms)

            # Runtime flags are independent: requesting only deformable contact
            # forces must not modify the affine-wrench output buffer.
            affine_wrenches = solver.arrays["affine_contact_wrenches"]
            affine_wrenches.fill_(123.0)
            solver._session.set_output_flags(1 << 2)
            solver.step()
            solver.synchronize()
            assert torch.all(affine_wrenches == 123.0).item()

            solver._session.set_output_flags(0)
            solver.step()
            assert solver.diagnostics["frame"] == 3

            # A non-finite external target must invalidate the native world
            # and cross the C ABI as an exception, never abort this process.
            target_storage[0, press_index, 0, 3] = torch.nan
            with pytest.raises(
                RuntimeError, match="libuipc batch world became invalid"
            ):
                solver.step()
            assert not solver.diagnostics["valid"]
        finally:
            solver.close()
            assert not tuple((project_root / "workspace").glob("batch_session_*"))
            del context


def test_real_libuipc_strict_failure_rewinds_memory_checkpoint() -> None:
    with tempfile.TemporaryDirectory(
        prefix="gobot-libuipc-strict-recovery-gpu-"
    ) as temporary_directory:
        project_root = Path(temporary_directory)
        scene_path = build_libuipc_test_scene(
            project_root, "soft_cube_press.jscn"
        )
        context = gobot.app.create_context()
        context.set_project_path(str(project_root))
        context.load_scene("res://" + scene_path.name)
        artifact = context.compile_ipc_scene_artifact()
        solver = LibuipcBatchSolver(
            artifact,
            num_envs=1,
            device="cuda:0",
            config=LibuipcBatchConfig(
                solver=LibuipcConfig(
                    fixed_time_step=0.01,
                    module_path=MODULE_PATH,
                    workspace=str(project_root / "workspace"),
                ),
                environments_per_shard=1,
                export_deformable_contact_forces=False,
            ),
        )
        try:
            transforms = solver.arrays["affine_transforms"]
            targets = solver.arrays["affine_targets"]
            initial_positions = solver.arrays["positions"].clone()
            initial_transforms = transforms.clone()
            press_index = next(
                index
                for index, body in enumerate(solver.affine_bodies)
                if str(body["path"]).endswith("/press_head")
            )
            targets.copy_(transforms)
            targets[0, press_index, 2, 3] -= 0.03
            solver.capture_checkpoint()
            solver.set_runtime_solver_options(
                newton_max_iterations=1,
                line_search_max_iterations=1,
                linear_system_tolerance_rate=1.0e-3,
                strict_convergence=True,
            )

            with pytest.raises(
                RuntimeError, match="libuipc batch world became invalid"
            ):
                solver.step()
            failed = solver.diagnostics
            assert not failed["valid"]
            assert failed["strict_convergence"]
            assert failed["solver_failure"] != 0
            assert failed["solver_failure_message"]

            solver.rewind_checkpoint()
            recovered = solver.diagnostics
            assert recovered["valid"]
            assert recovered["recovered"]
            assert recovered["frame"] == 0
            torch.testing.assert_close(
                solver.arrays["positions"], initial_positions
            )
            torch.testing.assert_close(transforms, initial_transforms)

            solver.set_runtime_solver_options(
                newton_max_iterations=32,
                line_search_max_iterations=16,
                linear_system_tolerance_rate=2.5e-4,
                strict_convergence=True,
            )
            solver.step()
            assert solver.diagnostics["valid"]
            assert solver.diagnostics["frame"] == 1
            solver.commit_checkpoint()
        finally:
            solver.close()
            del context


def test_real_libuipc_loose_static_box_collision() -> None:
    with tempfile.TemporaryDirectory(
        prefix="gobot-libuipc-static-gpu-"
    ) as temporary_directory:
        project_root = Path(temporary_directory)
        scene_path = build_libuipc_test_scene(
            project_root, "soft_cube_static_box.jscn"
        )
        context = gobot.app.create_context()
        context.set_project_path(str(project_root))
        context.load_scene("res://" + scene_path.name)
        artifact = context.compile_ipc_scene_artifact()
        solver = LibuipcBatchSolver(
            artifact,
            num_envs=1,
            device="cuda:0",
            config=LibuipcBatchConfig(
                solver=LibuipcConfig(
                    fixed_time_step=0.005,
                    module_path=MODULE_PATH,
                    workspace=str(project_root / "workspace"),
                ),
                environments_per_shard=1,
            ),
        )
        try:
            assert solver.capacities["affine_bodies_per_env"] == 0
            assert solver.capacities["static_colliders_per_env"] == 1
            assert solver.diagnostics["static_collider_count"] == 1
            for _ in range(80):
                solver.step()
            solver.synchronize()
            positions = solver.arrays["positions"]
            assert torch.isfinite(positions).all().item()
            assert positions[..., 2].amin().item() > -0.015
        finally:
            solver.close()
            del context


def test_real_libuipc_affine_output_excludes_static_colliders() -> None:
    with tempfile.TemporaryDirectory(
        prefix="gobot-libuipc-affine-static-gpu-"
    ) as temporary_directory:
        project_root = Path(temporary_directory)
        scene_path = build_libuipc_test_scene(
            project_root, "soft_cube_press_static_box.jscn"
        )
        context = gobot.app.create_context()
        context.set_project_path(str(project_root))
        context.load_scene("res://" + scene_path.name)
        artifact = context.compile_ipc_scene_artifact()
        solver = LibuipcBatchSolver(
            artifact,
            num_envs=1,
            device="cuda:0",
            config=LibuipcBatchConfig(
                solver=LibuipcConfig(
                    fixed_time_step=0.005,
                    module_path=MODULE_PATH,
                    workspace=str(project_root / "workspace"),
                ),
                environments_per_shard=1,
            ),
        )
        try:
            assert solver.capacities["affine_bodies_per_env"] == 2
            assert solver.capacities["static_colliders_per_env"] == 1
            positions = solver.arrays["positions"]
            transforms = solver.arrays["affine_transforms"]
            assert transforms.shape == (1, 2, 4, 4)
            torch.testing.assert_close(
                positions[0, 0],
                torch.tensor(
                    (-0.11, -0.11, 0.02),
                    dtype=positions.dtype,
                    device=positions.device,
                ),
            )
            torch.testing.assert_close(
                positions[0, -1],
                torch.tensor(
                    (0.11, 0.11, 0.18),
                    dtype=positions.dtype,
                    device=positions.device,
                ),
            )
        finally:
            solver.close()
            del context


def test_real_mujoco_libuipc_composite_step() -> None:
    with tempfile.TemporaryDirectory(
        prefix="gobot-mujoco-libuipc-gpu-"
    ) as temporary_directory:
        project_root = Path(temporary_directory)
        scene_path = build_libuipc_test_scene(
            project_root, "soft_cube_press.jscn"
        )
        context = gobot.app.create_context()
        context.set_project_path(str(project_root))
        context.load_scene("res://" + scene_path.name)
        artifact = CompiledMuJoCoIpcArtifact.from_context(context)
        provider = MuJoCoIpcProvider(
            artifact,
            config=MuJoCoIpcConfig(
                num_envs=2,
                device="cuda:0",
                environments_per_shard=2,
                capture_mujoco_graphs=True,
            ),
            libuipc_config=LibuipcBatchConfig(
                solver=LibuipcConfig(
                    fixed_time_step=0.002,
                    module_path=MODULE_PATH,
                    workspace=str(project_root / "workspace"),
                ),
                environments_per_shard=2,
            ),
        )
        try:
            assert provider.capabilities.device_native
            assert not provider.capabilities.graph_capture
            assert not provider.capabilities.masked_reset
            assert provider.rigid_solver.capabilities.graph_capture
            qpos_pointer = provider.arrays["qpos"].data_ptr()
            ipc_position_pointer = provider.arrays["ipc_positions"].data_ptr()

            provider.step()

            assert provider.arrays["qpos"].data_ptr() == qpos_pointer
            assert (
                provider.arrays["ipc_positions"].data_ptr()
                == ipc_position_pointer
            )
            assert torch.isfinite(provider.arrays["qpos"]).all().item()
            assert torch.isfinite(provider.arrays["xfrc_applied"]).all().item()
            assert torch.isfinite(provider.arrays["ipc_positions"]).all().item()
            assert torch.all(provider.arrays["time"] > 0.0).item()
            assert provider.diagnostics["frame"] == 1
            assert provider.diagnostics["libuipc"]["frame"] == 1

            provider.reset(
                torch.ones(2, dtype=torch.bool, device="cuda:0")
            )
            assert provider.diagnostics["frame"] == 0
            assert provider.arrays["qpos"].data_ptr() == qpos_pointer
        finally:
            provider.close()
            assert not tuple((project_root / "workspace").glob("batch_session_*"))
            del context


def test_real_mujoco_libuipc_solver_coupled_proxy_step() -> None:
    with tempfile.TemporaryDirectory(
        prefix="gobot-mujoco-libuipc-newton-gpu-"
    ) as temporary_directory:
        project_root = Path(temporary_directory)
        scene_path = build_libuipc_test_scene(
            project_root, "soft_cube_press.jscn"
        )
        context = gobot.app.create_context()
        context.set_project_path(str(project_root))
        context.load_scene("res://" + scene_path.name)
        artifact = CompiledMuJoCoIpcArtifact.from_context(context)
        def make_provider(
            *,
            capture_coupler_graphs: bool,
            export_deformable_contact_forces: bool = True,
            export_state: bool = True,
        ):
            mode = "captured" if capture_coupler_graphs else "eager"
            output = (
                "immediate" if export_deformable_contact_forces else "lazy"
            )
            return MuJoCoIpcProvider(
                artifact,
                config=MuJoCoIpcConfig(
                    num_envs=1,
                    device="cuda:0",
                    environments_per_shard=1,
                    coupling_iterations=2,
                    capture_mujoco_graphs=True,
                    capture_coupler_graphs=capture_coupler_graphs,
                ),
                libuipc_config=LibuipcBatchConfig(
                    solver=LibuipcConfig(
                        fixed_time_step=0.002,
                        module_path=MODULE_PATH,
                        workspace=str(
                            project_root / f"workspace_{mode}_{output}"
                        ),
                    ),
                    environments_per_shard=1,
                    export_deformable_state=export_state,
                    export_affine_state=export_state,
                    export_deformable_contact_forces=(
                        export_deformable_contact_forces
                    ),
                ),
            )

        provider = make_provider(capture_coupler_graphs=True)
        try:
            comparison_steps = 8
            initial_time = provider.arrays["time"].clone()
            provider.step(nsteps=comparison_steps)
            provider.synchronize()

            diagnostics = provider.diagnostics
            assert diagnostics["frame"] == comparison_steps
            assert diagnostics["libuipc"]["frame"] == comparison_steps
            assert diagnostics["actual_coupling_iterations"] == 2
            assert diagnostics["coupler_phase"] == "Idle"
            assert diagnostics["libuipc"]["checkpoint_active"] is False
            assert diagnostics["coupler_graph_captured"], diagnostics[
                "coupler_graph_capture_reason"
            ]
            assert diagnostics["phase_latency_ms"]
            assert torch.allclose(
                provider.arrays["time"],
                initial_time + comparison_steps * provider.fixed_time_step,
            )
            assert torch.isfinite(
                provider.arrays["ipc_affine_target_twists"]
            ).all().item()
            assert torch.isfinite(provider.arrays["xfrc_applied"]).all().item()
            captured_state = {
                name: provider.arrays[name].clone()
                for name in (
                    "qpos",
                    "qvel",
                    "xfrc_applied",
                    "ipc_positions",
                    "ipc_affine_transforms",
                    "ipc_affine_contact_wrenches",
                )
            }
        finally:
            provider.close()
            assert not tuple(
                (project_root / "workspace_captured_immediate").glob(
                    "batch_session_*"
                )
            )

        eager_provider = make_provider(capture_coupler_graphs=False)
        try:
            eager_provider.step(nsteps=comparison_steps)
            eager_provider.synchronize()
            eager_diagnostics = eager_provider.diagnostics
            assert not eager_diagnostics["coupler_graph_captured"]
            assert "disabled" in eager_diagnostics[
                "coupler_graph_capture_reason"
            ]
            for name, captured in captured_state.items():
                torch.testing.assert_close(
                    eager_provider.arrays[name], captured, rtol=0.0, atol=0.0
                )
        finally:
            eager_provider.close()
            assert not tuple(
                (project_root / "workspace_eager_immediate").glob(
                    "batch_session_*"
                )
            )

        lazy_provider = make_provider(
            capture_coupler_graphs=True,
            export_deformable_contact_forces=False,
            export_state=False,
        )
        try:
            lazy_provider.step(nsteps=comparison_steps)
            lazy_provider.refresh_state()
            lazy_provider.synchronize()
            assert not lazy_provider.diagnostics["libuipc"][
                "export_deformable_contact_forces"
            ]
            assert not lazy_provider.diagnostics["libuipc"]["export_deformable_state"]
            assert not lazy_provider.diagnostics["libuipc"]["export_affine_state"]
            for name, captured in captured_state.items():
                torch.testing.assert_close(
                    lazy_provider.arrays[name], captured, rtol=0.0, atol=0.0
                )
        finally:
            lazy_provider.close()
            assert not tuple(
                (project_root / "workspace_captured_lazy").glob(
                    "batch_session_*"
                )
            )
            del context


def test_mujoco_libuipc_editor_play_session() -> None:
    previous_module = os.environ.get("GOBOT_LIBUIPC_SOLVER_MODULE")
    if MODULE_PATH:
        os.environ["GOBOT_LIBUIPC_SOLVER_MODULE"] = MODULE_PATH
    context = gobot.app.create_context()
    script = None
    try:
        module = _load_mujoco_libuipc_play_script()
        context.set_project_path(str(MUJOCO_LIBUIPC_EXAMPLE))
        root = context.load_scene("res://soft_press_batch.jscn")
        script = module.Script()
        script._attach(root, root, context)
        script._ready()
        context.step(180)
        script._sync_scene()

        state = script.press_view.read_state()
        positions = script.provider.arrays["ipc_positions"]
        heights = positions[..., 2].amax(dim=1) - positions[..., 2].amin(dim=1)
        assert script.provider.num_envs == module.NUM_ENVS == 4
        assert len(script.display_roots) == 4
        assert len(script.display_deformable_bodies) == 4
        grid_positions = {
            tuple(round(float(value), 6) for value in root.position)
            for root in script.display_roots
        }
        assert len(grid_positions) == 4
        scripts = tuple(root.get_property("script") for root in script.display_roots)
        assert bool(scripts[0]) and all(not value for value in scripts[1:])
        assert script.tick == 180
        joint_positions = state.joint_position[:, 0]
        assert joint_positions[-1].item() < -0.1
        assert torch.all(joint_positions[1:] < joint_positions[:-1]).item()
        compression = 0.16 - heights
        assert compression[-1].item() > 0.005
        assert (compression.max() - compression.min()).item() > 0.001
        mappings = {
            mapping.link_name: mapping
            for mapping in script.provider.artifact.coupled_bodies
        }
        ground_id, press_id = script.provider.rigid_solver.resolve_object_ids(
            "body",
            (
                mappings["ground"].mujoco_body_name,
                mappings["press_head"].mujoco_body_name,
            ),
        )
        applied = script.provider.arrays["xfrc_applied"]
        assert torch.count_nonzero(applied[:, ground_id]) == 0
        assert torch.isfinite(applied[:, press_id]).all().item()
        assert torch.linalg.vector_norm(applied[:, press_id, :3]).max().item() > 0.0
        assert script.provider.rigid_solver.capabilities.graph_capture
        assert (
            script.provider.diagnostics["feedback_source"]
            == "native_contact_wrench"
        )
        assert script.provider.capabilities.exact_contact_wrench
        assert script.provider.capabilities.reset_scope == "full_batch_only"
        assert script.provider.diagnostics["graph_captured"] is False
    finally:
        if script is not None:
            script._exit_tree()
        context.clear_scene()
        if previous_module is None:
            os.environ.pop("GOBOT_LIBUIPC_SOLVER_MODULE", None)
        else:
            os.environ["GOBOT_LIBUIPC_SOLVER_MODULE"] = previous_module


def main() -> int:
    availability = LibuipcBatchSolver.availability(
        LibuipcBatchConfig(
            solver=LibuipcConfig(module_path=MODULE_PATH),
            environments_per_shard=2,
        )
    )
    assert availability.available, availability.reason
    test_real_libuipc_batch_device_buffers()
    test_real_libuipc_loose_static_box_collision()
    test_real_libuipc_affine_output_excludes_static_colliders()
    test_real_mujoco_libuipc_composite_step()
    test_real_mujoco_libuipc_solver_coupled_proxy_step()
    test_mujoco_libuipc_editor_play_session()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
