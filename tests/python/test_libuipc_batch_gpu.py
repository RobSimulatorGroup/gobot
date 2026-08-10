from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile


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
            ),
        )
        try:
            assert solver.capabilities.device_native
            assert solver.shard_count == 1
            assert solver.diagnostics["valid"]
            positions = solver.arrays["positions"]
            transforms = solver.arrays["affine_transforms"]
            assert positions.is_cuda and transforms.is_cuda
            assert positions.dtype == torch.float64
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
            target_storage.copy_(transforms)
            initial_positions = positions.clone()
            initial_transforms = transforms.clone()
            press_index = next(
                index
                for index, body in enumerate(solver.affine_bodies)
                if str(body["path"]).endswith("/press_head")
            )
            target_storage[0, press_index, 2, 3] -= 0.01
            position_pointer = positions.data_ptr()
            transform_pointer = transforms.data_ptr()

            solver.step()

            assert positions.data_ptr() == position_pointer
            assert transforms.data_ptr() == transform_pointer
            assert torch.isfinite(positions).all().item()
            assert torch.isfinite(transforms).all().item()
            assert (
                transforms[1, press_index, 2, 3]
                - transforms[0, press_index, 2, 3]
            ).item() > 0.005
            assert solver.diagnostics["frame"] == 1
            first_step_positions = positions.clone()
            first_step_transforms = transforms.clone()

            solver.reset(torch.ones(2, dtype=torch.bool, device="cuda:0"))
            assert solver.diagnostics["frame"] == 0
            assert positions.data_ptr() == position_pointer
            torch.testing.assert_close(positions, initial_positions)
            torch.testing.assert_close(transforms, initial_transforms)

            solver.step()
            torch.testing.assert_close(positions, first_step_positions)
            torch.testing.assert_close(transforms, first_step_transforms)
        finally:
            solver.close()
            assert not tuple((project_root / "workspace").glob("batch_session_*"))
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
        assert script.tick == 180
        assert state.joint_position[module.DISPLAY_ENV, 0].item() < -0.1
        assert (0.16 - heights[module.DISPLAY_ENV]).item() > 0.005
        assert script.provider.rigid_solver.capabilities.graph_capture
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
    test_real_mujoco_libuipc_composite_step()
    test_mujoco_libuipc_editor_play_session()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
