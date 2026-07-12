from __future__ import annotations

from pathlib import Path
import tempfile
import time

import gobot
import numpy as np


def _compiled_cartpole_artifact():
    context = gobot.app.create_context()
    temporary_directory = tempfile.TemporaryDirectory()
    project_path = Path(temporary_directory.name)
    context.set_project_path(str(project_path))
    root = gobot.scene.create_cartpole_scene(name="warp_cartpole")
    slider = root.find("rail/slider")
    hinge = root.find("rail/slider/cart/hinge")
    assert slider is not None and hinge is not None
    slider.drive_mode = gobot.JointDriveMode.Position
    hinge.drive_mode = gobot.JointDriveMode.Position
    gobot.save_scene(root, "res://warp_cartpole.jscn")
    context.load_scene("res://warp_cartpole.jscn")
    artifact = context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)
    assert context.has_world is False
    return context, temporary_directory, artifact


def main() -> None:
    context, temporary_directory, artifact = _compiled_cartpole_artifact()
    try:
        compiled = gobot.rl.CompiledSceneArtifact.from_mapping(artifact)
        assert compiled.format == "mjcf"
        assert compiled.robot_prefix("warp_cartpole") == "warp_cartpole_"
        assert compiled.dimensions["nq"] == 2
        assert compiled.content_digest == artifact["content_digest"]
        assert "<mujoco" in compiled.content

        availability = gobot.rl.MuJoCoWarpProvider.availability()
        if not availability.available:
            print(f"MuJoCo Warp optional smoke skipped: {availability.reason}")
            return

        import torch

        if not torch.cuda.is_available():
            print("MuJoCo Warp CUDA smoke skipped: Torch cannot access CUDA")
            return

        provider = gobot.rl.MuJoCoWarpProvider(
            artifact,
            num_envs=4,
            device="cuda:0",
            nconmax=64,
            njmax=128,
            overflow_check_interval=0,
        )
        try:
            arrays = provider.arrays
            qpos_pointer = arrays["qpos"].data_ptr()
            qvel_pointer = arrays["qvel"].data_ptr()
            ctrl_pointer = arrays["ctrl"].data_ptr()
            reset_mask_pointer = arrays["reset_mask"].data_ptr()
            initial_qpos = arrays["qpos"].clone()
            initial_qvel = arrays["qvel"].clone()
            initial_ctrl = arrays["ctrl"].clone()
            zero_actions = torch.zeros_like(arrays["ctrl"])
            layout = provider.resolve_robot_layout(
                "warp_cartpole",
                base_link="cart",
                joint_names=("slider", "hinge"),
                link_names=("cart", "pole"),
            )
            assert layout.actuator_modes == ("position", "position")
            provider.set_joint_position_targets(layout, zero_actions)

            provider.step(zero_actions, nsteps=2)
            provider.synchronize()
            provider.assert_finite()
            first_qpos = arrays["qpos"].clone()
            first_qvel = arrays["qvel"].clone()
            assert provider.arrays["qpos"].data_ptr() == qpos_pointer
            assert provider.arrays["qvel"].data_ptr() == qvel_pointer
            assert provider.arrays["ctrl"].data_ptr() == ctrl_pointer

            context.build_world(gobot.PhysicsBackendType.MuJoCoCpu)
            context.configure_batch_world(4)
            context.step_batch(2, workers=1)
            cpu_state = context.get_batch_robot_state(
                "warp_cartpole",
                "cart",
                ["slider", "hinge"],
            )
            np.testing.assert_allclose(
                cpu_state["joint_position"],
                first_qpos.cpu().numpy(),
                rtol=2.0e-4,
                atol=2.0e-5,
            )
            np.testing.assert_allclose(
                cpu_state["joint_velocity"],
                first_qvel.cpu().numpy(),
                rtol=2.0e-4,
                atol=2.0e-5,
            )

            all_envs = torch.ones(4, dtype=torch.bool, device="cuda:0")
            provider.reset(
                all_envs,
                qpos=initial_qpos,
                qvel=initial_qvel,
                ctrl=initial_ctrl,
            )
            provider.step(zero_actions, nsteps=2)
            provider.synchronize()
            torch.testing.assert_close(arrays["qpos"], first_qpos)
            torch.testing.assert_close(arrays["qvel"], first_qvel)

            before_reset = arrays["qpos"].clone()
            reset_qpos = before_reset.clone()
            reset_qpos[0, 0] = 0.25
            mask = torch.tensor([True, False, False, False], device="cuda:0")
            provider.reset(mask, qpos=reset_qpos)
            provider.synchronize()

            assert arrays["reset_mask"].data_ptr() == reset_mask_pointer
            assert not bool(arrays["reset_mask"].any())
            assert abs(float(arrays["qpos"][0, 0]) - 0.25) < 1.0e-5
            torch.testing.assert_close(arrays["qpos"][1:], before_reset[1:])
            assert provider.capabilities.graph_capture is True
            assert provider.capabilities.device_native is True

            provider.reset(mask, qpos=reset_qpos)
            provider.synchronize()
            reset_memory_before = torch.cuda.memory_allocated()
            for _ in range(10):
                provider.reset(mask, qpos=reset_qpos)
            provider.synchronize()
            assert torch.cuda.memory_allocated() == reset_memory_before

            provider.step(zero_actions)
            provider.synchronize()
            memory_before = torch.cuda.memory_allocated()
            start = time.perf_counter()
            for _ in range(100):
                provider.step(zero_actions)
            provider.synchronize()
            elapsed = time.perf_counter() - start
            memory_after = torch.cuda.memory_allocated()
            provider.assert_no_overflow()
            assert memory_after == memory_before
            assert elapsed > 0.0
            print(f"MuJoCo Warp smoke throughput: {400.0 / elapsed:.0f} env-steps/s")
        finally:
            provider.close()
    finally:
        context.clear_world()
        context.clear_scene()
        temporary_directory.cleanup()


if __name__ == "__main__":
    main()
