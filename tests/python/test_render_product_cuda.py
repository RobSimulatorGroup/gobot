import gc
import sys

try:
    import numpy as np
    import torch
except ImportError as error:
    print(f"CUDA render product test skipped: {error}")
    raise SystemExit(77)

try:
    import warp as wp
except ImportError:
    wp = None

import gobot


OUTPUTS = (
    "rgb",
    "linear_depth",
    "world_normal",
    "instance_id",
    "semantic_id",
)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA render product test skipped: Torch cannot access CUDA")
        return 77

    root = gobot.create_node("Node3D", "World")
    root.semantic_label = "crate"
    root.add_child(gobot.create_box_visual("box", (1.0, 1.0, 1.0)))
    camera = gobot.create_node("Camera3D", "camera")
    camera.set("eye", (0.0, -3.0, 0.0))
    camera.set("target", (0.0, 0.0, 0.0))
    camera.set("up", (0.0, 0.0, 1.0))
    camera.set("fov_y", 55.0)
    camera.set("z_near", 0.05)
    camera.set("z_far", 20.0)

    cpu_sensor = gobot.render.CameraSensor(
        camera,
        root=root,
        width=65,
        height=49,
        outputs=OUTPUTS,
        device="cpu",
        mode="minimal",
    )
    cuda_sensor = gobot.render.CameraSensor(
        camera,
        root=root,
        width=65,
        height=49,
        outputs=OUTPUTS,
        device="cuda",
        mode="minimal",
        frame_slots=1,
    )

    cpu_frame = cpu_sensor.capture()
    try:
        cuda_frame = cuda_sensor.capture()
    except RuntimeError as error:
        print(f"CUDA render product test skipped: {error}")
        return 77
    assert cuda_sensor.render_product.device == "cuda"

    cpu_depth = cpu_frame["linear_depth"].numpy()
    cpu_normal = cpu_frame["world_normal"].numpy()
    cpu_instance = cpu_frame["instance_id"].numpy()
    cuda_depth = cuda_frame["linear_depth"].numpy()
    cuda_normal = cuda_frame["world_normal"].numpy()
    cuda_instance = cuda_frame["instance_id"].numpy()
    cuda_semantic = cuda_frame["semantic_id"].numpy()

    assert abs(float(cuda_depth[24, 32]) - 2.5) < 1.0e-3
    np.testing.assert_allclose(cuda_normal[24, 32], (0.0, -1.0, 0.0), atol=1.0e-4)
    assert int(cuda_instance[24, 32]) != 0
    assert int(cuda_semantic[24, 32]) != 0
    assert np.isposinf(cuda_depth[0, 0])
    np.testing.assert_allclose(cuda_normal[0, 0], 0.0)
    assert int(cuda_instance[0, 0]) == 0
    np.testing.assert_allclose(cuda_depth[24, 32], cpu_depth[24, 32], atol=1.0e-3)
    np.testing.assert_allclose(cuda_normal[24, 32], cpu_normal[24, 32], atol=1.0e-4)
    assert int(cuda_instance[24, 32]) == int(cpu_instance[24, 32])
    assert set(cuda_frame.instance_id_to_path.values()) == {"box"}
    assert set(cuda_frame.semantic_id_to_label.values()) == {"crate"}

    torch_depth = torch.from_dlpack(cuda_frame["linear_depth"])
    torch_rgb = torch.from_dlpack(cuda_frame["rgb"])
    warp_depth = None
    warp_cuda_available = wp is not None and any(
        str(device).startswith("cuda") for device in wp.get_devices()
    )
    if warp_cuda_available:
        warp_depth = wp.from_dlpack(cuda_frame["linear_depth"])
    assert torch_depth.is_cuda and tuple(torch_depth.shape) == (49, 65)
    assert torch_rgb.is_cuda and tuple(torch_rgb.shape) == (49, 65, 3)
    assert tuple(torch_rgb.stride()) == (65 * 4, 4, 1)
    if warp_depth is not None:
        assert str(warp_depth.device).startswith("cuda")
        assert tuple(warp_depth.shape) == (49, 65)
    else:
        print("Warp CUDA DLPack check skipped: Warp cannot access a CUDA device")

    cuda_frame = None
    try:
        cuda_sensor.capture()
        raise AssertionError("DLPack consumers should retain the only CUDA frame slot")
    except RuntimeError as error:
        assert "frame pool exhausted" in str(error)

    del torch_depth, torch_rgb, warp_depth
    gc.collect()
    torch.cuda.synchronize()
    if warp_cuda_available:
        wp.synchronize()
    replacement = cuda_sensor.capture()
    assert replacement["linear_depth"].device == "cuda"
    shutdown_tensor = torch.from_dlpack(replacement["linear_depth"])

    auto_sensor = gobot.render.CameraSensor(
        camera,
        root=root,
        width=8,
        height=8,
        outputs=("linear_depth",),
        device="auto",
    )
    auto_sensor.capture()
    assert auto_sensor.render_product.device == "cuda"

    replacement = None
    cpu_frame = None
    cpu_sensor = None
    cuda_sensor = None
    auto_sensor = None
    gc.collect()
    gobot.render._shutdown_headless_render_context()
    assert abs(float(shutdown_tensor[24, 32]) - 2.5) < 1.0e-3
    del shutdown_tensor
    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
