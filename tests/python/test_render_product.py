import gc
import sys

import gobot
import numpy as np


OUTPUTS = (
    "rgb",
    "linear_depth",
    "world_normal",
    "instance_id",
    "semantic_id",
)


def main() -> int:
    root = gobot.create_node("Node3D", "World")
    root.semantic_label = "crate"
    box = gobot.create_box_visual("box", (1.0, 1.0, 1.0))
    root.add_child(box)

    camera = gobot.create_node("Camera3D", "camera")
    camera.set("eye", (0.0, -3.0, 0.0))
    camera.set("target", (0.0, 0.0, 0.0))
    camera.set("up", (0.0, 0.0, 1.0))
    camera.set("fov_y", 55.0)
    camera.set("z_near", 0.05)
    camera.set("z_far", 20.0)

    sensor = gobot.render.CameraSensor(
        camera,
        root=root,
        width=65,
        height=49,
        outputs=OUTPUTS,
        device="cpu",
        mode="minimal",
    )
    try:
        frame = sensor.capture()
    except RuntimeError as error:
        if "EGL" in str(error) or "egl" in str(error):
            print(f"render product test skipped: {error}")
            return 77
        raise

    assert set(frame.keys()) == set(OUTPUTS)
    rgb = frame["rgb"].numpy()
    depth = frame["linear_depth"].numpy()
    normal = frame["world_normal"].numpy()
    instance_id = frame["instance_id"].numpy()
    semantic_id = frame["semantic_id"].numpy()

    assert rgb.shape == (49, 65, 3) and rgb.dtype == np.uint8
    assert depth.shape == (49, 65) and depth.dtype == np.float32
    assert normal.shape == (49, 65, 3) and normal.dtype == np.float32
    assert instance_id.shape == (49, 65) and instance_id.dtype == np.uint32
    assert semantic_id.shape == (49, 65) and semantic_id.dtype == np.uint32
    assert not rgb.flags.writeable

    foreground = instance_id != 0
    background = ~foreground
    assert foreground.any()
    assert background.any()
    foreground_y, foreground_x = np.where(foreground)
    assert foreground[24, 32], (
        "center pixel missed rendered box; foreground bounds="
        f"x[{foreground_x.min()},{foreground_x.max()}] "
        f"y[{foreground_y.min()},{foreground_y.max()}]"
    )
    assert np.isfinite(depth[foreground]).all()
    assert (depth[foreground] > 0.0).all()
    assert np.isposinf(depth[background]).all()
    np.testing.assert_allclose(normal[background], 0.0)
    np.testing.assert_allclose(
        np.linalg.norm(normal[foreground], axis=1),
        1.0,
        rtol=1.0e-4,
        atol=1.0e-4,
    )
    center_depth = float(depth[24, 32])
    assert abs(center_depth - 2.5) < 1.0e-3, f"unexpected center depth: {center_depth}"
    np.testing.assert_allclose(normal[24, 32], (0.0, -1.0, 0.0), atol=1.0e-4)
    assert (semantic_id[foreground] != 0).all()
    assert (semantic_id[background] == 0).all()

    instance_values = set(int(value) for value in np.unique(instance_id[foreground]))
    semantic_values = set(int(value) for value in np.unique(semantic_id[foreground]))
    assert instance_values == set(frame.instance_id_to_path)
    assert set(frame.instance_id_to_path.values()) == {"box"}
    assert semantic_values == set(frame.semantic_id_to_label)
    assert set(frame.semantic_id_to_label.values()) == {"crate"}

    dlpack_depth = np.from_dlpack(frame["linear_depth"])
    np.testing.assert_array_equal(dlpack_depth, depth)
    assert frame["linear_depth"].__dlpack_device__() == (1, 0)

    legacy_rgb = gobot.render.capture_rgb(root=root, width=32, height=24)
    assert legacy_rgb.shape == (24, 32, 3)
    assert legacy_rgb.dtype == np.uint8
    legacy_overlay_rgb = gobot.render.capture_rgb(
        root=root,
        width=20,
        height=12,
        debug_arrows=(
            gobot.render.DebugArrow(
                start=(0.0, 0.0, 0.0),
                vector=(0.0, 0.0, 1.0),
            ),
        ),
    )
    assert legacy_overlay_rgb.shape == (12, 20, 3)

    single_slot = gobot.render.CameraSensor(
        camera,
        root=root,
        width=16,
        height=12,
        outputs=("linear_depth",),
        device="cpu",
        frame_slots=1,
    )
    held_frame = single_slot.capture()
    held_array = held_frame["linear_depth"].numpy()
    del held_frame
    try:
        single_slot.capture()
        raise AssertionError("retained NumPy view should hold the only frame slot")
    except RuntimeError as error:
        assert "frame pool exhausted" in str(error)
    del held_array
    gc.collect()
    single_slot.capture()

    cuda_sensor = gobot.render.CameraSensor(
        camera,
        root=root,
        width=8,
        height=8,
        outputs=("rgb",),
        device="cuda",
    )
    try:
        cuda_sensor.capture()
        raise AssertionError("explicit CUDA capture must not silently fall back")
    except RuntimeError as error:
        assert "does not fall back" in str(error) or "CUDA render-product" in str(error)

    sensor = None
    single_slot = None
    cuda_sensor = None
    frame = None
    gc.collect()
    gobot.render._shutdown_headless_render_context()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
