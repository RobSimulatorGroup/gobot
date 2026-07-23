import gc
import sys

import gobot
import numpy as np


def main() -> int:
    original_settings = gobot.render.get_raster_settings()
    root = gobot.create_node("Node3D", "World")

    ground = gobot.create_box_visual("ground", (6.0, 6.0, 0.1))
    ground.position = (0.0, 0.0, -0.55)
    ground.surface_color = (0.72, 0.75, 0.70, 1.0)
    root.add_child(ground)

    box = gobot.create_box_visual("box", (1.0, 1.0, 1.0))
    box.surface_color = (0.72, 0.28, 0.16, 1.0)
    root.add_child(box)

    outside = gobot.create_box_visual("outside", (1.0, 1.0, 1.0))
    outside.position = (100.0, 0.0, 0.0)
    root.add_child(outside)

    sun = gobot.create_node("DirectionalLight3D", "sun")
    sun.set("intensity", 3.0)
    sun.set("shadow_enabled", True)
    sun.set("rotation_degrees", (35.0, -25.0, 0.0))
    root.add_child(sun)

    environment = gobot.create_node("Environment3D", "environment")
    environment.set("ambient_intensity", 0.12)
    root.add_child(environment)

    camera = gobot.create_node("Camera3D", "camera")
    camera.set("eye", (3.2, -4.8, 2.8))
    camera.set("target", (0.0, 0.0, -0.1))
    camera.set("up", (0.0, 0.0, 1.0))
    camera.set("fov_y", 55.0)
    camera.set("z_near", 0.05)
    camera.set("z_far", 30.0)

    sensor = gobot.render.CameraSensor(
        camera,
        root=root,
        width=160,
        height=120,
        outputs=("rgb",),
        device="cpu",
        mode="realtime",
    )
    try:
        gobot.render.set_raster_settings(
            gobot.render.RasterSettings(
                frustum_culling=True,
                anti_aliasing="disabled",
                shadow_quality="medium",
                shadow_distance=20.0,
            )
        )
        shadowed = sensor.capture()["rgb"].numpy().copy()
        sun.set("shadow_enabled", False)
        unshadowed = sensor.capture()["rgb"].numpy().copy()
        sun.set("shadow_enabled", True)
    except RuntimeError as error:
        if "EGL" in str(error) or "egl" in str(error):
            print(f"raster pipeline test skipped: {error}")
            return 77
        raise

    shadow_difference = np.abs(shadowed.astype(np.int16) - unshadowed.astype(np.int16))
    shadow_metrics = (
        float(shadow_difference.mean()),
        int(shadow_difference.max()),
        int(np.count_nonzero(shadow_difference)),
    )
    assert shadow_difference.mean() > 0.15, shadow_metrics
    assert shadow_difference.max() > 10, shadow_metrics

    gobot.render.set_raster_settings(
        gobot.render.RasterSettings(
            frustum_culling=True,
            anti_aliasing="fxaa",
            shadow_quality="medium",
            shadow_distance=20.0,
        )
    )
    antialiased = sensor.capture()["rgb"].numpy().copy()
    assert np.any(antialiased != shadowed)

    aov_sensor = gobot.render.CameraSensor(
        camera,
        root=root,
        width=80,
        height=60,
        outputs=("linear_depth", "instance_id"),
        device="cpu",
        mode="minimal",
    )
    aov_frame = aov_sensor.capture()
    instance_id = aov_frame["instance_id"].numpy()
    outside_ids = {
        instance for instance, path in aov_frame.instance_id_to_path.items() if path == "outside"
    }
    assert len(outside_ids) == 1
    assert not np.isin(instance_id, list(outside_ids)).any()
    assert np.isfinite(aov_frame["linear_depth"].numpy()[instance_id != 0]).all()

    gobot.render.set_raster_settings(original_settings)
    sensor = None
    aov_sensor = None
    gc.collect()
    gobot.render._shutdown_headless_render_context()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
