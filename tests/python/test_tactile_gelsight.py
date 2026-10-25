"""Opt-in real CUDA test for the editor's single-world GelSight path."""

from __future__ import annotations
import json
import os
from pathlib import Path
import sys

if os.environ.get("GOBOT_RUN_TACTILE_GPU_TEST") != "1":
    print("Set GOBOT_RUN_TACTILE_GPU_TEST=1 for native CUDA tactile validation.")
    raise SystemExit(77)
import numpy as np
import gobot

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "examples/tactile_gelsight")
)
from press_geometry import START_HEIGHT, PRESS_HEIGHT
from gobot.ipc import LibuipcConfig, LibuipcProvider

ROOT = Path(__file__).resolve().parents[2]
context = gobot.app.create_context()
context.set_project_path(str(ROOT / "examples/tactile_gelsight"))
context.load_scene("res://gelsight_press.jscn")
artifact = context.compile_ipc_scene_artifact()
config = LibuipcConfig(
    fixed_time_step=1 / 120,
    gravity=(0, 0, 0),
    friction_coefficient=0.5,
    contact_activation_distance=0.0005,
    contact_resistance=1e10,
    affine_stiffness=1e8,
    kinematic_strength=10000.0,
    module_path=os.environ.get("GOBOT_LIBUIPC_TEST_MODULE_PATH", ""),
)
with LibuipcProvider(artifact, config=config) as provider:
    path = next(iter(provider.tactile_frames))
    head = provider.affine_bodies[0]["path"]
    rest = provider.arrays["positions"].copy()
    initial = provider.tactile_frames[path]
    assert initial["rgb"].shape == (240, 320, 3)
    assert initial["marker_motion"].shape == (63, 2)
    assert np.allclose(initial["marker_motion"], 0, atol=1e-5)
    assert np.allclose(initial["total_force"], 0)

    def step(x, z):
        transform = np.eye(4)
        transform[:3, 3] = (x, 0, z)
        provider.set_affine_target(head, transform)
        provider.step()
        frame = provider.tactile_frames[path]
        for value in frame.values():
            assert np.isfinite(value).all()
        return frame

    def press():
        for z in np.linspace(START_HEIGHT, PRESS_HEIGHT, 100):
            frame = step(0, float(z))
        return frame

    pressed = press()
    force = pressed["total_force"].copy()
    assert force[2] < -1e-5, force
    assert pressed["normal_pressure"].max() > 0
    assert pressed["indentation"].max() > 1e-5
    # The high-resolution camera must see through the torus center while seeing
    # the pressed annulus; the optical image may smooth around that contact core.
    assert pressed["contact_depth"][120, 160] > 0.033
    assert pressed["contact_depth"].min() < 0.028
    assert pressed["optical_indentation"].max() > 0.0003
    assert (
        pressed["optical_indentation"][120, 160]
        < pressed["optical_indentation"].max() * 0.5
    )
    max_shear = 0.0
    for x in np.linspace(0, 0.003, 60):
        sliding = step(float(x), PRESS_HEIGHT)
        max_shear = max(max_shear, float(np.linalg.norm(sliding["total_force"][:2])))
    assert max_shear > 1e-6, max_shear
    assert np.max(np.abs(sliding["marker_motion"])) > 0.01
    for z in np.linspace(PRESS_HEIGHT, START_HEIGHT + 0.002, 100):
        released = step(0.003, float(z))
    assert np.linalg.norm(released["total_force"]) < max(
        1e-4, np.linalg.norm(force) * 0.01
    )
    provider.reset()
    assert np.allclose(provider.arrays["positions"], rest, atol=1e-7)
    reset = provider.tactile_frames[path]
    assert reset["clock"][0] == 0 and np.allclose(reset["marker_motion"], 0, atol=1e-5)
    repeated = press()
    np.testing.assert_allclose(repeated["total_force"], force, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(
        repeated["indentation"], pressed["indentation"], rtol=1e-4, atol=1e-7
    )
    # Same controller command consumed by the editor's runtime gizmo.
    sys.path.insert(0, str(ROOT / "examples/tactile_gelsight"))
    from tactile_runtime import PressController

    provider.reset()
    controller = PressController(provider, head)
    manual = np.eye(4)
    manual[2, 3] = PRESS_HEIGHT
    controller.apply_commands({"press_target": manual})
    for _ in range(100):
        controller.step(config.fixed_time_step)
        provider.step()
    manual_frame = provider.tactile_frames[path]
    assert manual_frame["total_force"][2] < -1e-5
    assert manual_frame["normal_pressure"].max() > 0
    assert manual_frame["indentation"].max() > 1e-5
    provider.reset()
    controller.reset((0,))
    assert controller.manual_target is None
    print(
        json.dumps(
            {
                "press_force_N": force.tolist(),
                "peak_shear_N": max_shear,
                "indentation_m": float(pressed["indentation"].max()),
                "peak_pressure_Pa": float(pressed["normal_pressure"].max()),
                "repeatable_reset": True,
                "manual_target_contact": True,
            },
            indent=2,
        )
    )
