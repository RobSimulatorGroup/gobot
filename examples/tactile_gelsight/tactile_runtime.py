"""Worker-owned single-sensor UIPC experiment; only immutable data crosses threads."""

from __future__ import annotations
import numpy as np
from press_geometry import START_HEIGHT, PRESS_HEIGHT

FIXED_DT = 1 / 120
FIELDS = (
    "rgb",
    "rgb_marker",
    "depth",
    "indentation",
    "contact_depth",
    "optical_indentation",
    "normal_pressure",
    "shear_stress",
    "marker_positions",
    "marker_motion",
    "local_vertices",
    "total_force",
    "clock",
)


def target_at(time):
    """Smooth 10 s cycle: approach, press, slide, retract, hold."""
    t = time % 10.0

    def smooth(v):
        v = np.clip(v, 0.0, 1.0)
        return v * v * (3 - 2 * v)

    compression = smooth((t - 0.5) / 2.0) * (1.0 - smooth((t - 7.0) / 2.0))
    lateral = 0.003 * smooth((t - 3.0) / 3.0)
    transform = np.eye(4)
    transform[:3, 3] = (
        lateral,
        0.0,
        START_HEIGHT - (START_HEIGHT - PRESS_HEIGHT) * compression,
    )
    return transform


class PressController:
    def __init__(self, provider, path):
        self.provider = provider
        self.path = path
        self.time = 0.0
        self.manual_target = None
        self.current_target = target_at(0.0)

    def step(self, dt):
        if self.manual_target is None:
            self.current_target = target_at(self.time)
        else:
            delta = self.manual_target[:3, 3] - self.current_target[:3, 3]
            distance = float(np.linalg.norm(delta))
            if distance > 0:
                delta *= min(
                    1.0, 0.01 * dt / distance
                )  # At most 10 mm/s, preserving CCD sweeps.
            position = self.current_target[:3, 3] + delta
            self.current_target = self.manual_target.copy()
            self.current_target[:3, 3] = position
        self.provider.set_affine_target(self.path, self.current_target)
        self.time += dt

    def reset(self, environments):
        if tuple(environments) != (0,):
            raise ValueError("the tactile bench has one environment")
        self.time = 0.0
        self.manual_target = None
        self.current_target = target_at(0.0)
        self.provider.set_affine_target(self.path, self.current_target)

    def apply_commands(self, commands):
        if not commands:
            return
        if set(commands) != {"press_target"}:
            raise ValueError("unknown tactile bench command")
        target = np.asarray(commands["press_target"], dtype=np.float64)
        if (
            target.shape != (4, 4)
            or not np.isfinite(target).all()
            or not np.allclose(target[3], (0, 0, 0, 1))
            or not np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-4)
            or np.linalg.det(target[:3, :3]) <= 0
        ):
            raise ValueError("press_target must be a finite rigid 4x4 transform")
        target = target.copy()
        target[:3, 3] = np.clip(
            target[:3, 3], (-0.03, -0.03, PRESS_HEIGHT), (0.03, 0.03, 0.03)
        )
        self.manual_target = target


class TactileOutput:
    def __init__(self, provider, sensor_path):
        self.provider = provider
        self.sensor_path = sensor_path

    def snapshot(self, fields, environments):
        from gobot.ipc._libuipc_provider import _affine_transforms_to_poses

        if tuple(environments) != (0,):
            raise ValueError("the tactile bench has one environment")
        frame = self.provider.tactile_frames[self.sensor_path]
        result = {}
        for name in fields:
            if name == "affine.link_pose":
                result[name] = _affine_transforms_to_poses(
                    self.provider.arrays["affine_transforms"]
                )[None]
            elif name.startswith("tactile.") and name[8:] in FIELDS:
                result[name] = frame[name[8:]][None]
            else:
                raise KeyError(name)
        return result


def create(artifact, config, sensor_path, press_path):
    from gobot.ipc import LibuipcConfig, LibuipcProvider
    from gobot.sim import RuntimeComponents

    provider = LibuipcProvider(artifact, config=LibuipcConfig(**config))
    try:
        return RuntimeComponents(
            provider,
            PressController(provider, press_path),
            TactileOutput(provider, sensor_path),
        )
    except Exception:
        provider.close()
        raise
