"""Validated, name-mapped BeyondMimic/mjlab NPZ references; no simulator dependency."""

import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

FIELDS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


class Motion:
    def __init__(
        self, path, joint_names, body_names, *, dt=0.02, device="cpu", layout=None
    ):
        path = Path(path)
        self.digest = hashlib.sha256(path.read_bytes()).hexdigest()
        names = json.loads(Path(layout).read_text()) if layout is not None else {}
        with np.load(path, allow_pickle=False) as archive:
            for key in ("joint_names", "body_names", "fps"):
                if key in archive:
                    value = archive[key].tolist()
                    if key in names and names[key] != value:
                        raise ValueError(f"conflicting motion {key}")
                    names[key] = value
            if not all(k in names for k in ("joint_names", "body_names", "fps")):
                raise ValueError(
                    "motion needs fps, joint_names and body_names; supply motion_layout for unnamed mjlab NPZ"
                )
            fps = np.asarray(names["fps"]).reshape(-1)
            if (
                len(fps) != 1
                or not math.isfinite(float(fps[0]))
                or not math.isclose(float(fps[0]) * dt, 1.0, rel_tol=1e-5)
            ):
                raise ValueError(
                    "motion fps must match the control clock; resample before training"
                )
            self.fps = float(fps[0])
            for key in ("joint_names", "body_names"):
                if (
                    not isinstance(names[key], list)
                    or not all(isinstance(n, str) and n for n in names[key])
                    or len(set(names[key])) != len(names[key])
                ):
                    raise ValueError(f"{key} must be a unique name array")
            ji = self._indices(names["joint_names"], joint_names)
            bi = self._indices(names["body_names"], body_names)
            self.layout_digest = hashlib.sha256(
                json.dumps(names, sort_keys=True).encode()
            ).hexdigest()
            self.frames = len(archive["joint_pos"])
            if self.frames < 2:
                raise ValueError("motion requires at least two frames")
            for key in FIELDS:
                a = np.asarray(archive[key], dtype=np.float32)
                is_joint = key.startswith("joint_")
                expected = (
                    (self.frames, len(names["joint_names"]))
                    if is_joint
                    else (
                        self.frames,
                        len(names["body_names"]),
                        4 if "quat" in key else 3,
                    )
                )
                if a.shape != expected or not np.isfinite(a).all():
                    raise ValueError(f"{key} must be finite with shape {expected}")
                a = a[:, ji if is_joint else bi]
                if "quat" in key:
                    norms = np.linalg.norm(a, axis=-1, keepdims=True)
                    if not np.allclose(norms, 1, atol=1e-3):
                        raise ValueError(
                            "body_quat_w must contain unit wxyz quaternions"
                        )
                    # Preserve the shared reference bytes; the native loader does not renormalize.
                setattr(self, key, torch.as_tensor(a.copy(), device=device))

    @staticmethod
    def _indices(source, desired):
        missing = set(desired) - set(source)
        if missing:
            raise ValueError(f"missing motion names: {sorted(missing)}")
        return [source.index(n) for n in desired]

    def bodies(self, frames):
        return tuple(getattr(self, key)[frames] for key in FIELDS[2:])
