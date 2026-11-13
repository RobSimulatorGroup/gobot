"""Packed official motion tensors with injected scene kinematics.

The on-disk format also loads directly in the pinned mjlab_research MotionLoader.
Root quaternions are wxyz; both root velocities are in the world frame.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

FIELDS = {
    "root_pos_w": 3,
    "root_quat_w": 4,
    "joint_pos": 29,
    "root_lin_vel_w": 3,
    "root_ang_vel_w": 3,
    "joint_vel": 29,
}
SLICES = {
    "root_pos_w": (0, 3),
    "root_quat_w": (3, 7),
    "joint_pos": (7, 36),
    "root_lin_vel_w": (36, 39),
    "root_ang_vel_w": (39, 42),
    "joint_vel": (42, 71),
}
PROTOCOL = "gobot_supermimic_official_motion_v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inspect_library(path, *, verify_hashes=False):
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("protocol") != PROTOCOL or manifest.get("complete") is not True:
        raise ValueError("motion library is incomplete or has an unsupported protocol")
    if (
        manifest.get("fps") != 50.0
        or manifest.get("quaternion_order") != "wxyz"
        or manifest.get("root_velocity_frame") != "world"
    ):
        raise ValueError("official G1 motion requires 50 Hz, wxyz and world velocities")
    joints = json.loads((root / "joint_names.json").read_text())
    names = json.loads((root / "motion_names.json").read_text())
    ends = np.load(root / "motion_ends.npy", allow_pickle=False)
    if len(joints) != 29 or len(set(joints)) != 29 or joints != manifest["joint_names"]:
        raise ValueError("invalid motion joint order")
    if (
        ends.ndim != 1
        or ends.dtype != np.int64
        or not len(ends)
        or len(names) != len(ends)
        or len(set(names)) != len(names)
        or np.any(np.diff(np.r_[0, ends]) < 3)
        or int(ends[-1]) != manifest["frames"]
        or len(names) != manifest["clips"]
    ):
        raise ValueError("invalid clip names, lengths or boundaries")
    required = {f"{field}.npy" for field in FIELDS} | {
        "motion_ends.npy",
        "motion_names.json",
        "joint_names.json",
    }
    if set(manifest["files"]) != required:
        raise ValueError("motion manifest file set differs from protocol")
    for field, width in FIELDS.items():
        array = np.load(root / f"{field}.npy", mmap_mode="r", allow_pickle=False)
        if array.shape != (int(ends[-1]), width) or array.dtype not in (
            np.float16,
            np.float32,
        ):
            raise ValueError(f"invalid {field} shape or dtype")
    for name, entry in manifest["files"].items():
        if (root / name).stat().st_size != entry["bytes"]:
            raise ValueError(f"motion file size changed: {name}")
        if verify_hashes and digest(root / name) != entry["sha256"]:
            raise ValueError(f"motion file hash changed: {name}")
    return manifest


class MotionLibrary:
    def __init__(self, path, fk, joint_names, device="cpu", *, verify_hashes=True):
        self.path = Path(path)
        self.manifest = inspect_library(path, verify_hashes=verify_hashes)
        self.identity = digest(self.path / "manifest.json")
        if list(joint_names) != self.manifest["joint_names"]:
            raise ValueError("motion joint order differs from the authored G1 scene")
        self.fk = fk
        self.names = json.loads((self.path / "motion_names.json").read_text())
        self.motion_ends = torch.from_numpy(np.load(self.path / "motion_ends.npy")).to(
            device=device
        )
        self.time_step_total = self.manifest["frames"]
        for name in FIELDS:
            # Convert exactly once as the official loader does. Keep the source
            # quantization; never silently re-normalize quaternions or resample.
            array = np.load(self.path / f"{name}.npy", allow_pickle=False)
            setattr(
                self,
                name,
                torch.from_numpy(array).to(device=device, dtype=torch.float32),
            )

    def body_state(self, frames):
        return self.fk(
            self.root_pos_w[frames],
            self.root_quat_w[frames],
            self.joint_pos[frames],
            self.root_lin_vel_w[frames],
            self.root_ang_vel_w[frames],
            self.joint_vel[frames],
        )

    def full_state(self, frames):
        return torch.cat([getattr(self, name)[frames] for name in FIELDS], -1)
