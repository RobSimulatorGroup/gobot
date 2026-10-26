"""Download G1 locomotion references and compile features against the authored Gobot robot."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from .config import JOINT_NAMES, g1_amp_cfg
from gobot.rl.locomotion.scene import prepare_velocity_scene
from gobot.rl.locomotion.contract import robot_physics_digest
from gobot.rl.locomotion.motion import motion_features
from gobot.sim.providers import CompiledSceneArtifact
import gobot

DATASET = "lvhaidong/LAFAN1_Retargeting_Dataset"
API = f"https://huggingface.co/api/datasets/{DATASET}"


def read_url(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def resample(raw, source_fps=30., dt=.02):
    if raw.ndim != 2 or raw.shape[1] != 36 or len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError("motion must contain finite root position, xyzw quaternion and 29 joints")
    norms = np.linalg.norm(raw[:, 3:7], axis=1)
    if np.any(norms < 1e-6):
        raise ValueError("zero reference quaternion")
    times = np.arange(len(raw)) / source_fps
    target = np.arange(int(np.floor(times[-1] / dt)) + 1) * dt
    rotations = Slerp(times, Rotation.from_quat(raw[:, 3:7] / norms[:, None]))(target)
    p = np.stack([np.interp(target, times, raw[:, i]) for i in range(3)], 1)
    joints = np.stack([np.interp(target, times, raw[:, i]) for i in range(7, 36)], 1)
    v = np.gradient(p, dt, axis=0)
    omega = np.empty_like(v)
    omega[1:-1] = (rotations[2:] * rotations[:-2].inv()).as_rotvec() / (2 * dt)
    omega[0] = (rotations[1] * rotations[0].inv()).as_rotvec() / dt
    omega[-1] = (rotations[-1] * rotations[-2].inv()).as_rotvec() / dt
    return p, rotations, joints, v, omega, np.gradient(joints, dt, axis=0)


def prepare(output, *, offline=False):
    import mujoco
    output = Path(output)
    raw_dir = output.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_file = output.parent / "source.json"
    if not offline:
        revision = json.loads(read_url(API))["sha"]
        files = json.loads(read_url(f"{API}/tree/{revision}/g1"))
        files = [item["path"] for item in files if Path(item["path"]).name.startswith(("walk", "run", "sprint"))]
        def download(name):
            content = read_url(f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{name}")
            path = raw_dir / Path(name).name
            path.write_bytes(content)
            return {"name": path.name, "sha256": hashlib.sha256(content).hexdigest()}
        with ThreadPoolExecutor(max_workers=4) as pool:
            entries = list(pool.map(download, sorted(files)))
        source = {"dataset": DATASET, "revision": revision, "files": entries,
                  "license": "LAFAN1 CC BY-NC-ND 4.0; G1 retarget provenance retained"}
        source_file.write_text(json.dumps(source, indent=2) + "\n")
    else:
        source = json.loads(source_file.read_text())
    cfg = g1_amp_cfg(stage="flat")
    context, _, _ = prepare_velocity_scene(cfg)
    try:
        artifact = CompiledSceneArtifact.from_mapping(context.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu))
        model = mujoco.MjModel.from_xml_string(artifact.content)
        data = mujoco.MjData(model)
        prefix = artifact.robot_prefix("g1")
        joint_ids = [model.joint(prefix + name).id for name in JOINT_NAMES]
        qpos_ids = model.jnt_qposadr[joint_ids]
        links = [model.body(prefix + name).id for name in cfg.state_link_names]
        root_id = model.body(prefix + cfg.base_link).id
        arrays, clips = {}, []
        for i, item in enumerate(source["files"]):
            path = raw_dir / item["name"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"motion checksum mismatch: {path}")
            raw = np.loadtxt(path, delimiter=",")
            p, rotations, joints, v, omega, qd = resample(raw)
            xyzw = rotations.as_quat(); quat = xyzw[:, [3, 0, 1, 2]]
            link_positions = np.empty((len(p), len(links), 3), np.float32)
            base_positions = np.empty_like(p)
            for frame in range(len(p)):
                data.qpos[:] = model.qpos0
                data.qpos[:3] = p[frame]
                data.qpos[3:7] = quat[frame]
                data.qpos[qpos_ids] = joints[frame]
                mujoco.mj_kinematics(model, data)
                link_positions[frame] = data.xpos[links]
                base_positions[frame] = data.xpos[root_id]
            state = dict(base_position=base_positions, base_quaternion=quat, joint_position=joints,
                         joint_velocity=qd, base_linear_velocity_body=rotations.inv().apply(v),
                         base_angular_velocity_body=rotations.inv().apply(omega), link_position=link_positions)
            features = motion_features(state).numpy()
            lo, hi = model.jnt_range[joint_ids].T
            valid = np.isfinite(features).all(1)
            valid &= ((joints >= lo - .1) & (joints <= hi + .1)).all(1)
            valid &= (np.abs(qd) <= 2 * np.asarray(cfg.robot.velocity_limit)).all(1)
            valid &= rotations.as_matrix()[:, 2, 2] > .7
            # Exclude gross retarget artifacts, retaining ballistic running phases.
            ankle_z = link_positions[:, :2, 2].min(1)
            valid &= (ankle_z > -.12) & (ankle_z < .5)
            arrays[f"features_{i}"] = features.astype(np.float32)
            arrays[f"valid_{i}"] = valid
            arrays[f"root_{i}"] = np.concatenate((p, quat, v, omega), 1).astype(np.float32)
            arrays[f"joint_{i}"] = np.concatenate((joints, qd), 1).astype(np.float32)
            clips.append({"name": path.stem, "frames": len(features), "valid_frames": int(valid.sum()),
                          "gait": int(not path.name.startswith("walk")),
                          "split": "holdout" if "subject5" in path.stem else "train"})
            print(path.name, len(features), "frames,", int(valid.sum()), "valid", flush=True)
        metadata = {"schema": "gobot_motion_v1", "dt": .02, "source_fps": 30., "joint_names": JOINT_NAMES,
                    "link_names": cfg.state_link_names, "robot_digest": robot_physics_digest(artifact, "g1"),
                    "source": source, "clips": clips}
        arrays["metadata"] = np.array(json.dumps(metadata))
        np.savez_compressed(output, **arrays)
        print("Prepared", output, flush=True)
    finally:
        context.clear_scene()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "assets/motions/locomotion.npz")
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()
    prepare(args.output, offline=args.offline)

if __name__ == "__main__":
    main()
