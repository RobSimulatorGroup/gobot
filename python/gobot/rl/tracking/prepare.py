"""Prepare a Gobot G1 scene and named motion archive for tracking.

The source scene remains the authoring source. This utility requires the Gobot
examples checkout, but neither training nor evaluation imports mjlab.
"""

import argparse
import csv as csv_module
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import gobot
from gobot.rl.kinematics import SceneKinematics
from .config import G1_JOINTS, G1_BODIES, REFERENCE_REVISION
from .math import matrix, quaternion
from .motion import Motion, FIELDS


def load_csv(path, *, csv_format="lafan"):
    """Return metre/xyzw/radian coordinates, mapping SEED headers by joint name."""
    if csv_format == "lafan":
        return np.loadtxt(path, delimiter=",", dtype=np.float64)
    if csv_format != "seed":
        raise ValueError("csv_format must be lafan or seed")
    from scipy.spatial.transform import Rotation

    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv_module.reader(stream)
        header = next(reader)
    if len(set(header)) != len(header):
        raise ValueError("duplicate SEED CSV column")
    expected = (
        ["Frame"]
        + [f"root_translate{a}" for a in "XYZ"]
        + [f"root_rotate{a}" for a in "XYZ"]
        + [name + "_dof" for name in G1_JOINTS]
    )
    if set(header) != set(expected):
        raise ValueError(
            "SEED CSV needs Frame, root translation/rotation and named G1 joints"
        )
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.shape[1] != len(header) or not np.isfinite(data).all():
        raise ValueError("SEED CSV must contain finite numeric rows")
    indices = [header.index(n) for n in expected]
    data = data[:, indices]
    if len(data) > 1 and not np.allclose(np.diff(data[:, 0]), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("SEED Frame indices must be consecutive")
    pos = data[:, 1:4] / 100.0
    rotation = Rotation.from_euler("xyz", data[:, 4:7], degrees=True).as_quat()
    joints = np.deg2rad(data[:, 7:])
    return np.concatenate((pos, rotation, joints), axis=-1)


def convert_csv(path, kin, *, input_fps=30.0, dt=0.02, csv_format="lafan"):
    from scipy.spatial.transform import Rotation, Slerp

    raw = load_csv(path, csv_format=csv_format)
    if (
        raw.ndim != 2
        or raw.shape[1] != 7 + len(G1_JOINTS)
        or len(raw) < 3
        or not np.isfinite(raw).all()
    ):
        raise ValueError(
            "CSV must have root xyz, quaternion xyzw, and 29 G1 joint angles"
        )
    if not np.isfinite(input_fps) or input_fps <= 0:
        raise ValueError("input fps must be positive")
    norm = np.linalg.norm(raw[:, 3:7], axis=1)
    if (norm < 1e-8).any():
        raise ValueError("zero motion quaternion")
    t = np.arange(len(raw)) / input_fps
    out = np.arange(0, t[-1], dt)
    if len(out) < 3:
        raise ValueError("motion is too short at the target fps")
    rot = Slerp(t, Rotation.from_quat(raw[:, 3:7] / norm[:, None]))(out)
    pos = np.stack([np.interp(out, t, raw[:, i]) for i in range(3)], -1)
    joints = np.stack(
        [np.interp(out, t, raw[:, i]) for i in range(7, raw.shape[1])], -1
    )
    vel = np.gradient(pos, dt, axis=0)
    qd = np.gradient(joints, dt, axis=0)
    omega = (rot[2:] * rot[:-2].inv()).as_rotvec() / (2 * dt)
    omega = np.concatenate((omega[:1], omega, omega[-1:]))
    quat = rot.as_quat()[:, [3, 0, 1, 2]]
    return _from_root(kin, pos, quat, vel, omega, joints, qd)


def _from_root(kin, p, r, v, w, q, qd):
    p, r, v, w, q, qd = [
        torch.as_tensor(x, dtype=torch.float32) for x in (p, r, v, w, q, qd)
    ]
    # Bound peak memory for long clips.
    buffers = [[] for _ in range(4)]
    for lo in range(0, len(p), 256):
        sl = slice(lo, lo + 256)
        values = kin(p[sl], matrix(r[sl]), v[sl], w[sl], q[sl], qd[sl])
        for i, value in enumerate(values):
            buffers[i].append(quaternion(value) if i == 1 else value)
    return dict(
        zip(
            FIELDS,
            (q.numpy(), qd.numpy(), *(torch.cat(b).numpy() for b in buffers)),
            strict=True,
        )
    )


def prepare(
    output,
    *,
    source_project=None,
    csv=None,
    input_fps=None,
    csv_format="lafan",
    npz=None,
    layout=None,
    standing_smoke=False,
):
    from examples.g1.config import g1_velocity_cfg
    from gobot.rl.locomotion.scene import prepare_velocity_scene

    if sum((csv is not None, npz is not None, standing_smoke)) != 1:
        raise ValueError("choose CSV, NPZ, or standing smoke reference")
    if input_fps is None:
        input_fps = 120.0 if csv_format == "seed" else 30.0
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = g1_velocity_cfg(project_path=source_project)
    cfg.scene_path = "res://g1_flat_scene.jscn"
    context = gobot.app.create_context()
    try:
        context, robot_node, _ = prepare_velocity_scene(cfg, context=context)
        snapshot = context.compile_physics_scene_snapshot()
        kin = SceneKinematics(
            snapshot,
            robot_name="g1",
            base_link="pelvis",
            joint_names=G1_JOINTS,
            tracked_links=G1_BODIES,
        )
        if csv is not None:
            arrays = convert_csv(csv, kin, input_fps=input_fps, csv_format=csv_format)
        elif npz is not None:
            motion = Motion(npz, G1_JOINTS, G1_BODIES, layout=layout)
            arrays = {k: getattr(motion, k).cpu().numpy() for k in FIELDS}
        else:
            robot = next(r for r in snapshot["robots"] if r["name"] == "g1")
            joints = {j["name"]: j for j in robot["joints"]}
            q = np.array(
                [joints[n]["initial_position"] for n in G1_JOINTS], dtype=np.float32
            )[None].repeat(250, axis=0)
            p = np.zeros((250, 3), np.float32)
            p[:, 2] = 0.79
            r = np.zeros((250, 4), np.float32)
            r[:, 0] = 1
            arrays = _from_root(
                kin, p, r, np.zeros_like(p), np.zeros_like(p), q, np.zeros_like(q)
            )
        np.savez_compressed(
            output / "motion.npz",
            **arrays,
            fps=np.array(50.0),
            joint_names=np.array(G1_JOINTS),
            body_names=np.array(G1_BODIES),
        )
        # Save the configured robot separately so instance serialization does not
        # discard its child drive edits. Clone resource documents into the project;
        # mesh assets remain explicit absolute references to the source project.
        context.root.set("script", None)
        robot_document = json.loads((Path(cfg.project_path) / "g1.jscn").read_text())
        from gobot.rl.locomotion.math import find_node_by_name

        drive_fields = (
            "drive_mode",
            "drive_stiffness",
            "drive_damping",
            "armature",
            "effort_limit",
            "velocity_limit",
            "force_lower_limit",
            "force_upper_limit",
            "control_lower_limit",
            "control_upper_limit",
            "damping",
            "friction_loss",
        )
        for node in robot_document["__NODES__"]:
            if node["name"] in G1_JOINTS and node["type"] == "Joint3D":
                live = find_node_by_name(robot_node, node["name"])
                for name in drive_fields:
                    value = live.get(name)
                    node["properties"][name] = (
                        "Position" if name == "drive_mode" else float(value)
                    )
        (output / "robot.jscn").write_text(json.dumps(robot_document, indent=2))
        gobot.save_scene(context.root, str(output / "world.jscn"))
        source_root = Path(cfg.project_path).resolve()

        def rewrite(document):
            for resource in document.get("__EXT_RESOURCES__", ()):
                path = resource["__PATH__"]
                if path.startswith("res://"):
                    original = source_root / path[6:]
                    if (
                        resource["__TYPE__"] == "PackedScene"
                        and original.name == "g1.jscn"
                    ):
                        resource["__PATH__"] = str(output / "robot.jscn")
                    elif original.suffix == ".jres":
                        target = (
                            output / "resources" / original.relative_to(source_root)
                        )
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(
                            json.dumps(
                                rewrite(json.loads(original.read_text())), indent=2
                            )
                        )
                        resource["__PATH__"] = str(target)
                    else:
                        resource["__PATH__"] = str(original)
            for node in document.get("__NODES__", ()):
                node.get("properties", {})["script"] = None
            return document

        for name in ("robot.jscn", "world.jscn"):
            path = output / name
            path.write_text(
                json.dumps(rewrite(json.loads(path.read_text())), indent=2) + "\n"
            )
        (output / "project.gobot").write_text(
            json.dumps({"main_scene": "res://world.jscn"}, indent=2) + "\n"
        )
        (output / "tracking-source.json").write_text(
            json.dumps(
                dict(
                    reference_revision=REFERENCE_REVISION,
                    motion_source=str(csv or npz or "synthetic standing smoke"),
                    source_sha256=hashlib.sha256(
                        Path(csv or npz).read_bytes()
                    ).hexdigest()
                    if csv or npz
                    else None,
                    csv_format=csv_format if csv else None,
                    input_fps=input_fps if csv else None,
                    smoke_only=standing_smoke,
                    kinematics_digest=kin.digest,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        context.clear_scene()
    print(output)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-project", type=Path)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", type=Path)
    group.add_argument("--npz", type=Path)
    group.add_argument("--standing-smoke", action="store_true")
    p.add_argument("--input-fps", type=float)
    p.add_argument("--csv-format", choices=("lafan", "seed"), default="lafan")
    p.add_argument("--layout", type=Path)
    args = p.parse_args()
    prepare(**vars(args))


if __name__ == "__main__":
    main()
