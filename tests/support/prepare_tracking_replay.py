"""Build a Gobot Editor project for a recorded mjlab policy/reference rollout."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np


def prepare(scene, trajectory, evaluation, output):
    scene, trajectory, output = map(Path, (scene, trajectory, output))
    output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_binary_path(str(trajectory.with_suffix(".mjb")))
    data = mujoco.MjData(model)
    ids = [i for i in range(model.nbody) if model.body(i).name.startswith("robot/")]
    names = [model.body(i).name.removeprefix("robot/") for i in ids]
    if not names or names[0] != "pelvis":
        raise ValueError("expected pelvis as the first robot link")
    with np.load(trajectory) as trace:
        poses = {}
        for key, source in (("policy", "qpos"), ("reference", "reference_qpos")):
            frames = []
            for qpos in trace[source]:
                data.qpos[:] = qpos
                mujoco.mj_kinematics(model, data)
                frames.append(np.concatenate((data.xpos[ids], data.xquat[ids][:, [1, 2, 3, 0]]), axis=-1))
            poses[key] = np.asarray(frames, dtype=np.float32)
        np.savez_compressed(output / "replay.npz", **poses, link_names=names,
                            step_dt=trace["step_dt"],
                            anchor_position_error_m=trace["anchor_position_error_m"])
    shutil.copytree(scene.parent / "assets", output / "assets")
    shutil.copy2(evaluation, output / "evaluation.json")
    script = Path(__file__).resolve().parents[2] / "examples/g1/tracking/replay.py"
    shutil.copy2(script, output / "replay.py")
    original = json.loads(scene.read_text())
    document = copy.deepcopy(original)
    document["__EXT_RESOURCES__"].append(
        {"__ID__": "replay_script", "__TYPE__": "PythonScript", "__PATH__": "res://replay.py"})
    document["__NODES__"] = [{"name": "MjlabDanceReplay", "type": "Node3D", "parent": -1,
                               "properties": {"script": "ExtResource(replay_script)"}}]
    for name, color in (("Policy", (0.26, 0.57, 0.91)), ("Reference", (0.28, 0.79, 0.53))):
        material_id = name + "_Color"
        document["__SUB_RESOURCES__"].append({
            "__ID__": material_id, "__TYPE__": "PBRMaterial3D",
            "albedo": dict(zip(("red", "green", "blue", "alpha"), (*color, 1.0))),
            "roughness": 0.65, "metallic": 0.0,
        })
        start = len(document["__NODES__"])
        nodes = copy.deepcopy(original["__NODES__"])
        for node in nodes:
            node["parent"] = 0 if node["parent"] == -1 else node["parent"] + start
            properties = node["properties"]
            properties["script"] = None
            if node["type"] == "Robot3D":
                node["name"] = name
                properties["mode"] = "Motion"
            elif node["type"] == "CollisionShape3D":
                properties["disabled"] = True
            elif node["type"] == "MeshInstance3D":
                properties["material"] = f"SubResource({material_id})"
                properties["surface_color"] = dict(zip(("red", "green", "blue", "alpha"), (*color, 1.0)))
        document["__NODES__"].extend(nodes)
    (output / "world.jscn").write_text(json.dumps(document, indent=2) + "\n")
    (output / "project.gobot").write_text(json.dumps({
        "main_scene": "res://world.jscn",
        "editor_scene_views": {"res://world.jscn": {
            "eye": [0.0, -5.8, 2.7], "at": [0.0, 0.0, 0.8],
            "up": [0.0, 0.0, 1.0], "fov_y": 42.0,
        }},
    }, indent=2) + "\n")
    print(f"Prepared {len(poses['policy'])} frames and {len(names)} links: {output.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    prepare(**vars(p.parse_args()))
