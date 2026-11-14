"""Create a Gobot display project that runs native mjlab inference and physics."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import gobot


def prepare(scene, checkpoint, motion, output, extra_python_paths=()):
    repository = Path(__file__).resolve().parents[2]
    scene, checkpoint, motion, output = [Path(p).resolve() for p in (scene, checkpoint, motion, output)]
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(scene.parent / "assets", output / "assets")
    for name in ("mjlab_live.py", "mjlab_live_runtime.py", "mjlab_contacts.py"):
        shutil.copy2(repository / "examples/g1/tracking" / name, output / name)
    with np.load(motion) as data:
        names = data["body_names"].tolist()
        duration = (len(data["joint_pos"]) - 1) / float(data["fps"][0])
    original = json.loads(scene.read_text())
    doc = copy.deepcopy(original)
    doc["__EXT_RESOURCES__"].append({"__ID__": "live_script", "__TYPE__": "PythonScript",
                                      "__PATH__": "res://mjlab_live.py"})
    doc["__NODES__"] = [{"name": "LIVE_Mjlab_Dance", "type": "Node3D", "parent": -1,
                          "properties": {"script": "ExtResource(live_script)"}}]
    for name, color in (("Policy", (0.035, 0.24, 0.8)), ("Reference", (0.025, 0.55, 0.12))):
        material = name + "_Color"
        doc["__SUB_RESOURCES__"].append({
            "__ID__": material, "__TYPE__": "PBRMaterial3D",
            "albedo": dict(zip(("red", "green", "blue", "alpha"), (*color, 1.0))),
            "roughness": 0.65, "metallic": 0.0,
        })
        start = len(doc["__NODES__"])
        for original_node in original["__NODES__"]:
            node = copy.deepcopy(original_node)
            node["parent"] = 0 if node["parent"] == -1 else start + node["parent"]
            props = node["properties"]
            props["script"] = None
            if node["type"] == "Robot3D":
                node["name"] = name
                props["mode"] = "Motion"
            elif node["type"] == "MeshInstance3D":
                props["material"] = f"SubResource({material})"
            elif node["type"] == "CollisionShape3D":
                # These are display copies; collision runs in the native mjlab worker.
                props["disabled"] = True
            doc["__NODES__"].append(node)
    path = output / "world.jscn"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    ctx = gobot.app.create_context()
    try:
        ctx.set_project_path(str(output))
        root = ctx.load_scene("res://world.jscn")
        floor = gobot.create_box_visual("VisibleFloor_z0", (14, 14, 0.08), (0, 0, -0.04))
        floor.surface_color = (0.16, 0.19, 0.24, 1.0)
        root.add_child(floor)
        # Visible floor grid establishes contact height and distance scale.
        for i in range(-7, 8):
            for axis in (0, 1):
                size = (0.012, 14, 0.003) if axis == 0 else (14, 0.012, 0.003)
                position = (i, 0, 0.0015) if axis == 0 else (0, i, 0.0015)
                line = gobot.create_box_visual(f"FloorGrid_{axis}_{i + 7}", size, position)
                line.surface_color = (0.28, 0.31, 0.36, 1.0)
                root.add_child(line)
        gobot.save_scene(root, str(path))
    finally:
        ctx.clear_scene()
    settings = {
        "duration": duration, "link_names": names,
        "runtime": {"source": str(repository / "mjlab"), "checkpoint": str(checkpoint),
                    "motion": str(motion), "support": str(repository / "tests/support"),
                    "extra_python_paths": [str(Path(p).resolve()) for p in extra_python_paths],
                    "status_file": str(output / "live-status.json")},
    }
    (output / "live.json").write_text(json.dumps(settings, indent=2) + "\n")
    (output / "project.gobot").write_text(json.dumps({
        "main_scene": "res://world.jscn",
        "editor_scene_views": {"res://world.jscn": {
            "eye": [0, -5.8, 2.7], "at": [0, 0, 0.8], "up": [0, 0, 1], "fov_y": 42,
        }},
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--extra-python-paths", type=Path, nargs="*", default=())
    prepare(**vars(p.parse_args()))
