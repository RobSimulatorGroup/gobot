"""Prepare a Gobot Editor project for live tracking inference and reference display."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import gobot
from gobot.rl.training.workflow import update_config
from .training import Config


def walk(root):
    yield root
    for child in root.children:
        yield from walk(child)


def prepare(checkpoint, output, project_path=None, device="cuda:0"):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg = update_config(Config(), json.loads(checkpoint.with_name("config.json").read_text()))
    source = Path(project_path or cfg.project_path).resolve()
    output.mkdir(parents=True, exist_ok=False)
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(source))
        root = context.load_scene(cfg.scene_path)
        robot = next(node for node in walk(root) if node.type_name == "Robot3D" and node.name == cfg.robot_name)
        snapshot = context.compile_physics_scene_snapshot()
        spec = next(entry for entry in snapshot["robots"] if entry["name"] == cfg.robot_name)
        names = {cfg.base_link}
        while True:
            children = {joint["child_link"] for joint in spec["joints"] if joint["parent_link"] in names}
            if children <= names:
                break
            names.update(children)
        names = [link["name"] for link in spec["links"] if link["name"] in names and link.get("role") != "virtual_root"]
        gobot.save_scene(robot, str(output / "robot.jscn"))
    finally:
        context.clear_scene()
    model = json.loads((output / "robot.jscn").read_text())
    for resource in model["__EXT_RESOURCES__"]:
        if resource["__PATH__"].startswith("res://"):
            resource["__PATH__"] = str(source / resource["__PATH__"][6:])
    scene = deepcopy(model)
    scene["__EXT_RESOURCES__"].append({"__ID__": "play", "__TYPE__": "PythonScript", "__PATH__": "res://play.py"})
    scene["__NODES__"] = [{"type": "Node3D", "name": "Tracking", "parent": -1,
                             "properties": {"script": "ExtResource(play)"}}]
    for name, color in (("Policy", (0.035, 0.24, 0.8)), ("Reference", (0.025, 0.55, 0.12))):
        scene["__SUB_RESOURCES__"].append({"__ID__": name, "__TYPE__": "PBRMaterial3D",
            "albedo": dict(zip(("red", "green", "blue", "alpha"), (*color, 1.0))),
            "roughness": .65, "metallic": 0.0})
        start = len(scene["__NODES__"])
        nodes = deepcopy(model["__NODES__"])
        for node in nodes:
            node["parent"] = 0 if node["parent"] == -1 else start + node["parent"]
            node["properties"]["script"] = None
            if node["type"] == "Robot3D":
                node["name"] = name
                node["properties"]["mode"] = "Motion"
            elif node["type"] == "MeshInstance3D":
                node["properties"]["material"] = f"SubResource({name})"
            elif node["type"] == "CollisionShape3D":
                node["properties"]["disabled"] = True
        scene["__NODES__"].extend(nodes)
    (output / "play.py").write_text("from gobot.rl.tracking.editor_play import Script\n")
    (output / "world.jscn").write_text(json.dumps(scene, indent=2) + "\n")
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(output))
        root = context.load_scene("res://world.jscn")
        floor = gobot.create_box_visual("Floor", (14, 14, .08), (0, 0, -.04))
        floor.surface_color = (.16, .19, .24, 1)
        root.add_child(floor)
        gobot.save_scene(root, str(output / "world.jscn"))
    finally:
        context.clear_scene()
    (output / "robot.jscn").unlink()
    (output / "play.json").write_text(json.dumps(dict(
        link_names=names, control_dt=cfg.physics_dt * cfg.decimation,
        runtime=dict(checkpoint=str(checkpoint), project_path=str(source), device=device)), indent=2) + "\n")
    (output / "project.gobot").write_text(json.dumps({
        "main_scene": "res://world.jscn", "editor_scene_views": {"res://world.jscn": {
            "eye": [0, -5.8, 2.7], "at": [0, 0, .8], "up": [0, 0, 1], "fov_y": 42}}}, indent=2) + "\n")
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--project-path", type=Path)
    p.add_argument("--device", default="cuda:0")
    print(prepare(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
