"""Build the Gobot-native Allegro four-finger tactile grasp scene."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import gobot
from gobot.ipc._fabrication import create_fabricated_sensor


HERE = Path(__file__).resolve().parent
ASSET_MANIFEST = HERE / "allegro_assets.json"
PROJECT_ASSET_ROOT = Path("assets") / "wonik_allegro"
# Gobot node-name normalization removes punctuation from the URDF link names.
TIP_LINKS = ("link_30_tip", "link_70_tip", "link_110_tip", "link_150_tip")


def _find_descendant(node, name: str):
    if node.name == name:
        return node
    for child in node.children:
        match = _find_descendant(child, name)
        if match is not None:
            return match
    return None


def _stage_assets(asset_root: Path, project_root: Path) -> Path:
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    destination_root = project_root / PROJECT_ASSET_ROOT
    for entry in manifest["files"]:
        relative = Path(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"asset manifest contains unsafe path {relative}")
        source = asset_root / relative
        if (
            not source.is_file()
            or source.stat().st_size != int(entry["size"])
            or hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]
        ):
            raise RuntimeError(f"Allegro asset failed manifest validation: {relative}")
        destination = destination_root / relative
        if source.resolve() == destination.resolve():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return destination_root


def _create_soft_grasp_object():
    half_extent_x = 0.022
    half_extent_y = 0.065
    half_extent_z = 0.012
    vertices = [
        (-half_extent_x, -half_extent_y, -half_extent_z),
        (half_extent_x, -half_extent_y, -half_extent_z),
        (-half_extent_x, half_extent_y, -half_extent_z),
        (half_extent_x, half_extent_y, -half_extent_z),
        (-half_extent_x, -half_extent_y, half_extent_z),
        (half_extent_x, -half_extent_y, half_extent_z),
        (-half_extent_x, half_extent_y, half_extent_z),
        (half_extent_x, half_extent_y, half_extent_z),
    ]
    tetrahedra = [
        (0, 1, 3, 7),
        (0, 3, 2, 7),
        (0, 2, 6, 7),
        (0, 6, 4, 7),
        (0, 4, 5, 7),
        (0, 5, 1, 7),
    ]
    mesh = gobot.TetrahedralMesh()
    mesh.vertices = vertices
    mesh.tetrahedra = tetrahedra
    mesh.surface_triangles = []
    mesh.validate()

    body = gobot.create_node("DeformableBody3D", "soft_grasp_object")
    body.mesh = mesh
    body.position = (0.072, 0.0, -0.0072)
    body.density = 300.0
    body.young_modulus = 20000.0
    body.poisson_ratio = 0.42
    body.damping = 0.1
    body.self_collision_enabled = False
    return body


def build_scene(asset_root: Path):
    urdf = asset_root / "urdf" / "allegro_hand_description_left.urdf"
    license_path = asset_root / "LICENSE"
    if not urdf.is_file() or not license_path.is_file():
        raise RuntimeError(
            "Allegro assets are missing; run download_allegro_assets.py explicitly"
        )

    imported = gobot.load_scene(str(urdf))
    robot = imported.root
    robot.name = "allegro_left"
    robot.source_path = (
        "res://assets/wonik_allegro/urdf/allegro_hand_description_left.urdf"
    )
    shared_sensor = create_fabricated_sensor(
        name="tactile_0", resolution=(120, 160)
    )
    shared_sensor.config.damping = 20.0
    for index, tip_name in enumerate(TIP_LINKS):
        tip = _find_descendant(robot, tip_name)
        if tip is None:
            raise RuntimeError(f"Allegro URDF has no fingertip link {tip_name!r}")
        if index == 0:
            sensor = shared_sensor
        else:
            sensor = gobot.create_node("TactileSensor3D", f"tactile_{index}")
            sensor.config = shared_sensor.config
            sensor.gel_mesh = shared_sensor.gel_mesh
        sensor.position = (0.0, 0.0, 0.012)
        tip.add_child(sensor)

    root = gobot.create_node("Node3D", "allegro_tactile_grasp")
    root.add_child(robot)
    root.add_child(_create_soft_grasp_object())
    return root, imported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets",
        type=Path,
        default=HERE / "assets" / "wonik_allegro",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "allegro_tactile_grasp.jscn",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    gobot.app.context().set_project_path(str(output.parent))
    staged_assets = _stage_assets(
        args.assets.expanduser().resolve(), output.parent
    )
    root, imported = build_scene(staged_assets)
    gobot.save_scene(root, "res://" + output.name)
    del imported
    print(output)


if __name__ == "__main__":
    main()
