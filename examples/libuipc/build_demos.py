"""Build the Gobot-native libuipc demonstration scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Callable

import gobot

if __package__:
    from .fr3_robot import create_fr3_robot
else:
    from fr3_robot import create_fr3_robot


HERE = Path(__file__).resolve().parent
PLAY_SCRIPT_PATH = "res://libuipc_demo.py"
PLAY_SCRIPT_RESOURCE_ID = "libuipc_demo_script"
FR3_SOFT_BOX_SIZE = (0.050, 0.030, 0.025)
FR3_SOFT_BOX_CENTER = (0.50, 0.0, 0.211351)
FR3_SOFT_BOX_TABLE_GAP = 0.00025


def _box_tetrahedral_mesh(
    size: tuple[float, float, float],
    cells: tuple[int, int, int] = (3, 3, 3),
):
    size_x, size_y, size_z = (float(value) for value in size)
    cells_x, cells_y, cells_z = (int(value) for value in cells)
    if min(size_x, size_y, size_z) <= 0.0 or min(cells_x, cells_y, cells_z) <= 0:
        raise ValueError("tetrahedral box dimensions and cell counts must be positive")
    nx = cells_x + 1
    ny = cells_y + 1

    def vertex_index(ix: int, iy: int, iz: int) -> int:
        return iz * nx * ny + iy * nx + ix

    vertices = []
    for iz in range(cells_z + 1):
        z = -0.5 * size_z + size_z * iz / cells_z
        for iy in range(cells_y + 1):
            y = -0.5 * size_y + size_y * iy / cells_y
            for ix in range(cells_x + 1):
                x = -0.5 * size_x + size_x * ix / cells_x
                vertices.append((x, y, z))

    tetrahedra = []
    for iz in range(cells_z):
        for iy in range(cells_y):
            for ix in range(cells_x):
                v000 = vertex_index(ix, iy, iz)
                v100 = vertex_index(ix + 1, iy, iz)
                v010 = vertex_index(ix, iy + 1, iz)
                v110 = vertex_index(ix + 1, iy + 1, iz)
                v001 = vertex_index(ix, iy, iz + 1)
                v101 = vertex_index(ix + 1, iy, iz + 1)
                v011 = vertex_index(ix, iy + 1, iz + 1)
                v111 = vertex_index(ix + 1, iy + 1, iz + 1)
                tetrahedra.extend(
                    (
                        (v000, v100, v110, v111),
                        (v000, v110, v010, v111),
                        (v000, v010, v011, v111),
                        (v000, v011, v001, v111),
                        (v000, v001, v101, v111),
                        (v000, v101, v100, v111),
                    )
                )

    mesh = gobot.TetrahedralMesh()
    mesh.vertices = vertices
    mesh.tetrahedra = tetrahedra
    mesh.surface_triangles = []
    mesh.validate()
    return mesh


def _soft_box(
    name: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    *,
    young_modulus: float = 4.0e4,
    cells: tuple[int, int, int] = (3, 3, 3),
):
    body = gobot.create_node("DeformableBody3D", name)
    body.mesh = _box_tetrahedral_mesh(size, cells)
    body.position = position
    body.density = 650.0
    body.young_modulus = young_modulus
    body.poisson_ratio = 0.38
    body.damping = 0.08
    body.self_collision_enabled = False
    return body


def _box_visual(
    parent,
    name: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float, float],
    *,
    rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    visual = gobot.create_box_visual(name, size, position)
    visual.rotation_degrees = rotation_degrees
    visual.surface_color = color
    parent.add_child(visual)
    return visual


def _box_collision(
    parent,
    name: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    *,
    rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sliding_friction: float | None = None,
):
    collision = gobot.create_box_collision(name, size, position)
    collision.rotation_degrees = rotation_degrees
    if sliding_friction is not None:
        collision.physics_material = {
            "sliding_friction": sliding_friction,
            "torsional_friction": 0.005,
            "rolling_friction": 0.0001,
        }
    parent.add_child(collision)
    return collision


def _empty_link(
    name: str,
    position: tuple[float, float, float],
    mass: float,
    inertia_size: tuple[float, float, float],
):
    link = gobot.create_node("Link3D", name)
    link.position = position
    link.has_inertial = True
    link.mass = mass
    size_x, size_y, size_z = inertia_size
    link.inertia_diagonal = (
        mass * (size_y * size_y + size_z * size_z) / 12.0,
        mass * (size_x * size_x + size_z * size_z) / 12.0,
        mass * (size_x * size_x + size_y * size_y) / 12.0,
    )
    return link


def _scene_root(name: str):
    root = gobot.create_node("Node3D", name)
    colliders = gobot.create_node("Robot3D", "kinematic_colliders")
    root.add_child(colliders)
    return root, colliders


def _fr3_soft_grasp_scene():
    root, colliders = _scene_root("libuipc_fr3_soft_grasp")
    workspace = _empty_link("workspace", (0.0, 0.0, 0.0), 100.0, (1.3, 1.0, 0.2))
    _box_visual(
        workspace,
        "ground_visual",
        (1.30, 1.00, 0.05),
        (0.25, 0.0, -0.025),
        (0.16, 0.19, 0.22, 1.0),
    )
    _box_collision(
        workspace,
        "ground_collision",
        (1.30, 1.00, 0.05),
        (0.25, 0.0, -0.025),
    )
    table_top = (
        FR3_SOFT_BOX_CENTER[2]
        - 0.5 * FR3_SOFT_BOX_SIZE[2]
        - FR3_SOFT_BOX_TABLE_GAP
    )
    workbench_size = (0.36, 0.34, table_top)
    workbench_center = (0.50, 0.0, 0.5 * table_top)
    _box_visual(
        workspace,
        "workbench_visual",
        workbench_size,
        workbench_center,
        (0.32, 0.36, 0.39, 1.0),
    )
    _box_collision(
        workspace,
        "workbench_collision",
        workbench_size,
        workbench_center,
    )
    colliders.add_child(workspace)

    root.add_child(
        _soft_box(
            "soft_workpiece",
            FR3_SOFT_BOX_SIZE,
            FR3_SOFT_BOX_CENTER,
            young_modulus=3.0e4,
            cells=(5, 3, 3),
        )
    )

    root.add_child(create_fr3_robot())
    return root


def _normalize_scene_ids(scene_path: Path) -> None:
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    resources = scene.get("__EXT_RESOURCES__")
    nodes = scene.get("__NODES__")
    if not isinstance(resources, list) or not isinstance(nodes, list):
        raise RuntimeError(f"generated scene {scene_path.name} has no node/resource table")
    resources.sort(
        key=lambda entry: (
            entry.get("__PATH__") != PLAY_SCRIPT_PATH,
            str(entry.get("__PATH__", "")),
        )
    )
    external_replacements: dict[str, str] = {}
    external_type_counts: dict[str, int] = {}
    for entry in resources:
        old_id = str(entry["__ID__"])
        if entry.get("__PATH__") == PLAY_SCRIPT_PATH:
            new_id = PLAY_SCRIPT_RESOURCE_ID
        else:
            resource_type = str(entry["__TYPE__"]).lower()
            index = external_type_counts.get(resource_type, 0)
            external_type_counts[resource_type] = index + 1
            new_id = f"external_{resource_type}_{index}"
        external_replacements[old_id] = new_id
        entry["__ID__"] = new_id

    subresources = scene.get("__SUB_RESOURCES__", [])
    if not isinstance(subresources, list):
        raise RuntimeError(f"generated scene {scene_path.name} has no subresource table")
    type_counts: dict[str, int] = {}
    replacements: dict[str, str] = {}
    for entry in subresources:
        resource_type = str(entry["__TYPE__"])
        index = type_counts.get(resource_type, 0)
        type_counts[resource_type] = index + 1
        replacements[str(entry["__ID__"])] = f"{resource_type}_{index}"

    def rewrite(value):
        if isinstance(value, str):
            for old, new in external_replacements.items():
                if value == f"ExtResource({old})":
                    return f"ExtResource({new})"
            for old, new in replacements.items():
                if value == f"SubResource({old})":
                    return f"SubResource({new})"
            return value
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    scene = rewrite(scene)
    for entry in scene["__SUB_RESOURCES__"]:
        entry["__ID__"] = replacements[str(entry["__ID__"])]
    scene_path.write_text(
        json.dumps(scene, indent=4, ensure_ascii=True) + "\n", encoding="utf-8"
    )


SCENES: tuple[tuple[str, Callable[[], object]], ...] = (
    ("fr3_brick_grasp.jscn", _fr3_soft_grasp_scene),
)


def _stage_demo_project(output_dir: Path) -> None:
    if output_dir == HERE:
        return
    shutil.copy2(HERE / "libuipc_demo.py", output_dir / "libuipc_demo.py")
    shutil.copy2(HERE / "libuipc_runtime.py", output_dir / "libuipc_runtime.py")
    shutil.copy2(HERE / "fr3_grasp.py", output_dir / "fr3_grasp.py")
    shutil.copy2(HERE / "project.gobot", output_dir / "project.gobot")
    shutil.copytree(HERE / "assets", output_dir / "assets", dirs_exist_ok=True)


def build_demos(output_dir: Path = HERE) -> tuple[Path, ...]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _stage_demo_project(output_dir)
    destinations = []
    for filename, create_scene in SCENES:
        gobot.app.context().set_project_path(str(HERE))
        root = create_scene()
        destination = output_dir / filename
        gobot.app.context().set_project_path(str(output_dir))
        root.set("script", PLAY_SCRIPT_PATH)
        gobot.save_scene(root, "res://" + filename)
        _normalize_scene_ids(destination)
        destinations.append(destination)
    return tuple(destinations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    for destination in build_demos(args.output_dir):
        print(destination)


if __name__ == "__main__":
    main()
