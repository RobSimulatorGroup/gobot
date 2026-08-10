"""Build the authored scene used by the MuJoCo Warp + libuipc example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import gobot


HERE = Path(__file__).resolve().parent
SCENE_NAME = "soft_press_batch.jscn"
PLAY_SCRIPT_NAME = "mujoco_libuipc_play.py"
PLAY_SCRIPT_PATH = "res://" + PLAY_SCRIPT_NAME
PLAY_SCRIPT_RESOURCE_ID = "mujoco_libuipc_play_script"


def _box_tetrahedral_mesh(
    size: tuple[float, float, float],
    cells: tuple[int, int, int] = (2, 2, 2),
):
    size_x, size_y, size_z = (float(value) for value in size)
    cells_x, cells_y, cells_z = (int(value) for value in cells)
    if min(size_x, size_y, size_z) <= 0.0 or min(
        cells_x, cells_y, cells_z
    ) <= 0:
        raise ValueError("tetrahedral box dimensions and cells must be positive")
    nx = cells_x + 1
    ny = cells_y + 1

    def vertex_index(ix: int, iy: int, iz: int) -> int:
        return iz * nx * ny + iy * nx + ix

    vertices = [
        (
            -0.5 * size_x + size_x * ix / cells_x,
            -0.5 * size_y + size_y * iy / cells_y,
            -0.5 * size_z + size_z * iz / cells_z,
        )
        for iz in range(cells_z + 1)
        for iy in range(cells_y + 1)
        for ix in range(cells_x + 1)
    ]
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


def _set_box_inertia(link, mass: float, size: tuple[float, float, float]) -> None:
    size_x, size_y, size_z = size
    link.has_inertial = True
    link.mass = mass
    link.inertia_diagonal = (
        mass * (size_y * size_y + size_z * size_z) / 12.0,
        mass * (size_x * size_x + size_z * size_z) / 12.0,
        mass * (size_x * size_x + size_y * size_y) / 12.0,
    )


def _add_box_geometry(
    link,
    name: str,
    size: tuple[float, float, float],
    color: tuple[float, float, float, float],
) -> None:
    visual = gobot.create_box_visual(name + "_visual", size)
    visual.surface_color = color
    link.add_child(visual)
    collision = gobot.create_box_collision(name + "_collision", size)
    collision.friction = (0.8, 0.005, 0.0001)
    link.add_child(collision)


def create_scene():
    root = gobot.create_node("Node3D", "mujoco_libuipc_soft_press")

    colliders = gobot.create_node("Robot3D", "static_colliders")
    colliders.mode = gobot.RobotMode.Assembly
    ground_size = (0.8, 0.65, 0.05)
    ground = gobot.create_node("Link3D", "ground")
    ground.position = (0.0, 0.0, -0.025)
    _set_box_inertia(ground, 100.0, ground_size)
    _add_box_geometry(ground, "ground", ground_size, (0.16, 0.19, 0.22, 1.0))
    colliders.add_child(ground)
    root.add_child(colliders)

    press = gobot.create_node("Robot3D", "press")
    press.mode = gobot.RobotMode.Motion
    frame = gobot.create_node("Link3D", "press_frame")
    frame.role = gobot.LinkRole.VirtualRoot

    slide = gobot.create_node("Joint3D", "press_slide")
    slide.joint_type = gobot.JointType.Prismatic
    slide.parent_link = "press_frame"
    slide.child_link = "press_head"
    slide.axis = (0.0, 0.0, 1.0)
    slide.position = (0.0, 0.0, 0.34)
    slide.lower_limit = -0.17
    slide.upper_limit = 0.0
    slide.effort_limit = 2000.0
    slide.velocity_limit = 1.0
    slide.damping = 20.0
    slide.drive_mode = gobot.JointDriveMode.Position
    slide.drive_stiffness = 5000.0
    slide.drive_damping = 150.0
    slide.control_lower_limit = -0.17
    slide.control_upper_limit = 0.0

    head_size = (0.30, 0.30, 0.06)
    head = gobot.create_node("Link3D", "press_head")
    _set_box_inertia(head, 8.0, head_size)
    _add_box_geometry(head, "press_head", head_size, (0.82, 0.25, 0.12, 1.0))
    press.add_child(frame)
    frame.add_child(slide)
    slide.add_child(head)
    root.add_child(press)

    soft = gobot.create_node("DeformableBody3D", "compression_block")
    soft.mesh = _box_tetrahedral_mesh((0.22, 0.22, 0.16))
    soft.position = (0.0, 0.0, 0.10)
    soft.density = 650.0
    soft.young_modulus = 2.5e4
    soft.poisson_ratio = 0.38
    soft.damping = 0.08
    soft.self_collision_enabled = False
    root.add_child(soft)

    ground_coupling = gobot.create_node("PhysicsCoupling", "ground_coupling")
    ground_coupling.rigid_link_path = "../static_colliders/ground"
    ground_coupling.mode = gobot.PhysicsCouplingMode.OneWay
    ground_coupling.force_scale = 1.0
    ground_coupling.torque_scale = 1.0
    root.add_child(ground_coupling)

    press_coupling = gobot.create_node("PhysicsCoupling", "press_head_coupling")
    press_coupling.rigid_link_path = "../press/press_frame/press_slide/press_head"
    press_coupling.mode = gobot.PhysicsCouplingMode.TwoWay
    press_coupling.force_scale = 1.0
    press_coupling.torque_scale = 1.0
    root.add_child(press_coupling)
    return root


def _finalize_scene(scene_path: Path) -> None:
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    nodes = scene.get("__NODES__", [])
    roots = [entry for entry in nodes if int(entry.get("parent", -2)) == -1]
    if len(roots) != 1:
        raise RuntimeError("generated scene has no unique root node")
    resources = scene.get("__EXT_RESOURCES__", [])
    if not isinstance(resources, list):
        raise RuntimeError("generated scene has no external resource table")
    matches = [
        entry for entry in resources if entry.get("__PATH__") == PLAY_SCRIPT_PATH
    ]
    if len(matches) > 1:
        raise RuntimeError("generated scene has duplicate Play scripts")
    if matches:
        script = matches[0]
        script["__ID__"] = PLAY_SCRIPT_RESOURCE_ID
        script["__TYPE__"] = "PythonScript"
    else:
        resources.insert(
            0,
            {
                "__ID__": PLAY_SCRIPT_RESOURCE_ID,
                "__PATH__": PLAY_SCRIPT_PATH,
                "__TYPE__": "PythonScript",
            },
        )
    roots[0].setdefault("properties", {})["script"] = (
        f"ExtResource({PLAY_SCRIPT_RESOURCE_ID})"
    )

    subresources = scene.get("__SUB_RESOURCES__", [])
    if not isinstance(subresources, list):
        raise RuntimeError("generated scene has no subresource table")

    type_counts: dict[str, int] = {}
    replacements: dict[str, str] = {}
    for entry in subresources:
        resource_type = str(entry["__TYPE__"])
        index = type_counts.get(resource_type, 0)
        type_counts[resource_type] = index + 1
        replacements[str(entry["__ID__"])] = f"{resource_type}_{index}"

    def rewrite(value):
        if isinstance(value, str):
            for old, new in replacements.items():
                if value == f"SubResource({old})":
                    return f"SubResource({new})"
        elif isinstance(value, list):
            return [rewrite(item) for item in value]
        elif isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    scene = rewrite(scene)
    for entry in scene["__SUB_RESOURCES__"]:
        entry["__ID__"] = replacements[str(entry["__ID__"])]
    scene_path.write_text(
        json.dumps(scene, indent=4, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def _stage_project(output_dir: Path) -> None:
    if output_dir == HERE:
        return
    shutil.copy2(HERE / "build_scene.py", output_dir / "build_scene.py")
    shutil.copy2(HERE / PLAY_SCRIPT_NAME, output_dir / PLAY_SCRIPT_NAME)
    shutil.copy2(HERE / "project.gobot", output_dir / "project.gobot")


def build_scene(output_dir: Path = HERE) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _stage_project(output_dir)
    gobot.app.context().set_project_path(str(output_dir))
    destination = output_dir / SCENE_NAME
    gobot.save_scene(create_scene(), "res://" + SCENE_NAME)
    _finalize_scene(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    print(build_scene(args.output_dir))


if __name__ == "__main__":
    main()
