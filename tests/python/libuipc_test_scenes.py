"""Runtime-generated scenes used only by native libuipc regression tests."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import gobot


TEST_SCENE_NAMES = (
    "soft_cube_stack.jscn",
    "soft_cube_press.jscn",
)


def _box_tetrahedral_mesh(
    size: tuple[float, float, float],
    cells: tuple[int, int, int] = (3, 3, 3),
):
    size_x, size_y, size_z = (float(value) for value in size)
    cells_x, cells_y, cells_z = (int(value) for value in cells)
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


def _soft_box(
    name: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    *,
    young_modulus: float = 4.0e4,
):
    body = gobot.create_node("DeformableBody3D", name)
    body.mesh = _box_tetrahedral_mesh(size)
    body.position = position
    body.density = 650.0
    body.young_modulus = young_modulus
    body.poisson_ratio = 0.38
    body.damping = 0.08
    body.self_collision_enabled = False
    return body


def _kinematic_box(
    robot,
    name: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
):
    mass = 10.0
    size_x, size_y, size_z = size
    link = gobot.create_node("Link3D", name)
    link.position = position
    link.has_inertial = True
    link.mass = mass
    link.inertia_diagonal = (
        mass * (size_y * size_y + size_z * size_z) / 12.0,
        mass * (size_x * size_x + size_z * size_z) / 12.0,
        mass * (size_x * size_x + size_y * size_y) / 12.0,
    )
    link.add_child(gobot.create_box_collision(f"{name}_collision", size))
    robot.add_child(link)


def _scene_root(name: str):
    root = gobot.create_node("Node3D", name)
    colliders = gobot.create_node("Robot3D", "kinematic_colliders")
    root.add_child(colliders)
    return root, colliders


def _stack_scene():
    root, colliders = _scene_root("libuipc_soft_stack_test")
    root.add_child(
        _soft_box("lower_cube", (0.18, 0.18, 0.18), (-0.035, 0.0, 0.25))
    )
    root.add_child(
        _soft_box("upper_cube", (0.16, 0.16, 0.16), (0.045, 0.0, 0.50))
    )
    _kinematic_box(
        colliders,
        "ground",
        (1.0, 0.75, 0.05),
        (0.0, 0.0, -0.025),
    )
    return root


def _press_scene():
    root, colliders = _scene_root("libuipc_soft_press_test")
    root.add_child(
        _soft_box(
            "compression_block",
            (0.22, 0.22, 0.16),
            (0.0, 0.0, 0.10),
            young_modulus=2.5e4,
        )
    )
    _kinematic_box(
        colliders,
        "ground",
        (0.8, 0.65, 0.05),
        (0.0, 0.0, -0.025),
    )
    _kinematic_box(
        colliders,
        "press_head",
        (0.30, 0.30, 0.06),
        (0.0, 0.0, 0.34),
    )
    return root


_SCENE_BUILDERS: dict[str, Callable[[], object]] = {
    "soft_cube_stack.jscn": _stack_scene,
    "soft_cube_press.jscn": _press_scene,
}


def build_libuipc_test_scene(output_dir: Path, scene_name: str) -> Path:
    try:
        create_scene = _SCENE_BUILDERS[scene_name]
    except KeyError as error:
        raise ValueError(f"unknown libuipc test scene {scene_name!r}") from error
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gobot.app.context().set_project_path(str(output_dir))
    destination = output_dir / scene_name
    gobot.save_scene(create_scene(), "res://" + scene_name)
    return destination


__all__ = ["TEST_SCENE_NAMES", "build_libuipc_test_scene"]
