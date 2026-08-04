"""Deterministic tactile-gel fabrication used by Gobot examples and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FabricatedGel:
    vertices: tuple[tuple[float, float, float], ...]
    tetrahedra: tuple[tuple[int, int, int, int], ...]
    coat_vertex_indices: tuple[int, ...]
    stick_vertex_indices: tuple[int, ...]
    marker_positions: tuple[tuple[float, float], ...]
    marker_tetrahedra: tuple[int, ...]
    marker_barycentric: tuple[tuple[float, float, float, float], ...]


def fabricate_box_gel(
    *,
    size: tuple[float, float] = (0.03, 0.03),
    thickness: float = 0.003,
    cells: tuple[int, int] = (9, 9),
    resolution: tuple[int, int] = (240, 320),
) -> FabricatedGel:
    """Build a two-layer box gel with 8x8 markers on its contact surface."""

    size_x, size_y = (float(value) for value in size)
    cells_x, cells_y = (int(value) for value in cells)
    height, width = (int(value) for value in resolution)
    thickness = float(thickness)
    if size_x <= 0.0 or size_y <= 0.0 or thickness <= 0.0:
        raise ValueError("gel dimensions must be positive")
    if cells_x < 9 or cells_y < 9:
        raise ValueError("gel fabrication requires at least 9x9 surface cells")
    if height <= 0 or width <= 0:
        raise ValueError("tactile resolution must be positive")

    nx = cells_x + 1
    ny = cells_y + 1

    def vertex_index(ix: int, iy: int, iz: int) -> int:
        return iz * nx * ny + iy * nx + ix

    vertices: list[tuple[float, float, float]] = []
    for iz in range(2):
        z = thickness * iz
        for iy in range(ny):
            y = -0.5 * size_y + size_y * iy / cells_y
            for ix in range(nx):
                x = -0.5 * size_x + size_x * ix / cells_x
                vertices.append((x, y, z))

    tetrahedra: list[tuple[int, int, int, int]] = []
    for iy in range(cells_y):
        for ix in range(cells_x):
            v000 = vertex_index(ix, iy, 0)
            v100 = vertex_index(ix + 1, iy, 0)
            v010 = vertex_index(ix, iy + 1, 0)
            v110 = vertex_index(ix + 1, iy + 1, 0)
            v001 = vertex_index(ix, iy, 1)
            v101 = vertex_index(ix + 1, iy, 1)
            v011 = vertex_index(ix, iy + 1, 1)
            v111 = vertex_index(ix + 1, iy + 1, 1)
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

    first_tet: dict[int, tuple[int, int]] = {}
    for tetrahedron_index, tetrahedron in enumerate(tetrahedra):
        for local_index, index in enumerate(tetrahedron):
            first_tet.setdefault(index, (tetrahedron_index, local_index))

    marker_positions: list[tuple[float, float]] = []
    marker_tetrahedra: list[int] = []
    marker_barycentric: list[tuple[float, float, float, float]] = []
    pixel_margin_x = 0.1 * float(width - 1)
    pixel_margin_y = 0.1 * float(height - 1)
    for marker_y, iy in enumerate(range(1, 9)):
        for marker_x, ix in enumerate(range(1, 9)):
            gel_vertex = vertex_index(ix, iy, 0)
            tetrahedron_index, local_index = first_tet[gel_vertex]
            weights = [0.0, 0.0, 0.0, 0.0]
            weights[local_index] = 1.0
            marker_positions.append(
                (
                    pixel_margin_x
                    + (float(width - 1) - 2.0 * pixel_margin_x) * marker_x / 7.0,
                    pixel_margin_y
                    + (float(height - 1) - 2.0 * pixel_margin_y) * marker_y / 7.0,
                )
            )
            marker_tetrahedra.append(tetrahedron_index)
            marker_barycentric.append(tuple(weights))

    return FabricatedGel(
        vertices=tuple(vertices),
        tetrahedra=tuple(tetrahedra),
        coat_vertex_indices=tuple(vertex_index(ix, iy, 0) for iy in range(ny) for ix in range(nx)),
        stick_vertex_indices=tuple(vertex_index(ix, iy, 1) for iy in range(ny) for ix in range(nx)),
        marker_positions=tuple(marker_positions),
        marker_tetrahedra=tuple(marker_tetrahedra),
        marker_barycentric=tuple(marker_barycentric),
    )


def create_fabricated_sensor(
    *,
    name: str = "tactile_sensor",
    resolution: tuple[int, int] = (240, 320),
) -> Any:
    """Create one fully configured tactile sensor node."""

    import gobot

    fabricated = fabricate_box_gel(resolution=resolution)
    mesh = gobot.TetrahedralMesh()
    mesh.vertices = fabricated.vertices
    mesh.tetrahedra = fabricated.tetrahedra
    mesh.surface_triangles = []
    mesh.validate()

    config = gobot.TactileSensorConfig()
    config.image_height = int(resolution[0])
    config.image_width = int(resolution[1])
    config.coat_vertex_indices = fabricated.coat_vertex_indices
    config.stick_vertex_indices = fabricated.stick_vertex_indices
    config.marker_positions = fabricated.marker_positions
    config.marker_tetrahedra = fabricated.marker_tetrahedra
    config.marker_barycentric = fabricated.marker_barycentric

    sensor = gobot.create_node("TactileSensor3D", name)
    sensor.config = config
    sensor.gel_mesh = mesh
    return sensor


def create_fabricated_sensor_scene(
    *,
    name: str = "tactile_sensor",
    resolution: tuple[int, int] = (240, 320),
) -> Any:
    """Create a Gobot scene containing one fully configured tactile sensor."""

    import gobot

    sensor = create_fabricated_sensor(name=name, resolution=resolution)
    root = gobot.create_node("Node3D", "fabricated_tactile_sensor")
    root.add_child(sensor)
    return root


__all__: list[str] = []
