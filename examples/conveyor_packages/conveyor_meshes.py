"""Parcel volume and shell meshes shared by scene authoring and Play preview."""

import math
from typing import Any

import gobot
import numpy as np


def _positive_tetrahedron(
    vertices: list[tuple[float, float, float]],
    indices: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    points = np.asarray([vertices[index] for index in indices])
    signed_volume = float(
        (points[1] - points[0])
        @ np.cross(points[2] - points[0], points[3] - points[0])
    )
    if abs(signed_volume) <= 1.0e-14:
        raise ValueError("soft package mesh contains a degenerate tetrahedron")
    if signed_volume < 0.0:
        return (indices[1], indices[0], indices[2], indices[3])
    return indices


def package_mesh(
    size: tuple[float, float, float],
    cells: tuple[int, int, int] = (6, 4, 3),
    *,
    side_rounding: float = 0.0,
) -> Any:
    cells_x, cells_y, cells_z = cells
    if min(*size, cells_x, cells_y, cells_z) <= 0:
        raise ValueError("soft package dimensions and cells must be positive")
    if not 0.0 <= side_rounding < 0.5:
        raise ValueError("soft package side rounding must be in [0, 0.5)")
    half_x, half_y, half_z = (0.5 * float(value) for value in size)
    nx = cells_x + 1
    ny = cells_y + 1

    def vertex_index(ix: int, iy: int, iz: int) -> int:
        return iz * nx * ny + iy * nx + ix

    vertices: list[tuple[float, float, float]] = []
    for iz in range(cells_z + 1):
        w = 2.0 * iz / cells_z - 1.0
        for iy in range(cells_y + 1):
            v = 2.0 * iy / cells_y - 1.0
            for ix in range(cells_x + 1):
                u = 2.0 * ix / cells_x - 1.0
                center_x = max(0.0, 1.0 - u * u)
                center_y = max(0.0, 1.0 - v * v)
                if side_rounding > 0.0:
                    # Map the regular grid to a rounded pillow volume. The
                    # middle layers carry the full footprint, while the top
                    # and bottom layers draw inward. Height also fades toward
                    # the sealed perimeter, leaving a broad, inflated center.
                    layer_side_scale = 1.0 - side_rounding * w * w
                    x = (
                        half_x
                        * u
                        * (1.0 - side_rounding * (1.0 - center_y))
                        * layer_side_scale
                    )
                    y = (
                        half_y
                        * v
                        * (1.0 - side_rounding * (1.0 - center_x))
                        * layer_side_scale
                    )
                    center_fraction = math.sqrt(center_x * center_y)
                    # Match the asymmetric film cavity: the contents have a
                    # shallow underside and most of their loft above it. This
                    # preserves clearance from both sheets at initialization.
                    bottom = -(
                        0.04 + 0.15 * center_fraction
                    ) * size[2]
                    top = (
                        0.44 + 0.40 * center_fraction
                    ) * size[2]
                    layer = 0.5 * (w + 1.0)
                    z = bottom + layer * (top - bottom)
                    vertices.append((x, y, z))
                    continue
                # Keep a rounded rectangular footprint, but form the package
                # from asymmetric top and bottom sheets.  A symmetric solid
                # pillow leaves a bowl-shaped underside when it bridges the
                # table and belt; a real filled mailer has a broad, shallow
                # contact patch and most of its loft above that patch.
                x = half_x * u * (0.94 + 0.06 * center_y)
                y = half_y * v * (0.94 + 0.06 * center_x)
                distance_from_seam = max(
                    0.0, min(1.0 - abs(u), 1.0 - abs(v))
                )
                content_fraction = min(1.0, distance_from_seam / 0.16)
                content_fraction = content_fraction * content_fraction * (
                    3.0 - 2.0 * content_fraction
                )
                crown = 1.0 - 0.05 * (0.55 * u * u + 0.45 * v * v)
                edge_band = 4.0 * content_fraction * (
                    1.0 - content_fraction
                )
                wrinkle = (
                    0.025
                    * size[2]
                    * edge_band
                    * math.sin(5.0 * math.pi * u + 3.0 * math.pi * v)
                )
                center_crease = (
                    0.025
                    * size[2]
                    * math.exp(-55.0 * u * u)
                    * max(0.0, 1.0 - 1.35 * abs(v))
                    * content_fraction
                )
                seam_half_thickness = 0.018 * size[2]
                bottom = (
                    -seam_half_thickness
                    - 0.14 * size[2] * content_fraction * crown
                    + 0.20 * wrinkle
                )
                top = (
                    seam_half_thickness
                    + 0.82 * size[2] * content_fraction * crown
                    + wrinkle
                    - center_crease
                )
                layer = 0.5 * (w + 1.0)
                z = bottom + layer * (top - bottom)
                vertices.append((x, y, z))

    tetrahedra: list[tuple[int, int, int, int]] = []
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
                for tetrahedron in (
                    (v000, v100, v110, v111),
                    (v000, v110, v010, v111),
                    (v000, v010, v011, v111),
                    (v000, v011, v001, v111),
                    (v000, v001, v101, v111),
                    (v000, v101, v100, v111),
                ):
                    tetrahedra.append(
                        _positive_tetrahedron(vertices, tetrahedron)
                    )

    mesh = gobot.TetrahedralMesh()
    mesh.vertices = vertices
    mesh.tetrahedra = tetrahedra
    mesh.surface_triangles = []
    mesh.validate()
    return mesh


def mailer_shell_mesh(
    size: tuple[float, float, float],
    cells: tuple[int, int] = (18, 13),
) -> Any:
    cells_x, cells_y = cells
    if min(*size, cells_x, cells_y) <= 0:
        raise ValueError("soft mailer shell dimensions and cells must be positive")
    half_x, half_y, height = (
        0.5 * float(size[0]),
        0.5 * float(size[1]),
        float(size[2]),
    )
    nx = cells_x + 1
    ny = cells_y + 1
    layer_stride = nx * ny

    def vertex_index(layer: int, ix: int, iy: int) -> int:
        return layer * layer_stride + iy * nx + ix

    vertices: list[tuple[float, float, float]] = []
    for layer in range(2):
        for iy in range(ny):
            v = 2.0 * iy / cells_y - 1.0
            for ix in range(nx):
                u = 2.0 * ix / cells_x - 1.0
                center_x = max(0.0, 1.0 - u * u)
                center_y = max(0.0, 1.0 - v * v)
                x = half_x * u * (0.93 + 0.07 * center_y)
                y = half_y * v * (0.93 + 0.07 * center_x)
                seam_distance = max(
                    0.0, min(1.0 - abs(u), 1.0 - abs(v))
                )
                fill = min(1.0, seam_distance / 0.24)
                fill = fill * fill * (3.0 - 2.0 * fill)
                edge_band = 4.0 * fill * (1.0 - fill)
                # A heat-sealed perimeter starts coplanar. Film irregularity
                # belongs on the shoulder just inside that flange; placing the
                # largest opposite-signed wave directly on the free seam made
                # it curl into a tube as soon as gravity loaded the shell.
                seam_wave = (
                    0.003
                    * height
                    * edge_band
                    * math.sin(4.0 * math.pi * u - 3.0 * math.pi * v)
                )
                if layer == 0:
                    bottom_wrinkle = (
                        0.018
                        * height
                        * edge_band
                        * math.sin(5.0 * math.pi * u + 2.0 * math.pi * v)
                    )
                    z = (
                        -0.025 * height
                        - 0.245 * height * fill
                        + bottom_wrinkle
                        - seam_wave
                    )
                else:
                    diagonal_crease = (
                        0.085
                        * height
                        * math.exp(-90.0 * (u + 0.48 * v - 0.18) ** 2)
                        * fill
                    )
                    cross_crease = (
                        0.050
                        * height
                        * math.exp(-110.0 * (u - 0.60 * v + 0.28) ** 2)
                        * fill
                    )
                    top_wrinkle = (
                        0.040
                        * height
                        * edge_band
                        * math.sin(5.0 * math.pi * u + 3.0 * math.pi * v)
                    )
                    crown = 0.96 - 0.07 * (0.55 * u * u + 0.45 * v * v)
                    z = (
                        0.025 * height
                        + 0.785 * height * fill * crown
                        + top_wrinkle
                        - diagonal_crease
                        - cross_crease
                        + seam_wave
                    )
                vertices.append((x, y, z))

    triangles: list[tuple[int, int, int]] = []
    for iy in range(cells_y):
        for ix in range(cells_x):
            bottom_00 = vertex_index(0, ix, iy)
            bottom_10 = vertex_index(0, ix + 1, iy)
            bottom_01 = vertex_index(0, ix, iy + 1)
            bottom_11 = vertex_index(0, ix + 1, iy + 1)
            top_00 = vertex_index(1, ix, iy)
            top_10 = vertex_index(1, ix + 1, iy)
            top_01 = vertex_index(1, ix, iy + 1)
            top_11 = vertex_index(1, ix + 1, iy + 1)
            # Top faces +Z; bottom faces -Z.
            triangles.extend(
                (
                    (top_00, top_10, top_11),
                    (top_00, top_11, top_01),
                    (bottom_00, bottom_11, bottom_10),
                    (bottom_00, bottom_01, bottom_11),
                )
            )

    perimeter = (
        tuple((ix, 0) for ix in range(nx))
        + tuple((cells_x, iy) for iy in range(1, ny))
        + tuple((ix, cells_y) for ix in range(cells_x - 1, -1, -1))
        + tuple((0, iy) for iy in range(cells_y - 1, 0, -1))
    )
    for edge_index, (ix, iy) in enumerate(perimeter):
        next_ix, next_iy = perimeter[(edge_index + 1) % len(perimeter)]
        bottom_a = vertex_index(0, ix, iy)
        bottom_b = vertex_index(0, next_ix, next_iy)
        top_a = vertex_index(1, ix, iy)
        top_b = vertex_index(1, next_ix, next_iy)
        triangles.extend(
            ((bottom_a, bottom_b, top_b), (bottom_a, top_b, top_a))
        )

    mesh = gobot.SurfaceMesh()
    mesh.vertices = vertices
    mesh.triangles = triangles
    mesh.validate()
    return mesh
