# SPDX-FileCopyrightText: Copyright (c) 2026 Gobot contributors
# SPDX-License-Identifier: Apache-2.0
"""Standard Warp 1.15 numerical primitives for the private IPC runtime.

This module is intentionally not imported by :mod:`gobot.ipc`.  Importing it
loads Warp and is reserved for the numerical runtime and explicitly enabled GPU
tests.  The implementation is independent of Taccel's precompiled kernels; it
uses the published IPC barrier, conservative advancement, and compressible
Neo-Hookean definitions directly.
"""

from __future__ import annotations

import operator
from typing import Any

import warp as wp


_DISTANCE_EPSILON = 1.0e-30
_INVALID_ENERGY = 1.0e300


@wp.func
def _clamp_unit(value: wp.float64) -> wp.float64:
    return wp.min(wp.max(value, wp.float64(0.0)), wp.float64(1.0))


@wp.func
def point_segment_distance_squared(
    point: wp.vec3d,
    endpoint_a: wp.vec3d,
    endpoint_b: wp.vec3d,
) -> wp.float64:
    edge = endpoint_b - endpoint_a
    edge_length_squared = wp.dot(edge, edge)
    if edge_length_squared <= wp.float64(_DISTANCE_EPSILON):
        offset = point - endpoint_a
        return wp.dot(offset, offset)
    coordinate = _clamp_unit(wp.dot(point - endpoint_a, edge) / edge_length_squared)
    offset = point - (endpoint_a + coordinate * edge)
    return wp.dot(offset, offset)


@wp.func
def point_triangle_distance_squared(
    point: wp.vec3d,
    vertex_a: wp.vec3d,
    vertex_b: wp.vec3d,
    vertex_c: wp.vec3d,
) -> wp.float64:
    """Squared distance between a point and a closed triangle.

    The region tests follow the triangle Voronoi partition.  Degenerate input
    falls back to the three closed edges instead of dividing by a vanishing
    triangle area.
    """

    edge_ab = vertex_b - vertex_a
    edge_ac = vertex_c - vertex_a
    normal = wp.cross(edge_ab, edge_ac)
    scale = wp.dot(edge_ab, edge_ab) + wp.dot(edge_ac, edge_ac)
    if wp.dot(normal, normal) <= wp.float64(_DISTANCE_EPSILON) * scale * scale:
        return wp.min(
            point_segment_distance_squared(point, vertex_a, vertex_b),
            wp.min(
                point_segment_distance_squared(point, vertex_b, vertex_c),
                point_segment_distance_squared(point, vertex_c, vertex_a),
            ),
        )

    offset_a = point - vertex_a
    d1 = wp.dot(edge_ab, offset_a)
    d2 = wp.dot(edge_ac, offset_a)
    if d1 <= wp.float64(0.0) and d2 <= wp.float64(0.0):
        return wp.dot(offset_a, offset_a)

    offset_b = point - vertex_b
    d3 = wp.dot(edge_ab, offset_b)
    d4 = wp.dot(edge_ac, offset_b)
    if d3 >= wp.float64(0.0) and d4 <= d3:
        return wp.dot(offset_b, offset_b)

    vertex_region_c = d1 * d4 - d3 * d2
    if vertex_region_c <= wp.float64(0.0) and d1 >= wp.float64(0.0) and d3 <= wp.float64(0.0):
        coordinate = d1 / (d1 - d3)
        offset = point - (vertex_a + coordinate * edge_ab)
        return wp.dot(offset, offset)

    offset_c = point - vertex_c
    d5 = wp.dot(edge_ab, offset_c)
    d6 = wp.dot(edge_ac, offset_c)
    if d6 >= wp.float64(0.0) and d5 <= d6:
        return wp.dot(offset_c, offset_c)

    vertex_region_b = d5 * d2 - d1 * d6
    if vertex_region_b <= wp.float64(0.0) and d2 >= wp.float64(0.0) and d6 <= wp.float64(0.0):
        coordinate = d2 / (d2 - d6)
        offset = point - (vertex_a + coordinate * edge_ac)
        return wp.dot(offset, offset)

    vertex_region_a = d3 * d6 - d5 * d4
    edge_bc_d1 = d4 - d3
    edge_bc_d2 = d5 - d6
    if (
        vertex_region_a <= wp.float64(0.0)
        and edge_bc_d1 >= wp.float64(0.0)
        and edge_bc_d2 >= wp.float64(0.0)
    ):
        coordinate = edge_bc_d1 / (edge_bc_d1 + edge_bc_d2)
        offset = point - (vertex_b + coordinate * (vertex_c - vertex_b))
        return wp.dot(offset, offset)

    inverse_sum = wp.float64(1.0) / (vertex_region_a + vertex_region_b + vertex_region_c)
    coordinate_b = vertex_region_b * inverse_sum
    coordinate_c = vertex_region_c * inverse_sum
    offset = point - (vertex_a + coordinate_b * edge_ab + coordinate_c * edge_ac)
    return wp.dot(offset, offset)


@wp.func
def edge_edge_distance_squared(
    endpoint_a0: wp.vec3d,
    endpoint_a1: wp.vec3d,
    endpoint_b0: wp.vec3d,
    endpoint_b1: wp.vec3d,
) -> wp.float64:
    """Squared distance between two closed line segments."""

    edge_a = endpoint_a1 - endpoint_a0
    edge_b = endpoint_b1 - endpoint_b0
    offset = endpoint_a0 - endpoint_b0
    aa = wp.dot(edge_a, edge_a)
    bb = wp.dot(edge_b, edge_b)
    ba = wp.dot(edge_b, offset)
    coordinate_a = wp.float64(0.0)
    coordinate_b = wp.float64(0.0)

    if aa <= wp.float64(_DISTANCE_EPSILON) and bb <= wp.float64(_DISTANCE_EPSILON):
        return wp.dot(offset, offset)
    if aa <= wp.float64(_DISTANCE_EPSILON):
        coordinate_b = _clamp_unit(ba / bb)
    else:
        ca = wp.dot(edge_a, offset)
        if bb <= wp.float64(_DISTANCE_EPSILON):
            coordinate_a = _clamp_unit(-ca / aa)
        else:
            ab = wp.dot(edge_a, edge_b)
            denominator = aa * bb - ab * ab
            if denominator > wp.float64(_DISTANCE_EPSILON) * aa * bb:
                coordinate_a = _clamp_unit((ab * ba - ca * bb) / denominator)
            coordinate_b = (ab * coordinate_a + ba) / bb
            if coordinate_b < wp.float64(0.0):
                coordinate_b = wp.float64(0.0)
                coordinate_a = _clamp_unit(-ca / aa)
            elif coordinate_b > wp.float64(1.0):
                coordinate_b = wp.float64(1.0)
                coordinate_a = _clamp_unit((ab - ca) / aa)

    closest_offset = offset + coordinate_a * edge_a - coordinate_b * edge_b
    return wp.dot(closest_offset, closest_offset)


@wp.func
def ipc_barrier_energy(
    distance_squared: wp.float64,
    activation_distance_squared: wp.float64,
    stiffness: wp.float64,
) -> wp.float64:
    if distance_squared >= activation_distance_squared:
        return wp.float64(0.0)
    distance_squared = wp.max(distance_squared, wp.float64(_DISTANCE_EPSILON))
    delta = distance_squared - activation_distance_squared
    return -stiffness * delta * delta * wp.log(distance_squared / activation_distance_squared)


@wp.func
def ipc_barrier_gradient(
    distance_squared: wp.float64,
    activation_distance_squared: wp.float64,
    stiffness: wp.float64,
) -> wp.float64:
    if distance_squared >= activation_distance_squared:
        return wp.float64(0.0)
    distance_squared = wp.max(distance_squared, wp.float64(_DISTANCE_EPSILON))
    delta = distance_squared - activation_distance_squared
    return -stiffness * (
        wp.float64(2.0) * delta * wp.log(distance_squared / activation_distance_squared)
        + delta * delta / distance_squared
    )


@wp.func
def ipc_barrier_hessian(
    distance_squared: wp.float64,
    activation_distance_squared: wp.float64,
    stiffness: wp.float64,
) -> wp.float64:
    if distance_squared >= activation_distance_squared:
        return wp.float64(0.0)
    distance_squared = wp.max(distance_squared, wp.float64(_DISTANCE_EPSILON))
    delta = distance_squared - activation_distance_squared
    return stiffness * (
        -wp.float64(2.0) * wp.log(distance_squared / activation_distance_squared)
        - wp.float64(4.0) * delta / distance_squared
        + delta * delta / (distance_squared * distance_squared)
    )


@wp.func
def smooth_friction_energy(speed: wp.float64, smoothing_speed: wp.float64) -> wp.float64:
    """C2-smoothed norm with zero energy at zero tangential speed."""

    speed = wp.max(speed, wp.float64(0.0))
    if speed >= smoothing_speed:
        return speed - smoothing_speed / wp.float64(3.0)
    normalized = speed / smoothing_speed
    return speed * normalized - speed * normalized * normalized / wp.float64(3.0)


@wp.func
def smooth_friction_gradient(speed: wp.float64, smoothing_speed: wp.float64) -> wp.float64:
    speed = wp.max(speed, wp.float64(0.0))
    if speed >= smoothing_speed:
        return wp.float64(1.0)
    normalized = speed / smoothing_speed
    return wp.float64(2.0) * normalized - normalized * normalized


@wp.func
def smooth_friction_hessian(speed: wp.float64, smoothing_speed: wp.float64) -> wp.float64:
    speed = wp.max(speed, wp.float64(0.0))
    if speed >= smoothing_speed:
        return wp.float64(0.0)
    return wp.float64(2.0) * (wp.float64(1.0) - speed / smoothing_speed) / smoothing_speed


@wp.func
def _point_triangle_speed_bound(
    point_velocity: wp.vec3d,
    velocity_a: wp.vec3d,
    velocity_b: wp.vec3d,
    velocity_c: wp.vec3d,
) -> wp.float64:
    return wp.max(
        wp.length(point_velocity - velocity_a),
        wp.max(
            wp.length(point_velocity - velocity_b),
            wp.length(point_velocity - velocity_c),
        ),
    )


@wp.func
def point_triangle_ccd_fraction(
    point_start: wp.vec3d,
    point_end: wp.vec3d,
    triangle_a_start: wp.vec3d,
    triangle_a_end: wp.vec3d,
    triangle_b_start: wp.vec3d,
    triangle_b_end: wp.vec3d,
    triangle_c_start: wp.vec3d,
    triangle_c_end: wp.vec3d,
    thickness: wp.float64,
    tolerance: wp.float64,
) -> wp.float64:
    point_velocity = point_end - point_start
    velocity_a = triangle_a_end - triangle_a_start
    velocity_b = triangle_b_end - triangle_b_start
    velocity_c = triangle_c_end - triangle_c_start
    speed_bound = _point_triangle_speed_bound(point_velocity, velocity_a, velocity_b, velocity_c)
    fraction = wp.float64(0.0)

    for _iteration in range(64):
        point = point_start + fraction * point_velocity
        triangle_a = triangle_a_start + fraction * velocity_a
        triangle_b = triangle_b_start + fraction * velocity_b
        triangle_c = triangle_c_start + fraction * velocity_c
        distance = wp.sqrt(
            wp.max(
                point_triangle_distance_squared(point, triangle_a, triangle_b, triangle_c),
                wp.float64(0.0),
            )
        )
        gap = distance - thickness
        if gap <= tolerance:
            return fraction
        if speed_bound <= wp.float64(_DISTANCE_EPSILON):
            return wp.float64(1.0)
        step = gap / speed_bound
        step = wp.max(step, tolerance / speed_bound)
        if fraction + step > wp.float64(1.0):
            return wp.float64(1.0)
        fraction += step
    return wp.float64(1.0)


@wp.func
def _edge_edge_speed_bound(
    velocity_a0: wp.vec3d,
    velocity_a1: wp.vec3d,
    velocity_b0: wp.vec3d,
    velocity_b1: wp.vec3d,
) -> wp.float64:
    return wp.max(
        wp.max(
            wp.length(velocity_a0 - velocity_b0),
            wp.length(velocity_a0 - velocity_b1),
        ),
        wp.max(
            wp.length(velocity_a1 - velocity_b0),
            wp.length(velocity_a1 - velocity_b1),
        ),
    )


@wp.func
def edge_edge_ccd_fraction(
    edge_a0_start: wp.vec3d,
    edge_a0_end: wp.vec3d,
    edge_a1_start: wp.vec3d,
    edge_a1_end: wp.vec3d,
    edge_b0_start: wp.vec3d,
    edge_b0_end: wp.vec3d,
    edge_b1_start: wp.vec3d,
    edge_b1_end: wp.vec3d,
    thickness: wp.float64,
    tolerance: wp.float64,
) -> wp.float64:
    velocity_a0 = edge_a0_end - edge_a0_start
    velocity_a1 = edge_a1_end - edge_a1_start
    velocity_b0 = edge_b0_end - edge_b0_start
    velocity_b1 = edge_b1_end - edge_b1_start
    speed_bound = _edge_edge_speed_bound(velocity_a0, velocity_a1, velocity_b0, velocity_b1)
    fraction = wp.float64(0.0)

    for _iteration in range(64):
        edge_a0 = edge_a0_start + fraction * velocity_a0
        edge_a1 = edge_a1_start + fraction * velocity_a1
        edge_b0 = edge_b0_start + fraction * velocity_b0
        edge_b1 = edge_b1_start + fraction * velocity_b1
        distance = wp.sqrt(
            wp.max(
                edge_edge_distance_squared(edge_a0, edge_a1, edge_b0, edge_b1),
                wp.float64(0.0),
            )
        )
        gap = distance - thickness
        if gap <= tolerance:
            return fraction
        if speed_bound <= wp.float64(_DISTANCE_EPSILON):
            return wp.float64(1.0)
        step = gap / speed_bound
        step = wp.max(step, tolerance / speed_bound)
        if fraction + step > wp.float64(1.0):
            return wp.float64(1.0)
        fraction += step
    return wp.float64(1.0)


@wp.func
def _edge_matrix(edge_1: wp.vec3d, edge_2: wp.vec3d, edge_3: wp.vec3d) -> wp.mat33d:
    return wp.mat33d(
        edge_1[0], edge_2[0], edge_3[0],
        edge_1[1], edge_2[1], edge_3[1],
        edge_1[2], edge_2[2], edge_3[2],
    )


@wp.func
def tet_neo_hookean_energy(
    vertex_0: wp.vec3d,
    vertex_1: wp.vec3d,
    vertex_2: wp.vec3d,
    vertex_3: wp.vec3d,
    inverse_rest_matrix: wp.mat33d,
    rest_volume: wp.float64,
    shear_modulus: wp.float64,
    lame_lambda: wp.float64,
) -> wp.float64:
    deformation_gradient = _edge_matrix(
        vertex_1 - vertex_0,
        vertex_2 - vertex_0,
        vertex_3 - vertex_0,
    ) * inverse_rest_matrix
    determinant = wp.determinant(deformation_gradient)
    if determinant <= wp.float64(0.0):
        return wp.float64(_INVALID_ENERGY)
    log_determinant = wp.log(determinant)
    frobenius_squared = wp.float64(0.0)
    for row in range(3):
        for column in range(3):
            value = deformation_gradient[row, column]
            frobenius_squared += value * value
    density = (
        shear_modulus * wp.float64(0.5) * (frobenius_squared - wp.float64(3.0))
        - shear_modulus * log_determinant
        + lame_lambda * wp.float64(0.5) * log_determinant * log_determinant
    )
    return rest_volume * density


@wp.kernel
def point_triangle_distance_kernel(
    points: wp.array(dtype=wp.vec3d),
    triangle_a: wp.array(dtype=wp.vec3d),
    triangle_b: wp.array(dtype=wp.vec3d),
    triangle_c: wp.array(dtype=wp.vec3d),
    distance_squared: wp.array(dtype=wp.float64),
):
    index = wp.tid()
    distance_squared[index] = point_triangle_distance_squared(
        points[index], triangle_a[index], triangle_b[index], triangle_c[index]
    )


@wp.kernel
def edge_edge_distance_kernel(
    edge_a0: wp.array(dtype=wp.vec3d),
    edge_a1: wp.array(dtype=wp.vec3d),
    edge_b0: wp.array(dtype=wp.vec3d),
    edge_b1: wp.array(dtype=wp.vec3d),
    distance_squared: wp.array(dtype=wp.float64),
):
    index = wp.tid()
    distance_squared[index] = edge_edge_distance_squared(
        edge_a0[index], edge_a1[index], edge_b0[index], edge_b1[index]
    )


@wp.kernel
def barrier_kernel(
    distance_squared: wp.array(dtype=wp.float64),
    activation_distance_squared: wp.float64,
    stiffness: wp.float64,
    energy: wp.array(dtype=wp.float64),
    gradient: wp.array(dtype=wp.float64),
    hessian: wp.array(dtype=wp.float64),
):
    index = wp.tid()
    value = distance_squared[index]
    energy[index] = ipc_barrier_energy(value, activation_distance_squared, stiffness)
    gradient[index] = ipc_barrier_gradient(value, activation_distance_squared, stiffness)
    hessian[index] = ipc_barrier_hessian(value, activation_distance_squared, stiffness)


@wp.kernel
def friction_kernel(
    speed: wp.array(dtype=wp.float64),
    smoothing_speed: wp.float64,
    energy: wp.array(dtype=wp.float64),
    gradient: wp.array(dtype=wp.float64),
    hessian: wp.array(dtype=wp.float64),
):
    index = wp.tid()
    value = speed[index]
    energy[index] = smooth_friction_energy(value, smoothing_speed)
    gradient[index] = smooth_friction_gradient(value, smoothing_speed)
    hessian[index] = smooth_friction_hessian(value, smoothing_speed)


@wp.kernel
def point_triangle_ccd_kernel(
    point_start: wp.array(dtype=wp.vec3d),
    point_end: wp.array(dtype=wp.vec3d),
    triangle_a_start: wp.array(dtype=wp.vec3d),
    triangle_a_end: wp.array(dtype=wp.vec3d),
    triangle_b_start: wp.array(dtype=wp.vec3d),
    triangle_b_end: wp.array(dtype=wp.vec3d),
    triangle_c_start: wp.array(dtype=wp.vec3d),
    triangle_c_end: wp.array(dtype=wp.vec3d),
    thickness: wp.float64,
    tolerance: wp.float64,
    fraction: wp.array(dtype=wp.float64),
):
    index = wp.tid()
    fraction[index] = point_triangle_ccd_fraction(
        point_start[index],
        point_end[index],
        triangle_a_start[index],
        triangle_a_end[index],
        triangle_b_start[index],
        triangle_b_end[index],
        triangle_c_start[index],
        triangle_c_end[index],
        thickness,
        tolerance,
    )


@wp.kernel
def edge_edge_ccd_kernel(
    edge_a0_start: wp.array(dtype=wp.vec3d),
    edge_a0_end: wp.array(dtype=wp.vec3d),
    edge_a1_start: wp.array(dtype=wp.vec3d),
    edge_a1_end: wp.array(dtype=wp.vec3d),
    edge_b0_start: wp.array(dtype=wp.vec3d),
    edge_b0_end: wp.array(dtype=wp.vec3d),
    edge_b1_start: wp.array(dtype=wp.vec3d),
    edge_b1_end: wp.array(dtype=wp.vec3d),
    thickness: wp.float64,
    tolerance: wp.float64,
    fraction: wp.array(dtype=wp.float64),
):
    index = wp.tid()
    fraction[index] = edge_edge_ccd_fraction(
        edge_a0_start[index],
        edge_a0_end[index],
        edge_a1_start[index],
        edge_a1_end[index],
        edge_b0_start[index],
        edge_b0_end[index],
        edge_b1_start[index],
        edge_b1_end[index],
        thickness,
        tolerance,
    )


@wp.kernel
def tet_neo_hookean_kernel(
    positions: wp.array2d(dtype=wp.vec3d),
    inverse_rest_matrix: wp.array(dtype=wp.mat33d),
    rest_volume: wp.array(dtype=wp.float64),
    shear_modulus: wp.array(dtype=wp.float64),
    lame_lambda: wp.array(dtype=wp.float64),
    energy: wp.array(dtype=wp.float64),
    gradient: wp.array2d(dtype=wp.vec3d),
    hessian: wp.array3d(dtype=wp.float64),
    valid: wp.array(dtype=wp.int32),
):
    tet = wp.tid()
    x0 = positions[tet, 0]
    x1 = positions[tet, 1]
    x2 = positions[tet, 2]
    x3 = positions[tet, 3]
    inverse_rest = inverse_rest_matrix[tet]
    volume = rest_volume[tet]
    mu = shear_modulus[tet]
    lam = lame_lambda[tet]
    deformation_gradient = _edge_matrix(x1 - x0, x2 - x0, x3 - x0) * inverse_rest
    determinant = wp.determinant(deformation_gradient)

    if determinant <= wp.float64(0.0):
        energy[tet] = wp.float64(_INVALID_ENERGY)
        valid[tet] = 0
        for vertex in range(4):
            gradient[tet, vertex] = wp.vec3d(0.0)
        for row in range(12):
            for column in range(12):
                hessian[tet, row, column] = wp.float64(0.0)
        return

    valid[tet] = 1
    log_determinant = wp.log(determinant)
    inverse_transpose = wp.transpose(wp.inverse(deformation_gradient))
    first_piola = (
        mu * (deformation_gradient - inverse_transpose)
        + lam * log_determinant * inverse_transpose
    )
    gradient_edges = volume * first_piola * wp.transpose(inverse_rest)

    gradient[tet, 1] = wp.vec3d(
        gradient_edges[0, 0], gradient_edges[1, 0], gradient_edges[2, 0]
    )
    gradient[tet, 2] = wp.vec3d(
        gradient_edges[0, 1], gradient_edges[1, 1], gradient_edges[2, 1]
    )
    gradient[tet, 3] = wp.vec3d(
        gradient_edges[0, 2], gradient_edges[1, 2], gradient_edges[2, 2]
    )
    gradient[tet, 0] = -(gradient[tet, 1] + gradient[tet, 2] + gradient[tet, 3])

    energy[tet] = tet_neo_hookean_energy(
        x0, x1, x2, x3, inverse_rest, volume, mu, lam
    )

    inverse_gradient = wp.inverse(deformation_gradient)
    inverse_rest_transpose = wp.transpose(inverse_rest)
    inverse_term_scale = mu - lam * log_determinant
    for column in range(12):
        vertex = column / 3
        component = column - vertex * 3
        differential_edges = wp.mat33d(0.0)
        if vertex == 0:
            differential_edges[component, 0] = wp.float64(-1.0)
            differential_edges[component, 1] = wp.float64(-1.0)
            differential_edges[component, 2] = wp.float64(-1.0)
        else:
            differential_edges[component, vertex - 1] = wp.float64(1.0)

        differential_f = differential_edges * inverse_rest
        differential_log_determinant = wp.trace(inverse_gradient * differential_f)
        differential_piola = (
            mu * differential_f
            + inverse_term_scale
            * inverse_transpose
            * wp.transpose(differential_f)
            * inverse_transpose
            + lam * differential_log_determinant * inverse_transpose
        )
        differential_gradient_edges = volume * differential_piola * inverse_rest_transpose

        for row in range(12):
            row_vertex = row / 3
            row_component = row - row_vertex * 3
            if row_vertex == 0:
                hessian[tet, row, column] = -(
                    differential_gradient_edges[row_component, 0]
                    + differential_gradient_edges[row_component, 1]
                    + differential_gradient_edges[row_component, 2]
                )
            else:
                hessian[tet, row, column] = differential_gradient_edges[
                    row_component, row_vertex - 1
                ]


@wp.kernel
def project_psd_3x3_kernel(
    matrices: wp.array(dtype=wp.mat33d),
    eigenvalue_floor: wp.float64,
    projected: wp.array(dtype=wp.mat33d),
):
    index = wp.tid()
    symmetric = wp.float64(0.5) * (matrices[index] + wp.transpose(matrices[index]))
    eigenvectors = wp.mat33d()
    eigenvalues = wp.vec3d()
    wp.eig3(symmetric, eigenvectors, eigenvalues)
    for component in range(3):
        eigenvalues[component] = wp.max(eigenvalues[component], eigenvalue_floor)
    projected[index] = eigenvectors * wp.diag(eigenvalues) * wp.transpose(eigenvectors)


@wp.kernel
def _dot_vec3_partial_kernel(
    left: wp.array(dtype=wp.vec3d),
    right: wp.array(dtype=wp.vec3d),
    count: wp.int32,
    partial: wp.array(dtype=wp.float64),
):
    block, lane = wp.tid()
    index = block * wp.block_dim() + lane
    value = wp.float64(0.0)
    if index < count:
        value = wp.dot(left[index], right[index])
    total = wp.tile_sum(wp.tile(value))
    if lane == 0:
        partial[block] = total[0]


@wp.kernel
def _dot_scalar_partial_kernel(
    values: wp.array(dtype=wp.float64),
    count: wp.int32,
    partial: wp.array(dtype=wp.float64),
):
    block, lane = wp.tid()
    index = block * wp.block_dim() + lane
    value = wp.float64(0.0)
    if index < count:
        value = values[index]
    total = wp.tile_sum(wp.tile(value))
    if lane == 0:
        partial[block] = total[0]


@wp.kernel
def _invert_spd_blocks_kernel(
    diagonal: wp.array(dtype=wp.mat33d),
    inverse_diagonal: wp.array(dtype=wp.mat33d),
    invalid: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    symmetric = wp.float64(0.5) * (diagonal[index] + wp.transpose(diagonal[index]))
    leading_minor_1 = symmetric[0, 0]
    leading_minor_2 = (
        symmetric[0, 0] * symmetric[1, 1] - symmetric[0, 1] * symmetric[1, 0]
    )
    determinant = wp.determinant(symmetric)
    if (
        not wp.isfinite(symmetric)
        or leading_minor_1 <= wp.float64(_DISTANCE_EPSILON)
        or leading_minor_2 <= wp.float64(_DISTANCE_EPSILON)
        or determinant <= wp.float64(_DISTANCE_EPSILON)
    ):
        inverse_diagonal[index] = wp.mat33d(0.0)
        wp.atomic_max(invalid, 0, 1)
    else:
        inverse_diagonal[index] = wp.inverse(symmetric)


@wp.kernel
def _apply_block_preconditioner_kernel(
    inverse_diagonal: wp.array(dtype=wp.mat33d),
    residual: wp.array(dtype=wp.vec3d),
    preconditioned: wp.array(dtype=wp.vec3d),
):
    index = wp.tid()
    preconditioned[index] = inverse_diagonal[index] * residual[index]


@wp.kernel
def _initialize_cg_control_kernel(
    residual_squared: wp.array(dtype=wp.float64),
    rhs_squared: wp.array(dtype=wp.float64),
    relative_tolerance_squared: wp.float64,
    preconditioner_invalid: wp.array(dtype=wp.int32),
    tolerance_squared: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
    iteration_count: wp.array(dtype=wp.int32),
):
    tolerance = wp.max(
        relative_tolerance_squared * rhs_squared[0], wp.float64(_DISTANCE_EPSILON)
    )
    tolerance_squared[0] = tolerance
    invalid[0] = preconditioner_invalid[0]
    iteration_count[0] = 0
    if (
        preconditioner_invalid[0] != 0
        or not wp.isfinite(residual_squared[0])
        or residual_squared[0] <= tolerance
    ):
        active[0] = 0
    else:
        active[0] = 1


@wp.kernel
def _compute_cg_alpha_kernel(
    rho: wp.array(dtype=wp.float64),
    direction_dot_product: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
    alpha: wp.array(dtype=wp.float64),
):
    alpha[0] = wp.float64(0.0)
    denominator = direction_dot_product[0]
    numerator = rho[0]
    if active[0] != 0:
        if (
            not wp.isfinite(denominator)
            or not wp.isfinite(numerator)
            or denominator <= wp.float64(_DISTANCE_EPSILON)
            or numerator <= wp.float64(0.0)
        ):
            active[0] = 0
            invalid[0] = 1
        else:
            alpha[0] = numerator / denominator


@wp.kernel
def _update_cg_solution_residual_kernel(
    solution: wp.array(dtype=wp.vec3d),
    residual: wp.array(dtype=wp.vec3d),
    direction: wp.array(dtype=wp.vec3d),
    matrix_direction: wp.array(dtype=wp.vec3d),
    alpha: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    if active[0] != 0:
        step = alpha[0]
        solution[index] += step * direction[index]
        residual[index] -= step * matrix_direction[index]


@wp.kernel
def _finalize_cg_iteration_kernel(
    rho: wp.array(dtype=wp.float64),
    next_rho: wp.array(dtype=wp.float64),
    residual_squared: wp.array(dtype=wp.float64),
    tolerance_squared: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
    beta: wp.array(dtype=wp.float64),
    iteration_count: wp.array(dtype=wp.int32),
):
    beta[0] = wp.float64(0.0)
    if active[0] != 0:
        iteration_count[0] += 1
        if not wp.isfinite(residual_squared[0]) or not wp.isfinite(next_rho[0]):
            active[0] = 0
            invalid[0] = 1
        elif residual_squared[0] <= tolerance_squared[0]:
            active[0] = 0
            rho[0] = next_rho[0]
        elif rho[0] <= wp.float64(0.0) or next_rho[0] <= wp.float64(0.0):
            active[0] = 0
            invalid[0] = 1
        else:
            beta[0] = next_rho[0] / rho[0]
            rho[0] = next_rho[0]


@wp.kernel
def _update_cg_direction_kernel(
    preconditioned: wp.array(dtype=wp.vec3d),
    direction: wp.array(dtype=wp.vec3d),
    beta: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    if active[0] != 0:
        direction[index] = preconditioned[index] + beta[0] * direction[index]
    else:
        direction[index] = wp.vec3d(0.0)


class _FixedDotWorkspace:
    _BLOCK_SIZE = 256

    def __init__(self, length: int, device: Any) -> None:
        if length <= 0:
            raise ValueError("dot-product workspace length must be positive")
        self.length = length
        self.device = device
        self.partial: list[Any] = []
        partial_count = (length + self._BLOCK_SIZE - 1) // self._BLOCK_SIZE
        while True:
            self.partial.append(
                wp.empty(partial_count, dtype=wp.float64, device=device)
            )
            if partial_count == 1:
                break
            partial_count = (partial_count + self._BLOCK_SIZE - 1) // self._BLOCK_SIZE

    @property
    def result(self) -> Any:
        return self.partial[-1]

    def compute(self, left: Any, right: Any) -> Any:
        first_count = self.partial[0].shape[0]
        wp.launch(
            _dot_vec3_partial_kernel,
            dim=(first_count, self._BLOCK_SIZE),
            block_dim=self._BLOCK_SIZE,
            inputs=[left, right, self.length],
            outputs=[self.partial[0]],
            device=self.device,
        )
        for level in range(1, len(self.partial)):
            source = self.partial[level - 1]
            target = self.partial[level]
            wp.launch(
                _dot_scalar_partial_kernel,
                dim=(target.shape[0], self._BLOCK_SIZE),
                block_dim=self._BLOCK_SIZE,
                inputs=[source, source.shape[0]],
                outputs=[target],
                device=self.device,
            )
        return self.result


def assemble_bsr33_from_triplets(
    block_count: int,
    rows: Any,
    columns: Any,
    values: Any,
) -> Any:
    """Assemble a square 3x3 BSR matrix using Warp's public sparse API."""

    try:
        resolved_block_count = operator.index(block_count)
    except TypeError as error:
        raise TypeError("block_count must be an integer") from error
    if isinstance(block_count, bool) or resolved_block_count <= 0:
        raise ValueError("block_count must be a positive integer")
    if rows.shape != columns.shape or rows.shape[0] != values.shape[0]:
        raise ValueError("BSR triplet rows, columns, and values must have matching lengths")
    if rows.dtype != wp.int32 or columns.dtype != wp.int32:
        raise TypeError("BSR row and column arrays must use Warp int32")
    if values.dtype != wp.mat33d:
        raise TypeError("BSR values must use Warp mat33d")
    if rows.device != columns.device or rows.device != values.device:
        raise ValueError("BSR triplet arrays must share one device")

    from warp.sparse import bsr_from_triplets

    return bsr_from_triplets(
        rows_of_blocks=resolved_block_count,
        cols_of_blocks=resolved_block_count,
        rows=rows,
        columns=columns,
        values=values,
        prune_numerical_zeros=False,
    )


class FixedCgSolver:
    """Fixed-buffer block-Jacobi CG with device-side convergence state."""

    def __init__(
        self,
        matrix: Any,
        rhs: Any,
        solution: Any,
        *,
        max_iterations: int,
        relative_tolerance: float,
    ) -> None:
        try:
            resolved_iterations = operator.index(max_iterations)
        except TypeError as error:
            raise TypeError("max_iterations must be an integer") from error
        if isinstance(max_iterations, bool) or resolved_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        tolerance = float(relative_tolerance)
        if not 0.0 < tolerance < 1.0:
            raise ValueError("relative_tolerance must be between zero and one")
        if rhs.device != solution.device or matrix.device != rhs.device:
            raise ValueError("CG matrix, right-hand side, and solution must share one device")
        if rhs.shape != solution.shape:
            raise ValueError("CG right-hand side and solution must have the same shape")
        if len(rhs.shape) != 1 or rhs.dtype != wp.vec3d or solution.dtype != wp.vec3d:
            raise TypeError("fixed CG vectors must be one-dimensional Warp vec3d arrays")
        if matrix.nrow != matrix.ncol or matrix.nrow != rhs.shape[0]:
            raise ValueError("fixed CG requires a square BSR matrix matching the vector length")
        if matrix.block_shape != (3, 3) or matrix.scalar_type != wp.float64:
            raise TypeError("fixed CG requires a float64 3x3 BSR matrix")

        self.matrix = matrix
        self.rhs = rhs
        self.solution = solution
        self.max_iterations = resolved_iterations
        self.relative_tolerance = tolerance
        self.diagonal = wp.empty(matrix.nrow, dtype=wp.mat33d, device=matrix.device)
        self.inverse_diagonal = wp.empty_like(self.diagonal)
        self.preconditioner_invalid = wp.zeros(1, dtype=wp.int32, device=matrix.device)
        self.residual = wp.empty_like(rhs)
        self.preconditioned = wp.empty_like(rhs)
        self.direction = wp.empty_like(rhs)
        self.matrix_direction = wp.empty_like(rhs)
        self.rho = wp.empty(1, dtype=wp.float64, device=matrix.device)
        self.next_rho = wp.empty_like(self.rho)
        self.direction_dot_product = wp.empty_like(self.rho)
        self.residual_squared = wp.empty_like(self.rho)
        self.rhs_squared = wp.empty_like(self.rho)
        self.tolerance_squared = wp.empty_like(self.rho)
        self.alpha = wp.empty_like(self.rho)
        self.beta = wp.empty_like(self.rho)
        self.active = wp.empty(1, dtype=wp.int32, device=matrix.device)
        self.invalid = wp.empty_like(self.active)
        self.iteration_count = wp.empty_like(self.active)
        self._dot_workspace = _FixedDotWorkspace(rhs.shape[0], matrix.device)

    def _dot_into(self, left: Any, right: Any, target: Any) -> None:
        result = self._dot_workspace.compute(left, right)
        wp.copy(target, result, count=1)

    def refresh_preconditioner(self) -> None:
        from warp.sparse import bsr_get_diag

        self.preconditioner_invalid.zero_()
        bsr_get_diag(self.matrix, out=self.diagonal)
        wp.launch(
            _invert_spd_blocks_kernel,
            dim=self.matrix.nrow,
            inputs=[self.diagonal],
            outputs=[self.inverse_diagonal, self.preconditioner_invalid],
            device=self.matrix.device,
        )

    def solve(self) -> tuple[Any, Any, Any, Any]:
        from warp.sparse import bsr_mv

        self.refresh_preconditioner()
        self.residual.assign(self.rhs)
        bsr_mv(self.matrix, self.solution, self.residual, alpha=-1.0, beta=1.0)
        wp.launch(
            _apply_block_preconditioner_kernel,
            dim=self.matrix.nrow,
            inputs=[self.inverse_diagonal, self.residual],
            outputs=[self.preconditioned],
            device=self.matrix.device,
        )
        self.direction.assign(self.preconditioned)
        self._dot_into(self.residual, self.preconditioned, self.rho)
        self._dot_into(self.residual, self.residual, self.residual_squared)
        self._dot_into(self.rhs, self.rhs, self.rhs_squared)
        wp.launch(
            _initialize_cg_control_kernel,
            dim=1,
            inputs=[
                self.residual_squared,
                self.rhs_squared,
                self.relative_tolerance * self.relative_tolerance,
                self.preconditioner_invalid,
            ],
            outputs=[
                self.tolerance_squared,
                self.active,
                self.invalid,
                self.iteration_count,
            ],
            device=self.matrix.device,
        )

        for _iteration in range(self.max_iterations):
            bsr_mv(
                self.matrix,
                self.direction,
                self.matrix_direction,
                alpha=1.0,
                beta=0.0,
            )
            self._dot_into(
                self.direction, self.matrix_direction, self.direction_dot_product
            )
            wp.launch(
                _compute_cg_alpha_kernel,
                dim=1,
                inputs=[
                    self.rho,
                    self.direction_dot_product,
                    self.active,
                    self.invalid,
                ],
                outputs=[self.alpha],
                device=self.matrix.device,
            )
            wp.launch(
                _update_cg_solution_residual_kernel,
                dim=self.matrix.nrow,
                inputs=[
                    self.solution,
                    self.residual,
                    self.direction,
                    self.matrix_direction,
                    self.alpha,
                    self.active,
                ],
                device=self.matrix.device,
            )
            self._dot_into(self.residual, self.residual, self.residual_squared)
            wp.launch(
                _apply_block_preconditioner_kernel,
                dim=self.matrix.nrow,
                inputs=[self.inverse_diagonal, self.residual],
                outputs=[self.preconditioned],
                device=self.matrix.device,
            )
            self._dot_into(self.residual, self.preconditioned, self.next_rho)
            wp.launch(
                _finalize_cg_iteration_kernel,
                dim=1,
                inputs=[
                    self.rho,
                    self.next_rho,
                    self.residual_squared,
                    self.tolerance_squared,
                    self.active,
                    self.invalid,
                ],
                outputs=[self.beta, self.iteration_count],
                device=self.matrix.device,
            )
            wp.launch(
                _update_cg_direction_kernel,
                dim=self.matrix.nrow,
                inputs=[self.preconditioned, self.direction, self.beta, self.active],
                device=self.matrix.device,
            )

        return (
            self.iteration_count,
            self.residual_squared,
            self.tolerance_squared,
            self.invalid,
        )


__all__ = [
    "FixedCgSolver",
    "assemble_bsr33_from_triplets",
    "barrier_kernel",
    "edge_edge_ccd_kernel",
    "edge_edge_distance_kernel",
    "friction_kernel",
    "point_triangle_ccd_kernel",
    "point_triangle_distance_kernel",
    "project_psd_3x3_kernel",
    "tet_neo_hookean_kernel",
]
