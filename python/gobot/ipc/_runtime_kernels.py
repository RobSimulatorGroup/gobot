# SPDX-FileCopyrightText: Copyright (c) 2026 Gobot contributors
# SPDX-License-Identifier: Apache-2.0
"""Warp kernels used only by the private IPC correctness runtime."""

from __future__ import annotations

import warp as wp

from ._warp_kernels import (
    ipc_barrier_gradient,
    ipc_barrier_hessian,
    point_triangle_ccd_fraction,
    point_triangle_distance_squared,
)


_EPSILON = 1.0e-30


@wp.func
def _closest_point_barycentric(
    point: wp.vec3d,
    vertex_a: wp.vec3d,
    vertex_b: wp.vec3d,
    vertex_c: wp.vec3d,
) -> wp.vec3d:
    edge_ab = vertex_b - vertex_a
    edge_ac = vertex_c - vertex_a
    offset_a = point - vertex_a
    d1 = wp.dot(edge_ab, offset_a)
    d2 = wp.dot(edge_ac, offset_a)
    if d1 <= wp.float64(0.0) and d2 <= wp.float64(0.0):
        return wp.vec3d(1.0, 0.0, 0.0)

    offset_b = point - vertex_b
    d3 = wp.dot(edge_ab, offset_b)
    d4 = wp.dot(edge_ac, offset_b)
    if d3 >= wp.float64(0.0) and d4 <= d3:
        return wp.vec3d(0.0, 1.0, 0.0)

    vc = d1 * d4 - d3 * d2
    if vc <= wp.float64(0.0) and d1 >= wp.float64(0.0) and d3 <= wp.float64(0.0):
        coordinate = d1 / (d1 - d3)
        return wp.vec3d(wp.float64(1.0) - coordinate, coordinate, 0.0)

    offset_c = point - vertex_c
    d5 = wp.dot(edge_ab, offset_c)
    d6 = wp.dot(edge_ac, offset_c)
    if d6 >= wp.float64(0.0) and d5 <= d6:
        return wp.vec3d(0.0, 0.0, 1.0)

    vb = d5 * d2 - d1 * d6
    if vb <= wp.float64(0.0) and d2 >= wp.float64(0.0) and d6 <= wp.float64(0.0):
        coordinate = d2 / (d2 - d6)
        return wp.vec3d(wp.float64(1.0) - coordinate, 0.0, coordinate)

    va = d3 * d6 - d5 * d4
    d43 = d4 - d3
    d56 = d5 - d6
    if va <= wp.float64(0.0) and d43 >= wp.float64(0.0) and d56 >= wp.float64(0.0):
        coordinate = d43 / (d43 + d56)
        return wp.vec3d(0.0, wp.float64(1.0) - coordinate, coordinate)

    denominator = wp.float64(1.0) / (va + vb + vc)
    coordinate_b = vb * denominator
    coordinate_c = vc * denominator
    return wp.vec3d(
        wp.float64(1.0) - coordinate_b - coordinate_c,
        coordinate_b,
        coordinate_c,
    )


@wp.kernel
def update_attached_targets_kernel(
    body_pose: wp.array(dtype=wp.transform),
    particle_index: wp.array(dtype=wp.int32),
    body_index: wp.array(dtype=wp.int32),
    link_local_position: wp.array(dtype=wp.vec3d),
    target: wp.array(dtype=wp.vec3d),
):
    index = wp.tid()
    pose = body_pose[body_index[index]]
    local = link_local_position[index]
    local_float = wp.vec3(wp.float32(local[0]), wp.float32(local[1]), wp.float32(local[2]))
    world = wp.transform_point(pose, local_float)
    target[particle_index[index]] = wp.vec3d(
        wp.float64(world[0]), wp.float64(world[1]), wp.float64(world[2])
    )


@wp.kernel
def predict_kernel(
    position: wp.array(dtype=wp.vec3d),
    velocity: wp.array(dtype=wp.vec3d),
    mass: wp.array(dtype=wp.float64),
    kinematic: wp.array(dtype=wp.int32),
    kinematic_target: wp.array(dtype=wp.vec3d),
    gravity: wp.vec3d,
    damping: wp.array(dtype=wp.float64),
    dt: wp.float64,
    predicted: wp.array(dtype=wp.vec3d),
    iterate: wp.array(dtype=wp.vec3d),
):
    index = wp.tid()
    if kinematic[index] != 0:
        # Soft kinematic vertices have no inertia or body-force term. Start the
        # nonlinear solve from the current feasible mesh and let the quadratic
        # target energy move them without teleporting the surrounding tets.
        predicted[index] = position[index]
        iterate[index] = position[index]
        return
    decay = wp.exp(-damping[index] * dt)
    next_velocity = decay * velocity[index] + dt * gravity
    value = position[index] + dt * next_velocity
    predicted[index] = value
    iterate[index] = value


@wp.kernel
def initialize_newton_kernel(
    iterate: wp.array(dtype=wp.vec3d),
    predicted: wp.array(dtype=wp.vec3d),
    mass: wp.array(dtype=wp.float64),
    kinematic: wp.array(dtype=wp.int32),
    kinematic_target: wp.array(dtype=wp.vec3d),
    kinematic_stiffness: wp.float64,
    dt: wp.float64,
    gradient: wp.array(dtype=wp.vec3d),
    hessian: wp.array3d(dtype=wp.float64),
    contact_force: wp.array(dtype=wp.vec3d),
):
    index = wp.tid()
    contact_force[index] = wp.vec3d(0.0)
    for row in range(3):
        for column in range(3):
            hessian[index, row, column] = wp.float64(0.0)
    if kinematic[index] != 0:
        # The runtime divides the incremental potential by dt^2 so inertia can
        # be assembled as M / dt^2 while elastic and contact terms keep their
        # physical coefficients. Apply the same normalization to Taccel's
        # mass-weighted soft kinematic penalty.
        weighted_mass = mass[index] * kinematic_stiffness / (dt * dt)
        gradient[index] = weighted_mass * (
            iterate[index] - kinematic_target[index]
        )
        hessian[index, 0, 0] = weighted_mass
        hessian[index, 1, 1] = weighted_mass
        hessian[index, 2, 2] = weighted_mass
        return
    inverse_dt_squared_mass = mass[index] / (dt * dt)
    gradient[index] = inverse_dt_squared_mass * (iterate[index] - predicted[index])
    hessian[index, 0, 0] = inverse_dt_squared_mass
    hessian[index, 1, 1] = inverse_dt_squared_mass
    hessian[index, 2, 2] = inverse_dt_squared_mass


@wp.kernel
def gather_tetrahedra_kernel(
    position: wp.array(dtype=wp.vec3d),
    indices: wp.array2d(dtype=wp.int32),
    gathered: wp.array2d(dtype=wp.vec3d),
):
    tetrahedron, local_vertex = wp.tid()
    gathered[tetrahedron, local_vertex] = position[indices[tetrahedron, local_vertex]]


@wp.kernel
def tetrahedron_validity_kernel(
    position: wp.array(dtype=wp.vec3d),
    indices: wp.array2d(dtype=wp.int32),
    rest_volume: wp.array(dtype=wp.float64),
    valid: wp.array(dtype=wp.int32),
):
    tetrahedron = wp.tid()
    x0 = position[indices[tetrahedron, 0]]
    x1 = position[indices[tetrahedron, 1]]
    x2 = position[indices[tetrahedron, 2]]
    x3 = position[indices[tetrahedron, 3]]
    determinant = wp.dot(x1 - x0, wp.cross(x2 - x0, x3 - x0))
    minimum = wp.float64(6.0e-8) * rest_volume[tetrahedron]
    if wp.isfinite(determinant) and determinant > minimum:
        valid[tetrahedron] = 1
    else:
        valid[tetrahedron] = 0


@wp.kernel
def blend_line_search_kernel(
    origin: wp.array(dtype=wp.vec3d),
    candidate: wp.array(dtype=wp.vec3d),
    fraction: wp.float64,
    result: wp.array(dtype=wp.vec3d),
):
    index = wp.tid()
    result[index] = origin[index] + fraction * (candidate[index] - origin[index])


@wp.kernel
def initialize_ccd_fraction_kernel(
    fraction: wp.array(dtype=wp.float64),
):
    environment = wp.tid()
    fraction[environment] = wp.float64(1.0)


@wp.kernel
def point_triangle_line_search_kernel(
    origin: wp.array(dtype=wp.vec3d),
    candidate: wp.array(dtype=wp.vec3d),
    point_index: wp.array(dtype=wp.int32),
    triangle_a_index: wp.array(dtype=wp.int32),
    triangle_b_index: wp.array(dtype=wp.int32),
    triangle_c_index: wp.array(dtype=wp.int32),
    contact_environment: wp.array(dtype=wp.int32),
    thickness: wp.float64,
    tolerance: wp.float64,
    environment_fraction: wp.array(dtype=wp.float64),
):
    contact = wp.tid()
    point = point_index[contact]
    a = triangle_a_index[contact]
    b = triangle_b_index[contact]
    c = triangle_c_index[contact]
    start_distance_squared = point_triangle_distance_squared(
        origin[point], origin[a], origin[b], origin[c]
    )
    end_distance_squared = point_triangle_distance_squared(
        candidate[point], candidate[a], candidate[b], candidate[c]
    )
    start_distance = wp.sqrt(
        wp.max(start_distance_squared, wp.float64(0.0))
    )
    safe_thickness = wp.min(
        thickness, wp.float64(0.5) * start_distance
    )
    safe_tolerance = wp.min(
        tolerance, wp.float64(0.1) * start_distance
    )
    fraction = point_triangle_ccd_fraction(
        origin[point],
        candidate[point],
        origin[a],
        candidate[a],
        origin[b],
        candidate[b],
        origin[c],
        candidate[c],
        safe_thickness,
        safe_tolerance,
    )
    clearance = thickness + tolerance
    if start_distance_squared <= clearance * clearance:
        start_normal = wp.cross(origin[b] - origin[a], origin[c] - origin[a])
        end_normal = wp.cross(candidate[b] - candidate[a], candidate[c] - candidate[a])
        start_side = wp.dot(origin[point] - origin[a], start_normal)
        end_side = wp.dot(candidate[point] - candidate[a], end_normal)
        crossed_plane = (
            start_side < wp.float64(0.0) and end_side > wp.float64(0.0)
        ) or (
            start_side > wp.float64(0.0) and end_side < wp.float64(0.0)
        )
        # The micron-scale thickness is a numerical guard, not a physical
        # shell. Once a contact is inside that guard, allow same-side motion
        # while retaining CCD for an actual triangle-plane crossing.
        if (
            not crossed_plane
            and end_distance_squared > tolerance * tolerance
        ):
            fraction = wp.float64(1.0)
    wp.atomic_min(
        environment_fraction, contact_environment[contact], fraction
    )


@wp.kernel
def apply_ccd_fraction_kernel(
    origin: wp.array(dtype=wp.vec3d),
    candidate: wp.array(dtype=wp.vec3d),
    environment_fraction: wp.array(dtype=wp.float64),
    particles_per_environment: wp.int32,
    result: wp.array(dtype=wp.vec3d),
):
    index = wp.tid()
    environment = index / particles_per_environment
    fraction = environment_fraction[environment]
    if fraction < wp.float64(1.0):
        fraction *= wp.float64(0.9)
    result[index] = origin[index] + fraction * (candidate[index] - origin[index])


@wp.kernel
def accumulate_tetrahedra_kernel(
    indices: wp.array2d(dtype=wp.int32),
    tetrahedron_gradient: wp.array2d(dtype=wp.vec3d),
    tetrahedron_hessian: wp.array3d(dtype=wp.float64),
    gradient: wp.array(dtype=wp.vec3d),
    hessian: wp.array3d(dtype=wp.float64),
):
    tetrahedron, local_vertex = wp.tid()
    particle = indices[tetrahedron, local_vertex]
    value = tetrahedron_gradient[tetrahedron, local_vertex]
    wp.atomic_add(gradient, particle, value)


@wp.kernel
def point_triangle_contact_kernel(
    position: wp.array(dtype=wp.vec3d),
    velocity: wp.array(dtype=wp.vec3d),
    point_index: wp.array(dtype=wp.int32),
    triangle_a_index: wp.array(dtype=wp.int32),
    triangle_b_index: wp.array(dtype=wp.int32),
    triangle_c_index: wp.array(dtype=wp.int32),
    activation_distance_squared: wp.float64,
    stiffness: wp.float64,
    friction_coefficient: wp.float64,
    smoothing_speed: wp.float64,
    gradient: wp.array(dtype=wp.vec3d),
    contact_force: wp.array(dtype=wp.vec3d),
    active_count: wp.array(dtype=wp.int32),
    contact_barycentric: wp.array(dtype=wp.vec3d),
    contact_hessian: wp.array(dtype=wp.mat33d),
):
    contact = wp.tid()
    contact_barycentric[contact] = wp.vec3d(0.0)
    contact_hessian[contact] = wp.mat33d(0.0)
    point_id = point_index[contact]
    a_id = triangle_a_index[contact]
    b_id = triangle_b_index[contact]
    c_id = triangle_c_index[contact]
    point = position[point_id]
    a = position[a_id]
    b = position[b_id]
    c = position[c_id]
    distance_squared = point_triangle_distance_squared(point, a, b, c)
    if distance_squared >= activation_distance_squared:
        return

    barycentric = _closest_point_barycentric(point, a, b, c)
    closest = barycentric[0] * a + barycentric[1] * b + barycentric[2] * c
    offset = point - closest
    distance_squared = wp.max(wp.dot(offset, offset), wp.float64(_EPSILON))
    distance = wp.sqrt(distance_squared)
    normal = offset / distance
    barrier_first = ipc_barrier_gradient(
        distance_squared, activation_distance_squared, stiffness
    )
    barrier_second = ipc_barrier_hessian(
        distance_squared, activation_distance_squared, stiffness
    )
    point_gradient = wp.float64(2.0) * barrier_first * offset
    normal_force = -point_gradient

    triangle_velocity = (
        barycentric[0] * velocity[a_id]
        + barycentric[1] * velocity[b_id]
        + barycentric[2] * velocity[c_id]
    )
    relative_velocity = velocity[point_id] - triangle_velocity
    tangential_velocity = relative_velocity - wp.dot(relative_velocity, normal) * normal
    tangential_speed = wp.length(tangential_velocity)
    friction_force = wp.vec3d(0.0)
    if friction_coefficient > wp.float64(0.0) and tangential_speed > wp.float64(_EPSILON):
        smooth_scale = wp.min(
            wp.float64(1.0), tangential_speed / wp.max(smoothing_speed, wp.float64(_EPSILON))
        )
        friction_force = (
            -friction_coefficient
            * wp.length(normal_force)
            * smooth_scale
            * tangential_velocity
            / tangential_speed
        )
        point_gradient -= friction_force

    force = normal_force + friction_force
    wp.atomic_add(gradient, point_id, point_gradient)
    wp.atomic_add(contact_force, point_id, force)
    triangle_ids = wp.vec3i(a_id, b_id, c_id)
    for vertex in range(3):
        weight = barycentric[vertex]
        wp.atomic_add(gradient, triangle_ids[vertex], -weight * point_gradient)
        wp.atomic_add(contact_force, triangle_ids[vertex], -weight * force)

    normal_curvature = wp.max(
        wp.float64(2.0) * barrier_first
        + wp.float64(4.0) * distance_squared * barrier_second,
        wp.float64(0.0),
    )
    friction_curvature = (
        friction_coefficient
        * wp.length(normal_force)
        / wp.max(tangential_speed, smoothing_speed)
    )
    contact_barycentric[contact] = barycentric
    contact_hessian[contact] = wp.mat33d(
        normal_curvature * normal[0] * normal[0] + friction_curvature,
        normal_curvature * normal[0] * normal[1],
        normal_curvature * normal[0] * normal[2],
        normal_curvature * normal[1] * normal[0],
        normal_curvature * normal[1] * normal[1] + friction_curvature,
        normal_curvature * normal[1] * normal[2],
        normal_curvature * normal[2] * normal[0],
        normal_curvature * normal[2] * normal[1],
        normal_curvature * normal[2] * normal[2] + friction_curvature,
    )
    wp.atomic_add(active_count, 0, 1)


@wp.kernel
def write_diagonal_system_kernel(
    hessian: wp.array3d(dtype=wp.float64),
    gradient: wp.array(dtype=wp.vec3d),
    regularization: wp.float64,
    values: wp.array(dtype=wp.mat33d),
    rhs: wp.array(dtype=wp.vec3d),
    solution: wp.array(dtype=wp.vec3d),
):
    index = wp.tid()
    values[index] = wp.mat33d(
        hessian[index, 0, 0] + regularization,
        hessian[index, 0, 1],
        hessian[index, 0, 2],
        hessian[index, 1, 0],
        hessian[index, 1, 1] + regularization,
        hessian[index, 1, 2],
        hessian[index, 2, 0],
        hessian[index, 2, 1],
        hessian[index, 2, 2] + regularization,
    )
    rhs[index] = -gradient[index]
    solution[index] = wp.vec3d(0.0)


@wp.kernel
def write_tetrahedron_system_kernel(
    indices: wp.array2d(dtype=wp.int32),
    tetrahedron_hessian: wp.array3d(dtype=wp.float64),
    value_offset: wp.int32,
    values: wp.array(dtype=wp.mat33d),
):
    tetrahedron, row_vertex, column_vertex = wp.tid()
    destination = value_offset + tetrahedron * 16 + row_vertex * 4 + column_vertex
    stiffness = wp.float64(0.0)
    for component in range(12):
        stiffness = wp.max(
            stiffness,
            wp.abs(tetrahedron_hessian[tetrahedron, component, component]),
        )
    stiffness = wp.max(
        wp.float64(0.05) * stiffness / wp.float64(3.0),
        wp.float64(1.0e-12),
    )
    if row_vertex == column_vertex:
        stiffness *= wp.float64(3.0)
    else:
        stiffness = -stiffness
    values[destination] = wp.mat33d(
        stiffness, wp.float64(0.0), wp.float64(0.0),
        wp.float64(0.0), stiffness, wp.float64(0.0),
        wp.float64(0.0), wp.float64(0.0), stiffness,
    )


@wp.kernel
def write_point_triangle_system_kernel(
    barycentric: wp.array(dtype=wp.vec3d),
    contact_hessian: wp.array(dtype=wp.mat33d),
    value_offset: wp.int32,
    values: wp.array(dtype=wp.mat33d),
):
    contact, row_vertex, column_vertex = wp.tid()
    row_weight = wp.float64(1.0)
    column_weight = wp.float64(1.0)
    if row_vertex > 0:
        row_weight = -barycentric[contact][row_vertex - 1]
    if column_vertex > 0:
        column_weight = -barycentric[contact][column_vertex - 1]
    scale = row_weight * column_weight
    block = contact_hessian[contact]
    destination = value_offset + contact * 16 + row_vertex * 4 + column_vertex
    values[destination] = wp.mat33d(
        scale * block[0, 0], scale * block[0, 1], scale * block[0, 2],
        scale * block[1, 0], scale * block[1, 1], scale * block[1, 2],
        scale * block[2, 0], scale * block[2, 1], scale * block[2, 2],
    )


@wp.kernel
def apply_newton_step_kernel(
    iterate: wp.array(dtype=wp.vec3d),
    solution: wp.array(dtype=wp.vec3d),
    maximum_step: wp.float64,
):
    index = wp.tid()
    delta = solution[index]
    length = wp.length(delta)
    if length > maximum_step:
        delta *= maximum_step / length
    iterate[index] += delta


@wp.kernel
def finalize_particles_kernel(
    previous_position: wp.array(dtype=wp.vec3d),
    iterate: wp.array(dtype=wp.vec3d),
    dt: wp.float64,
    position: wp.array(dtype=wp.vec3d),
    velocity: wp.array(dtype=wp.vec3d),
    public_position: wp.array(dtype=wp.vec3),
    public_velocity: wp.array(dtype=wp.vec3),
    public_contact_force: wp.array(dtype=wp.vec3),
    contact_force: wp.array(dtype=wp.vec3d),
    invalid: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    value = iterate[index]
    next_velocity = (value - previous_position[index]) / dt
    if not wp.isfinite(value) or not wp.isfinite(next_velocity):
        wp.atomic_max(invalid, 0, 1)
        return
    position[index] = value
    velocity[index] = next_velocity
    public_position[index] = wp.vec3(
        wp.float32(value[0]), wp.float32(value[1]), wp.float32(value[2])
    )
    public_velocity[index] = wp.vec3(
        wp.float32(next_velocity[0]),
        wp.float32(next_velocity[1]),
        wp.float32(next_velocity[2]),
    )
    force = contact_force[index]
    public_contact_force[index] = wp.vec3(
        wp.float32(force[0]), wp.float32(force[1]), wp.float32(force[2])
    )


@wp.kernel
def copy_particles_to_newton_kernel(
    position: wp.array(dtype=wp.vec3d),
    velocity: wp.array(dtype=wp.vec3d),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    value = position[index]
    speed = velocity[index]
    particle_q[index] = wp.vec3(
        wp.float32(value[0]), wp.float32(value[1]), wp.float32(value[2])
    )
    particle_qd[index] = wp.vec3(
        wp.float32(speed[0]), wp.float32(speed[1]), wp.float32(speed[2])
    )


@wp.kernel
def reset_masked_particles_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    particles_per_environment: wp.int32,
    initial_position: wp.array(dtype=wp.vec3d),
    position: wp.array(dtype=wp.vec3d),
    velocity: wp.array(dtype=wp.vec3d),
    predicted: wp.array(dtype=wp.vec3d),
    iterate: wp.array(dtype=wp.vec3d),
    kinematic_target: wp.array(dtype=wp.vec3d),
    public_position: wp.array(dtype=wp.vec3),
    public_velocity: wp.array(dtype=wp.vec3),
    public_contact_force: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    environment = index / particles_per_environment
    if not reset_mask[environment]:
        return
    value = initial_position[index]
    position[index] = value
    velocity[index] = wp.vec3d(0.0)
    predicted[index] = value
    iterate[index] = value
    kinematic_target[index] = value
    public_position[index] = wp.vec3(
        wp.float32(value[0]), wp.float32(value[1]), wp.float32(value[2])
    )
    public_velocity[index] = wp.vec3(0.0)
    public_contact_force[index] = wp.vec3(0.0)


__all__: list[str] = []
