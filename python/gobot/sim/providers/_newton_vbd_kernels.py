"""Newton 1.6 VBD contact adapter. SDK implementation details stay here."""
import warp as wp
from newton._src.solvers.vbd.rigid_vbd_kernels import _eval_soft_ef_contact

_HAS_LOG_BARRIER = "use_log_barrier" in _eval_soft_ef_contact.input_types


@wp.kernel
def set_rigid_soft_damping(count: wp.array[int], damping: wp.array[float], value: float):
    i = wp.tid()
    if i < count[0]:
        damping[i] = value


@wp.kernel
def set_soft_surface_velocity(count: wp.array[int], shapes: wp.array[int], active: wp.array[int],
                              velocity: wp.array[wp.vec3], shape_body: wp.array[int],
                              body_q: wp.array[wp.transform], normals: wp.array[wp.vec3],
                              contact_velocity: wp.array[wp.vec3], friction: wp.array[float], mu: float):
    i = wp.tid()
    if i >= count[0]:
        return
    shape = shapes[i]
    if active[shape] != 0:
        # Newton subtracts this body-local surface velocity when evaluating
        # friction in the implicit contact solve. A belt is tangential motion,
        # not an extra normal velocity or a one-step force/velocity servo.
        n = normals[i]
        v = velocity[shape] - wp.dot(velocity[shape], n) * n
        body = shape_body[shape]
        if body >= 0:
            v = wp.transform_vector(wp.transform_inverse(body_q[body]), v)
        contact_velocity[i] = v
        friction[i] = mu


@wp.kernel
def track_tet_volume(q: wp.array[wp.vec3], indices: wp.array2d[int],
                     rest_inverse: wp.array[wp.mat33], minimum: wp.array[float]):
    i = wp.tid()
    origin = q[indices[i, 0]]
    edges = wp.matrix_from_cols(q[indices[i, 1]] - origin,
                                q[indices[i, 2]] - origin,
                                q[indices[i, 3]] - origin)
    ratio = wp.determinant(edges * rest_inverse[i])
    if not wp.isfinite(ratio):
        ratio = -wp.inf
    wp.atomic_min(minimum, 0, ratio)


@wp.kernel
def joint_drive_targets(dt: float, q_indices: wp.array[int], dofs: wp.array[int],
             kinds: wp.array[int], gains: wp.array[wp.vec3], limits: wp.array[wp.vec3],
             q: wp.array[float], qd: wp.array[float], requested: wp.array[float],
             limited_targets: wp.array[float], targets: wp.array[float],
             stiffness: wp.array[float], damping: wp.array[float]):
    i = wp.tid()
    dof, qi = dofs[i], q_indices[i]
    kp, kd = gains[i][0], gains[i][1]
    lower, upper, velocity_limit = limits[i][0], limits[i][1], limits[i][2]
    target = requested[dof]
    if velocity_limit > 0.0:
        target = limited_targets[i] + wp.clamp(target - limited_targets[i],
                                              -velocity_limit * dt, velocity_limit * dt)
    limited_targets[i] = target
    targets[dof] = target
    error = target - q[qi]
    if kinds[i] != 3:
        error = wp.atan2(wp.sin(error), wp.cos(error))
    # VBD solves these compliant drives implicitly with joints and contacts.
    # Bound the requested PD effort by scaling its impedance. This is a drive
    # command bound, not an exact bound on the subsequent constraint reaction.
    force = kp * error - kd * qd[dof]
    scale = float(1.0)
    if force > upper:
        scale = upper / force
    elif force < lower:
        scale = lower / force
    stiffness[dof] = kp * scale
    damping[dof] = kd * scale


@wp.kernel
def soft_contact_forces(
    dt: float, count: wp.array[int], indices: wp.array[wp.vec3i], barycentric: wp.array[wp.vec3],
    shape: wp.array[int], point: wp.array[wp.vec3], body_vel: wp.array[wp.vec3], normal: wp.array[wp.vec3],
    q: wp.array[wp.vec3], prev: wp.array[wp.vec3], radius: wp.array[float],
    body_q: wp.array[wp.transform], body_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector], com: wp.array[wp.vec3], shape_body: wp.array[int],
    margin: wp.array[float], ke: wp.array[float], kd: wp.array[float], mu: wp.array[float], epsilon: float,
    use_log_barrier: bool, contact_force: wp.array[wp.vec3], nodal_force: wp.array[wp.vec3], contact_position: wp.array[wp.vec3],
    maximum_penetration: wp.array[float],
):
    i = wp.tid()
    if i >= count[0]:
        return
    corners, weights = indices[i], barycentric[i]
    force = wp.vec3(0.0)
    position = wp.vec3(0.0)
    if wp.static(_HAS_LOG_BARRIER):
        force, hessian, position = _eval_soft_ef_contact(i, corners, weights, q, prev, radius,
                ke[i], kd[i], mu[i], epsilon, shape_body, body_q, body_prev, body_qd, com,
                shape, point, body_vel, normal, margin, dt, use_log_barrier)
    else:
        force, hessian, position = _eval_soft_ef_contact(i, corners, weights, q, prev, radius,
                ke[i], kd[i], mu[i], epsilon, shape_body, body_q, body_prev, body_qd, com,
                shape, point, body_vel, normal, margin, dt)
    contact_force[i] = force
    contact_position[i] = position
    soft_point = wp.vec3(0.0)
    soft_radius = float(0.0)
    for c in range(3):
        if corners[c] >= 0:
            wp.atomic_add(nodal_force, corners[c], weights[c] * force)
            soft_point += weights[c] * q[corners[c]]
            soft_radius = wp.max(soft_radius, radius[corners[c]])
    penetration = soft_radius + margin[shape[i]] - wp.dot(soft_point - position, normal[i])
    wp.atomic_max(maximum_penetration, 0, penetration)


@wp.kernel
def soft_belt_traction(count: wp.array[int], indices: wp.array[wp.vec3i], barycentric: wp.array[wp.vec3],
                       shape: wp.array[int], normal: wp.array[wp.vec3], force: wp.array[wp.vec3],
                       active: wp.array[int], traction: wp.array[wp.vec3]):
    i = wp.tid()
    if i >= count[0] or active[shape[i]] == 0:
        return
    tangential = force[i] - wp.dot(force[i], normal[i]) * normal[i]
    for c in range(3):
        node = indices[i][c]
        if node >= 0:
            weight = barycentric[i][c]
            wp.atomic_add(traction, node, weight * tangential)


@wp.kernel
def soft_external_forces(qd: wp.array[wp.vec3], mass: wp.array[float], damping: wp.array[float],
                         prescribed: wp.array[wp.vec3], forces: wp.array[wp.vec3]):
    i = wp.tid()
    forces[i] = prescribed[i] - mass[i] * damping[i] * qd[i]


@wp.kernel
def rigid_belt_load(dt: float, count: wp.array[int], shape0: wp.array[int], shape1: wp.array[int],
                    body0: wp.array[int], body1: wp.array[int], point0: wp.array[wp.vec3],
                    point1: wp.array[wp.vec3], normal: wp.array[wp.vec3], force: wp.array[wp.vec3],
                    velocity: wp.array[wp.vec3], active: wp.array[int], body_q: wp.array[wp.transform],
                    body_qd: wp.array[wp.spatial_vector], com: wp.array[wp.vec3], mass: wp.array[float],
                    mu: float, output: wp.array[wp.spatial_vector]):
    i = wp.tid()
    if i >= count[0]:
        return
    for side in range(2):
        surface = shape0[i]
        body = body1[i]
        point = point1[i]
        if side == 1:
            surface = shape1[i]
            body = body0[i]
            point = point0[i]
        if surface >= 0 and body >= 0 and mass[body] > 0.0 and active[surface] != 0:
            arm = point - wp.transform_point(body_q[body], com[body])
            v = wp.spatial_top(body_qd[body]) + wp.cross(wp.spatial_bottom(body_qd[body]), arm)
            load = wp.abs(wp.dot(force[i], normal[i]))
            slip = velocity[surface] - v
            slip -= wp.dot(slip, normal[i]) * normal[i]
            requested = mass[body] * slip / dt
            traction = wp.normalize(requested) * wp.min(wp.length(requested), mu * load)
            wp.atomic_add(output, body, wp.spatial_vector(traction, wp.cross(arm, traction)))


@wp.kernel
def copy_body_forces(source: wp.array[wp.spatial_vector], prescribed: wp.array[wp.spatial_vector],
                     destination: wp.array[wp.spatial_vector]):
    i = wp.tid()
    destination[i] = source[i] + prescribed[i]
