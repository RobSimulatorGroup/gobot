"""GPU-resident Warp CUDA kernels for batched FK, ESDF/primitive/self collision, and projected L-BFGS.

All kernels use standard public ``warp`` APIs (zero ``mujoco_warp._src`` dependency) and read dynamic
scene counts / optimization weights from small device buffers so CUDA Graphs remain valid across
in-place scene updates and penalty-continuation stages.
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})


# ---------------------------------------------------------------------------
# Device-side math & geometry helpers
# ---------------------------------------------------------------------------

@wp.func
def _rodrigues(axis: wp.vec3, angle: float) -> wp.mat33:
    s = wp.sin(angle)
    c = wp.cos(angle)
    omc = 1.0 - c
    x = axis[0]
    y = axis[1]
    z = axis[2]
    return wp.mat33(
        c + x * x * omc,       x * y * omc - z * s,   x * z * omc + y * s,
        y * x * omc + z * s,   c + y * y * omc,       y * z * omc - x * s,
        z * x * omc - y * s,   z * y * omc + x * s,   c + z * z * omc,
    )


@wp.func
def _rotation_log(m: wp.mat33) -> wp.vec3:
    """Shortest SO(3) rotation vector, matching C++ gobot::planning::rotation_log."""
    w = 1.0 + m[0, 0] + m[1, 1] + m[2, 2]
    x = 1.0 + m[0, 0] - m[1, 1] - m[2, 2]
    y = 1.0 - m[0, 0] + m[1, 1] - m[2, 2]
    z = 1.0 - m[0, 0] - m[1, 1] + m[2, 2]
    q0 = float(0.0)
    q1 = float(0.0)
    q2 = float(0.0)
    q3 = float(0.0)
    if w >= x and w >= y and w >= z:
        q0 = w
        q1 = m[2, 1] - m[1, 2]
        q2 = m[0, 2] - m[2, 0]
        q3 = m[1, 0] - m[0, 1]
    elif x >= y and x >= z:
        q0 = m[2, 1] - m[1, 2]
        q1 = x
        q2 = m[0, 1] + m[1, 0]
        q3 = m[0, 2] + m[2, 0]
    elif y >= z:
        q0 = m[0, 2] - m[2, 0]
        q1 = m[0, 1] + m[1, 0]
        q2 = y
        q3 = m[1, 2] + m[2, 1]
    else:
        q0 = m[1, 0] - m[0, 1]
        q1 = m[0, 2] + m[2, 0]
        q2 = m[1, 2] + m[2, 1]
        q3 = z
    if q0 < 0.0:
        q0 = -q0
        q1 = -q1
        q2 = -q2
        q3 = -q3
    vec = wp.vec3(q1, q2, q3)
    nrm = wp.length(vec)
    eps = 1.0e-7
    scale = float(0.0)
    if nrm > eps:
        scale = 2.0 * wp.atan2(nrm, q0) / nrm
    else:
        scale = 2.0 / wp.max(q0, eps)
    return vec * scale


@wp.func
def _so3_left_jacobian_inv(phi: wp.vec3) -> wp.mat33:
    theta = wp.length(phi)
    S = wp.mat33(
        0.0, -phi[2], phi[1],
        phi[2], 0.0, -phi[0],
        -phi[1], phi[0], 0.0,
    )
    S2 = S @ S
    eye = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    if theta < 1.0e-3:
        return eye - 0.5 * S + (1.0 / 12.0) * S2
    half = 0.5 * theta
    cot_half = wp.cos(half) / wp.sin(half)
    coeff = (1.0 - half * cot_half) / (theta * theta)
    return eye - 0.5 * S + coeff * S2


@wp.func
def _esdf_at(sdf: wp.array[float], ny: int, nz: int, ix: int, iy: int, iz: int) -> float:
    return sdf[(ix * ny + iy) * nz + iz]


@wp.func
def _esdf_collision_dist_grad(
    p: wp.vec3,
    sdf: wp.array[float],
    nx: int,
    ny: int,
    nz: int,
    origin: wp.vec3,
    resolution: float,
    from_occ: int,
    inflation_radius: float,
    max_distance: float,
    lipschitz_bound: float,
) -> wp.vec4:
    """Exact GPU counterpart of C++ VoxelESDF::collision_distance(p, &grad, true).

    Returns vec4(clearance_at_center, gx, gy, gz). Caller subtracts sphere_radius.
    """
    inv_res = 1.0 / resolution
    gx = (p[0] - origin[0]) * inv_res - 0.5
    gy = (p[1] - origin[1]) * inv_res - 0.5
    gz = (p[2] - origin[2]) * inv_res - 0.5

    max_gx = float(nx - 1)
    max_gy = float(ny - 1)
    max_gz = float(nz - 1)

    cgx = wp.clamp(gx, 0.0, max_gx)
    cgy = wp.clamp(gy, 0.0, max_gy)
    cgz = wp.clamp(gz, 0.0, max_gz)

    out_x = gx != cgx
    out_y = gy != cgy
    out_z = gz != cgz

    ix0 = int(0)
    ix1 = int(0)
    tx = float(0.0)
    if nx > 1:
        ix0 = wp.min(int(wp.floor(cgx)), nx - 2)
        ix1 = ix0 + 1
        tx = cgx - float(ix0)

    iy0 = int(0)
    iy1 = int(0)
    ty = float(0.0)
    if ny > 1:
        iy0 = wp.min(int(wp.floor(cgy)), ny - 2)
        iy1 = iy0 + 1
        ty = cgy - float(iy0)

    iz0 = int(0)
    iz1 = int(0)
    tz = float(0.0)
    if nz > 1:
        iz0 = wp.min(int(wp.floor(cgz)), nz - 2)
        iz1 = iz0 + 1
        tz = cgz - float(iz0)

    c000 = _esdf_at(sdf, ny, nz, ix0, iy0, iz0)
    c001 = _esdf_at(sdf, ny, nz, ix0, iy0, iz1)
    c010 = _esdf_at(sdf, ny, nz, ix0, iy1, iz0)
    c011 = _esdf_at(sdf, ny, nz, ix0, iy1, iz1)
    c100 = _esdf_at(sdf, ny, nz, ix1, iy0, iz0)
    c101 = _esdf_at(sdf, ny, nz, ix1, iy0, iz1)
    c110 = _esdf_at(sdf, ny, nz, ix1, iy1, iz0)
    c111 = _esdf_at(sdf, ny, nz, ix1, iy1, iz1)

    wx0 = 1.0 - tx
    wx1 = tx
    wy0 = 1.0 - ty
    wy1 = ty
    wz0 = 1.0 - tz
    wz1 = tz

    c00 = c000 * wz0 + c001 * wz1
    c01 = c010 * wz0 + c011 * wz1
    c10 = c100 * wz0 + c101 * wz1
    c11 = c110 * wz0 + c111 * wz1

    c0 = c00 * wy0 + c01 * wy1
    c1 = c10 * wy0 + c11 * wy1
    s = c0 * wx0 + c1 * wx1

    gin_x = float(0.0)
    gin_y = float(0.0)
    gin_z = float(0.0)
    if nx > 1:
        gin_x = (c1 - c0) * inv_res
    if ny > 1:
        gin_y = ((c01 - c00) * wx0 + (c11 - c10) * wx1) * inv_res
    if nz > 1:
        dz00 = c001 - c000
        dz01 = c011 - c010
        dz10 = c101 - c100
        dz11 = c111 - c110
        gin_z = ((dz00 * wy0 + dz01 * wy1) * wx0 + (dz10 * wy0 + dz11 * wy1) * wx1) * inv_res

    inside = not (out_x or out_y or out_z)
    if inside and (gin_x * gin_x + gin_y * gin_y + gin_z * gin_z) < 1.0e-12:
        cx = wp.clamp(int(wp.round(cgx)), 0, nx - 1)
        cy = wp.clamp(int(wp.round(cgy)), 0, ny - 1)
        cz = wp.clamp(int(wp.round(cgz)), 0, nz - 1)
        for step in range(1, 4):
            if (gin_x * gin_x + gin_y * gin_y + gin_z * gin_z) >= 1.0e-12:
                break
            xm = wp.max(0, cx - step)
            xp = wp.min(nx - 1, cx + step)
            ym = wp.max(0, cy - step)
            yp = wp.min(ny - 1, cy + step)
            zm = wp.max(0, cz - step)
            zp = wp.min(nz - 1, cz + step)
            if xp > xm:
                gin_x = (_esdf_at(sdf, ny, nz, xp, cy, cz) - _esdf_at(sdf, ny, nz, xm, cy, cz)) / (float(xp - xm) * resolution)
            if yp > ym:
                gin_y = (_esdf_at(sdf, ny, nz, cx, yp, cz) - _esdf_at(sdf, ny, nz, cx, ym, cz)) / (float(yp - ym) * resolution)
            if zp > zm:
                gin_z = (_esdf_at(sdf, ny, nz, cx, cy, zp) - _esdf_at(sdf, ny, nz, cx, cy, zm)) / (float(zp - zm) * resolution)

    val = s
    grad = wp.vec3(gin_x, gin_y, gin_z)

    if not inside:
        dx = (gx - cgx) * resolution
        dy = (gy - cgy) * resolution
        dz = (gz - cgz) * resolution
        d_out = wp.sqrt(dx * dx + dy * dy + dz * dz)
        offset = 0.5 * resolution if from_occ != 0 else 0.0
        edge = s + offset
        if edge >= 0.0:
            dist = wp.sqrt(edge * edge + d_out * d_out)
            if dist > 1.0e-12:
                inv_d = 1.0 / dist
                grad = wp.vec3(
                    dx * inv_d if out_x else edge * inv_d * gin_x,
                    dy * inv_d if out_y else edge * inv_d * gin_y,
                    dz * inv_d if out_z else edge * inv_d * gin_z,
                )
            else:
                grad = wp.vec3(0.0, 0.0, 0.0)
            val = dist - offset
        else:
            inv_do = 1.0 / d_out if d_out > 1.0e-12 else 0.0
            grad = wp.vec3(
                dx * inv_do if out_x else gin_x,
                dy * inv_do if out_y else gin_y,
                dz * inv_do if out_z else gin_z,
            )
            val = s + d_out

    if from_occ == 0:
        inv_l = 1.0 / lipschitz_bound
        return wp.vec4(val * inv_l, grad[0] * inv_l, grad[1] * inv_l, grad[2] * inv_l)

    # Conservative voxel interpolation allowance: (sqrt(1.5) - 0.5) * h.
    voxel_allowance = 0.724744871391589 * resolution
    shifted = val - voxel_allowance - inflation_radius
    if wp.abs(shifted) > max_distance:
        return wp.vec4(wp.clamp(shifted, -max_distance, max_distance), 0.0, 0.0, 0.0)
    return wp.vec4(shifted, grad[0], grad[1], grad[2])


@wp.func
def _primitive_sdf_grad(
    p: wp.vec3,
    kind: int,
    pos: wp.vec3,
    rot: wp.mat33,
    size: wp.vec3,
) -> wp.vec4:
    """Exact analytic signed distance and world-space unit gradient for static primitives."""
    if kind == 6:
        # Halfspace: free space is dot(p - pos, rot @ (0, 0, 1)) >= 0.
        n = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])
        d = wp.dot(p - pos, n)
        return wp.vec4(d, n[0], n[1], n[2])

    if kind == 2:
        diff = p - pos
        L = wp.length(diff)
        if L > 1.0e-8:
            g = diff / L
            return wp.vec4(L - size[0], g[0], g[1], g[2])
        return wp.vec4(-size[0], 1.0, 0.0, 0.0)

    pl = wp.transpose(rot) @ (p - pos)

    if kind == 1:
        # Oriented Box / OBB with half-extents size.
        ax = wp.abs(pl[0])
        ay = wp.abs(pl[1])
        az = wp.abs(pl[2])
        qx = ax - size[0]
        qy = ay - size[1]
        qz = az - size[2]
        sx = 1.0 if pl[0] >= 0.0 else -1.0
        sy = 1.0 if pl[1] >= 0.0 else -1.0
        sz = 1.0 if pl[2] >= 0.0 else -1.0
        ox = wp.max(qx, 0.0)
        oy = wp.max(qy, 0.0)
        oz = wp.max(qz, 0.0)
        out_len = wp.sqrt(ox * ox + oy * oy + oz * oz)
        if out_len > 1.0e-8:
            inv_l = 1.0 / out_len
            gl = wp.vec3(sx * ox * inv_l, sy * oy * inv_l, sz * oz * inv_l)
            gw = rot @ gl
            return wp.vec4(out_len, gw[0], gw[1], gw[2])
        # Inside or on boundary: closest face normal.
        if qx >= qy and qx >= qz:
            gw = rot @ wp.vec3(sx, 0.0, 0.0)
            return wp.vec4(qx, gw[0], gw[1], gw[2])
        if qy >= qz:
            gw = rot @ wp.vec3(0.0, sy, 0.0)
            return wp.vec4(qy, gw[0], gw[1], gw[2])
        gw = rot @ wp.vec3(0.0, 0.0, sz)
        return wp.vec4(qz, gw[0], gw[1], gw[2])

    if kind == 4:
        # Capsule: radius = size[0], half_height = size[1] along local Z.
        cz = wp.clamp(pl[2], -size[1], size[1])
        diff = wp.vec3(pl[0], pl[1], pl[2] - cz)
        L = wp.length(diff)
        if L > 1.0e-8:
            gw = rot @ (diff / L)
            return wp.vec4(L - size[0], gw[0], gw[1], gw[2])
        gw = rot @ wp.vec3(1.0, 0.0, 0.0)
        return wp.vec4(-size[0], gw[0], gw[1], gw[2])

    if kind == 3:
        # Cylinder: radius = size[0], half_height = size[1] along local Z.
        r_xy = wp.sqrt(pl[0] * pl[0] + pl[1] * pl[1])
        dr = r_xy - size[0]
        dz = wp.abs(pl[2]) - size[1]
        sz = 1.0 if pl[2] >= 0.0 else -1.0
        rx = pl[0] / r_xy if r_xy > 1.0e-8 else 1.0
        ry = pl[1] / r_xy if r_xy > 1.0e-8 else 0.0
        odr = wp.max(dr, 0.0)
        odz = wp.max(dz, 0.0)
        out_len = wp.sqrt(odr * odr + odz * odz)
        if out_len > 1.0e-8:
            inv_l = 1.0 / out_len
            gl = wp.vec3(rx * odr * inv_l, ry * odr * inv_l, sz * odz * inv_l)
            gw = rot @ gl
            return wp.vec4(out_len, gw[0], gw[1], gw[2])
        if dr >= dz:
            gw = rot @ wp.vec3(rx, ry, 0.0)
            return wp.vec4(dr, gw[0], gw[1], gw[2])
        gw = rot @ wp.vec3(0.0, 0.0, sz)
        return wp.vec4(dz, gw[0], gw[1], gw[2])

    return wp.vec4(1.0e30, 0.0, 0.0, 0.0)


@wp.func
def _huber_hinge(v: float, delta: float) -> wp.vec2:
    """Smooth C^1 Huber hinge penalty: returns vec2(value, d_value/d_v)."""
    if v <= 0.0:
        return wp.vec2(0.0, 0.0)
    if v <= delta:
        return wp.vec2(0.5 * v * v, v)
    return wp.vec2(delta * (v - 0.5 * delta), delta)


@wp.func
def _accum_point_grad(
    grad_q: wp.array3d[float],
    n: int,
    seg: int,
    T: int,
    fixed_start: int,
    fixed_end: int,
    w0: float,
    w1: float,
    body_id: int,
    p_world: wp.vec3,
    g_world: wp.vec3,
    scale: float,
    sid: int,
    J: int,
    body_ancestor: wp.array2d[wp.uint8],
    joint_prismatic: wp.array[wp.uint8],
    active_mask: wp.array2d[wp.uint8],
    req_idx: int,
    j_orig: wp.array2d[wp.vec3],
    j_axis: wp.array2d[wp.vec3],
):
    """Chain-rule world-point gradient g_world back to trajectory knots seg and seg+1."""
    k0_free = seg >= fixed_start and seg < T - fixed_end
    k1_free = (seg + 1) >= fixed_start and (seg + 1) < T - fixed_end
    if not k0_free and not k1_free:
        return
    for j in range(J):
        if active_mask[req_idx, j] == wp.uint8(0):
            continue
        if body_ancestor[body_id, j] == wp.uint8(0):
            continue
        axis = j_axis[sid, j]
        dq = float(0.0)
        if joint_prismatic[j] != wp.uint8(0):
            dq = wp.dot(g_world, axis) * scale
        else:
            dq = wp.dot(g_world, wp.cross(axis, p_world - j_orig[sid, j])) * scale
        if k0_free and w0 != 0.0:
            wp.atomic_add(grad_q, n, seg, j, w0 * dq)
        if k1_free and w1 != 0.0:
            wp.atomic_add(grad_q, n, seg + 1, j, w1 * dq)


# ---------------------------------------------------------------------------
# Direct FK & ESDF verification kernels (used by unit tests & diagnostics)
# ---------------------------------------------------------------------------

@wp.kernel
def fk_configs_kernel(
    q: wp.array2d[float],
    num_bodies: int,
    num_joints: int,
    parent: wp.array[int],
    kinds: wp.array[int],
    joint_idx: wp.array[int],
    body_branch: wp.array[int],
    a_rot: wp.array[wp.mat33],
    a_pos: wp.array[wp.vec3],
    b_rot: wp.array[wp.mat33],
    b_pos: wp.array[wp.vec3],
    axes: wp.array[wp.vec3],
    root_pos: wp.array[wp.vec3],
    root_rot: wp.array[wp.mat33],
    out_bpos: wp.array2d[wp.vec3],
    out_brot: wp.array2d[wp.mat33],
    out_jorig: wp.array2d[wp.vec3],
    out_jaxis: wp.array2d[wp.vec3],
):
    sid = wp.tid()
    for b in range(num_bodies):
        p = parent[b]
        if p < 0:
            br = body_branch[b]
            out_bpos[sid, b] = root_pos[br]
            out_brot[sid, b] = root_rot[br]
            continue
        j = joint_idx[b]
        kind = kinds[b]
        pr = out_brot[sid, p]
        pp = out_bpos[sid, p]
        anchor_r = pr @ a_rot[b]
        anchor_p = pp + pr @ a_pos[b]
        moving_r = anchor_r
        if kind == 1 or kind == 2:
            moving_r = anchor_r @ _rodrigues(axes[b], q[sid, j])
        if j >= 0:
            w_axis = anchor_r @ axes[b]
            out_jorig[sid, j] = anchor_p
            out_jaxis[sid, j] = w_axis
            if kind == 3:
                anchor_p = anchor_p + w_axis * q[sid, j]
        out_bpos[sid, b] = anchor_p + moving_r @ b_pos[b]
        out_brot[sid, b] = moving_r @ b_rot[b]


@wp.kernel
def query_esdf_spheres_kernel(
    bpos: wp.array2d[wp.vec3],
    brot: wp.array2d[wp.mat33],
    jorig: wp.array2d[wp.vec3],
    jaxis: wp.array2d[wp.vec3],
    num_spheres: int,
    num_joints: int,
    sphere_body: wp.array[int],
    sphere_center: wp.array[wp.vec3],
    sphere_radius: wp.array[float],
    sphere_esdf_active: wp.array[wp.uint8],
    body_ancestor: wp.array2d[wp.uint8],
    joint_prismatic: wp.array[wp.uint8],
    esdf_sdf: wp.array[float],
    scene_i32: wp.array[int],
    scene_f32: wp.array[float],
    out_dist: wp.array2d[float],
    out_grad_q: wp.array3d[float],
):
    tid = wp.tid()
    sid = tid // num_spheres
    sp = tid - sid * num_spheres
    has_esdf = scene_i32[0]
    if has_esdf == 0 or sphere_esdf_active[sp] == wp.uint8(0):
        out_dist[sid, sp] = 1.0e30
        for j in range(num_joints):
            out_grad_q[sid, sp, j] = 0.0
        return
    bid = sphere_body[sp]
    pw = bpos[sid, bid] + brot[sid, bid] @ sphere_center[sp]
    origin = wp.vec3(scene_f32[0], scene_f32[1], scene_f32[2])
    dg = _esdf_collision_dist_grad(
        pw, esdf_sdf, scene_i32[1], scene_i32[2], scene_i32[3],
        origin, scene_f32[3], scene_i32[4], scene_f32[4], scene_f32[5], scene_f32[6],
    )
    out_dist[sid, sp] = dg[0] - sphere_radius[sp]
    gw = wp.vec3(dg[1], dg[2], dg[3])
    for j in range(num_joints):
        if body_ancestor[bid, j] == wp.uint8(0):
            out_grad_q[sid, sp, j] = 0.0
        elif joint_prismatic[j] != wp.uint8(0):
            out_grad_q[sid, sp, j] = wp.dot(gw, jaxis[sid, j])
        else:
            out_grad_q[sid, sp, j] = wp.dot(gw, wp.cross(jaxis[sid, j], pw - jorig[sid, j]))


# ---------------------------------------------------------------------------
# Batched trajectory FK, Collision, Smoothness, Pose Constraints
# ---------------------------------------------------------------------------

@wp.kernel
def zero_eval_buffers_kernel(
    N: int,
    T: int,
    J: int,
    skip_accepted: int,
    accepted: wp.array[int],
    cost_col: wp.array[float],
    cost_pose: wp.array[float],
    cost_smooth: wp.array[float],
    cost_total: wp.array[float],
    max_col_viol: wp.array[float],
    max_other_viol: wp.array[float],
    grad_q: wp.array3d[float],
):
    n = wp.tid()
    if skip_accepted != 0 and accepted[n] != 0:
        return
    cost_col[n] = 0.0
    cost_pose[n] = 0.0
    cost_smooth[n] = 0.0
    cost_total[n] = 0.0
    max_col_viol[n] = 0.0
    max_other_viol[n] = 0.0
    for t in range(T):
        for j in range(J):
            grad_q[n, t, j] = 0.0


@wp.kernel
def fk_traj_samples_kernel(
    q: wp.array3d[float],
    M_samples: int,
    substeps: int,
    num_bodies: int,
    num_joints: int,
    skip_accepted: int,
    accepted: wp.array[int],
    parent: wp.array[int],
    kinds: wp.array[int],
    joint_idx: wp.array[int],
    body_branch: wp.array[int],
    a_rot: wp.array[wp.mat33],
    a_pos: wp.array[wp.vec3],
    b_rot: wp.array[wp.mat33],
    b_pos: wp.array[wp.vec3],
    axes: wp.array[wp.vec3],
    root_pos: wp.array[wp.vec3],
    root_rot: wp.array[wp.mat33],
    out_bpos: wp.array2d[wp.vec3],
    out_brot: wp.array2d[wp.mat33],
    out_jorig: wp.array2d[wp.vec3],
    out_jaxis: wp.array2d[wp.vec3],
):
    sid = wp.tid()
    n = sid // M_samples
    if skip_accepted != 0 and accepted[n] != 0:
        return
    m = sid - n * M_samples
    seg = m // substeps
    sub = m - seg * substeps
    u = float(sub) / float(substeps)
    w0 = 1.0 - u
    w1 = u

    for b in range(num_bodies):
        p = parent[b]
        if p < 0:
            br = body_branch[b]
            out_bpos[sid, b] = root_pos[br]
            out_brot[sid, b] = root_rot[br]
            continue
        j = joint_idx[b]
        kind = kinds[b]
        pr = out_brot[sid, p]
        pp = out_bpos[sid, p]
        anchor_r = pr @ a_rot[b]
        anchor_p = pp + pr @ a_pos[b]
        moving_r = anchor_r
        if j >= 0:
            qj = w0 * q[n, seg, j] + w1 * q[n, seg + 1, j]
            if kind == 1 or kind == 2:
                moving_r = anchor_r @ _rodrigues(axes[b], qj)
            w_axis = anchor_r @ axes[b]
            out_jorig[sid, j] = anchor_p
            out_jaxis[sid, j] = w_axis
            if kind == 3:
                anchor_p = anchor_p + w_axis * qj
        out_bpos[sid, b] = anchor_p + moving_r @ b_pos[b]
        out_brot[sid, b] = moving_r @ b_rot[b]


@wp.kernel
def eval_collision_cost_grad_kernel(
    M_samples: int,
    sphere_stride: int,
    substeps: int,
    T: int,
    J: int,
    K_seeds: int,
    fixed_start: int,
    fixed_end: int,
    skip_accepted: int,
    accepted: wp.array[int],
    bpos: wp.array2d[wp.vec3],
    brot: wp.array2d[wp.mat33],
    jorig: wp.array2d[wp.vec3],
    jaxis: wp.array2d[wp.vec3],
    body_ancestor: wp.array2d[wp.uint8],
    joint_prismatic: wp.array[wp.uint8],
    active_mask: wp.array2d[wp.uint8],
    sphere_body: wp.array[int],
    sphere_center: wp.array[wp.vec3],
    sphere_radius: wp.array[float],
    sphere_esdf_active: wp.array[wp.uint8],
    static_csr_off: wp.array[int],
    static_csr_idx: wp.array[int],
    self_csr_off: wp.array[int],
    self_csr_idx: wp.array[int],
    static_kind: wp.array[int],
    static_pos: wp.array[wp.vec3],
    static_rot: wp.array[wp.mat33],
    static_size: wp.array[wp.vec3],
    esdf_sdf: wp.array[float],
    scene_i32: wp.array[int],
    scene_f32: wp.array[float],
    opt_f32: wp.array[float],
    cost_col: wp.array[float],
    max_col_viol: wp.array[float],
    grad_q: wp.array3d[float],
):
    tid = wp.tid()
    sid = tid // sphere_stride
    sp = tid - sid * sphere_stride

    # scene_i32: [has_esdf, nx, ny, nz, from_occ, num_spheres, num_self_pairs, num_static_pairs]
    num_spheres = scene_i32[5]
    if sp >= num_spheres:
        return
    n = sid // M_samples
    if skip_accepted != 0 and accepted[n] != 0:
        return
    m = sid - n * M_samples
    seg = m // substeps
    sub = m - seg * substeps
    if seg == 0 and sub == 0:
        return

    req_idx = n // K_seeds
    u = float(sub) / float(substeps)
    w0 = 1.0 - u
    w1 = u

    has_esdf = scene_i32[0]
    nx = scene_i32[1]
    ny = scene_i32[2]
    nz = scene_i32[3]
    from_occ = scene_i32[4]

    w_col = opt_f32[2] / float(substeps)
    self_margin = opt_f32[6]
    world_margin = opt_f32[7]
    world_cushion = opt_f32[8]
    self_cushion = opt_f32[9]
    huber_delta = opt_f32[10]

    eta_world = world_margin + world_cushion
    eta_self = self_margin + self_cushion

    bid = sphere_body[sp]
    rad = sphere_radius[sp]
    pw = bpos[sid, bid] + brot[sid, bid] @ sphere_center[sp]

    local_cost = float(0.0)
    local_max_v = float(0.0)
    total_gw = wp.vec3(0.0, 0.0, 0.0)

    # 1. Sphere-to-ESDF
    if has_esdf != 0 and sphere_esdf_active[sp] != wp.uint8(0):
        origin = wp.vec3(scene_f32[0], scene_f32[1], scene_f32[2])
        res = scene_f32[3]
        inf_r = scene_f32[4]
        max_d = scene_f32[5]
        lip = scene_f32[6]
        dg = _esdf_collision_dist_grad(pw, esdf_sdf, nx, ny, nz, origin, res, from_occ, inf_r, max_d, lip)
        clearance = dg[0] - rad
        v = eta_world - clearance
        if v > 0.0:
            v_true = wp.max(0.0, world_margin - clearance)
            local_max_v = wp.max(local_max_v, v_true)
            hh = _huber_hinge(v, huber_delta)
            local_cost = local_cost + w_col * hh[0]
            total_gw = total_gw - (w_col * hh[1]) * wp.vec3(dg[1], dg[2], dg[3])

    # 2. Sphere-to-Static Analytic Primitives (CSR neighbors of sphere sp)
    st_beg = static_csr_off[sp]
    st_end = static_csr_off[sp + 1]
    for ci in range(st_beg, st_end):
        sb = static_csr_idx[ci]
        dg = _primitive_sdf_grad(pw, static_kind[sb], static_pos[sb], static_rot[sb], static_size[sb])
        clearance = dg[0] - rad
        v = eta_world - clearance
        if v > 0.0:
            v_true = wp.max(0.0, world_margin - clearance)
            local_max_v = wp.max(local_max_v, v_true)
            hh = _huber_hinge(v, huber_delta)
            local_cost = local_cost + w_col * hh[0]
            total_gw = total_gw - (w_col * hh[1]) * wp.vec3(dg[1], dg[2], dg[3])

    # 3. Self-Collision Sphere-to-Sphere (symmetric CSR neighbors of sphere sp)
    sf_beg = self_csr_off[sp]
    sf_end = self_csr_off[sp + 1]
    for ci in range(sf_beg, sf_end):
        sb = self_csr_idx[ci]
        bb = sphere_body[sb]
        pb = bpos[sid, bb] + brot[sid, bb] @ sphere_center[sb]
        diff = pw - pb
        L = wp.length(diff)
        clearance = L - (rad + sphere_radius[sb])
        v = eta_self - clearance
        if v > 0.0:
            v_true = wp.max(0.0, self_margin - clearance)
            local_max_v = wp.max(local_max_v, v_true)
            hh = _huber_hinge(v, huber_delta)
            # Symmetric CSR visits each unordered pair twice (a->b and b->a), so add 0.5*cost here.
            local_cost = local_cost + 0.5 * w_col * hh[0]
            gw = diff / L if L > 1.0e-8 else wp.vec3(1.0, 0.0, 0.0)
            total_gw = total_gw - (w_col * hh[1]) * gw

    if wp.dot(total_gw, total_gw) > 0.0:
        _accum_point_grad(
            grad_q, n, seg, T, fixed_start, fixed_end, w0, w1,
            bid, pw, total_gw, 1.0, sid, J,
            body_ancestor, joint_prismatic, active_mask, req_idx, jorig, jaxis,
        )
    if local_cost > 0.0:
        wp.atomic_add(cost_col, n, local_cost)
    if local_max_v > 0.0:
        wp.atomic_max(max_col_viol, n, local_max_v)


@wp.kernel
def eval_pose_constraints_kernel(
    num_pose_items: int,
    M_samples: int,
    substeps: int,
    T: int,
    J: int,
    K_seeds: int,
    fixed_start: int,
    fixed_end: int,
    skip_accepted: int,
    accepted: wp.array[int],
    bpos: wp.array2d[wp.vec3],
    brot: wp.array2d[wp.mat33],
    jorig: wp.array2d[wp.vec3],
    jaxis: wp.array2d[wp.vec3],
    body_ancestor: wp.array2d[wp.uint8],
    joint_prismatic: wp.array[wp.uint8],
    active_mask: wp.array2d[wp.uint8],
    # Compact pose constraint list across all requests in the chunk:
    pc_req: wp.array[int],
    pc_t: wp.array[int],
    pc_link: wp.array[int],
    pc_flags: wp.array[int],        # bit0=has_pos, bit1=has_rot
    pc_pos: wp.array[wp.vec3],
    pc_rot: wp.array[wp.mat33],
    pc_pos_tol: wp.array[wp.vec3],  # < 0 means free axis
    pc_rot_tol: wp.array[wp.vec3],  # < 0 means free axis
    pc_weight: wp.array[float],
    opt_f32: wp.array[float],
    cost_pose: wp.array[float],
    max_other_viol: wp.array[float],
    grad_q: wp.array3d[float],
):
    tid = wp.tid()
    idx = tid // K_seeds
    if idx >= num_pose_items:
        return
    k_seed = tid - idx * K_seeds
    req_idx = pc_req[idx]
    n = req_idx * K_seeds + k_seed
    if skip_accepted != 0 and accepted[n] != 0:
        return
    t = pc_t[idx]
    if t < fixed_start or t >= T - fixed_end:
        return

    sid = n * M_samples + t * substeps
    bid = pc_link[idx]
    p = bpos[sid, bid]
    R = brot[sid, bid]
    flags = pc_flags[idx]
    w_base = opt_f32[5] * pc_weight[idx]

    local_cost = float(0.0)
    local_viol = float(0.0)

    # Position component
    if (flags & 1) != 0:
        dp = p - pc_pos[idx]
        ptol = pc_pos_tol[idx]
        g_p = wp.vec3(0.0, 0.0, 0.0)
        if ptol[0] >= 0.0 and ptol[0] == ptol[1] and ptol[0] == ptol[2]:
            local_viol = wp.max(local_viol, wp.length(dp) - ptol[0])
        for a in range(3):
            if ptol[a] >= 0.0:
                err = dp[a]
                v = wp.abs(err) - ptol[a]
                if v > 0.0:
                    local_viol = wp.max(local_viol, v)
                local_cost = local_cost + 0.5 * w_base * err * err
                if a == 0:
                    g_p = wp.vec3(w_base * err, g_p[1], g_p[2])
                elif a == 1:
                    g_p = wp.vec3(g_p[0], w_base * err, g_p[2])
                else:
                    g_p = wp.vec3(g_p[0], g_p[1], w_base * err)
        for j in range(J):
            if active_mask[req_idx, j] == wp.uint8(0) or body_ancestor[bid, j] == wp.uint8(0):
                continue
            axis = jaxis[sid, j]
            dq = wp.dot(g_p, axis) if joint_prismatic[j] != wp.uint8(0) else wp.dot(g_p, wp.cross(axis, p - jorig[sid, j]))
            wp.atomic_add(grad_q, n, t, j, dq)

    # Rotation component
    if (flags & 2) != 0:
        rtol = pc_rot_tol[idx]
        free_count = int(0)
        free_axis = int(-1)
        for a in range(3):
            if rtol[a] < 0.0:
                free_count = free_count + 1
                free_axis = a
        Rt = wp.transpose(pc_rot[idx])
        if free_count == 0:
            phi = _rotation_log(Rt @ R)
            if rtol[0] == rtol[1] and rtol[0] == rtol[2]:
                local_viol = wp.max(local_viol, wp.length(phi) - rtol[0])
            for a in range(3):
                local_viol = wp.max(local_viol, wp.abs(phi[a]) - rtol[a])
            local_cost = local_cost + 0.5 * w_base * wp.dot(phi, phi)
            Jinv_Rt = _so3_left_jacobian_inv(phi) @ Rt
            w_phi = w_base * phi
            for j in range(J):
                if active_mask[req_idx, j] == wp.uint8(0) or body_ancestor[bid, j] == wp.uint8(0):
                    continue
                if joint_prismatic[j] != wp.uint8(0):
                    continue
                dphi = Jinv_Rt @ jaxis[sid, j]
                wp.atomic_add(grad_q, n, t, j, wp.dot(w_phi, dphi))
        elif free_count == 1:
            u = wp.vec3(R[0, free_axis], R[1, free_axis], R[2, free_axis])
            ut = Rt @ u
            a0 = (free_axis + 1) % 3
            a1 = (free_axis + 2) % 3
            sin_ang = wp.sqrt(ut[a0] * ut[a0] + ut[a1] * ut[a1])
            ang = wp.atan2(sin_ang, ut[free_axis])
            tol = wp.min(rtol[a0], rtol[a1])
            local_viol = wp.max(local_viol, ang - tol)
            # For a world angular perturbation, d(theta) is its component
            # along the unit rotation axis from the target axis to the link axis.
            local_cost = local_cost + 0.5 * w_base * ang * ang
            axis_target = wp.vec3(0.0, 0.0, 0.0)
            if sin_ang > 0.0:
                axis_target[a0] = -ut[a1] / sin_ang
                axis_target[a1] = ut[a0] / sin_ang
            elif ut[free_axis] < 0.0:
                # At the antipode the descent direction is nonunique.
                axis_target[a0] = 1.0
            g_rot = (w_base * ang) * (pc_rot[idx] @ axis_target)
            for j in range(J):
                if active_mask[req_idx, j] == wp.uint8(0) or body_ancestor[bid, j] == wp.uint8(0):
                    continue
                if joint_prismatic[j] == wp.uint8(0):
                    wp.atomic_add(grad_q, n, t, j, wp.dot(g_rot, jaxis[sid, j]))
        elif free_count == 2:
            phi = _rotation_log(Rt @ R)
            Jinv_Rt = _so3_left_jacobian_inv(phi) @ Rt
            for a in range(3):
                if rtol[a] < 0.0:
                    continue
                err = phi[a]
                local_viol = wp.max(local_viol, wp.abs(err) - rtol[a])
                local_cost = local_cost + 0.5 * w_base * err * err
                row_v = wp.vec3(Jinv_Rt[a, 0], Jinv_Rt[a, 1], Jinv_Rt[a, 2])
                for j in range(J):
                    if active_mask[req_idx, j] == wp.uint8(0) or body_ancestor[bid, j] == wp.uint8(0):
                        continue
                    if joint_prismatic[j] != wp.uint8(0):
                        continue
                    wp.atomic_add(grad_q, n, t, j, w_base * err * wp.dot(row_v, jaxis[sid, j]))

    if local_cost > 0.0:
        wp.atomic_add(cost_pose, n, local_cost)
    if local_viol > 0.0:
        wp.atomic_max(max_other_viol, n, local_viol)


@wp.kernel
def eval_smooth_limits_targets_kernel(
    q: wp.array3d[float],
    T: int,
    J: int,
    K_seeds: int,
    fixed_start: int,
    fixed_end: int,
    skip_accepted: int,
    accepted: wp.array[int],
    active_mask: wp.array2d[wp.uint8],
    velocity_limit: wp.array[float],
    jt_has: wp.array2d[wp.uint8],
    jt_vals: wp.array3d[float],
    has_start_vel: wp.array[wp.uint8],
    start_vel: wp.array2d[float],
    opt_f32: wp.array[float],
    cost_col: wp.array[float],
    cost_pose: wp.array[float],
    cost_smooth: wp.array[float],
    cost_total: wp.array[float],
    max_other_viol: wp.array[float],
    grad_q: wp.array3d[float],
):
    n = wp.tid()
    if skip_accepted != 0 and accepted[n] != 0:
        return
    req_idx = n // K_seeds

    w_v = opt_f32[0]
    w_a = opt_f32[1]
    w_vel = opt_f32[3]
    w_target = opt_f32[4]
    inv_dt = opt_f32[11]
    vel_scale = opt_f32[12]
    dt_start_smooth = opt_f32[13]

    c_smooth = float(0.0)
    c_extra = float(0.0)
    local_viol = float(0.0)

    # First-order velocity smoothness + optional fixed-dt velocity limit penalty.
    for t in range(T - 1):
        k0_free = t >= fixed_start and t < T - fixed_end
        k1_free = (t + 1) >= fixed_start and (t + 1) < T - fixed_end
        for j in range(J):
            if active_mask[req_idx, j] == wp.uint8(0):
                continue
            dq = q[n, t + 1, j] - q[n, t, j]
            c_smooth = c_smooth + 0.5 * w_v * dq * dq
            g_dq = w_v * dq
            if inv_dt > 0.0:
                vm = velocity_limit[j] * vel_scale
                if vm < 1.0e15:
                    speed = wp.abs(dq) * inv_dt
                    ex = speed - vm
                    if ex > 0.0:
                        local_viol = wp.max(local_viol, ex)
                        c_extra = c_extra + 0.5 * w_vel * ex * ex
                        sgn = 1.0 if dq >= 0.0 else -1.0
                        g_dq = g_dq + w_vel * ex * inv_dt * sgn
            if k0_free:
                grad_q[n, t, j] = grad_q[n, t, j] - g_dq
            if k1_free:
                grad_q[n, t + 1, j] = grad_q[n, t + 1, j] + g_dq

    # Second-order acceleration smoothness.
    if w_a > 0.0:
        for t in range(1, T - 1):
            k_prev = (t - 1) >= fixed_start and (t - 1) < T - fixed_end
            k_cur = t >= fixed_start and t < T - fixed_end
            k_next = (t + 1) >= fixed_start and (t + 1) < T - fixed_end
            for j in range(J):
                if active_mask[req_idx, j] == wp.uint8(0):
                    continue
                ddq = q[n, t + 1, j] - 2.0 * q[n, t, j] + q[n, t - 1, j]
                c_smooth = c_smooth + 0.5 * w_a * ddq * ddq
                ga = w_a * ddq
                if k_prev:
                    grad_q[n, t - 1, j] = grad_q[n, t - 1, j] + ga
                if k_cur:
                    grad_q[n, t, j] = grad_q[n, t, j] - 2.0 * ga
                if k_next:
                    grad_q[n, t + 1, j] = grad_q[n, t + 1, j] + ga

    # Optional start_velocity virtual-predecessor smoothness term at t=1.
    if has_start_vel[req_idx] != wp.uint8(0) and T > 2 and fixed_start == 1:
        dt_s = dt_start_smooth if dt_start_smooth > 0.0 else 0.1
        for j in range(J):
            if active_mask[req_idx, j] == wp.uint8(0):
                continue
            r = (q[n, 1, j] - q[n, 0, j]) - dt_s * start_vel[req_idx, j]
            c_smooth = c_smooth + 0.5 * w_a * r * r
            grad_q[n, 1, j] = grad_q[n, 1, j] + w_a * r

    # Waypoint joint targets (hard-pinned during L-BFGS; check tolerance for ranking).
    for t in range(fixed_start, T - fixed_end):
        if jt_has[req_idx, t] == wp.uint8(0):
            continue
        for j in range(J):
            if active_mask[req_idx, j] == wp.uint8(0):
                continue
            err = q[n, t, j] - jt_vals[req_idx, t, j]
            local_viol = wp.max(local_viol, wp.abs(err))
            c_extra = c_extra + 0.5 * w_target * err * err
            grad_q[n, t, j] = 0.0

    # Enforce zero gradient on fixed endpoints, hard-pinned joint targets, and locked joints.
    for t in range(T):
        is_fixed = t < fixed_start or t >= T - fixed_end or jt_has[req_idx, t] != wp.uint8(0)
        for j in range(J):
            if is_fixed or active_mask[req_idx, j] == wp.uint8(0):
                grad_q[n, t, j] = 0.0

    cost_smooth[n] = c_smooth
    cost_total[n] = c_smooth + c_extra + cost_col[n] + cost_pose[n]
    if local_viol > 0.0:
        wp.atomic_max(max_other_viol, n, local_viol)


# ---------------------------------------------------------------------------
# Batched Projected L-BFGS (zero host sync inside iteration loop)
# ---------------------------------------------------------------------------

@wp.kernel
def lbfgs_reset_history_kernel(
    hist_head: wp.array[int],
    hist_count: wp.array[int],
    accepted: wp.array[int],
):
    n = wp.tid()
    hist_head[n] = 0
    hist_count[n] = 0
    accepted[n] = 0


@wp.kernel
def lbfgs_direction_kernel(
    x: wp.array3d[float],
    g: wp.array3d[float],
    T: int,
    J: int,
    M_hist: int,
    max_step: float,
    lower: wp.array[float],
    upper: wp.array[float],
    s_hist: wp.array3d[float],      # [N, M_hist, D]
    y_hist: wp.array3d[float],      # [N, M_hist, D]
    rho_hist: wp.array2d[float],    # [N, M_hist]
    hist_head: wp.array[int],
    hist_count: wp.array[int],
    alpha_buf: wp.array2d[float],   # [N, M_hist]
    d_out: wp.array3d[float],       # [N, T, J]
    step_alpha: wp.array[float],
    accepted: wp.array[int],
):
    n = wp.tid()
    D = T * J
    cnt = hist_count[n]
    head = hist_head[n]

    # Initialize q_vec = projected gradient into d_out (so bound-active blocked components are 0).
    g_inf = float(0.0)
    for t in range(T):
        for j in range(J):
            gv = g[n, t, j]
            xv = x[n, t, j]
            if (xv <= lower[j] + 1.0e-7 and gv > 0.0) or (xv >= upper[j] - 1.0e-7 and gv < 0.0):
                gv = 0.0
            d_out[n, t, j] = gv
            g_inf = wp.max(g_inf, wp.abs(gv))

    if g_inf < 1.0e-9:
        for t in range(T):
            for j in range(J):
                d_out[n, t, j] = 0.0
        step_alpha[n] = 0.0
        accepted[n] = -1  # converged: skip trials and keep the current iterate
        return

    if cnt == 0:
        scale = 1.0
        if g_inf > max_step:
            scale = max_step / g_inf
        for t in range(T):
            for j in range(J):
                d_out[n, t, j] = -scale * d_out[n, t, j]
        step_alpha[n] = 1.0
        accepted[n] = 0
        return

    # Two-loop L-BFGS recursion (newest -> oldest).
    for back in range(cnt):
        slot = (head - 1 - back + M_hist * 4) % M_hist
        dot_sq = float(0.0)
        for t in range(T):
            row = t * J
            for j in range(J):
                dot_sq = dot_sq + s_hist[n, slot, row + j] * d_out[n, t, j]
        a = rho_hist[n, slot] * dot_sq
        alpha_buf[n, slot] = a
        for t in range(T):
            row = t * J
            for j in range(J):
                d_out[n, t, j] = d_out[n, t, j] - a * y_hist[n, slot, row + j]

    # Initial Hessian scaling gamma = (s^T y) / (y^T y) from most recent pair.
    newest = (head - 1 + M_hist * 4) % M_hist
    yy = float(0.0)
    for k in range(D):
        yv = y_hist[n, newest, k]
        yy = yy + yv * yv
    rho_new = rho_hist[n, newest]
    gamma = float(1.0)
    if yy > 1.0e-12 and rho_new > 0.0:
        gamma = wp.clamp(1.0 / (rho_new * yy), 1.0e-4, 1.0e2)
    for t in range(T):
        for j in range(J):
            d_out[n, t, j] = gamma * d_out[n, t, j]

    # Forward pass (oldest -> newest).
    for fwd in range(cnt):
        slot = (head - cnt + fwd + M_hist * 4) % M_hist
        dot_yr = float(0.0)
        for t in range(T):
            row = t * J
            for j in range(J):
                dot_yr = dot_yr + y_hist[n, slot, row + j] * d_out[n, t, j]
        beta = rho_hist[n, slot] * dot_yr
        diff = alpha_buf[n, slot] - beta
        for t in range(T):
            row = t * J
            for j in range(J):
                d_out[n, t, j] = d_out[n, t, j] + s_hist[n, slot, row + j] * diff

    # Negate to descent direction, mask bound-violating components, verify descent.
    g_dot_d = float(0.0)
    d_inf = float(0.0)
    for t in range(T):
        for j in range(J):
            dv = -d_out[n, t, j]
            xv = x[n, t, j]
            if (xv <= lower[j] + 1.0e-7 and dv < 0.0) or (xv >= upper[j] - 1.0e-7 and dv > 0.0):
                dv = 0.0
            d_out[n, t, j] = dv
            g_dot_d = g_dot_d + g[n, t, j] * dv
            d_inf = wp.max(d_inf, wp.abs(dv))

    if not wp.isfinite(g_dot_d) or g_dot_d >= -1.0e-12 or d_inf < 1.0e-12:
        # Fallback to projected negative gradient.
        scale = 1.0
        if g_inf > max_step:
            scale = max_step / g_inf
        for t in range(T):
            for j in range(J):
                gv = g[n, t, j]
                xv = x[n, t, j]
                if (xv <= lower[j] + 1.0e-7 and gv > 0.0) or (xv >= upper[j] - 1.0e-7 and gv < 0.0):
                    gv = 0.0
                d_out[n, t, j] = -scale * gv
    elif d_inf > max_step:
        scale = max_step / d_inf
        for t in range(T):
            for j in range(J):
                d_out[n, t, j] = scale * d_out[n, t, j]

    step_alpha[n] = 1.0
    accepted[n] = 0


@wp.kernel
def lbfgs_trial_step_kernel(
    x: wp.array3d[float],
    d: wp.array3d[float],
    T: int,
    J: int,
    K_seeds: int,
    fixed_start: int,
    fixed_end: int,
    lower: wp.array[float],
    upper: wp.array[float],
    active_mask: wp.array2d[wp.uint8],
    jt_has: wp.array2d[wp.uint8],
    step_alpha: wp.array[float],
    accepted: wp.array[int],
    x_trial: wp.array3d[float],
):
    tid = wp.tid()
    D = T * J
    n = tid // D
    if accepted[n] != 0:
        return
    rem = tid - n * D
    t = rem // J
    j = rem - t * J
    req_idx = n // K_seeds
    if t < fixed_start or t >= T - fixed_end or jt_has[req_idx, t] != wp.uint8(0) or active_mask[req_idx, j] == wp.uint8(0):
        x_trial[n, t, j] = x[n, t, j]
        return
    val = x[n, t, j] + step_alpha[n] * d[n, t, j]
    x_trial[n, t, j] = wp.clamp(val, lower[j], upper[j])


@wp.kernel
def lbfgs_armijo_check_kernel(
    x: wp.array3d[float],
    x_trial: wp.array3d[float],
    g: wp.array3d[float],
    f: wp.array[float],
    f_trial: wp.array[float],
    T: int,
    J: int,
    c1: float,
    shrink: float,
    step_alpha: wp.array[float],
    accepted: wp.array[int],
):
    n = wp.tid()
    if accepted[n] != 0:
        return
    ft = f_trial[n]
    fc = f[n]
    if not wp.isfinite(ft):
        step_alpha[n] = step_alpha[n] * shrink
        return
    # Exact projected-step directional derivative: g^T (x_trial - x).
    g_dot_dp = float(0.0)
    dp_norm2 = float(0.0)
    for t in range(T):
        for j in range(J):
            dp = x_trial[n, t, j] - x[n, t, j]
            g_dot_dp = g_dot_dp + g[n, t, j] * dp
            dp_norm2 = dp_norm2 + dp * dp
    if dp_norm2 <= 0.0:
        step_alpha[n] = step_alpha[n] * shrink
        return
    # Accept if Armijo condition holds on the projected step (small fp32 tolerance).
    if g_dot_dp < 0.0 and ft <= fc + c1 * g_dot_dp + 1.0e-6 * wp.abs(fc):
        accepted[n] = 1
    else:
        step_alpha[n] = step_alpha[n] * shrink


@wp.kernel
def lbfgs_update_history_kernel(
    x: wp.array3d[float],
    g: wp.array3d[float],
    f: wp.array[float],
    cost_smooth: wp.array[float],
    max_col_viol: wp.array[float],
    max_other_viol: wp.array[float],
    x_trial: wp.array3d[float],
    g_trial: wp.array3d[float],
    f_trial: wp.array[float],
    cost_smooth_trial: wp.array[float],
    max_col_viol_trial: wp.array[float],
    max_other_viol_trial: wp.array[float],
    T: int,
    J: int,
    M_hist: int,
    s_hist: wp.array3d[float],
    y_hist: wp.array3d[float],
    rho_hist: wp.array2d[float],
    hist_head: wp.array[int],
    hist_count: wp.array[int],
    accepted: wp.array[int],
):
    n = wp.tid()
    if accepted[n] < 0:
        return
    if accepted[n] == 0:
        # All line-search steps failed: preserve original x, g, f and reset L-BFGS history.
        hist_head[n] = 0
        hist_count[n] = 0
        return

    sy = float(0.0)
    ss = float(0.0)
    yy = float(0.0)
    for t in range(T):
        for j in range(J):
            sv = x_trial[n, t, j] - x[n, t, j]
            yv = g_trial[n, t, j] - g[n, t, j]
            sy = sy + sv * yv
            ss = ss + sv * sv
            yy = yy + yv * yv

    if sy > 1.0e-7 * wp.sqrt(ss * yy) and sy > 1.0e-12:
        slot = hist_head[n]
        for t in range(T):
            row = t * J
            for j in range(J):
                s_hist[n, slot, row + j] = x_trial[n, t, j] - x[n, t, j]
                y_hist[n, slot, row + j] = g_trial[n, t, j] - g[n, t, j]
        rho_hist[n, slot] = 1.0 / sy
        hist_head[n] = (slot + 1) % M_hist
        hist_count[n] = wp.min(M_hist, hist_count[n] + 1)

    for t in range(T):
        for j in range(J):
            x[n, t, j] = x_trial[n, t, j]
            g[n, t, j] = g_trial[n, t, j]
    f[n] = f_trial[n]
    cost_smooth[n] = cost_smooth_trial[n]
    max_col_viol[n] = max_col_viol_trial[n]
    max_other_viol[n] = max_other_viol_trial[n]


@wp.kernel
def rank_select_top_k_kernel(
    B_chunk: int,
    K_seeds: int,
    top_m: int,
    T: int,
    J: int,
    x: wp.array3d[float],
    cost_smooth: wp.array[float],
    max_col_viol: wp.array[float],
    max_other_viol: wp.array[float],
    seed_goal_idx: wp.array[int],
    out_top_x: wp.array4d[float],       # [B_chunk, top_m, T, J]
    out_top_goal: wp.array2d[int],      # [B_chunk, top_m]
    out_top_seed: wp.array2d[int],      # [B_chunk, top_m]
    out_top_viol: wp.array2d[float],    # [B_chunk, top_m]
    out_top_smooth: wp.array2d[float],  # [B_chunk, top_m]
    used_mask: wp.array2d[wp.uint8],    # [B_chunk, K_seeds]
):
    """Select top-M distinct ranked seeds per request directly on GPU (feasibility-first, then smoothness)."""
    b = wp.tid()
    if b >= B_chunk:
        return
    for k in range(K_seeds):
        used_mask[b, k] = wp.uint8(0)
    base = b * K_seeds
    m_count = wp.min(top_m, K_seeds)
    for rank in range(m_count):
        best_k = int(-1)
        best_feas = int(0)
        best_viol = float(1.0e30)
        best_smooth = float(1.0e30)
        for k in range(K_seeds):
            if used_mask[b, k] != wp.uint8(0):
                continue
            n = base + k
            v = wp.max(max_col_viol[n], max_other_viol[n])
            sm = cost_smooth[n]
            if not wp.isfinite(v) or not wp.isfinite(sm):
                v = 1.0e29
                sm = 1.0e29
            feas = 1 if v <= 1.0e-4 else 0
            better = False
            if best_k < 0:
                better = True
            elif feas > best_feas:
                better = True
            elif feas == best_feas:
                if feas == 1:
                    better = sm < best_smooth
                else:
                    better = (v < best_viol - 1.0e-5) or (wp.abs(v - best_viol) <= 1.0e-5 and sm < best_smooth)
            if better:
                best_k = k
                best_feas = feas
                best_viol = v
                best_smooth = sm
        if best_k < 0:
            best_k = 0
        used_mask[b, best_k] = wp.uint8(1)
        n_win = base + best_k
        out_top_goal[b, rank] = seed_goal_idx[n_win]
        out_top_seed[b, rank] = best_k
        out_top_viol[b, rank] = best_viol
        out_top_smooth[b, rank] = best_smooth
        for t in range(T):
            for j in range(J):
                out_top_x[b, rank, t, j] = x[n_win, t, j]
