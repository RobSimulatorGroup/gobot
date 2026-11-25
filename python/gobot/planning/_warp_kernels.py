"""MuJoCo Warp GPU kernels for indexed convex distance queries."""

import warp as wp
from mujoco_warp._src.collision_core import Geom
from mujoco_warp._src.collision_gjk import ccd, support
from mujoco_warp._src.types import GeomType, MJ_MAX_EPAFACES

wp.set_module_options({"enable_backward": False})
FrameMat = wp.types.matrix(shape=(3, 4), dtype=wp.float32)
WARP_ITERATIONS = 64
WARP_TILE = 16384
WARP_TYPES = (None, GeomType.BOX, GeomType.SPHERE, GeomType.CYLINDER,
              GeomType.CAPSULE, GeomType.MESH, GeomType.PLANE)


@wp.func
def _geom(frame: FrameMat, size: wp.vec3, span: wp.vec2i, vertices: wp.array[wp.vec3]):
    geom = Geom()
    geom.rot = wp.mat33(frame[0, 0], frame[0, 1], frame[0, 2],
                        frame[1, 0], frame[1, 1], frame[1, 2],
                        frame[2, 0], frame[2, 1], frame[2, 2])
    geom.pos = wp.vec3(frame[0, 3], frame[1, 3], frame[2, 3])
    geom.size = size
    geom.vert = vertices
    geom.vertadr = span[0]
    geom.vertnum = span[1]
    geom.graphadr = -1
    geom.index = -1
    return geom


@wp.kernel
def distances_indexed(
    offset: int, frames: wp.array2d[FrameMat], pairs: wp.array[wp.vec2i],
    rows: wp.array[int], pair_ids: wp.array[int],
    kinds: wp.array[int], sizes: wp.array[wp.vec3], spans: wp.array[wp.vec2i],
    vertices: wp.array[wp.vec3], vert: wp.array2d[wp.vec3], vert_index: wp.array2d[int],
    face: wp.array2d[int], face_pr: wp.array2d[wp.vec3], face_norm: wp.array2d[float],
    horizon: wp.array2d[int], overflow: wp.array[int], output: wp.array2d[float],
):
    slot = wp.tid()
    k = slot + offset
    sample = rows[k]
    ids = pairs[pair_ids[k]]
    a = _geom(frames[sample, ids[0]], sizes[ids[0]], spans[ids[0]], vertices)
    b = _geom(frames[sample, ids[1]], sizes[ids[1]], spans[ids[1]], vertices)
    b.pos = b.pos - a.pos
    a.pos = wp.vec3(0.0)
    ka = kinds[ids[0]]
    kb = kinds[ids[1]]
    dist = float(0.0)
    pa = wp.vec3()
    pb = wp.vec3()
    normal = wp.vec3()
    if kb == int(GeomType.PLANE.value):
        normal = b.rot @ wp.vec3(0.0, 0.0, 1.0)
        pa = support(a, ka, -normal).point
        dist = wp.dot(pa - b.pos, normal)
        pb = pa - dist * normal
    else:
        overflow[slot] = 0
        dist, count, pa, pb, _ = ccd(
            1.0e-6, 1.0e32, WARP_ITERATIONS, WARP_ITERATIONS,
            a, b, ka, kb, a.pos, b.pos, vert[slot], vert_index[slot],
            face[slot], face_pr[slot], face_norm[slot], horizon[slot],
            False, slot, overflow,
        )
        normal = wp.normalize(pa - pb) * wp.sign(dist)
        support_gap = wp.dot(normal, support(a, ka, -normal).point - support(b, kb, normal).point)
        if (count == 0 or overflow[slot] != 0 or wp.abs(dist) < 1.0e-6 or wp.length(normal) < 0.99
                or wp.abs(dist - support_gap) > 2.0e-5 or wp.abs(wp.dot(pa - pb, normal) - dist) > 2.0e-5):
            dist = wp.nan
    local_a = wp.transpose(a.rot) @ (pa - a.pos)
    local_b = wp.transpose(b.rot) @ (pb - b.pos)
    output[k, 0] = dist
    for axis in range(3):
        output[k, 1 + axis] = local_a[axis]
        output[k, 4 + axis] = local_b[axis]
        output[k, 7 + axis] = normal[axis]
