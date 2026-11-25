"""Batched convex distance and witnesses using the pinned MuJoCo Warp GJK/EPA kernels."""

import threading

import numpy as np
import warp as wp
from mujoco_warp._src.collision_core import Geom
from mujoco_warp._src.collision_gjk import ccd, support
from mujoco_warp._src.types import GeomType, MJ_MAX_EPAFACES

from ._coal import native

wp.set_module_options({"enable_backward": False})
_Frame = wp.types.matrix(shape=(3, 4), dtype=wp.float32)
_ITERATIONS = 64
_TILE = 8192
_GPU_THRESHOLD = 128
_TYPES = (None, GeomType.BOX, GeomType.SPHERE, GeomType.CYLINDER,
          GeomType.CAPSULE, GeomType.MESH, GeomType.PLANE)


@wp.func
def _geom(frame: _Frame, size: wp.vec3, span: wp.vec2i, vertices: wp.array[wp.vec3]):
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


@wp.func
def _eval_pair(slot: int, row: int, fa: _Frame, fb: _Frame, ids: wp.vec2i,
               kinds: wp.array[int], sizes: wp.array[wp.vec3], spans: wp.array[wp.vec2i],
               vertices: wp.array[wp.vec3], vert: wp.array2d[wp.vec3], vert_index: wp.array2d[int],
               face: wp.array2d[int], face_pr: wp.array2d[wp.vec3], face_norm: wp.array2d[float],
               horizon: wp.array2d[int], overflow: wp.array[int], output: wp.array2d[float]):
    a = _geom(fa, sizes[ids[0]], spans[ids[0]], vertices)
    b = _geom(fb, sizes[ids[1]], spans[ids[1]], vertices)
    # Compute in a translated world frame to retain precision far from the origin.
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
        dist, count, pa, pb, _ = ccd(1.0e-6, 1.0e32, _ITERATIONS, _ITERATIONS,
                                     a, b, ka, kb, a.pos, b.pos, vert[slot], vert_index[slot],
                                     face[slot], face_pr[slot], face_norm[slot], horizon[slot],
                                     False, slot, overflow)
        normal = wp.normalize(pa - pb) * wp.sign(dist)
        # The two tangent support planes must have the reported signed gap.
        # A flat/elongated hull can produce finite but inaccurate float32 simplex witnesses.
        support_gap = wp.dot(normal, support(a, ka, -normal).point - support(b, kb, normal).point)
        if (count == 0 or overflow[slot] != 0 or wp.abs(dist) < 1.0e-6 or wp.length(normal) < 0.99
                or wp.abs(dist - support_gap) > 2.0e-5 or wp.abs(wp.dot(pa - pb, normal) - dist) > 2.0e-5):
            dist = wp.nan
    local_a = wp.transpose(a.rot) @ (pa - a.pos)
    local_b = wp.transpose(b.rot) @ (pb - b.pos)
    output[row, 0] = dist
    for axis in range(3):
        output[row, 1 + axis] = local_a[axis]
        output[row, 4 + axis] = local_b[axis]
        output[row, 7 + axis] = normal[axis]


@wp.kernel
def _distances(offset: int, frames: wp.array2d[_Frame], pairs: wp.array[wp.vec2i],
               selected: wp.array[int], kinds: wp.array[int], sizes: wp.array[wp.vec3], spans: wp.array[wp.vec2i],
               vertices: wp.array[wp.vec3], vert: wp.array2d[wp.vec3], vert_index: wp.array2d[int],
               face: wp.array2d[int], face_pr: wp.array2d[wp.vec3], face_norm: wp.array2d[float],
               horizon: wp.array2d[int], overflow: wp.array[int], output: wp.array2d[float]):
    slot = wp.tid()
    row = slot + offset
    pair_id = row % pairs.shape[0]
    if selected.shape[0] > 0:
        pair_id = selected[row]
    ids = pairs[pair_id]
    _eval_pair(slot, row, frames[row, 0], frames[row, 1], ids,
               kinds, sizes, spans, vertices, vert, vert_index,
               face, face_pr, face_norm, horizon, overflow, output)


@wp.kernel
def _distances_indexed(offset: int, frames: wp.array2d[_Frame], pairs: wp.array[wp.vec2i],
                       rows: wp.array[int], pair_ids: wp.array[int],
                       kinds: wp.array[int], sizes: wp.array[wp.vec3], spans: wp.array[wp.vec2i],
                       vertices: wp.array[wp.vec3], vert: wp.array2d[wp.vec3], vert_index: wp.array2d[int],
                       face: wp.array2d[int], face_pr: wp.array2d[wp.vec3], face_norm: wp.array2d[float],
                       horizon: wp.array2d[int], overflow: wp.array[int], output: wp.array2d[float]):
    slot = wp.tid()
    k = slot + offset
    sample = rows[k]
    ids = pairs[pair_ids[k]]
    _eval_pair(slot, k, frames[sample, ids[0]], frames[sample, ids[1]], ids,
               kinds, sizes, spans, vertices, vert, vert_index,
               face, face_pr, face_norm, horizon, overflow, output)


class WarpPairs:
    """Immutable shared geometry with reusable GPU EPA workspace and small-batch CPU routing."""
    def __init__(self, shapes, pairs):
        self.device = wp.get_device()
        self._lock = threading.Lock()
        self.indices = wp.array(pairs, dtype=wp.vec2i, device=self.device)
        self.reference = native.Pairs([(shapes[i].geometry, shapes[j].geometry) for i, j in pairs])
        spans, sizes, vertices = [], [], []
        for shape in shapes:
            spec = shape.spec
            points = spec.get("hull_points", ())
            spans.append((len(vertices), len(points)))
            vertices.extend(points)
            if shape.kind == 1:
                sizes.append(np.asarray(spec["size"]) * 0.5)
            else:
                sizes.append((spec.get("radius", 0), 0.5 * spec.get("height", 0), 0))
        self.kinds = wp.array([int(_TYPES[s.kind]) for s in shapes], dtype=int, device=self.device)
        self.sizes = wp.array(sizes, dtype=wp.vec3, device=self.device)
        self.spans = wp.array(spans, dtype=wp.vec2i, device=self.device)
        self.vertices = wp.array(np.asarray(vertices, dtype=np.float32).reshape(-1, 3),
                                 dtype=wp.vec3, device=self.device)
        self._scratch_slots = 0
        self._scratch = None
        self._overflow = None
        self._output = None
        self._empty_sel = wp.empty(0, dtype=int, device=self.device)

    def _ensure_workspace(self, slots, count):
        if slots > self._scratch_slots:
            alloc = min(_TILE, max(slots, self._scratch_slots * 2, 256))
            nvert = 10 + 2 * _ITERATIONS
            nface = 6 + MJ_MAX_EPAFACES * _ITERATIONS
            self._scratch = [wp.empty((alloc, width), dtype=kind, device=self.device) for width, kind in
                             ((nvert, wp.vec3), (nvert, int), (nface, int), (nface, wp.vec3),
                              (nface, float), (64, int))]
            self._overflow = wp.empty(alloc, dtype=int, device=self.device)
            self._scratch_slots = alloc
        if self._output is None or self._output.shape[0] < count:
            alloc = max(count, 512 if self._output is None else self._output.shape[0] * 2)
            self._output = wp.empty((alloc, 10), dtype=float, device=self.device)

    def query(self, frames, verify, clearance, pair_indices=None):
        if verify:
            raise ValueError("original-geometry verification requires the authored collision model")
        samples, pair_count = frames.shape[:2]
        count = samples * pair_count
        if not count:
            return np.empty((samples, pair_count, 10), dtype=np.float32)
        with self._lock, wp.ScopedDevice(self.device):
            slots = min(count, _TILE)
            self._ensure_workspace(slots, count)
            poses = wp.array(np.ascontiguousarray(frames.reshape(count, 2, 3, 4), dtype=np.float32),
                             dtype=_Frame, device=self.device)
            selected = (self._empty_sel if pair_indices is None
                        else wp.array(np.asarray(pair_indices, dtype=np.int32), dtype=int, device=self.device))
            out_view = self._output[:count]
            for offset in range(0, count, slots):
                wp.launch(_distances, min(slots, count - offset),
                          inputs=[offset, poses, self.indices, selected, self.kinds, self.sizes,
                                  self.spans, self.vertices, *self._scratch, self._overflow, out_view])
            values = out_view.numpy().reshape(samples, pair_count, 10)
        # Resolve only indeterminate pairs in double precision, including exact contact.
        invalid = ~np.isfinite(values).all(axis=-1)
        rows, columns = np.nonzero(invalid)
        if len(rows):
            ids = columns if pair_indices is None else pair_indices[rows]
            packed = frames[rows, columns][:, None]
            values[rows, columns] = self.reference.query(packed, False, clearance, ids)[:, 0]
        return values

    def query_indexed(self, frames, endpoints, rows, pair_ids, verify=False):
        if verify:
            raise ValueError("original-geometry verification requires the authored collision model")
        count = len(rows)
        if not count:
            return np.empty((0, 10), dtype=np.float32)
        with self._lock, wp.ScopedDevice(self.device):
            slots = min(count, _TILE)
            self._ensure_workspace(slots, count)
            f32 = np.ascontiguousarray(frames, dtype=np.float32)
            poses = wp.array(f32, dtype=_Frame, device=self.device)
            rows_wp = wp.array(np.ascontiguousarray(rows, dtype=np.int32), dtype=int, device=self.device)
            pairs_wp = wp.array(np.ascontiguousarray(pair_ids, dtype=np.int32), dtype=int, device=self.device)
            out_view = self._output[:count]
            for offset in range(0, count, slots):
                wp.launch(_distances_indexed, min(slots, count - offset),
                          inputs=[offset, poses, self.indices, rows_wp, pairs_wp,
                                  self.kinds, self.sizes, self.spans, self.vertices,
                                  *self._scratch, self._overflow, out_view])
            values = out_view.numpy()
        invalid = ~np.isfinite(values).all(axis=-1)
        bad = np.flatnonzero(invalid)
        if len(bad):
            values[bad] = self.reference.query_indexed(frames, endpoints, rows[bad], pair_ids[bad], False)
        return values

    def first_violation(self, frames, endpoints, rows, pair_ids, margins, verify=False):
        if verify or len(rows) < _GPU_THRESHOLD:
            return self.reference.first_violation(frames, endpoints, rows, pair_ids, margins, verify)
        data = self.query_indexed(frames, endpoints, rows, pair_ids, False)
        minimum = np.maximum(0.0, margins[pair_ids] - 1e-6)
        failed = np.flatnonzero(~(np.isfinite(data[:, 0]) & (data[:, 0] >= minimum)))
        if len(failed):
            idx = int(failed[0])
            return int(rows[idx]), int(pair_ids[idx])
        return None

    def clearance_mask_indexed(self, frames, endpoints, rows, pair_ids, margins, verify=False):
        if verify or len(rows) < _GPU_THRESHOLD:
            return self.reference.clearance_mask_indexed(frames, endpoints, rows, pair_ids, margins, verify)
        data = self.query_indexed(frames, endpoints, rows, pair_ids, False)
        minimum = np.maximum(0.0, margins[pair_ids] - 1e-6)
        return np.isfinite(data[:, 0]) & (data[:, 0] >= minimum)
