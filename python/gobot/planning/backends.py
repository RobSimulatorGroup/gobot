"""Unified CollisionBackend abstraction for CPU Coal and GPU MuJoCo Warp."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Protocol

import numpy as np

from .world import PlanningWorld


@dataclass
class CollisionBatchEval:
    rows: np.ndarray            # (K,) sample indices
    pair_ids: np.ndarray        # (K,) Coal indices or -(active ESDF sphere index + 1)
    residuals: np.ndarray       # (K,) opt_margin - distance
    gradients: np.ndarray | None  # (K, J) when linearize=True
    frames: np.ndarray          # (S, N_shapes, 3, 4)
    seg_ok: np.ndarray          # (n_segs,) bool
    witnesses: np.ndarray | None  # (K, 6) world (p_a, p_b) when linearize=True


class CollisionBackend(Protocol):
    name: str
    world: PlanningWorld

    def eval_rows(
        self,
        samples: np.ndarray,
        opt_margins: np.ndarray,
        check_margins: np.ndarray,
        buffer: float,
        linearize: bool,
        group_size: int,
        segment_minimum: bool,
        snap=None,
    ) -> CollisionBatchEval: ...

    def query_indexed_convex(
        self, frames: np.ndarray, rows: np.ndarray, pair_ids: np.ndarray, snap=None,
    ) -> np.ndarray: ...

    def first_violation(
        self, frames: np.ndarray, rows: np.ndarray, pair_ids: np.ndarray,
        margins: np.ndarray, verify_original: bool, snap=None,
    ) -> tuple[int, int] | None: ...


class CoalBackend:
    """Native multi-threaded C++ Coal backend (single call for FK + AABB + GJK/EPA + Jacobian)."""

    name = "coal"

    def __init__(self, world: PlanningWorld) -> None:
        self.world = world

    def eval_rows(
        self,
        samples: np.ndarray,
        opt_margins: np.ndarray,
        check_margins: np.ndarray,
        buffer: float,
        linearize: bool,
        group_size: int,
        segment_minimum: bool,
        snap=None,
    ) -> CollisionBatchEval:
        s = self.world._native.snapshot() if snap is None else snap
        rows, pids, res, grad, frames, seg_ok, wit = s.eval_collision_rows(
            np.ascontiguousarray(samples, dtype=np.float64),
            opt_margins, check_margins, float(buffer), bool(linearize),
            int(group_size), bool(segment_minimum),
        )
        return CollisionBatchEval(rows, pids, res, grad, frames, seg_ok, wit)

    def query_indexed_convex(
        self, frames: np.ndarray, rows: np.ndarray, pair_ids: np.ndarray, snap=None,
    ) -> np.ndarray:
        s = self.world._native.snapshot() if snap is None else snap
        return s.query_distances_indexed(frames, rows, pair_ids, False)

    def first_violation(
        self, frames: np.ndarray, rows: np.ndarray, pair_ids: np.ndarray,
        margins: np.ndarray, verify_original: bool, snap=None,
    ) -> tuple[int, int] | None:
        s = self.world._native.snapshot() if snap is None else snap
        return s.first_violation_indexed(frames, rows, pair_ids, margins, verify_original)


_WARP_LOCK = threading.Lock()
_WARP_GPU_MIN_COUNT = 128


class MuJoCoWarpBackend:
    """Experimental GPU convex narrow-phase backend backed by MuJoCo Warp internal GJK/EPA kernels.

    Note: Uses private ``mujoco_warp._src`` APIs and CPU-GPU transfers per SQP step; for exact
    convex/mesh TrajOpt planning, prefer the native multi-threaded C++ ``CoalBackend``.
    """

    name = "mujoco_warp"

    def __init__(self, world: PlanningWorld) -> None:
        from . import _warp_kernels as wk

        self.world = world
        self._wp = wk.wp
        self._FrameMat = wk.FrameMat
        self._kernel = wk.distances_indexed
        self._types_map = wk.WARP_TYPES
        self._tile = wk.WARP_TILE
        self._iters = wk.WARP_ITERATIONS
        self._max_epa = wk.MJ_MAX_EPAFACES
        self._device = wk.wp.get_device()
        self._synced_topology = -1
        self._scratch_slots = 0
        self._scratch = None
        self._overflow = None
        self._output = None
        self._poses_buf = None
        self._rows_buf = None
        self._pairs_buf = None
        self._sync_geometry()

    def _sync_geometry(self, snap=None) -> None:
        s = self.world._native.snapshot() if snap is None else snap
        topo = s.topology_version
        if topo == self._synced_topology:
            return
        wp = self._wp
        kinds_raw, sizes_np, spans_np, verts_np, pairs_np = s.export_warp_geometry()
        with wp.ScopedDevice(self._device):
            self._kinds = wp.array([int(self._types_map[k]) for k in kinds_raw], dtype=int)
            self._sizes = wp.array(np.asarray(sizes_np, dtype=np.float32), dtype=wp.vec3)
            self._spans = wp.array(np.asarray(spans_np, dtype=np.int32), dtype=wp.vec2i)
            self._vertices = wp.array(np.asarray(verts_np, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3)
            self._pairs = wp.array(np.asarray(pairs_np, dtype=np.int32).reshape(-1, 2), dtype=wp.vec2i)
        self._synced_topology = topo

    def _ensure_workspace(self, slots: int, count: int) -> None:
        wp = self._wp
        if slots > self._scratch_slots:
            alloc = min(self._tile, max(slots, self._scratch_slots * 2, 256))
            nvert = 10 + 2 * self._iters
            nface = 6 + self._max_epa * self._iters
            self._scratch = [wp.empty((alloc, width), dtype=kind, device=self._device) for width, kind in
                             ((nvert, wp.vec3), (nvert, int), (nface, int), (nface, wp.vec3),
                              (nface, float), (64, int))]
            self._overflow = wp.empty(alloc, dtype=int, device=self._device)
            self._scratch_slots = alloc
        if self._output is None or self._output.shape[0] < count:
            alloc = max(count, 512 if self._output is None else self._output.shape[0] * 2)
            self._output = wp.empty((alloc, 10), dtype=float, device=self._device)

    def query_indexed_convex(
        self, frames: np.ndarray, rows: np.ndarray, pair_ids: np.ndarray, snap=None,
    ) -> np.ndarray:
        s = self.world._native.snapshot() if snap is None else snap
        count = len(rows)
        if not count:
            return np.empty((0, 10), dtype=np.float64)
        if count < _WARP_GPU_MIN_COUNT:
            return s.query_distances_indexed(frames, rows, pair_ids, False)
        if (rows.shape != (count,) or pair_ids.shape != (count,) or np.any(rows < 0)
                or np.any(rows >= len(frames)) or np.any(pair_ids < 0) or np.any(pair_ids >= s.num_pairs)):
            raise ValueError("sample or Coal pair index out of range")
        wp = self._wp
        with _WARP_LOCK:
            self._sync_geometry(s)
            with wp.ScopedDevice(self._device):
                slots = min(count, self._tile)
                self._ensure_workspace(slots, count)
                frames_f32 = np.ascontiguousarray(frames, dtype=np.float32)
                rows_i32 = np.ascontiguousarray(rows, dtype=np.int32)
                pairs_i32 = np.ascontiguousarray(pair_ids, dtype=np.int32)
                n_s, n_sh = frames_f32.shape[0], frames_f32.shape[1]
                if self._poses_buf is None or self._poses_buf.shape[0] < n_s or self._poses_buf.shape[1] != n_sh:
                    alloc_s = max(n_s, 64 if self._poses_buf is None else self._poses_buf.shape[0] * 2)
                    self._poses_buf = wp.empty((alloc_s, n_sh), dtype=self._FrameMat, device=self._device)
                if self._rows_buf is None or self._rows_buf.shape[0] < count:
                    alloc_c = max(count, 512 if self._rows_buf is None else self._rows_buf.shape[0] * 2)
                    self._rows_buf = wp.empty(alloc_c, dtype=int, device=self._device)
                    self._pairs_buf = wp.empty(alloc_c, dtype=int, device=self._device)
                poses = self._poses_buf[:n_s]
                rows_wp = self._rows_buf[:count]
                pairs_wp = self._pairs_buf[:count]
                wp.copy(poses, wp.from_numpy(frames_f32, dtype=self._FrameMat))
                wp.copy(rows_wp, wp.from_numpy(rows_i32, dtype=int))
                wp.copy(pairs_wp, wp.from_numpy(pairs_i32, dtype=int))
                out_view = self._output[:count]
                for offset in range(0, count, slots):
                    wp.launch(
                        self._kernel, min(slots, count - offset),
                        inputs=[offset, poses, self._pairs, rows_wp, pairs_wp,
                                self._kinds, self._sizes, self._spans, self._vertices,
                                *self._scratch, self._overflow, out_view],
                    )
                values = out_view.numpy().astype(np.float64, copy=False)
        invalid = ~np.isfinite(values).all(axis=-1)
        bad = np.flatnonzero(invalid)
        if len(bad):
            values[bad] = s.query_distances_indexed(frames, rows[bad], pair_ids[bad], False)
        return values

    def first_violation(
        self, frames: np.ndarray, rows: np.ndarray, pair_ids: np.ndarray,
        margins: np.ndarray, verify_original: bool, snap=None,
    ) -> tuple[int, int] | None:
        s = self.world._native.snapshot() if snap is None else snap
        if verify_original or len(rows) < _WARP_GPU_MIN_COUNT:
            return s.first_violation_indexed(frames, rows, pair_ids, margins, verify_original)
        data = self.query_indexed_convex(frames, rows, pair_ids, snap=s)
        minimum = np.maximum(0.0, margins[pair_ids] - 1e-6)
        failed = np.flatnonzero(~(np.isfinite(data[:, 0]) & (data[:, 0] >= minimum)))
        if len(failed):
            idx = int(failed[0])
            return int(rows[idx]), int(pair_ids[idx])
        return None

    def eval_rows(
        self,
        samples: np.ndarray,
        opt_margins: np.ndarray,
        check_margins: np.ndarray,
        buffer: float,
        linearize: bool,
        group_size: int,
        segment_minimum: bool,
        snap=None,
    ) -> CollisionBatchEval:
        s = self.world._native.snapshot() if snap is None else snap
        if s.has_esdf:
            return CoalBackend(self.world).eval_rows(samples, opt_margins, check_margins, buffer,
                                                     linearize, group_size, segment_minimum, snap=s)
        out = s.compute_frames(np.ascontiguousarray(samples, dtype=np.float64), linearize)
        if linearize:
            frames, jorig, jaxis = out
        else:
            (frames,) = out
            jorig = jaxis = None
        n_segs = (len(samples) + group_size - 1) // group_size if group_size > 0 else len(samples)
        seg_ok = np.ones(n_segs, dtype=np.bool_)
        if s.num_pairs == 0:
            empty_i = np.empty(0, dtype=np.int64)
            empty_d = np.empty(0, dtype=np.float64)
            empty_g = np.empty((0, s.num_joints), dtype=np.float64) if linearize else None
            empty_w = np.empty((0, 6), dtype=np.float64) if linearize else None
            return CollisionBatchEval(empty_i, empty_i, empty_d, empty_g, frames, seg_ok, empty_w)

        c_rows, c_pairs = s.screen_aabbs(frames, opt_margins + buffer, False, group_size)
        if not len(c_rows):
            empty_i = np.empty(0, dtype=np.int64)
            empty_d = np.empty(0, dtype=np.float64)
            empty_g = np.empty((0, s.num_joints), dtype=np.float64) if linearize else None
            empty_w = np.empty((0, 6), dtype=np.float64) if linearize else None
            return CollisionBatchEval(empty_i, empty_i, empty_d, empty_g, frames, seg_ok, empty_w)

        data = self.query_indexed_convex(frames, c_rows, c_pairs, snap=s)
        req_min = np.maximum(0.0, check_margins[c_pairs] - 1e-6)
        failed = ~(np.isfinite(data[:, 0]) & (data[:, 0] >= req_min))
        if failed.any() and group_size > 0:
            seg_ok[c_rows[failed] // group_size] = False
        residual = opt_margins[c_pairs] - data[:, 0]
        near = np.flatnonzero(residual > -buffer)
        if segment_minimum and len(near) and group_size > 0:
            group = (c_rows[near] // group_size) * len(opt_margins) + c_pairs[near]
            order = np.lexsort((-residual[near], group))
            first = np.r_[True, np.diff(group[order]) != 0]
            near = near[np.sort(order[first])]
        rows_n, pairs_n, res_n = c_rows[near], c_pairs[near], residual[near]
        grad = wit = None
        if linearize:
            if len(rows_n):
                grad, wit = s.compute_analytic_gradients(frames, jorig, jaxis, rows_n, pairs_n, data[near])
            else:
                grad = np.empty((0, s.num_joints), dtype=np.float64)
                wit = np.empty((0, 6), dtype=np.float64)
        return CollisionBatchEval(rows_n, pairs_n, res_n, grad, frames, seg_ok, wit)


def make_backend(world: PlanningWorld, backend: str = "coal") -> CollisionBackend:
    if backend == "coal":
        return CoalBackend(world)
    if backend == "mujoco_warp":
        return MuJoCoWarpBackend(world)
    raise ValueError("collision backend must be 'coal' or 'mujoco_warp'")
