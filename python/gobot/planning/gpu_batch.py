"""GPU-resident batched multi-seed projected L-BFGS trajectory optimizer & backend.

Keeps FK, ESDF / static analytic primitive / self-collision evaluation, pose constraints,
projected L-BFGS with Armijo backtracking, and top-M candidate selection resident on GPU
with zero per-iteration host syncs or PCIe round-trips. Final verification and Hermite
retiming always run in C++ TBB via ``MotionPlannerNative.finalize_candidates_batch`` on
the exact same pinned ``WorldSnapshot``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import threading
from time import perf_counter

import numpy as np

from .types import MotionPlanRequest, PlanDiagnostics, PlanResult, PoseConstraint
from .world import PlanningWorld


@dataclass(frozen=True)
class GpuBatchConfig:
    """Hyperparameters and safety/memory controls for the GPU-resident batch optimizer."""

    num_seeds: int = 4
    top_k_verify: int = 4
    lbfgs_iters_per_stage: tuple[int, ...] = (18, 18, 18)
    penalty_weights: tuple[float, ...] = (200.0, 2000.0, 15000.0)
    lbfgs_history: int = 6
    line_search_iters: int = 3
    armijo_c1: float = 1e-4
    line_search_shrink: float = 0.35
    max_step: float = 0.45
    w_smooth_vel: float = 0.5
    w_smooth_acc: float = 2.0
    w_vel_limit: float = 200.0
    w_joint_target: float = 2000.0
    w_pose_constraint: float = 500.0
    world_margin_cushion: float = 0.006
    self_margin_cushion: float = 0.0
    huber_delta: float = 0.15
    max_gpu_substeps: int = 6
    seed_noise_std: float = 0.40
    use_cuda_graph: bool = True
    cpu_fallback_on_fail: bool = True
    vram_budget_mb: int = 2048
    device: str | None = None
    rng_seed: int = 0

    def __post_init__(self) -> None:
        if self.num_seeds < 1:
            raise ValueError("num_seeds must be >= 1")
        if self.top_k_verify < 1:
            raise ValueError("top_k_verify must be >= 1")
        if len(self.lbfgs_iters_per_stage) == 0 or len(self.lbfgs_iters_per_stage) != len(self.penalty_weights):
            raise ValueError("lbfgs_iters_per_stage and penalty_weights must be non-empty and equal length")
        if any(it < 1 for it in self.lbfgs_iters_per_stage):
            raise ValueError("lbfgs_iters_per_stage entries must be >= 1")
        if self.lbfgs_history < 1:
            raise ValueError("lbfgs_history must be >= 1")
        if self.line_search_iters < 1:
            raise ValueError("line_search_iters must be >= 1")
        if self.vram_budget_mb < 64:
            raise ValueError("vram_budget_mb must be >= 64")
        if self.max_gpu_substeps < 1:
            raise ValueError("max_gpu_substeps must be >= 1")


@dataclass(frozen=True)
class GpuBatchStats:
    """Per-call timing and pass-rate diagnostics for GPU batch optimization."""

    num_requests: int = 0
    num_gpu_requests: int = 0
    num_direct_cpu_requests: int = 0
    top1_passed: int = 0
    topm_passed: int = 0
    cpu_fallback_count: int = 0
    failed_count: int = 0
    upload_ms: float = 0.0
    ik_ms: float = 0.0
    seed_ms: float = 0.0
    gpu_opt_ms: float = 0.0
    graph_capture_ms: float = 0.0
    verify_ms: float = 0.0
    fallback_ms: float = 0.0
    total_ms: float = 0.0
    graph_captured: bool = False
    graph_reused: bool = False


class _GpuSceneCache:
    """Grow-only GPU scene buffer cache keyed by immutable WorldSnapshot.version.

    Keeps device pointer addresses stable whenever updated scene data fits within existing
    buffer capacities, while exposing the exact pointer tuple so CUDA Graph caches automatically
    invalidate and recapture on any buffer reallocation.
    """

    def __init__(self, wp, device) -> None:
        self._wp = wp
        self._device = device
        self.synced_version: int = -1
        self.num_bodies: int = 0
        self.num_joints: int = 0
        self.num_branches: int = 0
        self.num_spheres: int = 0
        self.num_self_pairs: int = 0
        self.num_static: int = 0
        self.num_static_pairs: int = 0
        self.has_esdf: bool = False
        self.esdf_shape: tuple[int, int, int] = (1, 1, 1)

        with wp.ScopedDevice(device):
            self.scene_i32 = wp.zeros(8, dtype=wp.int32, device=device)
            self.scene_f32 = wp.zeros(7, dtype=wp.float32, device=device)

        # Grow-only buffers allocated on first sync.
        self.parent = None
        self.kinds = None
        self.joint_idx = None
        self.body_branch = None
        self.a_rot = None
        self.a_pos = None
        self.b_rot = None
        self.b_pos = None
        self.axes = None
        self.root_pos = None
        self.root_rot = None
        self.body_ancestor = None
        self.joint_prismatic = None
        self.lower = None
        self.upper = None
        self.velocity_limit = None

        self.sphere_body = None
        self.sphere_center = None
        self.sphere_radius = None
        self.sphere_esdf_active = None
        self.static_csr_off = None
        self.static_csr_idx = None
        self.self_csr_off = None
        self.self_csr_idx = None

        self.static_kind = None
        self.static_pos = None
        self.static_rot = None
        self.static_size = None

        self.esdf_sdf = None

    def _grow_copy_1d(self, attr: str, data_np: np.ndarray, dtype, min_cap: int = 1) -> None:
        wp = self._wp
        n = int(data_np.shape[0])
        buf = getattr(self, attr)
        need = max(min_cap, n)
        if buf is None or buf.shape[0] < need:
            cap = max(need, 16 if buf is None else buf.shape[0] * 2)
            buf = wp.zeros(cap, dtype=dtype, device=self._device)
            setattr(self, attr, buf)
        if n > 0:
            wp.copy(buf[:n], wp.from_numpy(np.ascontiguousarray(data_np), dtype=dtype))

    def _grow_copy_2d(self, attr: str, data_np: np.ndarray, dtype) -> None:
        wp = self._wp
        r, c = int(data_np.shape[0]), int(data_np.shape[1])
        buf = getattr(self, attr)
        if buf is None or buf.shape[0] < max(1, r) or buf.shape[1] != c:
            buf = wp.zeros((max(1, r), max(1, c)), dtype=dtype, device=self._device)
            setattr(self, attr, buf)
        if r > 0 and c > 0:
            wp.copy(buf[:r], wp.from_numpy(np.ascontiguousarray(data_np), dtype=dtype))

    def sync(self, snap) -> float:
        ver = int(snap.version)
        if ver == self.synced_version:
            return 0.0
        t0 = perf_counter()
        exp = snap.export_gpu_scene()
        wp = self._wp

        self.num_bodies = int(exp["num_bodies"])
        self.num_joints = int(exp["num_joints"])
        self.num_branches = int(exp["num_branches"])
        self.num_spheres = int(exp["num_spheres"])
        self.num_self_pairs = int(exp["self_sphere_pairs"].shape[0])
        self.num_static = int(exp["num_static"])
        self.num_static_pairs = int(exp["static_sphere_pairs"].shape[0])
        self.has_esdf = bool(exp["has_esdf"])

        esdf_arr = np.ascontiguousarray(exp["esdf_sdf"], dtype=np.float32)
        self.esdf_shape = (int(esdf_arr.shape[0]), int(esdf_arr.shape[1]), int(esdf_arr.shape[2]))
        esdf_flat = esdf_arr.reshape(-1)

        # Infinite bounds leave unbounded joints free during projected line search.
        lower_f = np.asarray(exp["lower"], dtype=np.float32)
        upper_f = np.asarray(exp["upper"], dtype=np.float32)
        vel_f = np.asarray(exp["velocity_limit"], dtype=np.float32)
        vel_f = np.where(np.isfinite(vel_f), vel_f, np.float32(1.0e20))

        with wp.ScopedDevice(self._device):
            self._grow_copy_1d("parent", exp["parent"], wp.int32)
            self._grow_copy_1d("kinds", exp["kinds"], wp.int32)
            self._grow_copy_1d("joint_idx", exp["joint_idx"], wp.int32)
            self._grow_copy_1d("body_branch", exp["body_branch"], wp.int32)
            self._grow_copy_1d("a_rot", exp["a_rot"], wp.mat33)
            self._grow_copy_1d("a_pos", exp["a_pos"], wp.vec3)
            self._grow_copy_1d("b_rot", exp["b_rot"], wp.mat33)
            self._grow_copy_1d("b_pos", exp["b_pos"], wp.vec3)
            self._grow_copy_1d("axes", exp["axes"], wp.vec3)
            self._grow_copy_1d("root_pos", exp["root_pos"], wp.vec3)
            self._grow_copy_1d("root_rot", exp["root_rot"], wp.mat33)
            self._grow_copy_2d("body_ancestor", exp["body_ancestor"], wp.uint8)
            self._grow_copy_1d("joint_prismatic", exp["joint_prismatic"], wp.uint8)
            self._grow_copy_1d("lower", lower_f, wp.float32)
            self._grow_copy_1d("upper", upper_f, wp.float32)
            self._grow_copy_1d("velocity_limit", vel_f, wp.float32)

            self._grow_copy_1d("sphere_body", exp["sphere_body"], wp.int32)
            self._grow_copy_1d("sphere_center", exp["sphere_center"], wp.vec3)
            self._grow_copy_1d("sphere_radius", exp["sphere_radius"], wp.float32)
            self._grow_copy_1d("sphere_esdf_active", exp["sphere_esdf_active"], wp.uint8)

            # Build per-sphere CSR adjacency lists for static obstacles and symmetric self-collision pairs.
            ns = self.num_spheres
            stat_pairs = np.asarray(exp["static_sphere_pairs"], dtype=np.int32).reshape(-1, 2)
            stat_adj: list[list[int]] = [[] for _ in range(ns)]
            for sa, sb in stat_pairs:
                stat_adj[int(sa)].append(int(sb))
            stat_off = np.zeros(ns + 1, dtype=np.int32)
            stat_flat: list[int] = []
            for s_i in range(ns):
                stat_off[s_i] = len(stat_flat)
                stat_flat.extend(stat_adj[s_i])
            stat_off[ns] = len(stat_flat)

            self_pairs = np.asarray(exp["self_sphere_pairs"], dtype=np.int32).reshape(-1, 2)
            self_adj: list[list[int]] = [[] for _ in range(ns)]
            for sa, sb in self_pairs:
                self_adj[int(sa)].append(int(sb))
                self_adj[int(sb)].append(int(sa))
            self_off = np.zeros(ns + 1, dtype=np.int32)
            self_flat: list[int] = []
            for s_i in range(ns):
                self_off[s_i] = len(self_flat)
                self_flat.extend(self_adj[s_i])
            self_off[ns] = len(self_flat)

            self._grow_copy_1d("static_csr_off", stat_off, wp.int32, min_cap=2)
            self._grow_copy_1d("static_csr_idx", np.asarray(stat_flat, dtype=np.int32), wp.int32, min_cap=1)
            self._grow_copy_1d("self_csr_off", self_off, wp.int32, min_cap=2)
            self._grow_copy_1d("self_csr_idx", np.asarray(self_flat, dtype=np.int32), wp.int32, min_cap=1)

            self._grow_copy_1d("static_kind", exp["static_kind"], wp.int32)
            self._grow_copy_1d("static_pos", exp["static_pos"], wp.vec3)
            self._grow_copy_1d("static_rot", exp["static_rot"], wp.mat33)
            self._grow_copy_1d("static_size", exp["static_size"], wp.vec3)

            self._grow_copy_1d("esdf_sdf", esdf_flat, wp.float32)

            meta_i = np.array([
                1 if self.has_esdf else 0,
                self.esdf_shape[0], self.esdf_shape[1], self.esdf_shape[2],
                int(exp["esdf_from_occupancy"]),
                self.num_spheres,
                self.num_self_pairs,
                self.num_static_pairs,
            ], dtype=np.int32)
            org = np.asarray(exp["esdf_origin"], dtype=np.float32)
            meta_f = np.array([
                org[0], org[1], org[2],
                float(exp["esdf_resolution"]),
                float(exp["esdf_inflation_radius"]),
                float(exp["esdf_max_distance"]),
                float(exp["esdf_lipschitz_bound"]),
            ], dtype=np.float32)
            wp.copy(self.scene_i32, wp.from_numpy(meta_i, dtype=wp.int32))
            wp.copy(self.scene_f32, wp.from_numpy(meta_f, dtype=wp.float32))

        self.synced_version = ver
        return (perf_counter() - t0) * 1000.0

    def ptrs_key(self) -> tuple[int, ...]:
        """Exact device-address signature of every buffer read by captured CUDA Graphs."""
        return (
            self.num_bodies, self.num_joints,
            self.scene_i32.ptr, self.scene_f32.ptr,
            self.parent.ptr, self.kinds.ptr, self.joint_idx.ptr, self.body_branch.ptr,
            self.a_rot.ptr, self.a_pos.ptr, self.b_rot.ptr, self.b_pos.ptr,
            self.axes.ptr, self.root_pos.ptr, self.root_rot.ptr,
            self.body_ancestor.ptr, self.joint_prismatic.ptr,
            self.lower.ptr, self.upper.ptr, self.velocity_limit.ptr,
            self.sphere_body.ptr, self.sphere_center.ptr, self.sphere_radius.ptr,
            self.sphere_esdf_active.ptr,
            self.sphere_body.shape[0],
            self.static_csr_off.ptr, self.static_csr_idx.ptr,
            self.self_csr_off.ptr, self.self_csr_idx.ptr,
            self.static_kind.ptr, self.static_pos.ptr, self.static_rot.ptr,
            self.static_size.ptr,
            self.esdf_sdf.ptr,
        )


class _GpuWorkspace:
    """Preallocated GPU buffers for one (B_chunk, K, T, J, substeps, M_hist, top_m, num_stages) shape."""

    def __init__(self, wp, device, *, B_chunk: int, K: int, T: int, J: int, num_bodies: int,
                 substeps: int, M_hist: int, top_m: int, num_stages: int, max_pose: int) -> None:
        self._wp = wp
        self._device = device
        self.B_chunk = B_chunk
        self.K = K
        self.N = B_chunk * K
        self.T = T
        self.J = J
        self.D = T * J
        self.M_samples = (T - 1) * substeps
        self.substeps = substeps
        self.M_hist = M_hist
        self.top_m = min(top_m, K)
        self.num_stages = num_stages
        self.max_pose = max(1, max_pose)

        N = self.N
        S = N * self.M_samples

        with wp.ScopedDevice(device):
            self.x = wp.zeros((N, T, J), dtype=wp.float32, device=device)
            self.g = wp.zeros((N, T, J), dtype=wp.float32, device=device)
            self.f = wp.zeros(N, dtype=wp.float32, device=device)
            self.cost_col = wp.zeros(N, dtype=wp.float32, device=device)
            self.cost_pose = wp.zeros(N, dtype=wp.float32, device=device)
            self.cost_smooth = wp.zeros(N, dtype=wp.float32, device=device)
            self.max_col_viol = wp.zeros(N, dtype=wp.float32, device=device)
            self.max_other_viol = wp.zeros(N, dtype=wp.float32, device=device)

            self.x_trial = wp.zeros((N, T, J), dtype=wp.float32, device=device)
            self.g_trial = wp.zeros((N, T, J), dtype=wp.float32, device=device)
            self.f_trial = wp.zeros(N, dtype=wp.float32, device=device)
            self.cost_col_trial = wp.zeros(N, dtype=wp.float32, device=device)
            self.cost_pose_trial = wp.zeros(N, dtype=wp.float32, device=device)
            self.cost_smooth_trial = wp.zeros(N, dtype=wp.float32, device=device)
            self.max_col_viol_trial = wp.zeros(N, dtype=wp.float32, device=device)
            self.max_other_viol_trial = wp.zeros(N, dtype=wp.float32, device=device)

            self.d = wp.zeros((N, T, J), dtype=wp.float32, device=device)
            self.s_hist = wp.zeros((N, M_hist, self.D), dtype=wp.float32, device=device)
            self.y_hist = wp.zeros((N, M_hist, self.D), dtype=wp.float32, device=device)
            self.rho_hist = wp.zeros((N, M_hist), dtype=wp.float32, device=device)
            self.alpha_buf = wp.zeros((N, M_hist), dtype=wp.float32, device=device)
            self.hist_head = wp.zeros(N, dtype=wp.int32, device=device)
            self.hist_count = wp.zeros(N, dtype=wp.int32, device=device)
            self.step_alpha = wp.zeros(N, dtype=wp.float32, device=device)
            self.accepted = wp.zeros(N, dtype=wp.int32, device=device)

            # FK buffers for all N * M_samples states.
            self.bpos = wp.zeros((S, num_bodies), dtype=wp.vec3, device=device)
            self.brot = wp.zeros((S, num_bodies), dtype=wp.mat33, device=device)
            self.jorig = wp.zeros((S, J), dtype=wp.vec3, device=device)
            self.jaxis = wp.zeros((S, J), dtype=wp.vec3, device=device)

            # Per-request metadata buffers.
            self.active_mask = wp.zeros((B_chunk, J), dtype=wp.uint8, device=device)
            self.jt_has = wp.zeros((B_chunk, T), dtype=wp.uint8, device=device)
            self.jt_vals = wp.zeros((B_chunk, T, J), dtype=wp.float32, device=device)
            self.has_start_vel = wp.zeros(B_chunk, dtype=wp.uint8, device=device)
            self.start_vel = wp.zeros((B_chunk, J), dtype=wp.float32, device=device)
            self.seed_goal_idx = wp.zeros(N, dtype=wp.int32, device=device)

            # Per-stage optimization parameter vectors (14 floats each).
            self.opt_stages = [wp.zeros(14, dtype=wp.float32, device=device) for _ in range(num_stages)]

            # Pose constraint compact buffers.
            mp = self.max_pose
            self.pc_req = wp.zeros(mp, dtype=wp.int32, device=device)
            self.pc_t = wp.zeros(mp, dtype=wp.int32, device=device)
            self.pc_link = wp.zeros(mp, dtype=wp.int32, device=device)
            self.pc_flags = wp.zeros(mp, dtype=wp.int32, device=device)
            self.pc_pos = wp.zeros(mp, dtype=wp.vec3, device=device)
            self.pc_rot = wp.zeros(mp, dtype=wp.mat33, device=device)
            self.pc_pos_tol = wp.zeros(mp, dtype=wp.vec3, device=device)
            self.pc_rot_tol = wp.zeros(mp, dtype=wp.vec3, device=device)
            self.pc_weight = wp.zeros(mp, dtype=wp.float32, device=device)

            # Top-M output selection buffers.
            self.used_mask = wp.zeros((B_chunk, K), dtype=wp.uint8, device=device)
            self.top_x = wp.zeros((B_chunk, self.top_m, T, J), dtype=wp.float32, device=device)
            self.top_goal = wp.zeros((B_chunk, self.top_m), dtype=wp.int32, device=device)
            self.top_seed = wp.zeros((B_chunk, self.top_m), dtype=wp.int32, device=device)
            self.top_viol = wp.zeros((B_chunk, self.top_m), dtype=wp.float32, device=device)
            self.top_smooth = wp.zeros((B_chunk, self.top_m), dtype=wp.float32, device=device)

    def ptrs_key(self) -> tuple[int, ...]:
        return (
            self.x.ptr, self.g.ptr, self.f.ptr,
            self.x_trial.ptr, self.g_trial.ptr, self.f_trial.ptr,
            self.bpos.ptr, self.brot.ptr, self.jorig.ptr, self.jaxis.ptr,
            self.active_mask.ptr, self.jt_has.ptr, self.jt_vals.ptr,
            self.pc_req.ptr, self.top_x.ptr,
            *(s.ptr for s in self.opt_stages),
        )


def _estimate_bytes_per_traj(T: int, J: int, num_bodies: int, substeps: int, M_hist: int) -> int:
    M_samples = (T - 1) * substeps
    D = T * J
    fk_bytes = M_samples * (num_bodies * (12 + 36) + J * (12 + 12))
    lbfgs_bytes = (5 * D + 2 * M_hist * D + 4 * M_hist + 32) * 4
    return int((fk_bytes + lbfgs_bytes) * 1.25)


def _build_piecewise_seeds(
    start_q: np.ndarray,
    goal_candidates: list[tuple[int, np.ndarray]],
    waypoints: int,
    fixed_start: int,
    fixed_end: int,
    joint_targets: dict[int, np.ndarray],
    active_mask: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    num_seeds: int,
    noise_std: float,
    user_seed: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate K smooth seeds [K, T, J] (float32) and goal_indices [K] (int32) via vectorized NumPy."""
    J = len(start_q)
    lo = np.where(np.isfinite(lower), lower, start_q - np.pi)
    hi = np.where(np.isfinite(upper), upper, start_q + np.pi)
    interior_anchors = sorted(
        (k, v) for k, v in joint_targets.items() if fixed_start <= k < waypoints - fixed_end
    )

    # Precompute piecewise linear templates for each unique goal candidate once.
    n_goals = len(goal_candidates)
    base_per_goal = np.empty((n_goals, waypoints, J), dtype=np.float64)
    for gi, (_, gq) in enumerate(goal_candidates):
        anchors = [(fixed_start - 1, start_q), *interior_anchors, (waypoints - fixed_end, gq)]
        out = base_per_goal[gi]
        out[:fixed_start] = start_q
        out[waypoints - fixed_end:] = gq
        for (i0, q0), (i1, q1) in zip(anchors, anchors[1:]):
            if i1 <= i0:
                continue
            alpha = np.linspace(0.0, 1.0, i1 - i0 + 1, dtype=np.float64)[:, None]
            out[i0:i1 + 1] = (1.0 - alpha) * q0[None, :] + alpha * q1[None, :]
        if not active_mask.all():
            out[:, ~active_mask] = start_q[~active_mask]

    goal_cycle = np.arange(num_seeds, dtype=np.int32) % n_goals
    goal_ids = np.asarray([int(goal_candidates[gi][0]) for gi in goal_cycle], dtype=np.int32)
    seeds64 = base_per_goal[goal_cycle].copy()

    if user_seed is not None:
        q = np.asarray(user_seed, dtype=np.float64).copy()
        q[:fixed_start] = start_q
        q[waypoints - fixed_end:] = goal_candidates[0][1]
        for idx, tgt in interior_anchors:
            q[idx] = tgt
        if not active_mask.all():
            q[:, ~active_mask] = start_q[~active_mask]
        seeds64[0] = q

    n_pert = num_seeds - n_goals
    if n_pert > 0:
        anchor_indices = [fixed_start - 1, *(k for k, _ in interior_anchors), waypoints - fixed_end]
        basis_1 = np.zeros(waypoints, dtype=np.float64)
        basis_2 = np.zeros(waypoints, dtype=np.float64)
        for i0, i1 in zip(anchor_indices, anchor_indices[1:]):
            if i1 - i0 <= 1:
                continue
            s = np.linspace(0.0, 1.0, i1 - i0 + 1, dtype=np.float64)
            basis_1[i0:i1 + 1] = np.sin(np.pi * s)
            basis_2[i0:i1 + 1] = np.sin(2.0 * np.pi * s)
        basis_1[:fixed_start] = 0.0
        basis_1[waypoints - fixed_end:] = 0.0
        basis_2[:fixed_start] = 0.0
        basis_2[waypoints - fixed_end:] = 0.0
        for k, _ in interior_anchors:
            basis_1[k] = 0.0
            basis_2[k] = 0.0

        span = np.minimum(hi - lo, 2.0 * np.pi)
        amp_scale = noise_std * np.minimum(1.0, span / 2.0)
        c1 = rng.normal(0.0, 1.0, size=(n_pert, J)) * amp_scale[None, :]
        c2 = 0.45 * rng.normal(0.0, 1.0, size=(n_pert, J)) * amp_scale[None, :]
        if n_pert >= 2:
            # Make first two perturbed seeds antipodal so both sides of a symmetric obstacle are explored.
            c1[1] = -c1[0]
            c2[1] = -c2[0]
        if not active_mask.all():
            c1[:, ~active_mask] = 0.0
            c2[:, ~active_mask] = 0.0
        seeds64[n_goals:] += basis_1[None, :, None] * c1[:, None, :] + basis_2[None, :, None] * c2[:, None, :]

    np.clip(seeds64, lower, upper, out=seeds64)
    return seeds64.astype(np.float32), goal_ids


def _is_pose_sequence(obj) -> bool:
    return (
        isinstance(obj, Sequence)
        and not isinstance(obj, (str, bytes, np.ndarray))
        and len(obj) > 0
        and isinstance(obj[0], PoseConstraint)
    )


class GpuBatchOptimizer:
    """GPU-resident multi-seed projected L-BFGS trajectory optimizer over a PlanningWorld."""

    def __init__(self, world: PlanningWorld, config: GpuBatchConfig | None = None) -> None:
        import warp as wp
        from . import _gpu_kernels as gk

        wp.init()
        cuda_devs = wp.get_cuda_devices()
        if not cuda_devs:
            raise RuntimeError("GpuBatchOptimizer requires a CUDA-capable GPU, but Warp found no CUDA devices")
        self._wp = wp
        self._gk = gk
        self.world = world
        self.config = GpuBatchConfig() if config is None else config
        self._device = wp.get_device(self.config.device) if self.config.device is not None else cuda_devs[0]
        if not self._device.is_cuda:
            raise RuntimeError(f"GpuBatchOptimizer requires a CUDA device, got {self._device}")
        self._lock = threading.Lock()
        self._scene_cache = _GpuSceneCache(wp, self._device)
        self._workspaces: dict[tuple, _GpuWorkspace] = {}
        self._graphs: dict[tuple, object] = {}

    @property
    def device(self) -> str:
        return str(self._device)

    def clear_caches(self) -> None:
        with self._lock:
            self._graphs.clear()
            self._workspaces.clear()

    def eval_fk_and_esdf(self, q: np.ndarray, snap=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate GPU FK body poses and GPU ESDF sphere clearances + joint gradients (for tests/diagnostics)."""
        wp = self._wp
        gk = self._gk
        s = self.world._native.snapshot() if snap is None else snap
        q_f32 = np.ascontiguousarray(np.asarray(q, dtype=np.float32))
        if q_f32.ndim != 2 or q_f32.shape[1] != s.num_joints:
            raise ValueError(f"q must have shape [S, {s.num_joints}]")
        S = int(q_f32.shape[0])
        with self._lock:
            self._scene_cache.sync(s)
            sc = self._scene_cache
            with wp.ScopedDevice(self._device):
                q_wp = wp.array(q_f32, dtype=wp.float32, device=self._device)
                bpos = wp.zeros((S, sc.num_bodies), dtype=wp.vec3, device=self._device)
                brot = wp.zeros((S, sc.num_bodies), dtype=wp.mat33, device=self._device)
                jorig = wp.zeros((S, sc.num_joints), dtype=wp.vec3, device=self._device)
                jaxis = wp.zeros((S, sc.num_joints), dtype=wp.vec3, device=self._device)
                wp.launch(
                    gk.fk_configs_kernel, dim=S,
                    inputs=[
                        q_wp, sc.num_bodies, sc.num_joints,
                        sc.parent, sc.kinds, sc.joint_idx, sc.body_branch,
                        sc.a_rot, sc.a_pos, sc.b_rot, sc.b_pos, sc.axes,
                        sc.root_pos, sc.root_rot,
                        bpos, brot, jorig, jaxis,
                    ],
                )
                NS = max(1, sc.num_spheres)
                out_d = wp.zeros((S, NS), dtype=wp.float32, device=self._device)
                out_g = wp.zeros((S, NS, sc.num_joints), dtype=wp.float32, device=self._device)
                if sc.num_spheres > 0:
                    wp.launch(
                        gk.query_esdf_spheres_kernel, dim=S * sc.num_spheres,
                        inputs=[
                            bpos, brot, jorig, jaxis,
                            sc.num_spheres, sc.num_joints,
                            sc.sphere_body, sc.sphere_center, sc.sphere_radius, sc.sphere_esdf_active,
                            sc.body_ancestor, sc.joint_prismatic,
                            sc.esdf_sdf, sc.scene_i32, sc.scene_f32,
                            out_d, out_g,
                        ],
                    )
                wp.synchronize_device(self._device)
                return (
                    bpos.numpy(),
                    brot.numpy(),
                    out_d.numpy()[:, :sc.num_spheres],
                    out_g.numpy()[:, :sc.num_spheres, :],
                )

    def _get_workspace(
        self, *, B_chunk: int, K: int, T: int, J: int, num_bodies: int,
        substeps: int, M_hist: int, top_m: int, num_stages: int, max_pose: int,
    ) -> _GpuWorkspace:
        # Bucket max_pose to powers of two to avoid workspace churn on varying pose constraint counts.
        pose_cap = 1
        while pose_cap < max_pose:
            pose_cap *= 2
        key = (B_chunk, K, T, J, num_bodies, substeps, M_hist, min(top_m, K), num_stages, pose_cap)
        ws = self._workspaces.get(key)
        if ws is None:
            # Cap cached workspaces so memory doesn't grow unbounded across many distinct shapes.
            if len(self._workspaces) >= 8:
                self._graphs.clear()
                self._workspaces.clear()
            ws = _GpuWorkspace(
                self._wp, self._device,
                B_chunk=B_chunk, K=K, T=T, J=J, num_bodies=num_bodies,
                substeps=substeps, M_hist=M_hist, top_m=top_m,
                num_stages=num_stages, max_pose=pose_cap,
            )
            self._workspaces[key] = ws
        return ws

    def _launch_eval(
        self, ws: _GpuWorkspace, sc: _GpuSceneCache, *,
        q_buf, grad_buf, f_buf, cost_col_buf, cost_pose_buf, cost_smooth_buf,
        max_col_buf, max_other_buf, opt_stage_buf,
        fixed_start: int, fixed_end: int, skip_accepted: int, num_pose_items: int,
    ) -> None:
        wp = self._wp
        gk = self._gk
        N = ws.N
        S = N * ws.M_samples

        wp.launch(
            gk.zero_eval_buffers_kernel, dim=N,
            inputs=[
                N, ws.T, ws.J, skip_accepted, ws.accepted,
                cost_col_buf, cost_pose_buf, cost_smooth_buf, f_buf,
                max_col_buf, max_other_buf, grad_buf,
            ],
        )
        wp.launch(
            gk.fk_traj_samples_kernel, dim=S,
            inputs=[
                q_buf, ws.M_samples, ws.substeps, sc.num_bodies, ws.J,
                skip_accepted, ws.accepted,
                sc.parent, sc.kinds, sc.joint_idx, sc.body_branch,
                sc.a_rot, sc.a_pos, sc.b_rot, sc.b_pos, sc.axes,
                sc.root_pos, sc.root_rot,
                ws.bpos, ws.brot, ws.jorig, ws.jaxis,
            ],
        )
        sphere_stride = int(sc.sphere_body.shape[0])
        wp.launch(
            gk.eval_collision_cost_grad_kernel, dim=S * sphere_stride,
            inputs=[
                ws.M_samples, sphere_stride, ws.substeps, ws.T, ws.J, ws.K, fixed_start, fixed_end,
                skip_accepted, ws.accepted,
                ws.bpos, ws.brot, ws.jorig, ws.jaxis,
                sc.body_ancestor, sc.joint_prismatic, ws.active_mask,
                sc.sphere_body, sc.sphere_center, sc.sphere_radius, sc.sphere_esdf_active,
                sc.static_csr_off, sc.static_csr_idx, sc.self_csr_off, sc.self_csr_idx,
                sc.static_kind, sc.static_pos, sc.static_rot, sc.static_size,
                sc.esdf_sdf, sc.scene_i32, sc.scene_f32, opt_stage_buf,
                cost_col_buf, max_col_buf, grad_buf,
            ],
        )
        if num_pose_items > 0:
            wp.launch(
                gk.eval_pose_constraints_kernel, dim=num_pose_items * ws.K,
                inputs=[
                    num_pose_items, ws.M_samples, ws.substeps, ws.T, ws.J, ws.K,
                    fixed_start, fixed_end, skip_accepted, ws.accepted,
                    ws.bpos, ws.brot, ws.jorig, ws.jaxis,
                    sc.body_ancestor, sc.joint_prismatic, ws.active_mask,
                    ws.pc_req, ws.pc_t, ws.pc_link, ws.pc_flags,
                    ws.pc_pos, ws.pc_rot, ws.pc_pos_tol, ws.pc_rot_tol, ws.pc_weight,
                    opt_stage_buf, cost_pose_buf, max_other_buf, grad_buf,
                ],
            )
        wp.launch(
            gk.eval_smooth_limits_targets_kernel, dim=N,
            inputs=[
                q_buf, ws.T, ws.J, ws.K, fixed_start, fixed_end,
                skip_accepted, ws.accepted, ws.active_mask, sc.velocity_limit,
                ws.jt_has, ws.jt_vals, ws.has_start_vel, ws.start_vel, opt_stage_buf,
                cost_col_buf, cost_pose_buf, cost_smooth_buf, f_buf, max_other_buf, grad_buf,
            ],
        )

    def _launch_optimize_pipeline(
        self, ws: _GpuWorkspace, sc: _GpuSceneCache, *,
        fixed_start: int, fixed_end: int, num_pose_items: int,
    ) -> None:
        wp = self._wp
        gk = self._gk
        cfg = self.config
        N = ws.N
        D = ws.D

        for stage_idx, n_iters in enumerate(cfg.lbfgs_iters_per_stage):
            opt_buf = ws.opt_stages[stage_idx]
            wp.launch(
                gk.lbfgs_reset_history_kernel, dim=N,
                inputs=[ws.hist_head, ws.hist_count, ws.accepted],
            )
            self._launch_eval(
                ws, sc,
                q_buf=ws.x, grad_buf=ws.g, f_buf=ws.f,
                cost_col_buf=ws.cost_col, cost_pose_buf=ws.cost_pose,
                cost_smooth_buf=ws.cost_smooth,
                max_col_buf=ws.max_col_viol, max_other_buf=ws.max_other_viol,
                opt_stage_buf=opt_buf,
                fixed_start=fixed_start, fixed_end=fixed_end,
                skip_accepted=0, num_pose_items=num_pose_items,
            )
            for _ in range(n_iters):
                wp.launch(
                    gk.lbfgs_direction_kernel, dim=N,
                    inputs=[
                        ws.x, ws.g, ws.T, ws.J, ws.M_hist, float(cfg.max_step),
                        sc.lower, sc.upper,
                        ws.s_hist, ws.y_hist, ws.rho_hist,
                        ws.hist_head, ws.hist_count, ws.alpha_buf,
                        ws.d, ws.step_alpha, ws.accepted,
                    ],
                )
                for _ in range(cfg.line_search_iters):
                    wp.launch(
                        gk.lbfgs_trial_step_kernel, dim=N * D,
                        inputs=[
                            ws.x, ws.d, ws.T, ws.J, ws.K, fixed_start, fixed_end,
                            sc.lower, sc.upper, ws.active_mask, ws.jt_has,
                            ws.step_alpha, ws.accepted, ws.x_trial,
                        ],
                    )
                    self._launch_eval(
                        ws, sc,
                        q_buf=ws.x_trial, grad_buf=ws.g_trial, f_buf=ws.f_trial,
                        cost_col_buf=ws.cost_col_trial, cost_pose_buf=ws.cost_pose_trial,
                        cost_smooth_buf=ws.cost_smooth_trial,
                        max_col_buf=ws.max_col_viol_trial, max_other_buf=ws.max_other_viol_trial,
                        opt_stage_buf=opt_buf,
                        fixed_start=fixed_start, fixed_end=fixed_end,
                        skip_accepted=1, num_pose_items=num_pose_items,
                    )
                    wp.launch(
                        gk.lbfgs_armijo_check_kernel, dim=N,
                        inputs=[
                            ws.x, ws.x_trial, ws.g, ws.f, ws.f_trial,
                            ws.T, ws.J, float(cfg.armijo_c1), float(cfg.line_search_shrink),
                            ws.step_alpha, ws.accepted,
                        ],
                    )
                wp.launch(
                    gk.lbfgs_update_history_kernel, dim=N,
                    inputs=[
                        ws.x, ws.g, ws.f, ws.cost_smooth, ws.max_col_viol, ws.max_other_viol,
                        ws.x_trial, ws.g_trial, ws.f_trial, ws.cost_smooth_trial,
                        ws.max_col_viol_trial, ws.max_other_viol_trial,
                        ws.T, ws.J, ws.M_hist,
                        ws.s_hist, ws.y_hist, ws.rho_hist,
                        ws.hist_head, ws.hist_count, ws.accepted,
                    ],
                )

        # Rank seeds per request and extract top-M directly on GPU.
        wp.launch(
            gk.rank_select_top_k_kernel, dim=ws.B_chunk,
            inputs=[
                ws.B_chunk, ws.K, ws.top_m, ws.T, ws.J,
                ws.x, ws.cost_smooth, ws.max_col_viol, ws.max_other_viol,
                ws.seed_goal_idx,
                ws.top_x, ws.top_goal, ws.top_seed, ws.top_viol, ws.top_smooth,
                ws.used_mask,
            ],
        )

    def optimize_chunk(
        self,
        snap,
        *,
        seeds_f32: np.ndarray,           # [B_chunk, K, T, J] float32
        seed_goal_idx: np.ndarray,       # [B_chunk, K] int32
        active_mask: np.ndarray,         # [B_chunk, J] uint8
        jt_has: np.ndarray,              # [B_chunk, T] uint8
        jt_vals: np.ndarray,             # [B_chunk, T, J] float32
        has_start_vel: np.ndarray,       # [B_chunk] uint8
        start_vel: np.ndarray,           # [B_chunk, J] float32
        pose_items: dict[str, np.ndarray] | None,
        fixed_start: int,
        fixed_end: int,
        substeps: int,
        self_margin: float,
        world_margin: float,
        dt: float | None,
        vel_scale: float,
        dt_start_smooth: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, bool, bool]:
        """Run GPU-resident batched L-BFGS on one homogeneous chunk.

        Returns:
            (top_x [B, M, T, J], top_goal [B, M], top_seed [B, M], top_viol [B, M],
             upload_ms, graph_capture_ms, gpu_opt_ms, graph_captured, graph_reused)
        """
        wp = self._wp
        cfg = self.config
        B_chunk, K, T, J = seeds_f32.shape
        num_pose = 0 if pose_items is None else int(pose_items["req"].shape[0])
        num_stages = len(cfg.lbfgs_iters_per_stage)

        with self._lock:
            upload_ms = self._scene_cache.sync(snap)
            sc = self._scene_cache
            ws = self._get_workspace(
                B_chunk=B_chunk, K=K, T=T, J=J, num_bodies=sc.num_bodies,
                substeps=substeps, M_hist=cfg.lbfgs_history,
                top_m=cfg.top_k_verify, num_stages=num_stages, max_pose=max(1, num_pose),
            )

            inv_dt = (1.0 / float(dt)) if (dt is not None and dt > 0.0) else 0.0
            stage_arrays = [
                np.array([
                    float(cfg.w_smooth_vel),
                    float(cfg.w_smooth_acc),
                    float(w_col),
                    float(cfg.w_vel_limit),
                    float(cfg.w_joint_target),
                    float(cfg.w_pose_constraint),
                    float(self_margin),
                    float(world_margin),
                    float(cfg.world_margin_cushion),
                    float(cfg.self_margin_cushion),
                    float(cfg.huber_delta),
                    float(inv_dt),
                    float(vel_scale),
                    float(dt_start_smooth),
                ], dtype=np.float32)
                for w_col in cfg.penalty_weights
            ]

            t_h2d0 = perf_counter()
            with wp.ScopedDevice(self._device):
                wp.copy(ws.x, wp.from_numpy(np.ascontiguousarray(seeds_f32.reshape(B_chunk * K, T, J)), dtype=wp.float32))
                wp.copy(ws.seed_goal_idx, wp.from_numpy(np.ascontiguousarray(seed_goal_idx.reshape(-1)), dtype=wp.int32))
                wp.copy(ws.active_mask, wp.from_numpy(np.ascontiguousarray(active_mask), dtype=wp.uint8))
                wp.copy(ws.jt_has, wp.from_numpy(np.ascontiguousarray(jt_has), dtype=wp.uint8))
                wp.copy(ws.jt_vals, wp.from_numpy(np.ascontiguousarray(jt_vals), dtype=wp.float32))
                wp.copy(ws.has_start_vel, wp.from_numpy(np.ascontiguousarray(has_start_vel), dtype=wp.uint8))
                wp.copy(ws.start_vel, wp.from_numpy(np.ascontiguousarray(start_vel), dtype=wp.float32))
                for s_idx, arr in enumerate(stage_arrays):
                    wp.copy(ws.opt_stages[s_idx], wp.from_numpy(arr, dtype=wp.float32))
                if num_pose > 0 and pose_items is not None:
                    wp.copy(ws.pc_req[:num_pose], wp.from_numpy(pose_items["req"], dtype=wp.int32))
                    wp.copy(ws.pc_t[:num_pose], wp.from_numpy(pose_items["t"], dtype=wp.int32))
                    wp.copy(ws.pc_link[:num_pose], wp.from_numpy(pose_items["link"], dtype=wp.int32))
                    wp.copy(ws.pc_flags[:num_pose], wp.from_numpy(pose_items["flags"], dtype=wp.int32))
                    wp.copy(ws.pc_pos[:num_pose], wp.from_numpy(pose_items["pos"], dtype=wp.vec3))
                    wp.copy(ws.pc_rot[:num_pose], wp.from_numpy(pose_items["rot"], dtype=wp.mat33))
                    wp.copy(ws.pc_pos_tol[:num_pose], wp.from_numpy(pose_items["pos_tol"], dtype=wp.vec3))
                    wp.copy(ws.pc_rot_tol[:num_pose], wp.from_numpy(pose_items["rot_tol"], dtype=wp.vec3))
                    wp.copy(ws.pc_weight[:num_pose], wp.from_numpy(pose_items["weight"], dtype=wp.float32))

                upload_ms += (perf_counter() - t_h2d0) * 1000.0
                graph_capture_ms = 0.0
                graph_captured = False
                graph_reused = False

                if cfg.use_cuda_graph:
                    graph_key = (
                        B_chunk, K, T, J, substeps, fixed_start, fixed_end,
                        ws.M_hist, cfg.lbfgs_iters_per_stage, cfg.line_search_iters,
                        num_pose, ws.top_m, float(cfg.max_step),
                        float(cfg.armijo_c1), float(cfg.line_search_shrink),
                        sc.ptrs_key(), ws.ptrs_key(),
                    )
                    graph = self._graphs.get(graph_key)
                    if graph is None:
                        if len(self._graphs) >= 16:
                            self._graphs.clear()
                        # Warm up kernels before capture (first-time JIT compilation happens outside capture).
                        wp.synchronize_device(self._device)
                        t_cap0 = perf_counter()
                        with wp.ScopedCapture(device=self._device) as capture:
                            self._launch_optimize_pipeline(
                                ws, sc,
                                fixed_start=fixed_start, fixed_end=fixed_end,
                                num_pose_items=num_pose,
                            )
                        graph = capture.graph
                        wp.synchronize_device(self._device)
                        graph_capture_ms = (perf_counter() - t_cap0) * 1000.0
                        self._graphs[graph_key] = graph
                        graph_captured = True
                    else:
                        graph_reused = True

                    t_opt0 = perf_counter()
                    wp.capture_launch(graph)
                    wp.synchronize_device(self._device)
                    gpu_opt_ms = (perf_counter() - t_opt0) * 1000.0
                else:
                    t_opt0 = perf_counter()
                    self._launch_optimize_pipeline(
                        ws, sc,
                        fixed_start=fixed_start, fixed_end=fixed_end,
                        num_pose_items=num_pose,
                    )
                    wp.synchronize_device(self._device)
                    gpu_opt_ms = (perf_counter() - t_opt0) * 1000.0

                top_x = ws.top_x.numpy().astype(np.float64, copy=False)
                top_goal = ws.top_goal.numpy()
                top_seed = ws.top_seed.numpy()
                top_viol = ws.top_viol.numpy()

            return (
                top_x, top_goal, top_seed, top_viol,
                upload_ms, graph_capture_ms, gpu_opt_ms,
                graph_captured, graph_reused,
            )


class GpuBatchBackend:
    """MotionPlanner backend backed by GPU-resident batched L-BFGS + C++ TBB request verification."""

    name = "gpu_batch"

    def __init__(self, world: PlanningWorld, *, config: GpuBatchConfig | None = None) -> None:
        self.world = world
        self.config = GpuBatchConfig() if config is None else config
        self.optimizer = GpuBatchOptimizer(world, self.config)
        self.last_stats = GpuBatchStats()

    def _compile_pose_items(self, chunk_pose_maps: list[dict[int, list]]) -> dict[str, np.ndarray] | None:
        reqs: list[int] = []
        ts: list[int] = []
        links: list[int] = []
        flags: list[int] = []
        pos_list: list[np.ndarray] = []
        rot_list: list[np.ndarray] = []
        ptol_list: list[np.ndarray] = []
        rtol_list: list[np.ndarray] = []
        weights: list[float] = []

        for local_i, pmap in enumerate(chunk_pose_maps):
            for t_idx, specs in pmap.items():
                for sp in specs:
                    reqs.append(local_i)
                    ts.append(int(t_idx))
                    links.append(int(sp.link_id))
                    fl = (1 if sp.has_position else 0) | (2 if sp.has_rotation else 0)
                    flags.append(fl)
                    pos_list.append(np.asarray(sp.position, dtype=np.float32))
                    rot_list.append(np.asarray(sp.rotation, dtype=np.float32))
                    pt = np.asarray(sp.pos_tol, dtype=np.float64)
                    rt = np.asarray(sp.rot_tol, dtype=np.float64)
                    ptol_list.append(np.where(np.isfinite(pt), pt, -1.0).astype(np.float32))
                    rtol_list.append(np.where(np.isfinite(rt), rt, -1.0).astype(np.float32))
                    weights.append(float(sp.weight))

        if not reqs:
            return None
        return {
            "req": np.asarray(reqs, dtype=np.int32),
            "t": np.asarray(ts, dtype=np.int32),
            "link": np.asarray(links, dtype=np.int32),
            "flags": np.asarray(flags, dtype=np.int32),
            "pos": np.ascontiguousarray(np.stack(pos_list, axis=0), dtype=np.float32),
            "rot": np.ascontiguousarray(np.stack(rot_list, axis=0), dtype=np.float32),
            "pos_tol": np.ascontiguousarray(np.stack(ptol_list, axis=0), dtype=np.float32),
            "rot_tol": np.ascontiguousarray(np.stack(rtol_list, axis=0), dtype=np.float32),
            "weight": np.asarray(weights, dtype=np.float32),
        }

    def _resolve_batch_endpoints(
        self,
        snap,
        requests: Sequence[MotionPlanRequest],
        masks: list[np.ndarray],
    ) -> tuple[list[np.ndarray | None], list[list[tuple[int, np.ndarray]] | None], list[str | None], float]:
        """Resolve start & goal joint configs across a batch (using parallel C++ solve_ik_batch for pose endpoints)."""
        t0 = perf_counter()
        B = len(requests)
        starts: list[np.ndarray | None] = [None] * B
        goals: list[list[tuple[int, np.ndarray]] | None] = [None] * B
        errors: list[str | None] = [None] * B

        lower, upper = snap.lower_limits, snap.upper_limits

        def ik_seeds(base, mask, count):
            yield base
            if count > 1:
                rng = np.random.default_rng(0)
                lo = np.where(np.isfinite(lower), lower, base - np.pi)
                hi = np.where(np.isfinite(upper), upper, base + np.pi)
                for _ in range(count - 1):
                    seed = base.copy()
                    seed[mask] = rng.uniform(lo[mask], hi[mask])
                    yield seed

        for i, req in enumerate(requests):
            if _is_pose_sequence(req.start):
                compiled = [self.world.compile_pose_constraint(t) for t in req.start]
                base = self.world.initial_positions.copy() if req.start_seed is None else self.world.configuration(req.start_seed)
                problems = [(compiled, seed) for seed in ik_seeds(base, masks[i], req.config.num_seeds)]
                q_out, ok_out, reasons = snap.solve_ik_batch(
                    problems, masks[i], req.config.margin, req.config.effective_world_margin, 40, True,
                )
                valid = np.flatnonzero(ok_out)
                if not len(valid):
                    errors[i] = f"start IK failed: {reasons[0] if reasons else 'IK failed'}"
                else:
                    best = valid[np.argmin(np.linalg.norm(q_out[valid] - base, axis=-1))]
                    starts[i] = q_out[best].copy()
            else:
                starts[i] = self.world.configuration(req.start)
                if np.any(starts[i] < lower) or np.any(starts[i] > upper):
                    errors[i] = "start or goal violates joint limits"

        # 2. Resolve goal endpoints (batch all pose goals in parallel per (mask, margin, wm) group!).
        pose_goal_groups: dict[tuple, list[tuple[int, int, list, np.ndarray]]] = {}
        for i, req in enumerate(requests):
            if errors[i] is not None or starts[i] is None:
                continue
            start_q = starts[i]
            mask = masks[i]
            cfg = req.config
            if _is_pose_sequence(req.goal):
                g_seed = start_q if req.goal_seed is None else self.world.configuration(req.goal_seed)
                compiled = [self.world.compile_pose_constraint(t) for t in req.goal]
                key = (mask.tobytes(), float(cfg.margin), float(cfg.effective_world_margin))
                for seed in ik_seeds(g_seed, mask, cfg.num_seeds):
                    pose_goal_groups.setdefault(key, []).append((i, 0, compiled, seed))
            elif (isinstance(req.goal, Sequence) and not isinstance(req.goal, (str, bytes, np.ndarray))
                  and len(req.goal) > 0 and _is_pose_sequence(req.goal[0])):
                g_seed = start_q if req.goal_seed is None else self.world.configuration(req.goal_seed)
                key = (mask.tobytes(), float(cfg.margin), float(cfg.effective_world_margin))
                for gid, ts in enumerate(req.goal):
                    compiled = [self.world.compile_pose_constraint(t) for t in ts]
                    pose_goal_groups.setdefault(key, []).append((i, gid, compiled, g_seed))
            elif (isinstance(req.goal, Sequence) and not isinstance(req.goal, (str, bytes, np.ndarray))
                  and len(req.goal) > 0 and not isinstance(req.goal[0], (int, float, np.number))):
                cands = []
                for gid, gq_raw in enumerate(req.goal):
                    gq = self.world.configuration(gq_raw).copy()
                    if not mask.all():
                        gq[~mask] = start_q[~mask]
                    cands.append((gid, gq))
                cands.sort(key=lambda item: float(np.linalg.norm(item[1] - start_q)))
                goals[i] = cands
            else:
                gq = self.world.configuration(req.goal).copy()
                if not mask.all():
                    gq[~mask] = start_q[~mask]
                goals[i] = [(0, gq)]

        for (mask_bytes, margin, wm), entries in pose_goal_groups.items():
            mask = np.frombuffer(mask_bytes, dtype=np.bool_)
            problems = [(compiled, g_seed) for _, _, compiled, g_seed in entries]
            q_out, ok_out, reasons = snap.solve_ik_batch(problems, mask, margin, wm, 40, True)
            per_req: dict[int, list[tuple[int, np.ndarray]]] = {}
            per_req_err: dict[int, str] = {}
            for row, (req_i, gid, _, _) in enumerate(entries):
                if ok_out[row]:
                    gq = q_out[row].copy()
                    if not mask.all():
                        gq[~mask] = starts[req_i][~mask]
                    per_req.setdefault(req_i, []).append((gid, gq))
                elif req_i not in per_req_err:
                    per_req_err[req_i] = reasons[row] or "IK failed"
            for req_i in {e[0] for e in entries}:
                cands = per_req.get(req_i, [])
                if not cands:
                    errors[req_i] = f"goal IK failed: {per_req_err.get(req_i, 'IK failed')}"
                else:
                    sq = starts[req_i]
                    req = requests[req_i]
                    if _is_pose_sequence(req.goal):
                        base = sq if req.goal_seed is None else self.world.configuration(req.goal_seed)
                        goals[req_i] = [min(cands, key=lambda item: np.linalg.norm(item[1] - base))]
                    else:
                        cands.sort(key=lambda item: np.linalg.norm(item[1] - sq))
                        goals[req_i] = cands

        for i, candidates in enumerate(goals):
            if candidates:
                goals[i] = [(gid, q) for gid, q in candidates if np.all((lower <= q) & (q <= upper))]
                if not goals[i]:
                    errors[i] = "start or goal violates joint limits"

        return starts, goals, errors, (perf_counter() - t0) * 1000.0

    def plan_single(self, native_planner, request: MotionPlanRequest) -> PlanResult:
        return self.plan_batch(native_planner, [request], chunk_size=1)[0]

    def plan_batch(
        self,
        native_planner,
        requests: Sequence[MotionPlanRequest],
        *,
        chunk_size: int = 64,
    ) -> tuple[PlanResult, ...]:
        from .planner import (
            _expand_path_constraints,
            _plan_result_from_native,
            _request_to_native_dict,
        )

        t_all0 = perf_counter()
        B = len(requests)
        pinned = self.world._native.snapshot()
        pinned_ver = int(pinned.version)
        J = self.world.num_joints
        num_bodies = int(pinned.num_bodies)
        cfg_gpu = self.config

        results: list[PlanResult | None] = [None] * B
        direct_cpu_indices: list[int] = []
        gpu_indices: list[int] = []

        # Route requests with global manifold_constraints directly to C++ on the same pinned snapshot.
        for i, req in enumerate(requests):
            if req.manifold_constraints:
                direct_cpu_indices.append(i)
            else:
                gpu_indices.append(i)

        if direct_cpu_indices:
            direct_reqs = [_request_to_native_dict(self.world, requests[i]) for i in direct_cpu_indices]
            raw_direct = native_planner.plan_batch_with_snapshot(pinned, direct_reqs, int(chunk_size))
            for idx, raw in zip(direct_cpu_indices, raw_direct):
                results[idx] = _plan_result_from_native(raw)

        total_upload_ms = 0.0
        total_ik_ms = 0.0
        total_seed_ms = 0.0
        total_gpu_opt_ms = 0.0
        total_graph_capture_ms = 0.0
        total_verify_ms = 0.0
        total_fallback_ms = 0.0
        top1_passed = 0
        topm_passed = 0
        cpu_fallback_count = 0
        failed_count = 0
        any_graph_captured = False
        any_graph_reused = False

        if gpu_indices:
            gpu_reqs = [requests[i] for i in gpu_indices]
            masks = [self.world.group_mask(req.group) for req in gpu_reqs]
            starts, goals, ep_errors, ik_ms = self._resolve_batch_endpoints(pinned, gpu_reqs, masks)
            total_ik_ms += ik_ms

            # Compile per-request constraints & group homogeneous requests for GPU chunking.
            buckets: dict[tuple, list[int]] = {}
            pose_maps: list[dict[int, list]] = [{} for _ in gpu_reqs]
            joint_maps: list[dict[int, np.ndarray]] = [{} for _ in gpu_reqs]
            native_req_dicts: list[dict | None] = [None] * len(gpu_reqs)

            endpoint_failures = []
            for local_i, req in enumerate(gpu_reqs):
                global_i = gpu_indices[local_i]
                cfg = req.config
                has_start_bc = (req.start_velocity is not None) or (req.start_acceleration is not None)
                fixed_start = 1 if has_start_bc else (2 if cfg.stop_at_endpoints else 1)
                fixed_end = 2 if cfg.stop_at_endpoints else 1
                if cfg.waypoints < fixed_start + fixed_end:
                    raise ValueError(f"waypoints must be >= {fixed_start + fixed_end}")
                if ep_errors[local_i] is not None or starts[local_i] is None or goals[local_i] is None:
                    if cfg_gpu.cpu_fallback_on_fail:
                        endpoint_failures.append(local_i)
                        continue
                    results[global_i] = PlanResult(
                        None, None, 0, ep_errors[local_i] or "endpoint resolution failed",
                        PlanDiagnostics(total_ms=ik_ms, ik_ms=ik_ms, seed_strategy="gpu_lbfgs"),
                        world_version=pinned_ver,
                    )
                    failed_count += 1
                    continue

                pmap, jmap = _expand_path_constraints(
                    self.world, cfg.waypoints, fixed_start, req.waypoint_targets,
                    req.path_constraints, fixed_end=fixed_end,
                )
                pose_maps[local_i] = pmap
                joint_maps[local_i] = jmap
                native_req_dicts[local_i] = _request_to_native_dict(self.world, req)

                substeps = max(1, min(int(cfg.collision_substeps), int(cfg_gpu.max_gpu_substeps)))
                bucket_key = (
                    int(cfg.waypoints),
                    int(fixed_start),
                    int(fixed_end),
                    substeps,
                    float(cfg.margin),
                    float(cfg.effective_world_margin),
                    None if cfg.dt is None else float(cfg.dt),
                    float(cfg.velocity_scale),
                )
                buckets.setdefault(bucket_key, []).append(local_i)

            if endpoint_failures:
                items = [{"request": _request_to_native_dict(self.world, gpu_reqs[i]), "candidates": []}
                         for i in endpoint_failures]
                t_fallback = perf_counter()
                fallback_results = native_planner.finalize_candidates_batch(pinned, items, True)
                total_fallback_ms += (perf_counter() - t_fallback) * 1000.0
                for i, item in zip(endpoint_failures, fallback_results):
                    result = _plan_result_from_native(item["result"])
                    results[gpu_indices[i]] = result
                    cpu_fallback_count += int(item["used_cpu_fallback"])
                    failed_count += int(not result.success)

            rng = np.random.default_rng(cfg_gpu.rng_seed)
            lo_lim = pinned.lower_limits
            hi_lim = pinned.upper_limits

            for bkey, bucket_locals in buckets.items():
                T, fixed_start, fixed_end, substeps, self_margin, world_margin, dt_val, vel_scale = bkey
                max_goals = max(len(goals[li]) for li in bucket_locals)
                max_req_seeds = max(int(gpu_reqs[li].config.num_seeds) for li in bucket_locals)
                K = max(cfg_gpu.num_seeds, max_goals, max_req_seeds)

                # Enforce VRAM budget automatically when choosing chunk size.
                bytes_per_traj = _estimate_bytes_per_traj(T, J, num_bodies, substeps, cfg_gpu.lbfgs_history)
                budget_bytes = cfg_gpu.vram_budget_mb * 1024 * 1024
                max_n_vram = max(K, int(budget_bytes // max(1, bytes_per_traj)))
                max_b_vram = max(1, max_n_vram // K)
                eff_chunk_b = max(1, min(len(bucket_locals), int(chunk_size), max_b_vram))

                for offset in range(0, len(bucket_locals), eff_chunk_b):
                    sub_locals = bucket_locals[offset:min(offset + eff_chunk_b, len(bucket_locals))]
                    Bc = len(sub_locals)

                    t_seed0 = perf_counter()
                    seeds_f32 = np.empty((Bc, K, T, J), dtype=np.float32)
                    seed_goals = np.empty((Bc, K), dtype=np.int32)
                    mask_u8 = np.empty((Bc, J), dtype=np.uint8)
                    jt_has = np.zeros((Bc, T), dtype=np.uint8)
                    jt_vals = np.zeros((Bc, T, J), dtype=np.float32)
                    has_sv = np.zeros(Bc, dtype=np.uint8)
                    sv_arr = np.zeros((Bc, J), dtype=np.float32)
                    sub_pose_maps: list[dict[int, list]] = []

                    for ci, li in enumerate(sub_locals):
                        req = gpu_reqs[li]
                        sq = starts[li]
                        glist = goals[li]
                        mk = masks[li]
                        jmap = joint_maps[li]
                        sub_pose_maps.append(pose_maps[li])
                        mask_u8[ci] = mk.astype(np.uint8)
                        for idx, tgt in jmap.items():
                            if fixed_start <= idx < T - fixed_end:
                                jt_has[ci, idx] = 1
                                jt_vals[ci, idx] = tgt.astype(np.float32)
                        if req.start_velocity is not None:
                            has_sv[ci] = 1
                            sv_arr[ci] = np.asarray(req.start_velocity, dtype=np.float32)
                        sd_k, gid_k = _build_piecewise_seeds(
                            sq, glist, T, fixed_start, fixed_end, jmap, mk,
                            lo_lim, hi_lim, K, cfg_gpu.seed_noise_std, req.seed, rng,
                        )
                        seeds_f32[ci] = sd_k
                        seed_goals[ci] = gid_k

                    pose_items = self._compile_pose_items(sub_pose_maps)
                    seed_ms = (perf_counter() - t_seed0) * 1000.0
                    total_seed_ms += seed_ms

                    (
                        top_x, top_goal, top_seed, _top_viol,
                        up_ms, cap_ms, opt_ms, captured, reused,
                    ) = self.optimizer.optimize_chunk(
                        pinned,
                        seeds_f32=seeds_f32,
                        seed_goal_idx=seed_goals,
                        active_mask=mask_u8,
                        jt_has=jt_has,
                        jt_vals=jt_vals,
                        has_start_vel=has_sv,
                        start_vel=sv_arr,
                        pose_items=pose_items,
                        fixed_start=fixed_start,
                        fixed_end=fixed_end,
                        substeps=substeps,
                        self_margin=self_margin,
                        world_margin=world_margin,
                        dt=dt_val,
                        vel_scale=vel_scale,
                    )
                    total_upload_ms += up_ms
                    total_graph_capture_ms += cap_ms
                    total_gpu_opt_ms += opt_ms
                    any_graph_captured = any_graph_captured or captured
                    any_graph_reused = any_graph_reused or reused

                    # Build request-level C++ verification batch for top-M candidates per request.
                    top_m = int(top_x.shape[1])
                    verify_items = []
                    for ci, li in enumerate(sub_locals):
                        cands = []
                        for r in range(top_m):
                            sk = int(top_seed[ci, r])
                            cands.append({
                                "positions": np.ascontiguousarray(top_x[ci, r], dtype=np.float64),
                                "goal_index": int(top_goal[ci, r]),
                                "seed_strategy": "gpu_lbfgs",
                                "initial_positions": np.ascontiguousarray(seeds_f32[ci, sk], dtype=np.float64),
                            })
                        verify_items.append({
                            "request": native_req_dicts[li],
                            "candidates": cands,
                        })

                    t_ver0 = perf_counter()
                    ver_out = native_planner.finalize_candidates_batch(
                        pinned, verify_items, bool(cfg_gpu.cpu_fallback_on_fail),
                    )
                    ver_wall_ms = (perf_counter() - t_ver0) * 1000.0

                    # Aggregate verify/fallback wall-clock breakdown proportionally.
                    sum_v = sum(float(v["verify_ms"]) for v in ver_out)
                    sum_fb = sum(float(v["fallback_ms"]) for v in ver_out)
                    denom = max(1e-9, sum_v + sum_fb)
                    total_verify_ms += ver_wall_ms * (sum_v / denom)
                    total_fallback_ms += ver_wall_ms * (sum_fb / denom)

                    per_req_gpu_ms = opt_ms / float(Bc)
                    per_req_seed_ms = seed_ms / float(Bc)
                    per_req_ik_ms = ik_ms / float(len(gpu_reqs))

                    for ci, (li, item_out) in enumerate(zip(sub_locals, ver_out)):
                        global_i = gpu_indices[li]
                        pr = _plan_result_from_native(item_out["result"])
                        acc_idx = int(item_out["accepted_candidate_index"])
                        used_fb = bool(item_out["used_cpu_fallback"])
                        if acc_idx == 0:
                            top1_passed += 1
                        if acc_idx >= 0:
                            topm_passed += 1
                        if used_fb:
                            cpu_fallback_count += 1
                        if not pr.success:
                            failed_count += 1

                        d = pr.diagnostics
                        merged_diag = PlanDiagnostics(
                            total_ms=per_req_ik_ms + per_req_seed_ms + per_req_gpu_ms + float(item_out["verify_ms"]) + float(item_out["fallback_ms"]),
                            ik_ms=per_req_ik_ms + d.ik_ms,
                            seed_ms=per_req_seed_ms + d.seed_ms,
                            sqp_ms=per_req_gpu_ms + d.sqp_ms,
                            validate_ms=d.validate_ms,
                            retime_ms=d.retime_ms,
                            qp_calls=d.qp_calls,
                            sqp_iterations=sum(cfg_gpu.lbfgs_iters_per_stage) if acc_idx >= 0 else d.sqp_iterations,
                            seed_strategy=d.seed_strategy,
                            witnesses=d.witnesses,
                            timed_out=d.timed_out,
                        )
                        results[global_i] = PlanResult(
                            trajectory=pr.trajectory,
                            initial_positions=pr.initial_positions,
                            goal_index=pr.goal_index,
                            reason=pr.reason,
                            diagnostics=merged_diag,
                            world_version=pr.world_version,
                            waypoint_indices=pr.waypoint_indices,
                        )

        total_ms = (perf_counter() - t_all0) * 1000.0
        self.last_stats = GpuBatchStats(
            num_requests=B,
            num_gpu_requests=len(gpu_indices),
            num_direct_cpu_requests=len(direct_cpu_indices),
            top1_passed=top1_passed,
            topm_passed=topm_passed,
            cpu_fallback_count=cpu_fallback_count,
            failed_count=failed_count,
            upload_ms=total_upload_ms,
            ik_ms=total_ik_ms,
            seed_ms=total_seed_ms,
            gpu_opt_ms=total_gpu_opt_ms,
            graph_capture_ms=total_graph_capture_ms,
            verify_ms=total_verify_ms,
            fallback_ms=total_fallback_ms,
            total_ms=total_ms,
            graph_captured=any_graph_captured,
            graph_reused=any_graph_reused,
        )
        return tuple(r for r in results if r is not None)
