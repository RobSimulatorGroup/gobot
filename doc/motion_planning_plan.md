# Gobot Motion Planning Roadmap

This document defines the four-phase engineering plan for Gobot's motion planning stack across [`modules/coal/`](../modules/coal/planning_world.hpp), [`python/gobot/planning/`](../python/gobot/planning/__init__.py), and [`src/gobot/editor/`](../src/gobot/editor/editor.cpp).

All work follows the repository architecture boundaries in [`AGENTS.md`](../AGENTS.md):

- `Scene` remains the authoring source of truth.
- `PlanningWorld` / `_coal_native` owns C++ kinematics, collision geometry, solvers, and geometric verification without depending on ImGui or OpenGL.
- `RenderBackend` renders debug visualization through explicit debug-draw primitives.
- `Editor` coordinates interactive planning gizmos, ghost previews, and panels without polluting user scene trees.

## Baseline (Before Phase 1)

Measured on Ryzen 7 3700X (8C/16T) + RTX 4070 (`cuda:0`) with exact Hermite velocity/acceleration bounds and convex + original triangle-mesh collision verification:

- **Panda TrajOpt** (`16` waypoints, `7` DOF): `~233 ms` total (`216 ms` SQP across `13` OSQP calls = `16.6 ms/QP`, `14 ms` retime).
- **Container Unloading** (`14` waypoints, `6` DOF, `350` shape pairs, no via): `~655 ms` total (`458 ms` in `_qp_step` across `133` OSQP calls = `3.45 ms/QP`, `121 ms` collision eval, `42 ms` retime).
- **Container Batch `B=4`** (`dt=None` TOPP): `~650 ms` to `2.1 s` (dominated by OSQP + Python/SciPy sparse matrix setup and outlier QP counts).

Key bottleneck finding: **70–90% of total planning latency is in Python/SciPy CSC assembly + OSQP ADMM calls** (`3.5–16.6 ms` per 72–112 variable QP), while C++ FK + AABB + Coal + analytic Jacobians take only `~0.55 ms` per evaluation.

---

## Phase 1: Extreme Performance (Completed)

Goal: cut single-trajectory planning latency and scale batched planning to `B = 32, 64, 128` while preserving exact Hermite velocity/acceleration bounds and tolerance-bounded continuous CCD (`< 0.5 * disp_tol`, with optional `strict_ccd=True` verification) across convex hulls and original triangle meshes.

### Phase 1 Measured Results (Ryzen 7 3700X + RTX 4070, Hermite + Mesh CCD Verification)
- **Panda TrajOpt** (`16` waypoints, `7` DOF): `233.4 ms` → **`50.0 ms`** (**4.7× faster**).
- **Container Unloading (no via)**: `666.1 ms` → **`179.8 ms`** (**3.7× faster**, duration `3.17 s` → `2.86 s`).
- **Container Batch (`sqp_fast`, `dt=None` TOPP, 100% solved & independently verified)**:
  - `B=1`: `coal` **`63.1 ms`**, `mujoco_warp` **`174.6 ms`**
  - `B=4`: `coal` **`205 ms`** (`51.2 ms/traj`, **10.5× faster** than `2.16 s`), `mujoco_warp` **`376 ms`** (`94.1 ms/traj`)
  - `B=8`: `coal` **`486 ms`** (`60.7 ms/traj`), `mujoco_warp` **`981 ms`** (`122.6 ms/traj`)
  - `B=32`: `coal` **`1.162 s`** (`36.3 ms/traj`), `mujoco_warp` **`2.895 s`** (`90.5 ms/traj`)
  - `B=64`: `coal` **`2.305 s`** (`36.0 ms/traj`), `mujoco_warp` **`5.265 s`** (`82.3 ms/traj`)
  - `B=128`: `coal` **`4.736 s`** (`37.0 ms/traj`), `mujoco_warp` **`11.028 s`** (`86.2 ms/traj`)
- **C++ Parallel Batch IK**: `0.14 ms` (`B=1`) → `12.35 ms` (`B=128`, `0.10 ms/solve`).

### 1.1 C++ Native L1-Hinge Box-Constrained QP Solver (`modules/coal/planning_qp.hpp`)
- Formulate the exact TrajOpt trust-region subproblem:
  $$\min_{l \le x \le u} \frac{1}{2} x^T H x + g^T x + \mu \sum_{i=1}^m \max(0, v_i + a_i^T x)$$
- Eliminate all $m$ slack variables analytically via **Proximal Augmented Lagrangian + Semismooth Newton (ProxAL-SSN)**:
  - Each Newton step solves only an $n \times n$ ($72 \le n \le 128$) SPD linear system $H + \epsilon I + \rho A_{\mathcal{A}}^T A_{\mathcal{A}} + \rho D_{\mathcal{B}}$ via Eigen `LLT`.
  - Reuse primal/dual warm starts across trust-region radius reductions without re-assembling matrices.
  - Expose directly to Python (`_coal_native.solve_l1_box_qp`) so both `CoalBackend` and `MuJoCoWarpBackend` benefit immediately, while keeping OSQP optional/fallback.

### 1.2 C++ Native SQP Engine (`modules/coal/planning_sqp.hpp`)
- Move the entire Coal-backend trust-region SQP loop (smoothness Hessian, analytic collision + pose constraint assembly, trust-region acceptance, failure-pair caching, and convex/original mesh verification) into C++ `PlanningWorld::solve_sqp_batch(...)`.
- Parallelize batch jobs and multi-seed racing directly across C++ threads (`ThreadPool`) with zero GIL contention and zero Python/SciPy overhead per SQP iteration.
- Support optional soft relative-velocity / segment-length balance when `dt=None` so TOPP-mode batch jobs converge in as few QPs as fixed-`dt` jobs without sacrificing optimality.

### 1.3 Fast Retiming & Continuous Collision Checking
- Vectorize / eliminate redundant inner work in `retime_trajectory` and `_finalize_trajectory` so Hermite retiming takes `< 3 ms`.
- Add C++ Conservative Advancement continuous collision checking in `PlanningWorld` to catch inter-sample grazing contacts and accelerate continuous verification.

### 1.4 Verification & Benchmarks
- Keep all existing regression tests passing and add dedicated C++ QP KKT optimality + C++/Python SQP equivalence tests.
- Refresh `examples/container_unloading/README.md` and `examples/panda_trajopt/README.md` with verified strict-Hermite benchmark tables.

---

## Phase 1.5: Architecture & Correctness Hardening (Completed)

Goal: resolve foundational soundness, thread-safety, layering, and CPU utilization issues before building Phases 2–4.

1. **Standalone C++ Planning Library (`libgobot_planning.so`)**:
   - Extracted [`gobot_planning`](../modules/coal/CMakeLists.txt) with zero Python/pybind11 dependency, including C++ TOPP retiming ([`planning_retime.hpp`](../modules/coal/planning_retime.hpp)), unified Hermite finalization, and [`MotionPlannerNative`](../modules/coal/planning_planner.hpp).
   - `_coal_native` and `python/gobot/planning/planner.py` are now thin wrappers over `gobot_planning`; future C++ editor tools (Phase 4) can link `gobot_planning` directly.
   - Added standalone C++ GoogleTest target `test_motion_planning_cpp`.
2. **Thread-Safe Immutable Copy-on-Write World Snapshots (`WorldSnapshot`)**:
   - All mutations on [`PlanningWorld`](../modules/coal/planning_world.hpp) clone lightweight metadata (sharing immutable `CollisionShape` pointers) and atomically swap the active `std::shared_ptr<const WorldSnapshot>`.
   - Native planners, atomic `validate_trajectory_collision(...)`, and Python-pinned `WorldSnapshot` handles (used across the entire `MuJoCoWarpBackend` solve path) execute lock-free against a single immutable snapshot, with explicit `frames`/`margins` dimension guards preventing OOB reads.
3. **Sound Kinematic-Chain Continuous Collision Checking (Linear + Hermite, Convex + Mesh)**:
   - Replaced chord-distance displacement estimates with rotation-invariant kinematic-chain Lipschitz upper bounds (`KinematicsModel::body_lipschitz_weights`), exact closed-form cubic Hermite joint total variation (`hermite_scalar_total_variation`), and exact analytical per-segment prismatic extremum bounds (`hermite_scalar_abs_max`, sound even without finite joint limits).
   - Added halfspace-plane AABB pruning and exact convex lower-bound pruning (`d_orig >= d_convex`) for original triangle meshes.
   - Covers both SQP linear segments and the actual played Hermite curves. By default (`strict_ccd=False`), any unresolved inter-sample dip below margin is bounded strictly by `< 0.5 * disp_tol` (`< 0.5 mm` at `disp_tol = 1e-3` in solver / `< 0.25 mm` at `disp_tol = 5e-4` in `validate_trajectory`); setting `strict_ccd=True` conservatively rejects any interval not proven safe within `disp_tol`.
4. **TBB Nested Work-Stealing Parallelism**:
   - Switched `ThreadPool` to `TBB::tbb` (`tbb::parallel_for`) when available, raising small-batch CPU utilization from `2–5 / 16` cores to `8.6–15.6 / 16` cores across `B = 1..32`.
   - `MuJoCoWarpBackend` is marked experimental; `PlanningWorld.from_snapshot` warns explicitly when joints omit `acceleration_limit`.
5. **Phase 1.5 Verified Speedups (Ryzen 7 3700X, TBB + C++ `MotionPlannerNative` + Kinematic-Chain Hermite CCD)**:
   - **Panda TrajOpt**: `44.1 ms` → **`30.9 ms`** (7.6× faster than pre-Phase 1).
   - **Container Unloading (no via)**: `135.8 ms` → **`121.9 ms`** (5.5× faster than pre-Phase 1).
   - **Container Batch (`coal`, `sqp_fast`, `dt=None` TOPP)**:
     - `B=1`: **`35.7 ms`**, `B=4`: **`50.0 ms`** (`12.5 ms/traj`), `B=8`: **`149 ms`** (`18.6 ms/traj`), `B=32`: **`402 ms`** (`12.6 ms/traj`), `B=64`: **`901 ms`** (`14.1 ms/traj`), `B=128`: **`1.666 s`** (`13.0 ms/traj`).

---

## Phase 2: Industrial Planning Capabilities

Goal: cover the core workflows expected in industrial manipulation and assembly cells.

1. **Incremental Scene Synchronization (`PlanningWorld.sync_from_scene`)**
   - Update obstacle and robot base transforms incrementally from a live Gobot `Scene` onto CoW snapshots without rebuilding meshes or BVHs.
2. **Cartesian Path Planner (`MotionPlanner.plan_cartesian_path`)**
   - Linear and circular arc end-effector interpolation with configurable translation/rotation step sizes.
   - Continuity-preserving IK (minimum joint jump + singularity damping) + collision, joint-limit, and actual continuous-curve Cartesian deviation checks.
3. **`C²` Quintic Spline & Jerk-Bounded Retiming**
   - Upgrade `Trajectory` representation to support `C²`-continuous quintic Hermite splines so joint accelerations are continuous at knots and jerk (`|j(t)| <= jerk_limit`) is strictly bounded.
4. **Constrained Global Planning (Manifold-Projected BiRRT)**
   - Extend C++ `plan_birrt` to project sampled states onto `PathConstraint` manifolds using existing analytic pose Jacobians.

---

## Phase 3: Dynamic Obstacle Avoidance & Perception

Goal: enable millisecond-level obstacle avoidance against live depth/point-cloud sensors and moving obstacles.

1. **CPU Voxel / 3D ESDF + Sphere-Cover Obstacles First**
   - Add `set_occupancy_grid`, `set_point_cloud`, and `set_esdf` APIs on `PlanningWorld`, plus automatic link sphere-cover approximation integrated into C++ SQP (fast O(1) trilinear distance/gradient lookup).
2. **Warm-Start Reactive Replanning (`10–30 Hz`)**
   - Add start-velocity boundary constraints to SQP and retiming (`q(0)=q_now, v(0)=v_now`), plus `replan(q_now, dq_now, prev_traj)` with shifted warm start and a moving-obstacle demo.
3. **Full GPU-Resident Batch Optimizer (Follow-up)**
   - Keep FK, sphere-to-ESDF collision/gradients, and a branchless parallel optimizer (L-BFGS / MPPI) resident in GPU memory with zero per-step PCIe transfers so large batches (`B >= 64..1024`) genuinely outperform 16-thread CPU Coal.

---

## Phase 4: Editor Interactive Motion Planning

Goal: provide an RViz/MoveIt-style interactive planning workflow inside `gobot_editor` linked directly against `gobot_planning` while keeping clean `Scene / RenderBackend / Editor` boundaries.

1. **Viewport Overlay Mesh Pass & Real-Time IK Ghost Preview**
   - Add an editor viewport overlay mesh draw path (without injecting ghost nodes into `SceneTree`) so dragging an end-effector gizmo on a selected `Robot3D` invokes C++ `gobot_planning` IK live and renders a translucent reachability-colored ghost robot.
2. **Motion Planning Editor Panel**
   - Select planning group, start/goal configurations, margins, velocity/acceleration scales, and debug overlays; trigger `Plan`, `Preview`, `Stop`, and `Re-Plan` asynchronously against immutable `WorldSnapshot`s.
3. **Timeline Scrubbing & Physics Tracking Handoff**
   - Scrub planned trajectories frame-by-frame with live end-effector paths and collision witnesses, export reusable trajectories, and optionally execute them through `SimulationServer` / `JointController` in MuJoCo.
