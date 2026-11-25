# Gobot Motion Planning Roadmap

This document defines the four-phase engineering plan for Gobot's motion planning stack across [`modules/coal/`](../modules/coal/planning_world.hpp), [`python/gobot/planning/`](../python/gobot/planning/__init__.py), and [`src/gobot/editor/`](../src/gobot/editor/editor.cpp).

All work follows the repository architecture boundaries in [`AGENTS.md`](../AGENTS.md):

- `Scene` remains the authoring source of truth.
- `PlanningWorld` / `_coal_native` owns C++ kinematics, collision geometry, solvers, and geometric verification without depending on ImGui or OpenGL.
- `RenderBackend` renders debug visualization through explicit debug-draw primitives.
- `Editor` coordinates interactive planning gizmos, ghost previews, and panels without polluting user scene trees.

## Baseline (Before Phase 1)

Measured on Ryzen 7 3700X (8C/16T) + RTX 4070 (`cuda:0`) with strict continuous Hermite velocity, acceleration, convex, and original triangle-mesh verification:

- **Panda TrajOpt** (`16` waypoints, `7` DOF): `~233 ms` total (`216 ms` SQP across `13` OSQP calls = `16.6 ms/QP`, `14 ms` retime).
- **Container Unloading** (`14` waypoints, `6` DOF, `350` shape pairs, no via): `~655 ms` total (`458 ms` in `_qp_step` across `133` OSQP calls = `3.45 ms/QP`, `121 ms` collision eval, `42 ms` retime).
- **Container Batch `B=4`** (`dt=None` TOPP): `~650 ms` to `2.1 s` (dominated by OSQP + Python/SciPy sparse matrix setup and outlier QP counts).

Key bottleneck finding: **70–90% of total planning latency is in Python/SciPy CSC assembly + OSQP ADMM calls** (`3.5–16.6 ms` per 72–112 variable QP), while C++ FK + AABB + Coal + analytic Jacobians take only `~0.55 ms` per evaluation.

---

## Phase 1: Extreme Performance (Completed)

Goal: cut single-trajectory planning latency and scale batched planning to `B = 32, 64, 128` while keeping 100% of strict continuous Hermite collision, velocity, and acceleration guarantees.

### Phase 1 Measured Results (Ryzen 7 3700X + RTX 4070, Strict Continuous Hermite + Mesh Verification)
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

### 1.3 Fast Retiming & Swept Continuous Collision Checking
- Vectorize / eliminate redundant inner work in `retime_trajectory` and `_finalize_trajectory` so Hermite retiming takes `< 2 ms`.
- Add C++ Swept Convex Hull / conservative continuous collision screening option in `PlanningWorld` to catch thin-link grazing contacts with fewer discrete substeps and accelerate continuous verification.

### 1.4 Verification & Benchmarks
- Keep all existing regression tests passing and add dedicated C++ QP KKT optimality + C++/Python SQP equivalence tests.
- Refresh `examples/container_unloading/README.md` and `examples/panda_trajopt/README.md` with verified strict-Hermite benchmark tables.

---

## Phase 2: Industrial Planning Capabilities

Goal: cover the core workflows expected in industrial manipulation and assembly cells.

1. **Cartesian Path Planner (`MotionPlanner.plan_cartesian_path`)**
   - Linear and circular arc end-effector interpolation with configurable translation/rotation step sizes.
   - Continuity-preserving IK (minimum joint jump + singularity damping) + collision and joint-limit checks, returning a retimed `Trajectory`.
2. **`C²` Quintic Spline & Jerk-Bounded Retiming**
   - Extend `retime.py` with optional `jerk_limit` and time-optimal forward/backward velocity integration (TOPP-RA style) with exact analytic velocity, acceleration, and jerk bound verification.
3. **Constrained Global Planning (Manifold-Projected BiRRT)**
   - Extend C++ `PlanningWorld::plan_birrt` to project sampled states onto `PathConstraint` manifolds using existing analytic pose Jacobians (e.g., orientation-locked transport around complex obstacles).
4. **Incremental Scene Synchronization (`PlanningWorld.sync_from_scene`)**
   - Update obstacle and robot base transforms incrementally from a live Gobot `Scene` without rebuilding meshes or BVHs.

---

## Phase 3: Dynamic Obstacle Avoidance & Perception

Goal: enable millisecond-level obstacle avoidance against live depth/point-cloud sensors and moving obstacles.

1. **Voxel / 3D ESDF Obstacle Layer (CPU + MuJoCo Warp GPU)**
   - Add `set_occupancy_grid`, `set_point_cloud`, and `set_esdf` APIs on `PlanningWorld`.
   - Auto-generate sphere-cover approximations for robot links to evaluate ESDF signed distance and analytic trilinear gradients in microseconds (CPU) or batched on GPU (Warp).
2. **Warm-Start Reactive Replanning (`10–30 Hz`)**
   - Add `ReactivePlanner` / `replan(q_now, dq_now, prev_traj)` that shifts the previous solution as warm start while enforcing continuity of current position and velocity.
   - Ship a dynamic-obstacle avoidance example (moving obstacle crossing the workspace).

---

## Phase 4: Editor Interactive Motion Planning

Goal: provide an RViz/MoveIt-style interactive planning workflow inside `gobot_editor` while keeping clean `Scene / RenderBackend / Editor` boundaries.

1. **End-Effector Interactive Gizmo & Real-Time IK Ghost Preview**
   - Dragging a target gizmo on a selected `Robot3D` invokes C++ IK live and renders a translucent ghost preview colored by collision/reachability state.
2. **Motion Planning Editor Panel**
   - Select planning group, start/goal configurations, margins, velocity/acceleration scales, and debug overlays; trigger `Plan`, `Preview`, `Stop`, and `Re-Plan` asynchronously without blocking the viewport.
3. **Timeline Scrubbing & Physics Tracking Handoff**
   - Scrub planned trajectories frame-by-frame with live end-effector paths and collision witnesses, export reusable trajectories, and optionally execute them through `SimulationServer` / `JointController` in MuJoCo.
