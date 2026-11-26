# Reactive Replanning Demo (7-DOF Franka Panda, 20 Hz)

Closed-loop reactive replanning benchmark/demo using `MotionPlanner.replan(...)` against a moving obstacle in both **Coal convex/mesh** mode and **live 3D point-cloud ESDF** mode.

## What It Verifies

- **3-tier replanning hierarchy** on one pinned `WorldSnapshot`:
  1. `reused_prev`: exact Hermite suffix slicing + atomic bounds/CCD/manifold validation when the obstacle is clear and state matches within numerical rounding tolerances.
  2. `warm_start`: shifted warm-start SQP seeded from the remaining suffix with exact start-velocity (`q(0) = q_now, dq(0) = dq_now`, and optional quintic `ddq(0) = ddq_now`).
  3. `global_fallback`: time-budgeted BiRRT/CBiRRT fallback preserving the exact start boundary when the previous homotopy class is blocked.
- **Scheduled switch-time continuity**. Each simulated tick validates the exact Hermite interval that the controller will execute up to the next switch. The planner starts the new trajectory at that interval’s endpoint. A result arriving after the switch is discarded.
- **Snapshot `world_version` consistency**: checks `result.world_version == world.version` before committing and re-validates against the current snapshot if the world changed during planning.
- **Time budget**. The native planner rejects results completed after its deadline. An in-flight collision query or numerical solve can return control after the budget expires. The demo measures per-path (`reused_prev`, `warm_start`, `global_fallback`) plus full-tick (`perception + replan + handoff`) P50/P95 latency reporting.

## Run

```bash
.venv/bin/python examples/reactive_replanning/demo.py --mode both --hz 20 --spline-order 3
.venv/bin/python examples/reactive_replanning/demo.py --mode coal --hz 20 --spline-order 5
```
