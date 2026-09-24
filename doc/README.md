# Documentation

Start with the [project README](../README.md) to install Gobot and open the editor.
Commands in these guides run from the repository root unless stated otherwise.

## Guides

| Topic | Guide |
| --- | --- |
| Source builds, native rebuilds, tests, wheels | [Build and test](building.md) |
| Example projects and editor discovery | [Examples](examples.md) |
| Training, evaluation, policy playback | [Go1](../examples/go1/README.md) |
| Engine ownership and API contracts | [Architecture](architecture.md) |
| CPU and CUDA training contracts and roadmap | [MuJoCo RL](mujoco_rl_plan.md) |
| MuJoCo SDK configuration | [MuJoCo setup](physics_mujoco_setup.md) |
| Native CPU batch integration | [mjbatch](mjbatch_cpu.md) |
| Import and runtime parity | [MJCF equivalence](mjcf_equivalence.md) |
| USD import and Newton provider | [Newton and OpenUSD](newton_openusd.md) |
| CUDA viewport and camera outputs | [Luisa rendering](luisa_rendering_plan.md) |
| 3D Gaussian environment assets | [Gaussian Splatting](gaussian_splatting.md) |
| Deformables and coupled simulation | [libuipc IPC](libuipc_ipc.md) |
| SuperDex timing and diagnostics | [SuperDex profiling](superdex_performance_diagnostics.md) |

## Development records

These documents preserve design decisions and measurements from specific
revisions. Use the guides above for current setup and API boundaries.

- [Architecture design notes](architecture_design_notes.md)
- [Physics runtime migration](physics_runtime_migration.md) and [follow-up](physics_runtime_followup.md)
- [Physics dependency updates](physics_dependency_updates.md)
- [Runtime performance work](runtime_performance_work.md)
- [Conveyor IPC acceptance](conveyor_ipc_acceptance.md)
- [Manipulation coupling benchmarks](ipc_manipulation_benchmarks.md)
