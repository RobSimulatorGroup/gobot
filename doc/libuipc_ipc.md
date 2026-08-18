# libuipc IPC Backend

Gobot builds its IPC contact solver from the pinned Apache-2.0 libuipc C++
sources in `3rdparty/libuipc`. The GPL-3.0 `pyuipc` bindings are not part of the
runtime dependency graph.

The integration keeps libuipc types private to `gobot_libuipc_solver`. Gobot's
scene compiler produces a backend-neutral IPC artifact, and the main engine
loads the solver through a versioned private C ABI. Installing or importing
Gobot never downloads libuipc or simulation assets.

## Source Build

libuipc is enabled by default. Initialize its pinned source and nested
dependencies:

```bash
git submodule update --init --recursive 3rdparty/libuipc
```

The pinned libuipc revision requires CMake 3.26 or newer. Gobot's integration is
validated with the CUDA 12.2 toolkit on Linux, so upgrading to CUDA 12.8 is not
required. The source build also needs system TBB and urdfdom development
packages (`libtbb-dev` and `liburdfdom-dev` on Ubuntu).

Gobot does not require or invoke vcpkg. It reuses its existing
Eigen/fmt/spdlog/JSON targets and obtains the remaining small C++ dependencies
at fixed Git commits through the same CPM source cache used by the rest of the
project. The pinned upstream CMake currently prints vcpkg-related status and
generates an unused `vcpkg.json` inside the build directory; no vcpkg executable
is called and no vcpkg installation prefix is created. Binary wheel
installation and Python import remain offline.

Configure Gobot normally:

```bash
cmake -S . -B build/libuipc -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DGOB_LIBUIPC_CUDA_ARCHITECTURES=native
cmake --build build/libuipc --target gobot_libuipc_solver --parallel
```

The first source configuration needs network access if the pinned CPM sources
are not already cached. Subsequent configurations reuse the cache. No package
manager executable or global dependency prefix is required.

Use `-DGOB_BUILD_LIBUIPC=OFF` for CPU-only CI or development machines without
the CUDA toolchain. That option removes the third-party solver module while
leaving the backend-neutral artifact compiler and module loader available.

## Runtime Boundary

The engine-facing API owns copied, stable state storage and exposes no libuipc
`World`, `Scene`, geometry, CUDA buffer, or solver handles. A solver session:

- consumes `IpcSceneArtifact` schema v5 and continues to read schemas v3/v4,
  normalizing the v3 missing static-collider table to empty; schema v5 adds an
  explicit rigid-system kind so standalone `RigidBody3D` records do not need a
  synthetic articulation wrapper;
- maps tetrahedral deformables to `StableNeoHookean` FEM;
- maps supported robot collision links to affine bodies, with soft transform
  targets for kinematic robots and fixed-root articulated dynamics for robots
  with joints;
- maps Gobot revolute, continuous, and prismatic joints to native libuipc
  affine-body constraints and absolute driving targets;
- solves all FEM/affine contact with libuipc IPC, CCD, and friction;
- retrieves body state and real per-vertex contact forces through libuipc
  feature interfaces;
- rejects an incompatible module ABI before creating a world.

The adapter supports tetrahedral deformables, box/sphere/capsule/cylinder/
triangle-mesh collision geometry, and driven revolute/prismatic articulation.
Loose `CollisionShape3D` nodes outside `Link3D` hierarchies compile as fixed IPC
colliders and never enter affine target or wrench arrays. `Terrain3D` remains
explicitly unsupported by the IPC compiler. Additional joint types and limits,
tactile image products, and articulated force feedback remain Gobot adapter
work; they do not change the public scene ownership boundary.

Schema v4 includes canonical `couplings` and `static_colliders` tables. Entries are sorted by coupling
node path and carry the coupling path, canonical `Link3D` path, robot/link
names, mode, per-binding force and torque scales, and a contiguous
`proxy_index`. In normal libuipc mode the solver still compiles the complete
robot. In `external_affine_proxies` mode it creates only the links named in
that table and preserves `proxy_index` order. There is no same-name link
discovery fallback.

Collision shapes and terrains use `PhysicsMaterial3D`, collision layer/mask,
and contact/rest offsets. The IPC compiler consumes only those neutral fields;
it does not read MuJoCo-specific friction or contact parameters.

The authored non-spatial `PhysicsCoupling` node exposes `enabled`,
`rigid_link_path`, `mode`, `force_scale`, and `torque_scale`. `OneWay` sends a
rigid pose to libuipc but never writes a reaction wrench. `TwoWay` additionally
writes proxy constraint feedback to MuJoCo. The referenced node must be a
`Link3D` under a `Robot3D` in the same compiled scene and must have at least one
enabled collision shape. Scales must be finite and non-negative, and two
enabled couplings cannot target the same link. Soft-body scope remains governed
by collision layer and mask.

## Python Provider

`gobot.ipc.LibuipcProvider` owns one native solver session and implements the
same Play lifecycle used by Gobot's other external providers. The first public
revision is intentionally single-environment and exposes full-scene reset,
affine transform and joint-position targets, stable read-only state arrays,
and explicit deformable/affine scene synchronization.

```python
from gobot.ipc import LibuipcConfig, LibuipcProvider

provider = LibuipcProvider.from_context(
    context,
    config=LibuipcConfig(fixed_time_step=0.01, device_index=0),
)
provider.step(nsteps=1)
positions = provider.arrays["positions"]
contact_forces = provider.arrays["contact_forces"]  # [vertex_count, 3], newtons
provider.set_joint_target("/World/Robot/shoulder_joint", 0.2)
provider.close()
```

`positions`, `velocities`, `contact_forces`, and `affine_transforms` are stable,
read-only NumPy arrays. `contact_forces` contains the physical libuipc contact
force accumulated at each deformable vertex, in newtons; it is not normalized
or display-scaled by the provider.

Tactile artifacts are rejected instead of being silently ignored. Tactile image
products remain native adapter work and are not currently exposed by
`gobot.ipc`.

## MuJoCo + libuipc Batch Co-simulation

`gobot.rl.MuJoCoIpcProvider` composes MuJoCo Warp rigid/articulation dynamics
with native libuipc FEM and IPC contact. The authored `.jscn` remains the only
source of truth. `CompiledMuJoCoIpcArtifact` stores the two compiled runtime
artifacts plus an explicit, validated mapping from each Gobot link path to its
MuJoCo runtime body and libuipc affine proxy.

```text
Gobot .jscn
    -> CompiledMuJoCoIpcArtifact
       -> MuJoCoWarpProvider     (rigid bodies and articulations)
       -> LibuipcBatchSolver     (FEM, soft contact, rigid proxies)
       -> SolverCoupledProxy     (pose/twist targets and reaction wrenches)
```

Collision ownership is fixed so a contact pair is never solved twice:

| Pair | Owner |
| --- | --- |
| rigid-rigid | MuJoCo |
| rigid-terrain | MuJoCo |
| deformable-deformable | libuipc |
| deformable-rigid | libuipc |
| deformable-static | libuipc |
| deformable-terrain | unsupported |

`SolverCoupledProxy` is the only composite integration path. At each shared
microstep it applies the current interface wrench, advances MuJoCo, transfers
the resulting body-origin pose/twist to libuipc, advances IPC, then harvests
and relaxes the new affine wrench. `OneWay` mappings transfer kinematics but
mask feedback; `TwoWay` mappings return the relaxed wrench to MuJoCo.

The global force and torque scales multiply each binding's scales. The Coupler
subtracts only its previous contribution from MuJoCo `xfrc_applied`, preserving
external forces written by callers. Native diagnostics identify the source as
`feedback_source=native_contact_wrench`; the explicit proxy pose-error fallback
reports `proxy_constraint`. A failed stage releases the Coupler-owned
wrench and faults the provider; another step is rejected until a successful
full reset, while `close()` remains available. Construction failure, reset,
and close also clear Coupler state.

Proxy coupling requires equal rigid/IPC microstep counts and equal microstep
`dt`. The x1 interactive path does not allocate or call rigid/IPC checkpoints.
For x2 and above, the Coupler captures preallocated MuJoCo state and one native
libuipc `World::dump()` checkpoint, then rewinds each additional iteration to
the same starting state. The rigid checkpoint stores only authoritative state;
MuJoCo `forward()` regenerates poses, transforms, subtree COM, and velocities
instead of copying those derived arrays. The final iterate is committed, so
`1`, `2`, or `4` coupling iterations still advance physical time by exactly one microstep.
Rigid `cvel` is converted from subtree-COM velocity to a body-origin,
world-frame `[linear_xyz, angular_xyz]` twist before upload. Fixed relaxation
and bounded Aitken relaxation are available; x1 defaults to fixed and x2+
defaults to Aitken. `OneWay` proxies receive both pose and twist but their feedback is always zero.
A failed capture, rewind, solve, or commit faults the composite provider until
a successful full reset.

The default capacity is 256 environments split into four fixed shards of 64.
Each shard owns one libuipc world and one subscene per environment. Cross-env
contact is disabled by subscene membership, and proxy-proxy contact is disabled
because MuJoCo owns rigid-rigid pairs. Storage addresses, body order, shard
count, and capacity stay fixed for the provider lifetime.

```python
from gobot.rl import (
    CompiledMuJoCoIpcArtifact,
    MuJoCoIpcConfig,
    MuJoCoIpcProvider,
)

artifact = CompiledMuJoCoIpcArtifact.from_context(context)
provider = MuJoCoIpcProvider(
    artifact,
    config=MuJoCoIpcConfig(
        num_envs=256,
        device="cuda:0",
        environments_per_shard=64,
    ),
)
provider.step(actions)
provider.reset(full_batch_mask)
```

The native batch C ABI is v4. In addition to target twists and single-slot
capture/rewind/commit operations, it exposes runtime output flags, explicit
output refresh, libuipc Newton/line-search/linear-tolerance settings, and
checkpoint/target/advance/reaction/state-sync phase timings. ABI v4 also
accepts the caller CUDA stream and reports device-native coupling/workspace
diagnostics. A v3 module is
rejected with an explicit version mismatch. The composite provider
intentionally supports full-batch reset only. Each
shard restores its frame-zero libuipc snapshot, including solver history and
contact caches, and each batch session uses an exclusive workspace that is
removed when the session closes. A partial mask raises an explicit unsupported
shard-recovery error. Solver microstep counts are explicit, positive integers;
`rigid_substeps` must equal `ipc_substeps`, and the two solvers must use the
same microstep `dt`. `SolverCoupledProxy` is the sole integration path: x1
advances without checkpoint storage, while x2 and above rewind both solvers
before each additional interface iteration.

`LibuipcBatchConfig(contact_constitution="al-ipc")` selects the experimental
AL-IPC contact pipeline and exposes its five native tuning parameters. The
pinned libuipc revision does not export AL contact gradients, so this mode
reports `exact_contact_wrench=false` and uses the existing affine
proxy-constraint reaction. It never silently falls back to standard IPC.
Standard IPC remains the default and retains direct contact/attachment wrench
feedback.

The composite provider reports `graph_capture=false` because native libuipc
shards execute outside graph capture. Its MuJoCo Warp subsolver may use its own
captured graph, and the Coupler separately captures graph-safe checkpoint,
rewind, rigid-kinematic gather, wrench relaxation/apply, and diagnostics-reduce
segments. `coupler_graph_captured` and its eager fallback reason are reported
independently. libuipc `World::advance()` and `World::sync()` remain required
host boundaries; this is not presented as one composite CUDA Graph.

FEM positions/velocities and affine proxy transforms are written into
pre-bound CUDA tensors. Affine target/twist staging and exact IPC contact-force
reduction now run device-to-device in persistent per-shard workspaces. ABI v4
accepts the caller CUDA stream, establishes the ordering needed by libuipc's
default-stream boundary, and reports workspace growth in diagnostics. Both
exchange paths and their phase timings are named in diagnostics and benchmark
output. Interface residual and Aitken coefficient stay in device scalars
during stepping; `.item()` occurs only when diagnostics are requested.

`LibuipcBatchConfig(export_deformable_contact_forces=False)` leaves the stable
device force buffer allocated but does not upload per-vertex contact forces on
each step. Exact affine contact wrenches are still exported every step. Call
`refresh_deformable_contact_forces()` to update the vertex buffer on demand;
diagnostics report the frame represented by that buffer. The default remains
`True` for compatibility and batch metrics.

Non-finite PCG residuals now invalidate the libuipc world and return a C ABI
error to the caller. They no longer abort the editor or training process, so
diagnostics remain available for identifying the failed frame and solve phase.

`export_deformable_state` and `export_affine_state` similarly control automatic
state export. `refresh_state()` updates both buffers without advancing physics.
The interactive editor profile uses lazy state export, while accurate and batch
profiles retain immediate output by default.

The opt-in GPU regression covers native checkpoint replay, loose static
collision, checkpoint-free SolverCoupledProxy x1, x2 rollback/replay, eager and
captured tensor exchange, and immediate/lazy contact-force output:

```bash
GOBOT_RUN_LIBUIPC_BATCH_GPU_TEST=1 \
ctest --test-dir build -R test_python_libuipc_batch_gpu --output-on-failure
```

The soft-press Play example compiles one artifact, then creates three
display-only runtime scene copies. Four environments are shown in a 2x2 grid
with different commands. The viewport callback performs one batched rigid pose
readback and one batched soft-vertex readback; headless training does neither.
Use `benchmark/mujoco_libuipc_batch_benchmark.py` to record warmup/repeated
median results. The following correctness matrix records 1- and 64-environment
curves for 1/2/4 Newton iterations and compares standard IPC with AL-IPC from
the same authored scene and controls:

```bash
uv run python benchmark/mujoco_libuipc_batch_benchmark.py \
  --environment-counts 1 64 \
  --coupling-iterations 1 2 4 \
  --contact-constitutions ipc al-ipc
```

Results include median step time, interface residual, Aitken coefficient,
ground-penetration proxy, and press wrench. The report marks AL-IPC eligible
only when median IPC step latency improves by at least 15 percent and the
configured physical bounds pass. This report does not change either default or
label AL-IPC as recommended in project documentation.

The dual-arm rope target has a separate same-process benchmark for the editor
quality profiles and Coupler graph toggle:

```bash
uv run python benchmark/dual_arm_rope_twist_benchmark.py \
  --environment-counts 1 64 \
  --module-path build/<matching-build>/python/gobot/libgobot_libuipc_solver.so
```

Add `--profiles interactive-balanced interactive-fast accurate` to rerun the
`12/6/2e-3` and `8/4/5e-3` admission candidates. When neither passes the
residual, attachment, slip, wrench, and exact-contact gates, the benchmark
selects the interactive profile's `16/8/1e-3` fallback.

The full interactive-cycle soak is a separate opt-in test because it advances
one environment for 40,000 physics steps. It uses the finite-torque task,
continues with zero wrist command after the normal safety hold, and does not
use the unbounded constant-speed showcase:

```bash
GOBOT_RUN_DUAL_ARM_ROPE_SOAK_GPU_TEST=1 \
GOBOT_LIBUIPC_TEST_MODULE_PATH=build/<matching-build>/python/gobot/libgobot_libuipc_solver.so \
uv run python tests/python/test_dual_arm_rope_twist_soak_gpu.py
```

It checks stable storage, graph capture, residual, grasp/attachment drift,
OneWay box/static-table rope-vertex penetration, exact wrench transfer, and a
five-second single-step runaway guard.

Shard-masked reset, full composite CUDA Graph capture, and coupled
sensor/randomization pipelines remain future work. None of those
extensions require changing `.jscn` as the authored source of truth.

## Examples

Open the native libuipc project and press Play:

```bash
uv run gobot_editor --path examples/libuipc
```

The default project scene uses the Franka Research 3 and Franka Hand asset from
Newton's `brick_stacking` example to grasp and lift a soft workpiece. The
checked-in upstream URDF provides the DAE visual meshes, nine arm/hand STL
collision meshes, inertial data, seven driven revolute joints, and two
prismatic finger joints with eight box colliders. No collision proxy geometry
is substituted. A `50 x 30 x 25 mm` tetrahedral FEM box rests directly on one
complete, uninterrupted tabletop; there is no retractable support beneath it.

Orange-red arrows visualize the real libuipc contact force accumulated at each
FEM vertex. The Physics panel's contact-force checkbox, scale, and maximum
length are the common controls for native backends and libuipc. Their displayed
lengths are logarithmically scaled and capped for readability, while
`provider.arrays["contact_forces"]` always retains the unscaled force vectors in
newtons. The native module combines the authored collider friction with the
provider friction coefficient, preserving a finite IPC contact gap during the
close and lift phases.

This is intentionally different from
`libuipc-samples/examples/86_panda_hydro_traj_cubes`: that upstream diagnostic
uses ten equal-sized cubes, disables gravity and contact, and replays only two
target rows. It is useful for testing affine joints but is not a robot geometry
or grasping example. Soft-soft contact and affine-press cases are generated as
test-only fixtures, while the user-facing project stays focused on the FR3
workflow. The scene reports `Provider: libuipc` in the Physics panel and can be
validated or stepped headlessly through `libuipc_demo.py`. The obsolete Warp
IPC Python provider and `examples/warp_ipc` project have been removed.

The real CUDA admission test is opt-in:

```bash
GOBOT_RUN_LIBUIPC_GPU_TEST=1 \
ctest --test-dir build -R test_python_libuipc_gpu --output-on-failure
```
