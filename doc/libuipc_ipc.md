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

- consumes only `IpcSceneArtifact` schema v1;
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

The adapter supports tetrahedral deformables, box/sphere/triangle-mesh robot
collision geometry, and driven revolute/prismatic articulation. Additional collision
shapes, joint types and limits, tactile image products, and articulated force
feedback remain Gobot adapter work; they do not change the public scene
ownership boundary.

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
       -> MuJoCoIpcCoupler       (pose targets and reaction wrenches)
```

Collision ownership is fixed so a contact pair is never solved twice:

| Pair | Owner |
| --- | --- |
| rigid-rigid | MuJoCo |
| rigid-terrain | MuJoCo |
| deformable-deformable | libuipc |
| deformable-rigid | libuipc |
| deformable-terrain | libuipc |

One composite tick performs the following ordered operations:

1. gather mapped MuJoCo body poses into the stable libuipc target tensor;
2. advance every isolated libuipc subscene by the same fixed timestep;
3. recover a contact reaction wrench from each proxy pose error, including
   mass, center-of-mass, inertia, and gravity compensation;
4. replace only the Coupler-owned contribution in MuJoCo `xfrc_applied`;
5. advance MuJoCo Warp once.

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

Batch v1 intentionally supports full-batch reset only. Each shard restores its
frame-zero libuipc snapshot, including solver history and contact caches, and
each batch session uses an exclusive workspace that is removed when the session
closes. The composite provider does not claim CUDA graph capture, although its
MuJoCo Warp subsolver may use a captured graph. FEM positions/velocities and
affine proxy transforms are written directly into pre-bound CUDA tensors. The
pinned libuipc public API has a device `BufferView` output but no corresponding
affine-transform input, so the small rigid target table currently uses one
device-to-host-to-device staging operation per shard and tick. The raw batch
contact-force and affine-wrench buffers are reserved and zero in v1; the Coupler
applies the gravity-compensated proxy reaction wrench instead. Removing target
staging and exporting native device contact gradients are the next
ABI-compatible performance steps.

The opt-in two-environment regression covers both the native batch solver and
the complete MuJoCo Warp + libuipc Coupler path:

```bash
GOBOT_RUN_LIBUIPC_BATCH_GPU_TEST=1 \
ctest --test-dir build -R test_python_libuipc_batch_gpu --output-on-failure
```

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
