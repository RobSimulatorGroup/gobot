# Native SuperDex package workcell

This example is a parcel-sorting station with a static manipulation table, a
separate outfeed conveyor, two floating LEAP Hands, three rigid cartons, two
closed-film mailers, and two tetrahedral fill bodies. SuperDex solves every
rigid, articulated, thin-shell, volumetric, and contact degree of freedom in
one native Gobot physics world.

The hands are the left and right Carnegie Mellon University LEAP Hand models
from Google DeepMind's MuJoCo Menagerie. Their source MJCF and meshes are
vendored unchanged under `assets/leap_hand`; `SOURCE.md` records the upstream
commit and license. Each hand retains all 16 finger joints. A hidden six-DOF
Cartesian wrist stage follows the demo trajectory without adding a visible
arm.

The scene has no `PhysicsCoupling` nodes or duplicate physics assets. Gobot's
SceneTree and `.jscn` remain the only authoring source. The Play and headless
scripts submit ordinary joint position targets and external-force commands to
`SimulationServer`; they never write runtime link poses or deformable vertices.

## Package models

The blue parcel is a 1,820-vertex closed thin shell with separate top and
bottom sheets, a sealed perimeter, asymmetric fullness, and deterministic
wrinkles. Its 0.30 kg film surrounds a 0.05 kg soft tetrahedral fill. The
yellow pouch uses a 2,072-vertex shell around a finer 1,620-vertex fill. Both
films use Neo-Hookean membrane behavior and authored shell bending stiffness.

At the start of a cycle the packages begin at their authored contact gap and settle on the
table. Both hands then contact and turn the blue mailer, push it toward the
belt, turn the yellow pouch, and turn the incoming small carton. Motion comes
from resolved contact: no package vertex is attached, teleported, or driven by
the trajectory.

The two soft-package turns use a table-supported sealed-edge pinch. Both palms
face down and descend from above. On each hand the index fingertip and thumb
oppose one another around the rear film edge. The middle and ring fingers bend
with the index finger so all three regular fingertips form one aligned jaw
opposite the thumb. The wrists keep that downward orientation and carry only
the pinched edge along a forward/lift arc over the opposite edge; no wrist
rotation or palm squeeze is used.

The conveyor collider stays fixed. Its prescribed `+X` surface velocity is
implemented through Gobot's backend-neutral force APIs:

- rigid packages receive a world-frame link force limited by measured normal
  contact and Coulomb friction;
- deformable packages receive per-node world-frame forces, and only nodes with
  measured belt contact can receive traction;
- the acceleration cap keeps the discrete velocity servo bounded.

## Build

SuperDex CPU is built by default. Initialize the pinned SDK fork for standalone
CMake builds (Python installation prepares the required submodules):

```bash
git submodule update --init --recursive 3rdparty/project_superdex
uv run cmake -S . -B build-superdex -G Ninja \
  -DGOB_BUILD_SUPERDEX=ON \
  -DGOB_BUILD_MUJOCO=OFF \
  -DGOB_BUILD_LIBUIPC=OFF
uv run cmake --build build-superdex -j
```

The default isolated build uses `GOB_SUPERDEX_SOURCE_DIR`; an already installed
core SDK can be selected with `GOB_SUPERDEX_ROOT`. The SDK precision is matched
to Gobot's `RealType`. `GOB_SUPERDEX_ENABLE_CUDA=ON` currently fails during
configuration because the pinned fork exports only the CPU solver; it never
silently creates a CPU world for a CUDA request.

## Editor Play Mode

Open the example from the installed Python environment and press Play:

```bash
uv run gobot_editor --path examples/conveyor_packages
```

After upgrading an older installation that omitted SuperDex, rebuild the
installed package, not just a separate standalone CMake build:

```bash
uv sync --reinstall-package gobot --no-build-isolation-package gobot \
  -C cmake.define.GOB_BUILD_SUPERDEX=ON
```

The script selects experimental `PhysicsBackendType.SuperDex`, CPU execution,
a 2 ms fixed step, one solver substep, and deformable contact-force output.
Use the Physics panel's contact-force option to show native contact arrows and
orange deformable belt-force resultants. Disable the option to clear them.

To run only the initial drop and settling phase:

```bash
GOBOT_CONVEYOR_DROP_ONLY=1 \
  uv run gobot_editor --path examples/conveyor_packages
```

The default `interactive` profile allows 16 Newton iterations.
`GOBOT_CONVEYOR_QUALITY=accurate` raises the cap to 48 iterations for dense
hand/film contact diagnostics; both profiles still exit early on convergence.

For a faster editor-only inspection, launch Play Mode explicitly in preview mode:

```bash
GOBOT_CONVEYOR_PREVIEW=1 uv run gobot_editor --path examples/conveyor_packages
```

The startup log reports `preview` or `validation`; plain `uv run gobot_editor`
uses validation mode. Preview mode changes only the runtime clone: it
uses 780 deformable nodes, disables thin-shell self-contact, caps Newton at
16 iterations with six line-search trials, and begins at the first hand approach. During soft-package
contact it pauses a command for up to three steps when measured penetration
exceeds 1 mm. The authored scene and headless validation profile retain the
full 6,292-node meshes and shell self-contact.

## Headless Run

For per-stage CPU timings over actual hand contact, run from the repository root:

```bash
uv run python benchmark/superdex_contact_benchmark.py \
  --output build/benchmarks/conveyor-contact-cpu
```

This keeps the full meshes and existing physics settings, waits for actual
bilateral hand contact, then records 100 warmup and 500 consecutive measured
physics ticks. Contact loss is reported rather than discarded. It requires the
profiling-enabled SDK fork; see [timing semantics and build instructions](../../doc/superdex_performance_diagnostics.md).
The normal runner can also opt into final-tick diagnostics with `--solver-timings`.

Regenerate the scene and run the same `SimulationServer` path without an editor:

```bash
uv run python examples/conveyor_packages/build_scene.py
uv run python examples/conveyor_packages/conveyor_packages_batch.py
```

The initial 300-step settling acceptance run is:

```bash
uv run python examples/conveyor_packages/conveyor_packages_batch.py \
  --steps 300 --quality interactive --phase-diagnostics
```

The JSON report includes initial/final parcel centers, final and maximum flip
angles, simultaneous left/right hand contact, pre-contact horizontal drift,
maximum penetration, force peaks, reset error, solver convergence, and
median/p95 step latency. Optional phase and force traces are available with
`--phase-diagnostics` and `--trace-force-flow`.

The first backend version supports one environment. `--num-envs` therefore
must remain `1`. `--execution cuda` is an explicit request: a CPU-only build or
machine without the required CUDA path fails instead of silently falling back.
CUDA remains experimental and reports `device_native=false` and
`graph_capture=false` because only the linear solve is intended for initial
acceleration.

## MuJoCo Warp + libuipc Grasp Acceptance

An opt-in headless trial reuses the existing two-way coupled provider. It is
separate from the default SuperDex Play session and currently **does not pass
physical grasp/flip acceptance**. Changing solvers alone does not make the
authored trajectory a successful grasp.

With a CUDA-capable MuJoCo Warp installation and Gobot's native libuipc module:

```bash
uv run --no-sync python examples/conveyor_packages/conveyor_mujoco_ipc.py \
  --report build/benchmarks/conveyor-ipc-acceptance/blue.json
```

The default contact controller requires 3,020 motion ticks plus up to 1,000
extra physics ticks waiting for a confirmed grasp. It stops early on acquisition
timeout or loss of grasp. `--controller open-loop` reproduces the original
3,020-tick trajectory without that safety gate.
It privately selects the blue film/fill, both hands, the worktable and the
stationary outfeed conveyor from this `.jscn`. It retains 2,600 soft nodes and
shell self-contact, adds explicit two-way couplings for all 34 hand collision
links, and uses one-way couplings for the stationary supports. No changes are
saved to the scene. There are no attachments, runtime vertex/pose writes,
conveyor traction, or extra package-driving forces.

`--compile-only` checks scene compilation and fingertip kinematics without
running CUDA. `--steps 2` checks the real GPU integration, but an incomplete
cycle always has `passed=false`. The process exits with code 1 on a physical
acceptance failure, incomplete trial, or solver error; inspect `failed_gates`,
`task_failure` and `error` separately. Compile-only success exits 0 but is not
physical success.

The original SuperDex pose leaves a 20.5 mm nominal gap for that solver's
contact radius. The contact controller aligns the thumb with all three regular
fingers in the air, descends toward a measured crest on the settled upper sheet,
then closes from a 50 mm opening toward a 2.5 mm unloaded gap. Grasp-point
tracking is limited to 50 mm/s and 60 mm total correction. Each hand stops
closing when it carries opposing loads; lifting requires both hands to maintain
at least 0.15 N on every fingertip for 100 ms, with proxy tracking within 1 mm
and fingertip forces below 20 N. Wrist rotation remains zero. These are
candidate joint commands, not a validated grasp; no film vertices are attached.
`--grasp-height-offset` and `--grasp-wait-steps` control acquisition placement
and timeout. `--controller open-loop --pinch-gap 0.0205` tests the original pose.

`--grip-feedback continuous` (default) adds bounded, independently filtered
force feedback for all four fingers after acquisition and throughout carrying.
Short contact loss pauses the trajectory while the fingers keep regulating;
20 ms sustained loss still fails. `--grip-feedback fixed` retains the previous
post-acquisition commands for comparison. The current candidate still loses
the grasp during lift; this is not yet a validated bag-flip controller.

The report includes per-finger reaction vectors, airborne pinch duration,
authored-face orientation, release/settling, proxy displacement bounds,
solver diagnostics, per-phase median/p95 step times, and sampled whole-device
memory use. See [acceptance criteria and results](../../doc/conveyor_ipc_acceptance.md)
for measurement limits and the next physical validation requirements. The
current runner does not validate cartons, multiple environments, or independent
batch resets.
The separate [coupling and 1/2/4/8 contact-stage benchmarks](../../doc/ipc_manipulation_benchmarks.md)
measure diagnostic workloads, not completed grasp throughput.

`--snapshot-dir build/benchmarks/conveyor-ipc-acceptance/states` additionally
saves measured vertices, velocities, hand/proxy transforms, wrenches and joint
positions as compressed NPZ files at the report trace boundaries.

The libuipc SDK must include the velocity-only state-update fix: older builds
discard friction history on every coupled step, even with friction enabled in
the scene. Rebuild/install both the native module and CUDA SDK after updating
the fork; the regression below tests device and host staging with and without
checkpoint rewind.

```bash
uv run --no-sync python -m pytest tests/python/test_conveyor_mujoco_ipc.py -q
GOBOT_RUN_CONVEYOR_IPC_GPU_TEST=1 uv run --no-sync python -m pytest \
  tests/python/test_conveyor_mujoco_ipc.py -q
GOBOT_RUN_LIBUIPC_BATCH_GPU_TEST=1 uv run --no-sync python -m pytest \
  tests/python/test_libuipc_batch_gpu.py -q -k velocity_staging
```
