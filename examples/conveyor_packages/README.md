# Newton package workcell

The default backend is **Newton VBD on CUDA**. Gobot compiles the authored
`conveyor_packages.jscn` into an owned physics snapshot; Newton solves the
LEAP hand joints, rigid cartons, closed film shells and tetrahedral contents
in one world. The same recipe runs in editor Play and from ordinary Python.
The reference sequence is pressing/releasing a soft parcel, pinching its edge,
lifting it onto a stopped conveyor, laying it flat and releasing it before
the belt starts. Two cartons
start 25 cm higher for flat and tilted drop trials.

The white and yellow mailers have rounded contents supporting both sides of a
closed film shell, a narrow raised weld, shallow corner folds and printed
shipping labels. The white pouch is about 8 cm thick after settling. Its softer
contents allow local fingertip compression while the film retains a filled
outline after release. These are authored scene meshes and materials, shared
by editor Play and video capture.

The yellow and white film materials use separate front/back UV panels, soft
crease normal maps and varying roughness. The 1024-by-512 atlases combine
short seam wrinkles with longer, curved folds across each face. Labels are printed only on the front
and use a matte finish. These maps change lighting, not collision geometry,
mass, stiffness or motion; the physical shell supplies the larger folds.
The scene includes its environment and key light so the same plastic finish
is visible in editor Play and captured video.

```bash
uv run gobot_editor --path examples/conveyor_packages

# Headless physics and measurement report
uv run python examples/conveyor_packages/conveyor_packages_batch.py \
  --backend newton --output /tmp/conveyor-newton.json

# Gobot-rendered MP4; --validate returns failure unless all acceptance checks pass
uv run python examples/conveyor_packages/conveyor_packages_batch.py \
  --backend newton --video /tmp/conveyor-newton.mp4 \
  --output /tmp/conveyor-newton.json --validate

# Short settling/carton-drop diagnostic
uv run python examples/conveyor_packages/conveyor_packages_batch.py \
  --backend newton --drop-only --output /tmp/conveyor-drop.json
```

Rebuild/install Gobot after changing the C++ snapshot or rendering code. The
project dependency currently pins Newton 1.6.0 and Warp 1.17.0. CUDA is required
by the default configuration; there is no silent CPU fallback. First startup
compiles Warp kernels and SDF contact assets. Subsequent runs reuse the cache.
Accuracy takes priority over real time: the default outer step is 2 ms with
8 solver substeps and 20 VBD iterations. `--substeps`, `--iterations`, `--device`,
`--steps`, `--fps`, `--width` and `--height` are explicit batch options.
After adding Python modules, refresh an existing editable installation with
`uv pip install --no-deps --no-build-isolation --reinstall -e .` from the repository
root. An old editable loader can otherwise mix source files with stale installed
copies; setting `PYTHONPATH` alone does not repair its module map.

Play builds and steps Newton on the simulation worker. Stop releases that
runtime; P requests a reset. Render snapshots update the Play clone only.
The normal scene render pass draws the deformed surfaces, so screenshots,
video and the editor consume the same geometry. Hidden contents remain hidden.

`conveyor_newton_runtime.py` owns the shared controller and runtime factory.
`conveyor_newton_batch.py` records measurements and video. Controllers submit
named hand-joint targets; they never prescribe a parcel pose, pin its vertices,
or add an attachment. Belt traction is applied only at measured contacts and
is limited by the contact normal load and friction. Soft parcels use the belt's
tangential velocity directly in Newton's implicit contact-friction solve,
including friction when the belt is stopped. The reported soft belt forces
are diagnostic resultants, not an extra force applied on the next step. Rigid
cartons retain explicit traction limited by their measured normal contact load.
The provider implementation
lives in `python/gobot/sim/providers/newton_vbd.py`; Newton-specific arrays and
contact details stay below the scene API.

The index and thumb pads enter above and below the raised front weld before
closing; their opposed faces support the film without piercing it. The white
film concentrates mesh rows near the sealed edges so a fingertip contacts
several rows instead of balancing on a few loose flange vertices. The hands
lift the pouch, hold it, and carry it 1.13 m over the stopped belt. They lower
until the bottom is supported, then move the held edge along a 30 cm radius
arc, lowering it 26 cm and drawing it back about 30 cm to lay the parcel on
its broad face. This preserves the supported span instead of compressing it
along a straight downward path. A supported pause
precedes opening the pads. The wrists then retreat and rise before extending
the remaining fingers. The belt accelerates to 0.35 m/s over 0.5 seconds after
this clearance and settling interval. No parcel coordinates are prescribed.

This avoids pushing the film against the hand's stepped knuckle housings,
which could store deformation and release it abruptly on withdrawal. The
housings use friction 0.4 while the grasp pads retain friction 3.0. The 3.8 m
conveyor gives the preceding cartons room to move ahead. Hand trajectories
use quintic interpolation with zero endpoint velocity and acceleration;
carrying takes 3 seconds, initial lowering 2 seconds and laying down 2.8 seconds.

The report samples compression/recovery, continuous lifted hold, maximum soft
contact penetration, reset error and wall time every 20 completed ticks and at
phase boundaries. Signed tetrahedron volume and peak rigid/soft penetration
are accumulated on every solver substep, so transient failures cannot disappear
between report samples. Local compression uses the mean height of the top-film
vertices under the six fingertip footprints, measured against their settled
height above the table. The broader central patch's 90th percentile compression
and each body's bounding height are reported separately. The full acceptance
thresholds are 10% local height compression, 80% local height recovery, the entire
film and its contents 10 cm above the table for 1 second and still held at
the end of the hold phase, at most
1 mm soft contact penetration, positive tetrahedron volumes and an exact
reset within 1 micrometre. Recovery also requires less than 2 cm horizontal
drift from the settled position, and shell/contents RMS speed below 3 cm/s
(maximum vertex speed 25 cm/s) over its final 0.2 seconds. A bag that launches
or keeps fluttering therefore cannot pass simply by regaining height.
The unattended yellow mailer is checked across the recording after the first
second: center drift must remain below 3 mm, height variation below 2 mm,
shell/contents RMS speed below 1 cm/s, RMS vertex displacement from rest below
2 mm and each vertex's displacement between samples below 1 mm. Instantaneous
maximum vertex speed is also reported, while acceptance bounds actual motion
to distinguish corner contact impulses from visible flutter.
At least two seconds of these samples are required; delayed oscillation cannot
be hidden by checking only the initial settling or the final frame.
A shortened run does not certify the whole motion.
The full run also requires the white parcel to reach the belt and travel at
least 20 cm along it during outfeed, rejecting a failed belt placement.
From the start of the supported pause through opening, hand withdrawal and
outfeed, the pouch center may rise at most 1.5 cm and its highest point at most
2.5 cm above the first supported-pause sample. The intentional vertical motion
during carrying and lowering is excluded from these supported-motion limits.
Its lowest point must stay within 5 mm of the belt/table surface. Instantaneous
RMS vertical vertex speed must stay below 15 cm/s. Between measurement samples,
no vertex may rise more than 5 mm, and RMS vertical displacement must stay below
2 mm. This catches a moving fold even when the overall bounds are falling.
The 0.2-second RMS is diagnostic only; averaging does not excuse a velocity spike.
The height and clearance checks apply to every report sample, so a transient
jump still fails even if the pouch lands flat in the final frame.
A detected solver failure is recorded in `failure` and makes the CLI exit with
an error; its partial measurements are retained. Drop-only runs check contact,
volume, completion and reset without claiming a successful grasp.

Rigid shape contacts, particle contacts, and rigid/film contacts use separate
damping. The stronger carton impact damping must not be averaged into the
gram-scale film contacts; a provider-local Newton adapter enforces that split.
The workcell also enables 20 ms of strain relaxation for the film and contents;
this damps changes of shape without resisting rigid translation. Joint
constraint damping reduces oscillation of the floating wrist stages. Static
table boxes retain vertex, edge and face contacts, including when a bag crosses
an edge. Opposing fingertip pads also collide with one another, preventing
the jaws from crossing through a closed film fold. The pinching poses were
fitted against the actual fingertip pad faces and palm collision geometry,
with non-overlapping jaws that open under compliant joint loads.
The yellow mailer's opposing film sheets have a 2.25 mm seam gap, clearing the
2 mm self-contact margin. Scaling the gap with its smaller bag height formerly
left only 1.95 mm, causing the weld to fight self-contact and develop motion
several seconds after settling. Its film mass remains 14 g and its initial
underside clearance remains 1 mm.
Its loaded underside uses shallower geometric wrinkles (at most 0.31 mm)
so the pre-shaped corrugations do not buckle against the table after settling.
Density is adjusted to retain the 14 g film mass; the visible face geometry
and the material's fine crease maps are retained.

The Newton adapter also makes edge-edge exclusion lists reciprocal before
capturing CUDA graphs. Newton 1.6's topological filter can treat the same pair
differently in the two query directions because bending adjacency contains
opposite hinge vertices as well as edge endpoints. Its edge-contact solver
evaluates each side separately, so an asymmetric filter can leave a force
without its reaction. Folded film could then start moving even with the belt
and hands stopped. The adapter preserves each exclusion in both directions;
non-neighbor self-contact and shell/contents contacts remain enabled. Bending
hinges also participate in particle coloring, including both opposite vertices.
The folded-film regression evaluates loaded internal contacts at a common pose
and checks both reciprocal pairs and zero resultant force. Before the repair,
that fixture produced a spurious resultant of about 3.7 N.

CUDA regression tests exercise flat and tilted carton drops, require their
post-impact rebound to stay below 1 cm and 2.5 cm respectively, and check that
a resting lightweight shell/contents pair loses motion energy. A moving/stopped
belt test checks transport distance, height and residual speed. An eight-second
yellow-mailer regression samples the full delayed-instability window. Full motion
recordings still need visual review for grasp slip and oscillation.

These checks establish motion stability, not exact energy conservation. VBD
solves an implicit-Euler variational objective, which is not the physical sum
of kinetic, gravitational and elastic energy. Friction and damping dissipate
energy, while hand drives and the moving belt can supply it. The isolated
yellow regression disables those drives and records kinetic energy throughout
eight seconds; after the first second it must stay below 1 microjoule. It does
not account for elastic/contact potential or motor work, so this is a delayed
motion-growth check rather than a complete energy budget.

Current provider scope is one environment with primitive/mesh rigid shapes,
fixed/prismatic/revolute joints, shells and tetrahedral solids. Unsupported
heightfields, couplings, sensors and kinematic deformables fail explicitly.
Newton 1.6 VBD does not impose exact actuator effort limits: this adapter
limits target slew and the requested PD drive impedance, but the implicit
constraint reaction can exceed that request. `exact_effort_limits` is therefore
false; reported drive efforts are estimates, not force-sensor measurements.

The native SuperDex implementation remains available with
`GOBOT_CONVEYOR_BACKEND=superdex` in Play or `--backend superdex` in the batch
entry point. Its settings and original motion sequence are documented below.

# SuperDex alternative

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

`conveyor_control.py` supplies the hand commands and belt forces for both entry
points. `conveyor_metrics.py` measures completed physics ticks and builds the
headless report. Scene definitions live in `conveyor_config.py`, motion timing
in `conveyor_profile.py`, and parcel mesh generation in `conveyor_meshes.py`.
`build_scene.py` composes and saves the scene; it attaches the Play script with
`root.set("script", "res://conveyor_packages_play.py")` before saving.

## Package models

The blue parcel is a 1,820-vertex closed thin shell with separate top and
bottom sheets, a sealed perimeter, asymmetric fullness, and deterministic
wrinkles, with white film and a barcode label. The continuous pillow curvature,
thin heat-sealed border and rounded contents give the parcel volume above and
below its weld. The contents support the shoulders, keeping the seam on the
side of the bag instead of letting a wide empty flange spread across the table.
Its 0.03 kg film surrounds a
0.32 kg soft tetrahedral fill. The yellow pouch uses a 2,072-vertex shell
(0.014 kg) around 0.151 kg of contents. Both fills use 189 vertices and rounded
edges with finite thickness to avoid sliver tetrahedra at the film seam. Both
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
GOBOT_CONVEYOR_BACKEND=superdex uv run gobot_editor --path examples/conveyor_packages
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
  GOBOT_CONVEYOR_BACKEND=superdex uv run gobot_editor --path examples/conveyor_packages
```

The default `interactive` profile allows 16 Newton iterations.
`GOBOT_CONVEYOR_QUALITY=accurate` raises the cap to 48 iterations for dense
hand/film contact diagnostics; both profiles still exit early on convergence.

For a faster editor-only inspection, launch Play Mode explicitly in preview mode:

```bash
GOBOT_CONVEYOR_PREVIEW=1 GOBOT_CONVEYOR_BACKEND=superdex uv run gobot_editor --path examples/conveyor_packages
```

The startup log reports `preview` or `validation`; plain `uv run gobot_editor`
uses validation mode. Preview mode changes only the runtime clone: it
uses 780 deformable nodes, disables thin-shell self-contact, caps Newton at
16 iterations with six line-search trials, and begins at the first hand approach. During soft-package
contact it pauses a command for up to three steps when measured penetration
exceeds 1 mm. The authored scene and headless validation profile retain the
full 4,270-node meshes and shell self-contact.

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
uv run python examples/conveyor_packages/conveyor_packages_batch.py --backend superdex
```

The initial 300-step settling acceptance run is:

```bash
uv run python examples/conveyor_packages/conveyor_packages_batch.py --backend superdex \
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
separate from the Newton default and the SuperDex alternative, and currently **does not pass
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
