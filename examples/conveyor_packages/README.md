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

Initialize the pinned SDK fork and enable the optional backend explicitly:

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

Build Gobot with `GOB_BUILD_SUPERDEX=ON`, open the example, and press Play:

```bash
uv run gobot_editor --path examples/conveyor_packages
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

For a faster editor-only inspection, launch Play Mode with
`GOBOT_CONVEYOR_PREVIEW=1`. Preview mode changes only the runtime clone: it
uses 780 deformable nodes, disables thin-shell self-contact, caps Newton at
12 iterations, and begins at the first hand approach. During soft-package
contact it pauses a command for up to three steps when measured penetration
exceeds 1 mm. The authored scene and headless validation profile retain the
full 6,292-node meshes and shell self-contact.

## Headless Run

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
