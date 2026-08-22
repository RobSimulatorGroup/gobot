# Rigid and deformable package conveyor

This example is an industrial parcel-sorting cell with three free rigid cartons,
one closed thin-film mailer, one tetrahedral pouch, and one integrated OpenArm
v2 bimanual robot. Incoming parcels queue on the left side of a wide static
sorting table. The robot sweeps the blue mailer across the table onto a
transverse front conveyor, then the conveyor carries it to the right-side
outfeed. The robot's operating surface is not itself a conveyor.

Each wrist carries a 1.50x Allegro Hand V3 visual: three articulated fingers and
an opposing thumb form a large, recognizable robotic hand. The source hand has
16 joints per side; this performance-oriented example compresses those poses
into the two existing OpenArm gripper synergy drives. The three fingers follow
one drive and the thumb follows the other. Physics remains intentionally
bounded: each side uses one horizontal palm pad, one short palm-heel sweep edge,
and eight visual-aligned finger collision proxies (two per digit). Each proxy
encloses its rendered mesh segments with a 6 mm contact margin, so the large
Allegro fingers cannot visibly enter a deformable before IPC contact activates.
Both visible palms and physical palm normals face down. The horizontal pad
supplies light pressure while the heel edge catches the mailer's rear wall, so
its forward motion comes from resolved IPC contact rather than a scripted
translation or hidden package force. Hand proxies collide with deformables but
not with the rigid sorting table, while soft packages collide with both table
and conveyor.

The modified OpenArm description and its required visual assets are vendored
under `assets/openarm_description`. They are derived from the official
`enactic/openarm_description` repository under Apache-2.0; `SOURCE.md` pins the
upstream commit and records the local geometry changes.

The left/right Allegro meshes and descriptions are vendored under
`assets/wonik_allegro` from the official Google DeepMind MuJoCo Menagerie model
under BSD-2-Clause. Its `SOURCE.md` pins the upstream commit.

The blue parcel is deliberately not a softened solid. Its visible skin is a
closed 532-vertex, 1,060-triangle film shell with separate top and bottom sheets,
a heat-sealed perimeter, shallow wrinkles, and asymmetric creases. libuipc
solves in-plane strain limiting and discrete shell bending, so the skin can fold
and wrinkle without acquiring the rounded underside of a solid pillow. A hidden
low-stiffness 252-vertex tetrahedral body represents the contents. The shell and
contents are independent IPC bodies: gravity supplies their weight and contact
transfers it between them and the table. The shell weighs about 0.10 kg and the
contents about 1.02 kg. The yellow pouch remains a 400-vertex, 1,512-tetrahedron
comparison body.

Every cycle begins with 300 physics ticks of isolated drop and settling. During
that interval the arms stay at their initial pose, grippers remain open, and the
belt velocity field is exactly zero. This makes the blue mailer fall roughly
20 cm, deform freely, and develop a flat support patch under its own weight
before manipulation begins. Mass-proportional vertex damping removes residual
bounce; it does not translate the parcel or target a position.

To keep the editor in this isolated drop/settle experiment indefinitely, use:

```bash
GOBOT_CONVEYOR_DROP_ONLY=1 \
  uv run gobot_editor --path examples/conveyor_packages
```

In this mode the arms and belt remain stationary after the initial 300 ticks;
the mailer continues settling under gravity until Play Mode is stopped.

Its drive model follows Newton v1.5.0's `basic_conveyor_forces` example:
collision geometry stays stationary while a prescribed surface-velocity field
carries the packages. Keeping the geometry fixed avoids travel limits and
continuously rebuilding or teleporting a belt collider.

MuJoCo Warp owns the frame and cartons. Per-contact normal forces are converted
into Coulomb-limited tangential carton forces that target the belt velocity on
the next step. The belt is authored with near-zero friction and configured as
normal-only in MuJoCo at runtime, so its static collision geometry supplies the
normal reaction while explicit velocity-field forces supply tangential
traction.
libuipc owns the soft packages. Because its public affine API
does not express a stationary surface velocity independently of pose, Gobot
binds a device-resident per-vertex external-force buffer to its deformable force
manager. Contacting package vertices receive the same Coulomb-limited velocity
field as the cartons. The belt remains a static `OneWay` proxy; no collider is
translated or teleported.
`SolverCoupledProxy` returns exact
soft-contact wrenches from each carton's `TwoWay` proxy to its MuJoCo free body.
The visible belt geometry stays fixed; only its thin markers move for feedback.

## Editor Play Mode

Open the project and press Play:

```bash
uv run gobot_editor --path examples/conveyor_packages
```

The raised, wide-set arms approach the blue mailer on the center of the static
table with open, downward-facing palms. They press and sweep it forward onto the
front belt. The fingers then open completely and pause while the belt ramps to
speed through real contact traction. Finally, the arms first move back and up to
clear the parcel, then retract fully while the package travels right through the
scanner. Press `P` or use the Physics panel Reset button to restart immediately.

The Physics panel's `Contact force arrows` flag controls deformable contact
visualization and CPU readback only. The GPU contact buffer remains enabled
because it drives the physical velocity field. The interactive profile refreshes
visible arrows every four physics steps. Turning the flag off clears them on the
next simulation callback. All vertices carrying at least `0.001 N` are shown.
Magenta arrows are the original per-vertex IPC contact forces. A cyan arrow
above each package shows its horizontal IPC resultant, making hand contact
and the table-to-belt sweep visible separately from the larger vertical support
forces. An orange
resultant shows the explicitly applied left-to-right belt velocity-field force.

The default is the low-latency `interactive` profile (`SolverCoupledProxy` x1).
Use the stricter x2/Aitken profile with:

```bash
GOBOT_CONVEYOR_QUALITY=accurate \
  uv run gobot_editor --path examples/conveyor_packages
```

An in-tree solver module is discovered automatically. It can also be selected
explicitly with `GOBOT_LIBUIPC_SOLVER_MODULE`.

## Headless Run

Regenerate the authored scene and run one complete GPU cycle:

```bash
uv run python examples/conveyor_packages/build_scene.py
uv run python examples/conveyor_packages/conveyor_packages_batch.py \
  --steps 1000 --refresh-contact-forces
```

To audit what moves a deformable package during every phase, add
`--trace-force-flow`. The trace reports positive and reverse contact peaks,
net cross-belt contact impulse, maximum forward displacement, the paired
arm-proxy reaction, belt external force, physics tick and phase, and package
deformation. This diagnostic launches a few extra GPU reductions per step and
therefore should not be used for latency measurements.

The JSON result reports rigid and soft package displacement, soft-package
height change, the number and span of vertices in the lowest 3 mm support patch,
final contact-force peak, interface residual, actual graph capture state, reset
error, and throughput. Contact-force readback remains off unless
`--refresh-contact-forces` is present. Use `--quality accurate` for x2/Aitken or
`--module-path` for a non-installed solver build.
