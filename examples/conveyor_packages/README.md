# LEAP Hand deformable-package workcell

This example is a parcel-sorting station with a deep static manipulation table
and a separate outfeed conveyor. Parcels queue on the left and two suspended LEAP
Hands work over the center of the table. They first turn the blue mailer and
push it onto the conveyor, then turn the smaller yellow pouch, and finally turn
the incoming rigid carton. The belt carries the blue mailer to the right
through the scanner. There are no robot arms in the scene.

The hands are the left and right Carnegie Mellon University LEAP Hand models
from Google DeepMind's MuJoCo Menagerie. Their source MJCF and meshes are
vendored unchanged under `assets/leap_hand`; `SOURCE.md` records the upstream
commit and license. Each hand retains all 16 articulated finger joints. A hidden
six-DOF Cartesian wrist stage lets the hand remain suspended and follow the demo
trajectory without adding a visible arm.

All official palm and finger collision boxes are imported. Gobot adds a tight
tip box to each distal digit and exposes 17 one-way IPC proxies per hand. For
both soft packages, the palms face one another, establish a symmetric 15 mm
side preload, close both fingertip rows to 45%, and roll forward together while
following the advancing side seams. The carton uses 15% closure. During the
blue push, both wrists return palm-down and the fingers remain lightly curved.

## Thin-shell mailer

The blue parcel is a closed film shell rather than a softened solid. Its mesh
has 1,820 vertices and 3,636 triangles: separate top and bottom sheets, a sealed
perimeter, asymmetric fullness, and small deterministic wrinkles. libuipc solves
in-plane strain and shell bending, so the surface can crease and fold without
turning into a rounded block.

A hidden, very soft tetrahedral body represents the loose contents. It supports
the film through contact but is not attached to the hands or moved by the
controller. The shell weighs 0.30 kg and the fill weighs 0.05 kg. At the start
of every cycle they drop about 18 cm, deform under gravity, and settle onto the
rear half of the table before either hand moves. That 42 cm setback leaves one
package length for the physical forward turnover, so the reversed mailer stays
on the table for the separate push phase. The bottom support patch therefore
comes from weight and contact, not from an authored flat base.

The yellow pouch uses the same closed-film construction in a lower pillow
shape. Its 2,072-vertex, 4,140-triangle shell surrounds a finer 1,620-vertex,
7,392-tetrahedron soft fill. The fill is widest at mid-height and tapers toward
both film sheets, so all four sides bow outward instead of forming vertical
walls. It remains puffed while the underside flattens under its own weight and
the film wrinkles during the two-hand turn.

The manipulation is entirely contact driven:

1. Both hands approach the blue mailer's short sides, touch the seams with open
   fingers, and then apply a symmetric 15 mm palm preload while both fingertip
   rows close to 45%.
2. Both wrists move toward the outfeed and execute the same 180-degree forward
   roll around their fingertip rows. They open only after the reverse face has
   reached the table, then withdraw sideways while the shell settles.
3. Both wrists return palm-down, approach behind the blue mailer, and push it in
   `+Y` onto the outfeed.
4. The hands first rise vertically clear of the blue mailer and move upstream
   at clearance height. They descend outside the smaller, fuller yellow pouch,
   establish a symmetric 15 mm side preload, close both fingertip rows to 45%,
   and execute one synchronized 180-degree wrist roll. Both wrists translate
   180 mm toward the outfeed during the turn so their fingertips remain on the
   advancing seams instead of sliding off when the pouch passes vertical.
5. They again rise before traversing upstream and turn the rigid incoming
   carton with a measured 2 mm palm preload and 15% finger closure. The hands
   lower the reversed carton onto the table, open, withdraw sideways, and only
   then rise and return at clearance height.
6. Only after the blue mailer reaches the belt does the stationary collider's
   velocity field carry it in `+X` toward the scanner.

No deformable vertex is attached, teleported, or position-driven. Before belt
contact, the only package forces are gravity and resolved IPC contact. On the
belt, a device-resident Coulomb-limited external force models the prescribed
surface velocity, following Newton's `basic_conveyor_forces` approach without
moving the collision geometry.

MuJoCo Warp owns the floating hands, rigid cartons, and stationary workcell.
libuipc owns the shell and volumetric packages. `SolverCoupledProxy` stages hand
poses into libuipc and returns the resolved contact wrench. Hand proxies use
`OneWay` coupling because the wrists are position driven; the deformables still
receive the equal physical contact response.

## Editor Play Mode

Open the example and press Play:

```bash
uv run gobot_editor --path examples/conveyor_packages
```

Use the Physics panel's `Contact force arrows` flag to inspect contact. Turning
it off clears the arrows immediately and disables their CPU readback. In the
interactive profile, visible forces refresh every four physics steps:

- magenta: per-vertex IPC contact force above `0.001 N`
- cyan: horizontal IPC resultant on each deformable package
- orange: belt velocity-field force

To watch only the initial free drop and long-term settling:

```bash
GOBOT_CONVEYOR_DROP_ONLY=1 \
  uv run gobot_editor --path examples/conveyor_packages
```

The default `interactive` profile uses one fixed-relaxation coupling pass and
updates scene geometry every two physics steps. The stricter profile uses two
passes with Aitken relaxation and refreshes display data every step:

```bash
GOBOT_CONVEYOR_QUALITY=accurate \
  uv run gobot_editor --path examples/conveyor_packages
```

## Headless Run

Regenerate the scene and run one complete 2 ms fixed-step cycle:

```bash
uv run python examples/conveyor_packages/build_scene.py
uv run python examples/conveyor_packages/conveyor_packages_batch.py \
  --refresh-contact-forces
```

The JSON output reports separate final and maximum flip angles for the blue
mailer, yellow pouch, and small carton, plus rigid and soft-body displacement,
support-patch size, contact-force peaks, interface residual, actual CUDA graph
state, reset error, latency, and throughput. Add
`--trace-force-flow` to record per-step contact and external resultants; that
mode adds GPU reductions and is intended for physics diagnosis rather than
performance measurement. `--phase-diagnostics` records only the device-side
state at motion boundaries and is useful for calibrating multi-object contact
without introducing per-step CPU synchronization.

An in-tree libuipc solver module is discovered automatically. Set
`GOBOT_LIBUIPC_SOLVER_MODULE` or pass `--module-path` to select another build.
