# LEAP Hand deformable-package workcell

This example is a parcel-sorting station with a static manipulation table and a
separate outfeed conveyor. Parcels queue on the left, two suspended LEAP Hands
work over the center of the table, and the selected parcel is pushed forward
onto the conveyor. The belt then carries it to the right through the scanner.
There are no robot arms in the scene.

The hands are the left and right Carnegie Mellon University LEAP Hand models
from Google DeepMind's MuJoCo Menagerie. Their source MJCF and meshes are
vendored unchanged under `assets/leap_hand`; `SOURCE.md` records the upstream
commit and license. Each hand retains all 16 articulated finger joints. A hidden
six-DOF Cartesian wrist stage lets the hand remain suspended and follow the demo
trajectory without adding a visible arm.

All official palm and finger collision boxes are imported. Gobot adds a tight
tip box to each distal digit and exposes 17 one-way IPC proxies per hand. During
the flip, the palms face one another and the three regular fingers curl upward
from the lower side of the mailer. The palms then close another 10 mm per side,
forming a repeatable physical envelope around the package. During the push, the
wrists return palm-down and the fingers remain lightly curved.

## Thin-shell mailer

The blue parcel is a closed film shell rather than a softened solid. Its mesh
has 1,080 vertices and 2,156 triangles: separate top and bottom sheets, a sealed
perimeter, asymmetric fullness, and small deterministic wrinkles. libuipc solves
in-plane strain and shell bending, so the surface can crease and fold without
turning into a rounded block.

A hidden, very soft tetrahedral body represents the loose contents. It supports
the film through contact but is not attached to the hands or moved by the
controller. The shell weighs 0.30 kg and the fill weighs 0.05 kg. At the start
of every cycle they drop about 18 cm, deform under gravity, and settle onto the
table before either hand moves. The bottom support patch therefore comes from
weight and contact, not from an authored flat base.

The manipulation is entirely contact driven:

1. Both hands approach the short sides, curl three fingers under the lower edge,
   and apply the 10 mm palm preload.
2. They lift the package clear of the table and roll 180 degrees around a fixed
   fingertip pivot.
3. The fingers open, the hands withdraw, and gravity finishes laying the reverse
   face onto the table.
4. Both wrists return palm-down, approach behind the package, and push it in
   `+Y` onto the outfeed.
5. Only after the package reaches the belt does the stationary collider's
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
  --steps 2980 --refresh-contact-forces
```

The JSON output reports final and maximum mailer flip angle, rigid and soft-body
displacement, support-patch size, contact-force peaks, interface residual,
actual CUDA graph state, reset error, latency, and throughput. Add
`--trace-force-flow` to record per-step contact and external resultants; that
mode adds GPU reductions and is intended for physics diagnosis rather than
performance measurement.

An in-tree libuipc solver module is discovered automatically. Set
`GOBOT_LIBUIPC_SOLVER_MODULE` or pass `--module-path` to select another build.
