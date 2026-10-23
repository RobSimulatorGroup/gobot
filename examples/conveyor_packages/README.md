# OpenArm v2 + Sharpa conveyor workcell

The default scene uses the bimanual OpenArm v2 model from
`enactic/openarm_mujoco`, with the existing Sharpa dexterous hands in place of
its native fingers. The original arm frames, joint axes, position/effort limits,
masses, inertia frames and visual geometry are preserved. Each arm has seven
active joints and each hand has 22. The right Sharpa is a locally mirrored
version of the supplied left hand.

The robot stands behind a 0.85 m sorting table. Its fixed arm mount is 1.29 m
above the floor, raised 12 cm from the first OpenArm preview so the arms are
less tightly folded. A fixed bracket extends the mount 26 cm over the rear
edge of the worktop. Incoming cartons wait on the right; the adjoining belt
runs left from the start. Table and belt are 1.00 m deep and share the same
physical surface height. The belt entrance is at X=-0.433 m, brought closer to
the compact arms. Yellow and white parcels sit at (-0.24, -0.06) and
(0.24, -0.10) m respectively.

The mailers retain their filled plastic shells, rounded deformable contents,
wrinkle/roughness maps and printed labels. Hands, arms, cartons, film and fill
are solved in the same Newton world. The controller only sends joint targets
and belt surface velocity: it never prescribes parcel poses, vertices or hand
attachments.

**Status:** this is a robot and motion integration example, not a validated
parcel-sorting policy. The previous ABLE/KUKA recordings do not validate
OpenArm. The OpenArm video is a short approach/contact preview; a complete
successful grasp, flip, release and two-parcel outfeed run remains unverified.
The lift is kept low to fit the replacement wrist's workspace. Task acceptance
continues to require actual flipped parcel faces and settled outfeed, so a
completed command sequence alone cannot pass it.

## Run

```bash
# Rebuild the authored scene and open it in the editor.
uv run python examples/conveyor_packages/build_scene.py --hand-model sharpa
uv run gobot_editor --path examples/conveyor_packages

# Record the commanded motion with Newton.
uv run python examples/conveyor_packages/conveyor_packages_batch.py \
  --backend newton --video build/conveyor-newton/conveyor-openarm-sharpa.mp4 \
  --output build/conveyor-newton/conveyor-openarm-sharpa.json
```

The default backend is Newton VBD on CUDA, pinned to Newton 1.6.0 and Warp
1.17.0. The outer step is 2 ms with eight substeps and 20 solver iterations.
Batch options include `--steps`, `--fps`, `--width`, `--height`, `--substeps`
and `--iterations`. `--validate` requires full task acceptance; a shortened
preview or an unsuccessful flip should fail that check. The first startup
compiles Warp kernels and mesh SDF assets; subsequent runs reuse the cache.

## Assets and motion

[OpenArm composition and source records](assets/openarm_sharpa/SOURCE.md) list
the pinned upstream revision, license, SHA-256 records, mounting fixture and
conversion details. [Sharpa provenance](assets/sharpa/SOURCE.md) documents the
hand inputs and mirroring. Rebuild the OpenArm composition offline with:

```bash
uv run python examples/conveyor_packages/assets/openarm_sharpa/prepare_assets.py
```

`build_scene.py` creates the editable `.jscn` authoring scene.
`conveyor_sharpa.py` builds the workcell and supplies URDF-based kinematics;
`conveyor_sharpa_grasp.py` defines contact points on the actual elastomer pads;
`conveyor_sharpa_motion.py` checks Cartesian paths against position limits,
palm direction, adjacent-sample joint continuity and arm/worktop clearance.
Near a wrist singularity it subdivides an interval before accepting either
arm's next target. It does not relax the guards. Faster intervals are retimed
to stay below the conservative 1.8 rad/s arm command limit.

`conveyor_sharpa_runtime.py` sends named joint targets and observes the white
parcel during the final side push. It bounds lateral corrections by the
current arm's reachable range without modifying the parcel. Reset clears
that feedback state. Wrist tracking is measured against the issued drive
targets after velocity limiting; palm direction comes from actual simulated
link orientations.

The waiting/return pose has downward-facing palms and relaxed fingers. The
pinch has a small index preload; a separate four-finger posture is used for
side pushes. Newton's compliant structural joints receive static gravity
feedforward computed from the URDF masses and centers of mass. Arm/finger
position gains are tuned for this Newton example, not copied from the upstream
MuJoCo actuators. Effort limits are approximate (`exact_effort_limits=false`).
The fixed mounting bracket's inertia is an explicit fixture assumption; the
source arm inertias remain unchanged.

Film remains deformable; the fill uses Poisson ratio 0.49 to resist collapse
under finger contacts. Belt friction is load-limited at measured contacts;
visual markers use the same completed simulation time. Mesh contacts use
Newton's mesh-owned SDFs. The report checks contact penetration, fill volumes,
tracking, palm direction, carton movement, parcel face orientation, outfeed
support/quietness and repeatable reset.

## Legacy LEAP scene

The previous stationary-hand drop and SuperDex sequences remain available:

```bash
uv run python examples/conveyor_packages/build_scene.py \
  --hand-model leap --output-dir build/conveyor-leap
uv run python examples/conveyor_packages/conveyor_packages_batch.py \
  --scene build/conveyor-leap/conveyor_packages.jscn --backend superdex
```

See [LEAP.md](LEAP.md) for that scene's configuration and validation notes.
