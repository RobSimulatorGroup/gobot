# MJLab Terrain Visual And Physics Parity

This document records Gobot's MJLab-style terrain implementation. The goal is
to keep terrain authoring, rendering, serialization, and physics snapshots
aligned while preserving Gobot scene data as the source of truth.

## Goals

- `Terrain3D` represents generated terrain as normal scene data.
- Visual terrain colors follow MJLab-style box colors and heightfield HSV
  elevation colors instead of the old single yellow height ramp.
- Terrain physics is generated from the same `Terrain3D` data used for
  rendering.
- Custom mesh terrain tiles, such as `tilted_grid`, render in the editor and
  compile into static MuJoCo mesh geoms.
- Python terrain generation remains a normal Gobot Python workflow and saves
  to `.jscn`.

## Scene Data Model

The main scene type is `Terrain3D` in
`include/gobot/scene/terrain_3d.hpp`.

Terrain data is split into three primitive lanes:

- `TerrainBox`
  - `center`
  - `size`
  - `rotation_degrees`
  - `color`
- `TerrainHeightField`
  - `center`
  - `size`
  - `rows`
  - `cols`
  - `heights`
  - `normalized_elevation`
  - `base_thickness`
  - `z_offset`
- `TerrainMeshPatch`
  - `center`
  - `rotation_degrees`
  - `vertices`
  - `indices`
  - `color`

`TerrainColorMode` has three modes:

- `SurfaceColor`: render with the node surface tint only.
- `HeightRamp`: keep the legacy/debug height ramp path.
- `MjLab`: use authored box/mesh colors and MJLab-style heightfield HSV
  elevation colors.

Python terrain generation defaults to `TerrainColorMode.MjLab`.

## Rendering

`Terrain3D::GetRenderMesh()` builds one `ArrayMesh` from all terrain lanes:

- boxes become box faces with per-vertex face color copied from
  `TerrainBox::color`;
- heightfields become grid meshes with vertex colors computed from
  `normalized_elevation`;
- mesh patches use authored vertices/indices transformed by the patch local
  transform and colored by `TerrainMeshPatch::color`.

The MJLab heightfield color formula is:

```text
hue        = 0.5 - elevation * 0.45
saturation = 0.6 - elevation * 0.2
value      = 0.4 + elevation * 0.3
```

The scene mesh shader now uses a simple headlight-style model:

- ambient: `0.30`
- diffuse: `0.60`
- specular: disabled

This keeps authored vertex colors visually dominant. Normal `MeshInstance3D`
nodes without vertex colors continue to receive white fallback vertex colors in
the render upload path.

## Physics

`PhysicsTerrainSnapshot` mirrors the three terrain lanes:

- `boxes`
- `heightfields`
- `mesh_patches`

The snapshot path lives in `src/gobot/physics/physics_world.cpp`.

For MuJoCo:

- boxes become `mjGEOM_BOX`;
- heightfields become `mjsHField` assets plus `mjGEOM_HFIELD`;
- mesh patches become in-memory `mjsMesh` assets plus `mjGEOM_MESH`;
- all terrain geoms reuse the `Terrain3D` contact parameters:
  `friction`, `contype`, `conaffinity`, `condim`, `solref`, `solimp`,
  `margin`, and `gap`.

Heightfield physical placement uses:

- `normalized_elevation` as the MuJoCo hfield normalized data when available;
- `z_offset` as the geom z offset baseline;
- `base_thickness` as the MuJoCo hfield base thickness.

If `normalized_elevation` is absent, the MuJoCo backend falls back to
normalizing `heights`.

## Python Generator

The Python generator lives in `python/gobot/terrain.py`.

The exported preset surface includes:

- `flat`
- `pyramid_stairs`
- `pyramid_stairs_inv`
- `hf_pyramid_slope`
- `hf_pyramid_slope_inv`
- `random_rough`
- `wave_terrain`
- `discrete_obstacles`
- `perlin_noise`
- `box_random_grid`
- `random_spread_boxes`
- `open_stairs`
- `random_stairs`
- `stepping_stones`
- `narrow_beams`
- `nested_rings`
- `tilted_grid`

`ROUGH_TERRAINS_CFG` is the default mixed rough-terrain preset set used by the
training-oriented rough-terrain preset set.

`mjlab_showcase_terrains()` is the visual showcase preset set used by the Go1
terrain example. It intentionally mixes flat road tiles, blue upward stairs,
red inverted stairs, radial mounds, radial pits, bumps, rough fields, waves,
and obstacle heightfields to match the MJLab documentation-style terrain board.
Curriculum placement advances by tile index, not just by column, so a square
showcase grid can cover the full preset list.

`go1_training_terrains()` is the Go1-scaled training preset set. It keeps the
same MJLab color language and terrain variety, but caps hills, pits, stairs,
roughness, and obstacle heights to ranges a small Go1 model can plausibly learn
to traverse before harder curriculum is introduced.

Continuity rule:

- presets that visually represent a continuous ground surface should generate a
  single heightfield for the whole sub-terrain tile;
- slope heightfields should meet neighboring tiles at `z = 0` along their
  borders; positive slopes rise toward the center platform, and inverted slopes
  sink toward the center platform;
- example rough-terrain slopes intentionally use stronger gains than the helper
  default so the editor scene reads clearly at normal viewport scale;
- sparse custom mesh patches are allowed only when the terrain type is
  intentionally made of separate plates, such as `tilted_grid`;
- box terrain should keep the authored top surface at the requested height,
  including negative heights for pits or inverted stairs;
- example scenes should prefer square or near-square terrain grids over long
  strips so the result reads as a continuous mosaic rather than disconnected
  rows.

SciPy is listed as a Python dependency so generator resize/interpolation
semantics can match MJLab-style terrain generation more closely. A bilinear
NumPy fallback remains in the helper for environments where SciPy is not
available during local development.

## Example Scene

`examples/go1/terrain_scene.jscn` is regenerated from the Python terrain
generator with the Go1-scaled training preset. It intentionally uses
`TerrainColorMode.MjLab` and contains mixed box and heightfield terrain data.

`examples/go1/go1_scene.jscn` is not regenerated by the terrain update.

## Verification

Focused checks:

```bash
cmake --build build --target gobot -j2
./build/tests/test_terrain_3d
./build/tests/test_physics_server
ctest --test-dir build -R "test_python_terrain_generation_scene|test_python_bindings_smoke" --output-on-failure
```

Expected coverage:

- `Terrain3D` stores boxes, heightfields, mesh patches, colors, spawn origins.
- MJLab render mode emits vertex colors.
- `.jscn` round-trip preserves `MjLab`, box colors, heightfield
  `normalized_elevation`, `z_offset`, and mesh patch data.
- physics snapshots include mesh patches.
- MuJoCo builds terrain boxes, hfields, and mesh geoms when the backend is
  compiled.

## Boundaries

- Terrain remains scene data. The Python API and `.jscn` do not expose OpenGL
  RIDs, ImGui state, or raw MuJoCo pointers.
- MuJoCo-specific mesh/hfield asset creation stays below the physics backend
  boundary.
- Terrain visuals use vertex colors for this milestone; this does not introduce
  a general terrain texture/material sampling system.
