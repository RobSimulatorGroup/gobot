# MuJoCo Backend Setup

Gobot keeps MuJoCo build-time configurable. The standard source and wheel build
enables MuJoCo CPU and may fetch the pinned release when no local package is
available. A minimal editor-only build can disable it explicitly:

```bash
cmake -S . -B build -DGOB_BUILD_MUJOCO=OFF
```

## Recommended: Local MuJoCo SDK

Download a MuJoCo release from the official `google-deepmind/mujoco` GitHub
releases page and extract it somewhere outside the Gobot source tree.

Configure Gobot with the extracted SDK root:

```bash
cmake -S . -B build \
  -DGOB_BUILD_MUJOCO=ON \
  -DGOB_MUJOCO_ROOT=/path/to/mujoco
```

`GOB_MUJOCO_ROOT` may point either to a CMake install prefix containing
`mujocoConfig.cmake`, or to the extracted SDK layout containing `include/` and
`lib/` or `bin/`.

If MuJoCo was installed into a normal CMake prefix, this also works:

```bash
cmake -S . -B build \
  -DGOB_BUILD_MUJOCO=ON \
  -DCMAKE_PREFIX_PATH=/path/to/mujoco
```

## Optional: Fetch MuJoCo During Configure

For a one-command source build, allow CMake to fetch the pinned official release:

```bash
cmake -S . -B build \
  -DGOB_BUILD_MUJOCO=ON \
  -DGOB_FETCH_MUJOCO=ON
```

The fetched release is controlled by:

```bash
-DGOB_MUJOCO_GIT_TAG=3.8.0
```

Set `-DGOB_FETCH_MUJOCO=OFF` when builds must remain offline and provide a local
SDK/package instead. MuJoCo's own CMake build uses `FetchContent` for some
dependencies, so first-time fetched builds can be slow on restricted networks.

## Not Using A Submodule Yet

MuJoCo is not added as a git submodule. Gobot first looks for a configured or
system package and otherwise uses the configurable fetch path. This keeps one
dependency source of truth and lets offline builds require a local SDK.

## Runtime Notes

When linking against a dynamic SDK library, make sure the MuJoCo shared library
is visible at runtime, for example:

```bash
export LD_LIBRARY_PATH=/path/to/mujoco/lib:$LD_LIBRARY_PATH
```

The normal authored path builds a MuJoCo model from Gobot scene data in `.jscn`:
robots, links, joints, collision shapes, contact parameters, and actuator
settings. `Robot3D.source_path` is retained only as a record of where an import
came from. Physics backends never load that file at
runtime; imported scenes must contain the authored links, joints, collision
shapes, sensors, and actuator settings needed to compile the model.

For the importer/runtime equivalence goals, see `doc/mjcf_equivalence.md`.

## Editor Smoke Test

The first editor-level check is that MuJoCo is actually selected and stepping:

1. Load or import a robot scene.
2. Open the `Physics` panel.
3. Select `MuJoCo CPU`.
4. Check that the backend reports `Available`.
5. Click `Build World`.
6. Click `Step` and watch the frame counter and joint state table.

If the frame counter advances and joint positions/velocities are populated, the
Gobot editor is creating a MuJoCo world and calling `mj_step`.

Visible falling/contact behavior needs extra scene conditions:

- The MuJoCo model must contain a ground plane or floor geom.
- The robot root must be free/floating if gravity should move the base.
- A fixed-base robot with no control targets may look unchanged even while
  MuJoCo is stepping correctly.
