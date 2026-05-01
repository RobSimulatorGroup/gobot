# MuJoCo Backend Setup

Gobot keeps MuJoCo optional. The default build does not download or link MuJoCo,
so editor and tests can still build on machines without the SDK.

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

This path is intentionally opt-in because MuJoCo's own CMake build uses
`FetchContent` for dependencies. It may be slow or fail on restricted networks.

## Not Using A Submodule Yet

MuJoCo is not added as a default git submodule because most Gobot builds do not
need to compile a physics backend, and a source checkout still needs additional
network downloads for MuJoCo dependencies. Keeping MuJoCo as an optional package
or opt-in fetch keeps the base editor clone smaller and easier to build.

## Runtime Notes

When linking against a dynamic SDK library, make sure the MuJoCo shared library
is visible at runtime, for example:

```bash
export LD_LIBRARY_PATH=/path/to/mujoco/lib:$LD_LIBRARY_PATH
```

The Gobot MuJoCo backend currently loads the first `Robot3D.source_path` as a
MuJoCo XML model through `mj_loadXML`. URDF support therefore follows MuJoCo's
compiler behavior, and advanced Gobot scene-to-MJCF export remains a later step.

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
