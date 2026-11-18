# Panda TrajOpt with Coal

Open the workcell from the repository root and click **Play**.

```sh
.venv/bin/gobot_editor --path examples/panda_trajopt
```

The translucent pink arm follows the initial joint interpolation. The solid arm
plays the optimized trajectory on the same clock. The pink and cyan curves
trace the hand. **Pause** holds both poses at the same instant. **Stop**
restores the edited scene. Move a fixture or either goal marker and press
**Play** to replan. The Console reports which obstacles the initial trajectory
intersects and the optimized trajectory's sampled minimum world clearance.

The arm has 11 convex collision pieces for its base, joint housing, arm
shells, palm and separate fingertips. The nine movable fixture nodes use four
faceted hull meshes, with distinct sizes, orientations and placements. The
rectangular worktop also participates in collision checks. Each hull's visual
and collision node references the same mesh. Three mechanically connected
pairs are excluded from self-collision tests at their shared mounting joints.

The example solves both endpoint poses with CPU Coal IK, then optimizes 16
joint waypoints at 0.14 s spacing. The optimization requests 25 mm of
environment clearance. Before playback, each segment is divided into 24
intervals and checked against the original triangles for 20 mm of world
clearance and 2 mm of self clearance. Positions between those samples have
no continuous collision guarantee.

## Build

Initialize Coal, enable `GOB_BUILD_COAL`, and build the Python package and
editor with the project's usual CMake options. The worker imports the Gobot
package from the editor's Python environment. With an editable installation,
the build option can be supplied when reinstalling the project.

```sh
git submodule update --init --recursive 3rdparty/coal
CMAKE_ARGS="-DGOB_BUILD_COAL=ON" uv pip install --no-deps -e .
```

Coal uses CMake 3.22+, Eigen, Boost serialization/filesystem and Assimp
development packages. Python scripts select the backend with
`Planner(snapshot, robots, collision_backend="coal")`. The default planner
uses the Torch backend.

The Panda joint frames and limits are adapted from the Apache-2.0 Genesis
`panda_bullet/panda.urdf` at revision `ffd119c30118ebb9745fc009349efb2112a8cb76`.
The scene's hulls are authored for this example. The trajectory task builds
on [PyRoki's TrajOpt example](https://github.com/chungmin99/pyroki/blob/388e43e1fc0d0ee382968d3dd72970fd62a0450c/examples/07_trajopt.py).
