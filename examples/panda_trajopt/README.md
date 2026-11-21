# Panda inspection with trust-region TrajOpt

Open the workcell from the repository root and click **Play**.

```sh
.venv/bin/gobot_editor --path examples/panda_trajopt
```

The translucent pink arm follows the initial joint interpolation. The solid arm
plays the optimized trajectory on the same clock. The pink and cyan curves
trace the hand. **Pause** holds both poses at the same instant. **Stop**
restores the edited scene. Move **Start**, **Goal**, the yellow **Inspection**
marker or a fixture and press **Play** to replan. At waypoint 8 (zero-based)
the hand must reach the marker inside the blue inspection gauge, pointing down
and turned 25 degrees about world Z from its endpoint orientation. The Console
reports the pose error, the obstacles hit by the seed and the minimum sampled
environment clearance along the optimized trajectory.

The arm has 11 convex collision pieces for its base, joint housing, arm
shells, palm and separate fingertips. The nine movable fixture nodes use four
faceted hull meshes, with distinct sizes, orientations and placements. The
worktop, gauge column and both jaws also participate in collision checks. Each hull's visual
and collision node references the same mesh. Three mechanically connected
pairs are excluded from self-collision tests at their shared mounting joints.

## Planning problem

The example solves endpoint IK with Coal, then optimizes 16 joint waypoints
together with fixed endpoints and 0.14 s between waypoints. The intermediate
hand position and orientation are constraints of that same solve. The
straight joint seed intersects three fixtures and the lower gauge jaw.

Gobot's CPU `solver="sqp"` follows the trust-region method in Tesseract
TrajOpt. Each iteration uses Torch FK and collision Jacobians, solves a convex
QP with OSQP, and compares predicted and actual L1 merit improvement before
accepting a step. The fixed-time quadratic costs use first and second joint differences
with zero velocity and acceleration targets. Callers select intermediate Cartesian constraints with
`waypoint_targets={index: PoseTarget(...)}`. The default solver is `"lm"`.
[Upstream SQP](https://github.com/tesseract-robotics/trajopt/blob/4664f89f3157113998e7c02ac2fefc45469a21c4/trajopt_sco/src/optimizers.cpp#L780-L915)
and [Cartesian constraint example](https://github.com/tesseract-robotics/trajopt/blob/4664f89f3157113998e7c02ac2fefc45469a21c4/trajopt/test/cart_position_optimization_unit.cpp#L101-L132).

The optimizer requests 25 mm of world clearance. It samples every segment
at least three times and refines its grid to limit the joint-space distance
between samples to 0.075 rad for this revolute arm. Before playback it checks
the original geometry at 361 poses for at least 20 mm of environment clearance
and 2 mm of self clearance. The reported clearance is the minimum sampled
convex distance. General mesh motion between samples has no continuous
collision guarantee. Tesseract also offers cast-based continuous checks.
[Upstream continuous evaluator](https://github.com/tesseract-robotics/trajopt/blob/4664f89f3157113998e7c02ac2fefc45469a21c4/trajopt/src/collision_terms.cpp#L1092-L1138).

## Build

Initialize Coal, enable `GOB_BUILD_COAL`, and build the Python package and
editor with the project's usual CMake options. The worker imports the Gobot
package from the editor's Python environment. With an editable installation,
the build option can be supplied when reinstalling the project.

```sh
git submodule update --init --recursive 3rdparty/coal
CMAKE_ARGS="-DGOB_BUILD_COAL=ON" uv sync --extra planning-sqp --reinstall-package gobot
```

Coal uses CMake 3.22+, Eigen, Boost serialization/filesystem and Assimp
development packages. Python scripts select the backend with
`Planner(snapshot, robots, collision_backend="coal")`. The default planner
uses the Torch backend.

The Panda joint frames and limits are adapted from the Apache-2.0 Genesis
`panda_bullet/panda.urdf` at revision `ffd119c30118ebb9745fc009349efb2112a8cb76`.
The scene's hulls are authored for this example. The trajectory task builds
on [PyRoki's TrajOpt example](https://github.com/chungmin99/pyroki/blob/388e43e1fc0d0ee382968d3dd72970fd62a0450c/examples/07_trajopt.py).
