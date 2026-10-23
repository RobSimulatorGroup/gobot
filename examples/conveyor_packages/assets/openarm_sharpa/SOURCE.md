# OpenArm v2 with Sharpa hands

Upstream: https://github.com/enactic/openarm_mujoco/tree/master/v2

Pinned revision: `a3c8a57ec15d385e3c1dbd3fd090ee486b7f258c`.
The original MJCF, mesh inputs, license and per-file SHA-256 hashes are in
`source/`. OpenArm is licensed under Apache-2.0; see `source/LICENSE`.
Sharpa inputs and their MIT license are in `../sharpa/`.

`prepare_assets.py` converts this specific v2 arm model into the example's
URDF composition. It preserves each arm body's frame, joint name/axis/range,
mass, center of mass and inertia frame. Source visual scale, reflection,
geom transform and material colors are retained. Collision meshes become
closed convex OBJ hulls, with CAD vertices welded at 10 micrometres.
The seventh wrist motor and its housing remain; only the stock finger bodies
are removed. Each wrist receives the existing 50 mm Sharpa adapter at local
Z=-105 mm, rotated 180 degrees about Y and then clocked by +30/-30 degrees for the left/right adapters. The right Sharpa hand is the existing
local mirrored derivation, not a manufacturer-supplied right-hand model.

The source pedestal stays at its original metric scale. A fixed 593 mm riser
places the bimanual mount at 1.29 m, suitable for the 0.85 m worktop. This is an
authored stationary workstation, without the upstream Cell enclosure, lift,
cameras or native grippers. A 260 mm forward bracket brings the mount over the worktop; its assumed mass is 1.2 kg. The two arm bases retain their source spacing.

This example uses Newton position drives and gravity feedforward, not the
source MuJoCo actuators or contact solver. It retains source effort limits and
sets a conservative 1.8 rad/s command limit; Newton's effort enforcement is
approximate. Importing the robot does not establish real-hardware grasp
performance or a successful simulated parcel flip.

Rebuild offline:

```bash
.venv/bin/python examples/conveyor_packages/assets/openarm_sharpa/prepare_assets.py
.venv/bin/python examples/conveyor_packages/build_scene.py --hand-model sharpa
```
