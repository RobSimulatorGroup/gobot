# OpenArm Assets

The arm and pinch-gripper meshes plus the generated bimanual URDF are from
[`enactic/openarm_description`](https://github.com/enactic/openarm_description)
at commit `1fba2cbc05001f05b4514120b70130b4ac06f409`.

The upstream files are licensed under Apache-2.0. See `LICENSE.txt` in this
directory. `openarm_v20_bimanual.urdf` is modified for this Gobot example:
the large body mesh is replaced with lightweight box geometry, and the two
fixed shoulder mounts are raised and spread apart for the conveyor workcell.
The bundled arm and gripper mesh files are unmodified.
