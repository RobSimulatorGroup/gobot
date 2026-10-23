# Sharpa hand and KUKA iiwa assets

Source: https://github.com/kushal2000/play2perfect/tree/main/assets/urdf/kuka_sharpa_description

Pinned revision: `ff2fc62873f6c6cd93164123c8358e9ce2d65c9b`.
The upstream repository supplies the included MIT license. Its URDF credits
IFL-CAMP/iiwa_stack for the iiwa model.

`source/` preserves the supplied URDF and meshes. `prepare_assets.py` creates
the left arm and a mirrored right counterpart. The latter is a local derivation,
not a separately supplied manufacturer right-hand model. Mirroring includes
mesh winding, joint axes/frames, centers of mass and inertia products. Joint
limits and masses are retained. Tactile pads are rendered dark rubber instead
of the upstream diagnostic green; arm housings use a neutral grey finish.

The empty, massless `iiwa14_link_ee` flange frame is folded into the following
fixed joint, preserving the hand transform without introducing a kinematic
child in the dynamic arm. Tiny upstream virtual-joint inertias are retained in
the asset; Newton applies its inertia validation when compiling the model.
The existing 50 g mount has an added cylindrical sleeve across its 50 mm
offset, with matching visual and collision geometry, so the wrist is connected.

Visual meshes retain the supplied geometry. Collision meshes are closed convex
hulls, with vertices welded at 10 micrometres and nearly collinear hull vertices
removed before rebuilding the hull. This removes CAD sliver triangles that the
physics snapshot compiler rejects. `prepare_assets.py` reproduces both URDFs
and these hulls; it does not remove individual faces and leave open surfaces.
