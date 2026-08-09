# Gobot libuipc demos

These scenes exercise Gobot's native C++ libuipc module. They do not import
Warp, Newton, Torch, or `pyuipc`.

Open the project and press Play:

```bash
uv run gobot_editor --path examples/libuipc
```

The project contains one featured robot scene and two focused checks:

- `fr3_brick_grasp.jscn`: the Franka Research 3 and Franka Hand used by
  Newton's `python -m newton.examples brick_stacking` example. The checked-in
  upstream URDF provides the DAE visual meshes, the nine arm/hand STL collision
  meshes, inertial data, seven driven revolute joints, and two prismatic
  fingers with their eight box colliders. It closes around, lifts, returns, and
  releases a `50 x 30 x 25 mm` tetrahedral FEM box resting directly on one
  complete, uninterrupted tabletop. No collision proxy geometry is substituted.
- `soft_cube_stack.jscn`: two independent FEM bodies exercising soft-soft and
  soft-affine contact, based on libuipc's multi-FEM cases.
- `soft_cube_press.jscn`: a constrained affine press cyclically compressing a
  soft block, based on `28_fem_periodically_pressed_tet`.

The translucent green surface is the live deformable mesh synchronized from
libuipc. Orange-red arrows show the real per-vertex contact forces exported by
libuipc. The Physics panel's `Contact force arrows`, `Force scale`, and
`Max force length` controls are shared with the native physics backends and
apply live to these arrows. Display lengths use the same logarithmic mapping as
the native renderer; `provider.arrays["contact_forces"]` remains an unscaled,
read-only array in newtons. The panel reports `Provider: libuipc`; if it reports
another provider, this project is not the active simulation scene.

The single workbench collision box supports the FEM box directly. Together
with the official finger collision boxes and the provider's IPC activation
distance, resistance, and friction, it keeps a finite contact gap as the
fingers close and the arm lifts the box.

The robot asset is copied from Newton's pinned `newton-assets` revision
`261cd1f429619d8ef4f546bd788ab9dea906b5e1`. See
`assets/franka_emika_panda/SOURCE.md`, `README.md`, and `LICENSE`.

Validate a scene without starting CUDA:

```bash
python examples/libuipc/libuipc_demo.py \
  --scene examples/libuipc/fr3_brick_grasp.jscn
```

Run a short native CUDA smoke test with:

```bash
python examples/libuipc/libuipc_demo.py \
  --scene examples/libuipc/fr3_brick_grasp.jscn --steps 2
```
