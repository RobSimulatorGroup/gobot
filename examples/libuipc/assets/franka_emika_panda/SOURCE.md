# Asset source

This directory contains the Franka Research 3 and Franka Hand asset used by
Newton's `python -m newton.examples brick_stacking` example.

- Upstream asset repository: `https://github.com/newton-physics/newton-assets`
- Pinned Newton asset revision: `261cd1f429619d8ef4f546bd788ab9dea906b5e1`
- Upstream package directory: `franka_emika_panda`

The demo imports `urdf/fr3_franka_hand.urdf` directly. Its DAE visual meshes,
STL arm/hand collision meshes, inertial data, joint limits, and finger box
collisions are used without geometry substitutes.

See `LICENSE` and `README.md` in this directory for upstream licensing and
provenance.
