# Newton G1

This example reproduces Newton 1.4's `robot_policy --robot g1_29dof`
control contract while keeping Gobot responsible for scene loading and
rendering.

Open it with:

```sh
uv run gobot_editor --path examples/newton_g1
```

The project hook downloads and verifies the pinned official Newton assets on
first open. After verification, Gobot imports Newton's canonical
`g1_isaac.usd` into a cached `.jscn` containing one `Robot3D` with its
`Link3D`, `Joint3D`, collision, and render nodes. Large meshes are stored in a
sibling `.meshes/` binary cache so the JSON scene stays compact. That Gobot
scene is both the rendered asset and the source compiled for `NewtonProvider`;
Newton's viewer is not used. The compiled model retains all 43 policy position
drives used by the official demo.

The source is the official
[`newton-physics/newton-assets`](https://github.com/newton-physics/newton-assets)
repository at commit `261cd1f429619d8ef4f546bd788ab9dea906b5e1`.
The hook downloads only `g1_isaac.usd` plus the ONNX policy, YAML contract,
and license. The USD is the exact asset used by Newton 1.4's policy demo; its
collision model is intentionally different from the similarly named structured
USD. Files and the versioned generated scene are cached
under `examples/newton_g1/assets`; later opens do not access the network unless
a file is missing or fails its size and Git blob hash check. The scene cache is
rebuilt when Gobot's importer version changes.

Press Play, then use `I`/`K` for forward/backward, `J`/`L` for lateral motion,
`U`/`O` for yaw, and `P` to reset. Playback requires a CUDA-capable GPU.
