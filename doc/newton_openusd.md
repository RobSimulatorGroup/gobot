# Newton And OpenUSD

Gobot keeps one authoring source of truth when OpenUSD and Newton are used
together:

```text
OpenUSD stage -> Gobot PackedScene / SceneTree -> compiled MJCF artifact
                                               -> NewtonProvider on CUDA
```

Newton does not reopen the project USD file. This keeps transforms, generated
terrain, robot names, and editor changes consistent with the other physics
providers, and avoids loading a second Python `usd-core` runtime into the same
process.

## Build OpenUSD Support

OpenUSD remains optional in a raw CMake build. Python source/editable builds and
official Linux x86-64 wheels enable it by default; wheels bundle the runtime.
The Python PEP 517 backend initializes the shallow checkouts at
`3rdparty/openusd` and `3rdparty/onetbb` and installs a minimal C++ OpenUSD SDK
below `build/openusd` automatically.

The sources are pinned Git submodules. Release jobs cache the compiled SDK and
package only the required runtime into wheels. Run the following workflow only
for a standalone CMake build or explicit SDK development:

```bash
git submodule update --init --depth=1 3rdparty/openusd 3rdparty/onetbb
cmake --workflow --preset dependencies-openusd

cmake -S . -B build/openusd-dev \
  -DGOB_BUILD_OPENUSD=ON \
  -DGOB_OPENUSD_ROOT="$PWD/build/openusd/install"
cmake --build build/openusd-dev -j
```

Configure the dependency-only build manually when source, build, install, or
parallelism settings need to be overridden. CMake and Ninja reuse unchanged
outputs instead of rebuilding them.

OpenUSD-enabled installs bundle the required shared libraries, oneTBB,
schema/plugin resources, and license files below `gobot/openusd`. Set
`-DGOB_BUNDLE_OPENUSD_RUNTIME=OFF` only when a system-managed OpenUSD runtime
is guaranteed on every target machine.

The first importer version supports `.usd`, `.usda`, and `.usdc` hierarchy,
stage units and up-axis conversion, local transforms, inherited visibility,
concave polygon triangulation, normals, `st` UVs, and constant
`UsdPreviewSurface` parameters. The importer uses `default` and `render`
purpose data and skips duplicate `proxy`/`guide` representations. For one
articulation it also maps rigid bodies, mass/inertia, fixed/revolute/prismatic
joints, limits, position drives, and box/sphere/cylinder/convex-mesh colliders
into backend-neutral Gobot nodes. Cameras, lights, animation, subdivision
evaluation, texture graphs, variants, payload UI, multi-articulation stages,
and advanced USD Physics schemas are not imported yet.

## Run Newton

`pip install gobot` on Linux x86-64 installs Newton and its compatible Warp and
MuJoCo-Warp runtime. Newton is a Python/Warp batch provider rather than a C++
`PhysicsBackendType`; Gobot compiles the active scene once and passes the
versioned artifact through the stable provider boundary:

```python
import gobot
import torch

app = gobot.app.create_context()
app.set_project_path("/path/to/project")
app.load_scene("res://world.jscn")
artifact = app.compile_scene_artifact(gobot.PhysicsBackendType.MuJoCoCpu)

with gobot.rl.NewtonProvider(
    artifact,
    num_envs=256,
    device="cuda:0",
    fixed_time_step=0.002,
) as provider:
    actions = torch.zeros_like(provider.arrays["ctrl"])
    arrays = provider.step(actions)
    joint_q = arrays["joint_q"]
    joint_qd = arrays["joint_qd"]
```

Use `resolve_robot_layout()` to translate stable Gobot robot, link, and joint
names into per-environment columns without exposing a Newton or MuJoCo model.
`set_joint_position_targets()` and `reset_robot_state()` then operate on that
layout. Select `use_mujoco_contacts=False` when the Newton collision pipeline
is required explicitly.

`NewtonModelConfig` provides explicit Newton-side joint-limit and contact
response overrides for policies that were trained with a known Newton model
profile. Its fields default to `None`, so ordinary providers preserve the
physics parameters compiled from the Gobot scene. The `newton_g1` example
uses the exact limit and contact defaults from Newton's released G1 policy
demo while keeping the imported Gobot scene as the model source of truth.
Contact overrides replace geom `solref` values; explicit MJCF contact pairs
remain authored scene data and take precedence.

The provider exposes stable CUDA Torch views for Newton-layout joint/body
state, controls, and reset masks. `joint_q` deliberately is not aliased as
MuJoCo `qpos`: Newton stores quaternion coordinates as `xyzw`, while MuJoCo
uses `wxyz`. It supports synchronous stepping and masked resets. Graph
capture and direct compilation from a backend-neutral physics snapshot are
follow-up work; the initial provider deliberately consumes Gobot's existing
validated MJCF artifact.

Gobot expands MuJoCo's legal one-value `solref` shorthand before Newton import.
Newton 1.4 otherwise fills the omitted second component with zero in its raw
custom-attribute path instead of MuJoCo's `1.0` default, which can make the
MuJoCo-Warp solve non-finite. Call `assert_finite()` and
`assert_no_overflow()` at evaluation or debug checkpoints.

Headless providers use the authored scene state and do not execute editor Play
scripts. A scene whose script computes a terrain-relative spawn pose must pass
that valid pose through `reset()` before the first simulation step.

`examples/newton_g1` is the end-to-end editor example: Gobot imports and
persists Newton's canonical `g1_isaac.usd` as one generated `.jscn` plus
compact binary mesh sidecars, compiles the physics artifact from that same
Gobot robot tree, runs the official ONNX policy with Warp-NN, and writes Newton
body transforms back to its `Link3D` nodes for Gobot rendering. The hook never introduces a
companion MJCF scene, and runtime playback never reopens the source USD.
