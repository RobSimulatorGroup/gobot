# Gobot Warp IPC examples

These examples author Gobot-native deformable and tactile scenes. They do not
download files during installation, import, or scene loading.

Generate a standalone tactile gel scene:

```bash
python examples/warp_ipc/fabricate_sensor.py \
  --output examples/warp_ipc/fabricated_sensor.jscn
```

Download the pinned Allegro asset set explicitly, then rebuild the checked-in
scene if needed:

```bash
python examples/warp_ipc/download_allegro_assets.py
python examples/warp_ipc/build_allegro_tactile_scene.py
```

`allegro_assets.json` pins the upstream `newton-assets` revision, every file
size, every SHA-256 digest, and the BSD-2-Clause license file. Downloaded files
live under the ignored `assets/` directory and are not included in wheels.

`allegro_tactile_grasp.jscn` is the authoring source for the four-finger scene.
It contains four sensors sharing one gel topology plus a separate tetrahedral
grasp object. The current correctness runtime uses Newton 1.4 and standard Warp
1.15 for float64 geometry, CCD, barrier/friction contact, tetrahedral
elasticity, BSR assembly, and fixed-buffer CG. It runs on CUDA without Taccel,
its Warp fork, or a Python 3.10 extension.

The exact multi-hand grasp-synthesis program shown in the Taccel paper was not
published under `Taccel/examples`. Gobot's closest reproducible port is the
Allegro four-finger sequence: 60 frames close the fingers and 20 frames lift the
kinematic hand. It explicitly renders and can export RGB, depth, local normals,
marker positions/flow, per-gel contact force/wrench, and deformable-object state.
Scene and trajectory validation can run without starting CUDA:

```bash
python examples/warp_ipc/allegro_tactile_grasp.py --validate-only
```

Run the complete 80-frame grasp and record every fifth frame for one environment
with:

```bash
python examples/warp_ipc/allegro_tactile_grasp.py \
  --num-envs 1 --selected-env 0 --record-every 5 \
  --output-dir /tmp/gobot-allegro-tactile
```

Each recorded `.npz` keeps the 2D signals separate from the 3D gel/object data;
rendering is explicit and no readback occurs when `--output-dir` is omitted.
The correctness runtime currently requires graph capture to remain disabled;
an explicit `--capture-graphs` request fails instead of silently falling back.

The real CUDA numerical admission test is opt-in and does not download assets:

```bash
GOBOT_RUN_WARP_IPC_GPU_TEST=1 \
  ctest --test-dir build -R test_python_warp_ipc_gpu_kernels --output-on-failure
```
