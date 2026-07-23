# Luisa CUDA Rendering

Gobot provides an optional LuisaCompute CUDA renderer for the editor viewport.
It extends the existing rendering boundary; it does not replace Gobot scene,
material, editor, or Python APIs.

Plain CMake builds remain OpenGL-only unless the option is enabled. Published
Python wheels enable it by default and include a shared module named
`libgobot_luisa_renderer.so`, which the OpenGL renderer discovers and loads at
runtime.

## Architecture

The data flow is:

```text
.jscn / live authored scene
  -> RenderServer snapshot compiler
  -> immutable RenderSceneSnapshot + RenderViewSnapshot
       -> OpenGL raster renderer
       -> optional LuisaCompute CUDA renderer module
       -> RenderProduct RGB/AOV frame slots
  -> editor viewport or camera sensor
```

`RenderSceneSnapshot` contains camera-independent CPU-side geometry, PBR
materials, texture image data, environment state, authored lights, stable
instance paths, and semantic labels. `RenderViewSnapshot` contains one pinhole
camera and its fingerprint. A renderer never traverses `Node`, retains
`Camera3D`, or reads editor selection state.

The optional module owns all Luisa and CUDA objects:

- Luisa `Context`, CUDA `Device`, stream, shaders, buffers, images, meshes, and
  acceleration structures
- geometry and texture caches keyed by Gobot resource identity and revision
- progressive accumulation, guide buffers, adaptive sampling, and presentation
- CUDA-OpenGL registration and synchronization

No Luisa or CUDA implementation header is included by `include/gobot` or
`src/gobot`. The private module ABI uses backend-neutral Gobot snapshots and is
versioned by `GOBOT_LUISA_RENDERER_ABI_VERSION`.

## Camera Render Products

`RenderProduct` is shared by editor-independent `CameraSensor` captures and
Python data collection. It allocates only requested outputs:

- `rgb`: sRGB `uint8 [H,W,3]`
- `linear_depth`: camera-forward metres in `float32 [H,W]`, background `+inf`
- `world_normal`: world-space unit vectors in `float32 [H,W,3]`, background zero
- `instance_id` and `semantic_id`: `uint32 [H,W]`, background/unlabelled zero

Instance ids derive from scene-relative paths; all surfaces of one mesh node
share an id. Semantic labels inherit from the nearest non-empty ancestor.
Every frame carries both id maps.

OpenGL writes all requested outputs in one MRT geometry pass and reads them
through the backend-neutral `RenderBuffer`. The Luisa ABI v2 writes primary-hit
AOVs to linear CUDA buffers. CUDA buffers export DLPack directly to Torch or
Warp; `numpy()` performs an explicit synchronous host copy. Public APIs expose
neither OpenGL textures nor CUDA/Luisa handles.

Sensor `minimal` captures are unaccumulated: OpenGL uses one raster geometry
pass, while Luisa uses one primary ray with direct lighting and hard shadows.
The editor's realtime and progressive quality modes remain separate.

Each product owns three frame slots by default. A frame, zero-copy CPU NumPy
view, or DLPack consumer keeps its slot alive, and capture fails explicitly
when the pool is exhausted. Capture is synchronous in this phase. Sensor
passes do not include editor debug drawing, gizmos, or overlays. The legacy
`capture_rgb()` helper is a one-output CPU render-product wrapper, retaining
its old overlay path only when debug arrows are explicitly requested.

## Implemented Viewport Modes

The camera-iris button in the 3D viewport exposes:

- `Raster`: OpenGL 4.6 PBR rasterization.
- `Ray Tracing Auto`: one-sample realtime rendering while camera or scene data
  changes, then progressive accumulation after the scene is stable.
- `Realtime Ray Tracing`: one new sample per frame and no retained
  accumulation.
- `Progressive Path Tracing`: accumulates samples until the configured limit.

Raster mode exposes frustum culling, FXAA, shadow quality, and shadow distance.
Ray modes expose target FPS, samples per frame, bounce count, denoising, and
adaptive quality. The popup also reports active samples, visible/culled item
counts, draw calls, per-stage timing, and backend status.

Implemented rendering features include:

- triangle meshes and instancing
- transform-only acceleration structure updates
- base-color, metallic-roughness, normal, occlusion, and emissive textures
- directional, point, and spot lights
- sky/ground environment colors and equirectangular environment textures
- GGX-style direct lighting, diffuse/glossy path bounces, Russian roulette,
  ACES tone mapping, and gamma conversion
- a small normal/depth-aware guide denoiser for low sample counts
- deterministic accumulation resets for camera, geometry, transform, material,
  and lighting changes
- direct CUDA writes into the editor's `GL_RGBA8` presentation texture
- an OpenGL depth prepass so editor debug geometry and overlays keep normal
  depth behavior
- conservative world-AABB frustum culling and deterministic opaque/alpha-mask/
  transparent draw queues
- back-to-front alpha blending without writing blended surfaces into geometry
  AOV attachments
- one texel-stabilized directional-light shadow map with low/medium/high PCF
  quality levels
- an LDR FXAA pass for RGB-only viewport output, applied before editor debug
  geometry and overlays

`Light3D.shadow_enabled`, `shadow_bias`, and `shadow_normal_bias` are serialized
backend-neutral scene properties. Shadow offsets use metres. Python callers can
configure the global viewport defaults with `gobot.render.RasterSettings`,
`get_raster_settings()`, and `set_raster_settings()`.

If module loading, CUDA initialization, scene synchronization, or interop fails,
the frame falls back to OpenGL and the renderer popup shows the exact error.
An empty scene also falls back instead of attempting to build an invalid empty
acceleration structure.

## Build LuisaCompute

The pinned `stable` LuisaCompute source is a git submodule. LuisaRender remains
an algorithm and implementation reference; it is not a Gobot runtime
dependency.

Requirements:

- Linux with an NVIDIA driver
- CUDA Toolkit 12.1 or newer
- CMake 3.26 or newer
- a C++20 compiler
- Ninja is recommended

Initialize the source and its pinned dependencies:

```bash
git submodule update --init 3rdparty/luisa_compute
git -C 3rdparty/luisa_compute submodule update --init --recursive
```

The build script rejects uninitialized, mismatched, conflicted, or dirty nested
submodules before configuring CMake. This prevents an extracted dependency tree
from being mistaken for the pinned Git checkout. Set
`GOB_LUISA_ALLOW_DIRTY_SOURCE=1` only when intentionally developing a LuisaCompute
dependency locally.

Build and install the isolated CUDA-only dependency:

```bash
scripts/build_luisa_compute.sh
```

The default install prefix is `build/luisa_compute/install`. Run this script
before `uv sync` or a local wheel build. The script keeps
Luisa's CMake options and bundled targets out of Gobot's build graph. It prefers
GCC 11 when available because GCC 12.3 on Ubuntu 22.04 can hit an internal
compiler error in Luisa's CUDA acceleration code.

Override tools or paths when needed:

```bash
CMAKE_BIN=/path/to/cmake \
GOB_LUISA_CC=/usr/bin/gcc-11 \
GOB_LUISA_CXX=/usr/bin/g++-11 \
GOB_LUISA_BUILD_DIR="$PWD/build/luisa_compute" \
GOB_LUISA_INSTALL_DIR="$PWD/build/luisa_compute/install" \
scripts/build_luisa_compute.sh
```

## Build Gobot

Configure a separate optional build:

```bash
cmake -S . -B build_luisa -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGOB_BUILD_LUISA_RENDERER=ON \
  -DGOB_LUISA_COMPUTE_ROOT="$PWD/build/luisa_compute/install"
cmake --build build_luisa --target gobot_editor -j
```

Run the source-build editor:

```bash
build_luisa/python/gobot/gobot_editor --path examples/go1
```

The build copies Luisa shared libraries, the CUDA backend plugin, and
`luisa_nvrtc` into `python/gobot/luisa`. The editor finds the renderer module
beside `libgobot.so` or the executable. `GOBOT_LUISA_RENDERER_LIBRARY` may point
to an explicit module path for diagnostics.

Published wheels contain those files already. Installing a wheel does not
require a system CUDA Toolkit or a local LuisaCompute checkout. It does require
an NVIDIA display driver with `libcuda.so.1` and the system libglvnd dispatch
libraries; shader code is compiled for the active GPU by the bundled
`luisa_nvrtc` helper. JIT cache files are written to
`$XDG_CACHE_HOME/gobot/luisa/v2` or `~/.cache/gobot/luisa/v2`, never into the
installed wheel.

The Python dependency set also installs Torch for RL and DLPack workflows.
Torch may install its own CUDA user-space packages, but the Gobot renderer does
not discover libraries through Torch or require Torch to initialize first.
Both stacks use the system NVIDIA driver; Gobot carries its own pinned Luisa
runtime so a Torch CUDA-minor upgrade cannot silently change renderer loading.

Release wheels are cross-built on standard GitHub-hosted Ubuntu runners. A
physical GPU is not required to compile CUDA code or link against the driver
stubs. LuisaCompute is built once per workflow in the pinned NVIDIA CUDA
development container. Its install tree and a compact CUDA build SDK containing
headers, `libcudart`, and the driver stub are passed to the Python-version wheel
matrix as a versioned tar artifact. Actual CUDA/OpenGL frame validation remains
in the independent GPU CI workflow on a self-hosted NVIDIA runner, so an offline
GPU machine cannot block wheel creation or PyPI publication.

## libglvnd And GPU Selection

libglvnd remains a system and NVIDIA-driver dispatch dependency. Gobot does not
vendor it as a third-party source dependency. Vendoring a second GL dispatch
stack would risk loading OpenGL and CUDA resources from different driver
implementations. The release repair step explicitly excludes `libEGL`,
`libGLdispatch`, `libGLX`, `libOpenGL`, and `libGL` from wheels, and the release
payload verifier rejects them if they are accidentally bundled.

At module creation, Gobot calls `cuGLGetDevices` for the current OpenGL context
and asks LuisaCompute to use that CUDA device. This is required on hybrid and
multi-GPU systems; choosing CUDA device 0 unconditionally can make
`cuGraphicsGLRegisterImage` fail even when both APIs work independently.

Useful diagnostics:

```bash
nvidia-smi
glxinfo -B
ldd build_luisa/python/gobot/libgobot_luisa_renderer.so
```

The OpenGL renderer must report the same NVIDIA GPU selected by CUDA. A Mesa,
llvmpipe, or non-CUDA OpenGL context cannot use direct CUDA presentation and
will use the normal raster fallback.

## Verification

Default CI does not require CUDA or LuisaCompute. Existing scene, resource,
OpenGL, and architecture tests cover the shared render contracts.

When `GOB_BUILD_LUISA_RENDERER=ON` and tests are enabled,
`test_luisa_renderer_module` loads the shared module and checks its ABI and
backend-neutral capability contract without creating a GPU device. GPU frame
validation additionally verifies:

- the module selects the OpenGL-associated NVIDIA device
- a non-empty scene produces nonblack finite pixels
- progressive sample count increases while the scene is stable
- camera or scene changes reset accumulation
- CUDA-OpenGL presentation succeeds without CPU readback
- five CUDA render-product outputs, background conventions, and id maps
- CPU/CUDA primary-hit depth, normal, and id parity
- direct Torch/Warp DLPack consumption when those packages expose CUDA
- frame-slot retention across DLPack consumers
- empty or unsupported scenes recover through OpenGL fallback

## Remaining Work

The editor and single-camera render-product paths are usable, but these are
separate follow-up milestones:

- persistent render scenes, dirty updates, asynchronous PBO/CUDA fences, and
  same-resolution multi-camera batching
- production denoising through an optional OptiX/OIDN integration
- stronger light and environment importance sampling and transparent material
  handling
- cascaded directional shadows, point/spot shadow maps, prefiltered IBL, SSAO,
  bloom, TAA, and order-independent transparency
- render-thread command submission so scene snapshot compilation and GPU work
  can overlap safely
- per-stage profiling for upload, acceleration build/update, tracing,
  denoising, interop, and sensor readback
- camera distortion/noise/exposure, motion vectors, point clouds, optical flow,
  and 2D/3D bounding boxes
