# Gaussian Splatting Environments

Gobot can load an inference-only 3D Gaussian scene as a visual environment for
the Raster editor viewport and camera render products. Training remains outside
the engine.

## Asset Layout

Keep the source 3DGS PLY and an optional proxy scene beside a `.gsplat`
manifest:

```json
{
  "__VERSION__": 1,
  "__TYPE__": "GaussianSplatScene",
  "ply": "warehouse.ply",
  "proxy_scene": "warehouse_proxy.jscn",
  "meters_per_unit": 1.0,
  "source_to_gobot": [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  ]
}
```

`ply` is required. Relative paths resolve from the manifest. The PLY may be
ASCII or binary little-endian and must use the standard 3DGS scalar properties:
position, `rot_0..3` in WXYZ order, `scale_0..2`, `opacity`, `f_dc_0..2`, and
optionally contiguous `f_rest_*` values for SH degree 1 through 3.

The optional proxy is expanded into normal Gobot nodes. Its mesh instances are
imported with `visible_in_rgb=false` and `cast_shadow=false`; they still provide
physics collision, depth, world normals, instance IDs, semantic IDs, and
occlusion against dynamic scene geometry. Without a proxy, the splat remains a
visual background and non-Gaussian AOV pixels retain their normal background
values.

## Runtime Behavior

- One enabled `GaussianSplat3D` is allowed in a render snapshot.
- Node and manifest transforms must be rigid with positive uniform scale.
- Raster viewport rendering uses direct CUDA-OpenGL presentation before normal
  scene geometry, FXAA, and editor overlays.
- `CameraSensor` CUDA frames composite Gaussian RGB directly into the device
  buffer. CPU/NumPy frames use the same CUDA render followed by synchronous
  readback.
- Ray-traced viewport modes do not render the Gaussian environment in the first
  implementation. Camera render products continue to use the minimal geometry
  pass for proxy AOVs.

## Installation And Build

Linux NVIDIA wheels contain AOT-compiled kernels and declare the CUDA 12 runtime
as a pip dependency. End users need a compatible NVIDIA driver, but do not need
`nvcc`, a system CUDA Toolkit, gsplat, or runtime network access.

For a source build, compile the pinned inference library before enabling the
Luisa module:

```bash
scripts/build_luisa_compute.sh
scripts/build_gsplat_inference.sh
cmake -S . -B build_cuda \
  -DGOB_BUILD_LUISA_RENDERER=ON \
  -DGOB_BUILD_GSPLAT_INFERENCE=ON \
  -DGOB_LUISA_COMPUTE_ROOT="$PWD/build/luisa_compute/install" \
  -DGOB_GSPLAT_INFERENCE_ROOT="$PWD/build/gsplat_inference/install"
```

The vendored raw-CUDA forward renderer is derived from gsplat commit
`bf302ef532a25175e44bd6bf7f777f599570748c`; attribution and modification notes
are stored in `3rdparty/gsplat_inference`. It does not vendor PyTorch/ATen,
training code, or the upstream HiGS macro-tile wrapper.

## Pretrained Environment Sample

The repository includes an optional example project backed by the pretrained
Tanks & Temples `train` scene. Its 166 MB PLY is downloaded and SHA-256 checked
on demand instead of being stored in Git or bundled into wheels:

```bash
python3 examples/gaussian_splatting/download_sample.py
gobot_editor --path examples/gaussian_splatting
```

The downloader is standalone and uses only the Python standard library. Do not
wrap it in `uv run`, which may synchronize the editable Gobot environment first.
Launch `gobot_editor` from an installed or activated environment. Plain
`uv run gobot_editor` may first synchronize and rebuild Gobot; editable builds
reuse the MuJoCo SDK installed in the project `.venv`. Use
`uv run --no-sync gobot_editor` when the environment is already synchronized
and no rebuild is wanted.

See `examples/gaussian_splatting/README.md` for the pinned source revision,
license declaration, file size, and checksum.
