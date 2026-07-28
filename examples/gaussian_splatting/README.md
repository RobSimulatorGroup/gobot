# Gaussian Splatting Environment

This example uses the pretrained Deep Blending `playroom` scene to exercise
Gobot's real 3DGS environment path. The large PLY is downloaded separately and
is not committed to Gobot or included in wheels.

Opening the project automatically runs its project-load hook. If the PLY is
missing, the editor downloads it into `assets/`, verifies its size and SHA-256,
and displays download progress in a modal dialog:

```bash
gobot_editor --path examples/gaussian_splatting
```

The downloader remains available as a manual command:

```bash
python3 examples/gaussian_splatting/download_sample.py
```

The downloader uses only the Python standard library. Run it directly with
`python3`; `uv run` may try to synchronize and rebuild the editable Gobot
environment before starting the download. Both automatic and manual downloads
reuse a verified PLY on later runs, and an interrupted `.part` download resumes
when the server supports HTTP range requests.

Run `gobot_editor` from an installed or activated Gobot environment. In a
source checkout, plain `uv run gobot_editor` synchronizes and may rebuild the
editable package using the MuJoCo SDK installed in `.venv`. When the environment
is already current, `.venv/bin/gobot_editor` or
`uv run --no-sync gobot_editor` skips that synchronization work. In a source
editable install, the launcher still runs a fast CMake/Ninja dependency check
so changed native code cannot start an older editor binary.

Viewport navigation follows common DCC controls: `Alt+Left Mouse` orbits,
`Middle Mouse` pans, the wheel or `Alt+Right Mouse` dollies, and holding
`Right Mouse` enables `WASD`/`QE` fly navigation. Hold `Shift` to move faster
or `Ctrl` for precise movement.

Use the Raster viewport mode. The first implementation does not show Gaussian
environments in ray-traced viewport modes. This visual-only sample does not
include proxy collision geometry, so its depth, normal, ID, and physics data
remain empty.

The saved editor view is not an arbitrary orbit camera. It is training camera
`DSC05573` from the official pretrained model, transformed by the same
`source_to_gobot` matrix as the splats. Starting from a training pose matters:
3DGS quality degrades quickly when the camera moves outside the capture volume.
The transform also raises the reconstruction by 3 m so Gobot's `z=0` editor
grid stays below the room instead of covering the initial camera view.

## Source

- PLY mirror: [`Voxel51/gaussian_splatting`](https://huggingface.co/datasets/Voxel51/gaussian_splatting)
  on Hugging Face
- Revision: `ed0588b29edea35e36dad784f73c1f502cc8a0d2`
- Asset: `FO_dataset/playroom/point_cloud/iteration_7000/point_cloud.ply`
- Size: 370,875,860 bytes
- SHA-256: `201bc92b65594727a3ecfbe7e658c09ac3f8be753e2e2024047cd3ea1fe31d8c`
- Dataset card license declaration: Apache-2.0
- Camera metadata: [official INRIA pretrained archive](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip),
  `playroom/cameras.json`, camera ID `29` (`DSC05573`)

The scene is a trained model of the Deep Blending `playroom` capture. The asset
is used only after an explicit download and remains subject to its source terms.
