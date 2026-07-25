# Gaussian Splatting Environment

This example uses the pretrained Tanks & Temples `train` scene to exercise
Gobot's real 3DGS environment path. The large PLY is downloaded separately and
is not committed to Gobot or included in wheels.

```bash
python3 examples/gaussian_splatting/download_sample.py
gobot_editor --path examples/gaussian_splatting
```

The downloader uses only the Python standard library. Run it directly with
`python3`; `uv run` may try to synchronize and rebuild the editable Gobot
environment before starting the download.

Run `gobot_editor` from an installed or activated Gobot environment. In a
source checkout whose `.venv` is already current, use
`.venv/bin/gobot_editor` or `uv run --no-sync gobot_editor`; plain `uv run`
may rebuild the editable package and fetch missing build inputs.

Use the Raster viewport mode. The first implementation does not show Gaussian
environments in ray-traced viewport modes. This visual-only sample does not
include proxy collision geometry, so its depth, normal, ID, and physics data
remain empty.

## Source

- Repository: [`datadude/gaussian_splatting`](https://huggingface.co/datadude/gaussian_splatting)
  on Hugging Face
- Revision: `65884107860281bfcde5b58904c327a923da7cc6`
- Asset: `train/point_cloud/iteration_7000/point_cloud.ply`
- Size: 165,633,787 bytes
- SHA-256: `e1bc6c22fa74db350a783385f578be0eb5465c1df0daaedb33fa10c99e10c380`
- Repository license declaration: MIT

The scene is a trained model of the Tanks & Temples `train` capture. The asset
is used only after an explicit download and remains subject to its source terms.
