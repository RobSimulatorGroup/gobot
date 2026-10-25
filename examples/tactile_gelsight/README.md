# GelSight Mini tactile bench

Open `project.gobot` in Gobot, or launch a freshly built editor:

```bash
.venv/bin/gobot_editor --path examples/tactile_gelsight
```

Press **Play**, then open **View → Tactile**. **View → Reset Layout** applies the
default arrangement: viewport and Python/Console in the center, Inspector above
Tactile/Physics on the right. Existing saved layouts are preserved. A single torus
approaches the gel,
presses it, slides sideways and retracts in a ten-second simulation-time cycle.
Pause, Step, Reset and Stop use the normal editor controls. Physics runs on the
simulation worker; editor frame rate does not change the fixed `1/120 s` step.
The viewport and tactile panel consume completed snapshots, published at most
30 times per second. Solver speed can be slower than real time.

To control the torus yourself, select **PressHead** or click its torus, then drag
the **Move** gizmo during Play. The first drag takes over from the automatic
trajectory; releasing the mouse holds the target. The UIPC soft constraint moves
the torus and generates physical contact forces. Translation is rate-limited to
10 mm/s and bounded to a 1 mm nominal indentation for this bench. Reset restores
the automatic cycle. Paused simulation consumes the target on Step/Resume.
Runtime targets never edit the authored scene or add undo entries.

The panel displays RGB, RGB with markers, indentation (mm), normal pressure (Pa)
and shear magnitude (Pa), plus the sensor-frame force in newtons. Motion arrows
show displacement from the reference marker positions without resampling IDs.
Pressure and shear are conservative estimates from the real IPC forces at the
sensing-surface FEM vertices. They use the reference gel's material grid; backing
constraint reactions are excluded.

The RGB optical response follows UniVTAC's split: a high-resolution depth
projection of the actual triangle-mesh contactors supplies the contact outline,
TAXIM's multiscale Gaussian model preserves the contact core while smoothing the
surrounding optical gel, and calibrated slopes produce RGB. `contact_depth` and
`optical_indentation` expose those image products. `depth` and `indentation`
remain the unfiltered FEM gel measurements; marker motion and contact forces
still come from UIPC. The generic fallback for primitive contactors uses smooth
FEM normals. These reference simulation parameters have not been calibrated
against measured hardware force data.

The press head is a horizontal torus (14 mm outer diameter, 6 mm hole). Its
single `TriangleMeshShape3D` collision surface preserves the hole without filling
it with a convex hull. SceneTree shows one `Collision` child.

The authored scene contains the converted original UniVTAC GelSight case, the
169-vertex/424-tetrahedron gel, 63 markers, and the optical calibration. The scene
is self-contained except for its local Play scripts. Runtime does not need
UniVTAC, Isaac Sim, USD Python, SciPy, or network access. The native libuipc CUDA
module is required. Model provenance and source hashes are in
`asset_provenance.json`; source notices are retained alongside the scene.

The sensor's origin is centered on the resting top surface, +Z points outward,
+X is image-right, and +Y is image-up. All authoring and physics quantities use SI
units. Initial shell and gel resources remain unchanged during Play.

To regenerate assets offline after downloading the pinned UniVTAC LFS `Case.usd`
and `Gelpad_high_res.usd` files:

```bash
uv run --no-project --with usd-core --with numpy --with scipy \
  python tools/import_gelsight_mini.py \
  --univtac /path/to/UniVTAC --case /path/to/Case.usd --gel /path/to/Gelpad_high_res.usd
```

The converter preserves the original tetrahedral topology, fixes orientation,
extracts the sensing/backing vertex sets, embeds markers in existing tetrahedra,
and converts numeric calibration data with pickle loading disabled.

Validation:

```bash
build/tests/test_tactile_sensor_processor
ctest --test-dir build -R test_python_tactile_scene --output-on-failure
GOBOT_RUN_TACTILE_GPU_TEST=1 \
  ctest --test-dir build -R test_python_tactile_gelsight --output-on-failure
```

For a separately built solver, supply its path with
`GOBOT_LIBUIPC_TEST_MODULE_PATH` when running the test. Batch environments and the
GF225/Xense WS presets remain future additions.
