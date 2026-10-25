"""Offline UniVTAC -> Gobot asset conversion. Requires usd-core, numpy, scipy.

Runtime uses only the generated .jscn; Isaac Sim and USD Python are not needed.
Original meshes and calibration are preserved, with a sensor frame at the gel top.
"""

from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom
from scipy.ndimage import gaussian_filter
import sys

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "examples/tactile_gelsight")
)
from press_geometry import replace_press

REVISION = "0dafa10262e22f486f160a55d6f11aeab12d8e7b"


def matrix(value):
    a = np.asarray(value)
    if a.ndim == 1:
        a = a[:, None]
    return {
        "matrix_data": {
            "rows": a.shape[0],
            "cols": a.shape[1],
            "storage": a.flatten(order="F").tolist(),
        }
    }


def convert(source, case_path, gel_path, output):
    stage = Usd.Stage.Open(str(gel_path))
    prim = stage.GetPrimAtPath("/Gelpad_high_res/mesh")
    vertices = np.asarray(prim.GetAttribute("tet_points").Get(), dtype=float)
    tets = np.asarray(prim.GetAttribute("tet_indices").Get(), dtype=np.int64).reshape(
        -1, 4
    )
    top = vertices[:, 2].max()
    vertices[:, 2] -= top
    for tet in tets:
        if np.linalg.det((vertices[tet[1:]] - vertices[tet[0]]).T) < 0:
            tet[1], tet[2] = tet[2], tet[1]
    faces = defaultdict(list)
    for ti, (a, b, c, d) in enumerate(tets):
        for f in ((b, c, d), (a, d, c), (a, b, d), (a, c, b)):
            faces[tuple(sorted(f))].append((f, ti))
    boundary = [items[0] for items in faces.values() if len(items) == 1]
    surface = np.asarray([f for f, _ in boundary], dtype=int)
    coat = np.flatnonzero(np.isclose(vertices[:, 2], 0, atol=1e-6))
    stick = np.flatnonzero(np.isclose(vertices[:, 2], vertices[:, 2].min(), atol=1e-6))
    skin = [(f, ti) for f, ti in boundary if all(i in coat for i in f)]
    markers = []
    marker_tets = []
    weights = []
    camera_z = -0.0283
    fx = 320 * 0.0283 / 0.0266
    fy = 240 * 0.0283 / 0.0209
    for y in np.arange(7) * 0.00245833 - 3 * 0.00245833:
        for x in np.arange(9) * 0.00240625 - 4 * 0.00240625:
            for face, ti in skin:
                triangle = vertices[list(face)]
                w = np.linalg.solve(
                    np.vstack((triangle[:, :2].T, np.ones(3))), [x, y, 1]
                )
                if min(w) >= -1e-7:
                    w = np.maximum(w, 0.0)
                    w /= w.sum()
                    bary = np.zeros(4)
                    for i, vertex in enumerate(face):
                        bary[np.flatnonzero(tets[ti] == vertex)[0]] = w[i]
                    markers.append(
                        matrix(
                            [fx * x / (-camera_z) + 160, -fy * y / (-camera_z) + 120]
                        )
                    )
                    marker_tets.append(ti)
                    weights.append(matrix(bary))
                    break
            else:
                raise ValueError("marker lies outside the original FEM skin")
    root = (
        source
        / "third_party/TacEx/source/tacex_assets/tacex_assets/data/Sensors/GelSight_Mini/calibs/640x480"
    )
    with np.load(root / "polycalib.npz", allow_pickle=False) as data:
        coefficients = (
            np.stack([data["grad_b"], data["grad_g"], data["grad_r"]]) / 255.0
        )
    with np.load(root / "dataPack.npz", allow_pickle=False) as data:
        background = data["f0"][..., ::-1].astype(float) / 255.0
    params = json.loads((root / "params.json").read_text())
    # TAXIM initial-frame preprocessing (reflect padding, finite Gaussian support).
    sigma = np.array(params["simulator"]["initial_frame_sigma_rel"]) * [640, 480]
    kernel = (
        np.rint(np.sqrt(-2 * np.log(1e-5 * np.sqrt(2 * np.pi) * sigma)) * sigma).astype(
            int
        )
        // 2
        * 2
        + 1
    )
    blurred = gaussian_filter(
        background,
        sigma=(sigma[1], sigma[0], 0),
        radius=(kernel[1] // 2, kernel[0] // 2, 0),
        mode="mirror",
    )
    background = 0.15 * blurred + 0.85 * background
    # Bilinear half-pixel centers for the exact 2:1 reduction.
    background = (
        background[::2, ::2]
        + background[1::2, ::2]
        + background[::2, 1::2]
        + background[1::2, 1::2]
    ) * 0.25
    stage = Usd.Stage.Open(str(case_path))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/GelsightMiniCase/case/mesh"))
    cv = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    transform = np.asarray(mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    cv = cv @ transform[:3, :3].T + transform[:3, 3]
    cv[:, 2] += camera_z
    raw = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    counts = mesh.GetFaceVertexCountsAttr().Get()
    ci = []
    start = 0
    for n in counts:
        polygon = raw[start : start + n]
        start += n
        for i in range(1, n - 1):
            ci.extend([int(polygon[0]), int(polygon[i]), int(polygon[i + 1])])
    resources = []

    def resource(kind, name, **values):
        resources.append({"__TYPE__": kind, "__ID__": name, **values})
        return f"SubResource({name})"

    def color(r, g, b):
        return dict(red=r, green=g, blue=b, alpha=1.0)

    optical = resource(
        "TactileOpticalCalibration",
        "gsmini_taxim",
        width=320,
        height=240,
        bins=125,
        calibration_width=640,
        calibration_height=480,
        pixel_size=params["sensor"]["pixmm"] / 1000.,
        coefficients=coefficients.ravel().tolist(),
        background=background.ravel().tolist(),
    )
    tetmesh = resource(
        "TetrahedralMesh",
        "gsmini_fem",
        vertices=[matrix(v) for v in vertices],
        tetrahedra=tets.ravel().tolist(),
        surface_triangles=surface.ravel().tolist(),
    )
    camera = np.eye(4)
    camera[2, 3] = camera_z
    config = resource(
        "TactileSensorConfig",
        "gsmini",
        image_width=320,
        image_height=240,
        near_plane=0.020,
        far_plane=0.034,
        pixel_size=0.0266 / 320,
        camera_intrinsics=matrix([fx, fy, 160, 120]),
        camera_transform={"matrix_data": camera.flatten(order="F").tolist()},
        density=10000.0,
        young_modulus=100000.0,
        poisson_ratio=0.49,
        damping=0.0,
        friction_coefficient=1.0,
        attachment_strength=10000.0,
        coat_vertex_indices=coat.tolist(),
        stick_vertex_indices=stick.tolist(),
        marker_positions=markers,
        marker_tetrahedra=marker_tets,
        marker_barycentric=weights,
        rgb_model="taxim_v1",
        optical_calibration=optical,
    )
    case = resource(
        "ArrayMesh",
        "gsmini_case",
        surfaces=[
            dict(
                vertices=[matrix(v) for v in cv],
                indices=ci,
                normals=[],
                tangents=[],
                uv0=[],
                colors=[],
                material=None,
            )
        ],
    )
    sphere = resource(
        "SphereMesh", "press_visual", radius=0.005, radial_segments=32, rings=16
    )
    shape = resource("SphereShape3D", "press_shape", radius=0.005)
    material = resource(
        "PhysicsMaterial3D",
        "press_material",
        sliding_friction=1.0,
        torsional_friction=0.0,
        rolling_friction=0.0,
    )
    nodes = []

    def node(kind, name, parent, **properties):
        nodes.append(dict(type=kind, name=name, parent=parent, properties=properties))
        return len(nodes) - 1

    world = node("Node3D", "TactileBench", -1, script="ExtResource(play)")
    sensor = node(
        "TactileSensor3D",
        "GelSightMini",
        world,
        config=config,
        gel_mesh=tetmesh,
        sensor_period=0.0,
        visualize_debug=True,
    )
    node(
        "MeshInstance3D",
        "Case",
        sensor,
        mesh=case,
        surface_color=color(0.07, 0.075, 0.09),
    )
    press = node(
        "RigidBody3D",
        "PressHead",
        world,
        position=matrix([0, 0, 0.007]),
        has_inertial=True,
        mass=0.05,
        inertia_diagonal=matrix([5e-7] * 3),
    )
    node(
        "MeshInstance3D",
        "Sphere",
        press,
        mesh=sphere,
        surface_color=color(0.92, 0.45, 0.12),
    )
    node("CollisionShape3D", "Collision", press, shape=shape, physics_material=material)
    scene = {
        "__VERSION__": 3,
        "__META_TYPE__": "SCENE",
        "__TYPE__": "PackedScene",
        "__EXT_RESOURCES__": [
            {
                "__ID__": "play",
                "__TYPE__": "PythonScript",
                "__PATH__": "res://tactile_play.py",
            }
        ],
        "__SUB_RESOURCES__": resources,
        "__NODES__": nodes,
    }
    replace_press(scene)
    output.mkdir(parents=True, exist_ok=True)
    (output / "gelsight_press.jscn").write_text(
        json.dumps(scene, separators=(",", ":")) + "\n"
    )
    (output / "project.gobot").write_text(
        json.dumps(
            {
                "main_scene": "res://gelsight_press.jscn",
                "editor_scene_views": {
                    "res://gelsight_press.jscn": {
                        "eye": [0.055, -0.070, 0.060],
                        "at": [0, 0, -0.008],
                        "up": [0, 0, 1],
                        "fov_y": 40,
                    }
                },
            },
            indent=2,
        )
        + "\n"
    )
    sources = [case_path, gel_path, root / "polycalib.npz", root / "dataPack.npz"]
    (output / "asset_provenance.json").write_text(
        json.dumps(
            {
                "repository": "univtac/UniVTAC",
                "revision": REVISION,
                "sensor": "GelSight Mini",
                "frame": "gel top centered, +Z outward, metres",
                "vertices": len(vertices),
                "tetrahedra": len(tets),
                "markers": len(markers),
                "source_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "Converted:",
        len(vertices),
        "vertices,",
        len(tets),
        "tetrahedra,",
        len(coat),
        "skin vertices,",
        len(stick),
        "backing vertices",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--univtac", type=Path, required=True)
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--gel", type=Path, required=True)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples/tactile_gelsight",
    )
    a = p.parse_args()
    convert(a.univtac, a.case, a.gel, a.output)
