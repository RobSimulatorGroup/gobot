"""CPU contract for the shipped GelSight asset, resource codec and optical processor."""

import json
from pathlib import Path
import struct
import tempfile
import numpy as np
import gobot

root = Path(__file__).resolve().parents[2]
c = gobot.app.create_context()
c.set_project_path(str(root / "examples/tactile_gelsight"))
c.load_scene("res://gelsight_press.jscn")
artifact = c.compile_ipc_scene_artifact()
sensor = json.loads(artifact["manifest"])["tactile_sensors"][0]
assert sensor["resolution"] == [240, 320]
assert sensor["gel_vertex_count"] == 169 and sensor["gel_tetrahedron_count"] == 424
assert len(sensor["marker_positions"]) == 63
camera = np.asarray(sensor["camera_transform"]["matrix_row_major"]).reshape(4, 4)
np.testing.assert_allclose(camera[:3, 3], (0, 0, -0.0283), atol=1e-8)
assert sensor["optical_calibration"]["bins"] == 125
blob = next(b["data"] for b in artifact["blobs"] if b["id"] == sensor["gel_mesh_blob"])
_, v, t, f = struct.unpack_from("<IIII", blob, 8)
vertices = np.frombuffer(blob, dtype="<f8", offset=24, count=v * 3).reshape(v, 3)
tets = np.frombuffer(blob, dtype="<u4", offset=24 + v * 24, count=t * 4)
surface = np.frombuffer(blob, dtype="<u4", offset=24 + v * 24 + t * 16, count=f * 3)
processor = gobot._core._TactileSensorProcessor(
    sensor, vertices, tets.tolist(), surface.tolist()
)
frame = processor.process(vertices, np.zeros_like(vertices), np.eye(4), 0, 0.0)
assert frame["rgb"].shape == (240, 320, 3) and frame["rgb"].dtype == np.uint8
assert np.ptp(frame["rgb"]) > 20
assert np.allclose(frame["indentation"], 0, atol=1e-7)
assert np.allclose(frame["marker_motion"], 0, atol=1e-5)
assert np.isclose(frame["depth"][120, 160], 0.0283, atol=1e-7)
with tempfile.TemporaryDirectory(prefix="gobot-tactile-roundtrip-") as directory:
    path = str(Path(directory) / "roundtrip.jscn")
    gobot.save_scene(c.root, path)
    c.load_scene(path)
    restored = json.loads(c.compile_ipc_scene_artifact()["manifest"])[
        "tactile_sensors"
    ][0]
    assert restored == sensor
print("GelSight asset, camera, optics, markers and save/load contracts passed.")

# Runtime gizmo inputs are bounded control data, not authoring transform edits.
press = next(node for node in c.root.children if node.type_name == "RigidBody3D")
position = press.position
c.register_runtime_pose_target(press, (-0.03, -0.03, 0.004), (0.03, 0.03, 0.03))
assert c.get_runtime_pose_target(press) is None
pose = np.eye(4)
pose[:3, 3] = (0.1, -0.1, 0)
c.set_runtime_pose_target(press, pose)
np.testing.assert_allclose(
    c.get_runtime_pose_target(press)[:3, 3], (0.03, -0.03, 0.004), atol=1e-8
)
assert tuple(press.position) == tuple(position) and not c.scene_dirty and not c.can_undo
c.clear_runtime_pose_target(press)
assert c.get_runtime_pose_target(press) is None

# The worker controller takes over smoothly and reset returns to the demo cycle.
import sys

sys.path.insert(0, str(root / "examples/tactile_gelsight"))
from tactile_runtime import PressController
from press_geometry import START_HEIGHT, PRESS_HEIGHT


class Sink:
    def set_affine_target(self, path, value):
        self.target = np.array(value)


sink = Sink()
controller = PressController(sink, "press")
pose = np.eye(4)
pose[2, 3] = PRESS_HEIGHT
controller.apply_commands({"press_target": pose})
controller.step(0.01)
np.testing.assert_allclose(sink.target[:3, 3], (0, 0, START_HEIGHT - 0.0001), atol=1e-8)
controller.reset((0,))
assert controller.manual_target is None
np.testing.assert_allclose(sink.target[:3, 3], (0, 0, START_HEIGHT), atol=1e-8)
print("Runtime pose target and manual press controller contracts passed.")

# The torus has a real open hole in its triangle collision geometry.
manifest = json.loads(artifact["manifest"])
press_shapes = manifest["robots"][0]["links"][0]["collision_shapes"]
assert len(press_shapes) == 1 and press_shapes[0]["shape_type"] == "triangle_mesh"
assert press_shapes[0]["convex"] is False
for shape in press_shapes:
    mesh = next(b["data"] for b in artifact["blobs"] if b["id"] == shape["mesh_blob"])
    _, nv, nt = struct.unpack_from("<III", mesh, 8)
    points = np.frombuffer(mesh, dtype="<f8", offset=20, count=nv * 3).reshape(nv, 3)
    assert np.linalg.norm(points[:, :2], axis=1).min() > 0.0029
print("Hollow torus collision geometry passed.")

# A backend that convexifies meshes must report the unsupported shape explicitly.
try:
    c.backend_type = gobot.PhysicsBackendType.MuJoCoCpu
    c.build_world(gobot.PhysicsBackendType.MuJoCoCpu)
except RuntimeError as error:
    assert "TriangleMeshShape3D" in str(error)
else:
    raise AssertionError("MuJoCo silently convexified the torus")
