from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any

import gobot
import numpy as np

from gobot.ipc import (
    CompiledIpcSceneArtifact,
    DeformableBatchState,
    TactileBatchState,
    WarpIpcConfig,
    WarpIpcProvider,
)
from gobot.ipc._artifact import _decode_mesh_blob, _decode_surface_mesh_blob
from gobot.ipc._fabrication import create_fabricated_sensor_scene, fabricate_box_gel
from gobot.rl.providers import GraphInvalidatedError, ProviderUnavailableError


class _Tensor:
    def __init__(self, value: Any):
        self.array = np.asarray(value)

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> "_Tensor":
        return cls(np.zeros(shape, dtype=np.float32))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    def clone(self) -> "_Tensor":
        return _Tensor(self.array.copy())

    def copy_(self, other: "_Tensor") -> "_Tensor":
        np.copyto(self.array, other.array)
        return self

    def data_ptr(self) -> int:
        return int(self.array.__array_interface__["data"][0])

    def detach(self) -> "_Tensor":
        return self

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.array

    def __getitem__(self, key: Any) -> "_Tensor":
        return _Tensor(self.array[key])


def _copy_state(current: Any | None, source: Any) -> Any:
    if current is None:
        return type(source)(*(value.clone() for value in source.__dict__.values()))
    for target, value in zip(current.__dict__.values(), source.__dict__.values(), strict=True):
        target.copy_(value)
    return current


class _DeformableAdapter:
    def __init__(self, session: "_Session", body_count: int, max_vertices: int):
        self.session = session
        shape = (session.num_envs, body_count, max_vertices, 3)
        self.source = DeformableBatchState(*(_Tensor.zeros(shape) for _ in range(3)))
        self.targets = None
        self.last_reset = None

    def read_state(self, state: DeformableBatchState | None) -> DeformableBatchState:
        return _copy_state(state, self.source)

    def set_kinematic_targets(self, targets: Any, *, target_mask: Any | None) -> None:
        self.targets = (targets, target_mask)

    def reset(self, reset_mask: Any, **state: Any) -> dict[str, Any]:
        self.last_reset = (reset_mask, state)
        return self.session.arrays


class _TactileAdapter:
    def __init__(
        self,
        session: "_Session",
        sensor_count: int,
        resolution: tuple[int, int],
        gel_vertices: int,
        marker_count: int,
    ):
        self.session = session
        n, s = session.num_envs, sensor_count
        h, w = resolution
        self.source = TactileBatchState(
            rgb=_Tensor.zeros((n, s, h, w, 3)),
            depth=_Tensor.zeros((n, s, h, w)),
            normal=_Tensor.zeros((n, s, h, w, 3)),
            marker_position=_Tensor.zeros((n, s, marker_count, 2)),
            marker_flow=_Tensor.zeros((n, s, marker_count, 2)),
            contact_force=_Tensor.zeros((n, s, gel_vertices, 3)),
            contact_wrench=_Tensor.zeros((n, s, 6)),
        )
        self.render_count = 0

    def read_state(self, state: TactileBatchState | None) -> TactileBatchState:
        return _copy_state(state, self.source)

    def render(self) -> None:
        self.render_count += 1
        self.source.rgb.array.fill(float(self.render_count))

    def reset(self, reset_mask: Any, **state: Any) -> dict[str, Any]:
        del reset_mask, state
        return self.session.arrays


class _Session:
    def __init__(self, num_envs: int, *, graph_captured: bool = False):
        self.num_envs = num_envs
        self.graph_captured = graph_captured
        self._arrays = {"active": _Tensor.zeros((num_envs,))}
        self.deformable_adapter = None
        self.tactile_adapter = None
        self.step_count = 0
        self.last_reset = None
        self.closed = False

    @property
    def arrays(self) -> dict[str, Any]:
        return self._arrays

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {"newton_iteration": 3, "cg_iteration": 8}

    def create_deformable_view_adapter(self, spec: Any, entries: Any) -> _DeformableAdapter:
        self.deformable_adapter = _DeformableAdapter(
            self, len(spec.body_names), max(int(entry["vertex_count"]) for entry in entries)
        )
        return self.deformable_adapter

    def create_tactile_view_adapter(self, spec: Any, entries: Any) -> _TactileAdapter:
        first = entries[0]
        self.tactile_adapter = _TactileAdapter(
            self,
            len(spec.sensor_names),
            tuple(first["resolution"]),
            int(first["gel_vertex_count"]),
            len(first["marker_positions"]),
        )
        return self.tactile_adapter

    def step(self, actions: Any, *, nsteps: int) -> None:
        del actions
        self.step_count += nsteps

    def reset(self, reset_mask: Any, **state: Any) -> None:
        self.last_reset = (reset_mask, state)

    def close(self) -> None:
        self.closed = True


class _SceneContext:
    def __init__(self):
        self.update = None

    def apply_deformable_vertices(self, bodies: Any, positions: Any, counts: Any) -> None:
        self.update = (tuple(bodies), positions.copy(), tuple(counts))


def _encode_mesh_blob(vertex_count: int, *, scale: float = 1.0) -> tuple[bytes, str]:
    if vertex_count not in (4, 5):
        raise ValueError("test mesh fixture supports four or five vertices")
    vertices = [
        (0.0, 0.0, 0.0),
        (scale, 0.0, 0.0),
        (0.0, scale, 0.0),
        (0.0, 0.0, scale),
    ]
    if vertex_count == 5:
        vertices.append((0.0, 0.0, -scale))
        tetrahedra = (0, 1, 2, 3, 0, 2, 1, 4)
        surface = (
            1, 2, 3,
            0, 3, 2,
            0, 1, 3,
            2, 1, 4,
            0, 4, 1,
            0, 2, 4,
        )
    else:
        tetrahedra = (0, 1, 2, 3)
        surface = (1, 2, 3, 0, 3, 2, 0, 1, 3, 0, 2, 1)
    tetrahedron_count = len(tetrahedra) // 4
    surface_triangle_count = len(surface) // 3
    data = bytearray(b"GOBTIPC1")
    data.extend(
        struct.pack(
            "<IIII",
            1,
            vertex_count,
            tetrahedron_count,
            surface_triangle_count,
        )
    )
    data.extend(struct.pack(f"<{vertex_count * 3}d", *(value for vertex in vertices for value in vertex)))
    data.extend(struct.pack(f"<{len(tetrahedra)}I", *tetrahedra))
    data.extend(struct.pack(f"<{len(surface)}I", *surface))

    topology = bytearray(b"GOBTIPCTOP1\0")
    topology.extend(
        struct.pack(
            "<III", vertex_count, tetrahedron_count, surface_triangle_count
        )
    )
    topology.extend(struct.pack(f"<{len(tetrahedra)}I", *tetrahedra))
    topology.extend(struct.pack(f"<{len(surface)}I", *surface))
    return bytes(data), "sha256:" + hashlib.sha256(topology).hexdigest()


def _encode_surface_mesh_blob() -> bytes:
    data = bytearray(b"GOBTTRI1")
    data.extend(struct.pack("<III", 1, 3, 1))
    data.extend(
        struct.pack(
            "<9d",
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        )
    )
    data.extend(struct.pack("<3I", 0, 1, 2))
    return bytes(data)


def _artifact_mapping(
    *, mismatched_sensor: bool = False, separate_sensor_mesh: bool = False
) -> dict[str, Any]:
    blob_data, topology_digest = _encode_mesh_blob(4)
    padded_blob_data, _ = _encode_mesh_blob(5)
    alternate_blob_data, alternate_topology_digest = _encode_mesh_blob(4, scale=0.5)
    assert alternate_topology_digest == topology_digest
    blob_id = "sha256:" + hashlib.sha256(blob_data).hexdigest()
    padded_blob_id = "sha256:" + hashlib.sha256(padded_blob_data).hexdigest()
    alternate_blob_id = "sha256:" + hashlib.sha256(alternate_blob_data).hexdigest()
    raw_blob_values = [(blob_id, blob_data), (padded_blob_id, padded_blob_data)]
    if separate_sensor_mesh:
        raw_blob_values.append((alternate_blob_id, alternate_blob_data))
    blob_values = sorted(
        raw_blob_values, key=lambda item: item[0]
    )
    resolution_b = [9, 8] if mismatched_sensor else [6, 8]
    identity_transform = {
        "matrix_row_major": [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    def deformable(
        name: str,
        mesh_blob: str,
        vertex_count: int,
        tetrahedron_count: int,
        surface_triangle_count: int,
    ) -> dict[str, Any]:
        return {
            "collision_layer": 1,
            "collision_mask": 0xFFFFFFFF,
            "damping": 0.0,
            "density": 1000.0,
            "kinematic": False,
            "mesh_blob": mesh_blob,
            "name": name,
            "path": f"/world/{name}",
            "poisson_ratio": 0.4,
            "self_collision": False,
            "surface_triangle_count": surface_triangle_count,
            "tetrahedron_count": tetrahedron_count,
            "transform": identity_transform,
            "vertex_count": vertex_count,
            "young_modulus": 100000.0,
        }

    manifest_data = {
        "blobs": [
            {
                "byte_length": len(data),
                "encoding": "gobot.tetrahedral-mesh.le.v1",
                "id": content_id,
                "sha256": content_id,
            }
            for content_id, data in blob_values
        ],
        "deformable_bodies": [
            deformable("pad_a", blob_id, 4, 1, 4),
            deformable("pad_b", padded_blob_id, 5, 2, 6),
        ],
        "format": "gobot-ipc",
        "producer": "gobot",
        "producer_version": "0.1.0",
        "robots": [],
        "scene_name": "world",
        "schema_version": 1,
        "tactile_sensors": [
            {
                "collision_layer": 1,
                "collision_mask": 0xFFFFFFFF,
                "coat_vertex_indices": [0, 1, 2],
                "damping": 0.0,
                "density": 1000.0,
                "enabled": True,
                "far_plane": 0.05,
                "friction_coefficient": 1.0,
                "gel_mesh_blob": alternate_blob_id if separate_sensor_mesh else blob_id,
                "gel_topology_sha256": topology_digest,
                "gel_tetrahedron_count": 1,
                "gel_vertex_count": 4,
                "marker_barycentric": [[0.25, 0.25, 0.25, 0.25]],
                "marker_positions": [[2.0, 3.0]],
                "marker_tetrahedra": [0],
                "name": "taxel_a",
                "near_plane": 0.0,
                "path": "/world/taxel_a",
                "pixel_size": 0.0001,
                "poisson_ratio": 0.4,
                "resolution": [6, 8],
                "rgb_model": "gobot_deterministic_v1",
                "stick_vertex_indices": [3],
                "transform": identity_transform,
                "young_modulus": 500000.0,
            },
            {
                "collision_layer": 1,
                "collision_mask": 0xFFFFFFFF,
                "coat_vertex_indices": [0, 1, 2],
                "damping": 0.0,
                "density": 1000.0,
                "enabled": True,
                "far_plane": 0.05,
                "friction_coefficient": 1.0,
                "gel_mesh_blob": blob_id,
                "gel_topology_sha256": topology_digest,
                "gel_tetrahedron_count": 1,
                "gel_vertex_count": 4,
                "marker_barycentric": [[0.25, 0.25, 0.25, 0.25]],
                "marker_positions": [[2.0, 3.0]],
                "marker_tetrahedra": [0],
                "name": "taxel_b",
                "near_plane": 0.0,
                "path": "/world/taxel_b",
                "pixel_size": 0.0001,
                "poisson_ratio": 0.4,
                "resolution": resolution_b,
                "rgb_model": "gobot_deterministic_v1",
                "stick_vertex_indices": [3],
                "transform": identity_transform,
                "young_modulus": 500000.0,
            },
        ],
    }
    manifest = json.dumps(manifest_data, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "producer": "gobot",
        "producer_version": "0.1.0",
        "format": "gobot-ipc",
        "manifest": manifest,
        "manifest_sha256": "sha256:" + hashlib.sha256(manifest.encode()).hexdigest(),
        "blobs": [
            {
                "id": content_id,
                "encoding": "gobot.tetrahedral-mesh.le.v1",
                "sha256": content_id,
                "data": data,
            }
            for content_id, data in blob_values
        ],
    }


def test_import_is_lightweight_and_artifact_validation_is_strict() -> None:
    assert "torch" not in sys.modules
    assert "warp" not in sys.modules
    assert "newton" not in sys.modules

    mapping = _artifact_mapping()
    artifact = CompiledIpcSceneArtifact.from_mapping(mapping)
    assert artifact.digest == mapping["manifest_sha256"]
    assert len(artifact.blobs) == 2
    assert artifact.to_mapping()["blobs"][0]["data"].startswith(b"GOBTIPC1")
    assert isinstance(artifact.tactile_sensors[0]["resolution"], tuple)
    try:
        artifact.tactile_sensors[0]["resolution"] = (1, 1)
    except TypeError:
        pass
    else:
        raise AssertionError("validated IPC artifact metadata remained mutable")

    changed = dict(mapping)
    changed["manifest_sha256"] = "sha256:" + "0" * 64
    try:
        CompiledIpcSceneArtifact.from_mapping(changed)
    except ValueError as error:
        assert "digest mismatch" in str(error)
    else:
        raise AssertionError("a tampered IPC artifact digest was accepted")

    spaced = dict(mapping)
    spaced["manifest"] = json.dumps(json.loads(mapping["manifest"]), sort_keys=True)
    spaced["manifest_sha256"] = "sha256:" + hashlib.sha256(
        spaced["manifest"].encode()
    ).hexdigest()
    try:
        CompiledIpcSceneArtifact.from_mapping(spaced)
    except ValueError as error:
        assert "non-canonical whitespace" in str(error)
    else:
        raise AssertionError("a non-canonical IPC manifest was accepted")

    duplicate_tet_blob = bytearray(b"GOBTIPC1")
    duplicate_tet_blob.extend(struct.pack("<IIII", 1, 4, 2, 0))
    duplicate_tet_blob.extend(
        struct.pack(
            "<12d",
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        )
    )
    duplicate_tet_blob.extend(struct.pack("<8I", 0, 1, 2, 3, 0, 1, 2, 3))
    try:
        _decode_mesh_blob(bytes(duplicate_tet_blob))
    except ValueError as error:
        assert "duplicate tetrahedron" in str(error)
    else:
        raise AssertionError("an IPC blob with a duplicate tetrahedron was accepted")

    surface = _decode_surface_mesh_blob(_encode_surface_mesh_blob())
    assert surface == {"vertex_count": 3, "triangle_count": 1}
    degenerate_surface = bytearray(_encode_surface_mesh_blob())
    struct.pack_into("<3I", degenerate_surface, len(degenerate_surface) - 12, 0, 1, 1)
    try:
        _decode_surface_mesh_blob(bytes(degenerate_surface))
    except ValueError as error:
        assert "repeated triangle vertex" in str(error)
    else:
        raise AssertionError("a degenerate IPC triangle mesh blob was accepted")


def test_scene_roundtrip_compiles_canonical_artifact() -> None:
    with tempfile.TemporaryDirectory(prefix="gobot-ipc-") as project:
        context = gobot.app.create_context()
        context.set_project_path(project)

        mesh = gobot.TetrahedralMesh()
        mesh.vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        mesh.tetrahedra = [[0, 1, 2, 3]]
        mesh.surface_triangles = []
        mesh.validate()

        root = gobot.create_node("Node3D", "ipc_world")
        body = gobot.create_node("DeformableBody3D", "pad")
        body.mesh = mesh
        body.density = 1060.0
        body.self_collision_enabled = True
        root.add_child(body)

        config = gobot.TactileSensorConfig()
        config.image_width = 16
        config.image_height = 12
        config.coat_vertex_indices = [0, 1, 2]
        config.stick_vertex_indices = [3]
        config.marker_positions = [(4.0, 5.0)]
        config.marker_tetrahedra = [0]
        config.marker_barycentric = [(0.25, 0.25, 0.25, 0.25)]
        sensor = gobot.create_node("TactileSensor3D", "taxel")
        sensor.config = config
        sensor.gel_mesh = mesh
        robot = gobot.create_node("Robot3D", "hand")
        fingertip = gobot.create_node("Link3D", "fingertip")
        fingertip.add_child(
            gobot.create_box_collision(
                "fingertip_collision", (0.02, 0.03, 0.04)
            )
        )
        fingertip.add_child(sensor)
        robot.add_child(fingertip)
        root.add_child(robot)

        scene_path = "res://ipc_roundtrip.jscn"
        gobot.save_scene(root, scene_path)
        serialized = json.loads((Path(project) / "ipc_roundtrip.jscn").read_text())
        assert serialized["__META_TYPE__"] == "SCENE"
        assert serialized["__NODES__"]

        context.load_scene(scene_path)
        assert [child.type for child in context.root.children] == [
            "DeformableBody3D",
            "Robot3D",
        ]
        loaded_body = context.root.find("pad")
        assert loaded_body.mesh.vertex_count == 4
        assert loaded_body.self_collision_enabled
        loaded_sensor = context.root.find("hand/fingertip/taxel")
        assert loaded_sensor.config.image_width == 16
        assert loaded_sensor.config.coat_vertex_indices == [0, 1, 2]
        assert loaded_sensor.gel_mesh.tetrahedron_count == 1

        first_mapping = context.compile_ipc_scene_artifact()
        first = CompiledIpcSceneArtifact.from_mapping(first_mapping)
        second = CompiledIpcSceneArtifact.from_mapping(
            context.compile_ipc_scene_artifact()
        )
        assert first.manifest == second.manifest
        assert len(first.blobs) == 1
        assert first.deformable_bodies[0]["path"].endswith("/pad")
        assert first.tactile_sensors[0]["resolution"] == (12, 16)
        assert first.tactile_sensors[0]["attachment"]["link_path"].endswith(
            "/hand/fingertip"
        )
        assert len(first.robots) == 1
        collision_shapes = first.robots[0]["links"][0]["collision_shapes"]
        assert len(collision_shapes) == 1
        assert collision_shapes[0]["shape_type"] == "box"
        np.testing.assert_allclose(
            collision_shapes[0]["size"], (0.02, 0.03, 0.04), atol=1.0e-8
        )

        positions = np.asarray(
            [[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.1]]],
            dtype=np.float32,
        )
        context.apply_deformable_vertices((loaded_body,), positions, (4,))
        after_sync = CompiledIpcSceneArtifact.from_mapping(
            context.compile_ipc_scene_artifact()
        )
        assert after_sync.digest == first.digest


def test_sensor_fabrication_produces_valid_8x8_marker_scene() -> None:
    fabricated = fabricate_box_gel(resolution=(120, 160))
    assert len(fabricated.vertices) == 200
    assert len(fabricated.tetrahedra) == 486
    assert len(fabricated.marker_positions) == 64
    assert set(fabricated.coat_vertex_indices).isdisjoint(
        fabricated.stick_vertex_indices
    )
    assert all(abs(sum(weights) - 1.0) < 1.0e-12 for weights in fabricated.marker_barycentric)

    with tempfile.TemporaryDirectory(prefix="gobot-ipc-fabrication-") as project:
        context = gobot.app.create_context()
        context.set_project_path(project)
        root = create_fabricated_sensor_scene(resolution=(120, 160))
        gobot.save_scene(root, "res://fabricated_sensor.jscn")
        context.load_scene("res://fabricated_sensor.jscn")
        sensor = context.root.find("tactile_sensor")
        assert sensor.gel_mesh.vertex_count == 200
        assert sensor.gel_mesh.tetrahedron_count == 486
        assert len(sensor.config.coat_vertex_indices) == 100
        assert len(sensor.config.stick_vertex_indices) == 100
        artifact = CompiledIpcSceneArtifact.from_mapping(
            context.compile_ipc_scene_artifact()
        )
        assert artifact.tactile_sensors[0]["resolution"] == (120, 160)
        assert len(artifact.tactile_sensors[0]["marker_positions"]) == 64


def test_views_keep_storage_stable_and_render_is_explicit() -> None:
    session = _Session(4)
    provider = WarpIpcProvider(
        _artifact_mapping(),
        num_envs=4,
        config=WarpIpcConfig(device="cpu", capture_graphs=False),
        _session=session,
    )
    assert provider.capabilities.name == "Warp IPC"
    assert provider.capacities["pt"] == 131072
    assert provider.diagnostics["cg_iteration"] == 8

    deformable = provider.create_deformable_view(body_names=("pad_a", "pad_b"))
    state = deformable.read_state()
    pointers = tuple(value.data_ptr() for value in state.__dict__.values())
    session.deformable_adapter.source.position.array[2, 0, 0] = (1.0, 2.0, 3.0)
    updated = deformable.read_state()
    assert updated is state
    assert tuple(value.data_ptr() for value in updated.__dict__.values()) == pointers
    assert updated.position.shape == (4, 2, 5, 3)
    assert deformable.vertex_counts == (4, 5)

    targets = _Tensor.zeros((4, 2, 5, 3))
    target_mask = _Tensor(np.asarray([True, False, True, False]))
    deformable.set_kinematic_targets(targets, target_mask=target_mask)
    assert session.deformable_adapter.targets == (targets, target_mask)
    deformable.reset(target_mask, position=targets)
    assert session.deformable_adapter.last_reset[0] is target_mask

    context = _SceneContext()
    deformable.bind_scene(context, ("body-a", "body-b"))
    deformable.sync_scene(env_index=2)
    assert context.update[0] == ("body-a", "body-b")
    assert context.update[1].shape == (2, 5, 3)
    assert context.update[2] == (4, 5)

    tactile = provider.create_tactile_view(sensor_names=("taxel_a", "taxel_b"))
    first = tactile.read_state()
    tactile_pointers = tuple(value.data_ptr() for value in first.__dict__.values())
    assert session.tactile_adapter.render_count == 0
    assert first.rgb.shape == (4, 2, 6, 8, 3)
    rendered = tactile.render()
    assert rendered is first
    assert session.tactile_adapter.render_count == 1
    assert tuple(value.data_ptr() for value in rendered.__dict__.values()) == tactile_pointers
    assert np.all(rendered.rgb.array == 1.0)

    provider.step(nsteps=3)
    assert session.step_count == 3
    try:
        provider.step(nsteps=1.5)
    except TypeError as error:
        assert "integer" in str(error)
    else:
        raise AssertionError("a fractional Warp IPC step count was accepted")
    assert session.step_count == 3
    provider.reset(target_mask, seed=7)
    assert session.last_reset == (target_mask, {"seed": 7})
    provider.close()
    assert session.closed
    try:
        deformable.read_state()
    except RuntimeError as error:
        assert "stale" in str(error)
    else:
        raise AssertionError("a view remained usable after provider close")


def test_view_rejects_replaced_state_storage() -> None:
    session = _Session(1)
    provider = WarpIpcProvider(
        _artifact_mapping(),
        num_envs=1,
        config=WarpIpcConfig(device="cpu", capture_graphs=False),
        _session=session,
    )
    deformable = provider.create_deformable_view(body_names=("pad_a",))
    deformable.read_state()
    object.__setattr__(
        session.deformable_adapter.source,
        "position",
        _Tensor.zeros((1, 1, 4, 3)),
    )
    session.deformable_adapter.read_state = (
        lambda state: session.deformable_adapter.source
    )
    try:
        deformable.read_state()
    except GraphInvalidatedError as error:
        assert "state storage changed" in str(error)
    else:
        raise AssertionError("replaced deformable view storage was accepted")
    provider.close()


def test_provider_rejects_mismatched_views_and_invalidated_storage() -> None:
    topology_session = _Session(1)
    topology_provider = WarpIpcProvider(
        _artifact_mapping(separate_sensor_mesh=True),
        num_envs=1,
        config=WarpIpcConfig(device="cpu", capture_graphs=False),
        _session=topology_session,
    )
    topology_provider.create_tactile_view(sensor_names=("taxel_a", "taxel_b"))
    topology_provider.close()

    mismatch_session = _Session(1)
    mismatch_provider = WarpIpcProvider(
        _artifact_mapping(mismatched_sensor=True),
        num_envs=1,
        config=WarpIpcConfig(device="cpu", capture_graphs=False),
        _session=mismatch_session,
    )
    try:
        mismatch_provider.create_tactile_view(sensor_names=("taxel_a", "taxel_b"))
    except ValueError as error:
        assert "identical resolution" in str(error)
    else:
        raise AssertionError("a heterogeneous tactile view was accepted")
    mismatch_provider.close()

    session = _Session(1)
    provider = WarpIpcProvider(
        _artifact_mapping(),
        num_envs=1,
        config=WarpIpcConfig(device="cpu", capture_graphs=False),
        _session=session,
    )
    session._arrays["active"] = _Tensor.zeros((1,))
    try:
        provider.step()
    except GraphInvalidatedError as error:
        assert "storage changed" in str(error)
    else:
        raise AssertionError("replaced provider storage did not invalidate the graph contract")
    provider.close()

    uncaptured = _Session(1, graph_captured=False)
    try:
        WarpIpcProvider(
            _artifact_mapping(),
            num_envs=1,
            config=WarpIpcConfig(device="cpu", capture_graphs=True),
            _session=uncaptured,
        )
    except ProviderUnavailableError as error:
        assert "graph capture" in str(error)
    else:
        raise AssertionError("explicit graph capture accepted an uncaptured session")
    assert uncaptured.closed


if __name__ == "__main__":
    test_import_is_lightweight_and_artifact_validation_is_strict()
    test_scene_roundtrip_compiles_canonical_artifact()
    test_sensor_fabrication_produces_valid_8x8_marker_scene()
    test_views_keep_storage_stable_and_render_is_explicit()
    test_view_rejects_replaced_state_storage()
    test_provider_rejects_mismatched_views_and_invalidated_storage()
