from __future__ import annotations

from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile

import mujoco
import numpy as np
from scipy.spatial import cKDTree
import trimesh

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "conveyor_packages"
SCENE = EXAMPLE / "conveyor_packages.jscn"


def _matrix_vector(
    properties: dict[str, object],
    name: str,
    default: tuple[float, float, float],
) -> np.ndarray:
    value = properties.get(name)
    if not isinstance(value, dict) or "matrix_data" not in value:
        return np.asarray(default, dtype=np.float64)
    return np.asarray(
        value["matrix_data"]["storage"], dtype=np.float64
    )


def _rotation_matrix(rotation_degrees: np.ndarray) -> np.ndarray:
    x, y, z = np.radians(rotation_degrees)
    rotation_x = np.asarray(
        ((1.0, 0.0, 0.0),
         (0.0, math.cos(x), -math.sin(x)),
         (0.0, math.sin(x), math.cos(x))),
        dtype=np.float64,
    )
    rotation_y = np.asarray(
        ((math.cos(y), 0.0, math.sin(y)),
         (0.0, 1.0, 0.0),
         (-math.sin(y), 0.0, math.cos(y))),
        dtype=np.float64,
    )
    rotation_z = np.asarray(
        ((math.cos(z), -math.sin(z), 0.0),
         (math.sin(z), math.cos(z), 0.0),
         (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    return rotation_z @ rotation_y @ rotation_x


def _scene_world_transform_list(
    scene: dict[str, object],
) -> list[np.ndarray]:
    transforms: list[np.ndarray] = []
    for node in scene["__NODES__"]:
        properties = node.get("properties", {})
        local = np.eye(4, dtype=np.float64)
        local[:3, :3] = _rotation_matrix(
            _matrix_vector(
                properties, "rotation_degrees", (0.0, 0.0, 0.0)
            )
        ) @ np.diag(
            _matrix_vector(properties, "scale", (1.0, 1.0, 1.0))
        )
        local[:3, 3] = _matrix_vector(
            properties, "position", (0.0, 0.0, 0.0)
        )
        parent = int(node["parent"])
        world = transforms[parent] @ local if parent >= 0 else local
        transforms.append(world)
    return transforms


def _descendant_world_transform(
    scene: dict[str, object], ancestor_name: str, node_name: str
) -> np.ndarray:
    index = _descendant_node_index(scene, ancestor_name, node_name)
    return _scene_world_transform_list(scene)[index]


def _descendant_node_index(
    scene: dict[str, object], ancestor_name: str, node_name: str
) -> int:
    nodes = scene["__NODES__"]
    ancestor = next(
        index
        for index, node in enumerate(nodes)
        if node["name"] == ancestor_name
    )
    for index, node in enumerate(nodes):
        if node["name"] != node_name:
            continue
        parent = index
        while parent >= 0 and parent != ancestor:
            parent = int(nodes[parent]["parent"])
        if parent == ancestor:
            return index
    raise KeyError(f"{node_name!r} is not below {ancestor_name!r}")


@lru_cache(maxsize=None)
def _mesh_vertices(path: Path) -> np.ndarray:
    mesh = trimesh.load_mesh(path, process=False)
    return np.asarray(mesh.vertices, dtype=np.float64)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _builder():
    return _load_module(
        "gobot_conveyor_test_builder", EXAMPLE / "build_scene.py"
    )


@lru_cache(maxsize=1)
def _profile():
    return _load_module(
        "gobot_conveyor_test_profile", EXAMPLE / "conveyor_profile.py"
    )


@lru_cache(maxsize=1)
def _forces():
    return _load_module(
        "gobot_conveyor_test_forces", EXAMPLE / "conveyor_forces.py"
    )


@lru_cache(maxsize=1)
def _play():
    return _load_module(
        "gobot_conveyor_test_play", EXAMPLE / "conveyor_packages_play.py"
    )


def _tetrahedral_mass(mesh: object, density: float) -> float:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    tetrahedra = np.asarray(mesh.tetrahedra, dtype=np.int64)
    points = vertices[tetrahedra]
    signed_six_volumes = np.einsum(
        "ij,ij->i",
        points[:, 1] - points[:, 0],
        np.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 0]),
    )
    return float(np.abs(signed_six_volumes).sum() / 6.0) * float(density)


def _surface_mass(mesh: object, thickness: float, density: float) -> float:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    points = vertices[triangles]
    area = 0.5 * np.linalg.norm(
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
        axis=1,
    ).sum()
    return float(area) * float(thickness) * float(density)


def test_scene_is_reproducible() -> None:
    builder = _builder()
    with tempfile.TemporaryDirectory(
        prefix="gobot-conveyor-packages-scene-"
    ) as temporary:
        output = Path(temporary)
        generated = builder.build_scene(output)
        assert generated.read_bytes() == SCENE.read_bytes()
        assert {
            path.name for path in output.iterdir() if path.is_file()
        } == {
            "README.md",
            "build_scene.py",
            "conveyor_forces.py",
            "conveyor_packages.jscn",
            "conveyor_packages_batch.py",
            "conveyor_packages_play.py",
            "conveyor_profile.py",
            "project.gobot",
        }
        assert (output / "assets").is_symlink()
        assert (output / "assets").resolve() == (builder.HERE / "assets").resolve()


def _legacy_scene_has_play_script_and_industrial_visuals() -> None:
    scene = json.loads(SCENE.read_text(encoding="utf-8"))
    roots = [node for node in scene["__NODES__"] if node["parent"] == -1]
    assert len(roots) == 1
    assert roots[0]["properties"]["script"] == (
        "ExtResource(conveyor_packages_play_script)"
    )
    resource = next(
        value
        for value in scene["__EXT_RESOURCES__"]
        if value["__ID__"] == "conveyor_packages_play_script"
    )
    assert resource["__PATH__"] == "res://conveyor_packages_play.py"
    names = {node["name"] for node in scene["__NODES__"]}
    assert {
        "scanner_crossbar",
        "scanner_camera",
        "worktable_surface_visual",
        "worktable_surface_collision",
        "incoming_package_lane",
        "manual_sorting_zone",
        "worktable_transfer_lip",
        "end_stop_visual",
        "end_stop_collision",
        "openarm_bimanual",
        "openarm_base_pedestal",
        "openarm_shoulder_shroud",
        "openarm_camera_mast",
        "openarm_sensor_head",
        "openarm_left_allegro_palm",
        "openarm_right_allegro_palm",
        "openarm_left_allegro_thumb_distal",
        "openarm_right_allegro_thumb_distal",
        "openarm_left_allegro_ff_tip",
        "openarm_right_allegro_ff_tip",
        "openarm_left_allegro_rf_tip",
        "openarm_right_allegro_rf_tip",
        "openarm_left_allegro_ff_tip_proxy",
        "openarm_right_allegro_thumb_tip_proxy",
        "openarm_left_sweep_edge_proxy",
        "openarm_right_sweep_edge_proxy",
    } <= names
    resource_paths = {
        resource["__PATH__"] for resource in scene["__EXT_RESOURCES__"]
    }
    assert {
        "res://assets/wonik_allegro/assets/base_link_left.stl",
        "res://assets/wonik_allegro/assets/base_link.stl",
        "res://assets/wonik_allegro/assets/link_15.0_tip.stl",
    } <= resource_paths
    assert math.isclose(_builder().ALLEGRO_HAND_SCALE, 1.50)
    assert math.isclose(_builder().HAND_DOWN_ANGLE_DEGREES, 60.0)
    assert math.isclose(_builder().HAND_PRESS_ANGLE_DEGREES, 90.0)
    np.testing.assert_allclose(
        _builder().HAND_PUSH_PAD_SIZE, (0.180, 0.025, 0.260)
    )
    np.testing.assert_allclose(
        _builder().HAND_SWEEP_EDGE_SIZE, (0.180, 0.025, 0.105)
    )
    assert math.isclose(
        float(np.linalg.det(_builder().HAND_PUSH_PAD_REFERENCE_ROTATION)),
        1.0,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        float(np.linalg.det(_builder().HAND_VISUAL_WORLD_ADJUSTMENT)),
        1.0,
        abs_tol=1.0e-12,
    )
    pressure_normal = _builder().HAND_PUSH_PAD_WORLD_ROTATION[:, 1]
    assert abs(float(pressure_normal[1])) < 1.0e-12
    assert pressure_normal[2] < -0.99
    assert math.isclose(
        float(np.linalg.det(_builder().HAND_PUSH_PAD_WORLD_ROTATION)),
        1.0,
        abs_tol=1.0e-12,
    )
    for world_offset, local_offset in zip(
        _builder().HAND_PUSH_PAD_WORLD_OFFSETS,
        _builder().HAND_PUSH_PAD_LOCAL_OFFSETS,
        strict=True,
    ):
        np.testing.assert_allclose(
            _builder().HAND_PUSH_PAD_REFERENCE_ROTATION
            @ np.asarray(local_offset),
            world_offset,
            atol=1.0e-12,
        )
    for world_offset, local_offset in zip(
        _builder().HAND_SWEEP_EDGE_WORLD_OFFSETS,
        _builder().HAND_SWEEP_EDGE_LOCAL_OFFSETS,
        strict=True,
    ):
        np.testing.assert_allclose(
            _builder().HAND_PUSH_PAD_REFERENCE_ROTATION
            @ np.asarray(local_offset),
            world_offset,
            atol=1.0e-12,
        )
    world_transforms = _scene_world_transforms(scene)
    for side in ("left", "right"):
        palm_transform = world_transforms[
            f"openarm_{side}_allegro_palm"
        ]
        palm_normal = palm_transform[:3, 0]
        palm_normal /= np.linalg.norm(palm_normal)
        if palm_normal[2] > 0.0:
            palm_normal *= -1.0
        assert abs(float(palm_normal[0])) < 0.02
        assert abs(float(palm_normal[1])) < 0.02
        assert palm_normal[2] < -0.99
        palm_position = palm_transform[:3, 3]
        fingertip_positions = np.stack(
            tuple(
                world_transforms[
                    f"openarm_{side}_allegro_{finger}_tip"
                ][:3, 3]
                for finger in ("ff", "mf", "rf")
            )
        )
        finger_direction = fingertip_positions.mean(axis=0) - palm_position
        assert finger_direction[1] > 0.10
        assert abs(float(finger_direction[2])) < finger_direction[1]
    belt_surface = next(
        node
        for node in scene["__NODES__"]
        if node["name"] == "belt_surface"
    )
    np.testing.assert_allclose(
        belt_surface["properties"]["position"]["matrix_data"]["storage"],
        (
            _builder().BELT_CENTER_X,
            _builder().BELT_CENTER_Y,
            _builder().BELT_CENTER_Z,
        ),
    )
    worktable = next(
        node
        for node in scene["__NODES__"]
        if node["name"] == "worktable_surface_collision"
    )
    np.testing.assert_allclose(
        worktable["properties"]["position"]["matrix_data"]["storage"],
        (
            _builder().WORKTABLE_CENTER_X,
            _builder().WORKTABLE_CENTER_Y,
            _builder().WORKTABLE_TOP_Z
            - 0.5 * _builder().WORKTABLE_THICKNESS,
        ),
    )
    robot = next(
        node
        for node in scene["__NODES__"]
        if node["name"] == "openarm_bimanual"
    )
    np.testing.assert_allclose(
        robot["properties"]["position"]["matrix_data"]["storage"],
        _builder().OPENARM_ROOT_POSITION,
    )
    soft_specs = {
        str(spec["name"]): spec for spec in _builder().SOFT_PACKAGE_SPECS
    }
    blue_spec = soft_specs["soft_mailer_blue"]
    fill_spec = soft_specs["soft_mailer_blue_fill"]
    yellow_spec = soft_specs["soft_pouch_yellow"]
    yellow_fill_spec = soft_specs["soft_pouch_yellow_fill"]
    rigid_specs = {
        str(spec["name"]): spec for spec in _builder().RIGID_BOX_SPECS
    }
    assert (
        float(rigid_specs["carton_small"]["position"][0])
        < float(yellow_spec["position"][0])
        < float(blue_spec["position"][0])
        < float(rigid_specs["carton_wide"]["position"][0])
        < float(rigid_specs["carton_tall"]["position"][0])
    )
    carton_right_edge = (
        float(rigid_specs["carton_small"]["position"][0])
        + 0.5 * float(rigid_specs["carton_small"]["size"][0])
    )
    yellow_left_edge = (
        float(yellow_spec["position"][0])
        - 0.5 * float(yellow_spec["size"][0])
    )
    assert yellow_left_edge - carton_right_edge >= 0.30
    belt_near_edge = (
        _builder().BELT_CENTER_Y - 0.5 * _builder().BELT_WIDTH
    )
    assert (
        float(rigid_specs["carton_small"]["position"][1])
        + 0.5 * float(rigid_specs["carton_small"]["size"][1])
        < belt_near_edge
    )
    assert (
        float(yellow_spec["position"][1])
        + 0.5 * float(yellow_spec["size"][1])
        < belt_near_edge
    )
    assert (
        float(blue_spec["position"][1])
        + 0.5 * float(blue_spec["size"][1])
        < belt_near_edge
    )
    for name in ("carton_wide", "carton_tall"):
        assert abs(
            float(rigid_specs[name]["position"][1])
            - _builder().BELT_CENTER_Y
        ) < 0.5 * _builder().BELT_WIDTH
    assert math.isclose(
        float(blue_spec["position"][2]),
        _builder().WORKTABLE_TOP_Z + 0.0461,
    )
    assert math.isclose(float(blue_spec["position"][1]), -0.375)
    assert math.isclose(_builder().WORKTABLE_SLIDING_FRICTION, 0.30)
    assert blue_spec["model"] == "thin_shell"
    assert float(blue_spec["young_modulus"]) >= 1.0e5
    assert float(blue_spec["bending_stiffness"]) <= 5.0e-4
    assert float(blue_spec["thickness"]) <= 1.5e-3
    assert float(blue_spec["damping"]) >= 5.0
    assert tuple(blue_spec["cells"]) == (18, 13)
    assert fill_spec["model"] == "volumetric"
    assert not bool(fill_spec["visible"])
    assert math.isclose(
        float(blue_spec["position"][2]) - float(fill_spec["position"][2]),
        0.010,
    )
    assert float(fill_spec["young_modulus"]) <= 1.0e4
    assert tuple(fill_spec["cells"]) == (8, 6, 3)
    assert yellow_spec["model"] == "thin_shell"
    assert tuple(yellow_spec["cells"]) == (28, 21)
    assert float(yellow_spec["size"][2]) > float(blue_spec["size"][2])
    assert yellow_fill_spec["model"] == "volumetric"
    assert tuple(yellow_fill_spec["cells"]) == (11, 8, 5)
    assert not bool(yellow_fill_spec["visible"])

    yellow_fill_mesh = builder._soft_package_mesh(
        yellow_fill_spec["size"],
        yellow_fill_spec["cells"],
        side_rounding=float(yellow_fill_spec["side_rounding"]),
    )
    yellow_fill_vertices = np.asarray(
        yellow_fill_mesh.vertices, dtype=np.float64
    )
    yellow_fill_layers = yellow_fill_vertices.reshape(
        yellow_fill_spec["cells"][2] + 1,
        -1,
        3,
    )
    middle_layer = yellow_fill_layers[
        yellow_fill_spec["cells"][2] // 2
    ]
    for axis in (0, 1):
        middle_extent = float(np.ptp(middle_layer[:, axis]))
        bottom_extent = float(np.ptp(yellow_fill_layers[0, :, axis]))
        top_extent = float(np.ptp(yellow_fill_layers[-1, :, axis]))
        assert middle_extent > 1.08 * bottom_extent
        assert middle_extent > 1.08 * top_extent
    blue_mesh = _builder()._soft_mailer_shell_mesh(
        blue_spec["size"], blue_spec["cells"]
    )
    blue_vertices = np.asarray(blue_mesh.vertices, dtype=np.float64)
    blue_triangles = np.asarray(blue_mesh.triangles, dtype=np.int64)
    layer_size = (blue_spec["cells"][0] + 1) * (
        blue_spec["cells"][1] + 1
    )
    bottom_z = blue_vertices[:layer_size, 2]
    top_z = blue_vertices[layer_size:, 2]
    assert float(top_z.max() - bottom_z.min()) > 0.08
    assert float(np.ptp(top_z)) > 0.04
    edge_counts: dict[tuple[int, int], int] = {}
    for triangle in blue_triangles:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    assert edge_counts
    assert set(edge_counts.values()) == {2}
    initial_bottom_gap = (
        float(blue_spec["position"][2])
        + float(bottom_z.min())
        - _builder().WORKTABLE_TOP_Z
    )
    assert math.isclose(initial_bottom_gap, 0.011, abs_tol=1.0e-6)
    assert (
        _builder().OPENARM_ROOT_POSITION[1]
        < _builder().BELT_CENTER_Y - 0.5 * _builder().BELT_WIDTH
    )
    collision_nodes = {
        node["name"]: node["properties"]
        for node in scene["__NODES__"]
        if node["name"]
        in {
            "worktable_surface_collision",
            "openarm_left_palm_proxy",
            "soft_mailer_blue",
        }
    }
    assert (
        collision_nodes["worktable_surface_collision"]["collision_layer"],
        collision_nodes["worktable_surface_collision"]["collision_mask"],
    ) == (
        _builder().RIGID_COLLISION_LAYER,
        _builder().RIGID_COLLISION_MASK,
    )
    assert (
        collision_nodes["openarm_left_palm_proxy"]["collision_layer"],
        collision_nodes["openarm_left_palm_proxy"]["collision_mask"],
    ) == (
        _builder().HAND_COLLISION_LAYER,
        _builder().HAND_COLLISION_MASK,
    )
    assert (
        collision_nodes["soft_mailer_blue"]["collision_layer"],
        collision_nodes["soft_mailer_blue"]["collision_mask"],
    ) == (
        _builder().DEFORMABLE_COLLISION_LAYER,
        _builder().DEFORMABLE_COLLISION_MASK,
    )
    assert (
        _builder().WORKTABLE_CENTER_Y
        + 0.5 * _builder().WORKTABLE_DEPTH
        < _builder().BELT_CENTER_Y - 0.5 * _builder().BELT_WIDTH
    )
    assert len({name for name in names if name.startswith("belt_marker_")}) >= 12
    wrist_joints = [
        node
        for node in scene["__NODES__"]
        if node["name"] in {"openarm_left_joint7", "openarm_right_joint7"}
    ]
    assert len(wrist_joints) == 2
    for wrist in wrist_joints:
        properties = wrist["properties"]
        assert properties["drive_mode"] == "Position"
        assert math.isclose(properties["drive_stiffness"], 42.0)
        assert math.isclose(properties["drive_damping"], 5.0)
        assert math.isclose(properties["effort_limit"], 7.0)
    left_link1 = next(
        node
        for node in scene["__NODES__"]
        if node["name"] == "openarm_left_link1"
    )
    baked_y_degrees = left_link1["properties"]["rotation_degrees"][
        "matrix_data"
    ]["storage"][1]
    assert math.isclose(
        baked_y_degrees,
        math.degrees(_builder().ARM_INITIAL_POSES[0][0]),
    )
    assert not math.isclose(baked_y_degrees, 0.0)
    shoulder_mounts = {
        node["name"]: node["properties"]["position"]["matrix_data"][
            "storage"
        ]
        for node in scene["__NODES__"]
        if node["name"]
        in {
            "openarm_left_base_link_mount_joint",
            "openarm_right_base_link_mount_joint",
        }
    }
    np.testing.assert_allclose(
        shoulder_mounts["openarm_left_base_link_mount_joint"],
        (0.0, 0.22, 0.94),
    )
    np.testing.assert_allclose(
        shoulder_mounts["openarm_right_base_link_mount_joint"],
        (0.0, -0.22, 0.94),
    )
    assert all(
        not resource["__PATH__"].startswith("res://../")
        for resource in scene["__EXT_RESOURCES__"]
    )


def _legacy_allegro_visual_meshes_are_enclosed_by_contact_proxies() -> None:
    builder = _builder()
    scene = json.loads(SCENE.read_text(encoding="utf-8"))
    nodes = {str(node["name"]): node for node in scene["__NODES__"]}
    transforms = _scene_world_transforms(scene)
    external_resources = {
        str(resource["__ID__"]): str(resource["__PATH__"])
        for resource in scene["__EXT_RESOURCES__"]
    }
    sub_resources = {
        str(resource["__ID__"]): resource
        for resource in scene["__SUB_RESOURCES__"]
    }

    proxy_names = {
        name
        for name in nodes
        if name.startswith("openarm_")
        and "_allegro_" in name
        and name.endswith("_proxy")
    }
    assert len(proxy_names) == 16
    visual_names = {
        name
        for name in nodes
        if re.fullmatch(
            r"openarm_(left|right)_allegro_"
            r"(ff|mf|rf|thumb)_(base|proximal|medial|distal|tip)",
            name,
        )
    }
    assert len(visual_names) == 40
    tolerance = 5.0e-7
    for visual_name in sorted(visual_names):
        name_match = re.fullmatch(
            r"(openarm_(left|right)_allegro_(ff|mf|rf|thumb))_"
            r"(base|proximal|medial|distal|tip)",
            visual_name,
        )
        assert name_match is not None
        segment = name_match.group(4)
        proxy_group = "root" if segment in {"base", "proximal"} else "tip"
        proxy_name = f"{name_match.group(1)}_{proxy_group}_proxy"
        assert proxy_name in proxy_names
        visual = nodes[visual_name]
        proxy = nodes[proxy_name]

        mesh_match = re.fullmatch(
            r"ExtResource\(([^)]+)\)", str(visual["properties"]["mesh"])
        )
        shape_match = re.fullmatch(
            r"SubResource\(([^)]+)\)", str(proxy["properties"]["shape"])
        )
        assert mesh_match is not None
        assert shape_match is not None
        mesh_path = external_resources[mesh_match.group(1)]
        assert mesh_path.startswith("res://")
        mesh = trimesh.load_mesh(
            EXAMPLE / mesh_path.removeprefix("res://"), process=False
        )
        vertices = np.column_stack(
            (np.asarray(mesh.vertices, dtype=np.float64), np.ones(len(mesh.vertices)))
        )
        world_vertices = transforms[visual_name] @ vertices.T
        proxy_vertices = (
            np.linalg.inv(transforms[proxy_name]) @ world_vertices
        ).T[:, :3]
        shape = sub_resources[shape_match.group(1)]
        half_size = 0.5 * _matrix_vector(shape, "size", (0.0, 0.0, 0.0))
        clearance = half_size - np.abs(proxy_vertices)
        assert float(np.min(clearance)) >= (
            builder.ALLEGRO_VISUAL_PROXY_MARGIN - tolerance
        ), proxy_name

        properties = proxy["properties"]
        assert (
            properties["collision_layer"], properties["collision_mask"]
        ) == (builder.HAND_COLLISION_LAYER, builder.HAND_COLLISION_MASK)
    for proxy_name in proxy_names:
        proxy = nodes[proxy_name]
        parent = scene["__NODES__"][int(proxy["parent"])]
        expected_link = (
            "ee_link2" if "_thumb_" in proxy_name else "ee_link1"
        )
        assert str(parent["name"]).endswith(expected_link)


def test_scene_compiles_to_native_superdex_conveyor_contract() -> None:
    builder = _builder()
    scene = json.loads(SCENE.read_text(encoding="utf-8"))
    nodes = scene["__NODES__"]

    assert not any(node["type"] == "PhysicsCoupling3D" for node in nodes)
    assert {
        str(node["name"])
        for node in nodes
        if node["type"] == "Robot3D"
    } == {
        "warehouse_frame",
        "conveyor",
        *builder.LEAP_ROBOT_NAMES,
    }
    assert {
        str(node["name"])
        for node in nodes
        if node["type"] == "RigidBody3D"
    } == {
        str(spec["name"]) for spec in builder.RIGID_BOX_SPECS
    }

    deformables = [
        node for node in nodes if node["type"] == "DeformableBody3D"
    ]
    assert [str(node["name"]) for node in deformables] == [
        str(spec["name"]) for spec in builder.SOFT_PACKAGE_SPECS
    ]
    for node, spec in zip(
        deformables, builder.SOFT_PACKAGE_SPECS, strict=True
    ):
        properties = node["properties"]
        model = str(spec.get("model", "volumetric"))
        assert properties["model"] == (
            "ThinShell" if model == "thin_shell" else "Volumetric"
        )
        assert str(properties["physics_material"]).startswith("SubResource(")
        internal_fill = bool(spec.get("internal_fill", False))
        assert (
            properties["collision_layer"], properties["collision_mask"]
        ) == (
            (
                builder.DEFORMABLE_FILL_COLLISION_LAYER
                if internal_fill
                else builder.DEFORMABLE_COLLISION_LAYER
            ),
            (
                builder.DEFORMABLE_FILL_COLLISION_MASK
                if internal_fill
                else builder.DEFORMABLE_COLLISION_MASK
            ),
        )
        assert bool(properties["self_collision_enabled"]) == bool(
            model == "thin_shell"
        )
        if model == "thin_shell":
            assert properties["mesh"] is None
            assert properties["surface_mesh"] is not None
            assert math.isclose(
                float(properties["bending_stiffness"]),
                float(spec["bending_stiffness"]),
                rel_tol=1.0e-6,
            )
        else:
            assert properties["mesh"] is not None


def test_scene_has_play_script_and_leap_hands() -> None:
    builder = _builder()
    scene = json.loads(SCENE.read_text(encoding="utf-8"))
    roots = [node for node in scene["__NODES__"] if node["parent"] == -1]
    assert len(roots) == 1
    assert roots[0]["properties"]["script"] == (
        "ExtResource(conveyor_packages_play_script)"
    )
    resource = next(
        value
        for value in scene["__EXT_RESOURCES__"]
        if value["__ID__"] == "conveyor_packages_play_script"
    )
    assert resource["__PATH__"] == "res://conveyor_packages_play.py"

    names = {str(node["name"]) for node in scene["__NODES__"]}
    assert {
        "scanner_crossbar",
        "scanner_camera",
        "worktable_surface_visual",
        "worktable_surface_collision",
        "incoming_package_lane",
        "end_stop_visual",
        "end_stop_collision",
        "leap_left",
        "leap_right",
        "leap_left_if_tip_collision",
        "leap_right_if_tip_collision",
        "leap_left_th_tip_collision",
        "leap_right_th_tip_collision",
    } <= names
    assert "manual_sorting_zone" not in names
    assert "worktable_transfer_lip" not in names
    assert not any("openarm" in name.lower() for name in names)
    assert not any("allegro" in name.lower() for name in names)

    external_resource_paths = {
        str(resource["__ID__"]): str(resource["__PATH__"])
        for resource in scene["__EXT_RESOURCES__"]
    }
    resource_paths = set(external_resource_paths.values())
    assert {
        "res://assets/leap_hand/assets/palm_left.obj",
        "res://assets/leap_hand/assets/palm_right.obj",
        "res://assets/leap_hand/assets/tip.obj",
        "res://assets/leap_hand/assets/thumb_tip.obj",
    } <= resource_paths
    assert not any(
        "openarm" in path or "allegro" in path for path in resource_paths
    )
    assert all(not path.startswith("res://../") for path in resource_paths)
    for filename in ("left_hand.xml", "right_hand.xml", "SOURCE.md", "LICENSE"):
        assert (builder.LEAP_ASSET_ROOT / filename).is_file()

    assert math.isclose(
        float(np.linalg.det(builder.HAND_PALM_ALIGNMENT_ROTATION)),
        1.0,
        abs_tol=1.0e-12,
    )
    scene_nodes = {
        str(node["name"]): node
        for node in scene["__NODES__"]
        if str(node["name"]).startswith("leap_")
    }
    for side_index, side in enumerate(builder.HAND_SIDES):
        robot_name = f"leap_{side}"
        robot = scene_nodes[robot_name]
        assert robot["properties"]["source_path"] == (
            f"res://assets/leap_hand/{builder.LEAP_MJCF_BY_SIDE[side_index]}"
        )
        np.testing.assert_allclose(
            _matrix_vector(
                robot["properties"], "position", (0.0, 0.0, 0.0)
            ),
            builder.HAND_ROOT_POSITIONS[side_index],
            atol=1.0e-7,
        )

        # Compare the generated hand directly with Menagerie's MJCF forward
        # kinematics. This covers every coupled link and rendered mesh, so a
        # quaternion-order, hierarchy, or left/right import regression cannot
        # silently turn the hand into a plausible-looking but invalid model.
        source_model = mujoco.MjModel.from_xml_path(
            str(
                builder.LEAP_ASSET_ROOT
                / builder.LEAP_MJCF_BY_SIDE[side_index]
            )
        )
        source_data = mujoco.MjData(source_model)
        mujoco.mj_forward(source_model, source_data)
        root_position = np.asarray(
            builder.HAND_ROOT_POSITIONS[side_index], dtype=np.float64
        )
        alignment = np.asarray(
            builder.HAND_PALM_ALIGNMENT_ROTATION, dtype=np.float64
        )
        for link_name in builder.LEAP_CONTACT_LINK_NAMES:
            body_id = mujoco.mj_name2id(
                source_model, mujoco.mjtObj.mjOBJ_BODY, link_name
            )
            assert body_id >= 0
            imported = _descendant_world_transform(
                scene, robot_name, link_name
            )
            np.testing.assert_allclose(
                imported[:3, 3],
                root_position + alignment @ source_data.xpos[body_id],
                atol=2.0e-6,
            )
            np.testing.assert_allclose(
                imported[:3, :3],
                alignment @ source_data.xmat[body_id].reshape(3, 3),
                atol=2.0e-6,
            )

        for geom_id in range(source_model.ngeom):
            if source_model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            geom_name = mujoco.mj_id2name(
                source_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            assert geom_name is not None
            imported_index = _descendant_node_index(
                scene, robot_name, geom_name
            )
            imported = _scene_world_transform_list(scene)[imported_index]
            mesh_reference = str(
                scene["__NODES__"][imported_index]["properties"]["mesh"]
            )
            assert mesh_reference.startswith("ExtResource(")
            assert mesh_reference.endswith(")")
            mesh_resource_id = mesh_reference[len("ExtResource(") : -1]
            mesh_path = external_resource_paths[mesh_resource_id]
            assert mesh_path.startswith("res://")
            raw_vertices = _mesh_vertices(
                EXAMPLE / mesh_path.removeprefix("res://")
            )
            imported_vertices = (
                imported[:3, :3] @ raw_vertices.T
            ).T + imported[:3, 3]

            mesh_id = int(source_model.geom_dataid[geom_id])
            vertex_address = int(source_model.mesh_vertadr[mesh_id])
            vertex_count = int(source_model.mesh_vertnum[mesh_id])
            compiled_vertices = np.asarray(
                source_model.mesh_vert[
                    vertex_address : vertex_address + vertex_count
                ],
                dtype=np.float64,
            )
            source_vertices = (
                source_data.geom_xmat[geom_id].reshape(3, 3)
                @ compiled_vertices.T
            ).T + source_data.geom_xpos[geom_id]
            expected_vertices = (
                alignment @ source_vertices.T
            ).T + root_position

            imported_to_expected = cKDTree(expected_vertices).query(
                imported_vertices
            )[0]
            expected_to_imported = cKDTree(imported_vertices).query(
                expected_vertices
            )[0]
            assert float(imported_to_expected.max()) <= 2.0e-5, geom_name
            assert float(expected_to_imported.max()) <= 2.0e-5, geom_name

        palm = _descendant_world_transform(scene, robot_name, "palm")
        np.testing.assert_allclose(
            palm[:3, 2], (0.0, 0.0, 1.0), atol=1.0e-6
        )
        finger_vectors = []
        for finger_name in ("if", "mf", "rf"):
            base = _descendant_world_transform(
                scene, robot_name, f"{finger_name}_bs"
            )[:3, 3]
            distal = _descendant_world_transform(
                scene, robot_name, f"{finger_name}_ds"
            )[:3, 3]
            finger_vectors.append(distal - base)
        finger_vectors = np.stack(finger_vectors)
        assert np.all(finger_vectors[:, 1] > 0.08)
        assert np.all(np.abs(finger_vectors[:, 2]) < 0.02)

        index_tip = _descendant_world_transform(
            scene, robot_name, "if_ds"
        )[:3, 3]
        ring_tip = _descendant_world_transform(
            scene, robot_name, "rf_ds"
        )[:3, 3]
        thumb_tip = _descendant_world_transform(
            scene, robot_name, "th_ds"
        )[:3, 3]
        inward_sign = 1.0 if side == "left" else -1.0
        assert inward_sign * (index_tip[0] - ring_tip[0]) > 0.08
        assert inward_sign * (thumb_tip[0] - palm[0, 3]) > 0.07

    belt_surface = next(
        node for node in scene["__NODES__"] if node["name"] == "belt_surface"
    )
    np.testing.assert_allclose(
        _matrix_vector(
            belt_surface["properties"], "position", (0.0, 0.0, 0.0)
        ),
        (builder.BELT_CENTER_X, builder.BELT_CENTER_Y, builder.BELT_CENTER_Z),
    )
    assert (
        builder.WORKTABLE_CENTER_Y + 0.5 * builder.WORKTABLE_DEPTH
        < builder.BELT_CENTER_Y - 0.5 * builder.BELT_WIDTH
    )
    assert len({name for name in names if name.startswith("belt_marker_")}) >= 12

    soft_specs = {
        str(spec["name"]): spec for spec in builder.SOFT_PACKAGE_SPECS
    }
    blue_spec = soft_specs["soft_mailer_blue"]
    fill_spec = soft_specs["soft_mailer_blue_fill"]
    yellow_spec = soft_specs["soft_pouch_yellow"]
    yellow_fill_spec = soft_specs["soft_pouch_yellow_fill"]
    assert math.isclose(float(blue_spec["position"][1]), -0.375)
    assert blue_spec["model"] == "thin_shell"
    assert tuple(blue_spec["cells"]) == (34, 25)
    assert float(blue_spec["young_modulus"]) >= 1.0e5
    assert float(blue_spec["bending_stiffness"]) <= 5.0e-4
    assert float(blue_spec["thickness"]) <= 1.5e-3
    assert fill_spec["model"] == "volumetric"
    assert bool(fill_spec["internal_fill"])
    assert tuple(fill_spec["cells"]) == (12, 9, 5)
    assert math.isclose(float(fill_spec["size"][2]), 0.070)
    assert builder.WORKTABLE_TOP_Z < float(fill_spec["position"][2])
    assert float(fill_spec["position"][2]) < float(blue_spec["position"][2])
    assert not bool(fill_spec["visible"])
    assert yellow_spec["model"] == "thin_shell"
    assert tuple(yellow_spec["cells"]) == (36, 27)
    assert float(yellow_spec["size"][0]) < float(blue_spec["size"][0])
    assert float(yellow_spec["size"][1]) < float(blue_spec["size"][1])
    assert float(yellow_spec["size"][2]) < float(blue_spec["size"][2])
    assert yellow_fill_spec["model"] == "volumetric"
    assert bool(yellow_fill_spec["internal_fill"])
    assert tuple(yellow_fill_spec["cells"]) == (14, 11, 8)
    assert math.isclose(float(yellow_fill_spec["size"][2]), 0.065)
    assert math.isclose(float(yellow_fill_spec["side_rounding"]), 0.08)
    assert builder.WORKTABLE_TOP_Z < float(yellow_fill_spec["position"][2])
    assert (
        float(yellow_fill_spec["position"][2])
        < float(yellow_spec["position"][2])
    )
    assert (
        float(yellow_fill_spec["size"][2])
        < float(fill_spec["size"][2])
    )
    assert (
        float(yellow_spec["size"][0])
        - float(yellow_fill_spec["size"][0])
        >= 0.0349
    )
    assert not bool(yellow_fill_spec["visible"])
    assert not (
        builder.HAND_COLLISION_MASK
        & builder.DEFORMABLE_FILL_COLLISION_LAYER
    )
    assert (
        builder.DEFORMABLE_COLLISION_MASK
        & builder.DEFORMABLE_FILL_COLLISION_LAYER
    )
    assert (
        builder.DEFORMABLE_FILL_COLLISION_MASK
        & builder.DEFORMABLE_COLLISION_LAYER
    )

    blue_mesh = builder._soft_mailer_shell_mesh(
        blue_spec["size"], blue_spec["cells"]
    )
    blue_vertices = np.asarray(blue_mesh.vertices, dtype=np.float64)
    blue_triangles = np.asarray(blue_mesh.triangles, dtype=np.int64)
    layer_size = (blue_spec["cells"][0] + 1) * (
        blue_spec["cells"][1] + 1
    )
    bottom_z = blue_vertices[:layer_size, 2]
    top_z = blue_vertices[layer_size:, 2]
    blue_perimeter = np.asarray(
        [
            iy * (blue_spec["cells"][0] + 1) + ix
            for iy in range(blue_spec["cells"][1] + 1)
            for ix in range(blue_spec["cells"][0] + 1)
            if ix in (0, blue_spec["cells"][0])
            or iy in (0, blue_spec["cells"][1])
        ],
        dtype=np.int64,
    )
    assert float(top_z.max() - bottom_z.min()) > 0.10
    assert float(np.ptp(top_z)) > 0.06
    assert float(np.ptp(bottom_z[blue_perimeter])) <= 1.0e-12
    assert float(np.ptp(top_z[blue_perimeter])) <= 1.0e-12
    edge_counts: dict[tuple[int, int], int] = {}
    for triangle in blue_triangles:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    assert edge_counts and set(edge_counts.values()) == {2}
    blue_bottom_gap = (
        float(blue_spec["position"][2])
        + float(bottom_z.min())
        - builder.WORKTABLE_TOP_Z
    )
    assert math.isclose(blue_bottom_gap, 0.011, abs_tol=1.0e-6)

    yellow_mesh = builder._soft_mailer_shell_mesh(
        yellow_spec["size"], yellow_spec["cells"]
    )
    yellow_vertices = np.asarray(yellow_mesh.vertices, dtype=np.float64)
    yellow_triangles = np.asarray(yellow_mesh.triangles, dtype=np.int64)
    yellow_layer_size = (yellow_spec["cells"][0] + 1) * (
        yellow_spec["cells"][1] + 1
    )
    yellow_bottom_z = yellow_vertices[:yellow_layer_size, 2]
    yellow_top_z = yellow_vertices[yellow_layer_size:, 2]
    yellow_perimeter = np.asarray(
        [
            iy * (yellow_spec["cells"][0] + 1) + ix
            for iy in range(yellow_spec["cells"][1] + 1)
            for ix in range(yellow_spec["cells"][0] + 1)
            if ix in (0, yellow_spec["cells"][0])
            or iy in (0, yellow_spec["cells"][1])
        ],
        dtype=np.int64,
    )
    assert len(blue_triangles) == 3636
    assert len(yellow_triangles) == 4140
    assert len(yellow_triangles) > len(blue_triangles)
    assert 0.10 < float(
        yellow_top_z.max() - yellow_bottom_z.min()
    ) < 0.14
    assert float(np.ptp(yellow_bottom_z[yellow_perimeter])) <= 1.0e-12
    assert float(np.ptp(yellow_top_z[yellow_perimeter])) <= 1.0e-12
    yellow_bottom_gap = (
        float(yellow_spec["position"][2])
        + float(yellow_bottom_z.min())
        - builder.WORKTABLE_TOP_Z
    )
    assert math.isclose(
        yellow_bottom_gap, 0.0091059, abs_tol=1.0e-6
    )

    properties_by_unique_name = {
        str(node["name"]): node["properties"]
        for node in scene["__NODES__"]
        if node["name"]
        in {
            "worktable_surface_collision",
            "leap_left_if_tip_collision",
            "soft_mailer_blue",
            "carton_small_collision",
        }
    }
    assert (
        properties_by_unique_name["worktable_surface_collision"]["collision_layer"],
        properties_by_unique_name["worktable_surface_collision"]["collision_mask"],
    ) == (builder.RIGID_COLLISION_LAYER, builder.RIGID_COLLISION_MASK)
    assert (
        properties_by_unique_name["leap_left_if_tip_collision"]["collision_layer"],
        properties_by_unique_name["leap_left_if_tip_collision"]["collision_mask"],
    ) == (builder.HAND_COLLISION_LAYER, builder.HAND_COLLISION_MASK)
    assert (
        properties_by_unique_name["soft_mailer_blue"]["collision_layer"],
        properties_by_unique_name["soft_mailer_blue"]["collision_mask"],
    ) == (
        builder.DEFORMABLE_COLLISION_LAYER,
        builder.DEFORMABLE_COLLISION_MASK,
    )
    assert (
        properties_by_unique_name["carton_small_collision"]["collision_layer"],
        properties_by_unique_name["carton_small_collision"]["collision_mask"],
    ) == (
        builder.RIGID_PACKAGE_COLLISION_LAYER,
        builder.RIGID_PACKAGE_COLLISION_MASK,
    )
    assert builder.HAND_COLLISION_MASK & builder.RIGID_PACKAGE_COLLISION_LAYER
    assert builder.HAND_COLLISION_MASK & builder.RIGID_COLLISION_LAYER

    stage_joint_names = {
        name
        for names_by_side in builder.HAND_STAGE_JOINT_NAMES_BY_SIDE
        for name in names_by_side
    }
    stage_joints = [
        node for node in scene["__NODES__"] if node["name"] in stage_joint_names
    ]
    assert len(stage_joints) == 12
    for joint in stage_joints:
        properties = joint["properties"]
        assert properties["drive_mode"] == "Position"
        dof_name = joint["name"].rsplit("_", 1)[-1]
        linear = dof_name in {"x", "y", "z"}
        expected_stiffness = (
            builder.HAND_STAGE_LINEAR_STIFFNESS
            if linear
            else builder.HAND_STAGE_ANGULAR_STIFFNESS
        )
        assert math.isclose(properties["drive_stiffness"], expected_stiffness)
        assert math.isclose(
            properties["armature"],
            builder.HAND_STAGE_ARMATURE,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
        expected_limit = (
            builder.HAND_STAGE_TRANSLATION_RANGE
            if linear
            else (
                builder.HAND_STAGE_ROLL_RANGE
                if dof_name == "roll"
                else builder.HAND_STAGE_ROTATION_RANGE
            )
        )
        assert math.isclose(
            properties["lower_limit"],
            -expected_limit,
            rel_tol=0.0,
            abs_tol=1.0e-5,
        )
        assert math.isclose(
            properties["upper_limit"],
            expected_limit,
            rel_tol=0.0,
            abs_tol=1.0e-5,
        )


def _legacy_quality_profiles_and_belt_schedule() -> None:
    profile = _profile()
    assert math.isclose(profile.IPC_CONTACT_ACTIVATION_DISTANCE, 2.0e-3)
    assert math.isclose(profile.IPC_CONTACT_RESISTANCE, 1.0e8)
    interactive = profile.quality_profile("interactive")
    accurate = profile.quality_profile("accurate")
    assert (
        interactive.coupling_iterations,
        interactive.relaxation_mode,
        interactive.scene_sync_interval,
        interactive.contact_refresh_interval,
    ) == (1, "fixed", 2, 4)
    assert (
        accurate.coupling_iterations,
        accurate.relaxation_mode,
        accurate.scene_sync_interval,
        accurate.contact_refresh_interval,
    ) == (2, "aitken", 1, 1)
    assert (
        interactive.newton_max_iterations,
        interactive.line_search_max_iterations,
        interactive.linear_system_tolerance_rate,
    ) == (48, 8, 2.0e-3)
    assert (
        accurate.newton_max_iterations,
        accurate.line_search_max_iterations,
        accurate.linear_system_tolerance_rate,
    ) == (16, 8, 1.0e-3)
    speeds = [
        profile.belt_speed_at_tick(tick)
        for tick in range(profile.CYCLE_TICKS)
    ]
    assert all(value == 0.0 for value in speeds[: profile.BELT_START_TICKS])
    ramp_start = profile.BELT_START_TICKS
    assert 0.0 < speeds[ramp_start] < speeds[
        ramp_start + profile.RAMP_TICKS - 1
    ]
    assert math.isclose(
        speeds[ramp_start + profile.RAMP_TICKS - 1], profile.BELT_SPEED
    )
    assert math.isclose(
        speeds[ramp_start + profile.RAMP_TICKS], profile.BELT_SPEED
    )
    assert speeds[-1] == 0.0
    assert math.isclose(
        sum(speeds) * profile.FIXED_DT,
        0.48,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    drop_end = profile.DROP_SETTLE_TICKS
    assert profile.cycle_phase(0) == "drop_settle"
    assert profile.cycle_phase(drop_end - 1) == "drop_settle"
    assert profile.cycle_phase(drop_end) == "approach"
    assert profile.cycle_phase(drop_end + profile.APPROACH_TICKS) == "grip"
    assert profile.cycle_phase(
        drop_end + profile.APPROACH_TICKS + profile.GRIP_SETTLE_TICKS
    ) == "push"
    assert profile.cycle_phase(
        drop_end
        + profile.APPROACH_TICKS
        + profile.GRIP_SETTLE_TICKS
        + profile.PUSH_TICKS
    ) == "release"
    open_settle_start = (
        drop_end
        + profile.APPROACH_TICKS
        + profile.GRIP_SETTLE_TICKS
        + profile.PUSH_TICKS
        + profile.RELEASE_TICKS
    )
    assert profile.cycle_phase(open_settle_start) == "open_settle"
    clear_start = open_settle_start + profile.OPEN_SETTLE_TICKS
    assert profile.cycle_phase(clear_start) == "clear"
    retract_start = clear_start + profile.ARM_CLEAR_TICKS
    assert profile.cycle_phase(retract_start) == "retract"
    assert profile.cycle_phase(profile.MANIPULATION_TICKS) == "settle"
    assert profile.cycle_phase(profile.CYCLE_TICKS - 1) == "settle"
    grip_fractions = [
        profile.gripper_close_fraction_at_tick(tick)
        for tick in range(profile.MANIPULATION_TICKS + 1)
    ]
    assert all(value == 0.0 for value in grip_fractions[:drop_end])
    assert 0.0 < grip_fractions[drop_end] < 1.0
    assert math.isclose(
        grip_fractions[drop_end + profile.APPROACH_TICKS - 1],
        1.0,
    )
    assert grip_fractions[-1] == 0.0
    assert profile.arm_grip_fraction_at_tick(drop_end - 1) == 0.0
    assert 0.0 < profile.arm_grip_fraction_at_tick(drop_end) < 1.0
    assert profile.arm_grip_fraction_at_tick(
        drop_end + profile.APPROACH_TICKS - 1
    ) == 1.0
    assert profile.arm_grip_fraction_at_tick(profile.MANIPULATION_TICKS) == 1.0
    push_start = (
        drop_end + profile.APPROACH_TICKS + profile.GRIP_SETTLE_TICKS
    )
    assert profile.arm_push_fraction_at_tick(push_start - 1) == 0.0
    assert 0.0 < profile.arm_push_fraction_at_tick(push_start) < 1.0
    assert profile.arm_push_fraction_at_tick(
        push_start + profile.PUSH_TICKS - 1
    ) == 1.0
    assert profile.arm_clear_fraction_at_tick(
        clear_start - 1
    ) == 0.0
    assert 0.0 < profile.arm_clear_fraction_at_tick(
        clear_start
    ) < 1.0
    assert profile.arm_clear_fraction_at_tick(retract_start - 1) == 1.0
    assert profile.arm_retract_fraction_at_tick(retract_start - 1) == 0.0
    assert 0.0 < profile.arm_retract_fraction_at_tick(retract_start) < 1.0
    assert profile.arm_retract_fraction_at_tick(
        profile.MANIPULATION_TICKS - 1
    ) == 1.0
    assert tuple(len(target) for target in profile.ARM_GRIP_TARGETS) == (7, 7)
    assert tuple(len(target) for target in profile.ARM_PUSH_TARGETS) == (7, 7)
    assert tuple(len(target) for target in profile.ARM_CLEAR_TARGETS) == (7, 7)
    assert tuple(len(target) for target in profile.ARM_RETRACT_TARGETS) == (7, 7)
    assert profile.FINGER_GRIP_OFFSETS[0] < 0.0
    assert profile.FINGER_GRIP_OFFSETS[1] > 0.0

    for spec, expected_mass in zip(
        _builder().SOFT_PACKAGE_SPECS,
        profile.SOFT_PACKAGE_MASSES,
        strict=True,
    ):
        if spec.get("model", "volumetric") == "thin_shell":
            mesh = _builder()._soft_mailer_shell_mesh(
                spec["size"], spec["cells"]
            )
            mass = _surface_mass(mesh, spec["thickness"], spec["density"])
        else:
            mesh = _builder()._soft_package_mesh(
                spec["size"],
                spec["cells"],
                side_rounding=float(spec.get("side_rounding", 0.0)),
            )
            mass = _tetrahedral_mass(mesh, spec["density"])
        assert math.isclose(mass, expected_mass, rel_tol=1.0e-12)
    assert math.isclose(
        sum(profile.SOFT_PACKAGE_MASSES[:2]),
        0.35,
        rel_tol=1.0e-12,
    )


def test_quality_profiles_and_flip_push_schedule() -> None:
    profile = _profile()
    interactive = profile.quality_profile("interactive")
    accurate = profile.quality_profile("accurate")
    assert (
        interactive.name,
        interactive.contact_refresh_interval,
        interactive.newton_max_iterations,
        interactive.line_search_max_iterations,
    ) == ("interactive", 4, 16, 8)
    assert (
        accurate.name,
        accurate.contact_refresh_interval,
        accurate.newton_max_iterations,
        accurate.line_search_max_iterations,
    ) == ("accurate", 1, 48, 8)

    speeds = [
        profile.belt_speed_at_tick(tick)
        for tick in range(profile.CYCLE_TICKS)
    ]
    assert all(value == 0.0 for value in speeds[: profile.BELT_START_TICKS])
    ramp_start = profile.BELT_START_TICKS
    assert 0.0 < speeds[ramp_start] < speeds[
        ramp_start + profile.RAMP_TICKS - 1
    ]
    assert math.isclose(
        speeds[ramp_start + profile.RAMP_TICKS - 1], profile.BELT_SPEED
    )
    assert math.isclose(
        speeds[ramp_start + profile.RAMP_TICKS], profile.BELT_SPEED
    )
    assert speeds[-1] == 0.0
    assert math.isclose(
        sum(speeds) * profile.FIXED_DT,
        0.60,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )

    phases_and_durations = tuple(
        (segment.phase, segment.duration)
        for segment in profile.HAND_MOTION_SEGMENTS
    )
    assert tuple(phase for phase, _ in phases_and_durations) == (
        "drop_settle",
        "blue_flip_approach",
        "blue_flip_contact",
        "blue_flip_grip",
        "blue_flip_stabilize",
        "blue_flip_rotate",
        "blue_flip_place",
        "blue_flip_turnover",
        "blue_flip_release",
        "blue_flip_clear",
        "blue_flip_settle",
        "blue_reorient",
        "blue_push_approach",
        "blue_push",
        "blue_push_release",
        "blue_push_clear",
        "blue_push_depart",
        "yellow_flip_transfer",
        "yellow_flip_approach",
        "yellow_flip_contact",
        "yellow_flip_grip",
        "yellow_flip_stabilize",
        "yellow_flip_rotate",
        "yellow_flip_place",
        "yellow_flip_release",
        "yellow_flip_clear",
        "yellow_flip_settle",
        "yellow_flip_depart",
        "carton_flip_transfer",
        "carton_flip_approach",
        "carton_flip_grip",
        "carton_flip_stabilize",
        "carton_flip_lift",
        "carton_flip_rotate",
        "carton_flip_place",
        "carton_flip_release",
        "carton_flip_clear",
        "carton_flip_depart",
        "return_home",
    )
    phase_starts = {}
    tick = 0
    for phase_name, duration in phases_and_durations:
        phase_starts[phase_name] = tick
        assert profile.cycle_phase(tick) == phase_name
        assert profile.cycle_phase(tick + duration - 1) == phase_name
        tick += duration
    assert tick == profile.MANIPULATION_TICKS
    assert profile.cycle_phase(tick) == "settle"
    assert profile.cycle_phase(profile.CYCLE_TICKS - 1) == "settle"
    assert profile.BELT_START_TICKS == (
        phase_starts["blue_push"] + profile.PUSH_TICKS
    )
    for previous, following in zip(
        profile.HAND_MOTION_SEGMENTS[:-1],
        profile.HAND_MOTION_SEGMENTS[1:],
        strict=True,
    ):
        np.testing.assert_allclose(previous.end, following.start, atol=1.0e-12)

    flip_grip_start = phase_starts["blue_flip_grip"]
    assert profile.finger_close_fractions_at_tick(
        flip_grip_start - 1
    ) == (0.0, 0.0)
    blue_close = profile.finger_close_fractions_at_tick(
        flip_grip_start + profile.FLIP_GRIP_TICKS - 1
    )
    np.testing.assert_allclose(
        blue_close,
        profile.BLUE_FLIP_FINGER_CLOSE_FRACTIONS,
        atol=1.0e-12,
    )
    assert math.isclose(
        profile.finger_close_fraction_at_tick(
            flip_grip_start + profile.FLIP_GRIP_TICKS - 1
        ),
        max(profile.BLUE_FLIP_FINGER_CLOSE_FRACTIONS),
    )
    assert all(
        len(targets) == 6
        for pair in (
            profile.HAND_STAGE_HOME_TARGETS,
            profile.HAND_STAGE_BLUE_FLIP_GRIP_TARGETS,
            profile.HAND_STAGE_BLUE_FLIPPED_TARGETS,
            profile.HAND_STAGE_YELLOW_FLIPPED_TARGETS,
            profile.HAND_STAGE_PUSH_TARGETS,
        )
        for targets in pair
    )
    assert all(
        len(controls) == 22
        for controls in profile.hand_controls_at_tick(flip_grip_start)
    )
    blue_contact_shift = (
        np.asarray(profile.HAND_STAGE_BLUE_FLIP_CONTACT_TARGETS)
        - np.asarray(profile.HAND_STAGE_BLUE_FLIP_APPROACH_TARGETS)
    )
    np.testing.assert_allclose(
        blue_contact_shift,
        (
            (0.0, 0.0, -profile.SOFT_FLIP_APPROACH_CLEARANCE, 0.0, 0.0, 0.0),
            (0.0, 0.0, -profile.SOFT_FLIP_APPROACH_CLEARANCE, 0.0, 0.0, 0.0),
        ),
        atol=1.0e-12,
    )
    grip_preload = tuple(
        grip[0] - contact[0]
        for contact, grip in zip(
            profile.HAND_STAGE_BLUE_FLIP_CONTACT_TARGETS,
            profile.HAND_STAGE_BLUE_FLIP_GRIP_TARGETS,
            strict=True,
        )
    )
    np.testing.assert_allclose(
        grip_preload,
        (profile.BLUE_FLIP_PALM_PRELOAD, -profile.BLUE_FLIP_PALM_PRELOAD),
        atol=1.0e-12,
    )
    assert profile.BLUE_FLIP_PALM_PRELOAD == 0.0
    np.testing.assert_allclose(
        np.asarray(profile.HAND_STAGE_BLUE_FLIP_STABILIZE_TARGETS)
        - np.asarray(profile.HAND_STAGE_BLUE_FLIP_GRIP_TARGETS),
        (
            (0.0, 0.0, profile.BLUE_FLIP_STABILIZE_LIFT, 0.0, 0.0, 0.0),
            (0.0, 0.0, profile.BLUE_FLIP_STABILIZE_LIFT, 0.0, 0.0, 0.0),
        ),
        atol=1.0e-12,
    )
    blue_turnover_delta = (
        np.asarray(profile.HAND_STAGE_BLUE_FLIPPED_TARGETS)
        - np.asarray(profile.HAND_STAGE_BLUE_FLIP_STABILIZE_TARGETS)
    )
    expected_blue_flipped = profile._pinched_edge_turnover_targets(
        profile.HAND_STAGE_BLUE_FLIP_STABILIZE_TARGETS,
        fraction=1.0,
        forward_travel=profile.BLUE_FLIP_EDGE_FORWARD_TRAVEL,
        end_lift=profile.BLUE_FLIP_EDGE_END_LIFT,
        arc_height=profile.BLUE_FLIP_EDGE_ARC_HEIGHT,
    )
    np.testing.assert_allclose(
        profile.HAND_STAGE_BLUE_FLIPPED_TARGETS,
        expected_blue_flipped,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        blue_turnover_delta,
        (
            (
                0.0,
                profile.BLUE_FLIP_EDGE_FORWARD_TRAVEL,
                profile.BLUE_FLIP_EDGE_END_LIFT,
                0.0,
                0.0,
                0.0,
            ),
            (
                0.0,
                profile.BLUE_FLIP_EDGE_FORWARD_TRAVEL,
                profile.BLUE_FLIP_EDGE_END_LIFT,
                0.0,
                0.0,
                0.0,
            ),
        ),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(profile.HAND_STAGE_BLUE_FLIP_CLEAR_TARGETS)
        - np.asarray(profile.HAND_STAGE_BLUE_FLIP_RELEASE_TARGETS),
        (
            (
                -profile.BLUE_FLIP_CLEAR_SIDE_RETREAT,
                0.0,
                profile.BLUE_FLIP_CLEAR_LIFT,
                0.0,
                0.0,
                0.0,
            ),
            (
                profile.BLUE_FLIP_CLEAR_SIDE_RETREAT,
                0.0,
                profile.BLUE_FLIP_CLEAR_LIFT,
                0.0,
                0.0,
                0.0,
            ),
        ),
        atol=1.0e-12,
    )
    assert profile.BLUE_FLIP_FINGER_CLOSE_FRACTIONS == (1.0, 1.0)
    assert math.isclose(profile.BLUE_POST_FLIP_ROLL, 0.0)
    assert 0.40 < profile.BLUE_FLIP_EDGE_FORWARD_TRAVEL < 0.50
    assert profile.BLUE_FLIP_STABILIZE_LIFT > 0.32
    assert profile.BLUE_FLIP_EDGE_END_LIFT == 0.0
    assert 0.0 < profile.BLUE_FLIP_EDGE_ARC_HEIGHT <= 0.05
    assert 0.08 < profile.BLUE_FLIP_PLACE_DROP < 0.10
    assert 0.40 < profile.BLUE_FLIP_TURNOVER_FORWARD_TRAVEL < 0.50
    assert 0.20 < profile.BLUE_FLIP_TURNOVER_DROP < 0.30
    expected_blue_turnover = profile._table_pivot_turnover_targets(
        profile.HAND_STAGE_BLUE_FLIP_PLACE_TARGETS,
        fraction=1.0,
        forward_travel=profile.BLUE_FLIP_TURNOVER_FORWARD_TRAVEL,
        drop=profile.BLUE_FLIP_TURNOVER_DROP,
    )
    np.testing.assert_allclose(
        profile.HAND_STAGE_BLUE_FLIP_TURNOVER_TARGETS,
        expected_blue_turnover,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(profile.HAND_STAGE_PUSH_APPROACH_TARGETS)[:, 1],
        (-0.8000, -0.8705),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(profile.HAND_STAGE_PUSH_RELEASE_TARGETS)
        - np.asarray(profile.HAND_STAGE_PUSH_TARGETS),
        (
            (0.0, -profile.PUSH_RELEASE_RETREAT, 0.0, 0.0, 0.0, 0.0),
            (0.0, -profile.PUSH_RELEASE_RETREAT, 0.0, 0.0, 0.0, 0.0),
        ),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(profile.HAND_STAGE_PUSH_CLEAR_TARGETS)
        - np.asarray(profile.HAND_STAGE_PUSH_RELEASE_TARGETS),
        (
            (0.0, 0.0, profile.PUSH_CLEAR_LIFT, 0.0, 0.0, 0.0),
            (0.0, 0.0, profile.PUSH_CLEAR_LIFT, 0.0, 0.0, 0.0),
        ),
        atol=1.0e-12,
    )
    assert profile.YELLOW_FLIP_FINGER_CLOSE_FRACTIONS == (1.0, 1.0)
    assert profile.CARTON_FLIP_FINGER_CLOSE_FRACTION <= 0.25
    yellow_contact_shift = (
        np.asarray(profile.HAND_STAGE_YELLOW_FLIP_CONTACT_TARGETS)
        - np.asarray(profile.HAND_STAGE_YELLOW_FLIP_APPROACH_TARGETS)
    )
    np.testing.assert_allclose(
        yellow_contact_shift,
        (
            (0.0, 0.0, -profile.SOFT_FLIP_APPROACH_CLEARANCE, 0.0, 0.0, 0.0),
            (0.0, 0.0, -profile.SOFT_FLIP_APPROACH_CLEARANCE, 0.0, 0.0, 0.0),
        ),
        atol=1.0e-12,
    )
    yellow_grip_shift = (
        np.asarray(profile.HAND_STAGE_YELLOW_FLIP_GRIP_TARGETS)
        - np.asarray(profile.HAND_STAGE_YELLOW_FLIP_CONTACT_TARGETS)
    )
    np.testing.assert_allclose(
        yellow_grip_shift,
        (
            (profile.YELLOW_FLIP_PALM_PRELOAD, 0.0, 0.0, 0.0, 0.0, 0.0),
            (-profile.YELLOW_FLIP_PALM_PRELOAD, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        atol=1.0e-12,
    )
    assert profile.YELLOW_FLIP_PALM_PRELOAD == 0.0
    assert all(
        target[3:] == (0.0, 0.0, 0.0)
        for targets in (
            profile.HAND_STAGE_BLUE_FLIP_APPROACH_TARGETS,
            profile.HAND_STAGE_BLUE_FLIP_CONTACT_TARGETS,
            profile.HAND_STAGE_BLUE_FLIP_GRIP_TARGETS,
            profile.HAND_STAGE_BLUE_FLIP_STABILIZE_TARGETS,
                profile.HAND_STAGE_BLUE_FLIPPED_TARGETS,
                profile.HAND_STAGE_BLUE_FLIP_PLACE_TARGETS,
                profile.HAND_STAGE_BLUE_FLIP_TURNOVER_TARGETS,
            profile.HAND_STAGE_YELLOW_FLIP_APPROACH_TARGETS,
            profile.HAND_STAGE_YELLOW_FLIP_CONTACT_TARGETS,
            profile.HAND_STAGE_YELLOW_FLIP_GRIP_TARGETS,
            profile.HAND_STAGE_YELLOW_FLIP_STABILIZE_TARGETS,
            profile.HAND_STAGE_YELLOW_FLIPPED_TARGETS,
            profile.HAND_STAGE_YELLOW_FLIP_PLACE_TARGETS,
        )
        for target in targets
    )
    blue_controls = profile.hand_controls_at_tick(
        flip_grip_start + profile.FLIP_GRIP_TICKS - 1
    )
    for controls, expected_targets, close_fraction in zip(
        blue_controls,
        profile.LEAP_BLUE_AIR_PINCH_TARGETS_BY_SIDE,
        profile.BLUE_FLIP_FINGER_CLOSE_FRACTIONS,
        strict=True,
    ):
        np.testing.assert_allclose(
            controls[6:],
            np.asarray(expected_targets) * close_fraction,
            atol=1.0e-12,
        )
    blue_push_start = phase_starts["blue_push"]
    for controls in profile.hand_controls_at_tick(blue_push_start):
        np.testing.assert_allclose(
            controls[6:],
            np.asarray(profile.LEAP_FINGER_CLOSE_TARGETS)
            * profile.PUSH_FINGER_CLOSE_FRACTION,
            atol=1.0e-12,
        )
    assert all(
        math.isclose(target[3], profile.BLUE_POST_FLIP_ROLL)
        and target[4:] == (0.0, 0.0)
        for target in profile.HAND_STAGE_PUSH_TARGETS
    )
    for phase_name, duration, expected_fractions in (
        (
            "yellow_flip_grip",
            profile.YELLOW_FLIP_GRIP_TICKS,
            profile.YELLOW_FLIP_FINGER_CLOSE_FRACTIONS,
        ),
        (
            "carton_flip_grip",
            profile.CARTON_FLIP_GRIP_TICKS,
            (profile.CARTON_FLIP_FINGER_CLOSE_FRACTION,) * 2,
        ),
    ):
        start = phase_starts[phase_name]
        assert profile.finger_close_fractions_at_tick(start - 1) == (0.0, 0.0)
        assert all(
            fraction > 0.0
            for fraction in profile.finger_close_fractions_at_tick(start)
        )
        np.testing.assert_allclose(
            profile.finger_close_fractions_at_tick(start + duration - 1),
            expected_fractions,
            atol=1.0e-12,
        )
    yellow_release_end = (
        phase_starts["yellow_flip_release"]
        + profile.YELLOW_FLIP_RELEASE_TICKS
        - 1
    )
    blue_release_end = (
        phase_starts["blue_flip_release"] + profile.FLIP_RELEASE_TICKS - 1
    )
    np.testing.assert_allclose(
        profile.finger_close_fractions_at_tick(blue_release_end),
        (profile.BLUE_FLIP_RELEASE_CLOSE_FRACTION,) * 2,
        atol=1.0e-12,
    )
    release_fraction = profile.SOFT_FLIP_RELEASE_CLOSE_FRACTION
    np.testing.assert_allclose(
        profile.finger_close_fractions_at_tick(yellow_release_end),
        (release_fraction, release_fraction),
        atol=1.0e-12,
    )
    for controls, targets in zip(
        profile.hand_controls_at_tick(yellow_release_end),
        profile.LEAP_SOFT_PINCH_TARGETS_BY_SIDE,
        strict=True,
    ):
        np.testing.assert_allclose(
            controls[6:], np.asarray(targets) * release_fraction,
            atol=1.0e-12,
        )
    for prefix, clear_ticks, settle_ticks, clear_fraction in (
        (
            "blue",
            profile.FLIP_CLEAR_TICKS,
            profile.FLIP_SETTLE_TICKS,
            profile.BLUE_FLIP_RELEASE_CLOSE_FRACTION,
        ),
        (
            "yellow",
            profile.YELLOW_FLIP_CLEAR_TICKS,
            profile.YELLOW_FLIP_SETTLE_TICKS,
            release_fraction,
        ),
    ):
        clear_end = (
            phase_starts[f"{prefix}_flip_clear"]
            + clear_ticks - 1
        )
        settle_end = (
            phase_starts[f"{prefix}_flip_settle"]
            + settle_ticks - 1
        )
        np.testing.assert_allclose(
            profile.finger_close_fractions_at_tick(clear_end),
            (clear_fraction, clear_fraction),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            profile.finger_close_fractions_at_tick(settle_end),
            (0.0, 0.0),
            atol=1.0e-12,
        )
    np.testing.assert_allclose(
        np.asarray(profile.HAND_STAGE_YELLOW_FLIP_RELEASE_TARGETS)
        - np.asarray(profile.HAND_STAGE_YELLOW_FLIP_PLACE_TARGETS),
        0.0,
        atol=1.0e-12,
    )
    expected_yellow_flipped = profile._pinched_edge_turnover_targets(
        profile.HAND_STAGE_YELLOW_FLIP_STABILIZE_TARGETS,
        fraction=1.0,
        forward_travel=profile.YELLOW_FLIP_EDGE_FORWARD_TRAVEL,
        end_lift=profile.YELLOW_FLIP_EDGE_END_LIFT,
        arc_height=profile.YELLOW_FLIP_EDGE_ARC_HEIGHT,
    )
    np.testing.assert_allclose(
        profile.HAND_STAGE_YELLOW_FLIPPED_TARGETS,
        expected_yellow_flipped,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(profile.HAND_STAGE_YELLOW_FLIP_CLEAR_TARGETS)
        - np.asarray(profile.HAND_STAGE_YELLOW_FLIP_RELEASE_TARGETS),
        (
            (0.0, 0.0, profile.YELLOW_FLIP_CLEAR_LIFT, 0.0, 0.0, 0.0),
            (0.0, 0.0, profile.YELLOW_FLIP_CLEAR_LIFT, 0.0, 0.0, 0.0),
        ),
        atol=1.0e-12,
    )
    for blue, yellow, carton in zip(
        profile.HAND_STAGE_BLUE_FLIP_GRIP_TARGETS,
        profile.HAND_STAGE_YELLOW_FLIP_GRIP_TARGETS,
        profile.HAND_STAGE_CARTON_FLIP_GRIP_TARGETS,
        strict=True,
    ):
        assert carton[0] < yellow[0] < blue[0]
    assert all(
        math.isclose(target[2], 0.15, abs_tol=1.0e-12)
        for target in (
            *profile.HAND_STAGE_BLUE_PUSH_DEPART_TARGETS,
            *profile.HAND_STAGE_YELLOW_FLIP_TRANSFER_TARGETS,
            *profile.HAND_STAGE_YELLOW_FLIP_DEPART_TARGETS,
            *profile.HAND_STAGE_CARTON_FLIP_TRANSFER_TARGETS,
            *profile.HAND_STAGE_CARTON_FLIP_DEPART_TARGETS,
        )
    )
    carton_preload = tuple(
        grip[0] - approach[0]
        for approach, grip in zip(
            profile.HAND_STAGE_CARTON_FLIP_APPROACH_TARGETS,
            profile.HAND_STAGE_CARTON_FLIP_GRIP_TARGETS,
            strict=True,
        )
    )
    np.testing.assert_allclose(
        carton_preload,
        (profile.CARTON_FLIP_PALM_PRELOAD, -profile.CARTON_FLIP_PALM_PRELOAD),
        atol=1.0e-12,
    )
    assert 0.0 < profile.CARTON_FLIP_PALM_PRELOAD <= 0.002
    assert math.isclose(
        profile.CARTON_FLIP_FINGER_CLOSE_FRACTION,
        0.15,
        abs_tol=1.0e-12,
    )
    for flipped, placed, released in zip(
        profile.HAND_STAGE_CARTON_FLIPPED_LIFT_TARGETS,
        profile.HAND_STAGE_CARTON_FLIP_PLACE_TARGETS,
        profile.HAND_STAGE_CARTON_FLIP_RELEASE_TARGETS,
        strict=True,
    ):
        assert math.isclose(
            flipped[2] - placed[2],
            profile.CARTON_FLIP_PLACE_DROP,
            abs_tol=1.0e-12,
        )
        np.testing.assert_allclose(placed, released, atol=1.0e-12)
    for flipped, placed, released in zip(
        profile.HAND_STAGE_YELLOW_FLIPPED_TARGETS,
        profile.HAND_STAGE_YELLOW_FLIP_PLACE_TARGETS,
        profile.HAND_STAGE_YELLOW_FLIP_RELEASE_TARGETS,
        strict=True,
    ):
        assert math.isclose(
            flipped[2] - placed[2],
            profile.YELLOW_FLIP_PLACE_DROP,
            abs_tol=1.0e-12,
        )
        np.testing.assert_allclose(placed, released, atol=1.0e-12)
    for flipped, placed, turned, released in zip(
        profile.HAND_STAGE_BLUE_FLIPPED_TARGETS,
        profile.HAND_STAGE_BLUE_FLIP_PLACE_TARGETS,
        profile.HAND_STAGE_BLUE_FLIP_TURNOVER_TARGETS,
        profile.HAND_STAGE_BLUE_FLIP_RELEASE_TARGETS,
        strict=True,
    ):
        assert math.isclose(
            turned[1] - placed[1],
            profile.BLUE_FLIP_TURNOVER_FORWARD_TRAVEL,
            abs_tol=1.0e-12,
        )
        assert math.isclose(
            placed[2] - turned[2],
            profile.BLUE_FLIP_TURNOVER_DROP,
            abs_tol=1.0e-12,
        )
        assert math.isclose(
            flipped[2] - placed[2],
            profile.BLUE_FLIP_PLACE_DROP,
            abs_tol=1.0e-12,
        )
        np.testing.assert_allclose(turned, released, atol=1.0e-12)
    assert (
        profile.HAND_STAGE_CARTON_FLIP_CLEAR_TARGETS[0][0]
        <= profile.HAND_STAGE_CARTON_FLIP_RELEASE_TARGETS[0][0]
    )
    assert (
        profile.HAND_STAGE_CARTON_FLIP_CLEAR_TARGETS[1][0]
        > profile.HAND_STAGE_CARTON_FLIP_RELEASE_TARGETS[1][0]
    )
    assert all(
        all(
            abs(value) <= _builder().HAND_STAGE_TRANSLATION_RANGE
            for value in target[:3]
        )
        for segment in profile.HAND_MOTION_SEGMENTS
        for pair in (segment.start, segment.end)
        for target in pair
    )
    assert all(
        math.isclose(end[3] - start[3], -0.5 * math.pi)
        for start, end in zip(
            profile.HAND_STAGE_CARTON_FLIP_LIFT_TARGETS,
            profile.HAND_STAGE_CARTON_FLIPPED_LIFT_TARGETS,
            strict=True,
        )
    )
    np.testing.assert_allclose(
        np.asarray(profile.HAND_STAGE_CARTON_FLIPPED_LIFT_TARGETS)[:, 1]
        - np.asarray(profile.HAND_STAGE_CARTON_FLIP_LIFT_TARGETS)[:, 1],
        profile.CARTON_FLIP_FOLLOW_THROUGH,
        atol=1.0e-12,
    )
    for phase_name, duration, before_targets, after_targets in (
        (
            "blue_flip_rotate",
            profile.FLIP_ROTATE_TICKS,
            profile.HAND_STAGE_BLUE_FLIP_STABILIZE_TARGETS,
            profile.HAND_STAGE_BLUE_FLIPPED_TARGETS,
        ),
        (
            "yellow_flip_rotate",
            profile.YELLOW_FLIP_ROTATE_TICKS,
            profile.HAND_STAGE_YELLOW_FLIP_STABILIZE_TARGETS,
            profile.HAND_STAGE_YELLOW_FLIPPED_TARGETS,
        ),
        (
            "carton_flip_rotate",
            profile.CARTON_FLIP_ROTATE_TICKS,
            profile.HAND_STAGE_CARTON_FLIP_LIFT_TARGETS,
            profile.HAND_STAGE_CARTON_FLIPPED_LIFT_TARGETS,
        ),
    ):
        for relative_tick in (0, duration // 4, duration // 2 - 1, duration - 1):
            stage_targets = profile.hand_stage_targets_at_tick(
                phase_starts[phase_name] + relative_tick
            )
            if phase_name in {"blue_flip_rotate", "yellow_flip_rotate"}:
                fraction = profile.smoothstep(
                    float(relative_tick + 1) / duration
                )
                if phase_name == "blue_flip_rotate":
                    forward_travel = profile.BLUE_FLIP_EDGE_FORWARD_TRAVEL
                    end_lift = profile.BLUE_FLIP_EDGE_END_LIFT
                    arc_height = profile.BLUE_FLIP_EDGE_ARC_HEIGHT
                else:
                    forward_travel = profile.YELLOW_FLIP_EDGE_FORWARD_TRAVEL
                    end_lift = profile.YELLOW_FLIP_EDGE_END_LIFT
                    arc_height = profile.YELLOW_FLIP_EDGE_ARC_HEIGHT
                expected_targets = profile._pinched_edge_turnover_targets(
                    before_targets,
                    fraction=fraction,
                    forward_travel=forward_travel,
                    end_lift=end_lift,
                    arc_height=arc_height,
                )
            else:
                expected_targets = profile._transition(
                    relative_tick, duration, before_targets, after_targets
                )
            np.testing.assert_allclose(
                stage_targets, expected_targets, atol=1.0e-12
            )
    assert all(
        abs(target[3]) <= _builder().HAND_STAGE_ROLL_RANGE
        for segment in profile.HAND_MOTION_SEGMENTS
        for pair in (segment.start, segment.end)
        for target in pair
    )
    large_roll_phases = tuple(
        segment.phase
        for segment in profile.HAND_MOTION_SEGMENTS
        if any(
            abs(end[3] - start[3]) > 0.5 * math.pi + 1.0e-12
            for start, end in zip(segment.start, segment.end, strict=True)
        )
    )
    assert large_roll_phases == ()
    assert all(
        abs(end[3] - start[3]) <= 1.5 * math.pi + 1.0e-12
        for segment in profile.HAND_MOTION_SEGMENTS
        for start, end in zip(segment.start, segment.end, strict=True)
    )
    for segment in profile.HAND_MOTION_SEGMENTS:
        expected_scale = (
            0.0
            if segment.phase in profile.SOFT_DAMPING_DISABLED_PHASES
            else 1.0
        )
        assert profile.soft_damping_scale_at_tick(
            phase_starts[segment.phase]
        ) == expected_scale

    for spec, expected_mass in zip(
        _builder().SOFT_PACKAGE_SPECS,
        profile.SOFT_PACKAGE_MASSES,
        strict=True,
    ):
        if spec.get("model", "volumetric") == "thin_shell":
            mesh = _builder()._soft_mailer_shell_mesh(
                spec["size"], spec["cells"]
            )
            mass = _surface_mass(mesh, spec["thickness"], spec["density"])
        else:
            mesh = _builder()._soft_package_mesh(
                spec["size"],
                spec["cells"],
                side_rounding=float(spec.get("side_rounding", 0.0)),
            )
            mass = _tetrahedral_mass(mesh, spec["density"])
        assert math.isclose(mass, expected_mass, rel_tol=1.0e-12)
    assert math.isclose(
        sum(profile.SOFT_PACKAGE_MASSES[:2]),
        0.35,
        rel_tol=1.0e-12,
    )
    assert math.isclose(
        sum(profile.SOFT_PACKAGE_MASSES[2:]),
        0.165,
        rel_tol=1.0e-12,
    )


def test_soft_package_targets_form_opposed_leap_fingertip_pinches() -> None:
    builder = _builder()
    profile = _profile()
    assert len(profile.LEAP_SOFT_PINCH_TARGETS_BY_SIDE) == 2

    def body_box(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body_name: str,
        bounds: tuple[tuple[float, ...], tuple[float, ...]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower = np.asarray(bounds[0], dtype=np.float64)
        upper = np.asarray(bounds[1], dtype=np.float64)
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        rotation = data.xmat[body_id].reshape(3, 3)
        center = data.xpos[body_id] + rotation @ (0.5 * (lower + upper))
        return center, 0.5 * (upper - lower), rotation

    for side_index, (side, targets) in enumerate(zip(
        builder.HAND_SIDES,
        profile.LEAP_SOFT_PINCH_TARGETS_BY_SIDE,
        strict=True,
    )):
        model = mujoco.MjModel.from_xml_path(
            str(builder.LEAP_ASSET_ROOT / f"{side}_hand.xml")
        )
        data = mujoco.MjData(model)
        for joint_name, target in zip(
            builder.LEAP_FINGER_JOINT_NAMES, targets, strict=True
        ):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            joint_range = model.jnt_range[joint_id]
            assert joint_range[0] <= target <= joint_range[1]
            data.qpos[model.jnt_qposadr[joint_id]] = target
        mujoco.mj_forward(model, data)

        index_center, _, _ = body_box(
            model, data, "if_ds", builder.LEAP_TIP_BOUNDS
        )
        middle_center, middle_half_size, middle_rotation = body_box(
            model, data, "mf_ds", builder.LEAP_TIP_BOUNDS
        )
        thumb_center, thumb_half_size, thumb_rotation = body_box(
            model, data, "th_ds", builder.LEAP_THUMB_TIP_BOUNDS
        )
        pinch_delta = middle_center - thumb_center
        center_gap = float(np.linalg.norm(pinch_delta))
        pinch_axis = pinch_delta / center_gap
        middle_radius = float(
            np.abs(pinch_axis @ middle_rotation) @ middle_half_size
        )
        thumb_radius = float(
            np.abs(pinch_axis @ thumb_rotation) @ thumb_half_size
        )
        surface_gap = center_gap - middle_radius - thumb_radius

        assert 0.054 < center_gap < 0.055
        assert np.linalg.norm(pinch_delta[1:]) < 0.0001
        assert math.isclose(surface_gap, 0.0205, abs_tol=0.0001)

        source_pinch_center = 0.5 * (middle_center + thumb_center)
        target = profile.HAND_STAGE_YELLOW_FLIP_CONTACT_TARGETS[
            side_index
        ]
        assert target[3:] == (0.0, 0.0, 0.0)
        world_pinch_center = (
            np.asarray(builder.HAND_ROOT_POSITIONS[side_index])
            + np.asarray(target[:3])
            + np.asarray(builder.HAND_PALM_ALIGNMENT_ROTATION)
            @ source_pinch_center
        )
        np.testing.assert_allclose(
            world_pinch_center,
            profile.YELLOW_FLIP_PINCH_WORLD_CENTERS[side_index],
            atol=2.0e-6,
        )

        regular_centers = [index_center, middle_center]
        for body_name in ("rf_ds",):
            regular_center, _, _ = body_box(
                model, data, body_name, builder.LEAP_TIP_BOUNDS
            )
            regular_centers.append(regular_center)
            np.testing.assert_allclose(
                regular_center[[0, 2]],
                index_center[[0, 2]],
                atol=0.0002,
            )

        lateral_steps = np.diff(
            np.asarray([center[1] for center in regular_centers])
        )
        expected_step = -0.0454 if side == "left" else 0.0454
        np.testing.assert_allclose(
            lateral_steps, (expected_step, expected_step), atol=0.0001
        )
        for offset in (4, 8):
            np.testing.assert_allclose(
                targets[offset:offset + 4], targets[0:4], atol=0.0
            )
        np.testing.assert_allclose(
            np.asarray(targets)[[1, 5, 9]], 0.0, atol=0.0
        )

    assert len(profile.LEAP_BLUE_AIR_PINCH_TARGETS_BY_SIDE) == 2
    palm_alignment = np.asarray(builder.HAND_PALM_ALIGNMENT_ROTATION)
    for side_index, (side, targets) in enumerate(zip(
        builder.HAND_SIDES,
        profile.LEAP_BLUE_AIR_PINCH_TARGETS_BY_SIDE,
        strict=True,
    )):
        model = mujoco.MjModel.from_xml_path(
            str(builder.LEAP_ASSET_ROOT / f"{side}_hand.xml")
        )
        data = mujoco.MjData(model)
        for joint_name, target_value in zip(
            builder.LEAP_FINGER_JOINT_NAMES, targets, strict=True
        ):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            joint_range = model.jnt_range[joint_id]
            assert joint_range[0] <= target_value <= joint_range[1]
            data.qpos[model.jnt_qposadr[joint_id]] = target_value
        mujoco.mj_forward(model, data)

        index_center, _, _ = body_box(
            model, data, "if_ds", builder.LEAP_TIP_BOUNDS
        )
        middle_center, middle_half_size, middle_rotation = body_box(
            model, data, "mf_ds", builder.LEAP_TIP_BOUNDS
        )
        thumb_center, thumb_half_size, thumb_rotation = body_box(
            model, data, "th_ds", builder.LEAP_THUMB_TIP_BOUNDS
        )
        pinch_delta = middle_center - thumb_center
        center_gap = float(np.linalg.norm(pinch_delta))
        pinch_axis = pinch_delta / center_gap
        middle_radius = float(
            np.abs(pinch_axis @ middle_rotation) @ middle_half_size
        )
        thumb_radius = float(
            np.abs(pinch_axis @ thumb_rotation) @ thumb_half_size
        )
        surface_gap = center_gap - middle_radius - thumb_radius
        world_delta = palm_alignment @ pinch_delta

        assert abs(world_delta[0]) < 1.0e-6
        assert 0.0469 < world_delta[1] < 0.0471
        assert -0.0301 < world_delta[2] < -0.0299
        assert 0.0204 < surface_gap < 0.0206

        source_pinch_center = 0.5 * (middle_center + thumb_center)
        stage_target = profile.HAND_STAGE_BLUE_FLIP_CONTACT_TARGETS[
            side_index
        ]
        assert stage_target[3:] == (0.0, 0.0, 0.0)
        world_pinch_center = (
            np.asarray(builder.HAND_ROOT_POSITIONS[side_index])
            + np.asarray(stage_target[:3])
            + palm_alignment @ source_pinch_center
        )
        np.testing.assert_allclose(
            world_pinch_center,
            profile.BLUE_FLIP_PINCH_WORLD_CENTERS[side_index],
            atol=2.0e-6,
        )

        for body_name in ("mf_ds", "rf_ds"):
            regular_center, _, _ = body_box(
                model, data, body_name, builder.LEAP_TIP_BOUNDS
            )
            np.testing.assert_allclose(
                regular_center[[0, 2]],
                index_center[[0, 2]],
                atol=0.0002,
            )
        for offset in (4, 8):
            np.testing.assert_allclose(
                targets[offset:offset + 4], targets[0:4], atol=0.0
            )
        np.testing.assert_allclose(
            np.asarray(targets)[[1, 5, 9]], 0.0, atol=0.0
        )

    left, right = profile.LEAP_SOFT_PINCH_TARGETS_BY_SIDE
    assert left[13] < 0.0 < right[13]
    blue_left, blue_right = profile.LEAP_BLUE_AIR_PINCH_TARGETS_BY_SIDE
    assert blue_left[13] < 0.0 < blue_right[13]


class _FakeNativeContext:
    def __init__(self, state: dict[str, object]) -> None:
        self.state = state
        self.link_force_calls: list[
            tuple[str, str, tuple[float, ...], tuple[float, ...]]
        ] = []
        self.deformable_force_calls: list[tuple[int, np.ndarray]] = []

    def get_physics_state_view(self) -> dict[str, object]:
        return self.state

    def set_link_external_force(
        self,
        robot_name: str,
        link_name: str,
        point: tuple[float, ...],
        force: tuple[float, ...],
    ) -> None:
        self.link_force_calls.append(
            (
                str(robot_name),
                str(link_name),
                tuple(float(value) for value in point),
                tuple(float(value) for value in force),
            )
        )

    def set_deformable_external_forces(
        self, stable_id: int, forces: object
    ) -> None:
        array = np.asarray(forces, dtype=np.float64)
        assert array.ndim == 2 and array.shape[1] == 3
        self.deformable_force_calls.append((int(stable_id), array.copy()))


def _identity_transform(position: tuple[float, float, float]) -> dict[str, object]:
    return {
        "position": position,
        "matrix": (
            (1.0, 0.0, 0.0, position[0]),
            (0.0, 1.0, 0.0, position[1]),
            (0.0, 0.0, 1.0, position[2]),
            (0.0, 0.0, 0.0, 1.0),
        ),
    }


def _native_rigid_state() -> dict[str, object]:
    def robot(name: str, velocity_x: float, x: float) -> dict[str, object]:
        return {
            "name": name,
            "links": [
                {
                    "name": name,
                    "linear_velocity": (velocity_x, 0.0, 0.0),
                    "global_transform": _identity_transform((x, 0.58, 0.62)),
                }
            ],
        }

    return {
        "robots": [
            robot("carton_small", 0.0, -0.2),
            robot("carton_wide", 0.0, 0.2),
        ],
        "contacts": [
            {
                "robot_name": "carton_small",
                "link_name": "carton_small",
                "other_robot_name": "conveyor",
                "other_link_name": "belt_surface",
                "normal_force": 6.0,
            },
            {
                "robot_name": "carton_small",
                "link_name": "carton_small",
                "other_robot_name": "conveyor",
                "other_link_name": "belt_surface",
                "normal_force": 4.0,
            },
            {
                "robot_name": "carton_wide",
                "link_name": "carton_wide",
                "other_robot_name": "conveyor",
                "other_link_name": "belt_surface",
                "normal_force": 30.0,
            },
        ],
        "deformables": [],
    }


def _native_deformable_state(
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, object]:
    def body(
        name: str,
        stable_id: int,
        x_offset: float,
        normal_forces: tuple[float, float, float],
    ) -> dict[str, object]:
        vertices = np.asarray(
            (
                (x_offset, 0.58, 0.56),
                (x_offset + 0.2, 0.58, 0.56),
                (1.5, 0.58, 0.56),
            ),
            dtype=np.float64,
        )
        velocities = np.repeat(
            np.asarray(velocity, dtype=np.float64)[None, :],
            len(vertices),
            axis=0,
        )
        contacts = np.zeros_like(vertices)
        contacts[:, 2] = normal_forces
        return {
            "name": name,
            "stable_id": stable_id,
            "global_transform": _identity_transform((0.0, 0.0, 0.0)),
            "local_vertices": vertices.copy(),
            "world_vertices": vertices,
            "local_velocities": velocities,
            "contact_forces_world": contacts,
        }

    return {
        "robots": [],
        "contacts": [],
        "deformables": [
            body("soft_mailer_blue", 101, -0.2, (2.0, 4.0, 6.0)),
            body("soft_pouch_yellow", 202, -0.1, (1.0, 2.0, 3.0)),
        ],
    }


def test_native_rigid_conveyor_force_is_coulomb_limited() -> None:
    context = _FakeNativeContext(_native_rigid_state())
    model = _forces().ConveyorForceModel(
        context,
        ("carton_small", "carton_wide"),
        (1.0, 2.0),
        belt_robot="conveyor",
        belt_link="belt_surface",
        friction_coefficient=0.5,
        fixed_dt=0.1,
        max_acceleration=4.0,
    )

    assert np.allclose(model.apply(1.0), (4.0, 8.0))
    assert [call[:2] for call in context.link_force_calls] == [
        ("carton_small", "carton_small"),
        ("carton_wide", "carton_wide"),
    ]
    assert np.allclose(
        [call[3] for call in context.link_force_calls],
        ((4.0, 0.0, 0.0), (8.0, 0.0, 0.0)),
    )

    model.clear()
    assert np.count_nonzero(
        np.asarray([call[3] for call in context.link_force_calls[-2:]])
    ) == 0
    assert np.count_nonzero(model.normal_force) == 0
    assert np.count_nonzero(model.drive_force) == 0


def test_native_deformable_force_is_nodal_and_belt_local() -> None:
    context = _FakeNativeContext(_native_deformable_state())
    model = _forces().DeformableConveyorForceModel(
        context,
        ("soft_mailer_blue", "soft_pouch_yellow"),
        (0.6, 0.9),
        friction_coefficient=0.5,
        fixed_dt=0.1,
        belt_half_length=1.0,
        belt_half_width=0.4,
        belt_top=0.56,
        belt_center_y=0.58,
    )

    applied = model.apply(1.0)
    assert np.allclose(
        model.drive_force["soft_mailer_blue"], (1.0, 1.2, 0.0)
    )
    assert np.allclose(
        model.drive_force["soft_pouch_yellow"], (0.5, 1.0, 0.0)
    )
    assert [stable_id for stable_id, _ in context.deformable_force_calls] == [
        101,
        202,
    ]
    for name, (_, force) in zip(
        model.body_names, context.deformable_force_calls, strict=True
    ):
        assert force.shape == (3, 3)
        assert force.flags.c_contiguous
        assert np.allclose(force, applied[name])

    model.clear()
    assert [stable_id for stable_id, _ in context.deformable_force_calls[-2:]] == [
        101,
        202,
    ]
    assert all(
        np.count_nonzero(force) == 0
        for _, force in context.deformable_force_calls[-2:]
    )


def test_native_deformable_velocity_damping_is_mass_normalized() -> None:
    velocity = np.asarray((1.0, -2.0, 0.5), dtype=np.float64)
    context = _FakeNativeContext(
        _native_deformable_state(tuple(float(value) for value in velocity))
    )
    model = _forces().DeformableConveyorForceModel(
        context,
        ("soft_mailer_blue", "soft_pouch_yellow"),
        (0.6, 0.9),
        friction_coefficient=0.5,
        fixed_dt=0.1,
        belt_half_length=1.0,
        belt_half_width=0.4,
        belt_top=0.56,
        belt_center_y=0.58,
        velocity_damping_rates=(2.0, 4.0),
    )

    applied = model.apply(1.0)
    assert np.allclose(applied["soft_mailer_blue"], -0.4 * velocity)
    assert np.allclose(applied["soft_pouch_yellow"], -1.2 * velocity)

    scaled = model.apply(1.0, damping_scale=0.25)
    assert np.allclose(scaled["soft_mailer_blue"], -0.1 * velocity)
    assert np.allclose(scaled["soft_pouch_yellow"], -0.3 * velocity)
    for invalid_scale in (-1.0, math.inf, math.nan):
        with np.testing.assert_raises(ValueError):
            model.apply(1.0, damping_scale=invalid_scale)


def test_native_force_arrows_read_physics_scene_state() -> None:
    play = _play()
    arrows = play._contact_arrows(
        {
            "contacts": [
                {
                    "position": (0.1, 0.2, 0.3),
                    "force": (0.0, 3.0, 4.0),
                },
                {
                    "position": (0.0, 0.0, 0.0),
                    "force": (0.0, 0.0, 0.0),
                },
            ]
        },
        force_scale=0.08,
        max_force_length=0.8,
    )

    assert len(arrows) == 1
    assert np.allclose(arrows[0].start, (0.1, 0.2, 0.3))
    assert np.allclose(arrows[0].vector, (0.0, 0.6, 0.8))
    assert arrows[0].color == play.CONTACT_FORCE_ARROW_COLOR
    assert arrows[0].label == "contact 5 N"
    assert math.isclose(
        play._max_contact_penetration(
            {
                "contacts": [
                    {"distance": -0.0004},
                    {"distance": -0.0012},
                    {"distance": 0.01},
                ]
            }
        ),
        0.0012,
    )


def test_editor_preview_uses_runtime_only_coarse_superdex_meshes() -> None:
    play = _play()
    builder = _builder()

    class PreviewBody:
        pass

    nodes = {
        name: PreviewBody() for name in play.PREVIEW_SOFT_PACKAGE_CELLS
    }
    authored_cells = {
        str(spec["name"]): tuple(spec["cells"])
        for spec in builder.SOFT_PACKAGE_SPECS
    }

    total_nodes = play._apply_preview_deformable_meshes(nodes, builder)

    assert total_nodes == 780
    assert len(nodes["soft_mailer_blue"].surface_mesh.vertices) == 260
    assert len(nodes["soft_mailer_blue_fill"].mesh.vertices) == 72
    assert len(nodes["soft_pouch_yellow"].surface_mesh.vertices) == 308
    assert len(nodes["soft_pouch_yellow_fill"].mesh.vertices) == 140
    assert nodes["soft_mailer_blue"].self_collision_enabled is False
    assert nodes["soft_pouch_yellow"].self_collision_enabled is False
    assert {
        str(spec["name"]): tuple(spec["cells"])
        for spec in builder.SOFT_PACKAGE_SPECS
    } == authored_cells == {
        "soft_mailer_blue": (34, 25),
        "soft_mailer_blue_fill": (12, 9, 5),
        "soft_pouch_yellow": (36, 27),
        "soft_pouch_yellow_fill": (14, 11, 8),
    }
    assert play.PREVIEW_NEWTON_ITERATIONS == 16
    assert play.PREVIEW_LINE_SEARCH_ITERATIONS == 6
    assert math.isclose(play.PREVIEW_PENETRATION_LIMIT_METERS, 1.0e-3)
    assert play.PREVIEW_PENETRATION_HOLD_STEPS == 3


def test_runtime_uses_native_superdex_state_and_external_forces() -> None:
    play_source = (EXAMPLE / "conveyor_packages_play.py").read_text(
        encoding="utf-8"
    )
    batch_source = (EXAMPLE / "conveyor_packages_batch.py").read_text(
        encoding="utf-8"
    )
    force_source = (EXAMPLE / "conveyor_forces.py").read_text(
        encoding="utf-8"
    )
    builder_source = (EXAMPLE / "build_scene.py").read_text(encoding="utf-8")

    for source in (play_source, batch_source, force_source):
        assert "torch" not in source.lower()
        assert "provider" not in source.lower()
        assert "set_runtime_vertices" not in source
    assert "PhysicsCoupling3D" not in builder_source
    assert "PhysicsBackendType.SuperDex" in play_source
    assert "PhysicsBackendType.SuperDex" in batch_source
    assert "record_deformable_contact_forces" in play_source
    assert "record_deformable_contact_forces" in batch_source
    assert "get_physics_state" in play_source
    assert "get_physics_state" in batch_source
    assert "get_physics_state" in force_source
    assert "set_link_external_force" in force_source
    assert "set_deformable_external_forces" in force_source
    assert ".item()" not in force_source


def main() -> None:
    test_scene_is_reproducible()
    test_scene_compiles_to_native_superdex_conveyor_contract()
    test_scene_has_play_script_and_leap_hands()
    test_quality_profiles_and_flip_push_schedule()
    test_native_rigid_conveyor_force_is_coulomb_limited()
    test_native_deformable_force_is_nodal_and_belt_local()
    test_native_deformable_velocity_damping_is_mass_normalized()
    test_native_force_arrows_read_physics_scene_state()
    test_editor_preview_uses_runtime_only_coarse_superdex_meshes()
    test_runtime_uses_native_superdex_state_and_external_forces()


if __name__ == "__main__":
    main()
