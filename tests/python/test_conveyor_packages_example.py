from __future__ import annotations

from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial import cKDTree
import trimesh


OPTIONAL_DEPENDENCY_SKIP_CODE = 77

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    print("Conveyor packages example skipped: torch is unavailable")
    raise SystemExit(OPTIONAL_DEPENDENCY_SKIP_CODE) from error

import gobot
from gobot.rl import CompiledMuJoCoIpcArtifact


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


@lru_cache(maxsize=1)
def _artifact() -> CompiledMuJoCoIpcArtifact:
    context = gobot.app.create_context()
    try:
        context.set_project_path(str(EXAMPLE))
        context.load_scene("res://" + SCENE.name)
        return CompiledMuJoCoIpcArtifact.from_context(context)
    finally:
        context.clear_scene()


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


def test_scene_compiles_to_mixed_package_conveyor_contract() -> None:
    builder = _builder()
    artifact = _artifact()
    assert artifact.mujoco.dimensions == {
        "nq": 65,
        "nv": 62,
        "nu": 44,
        "nbody": 50,
        "njoint": 47,
        "ngeom": 153,
        "nsensor": 0,
        "nhfield": 0,
    }
    rigid_names = tuple(str(spec["name"]) for spec in builder.RIGID_BOX_SPECS)
    assert [robot.name for robot in artifact.mujoco.robots] == [
        "warehouse_frame",
        "conveyor",
        *builder.LEAP_ROBOT_NAMES,
        *rigid_names,
    ]
    mappings = [
        (mapping.robot_name, mapping.link_name, mapping.mode)
        for mapping in artifact.coupled_bodies
    ]
    assert ("conveyor", "belt_surface", "OneWay") in mappings
    assert ("warehouse_frame", "frame", "OneWay") in mappings
    for name in rigid_names:
        assert (name, name, "TwoWay") in mappings
    for robot_name in builder.LEAP_ROBOT_NAMES:
        assert {
            link_name
            for mapped_robot, link_name, mode in mappings
            if mapped_robot == robot_name and mode == "OneWay"
        } == set(builder.LEAP_CONTACT_LINK_NAMES)
    assert [body["name"] for body in artifact.ipc.deformable_bodies] == [
        str(spec["name"]) for spec in builder.SOFT_PACKAGE_SPECS
    ]
    for body, spec in zip(
        artifact.ipc.deformable_bodies,
        builder.SOFT_PACKAGE_SPECS,
        strict=True,
    ):
        model = str(spec.get("model", "volumetric"))
        assert body["model"] == model
        if model == "thin_shell":
            cells_x, cells_y = spec["cells"]
            assert int(body["vertex_count"]) == (
                2 * (cells_x + 1) * (cells_y + 1)
            )
            assert int(body["tetrahedron_count"]) == 0
            assert int(body["surface_triangle_count"]) == (
                4 * cells_x * cells_y + 4 * (cells_x + cells_y)
            )
            assert math.isclose(
                float(body["thickness"]), spec["thickness"], rel_tol=1.0e-6
            )
            assert math.isclose(
                float(body["bending_stiffness"]),
                spec["bending_stiffness"],
                rel_tol=1.0e-6,
            )
        else:
            cells_x, cells_y, cells_z = spec["cells"]
            assert int(body["vertex_count"]) == (
                (cells_x + 1) * (cells_y + 1) * (cells_z + 1)
            )
            assert int(body["tetrahedron_count"]) == (
                6 * cells_x * cells_y * cells_z
            )
            assert int(body["surface_triangle_count"]) == 4 * (
                cells_x * cells_y
                + cells_x * cells_z
                + cells_y * cells_z
            )
        assert math.isclose(
            float(body["young_modulus"]), float(spec["young_modulus"])
        )
        assert math.isclose(
            float(body["density"]),
            float(spec["density"]),
            rel_tol=1.0e-6,
        )

    conveyor = next(
        robot for robot in artifact.ipc.robots if robot["name"] == "conveyor"
    )
    belt_shape = conveyor["links"][0]["collision_shapes"][0]
    assert belt_shape["name"] == "moving_belt_collision"
    assert math.isclose(
        float(belt_shape["material"]["sliding_friction"]),
        1.0e-5,
        rel_tol=1.0e-5,
    )
    assert all(
        math.isclose(float(actual), float(expected), rel_tol=1.0e-6)
        for actual, expected in zip(
            belt_shape["size"],
            (
                builder.BELT_PROXY_LENGTH,
                builder.BELT_WIDTH,
                builder.BELT_THICKNESS,
            ),
            strict=True,
        )
    )
    assert artifact.mujoco.robots[0].joint_names == ()
    assert artifact.mujoco.robots[1].joint_names == ()
    for robot, expected_names in zip(
        artifact.mujoco.robots[2:4],
        builder.HAND_JOINT_NAMES_BY_SIDE,
        strict=True,
    ):
        assert robot.joint_names == tuple(
            f"{robot.name}_{name}" for name in expected_names
        )
    assert all(
        len(robot.joint_names) == 1
        for robot in artifact.mujoco.robots[4:]
    )
    belt_mapping = artifact.coupled_bodies[0]
    assert belt_mapping.force_scale == 1.0
    assert belt_mapping.torque_scale == 1.0


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
        _builder().WORKTABLE_TOP_Z + 0.235,
    )
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
    assert tuple(yellow_spec["cells"]) == (9, 7, 4)
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
    assert initial_bottom_gap >= 0.18
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
    assert blue_spec["model"] == "thin_shell"
    assert tuple(blue_spec["cells"]) == (26, 19)
    assert float(blue_spec["young_modulus"]) >= 1.0e5
    assert float(blue_spec["bending_stiffness"]) <= 5.0e-4
    assert float(blue_spec["thickness"]) <= 1.5e-3
    assert fill_spec["model"] == "volumetric"
    assert tuple(fill_spec["cells"]) == (10, 8, 4)
    assert not bool(fill_spec["visible"])
    assert tuple(yellow_spec["cells"]) == (9, 7, 4)

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
    assert float(top_z.max() - bottom_z.min()) > 0.10
    assert float(np.ptp(top_z)) > 0.06
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
    assert (
        float(blue_spec["position"][2])
        + float(bottom_z.min())
        - builder.WORKTABLE_TOP_Z
        >= 0.18
    )

    properties_by_unique_name = {
        str(node["name"]): node["properties"]
        for node in scene["__NODES__"]
        if node["name"]
        in {
            "worktable_surface_collision",
            "leap_left_if_tip_collision",
            "soft_mailer_blue",
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
        linear = joint["name"].rsplit("_", 1)[-1] in {"x", "y", "z"}
        expected_stiffness = (
            builder.HAND_STAGE_LINEAR_STIFFNESS
            if linear
            else builder.HAND_STAGE_ANGULAR_STIFFNESS
        )
        assert math.isclose(properties["drive_stiffness"], expected_stiffness)


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
    ) == (32, 8, 2.0e-3)
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
            mesh = _builder()._soft_package_mesh(spec["size"], spec["cells"])
            mass = _tetrahedral_mass(mesh, spec["density"])
        assert math.isclose(mass, expected_mass, rel_tol=1.0e-12)
    assert math.isclose(
        sum(profile.SOFT_PACKAGE_MASSES[:2]),
        0.35,
        rel_tol=1.0e-12,
    )


def test_quality_profiles_and_flip_push_schedule() -> None:
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

    phases_and_durations = (
        ("drop_settle", profile.DROP_SETTLE_TICKS),
        ("flip_approach", profile.FLIP_APPROACH_TICKS),
        ("flip_grip", profile.FLIP_GRIP_TICKS),
        ("flip_lift", profile.FLIP_LIFT_TICKS),
        ("flip_rotate", profile.FLIP_ROTATE_TICKS),
        ("flip_place", profile.FLIP_PLACE_TICKS),
        ("flip_release", profile.FLIP_RELEASE_TICKS),
        ("flip_clear", profile.FLIP_CLEAR_TICKS),
        ("reorient", profile.REORIENT_TICKS),
        ("push_approach", profile.PUSH_APPROACH_TICKS),
        ("push", profile.PUSH_TICKS),
        ("push_release", profile.PUSH_RELEASE_TICKS),
        ("push_clear", profile.PUSH_CLEAR_TICKS),
    )
    tick = 0
    for phase_name, duration in phases_and_durations:
        assert profile.cycle_phase(tick) == phase_name
        assert profile.cycle_phase(tick + duration - 1) == phase_name
        tick += duration
    assert tick == profile.MANIPULATION_TICKS
    assert profile.cycle_phase(tick) == "settle"
    assert profile.cycle_phase(profile.CYCLE_TICKS - 1) == "settle"

    flip_grip_start = profile.DROP_SETTLE_TICKS + profile.FLIP_APPROACH_TICKS
    assert profile.finger_close_fraction_at_tick(flip_grip_start - 1) == 0.0
    assert 0.0 < profile.finger_close_fraction_at_tick(flip_grip_start)
    assert math.isclose(
        profile.finger_close_fraction_at_tick(
            flip_grip_start + profile.FLIP_GRIP_TICKS - 1
        ),
        profile.FLIP_FINGER_CLOSE_FRACTION,
    )
    assert all(
        len(targets) == 6
        for pair in (
            profile.HAND_STAGE_HOME_TARGETS,
            profile.HAND_STAGE_FLIP_GRIP_TARGETS,
            profile.HAND_STAGE_FLIPPED_LIFT_TARGETS,
            profile.HAND_STAGE_PUSH_TARGETS,
        )
        for targets in pair
    )
    assert all(
        len(controls) == 22
        for controls in profile.hand_controls_at_tick(flip_grip_start)
    )
    grip_preload = tuple(
        grip[0] - approach[0]
        for approach, grip in zip(
            profile.HAND_STAGE_FLIP_APPROACH_TARGETS,
            profile.HAND_STAGE_FLIP_GRIP_TARGETS,
            strict=True,
        )
    )
    np.testing.assert_allclose(grip_preload, (0.010, -0.010), atol=1.0e-12)
    for grip, lift, flipped in zip(
        profile.HAND_STAGE_FLIP_GRIP_TARGETS,
        profile.HAND_STAGE_FLIP_LIFT_TARGETS,
        profile.HAND_STAGE_FLIPPED_LIFT_TARGETS,
        strict=True,
    ):
        assert math.isclose(grip[0], lift[0], abs_tol=1.0e-12)
        assert math.isclose(lift[0], flipped[0], abs_tol=1.0e-12)
    for before, after in zip(
        profile.HAND_STAGE_FLIP_LIFT_TARGETS,
        profile.HAND_STAGE_FLIPPED_LIFT_TARGETS,
        strict=True,
    ):
        assert math.isclose(abs(after[3] - before[3]), math.pi)
    rotate_start = (
        profile.DROP_SETTLE_TICKS
        + profile.FLIP_APPROACH_TICKS
        + profile.FLIP_GRIP_TICKS
        + profile.FLIP_LIFT_TICKS
    )
    for relative_tick in (
        0,
        profile.FLIP_ROTATE_TICKS // 4,
        profile.FLIP_ROTATE_TICKS // 2 - 1,
        profile.FLIP_ROTATE_TICKS - 1,
    ):
        stage_targets = profile.hand_stage_targets_at_tick(
            rotate_start + relative_tick
        )
        for before, after, target in zip(
            profile.HAND_STAGE_FLIP_LIFT_TARGETS,
            profile.HAND_STAGE_FLIPPED_LIFT_TARGETS,
            stage_targets,
            strict=True,
        ):
            pivot = 0.5 * (
                np.asarray(before[1:3]) + np.asarray(after[1:3])
            )
            expected_radius = np.linalg.norm(
                np.asarray(before[1:3]) - pivot
            )
            assert math.isclose(
                np.linalg.norm(np.asarray(target[1:3]) - pivot),
                expected_radius,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
    assert all(
        target[3:] == (0.0, 0.0, 0.0)
        for target in profile.HAND_STAGE_PUSH_TARGETS
    )

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
            mesh = _builder()._soft_package_mesh(spec["size"], spec["cells"])
            mass = _tetrahedral_mass(mesh, spec["density"])
        assert math.isclose(mass, expected_mass, rel_tol=1.0e-12)
    assert math.isclose(
        sum(profile.SOFT_PACKAGE_MASSES[:2]),
        0.35,
        rel_tol=1.0e-12,
    )


class _FakeView:
    def __init__(self, velocity: torch.Tensor) -> None:
        self.state = SimpleNamespace(base_velocity=velocity)

    def read_state(self):
        return self.state


class _FakeRigidSolver:
    def __init__(self, sensors: dict[str, dict[str, torch.Tensor]]) -> None:
        self._torch = torch
        self._sensors = sensors

    def contact_sensor(self, name: str):
        return self._sensors[name]


class _FakeBeltRigidSolver:
    def __init__(self) -> None:
        self._torch = torch
        self._arrays = {
            "geom_friction": torch.full((1, 4, 3), 0.6),
            "geom_condim": torch.full((4,), 3, dtype=torch.int32),
            "geom_priority": torch.zeros(4, dtype=torch.int32),
        }
        self.recompute_calls = 0

    def resolve_object_ids(self, object_type: str, names: tuple[str, ...]):
        assert object_type == "geom"
        assert names == ("conveyor_moving_belt_collision",)
        return (2,)

    def model_array(self, name: str) -> torch.Tensor:
        return self._arrays[name]

    def recompute_constants(self) -> None:
        self.recompute_calls += 1


class _FakeProvider:
    def __init__(self, sensors: dict[str, dict[str, torch.Tensor]]) -> None:
        self.num_envs = 2
        self.rigid_solver = _FakeRigidSolver(sensors)
        self.arrays = {"xfrc_applied": torch.zeros(2, 4, 6)}
        self.sense_calls = 0

    def sense(self) -> None:
        self.sense_calls += 1


def test_mujoco_belt_material_overrides_parcel_friction() -> None:
    rigid = _FakeBeltRigidSolver()
    provider = SimpleNamespace(rigid_solver=rigid)
    _forces().configure_mujoco_velocity_field_belt(
        provider, "conveyor_moving_belt_collision"
    )

    assert rigid.recompute_calls == 1
    assert torch.count_nonzero(rigid._arrays["geom_friction"][:, 2]) == 0
    assert torch.all(rigid._arrays["geom_friction"][:, (0, 1, 3)] == 0.6)
    assert rigid._arrays["geom_condim"].tolist() == [3, 3, 1, 3]
    assert rigid._arrays["geom_priority"].tolist() == [0, 0, 1, 0]


class _FakeDeformableProvider:
    def __init__(self) -> None:
        self.num_envs = 2
        self.rigid_solver = SimpleNamespace(_torch=torch)
        positions = torch.zeros(2, 6, 3)
        positions[..., 1] = 0.58
        positions[:, 0, 1] = 0.0
        positions[..., 2] = 0.56
        positions[:, 5, 2] = 0.75
        velocities = torch.zeros_like(positions)
        contact_forces = torch.zeros_like(positions)
        contact_forces[..., 2] = torch.tensor(
            [[2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
             [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
        )
        self.arrays = {
            "ipc_positions": positions,
            "ipc_velocities": velocities,
            "ipc_contact_forces": contact_forces,
            "ipc_external_forces": torch.zeros_like(positions),
        }


def test_rigid_conveyor_force_is_coulomb_limited_and_composable() -> None:
    def sensor(normal_forces: tuple[tuple[float, float], ...]):
        force = torch.zeros(2, 2, 3)
        force[..., 0] = torch.tensor(normal_forces)
        return {
            "force": force,
            "found": torch.tensor([[True, True], [True, True]]),
        }

    sensors = {
        "first": sensor(((6.0, 4.0), (3.0, 2.0))),
        "second": sensor(((12.0, 8.0), (5.0, 3.0))),
    }
    provider = _FakeProvider(sensors)
    provider.arrays["xfrc_applied"][:, 1, 2] = 3.0
    provider.arrays["xfrc_applied"][:, 3, 2] = -2.0
    external = provider.arrays["xfrc_applied"].clone()
    views = (
        _FakeView(torch.zeros(2, 6)),
        _FakeView(torch.zeros(2, 6)),
    )
    model = _forces().ConveyorForceModel(
        provider,
        views,
        (1, 3),
        ("first", "second"),
        (1.0, 2.0),
        friction_coefficient=0.5,
        fixed_dt=0.1,
    )

    model.apply(torch.tensor([1.0, 0.5]))
    assert provider.sense_calls == 1
    assert torch.allclose(
        model.drive_force,
        torch.tensor([[4.0, 8.0], [2.5, 4.0]]),
    )
    assert torch.allclose(
        provider.arrays["xfrc_applied"][:, (1, 3), 0],
        model.drive_force,
    )
    assert torch.allclose(
        provider.arrays["xfrc_applied"][:, (1, 3), 2],
        external[:, (1, 3), 2],
    )

    for value in sensors.values():
        value["force"].zero_()
        value["found"].zero_()
    model.apply(0.0)
    assert torch.allclose(provider.arrays["xfrc_applied"], external)
    model.clear()
    assert torch.allclose(provider.arrays["xfrc_applied"], external)


def test_deformable_conveyor_force_is_coulomb_limited_and_belt_local() -> None:
    provider = _FakeDeformableProvider()
    entries = (
        {"element_offset": 0, "element_count": 3},
        {"element_offset": 3, "element_count": 3},
    )
    model = _forces().DeformableConveyorForceModel(
        provider,
        entries,
        (0.6, 0.9),
        friction_coefficient=0.5,
        fixed_dt=0.1,
        belt_half_length=1.0,
        belt_half_width=0.4,
        belt_top=0.56,
        belt_center_y=0.58,
    )

    model.apply(torch.tensor([1.0, 0.5]))
    expected = torch.tensor(
        [[0.0, 1.2, 1.2, 1.8, 1.8, 0.0],
         [0.0, 1.0, 1.2, 1.8, 1.8, 0.0]]
    )
    assert torch.allclose(model.drive_force, expected)
    assert torch.allclose(
        provider.arrays["ipc_external_forces"][..., 0], expected
    )
    assert torch.count_nonzero(
        provider.arrays["ipc_external_forces"][..., 1:]
    ) == 0
    model.clear()
    assert torch.count_nonzero(provider.arrays["ipc_external_forces"]) == 0


def test_deformable_velocity_damping_is_mass_normalized_per_body() -> None:
    provider = _FakeDeformableProvider()
    provider.arrays["ipc_contact_forces"].zero_()
    velocity = torch.tensor((1.0, -2.0, 0.5))
    provider.arrays["ipc_velocities"].copy_(velocity)
    entries = (
        {"element_offset": 0, "element_count": 3},
        {"element_offset": 3, "element_count": 3},
    )
    model = _forces().DeformableConveyorForceModel(
        provider,
        entries,
        (0.6, 0.9),
        friction_coefficient=0.5,
        fixed_dt=0.1,
        belt_half_length=1.0,
        belt_half_width=0.4,
        belt_top=0.56,
        belt_center_y=0.58,
        velocity_damping_rates=(2.0, 4.0),
    )

    model.apply(0.0)
    expected = torch.empty_like(provider.arrays["ipc_external_forces"])
    expected[:, :3].copy_(velocity).mul_(-0.4)
    expected[:, 3:].copy_(velocity).mul_(-1.2)
    assert torch.allclose(provider.arrays["ipc_external_forces"], expected)
    assert torch.count_nonzero(model.drive_force) == 0


def test_deformable_force_arrows_expose_horizontal_and_belt_resultants() -> None:
    play = _play()
    assert play._merge_contiguous_body_ranges(
        ((0, 2), (2, 5), (5, 7)), ((0, 1), (2,))
    ) == ((0, 5), (5, 7))
    positions = np.asarray(
        (
            (0.0, 0.0, 0.5),
            (0.2, 0.0, 0.6),
            (0.4, 0.0, 0.55),
        ),
        dtype=np.float64,
    )
    forces = np.asarray(
        (
            (0.0, 2.0, 20.0),
            (0.0, 3.0, 10.0),
            (0.0, 0.0, 5.0),
        ),
        dtype=np.float64,
    )
    arrows = play._body_resultant_force_arrows(
        positions,
        forces,
        ((0, 2), (2, 3)),
        ("push_target", "supported_only"),
        horizontal_only=True,
        color=play.CONTACT_HORIZONTAL_RESULTANT_COLOR,
        label="horizontal IPC resultant",
        force_scale=0.08,
        max_force_length=0.8,
    )

    assert len(arrows) == 1
    assert np.allclose(arrows[0].vector, (0.0, 1.0, 0.0))
    assert np.allclose(arrows[0].start, (0.1, 0.0, 0.625))
    assert arrows[0].color == play.CONTACT_HORIZONTAL_RESULTANT_COLOR
    assert "push_target horizontal IPC resultant 5 N" == arrows[0].label

    belt_arrows = play._body_resultant_force_arrows(
        positions,
        np.asarray((2.0, 3.0, 0.0), dtype=np.float64),
        ((0, 2), (2, 3)),
        ("push_target", "supported_only"),
        horizontal_only=False,
        force_axis=(1.0, 0.0, 0.0),
        color=play.EXTERNAL_FORCE_RESULTANT_COLOR,
        label="belt drive resultant",
        force_scale=0.08,
        max_force_length=0.8,
    )
    assert len(belt_arrows) == 1
    assert np.allclose(belt_arrows[0].vector, (1.0, 0.0, 0.0))
    assert "push_target belt drive resultant 5 N" == belt_arrows[0].label


def test_runtime_uses_one_velocity_field_and_lazy_contact_output() -> None:
    play_source = (EXAMPLE / "conveyor_packages_play.py").read_text(
        encoding="utf-8"
    )
    batch_source = (EXAMPLE / "conveyor_packages_batch.py").read_text(
        encoding="utf-8"
    )
    force_source = (EXAMPLE / "conveyor_forces.py").read_text(
        encoding="utf-8"
    )
    assert "set_proxy_twist_override" in play_source
    assert "ConveyorForceModel" in play_source
    assert "DeformableConveyorForceModel" in play_source
    assert "export_deformable_contact_forces=True" in play_source
    assert "contact_refresh_interval" in play_source
    assert "GOBOT_CONVEYOR_DROP_ONLY" in play_source
    assert "hand_controls_at_tick" in play_source
    assert "self.soft_force_model.drive_force[0]" in play_source
    assert 'self.provider.arrays["ipc_external_forces"][0]' not in play_source
    assert "profile_module.IPC_CONTACT_ACTIVATION_DISTANCE" in play_source
    assert "profile_module.IPC_CONTACT_RESISTANCE" in play_source
    assert "self.profile_module.IPC_CONTACT_FRICTION" not in play_source
    assert "hand_controls_at_tick" in batch_source
    assert "ConveyorForceModel" in batch_source
    assert "DeformableConveyorForceModel" in batch_source
    assert "--refresh-contact-forces" in batch_source
    assert "--trace-force-flow" in batch_source
    assert "basic_conveyor_forces" in force_source
    assert "configure_mujoco_velocity_field_belt" in force_source
    assert ".item()" not in force_source


def main() -> None:
    test_scene_is_reproducible()
    test_scene_compiles_to_mixed_package_conveyor_contract()
    test_scene_has_play_script_and_leap_hands()
    test_quality_profiles_and_flip_push_schedule()
    test_mujoco_belt_material_overrides_parcel_friction()
    test_rigid_conveyor_force_is_coulomb_limited_and_composable()
    test_deformable_conveyor_force_is_coulomb_limited_and_belt_local()
    test_deformable_velocity_damping_is_mass_normalized_per_body()
    test_deformable_force_arrows_expose_horizontal_and_belt_resultants()
    test_runtime_uses_one_velocity_field_and_lazy_contact_output()


if __name__ == "__main__":
    main()
