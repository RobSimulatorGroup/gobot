from __future__ import annotations

from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
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


def _scene_world_transforms(scene: dict[str, object]) -> dict[str, np.ndarray]:
    transforms: list[np.ndarray] = []
    result: dict[str, np.ndarray] = {}
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
        result[str(node["name"])] = world
    return result


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
        "nq": 39,
        "nv": 36,
        "nu": 18,
        "nbody": 27,
        "njoint": 21,
        "ngeom": 32,
        "nsensor": 0,
        "nhfield": 0,
    }
    rigid_names = tuple(str(spec["name"]) for spec in builder.RIGID_BOX_SPECS)
    assert [robot.name for robot in artifact.mujoco.robots] == [
        "warehouse_frame",
        "conveyor",
        "openarm_bimanual",
        *rigid_names,
    ]
    assert [
        (mapping.robot_name, mapping.link_name, mapping.mode)
        for mapping in artifact.coupled_bodies
    ] == [
        ("conveyor", "belt_surface", "OneWay"),
        ("carton_small", "carton_small", "TwoWay"),
        ("carton_tall", "carton_tall", "TwoWay"),
        ("carton_wide", "carton_wide", "TwoWay"),
        ("openarm_bimanual", "openarm_left_ee_base_link", "OneWay"),
        ("openarm_bimanual", "openarm_left_ee_link1", "OneWay"),
        ("openarm_bimanual", "openarm_left_ee_link2", "OneWay"),
        ("openarm_bimanual", "openarm_right_ee_base_link", "OneWay"),
        ("openarm_bimanual", "openarm_right_ee_link1", "OneWay"),
        ("openarm_bimanual", "openarm_right_ee_link2", "OneWay"),
        ("warehouse_frame", "frame", "OneWay"),
    ]
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
    assert len(artifact.mujoco.robots[2].joint_names) == sum(
        len(names) for names in builder.ARM_JOINT_NAMES_BY_SIDE
    )
    assert all(
        len(robot.joint_names) == 1
        for robot in artifact.mujoco.robots[3:]
    )
    belt_mapping = artifact.coupled_bodies[0]
    assert belt_mapping.force_scale == 1.0
    assert belt_mapping.torque_scale == 1.0


def test_scene_has_play_script_and_industrial_visuals() -> None:
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


def test_allegro_visual_meshes_are_enclosed_by_contact_proxies() -> None:
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


def test_quality_profiles_and_belt_schedule() -> None:
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
        1.1248596649061466,
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
    assert "ARM_PUSH_TARGETS" in play_source
    assert "profile_module.IPC_CONTACT_ACTIVATION_DISTANCE" in play_source
    assert "profile_module.IPC_CONTACT_RESISTANCE" in play_source
    assert "self.profile_module.IPC_CONTACT_FRICTION" not in play_source
    assert "arm_push_fraction_at_tick" in batch_source
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
    test_scene_has_play_script_and_industrial_visuals()
    test_quality_profiles_and_belt_schedule()
    test_mujoco_belt_material_overrides_parcel_friction()
    test_rigid_conveyor_force_is_coulomb_limited_and_composable()
    test_deformable_conveyor_force_is_coulomb_limited_and_belt_local()
    test_deformable_velocity_damping_is_mass_normalized_per_body()
    test_deformable_force_arrows_expose_horizontal_and_belt_resultants()
    test_runtime_uses_one_velocity_field_and_lazy_contact_output()


if __name__ == "__main__":
    main()
