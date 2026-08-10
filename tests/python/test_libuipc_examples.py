from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import gobot
import numpy as np
from gobot.ipc import CompiledIpcSceneArtifact
from gobot.rl import CompiledMuJoCoIpcArtifact

from libuipc_test_scenes import TEST_SCENE_NAMES, build_libuipc_test_scene


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples" / "libuipc"
MUJOCO_LIBUIPC_EXAMPLE_ROOT = ROOT / "examples" / "mujoco_libuipc"
SCENE_NAMES = ("fr3_brick_grasp.jscn",)


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "gobot_libuipc_demo_builder", EXAMPLE_ROOT / "build_demos.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_play_script():
    spec = importlib.util.spec_from_file_location(
        "gobot_libuipc_demo_play", EXAMPLE_ROOT / "libuipc_demo.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_mujoco_libuipc_builder():
    spec = importlib.util.spec_from_file_location(
        "gobot_mujoco_libuipc_example_builder",
        MUJOCO_LIBUIPC_EXAMPLE_ROOT / "build_scene.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checked_in_libuipc_scenes_are_reproducible() -> None:
    builder = _load_builder()
    with tempfile.TemporaryDirectory(prefix="gobot-libuipc-demos-") as directory:
        generated = builder.build_demos(Path(directory))
        assert tuple(path.name for path in generated) == SCENE_NAMES
        for name in SCENE_NAMES:
            assert (Path(directory) / name).read_bytes() == (
                EXAMPLE_ROOT / name
            ).read_bytes()
        assert (Path(directory) / "libuipc_demo.py").is_file()
        assert (Path(directory) / "project.gobot").is_file()
        assert (
            Path(directory)
            / "assets"
            / "franka_emika_panda"
            / "urdf"
            / "fr3_franka_hand.urdf"
        ).is_file()
        context = gobot.app.create_context()
        context.set_project_path(directory)
        context.load_scene("res://fr3_brick_grasp.jscn")
        artifact = CompiledIpcSceneArtifact.from_mapping(
            context.compile_ipc_scene_artifact()
        )
        assert len(artifact.deformable_bodies) == 1


def test_mujoco_libuipc_batch_example_is_reproducible_and_mapped() -> None:
    builder = _load_mujoco_libuipc_builder()
    with tempfile.TemporaryDirectory(
        prefix="gobot-mujoco-libuipc-example-"
    ) as directory:
        generated = builder.build_scene(Path(directory))
        checked_in = MUJOCO_LIBUIPC_EXAMPLE_ROOT / "soft_press_batch.jscn"
        assert generated.read_bytes() == checked_in.read_bytes()
        assert (Path(directory) / "mujoco_libuipc_play.py").is_file()
        assert (Path(directory) / "project.gobot").is_file()

        context = gobot.app.create_context()
        context.set_project_path(directory)
        context.load_scene("res://soft_press_batch.jscn")
        artifact = CompiledMuJoCoIpcArtifact.from_context(context)
        assert artifact.mujoco.dimensions["nq"] == 1
        assert artifact.mujoco.dimensions["nu"] == 1
        assert len(artifact.ipc.deformable_bodies) == 1
        assert [
            (mapping.robot_name, mapping.link_name)
            for mapping in artifact.coupled_bodies
        ] == [
            ("static_colliders", "ground"),
            ("press", "press_head"),
        ]

        scene = json.loads(generated.read_text(encoding="utf-8"))
        root = next(node for node in scene["__NODES__"] if node["parent"] == -1)
        assert root["properties"]["script"] == (
            "ExtResource(mujoco_libuipc_play_script)"
        )
        assert any(
            resource["__ID__"] == "mujoco_libuipc_play_script"
            and resource["__PATH__"] == "res://mujoco_libuipc_play.py"
            and resource["__TYPE__"] == "PythonScript"
            for resource in scene["__EXT_RESOURCES__"]
        )
        slide = next(
            node
            for node in scene["__NODES__"]
            if node["name"] == "press_slide"
        )
        assert slide["type"] == "Joint3D"
        assert slide["properties"]["joint_type"] == "Prismatic"
        assert slide["properties"]["drive_mode"] == "Position"


def test_mujoco_libuipc_play_script_uses_the_composite_gpu_provider() -> None:
    source = (
        MUJOCO_LIBUIPC_EXAMPLE_ROOT / "mujoco_libuipc_play.py"
    ).read_text(encoding="utf-8")
    assert "class Script(gobot.NodeScript)" in source
    assert "MuJoCoIpcProvider" in source
    assert "LibuipcBatchSolver" in source
    assert "ProviderPlaySession" in source
    assert "apply_deformable_vertices" in source
    assert "press_view.sync_scene" in source
    assert "NUM_ENVS = 4" in source


def test_libuipc_scenes_compile_to_supported_native_contracts() -> None:
    expected_counts = {
        "fr3_brick_grasp.jscn": (1, 11),
    }
    for name in SCENE_NAMES:
        scene = json.loads((EXAMPLE_ROOT / name).read_text(encoding="utf-8"))
        root = next(node for node in scene["__NODES__"] if node["parent"] == -1)
        assert root["properties"]["script"] == "ExtResource(libuipc_demo_script)"
        scripts = [
            entry
            for entry in scene["__EXT_RESOURCES__"]
            if entry["__PATH__"] == "res://libuipc_demo.py"
        ]
        assert len(scripts) == 1
        assert scripts[0]["__TYPE__"] == "PythonScript"
        assert not any(node["type"] == "TactileSensor3D" for node in scene["__NODES__"])

        context = gobot.app.create_context()
        context.set_project_path(str(EXAMPLE_ROOT))
        context.load_scene("res://" + name)
        artifact = CompiledIpcSceneArtifact.from_mapping(
            context.compile_ipc_scene_artifact()
        )
        affine_links = [
            link
            for robot in artifact.robots
            for link in robot["links"]
            if any(
                not shape.get("disabled", False)
                for shape in link["collision_shapes"]
            )
        ]
        assert (len(artifact.deformable_bodies), len(affine_links)) == expected_counts[name]
        assert not artifact.tactile_sensors
        if name == "fr3_brick_grasp.jscn":
            node_names = {node["name"] for node in scene["__NODES__"]}
            assert "workbench_visual" in node_names
            assert "workbench_collision" in node_names
            assert not {
                "workbench_left_visual",
                "workbench_left_collision",
                "workbench_right_visual",
                "workbench_right_collision",
                "brick_support",
                "brick_support_visual",
                "brick_support_collision",
            } & node_names

            tetrahedral_meshes = [
                resource
                for resource in scene["__SUB_RESOURCES__"]
                if resource["__TYPE__"] == "TetrahedralMesh"
            ]
            assert len(tetrahedral_meshes) == 1
            soft_mesh = tetrahedral_meshes[0]
            vertices = [
                vertex["matrix_data"]["storage"]
                for vertex in soft_mesh["vertices"]
            ]
            extents = tuple(
                max(vertex[axis] for vertex in vertices)
                - min(vertex[axis] for vertex in vertices)
                for axis in range(3)
            )
            assert all(
                abs(actual - expected) < 1.0e-6
                for actual, expected in zip(extents, (0.050, 0.030, 0.025))
            )
            assert len(vertices) == 96
            assert len(soft_mesh["tetrahedra"]) // 4 == 270

            soft_transform = artifact.deformable_bodies[0]["transform"][
                "matrix_row_major"
            ]
            workbench = next(
                shape
                for link in affine_links
                for shape in link["collision_shapes"]
                if shape["name"] == "workbench_collision"
            )
            soft_bottom = float(soft_transform[11]) + min(
                float(vertex[2]) for vertex in vertices
            )
            workbench_top = (
                float(workbench["transform"]["matrix_row_major"][11])
                + 0.5 * float(workbench["size"][2])
            )
            assert abs((soft_bottom - workbench_top) - 2.5e-4) < 1.0e-6

            joints = [
                joint for robot in artifact.robots for joint in robot["joints"]
            ]
            assert {joint["name"] for joint in joints} == {
                "fr3_joint1",
                "fr3_joint2",
                "fr3_joint3",
                "fr3_joint4",
                "fr3_joint5",
                "fr3_joint6",
                "fr3_joint7",
                "fr3_finger_joint1",
                "fr3_finger_joint2",
            }
            assert sum(int(joint["joint_type"]) == 1 for joint in joints) == 7
            assert sum(int(joint["joint_type"]) == 3 for joint in joints) == 2
            assert {int(joint["joint_type"]) for joint in joints} == {1, 3}

            shapes_by_name = {
                shape["name"]: shape
                for link in affine_links
                for shape in link["collision_shapes"]
                if not shape.get("disabled", False)
            }
            assert "workbench_collision" in shapes_by_name
            assert "brick_support_collision" not in shapes_by_name
            assert not any(
                shape_name.endswith("_internal_collision")
                for shape_name in shapes_by_name
            )
            triangle_shapes = {
                shape_name: shape
                for shape_name, shape in shapes_by_name.items()
                if shape["shape_type"] == "triangle_mesh"
            }
            assert set(triangle_shapes) == {
                *(f"fr3_link{index}_collision" for index in range(8)),
                "fr3_hand_collision",
            }
            assert sum(
                int(shape["triangle_count"])
                for shape in triangle_shapes.values()
            ) == 2300
            assert triangle_shapes["fr3_link0_collision"]["contact_type"] == 2
            assert triangle_shapes["fr3_link0_collision"]["contact_affinity"] == 2
            assert workbench["contact_type"] == 1
            assert workbench["contact_affinity"] == 1
            blob_encodings = {
                entry["id"]: entry["encoding"]
                for entry in artifact.manifest_data["blobs"]
            }
            assert {
                blob_encodings[shape["mesh_blob"]]
                for shape in triangle_shapes.values()
            } == {"gobot.triangle-mesh.le.v1"}
            finger_shapes = [
                shape
                for shape_name, shape in shapes_by_name.items()
                if shape_name.startswith(
                    ("fr3_leftfinger_collision", "fr3_rightfinger_collision")
                )
            ]
            assert len(finger_shapes) == 8
            assert all(shape["shape_type"] == "box" for shape in finger_shapes)
            assert all(
                abs(shape["friction"][0] - 1.25) < 1.0e-6
                for shape in finger_shapes
            )

            links_by_name = {
                link["name"]: link
                for robot in artifact.robots
                for link in robot["links"]
            }
            np.testing.assert_allclose(
                links_by_name["fr3_link1"]["inertia_off_diagonal"],
                (1.33179e-5, -0.0001140478, -0.00199503),
                rtol=2.0e-6,
                atol=1.0e-10,
            )
            link7 = links_by_name["fr3_link7"]
            assert abs(link7["mass"] - (0.6271432862 + 0.6544)) < 1.0e-6
            np.testing.assert_allclose(
                link7["center_of_mass"],
                (0.00650101, 0.00853633, 0.05731151),
                rtol=2.0e-6,
                atol=1.0e-8,
            )
            assert np.linalg.norm(link7["inertia_off_diagonal"]) > 1.0e-4


def test_libuipc_contact_scenes_are_generated_test_fixtures() -> None:
    expected_counts = {
        "soft_cube_stack.jscn": (2, 1),
        "soft_cube_press.jscn": (1, 2),
    }
    with tempfile.TemporaryDirectory(prefix="gobot-libuipc-fixtures-") as directory:
        project_path = Path(directory)
        for scene_name in TEST_SCENE_NAMES:
            destination = build_libuipc_test_scene(project_path, scene_name)
            scene = json.loads(destination.read_text(encoding="utf-8"))
            root = next(node for node in scene["__NODES__"] if node["parent"] == -1)
            assert root["properties"]["script"] is None

            context = gobot.app.create_context()
            context.set_project_path(directory)
            context.load_scene("res://" + scene_name)
            artifact = CompiledIpcSceneArtifact.from_mapping(
                context.compile_ipc_scene_artifact()
            )
            affine_links = [
                link
                for robot in artifact.robots
                for link in robot["links"]
                if any(
                    not shape.get("disabled", False)
                    for shape in link["collision_shapes"]
                )
            ]
            assert (
                len(artifact.deformable_bodies),
                len(affine_links),
            ) == expected_counts[scene_name]
            assert all(
                shape["shape_type"] == "box"
                for link in affine_links
                for shape in link["collision_shapes"]
                if not shape.get("disabled", False)
            )


def test_fr3_asset_has_complete_licensed_urdf_and_meshes() -> None:
    asset_root = EXAMPLE_ROOT / "assets" / "franka_emika_panda"
    robot = ET.parse(asset_root / "urdf" / "fr3_franka_hand.urdf").getroot()
    assert robot.get("name") == "fr3"
    assert {link.get("name") for link in robot.findall("link")} >= {
        *(f"fr3_link{index}" for index in range(8)),
        "fr3_hand",
        "fr3_leftfinger",
        "fr3_rightfinger",
    }
    meshes = {
        mesh.get("filename") for mesh in robot.findall("./link/visual/geometry/mesh")
    }
    assert len(meshes) == 10
    assert all(path and path.endswith(".dae") for path in meshes)
    assert all(
        (asset_root / path.removeprefix("package://franka_emika_panda/")).is_file()
        for path in meshes
    )
    collision_meshes = {
        mesh.get("filename")
        for mesh in robot.findall("./link/collision/geometry/mesh")
    }
    assert len(collision_meshes) == 9
    assert all(path and path.endswith(".stl") for path in collision_meshes)
    assert all(
        (asset_root / path.removeprefix("package://franka_emika_panda/")).is_file()
        for path in collision_meshes
    )
    assert "Apache License" in (asset_root / "LICENSE").read_text(encoding="utf-8")
    assert (asset_root / "SOURCE.md").is_file()


def test_fr3_scene_uses_urdf_visual_and_collision_meshes() -> None:
    scene = json.loads(
        (EXAMPLE_ROOT / "fr3_brick_grasp.jscn").read_text(encoding="utf-8")
    )
    meshes = [
        node
        for node in scene["__NODES__"]
        if node["type"] == "MeshInstance3D" and node["name"].startswith("fr3_")
    ]
    assert len(meshes) == 11
    assert all(
        str(node["properties"]["mesh"]).startswith("ExtResource(")
        for node in meshes
    )
    resource_paths = {
        entry["__PATH__"]
        for entry in scene["__EXT_RESOURCES__"]
        if entry["__TYPE__"] == "Mesh"
    }
    assert len([path for path in resource_paths if path.endswith(".dae")]) == 10
    assert len([path for path in resource_paths if path.endswith(".stl")]) == 9
    robot_collisions = [
        node
        for node in scene["__NODES__"]
        if node["type"] == "CollisionShape3D"
        and node["name"].startswith("fr3_")
    ]
    assert len(robot_collisions) == 17
    assert all(
        node["properties"]["visible"] is False for node in robot_collisions
    )
    assert not any(
        node["name"].endswith("_internal_collision")
        for node in scene["__NODES__"]
    )
    assert not any(
        node["name"].startswith("fr3_")
        and str(node.get("properties", {}).get("mesh", "")).startswith(
            "SubResource(BoxMesh"
        )
        for node in scene["__NODES__"]
    )


def test_libuipc_play_script_uses_only_the_native_provider() -> None:
    source = (EXAMPLE_ROOT / "libuipc_demo.py").read_text(encoding="utf-8")
    assert "LibuipcProvider" in source
    assert "ProviderPlaySession" in source
    assert "WarpIpcProvider" not in source
    assert "torch" not in source
    assert "warp" not in source.lower()
    assert "set_debug_arrows" in source
    assert "clear_debug_arrows" in source
    assert "get_physics_debug_settings" in source
    assert "brick_support" not in source

    project = json.loads((EXAMPLE_ROOT / "project.gobot").read_text(encoding="utf-8"))
    assert project["main_scene"] == "res://fr3_brick_grasp.jscn"
    assert set(project["editor_scene_views"]) == {
        "res://" + name for name in SCENE_NAMES
    }


def test_libuipc_play_script_builds_bounded_contact_force_arrows() -> None:
    demo = _load_play_script()
    positions = np.zeros((30, 3), dtype=np.float64)
    positions[:, 0] = np.arange(30, dtype=np.float64)
    forces = np.zeros_like(positions)
    forces[:, 2] = np.arange(1, 31, dtype=np.float64)

    arrows = demo._contact_force_arrows(
        positions,
        forces,
        force_scale=0.08,
        max_force_length=0.8,
    )
    assert len(arrows) == demo.CONTACT_FORCE_ARROW_MAX_COUNT
    assert tuple(arrows[0].start) == (29.0, 0.0, 0.0)
    assert tuple(arrows[0].vector) == (0.0, 0.0, 1.0)
    assert arrows[0].label == "30 N"
    assert 0.0 < arrows[-1].scale <= arrows[0].scale
    assert abs(arrows[0].scale - 0.08 * np.log1p(30.0)) < 1.0e-12
    capped = demo._contact_force_arrows(
        positions,
        forces,
        force_scale=0.08,
        max_force_length=0.01,
    )
    assert capped[0].scale == 0.01
    assert demo._contact_force_arrows(positions, np.zeros_like(forces)) == []


def main() -> int:
    test_checked_in_libuipc_scenes_are_reproducible()
    test_mujoco_libuipc_batch_example_is_reproducible_and_mapped()
    test_mujoco_libuipc_play_script_uses_the_composite_gpu_provider()
    test_libuipc_scenes_compile_to_supported_native_contracts()
    test_libuipc_contact_scenes_are_generated_test_fixtures()
    test_fr3_asset_has_complete_licensed_urdf_and_meshes()
    test_fr3_scene_uses_urdf_visual_and_collision_meshes()
    test_libuipc_play_script_uses_only_the_native_provider()
    test_libuipc_play_script_builds_bounded_contact_force_arrows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
