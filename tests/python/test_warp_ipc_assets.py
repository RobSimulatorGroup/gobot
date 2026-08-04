from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "warp_ipc"


def _load_downloader():
    path = EXAMPLE_ROOT / "download_allegro_assets.py"
    spec = importlib.util.spec_from_file_location("gobot_warp_ipc_asset_downloader", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_grasp_example():
    path = EXAMPLE_ROOT / "allegro_tactile_grasp.py"
    spec = importlib.util.spec_from_file_location("gobot_warp_ipc_allegro_grasp", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_allegro_asset_manifest_is_pinned_and_license_complete() -> None:
    manifest = json.loads(
        (EXAMPLE_ROOT / "allegro_assets.json").read_text(encoding="utf-8")
    )
    assert manifest["source"] == "https://github.com/newton-physics/newton-assets.git"
    assert len(manifest["revision"]) == 40
    assert manifest["license"] == {"path": "LICENSE", "spdx": "BSD-2-Clause"}
    files = manifest["files"]
    paths = [entry["path"] for entry in files]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    assert "LICENSE" in paths
    assert "urdf/allegro_hand_description_left.urdf" in paths
    assert sum(int(entry["size"]) for entry in files) == manifest["total_size"]
    assert all(int(entry["size"]) > 0 for entry in files)
    assert all(len(entry["sha256"]) == 64 for entry in files)


def test_checked_in_allegro_scene_is_portable_and_deduplicated() -> None:
    scene_path = EXAMPLE_ROOT / "allegro_tactile_grasp.jscn"
    source = scene_path.read_text(encoding="utf-8")
    assert "/home/" not in source
    assert "package://" not in source
    assert (
        "res://assets/wonik_allegro/urdf/allegro_hand_description_left.urdf"
        in source
    )

    scene = json.loads(source)
    node_types = [node["type"] for node in scene["__NODES__"]]
    subresource_types = [
        resource["__TYPE__"] for resource in scene["__SUB_RESOURCES__"]
    ]
    assert node_types.count("Robot3D") == 1
    assert node_types.count("Link3D") == 21
    assert node_types.count("Joint3D") == 20
    assert node_types.count("TactileSensor3D") == 4
    assert node_types.count("DeformableBody3D") == 1
    assert subresource_types.count("TactileSensorConfig") == 1
    assert subresource_types.count("TetrahedralMesh") == 2
    deformable = next(
        node for node in scene["__NODES__"] if node["type"] == "DeformableBody3D"
    )
    assert deformable["properties"]["kinematic"] is False
    object_position = deformable["properties"]["position"]["matrix_data"]["storage"]
    assert abs(float(object_position[2]) + 0.0072) < 1.0e-8


def test_downloader_is_explicit_and_validates_before_installing() -> None:
    payload = b"pinned Allegro fixture"
    digest = hashlib.sha256(payload).hexdigest()
    fixture_manifest = {
        "asset_root": "fixture",
        "files": [{"path": "LICENSE", "sha256": digest, "size": len(payload)}],
        "license": {"path": "LICENSE", "spdx": "BSD-2-Clause"},
        "revision": "1" * 40,
        "source": "https://github.com/newton-physics/newton-assets.git",
        "total_size": len(payload),
    }

    with tempfile.TemporaryDirectory(prefix="gobot-ipc-assets-") as directory:
        root = Path(directory)
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(fixture_manifest), encoding="utf-8")
        output = root / "assets"

        with patch("urllib.request.urlopen") as urlopen_during_import:
            downloader = _load_downloader()
        urlopen_during_import.assert_not_called()
        downloader.MANIFEST_PATH = manifest_path

        with patch.object(
            downloader.urllib.request,
            "urlopen",
            return_value=io.BytesIO(payload),
        ) as urlopen:
            downloader.download_assets(output)
        urlopen.assert_called_once()
        assert (output / "LICENSE").read_bytes() == payload
        assert not (output / "LICENSE.part").exists()

        with patch.object(
            downloader.urllib.request,
            "urlopen",
            side_effect=AssertionError("validated assets must not be downloaded again"),
        ):
            downloader.download_assets(output)


def _grasp_artifact_fixture(example):
    identity = {
        "matrix_row_major": [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    }
    source_order = (
        "joint_80",
        "joint_90",
        "joint_100",
        "joint_110",
        "joint_120",
        "joint_130",
        "joint_140",
        "joint_150",
        "joint_00",
        "joint_10",
        "joint_20",
        "joint_30",
        "joint_40",
        "joint_50",
        "joint_60",
        "joint_70",
    )
    joints = tuple(
        {
            "joint_type": 1,
            "lower_limit": -0.5 if name != "joint_120" else 0.263,
            "name": name,
            "upper_limit": 1.8,
        }
        for name in source_order
    )
    sensors = tuple(
        {
            "attachment": {
                "link_path": f"/root/hand/{tip_name}",
            },
            "gel_topology_sha256": "sha256:" + "1" * 64,
            "gel_vertex_count": 200,
            "marker_positions": tuple((0.0, 0.0) for _ in range(64)),
            "name": f"tactile_{index}",
            "path": f"/root/hand/{tip_name}/tactile_{index}",
            "resolution": (120, 160),
        }
        for index, tip_name in enumerate(
            ("link_30_tip", "link_70_tip", "link_110_tip", "link_150_tip")
        )
    )
    return SimpleNamespace(
        deformable_bodies=(
            {"name": "soft_grasp_object", "path": "/root/soft_grasp_object"},
        ),
        digest="sha256:" + "2" * 64,
        robots=(
            {
                "joints": joints,
                "links": (
                    {"name": "palm_link", "path": "/root/hand/palm_link"},
                    *(
                        {"name": tip_name, "path": f"/root/hand/{tip_name}"}
                        for tip_name in (
                            "link_30_tip",
                            "link_70_tip",
                            "link_110_tip",
                            "link_150_tip",
                        )
                    ),
                ),
                "name": "allegro_left",
                "root_link_paths": ("/root/hand/palm_link",),
                "transform": identity,
            },
        ),
        tactile_sensors=sensors,
    )


def test_allegro_grasp_port_preserves_control_and_signal_contract() -> None:
    example = _load_grasp_example()
    artifact = _grasp_artifact_fixture(example)
    program = example.build_grasp_program(artifact)

    assert program.joint_names == tuple(
        joint["name"] for joint in artifact.robots[0]["joints"]
    )
    assert program.frame_count == 80
    first = program.frame(0)
    closed = program.frame(59)
    first_lift = program.frame(60)
    final = program.frame(79)
    assert first.phase == "close"
    assert closed.phase == "close"
    assert closed.joint_position == program.closed_joint_position
    assert first_lift.phase == "lift"
    assert first_lift.base_pose_xyzw[2] == 0.0025
    assert final.base_pose_xyzw[2] == 0.05
    assert final.joint_position == program.closed_joint_position
    assert all(
        lower <= target <= upper
        for target, (lower, upper) in zip(
            program.closed_joint_position, program.joint_limits, strict=True
        )
    )

    summary = example._summary(artifact, program)
    assert summary["sensor_count"] == 4
    assert summary["sensor_resolution"] == [120, 160]
    assert summary["phases"] == {"close": 60, "lift": 20}
    assert summary["signals_2d"] == [
        "rgb",
        "depth",
        "normal",
        "marker_position",
        "marker_flow",
    ]
    assert summary["signals_3d"] == [
        "contact_force",
        "contact_wrench",
        "deformable_position",
        "deformable_velocity",
    ]

    for invalid in (-1, 80):
        try:
            program.frame(invalid)
        except IndexError:
            pass
        else:
            raise AssertionError("out-of-range Allegro grasp frame was accepted")


def test_allegro_grasp_port_keeps_render_and_readback_explicit() -> None:
    source = (EXAMPLE_ROOT / "allegro_tactile_grasp.py").read_text(encoding="utf-8")
    assert "tactile_view.render()" in source
    assert "set_base_pose_targets" in source
    assert "import torch" in source
    assert "from taccel" not in source.lower()
    assert "import warp" not in source
    assert "np.savez_compressed" in source
    assert "set_kinematic_targets" not in source


def test_allegro_grasp_port_runs_all_commands_without_implicit_readback() -> None:
    example = _load_grasp_example()
    artifact = _grasp_artifact_fixture(example)
    program = example.build_grasp_program(artifact)

    class RobotView:
        def __init__(self):
            self.reset_value = None
            self.joint_targets = []
            self.base_targets = []

        def reset(self, reset_mask, **state):
            self.reset_value = (reset_mask.clone(), state)

        def set_position_targets(self, targets):
            self.joint_targets.append(targets.clone())

        def set_base_pose_targets(self, targets):
            self.base_targets.append(targets.clone())

    class TactileView:
        def __init__(self):
            self.render_count = 0

        def render(self):
            self.render_count += 1
            return object()

    class DeformableView:
        def __init__(self):
            self.kinematic_targets = []

        def read_state(self):
            import torch

            return SimpleNamespace(position=torch.zeros((2, 1, 8, 3)))

        def set_kinematic_targets(self, targets, *, target_mask=None):
            self.kinematic_targets.append(
                (targets.clone(), None if target_mask is None else target_mask.clone())
            )

    class RuntimeProvider:
        last = None

        @staticmethod
        def availability():
            return SimpleNamespace(available=True, reason="")

        def __init__(self, artifact, *, num_envs, config):
            self.artifact = artifact
            self.num_envs = num_envs
            self.config = config
            self.graph_captured = False
            self.robot_view = RobotView()
            self.tactile_view = TactileView()
            self.deformable_view = DeformableView()
            self.step_count = 0
            RuntimeProvider.last = self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def create_robot_view(self, **names):
            assert names["joint_names"] == program.joint_names
            return self.robot_view

        def create_tactile_view(self, *, sensor_names):
            assert sensor_names == program.tactile_sensor_paths
            return self.tactile_view

        def create_deformable_view(self, *, body_names):
            assert body_names == program.deformable_body_paths
            return self.deformable_view

        def step(self, *, nsteps):
            self.step_count += nsteps

    with patch.object(example, "WarpIpcProvider", RuntimeProvider):
        example.run(
            None,
            artifact,
            program,
            num_envs=2,
            device="cpu",
            capture_graphs=False,
            output_dir=None,
            record_every=1,
            selected_env=0,
        )

    provider = RuntimeProvider.last
    assert provider is not None
    assert provider.step_count == 80 * example.PHYSICS_SUBSTEPS
    assert provider.tactile_view.render_count == 80
    assert len(provider.robot_view.joint_targets) == 80 * example.PHYSICS_SUBSTEPS
    assert len(provider.robot_view.base_targets) == 80 * example.PHYSICS_SUBSTEPS
    assert provider.deformable_view.kinematic_targets == []
    assert provider.robot_view.reset_value is not None
    assert tuple(provider.robot_view.reset_value[1]["joint_position"].shape) == (2, 16)
    final_base = provider.robot_view.base_targets[-1]
    assert abs(float(final_base[0, 2]) - 0.05) < 1.0e-7


if __name__ == "__main__":
    test_allegro_asset_manifest_is_pinned_and_license_complete()
    test_checked_in_allegro_scene_is_portable_and_deduplicated()
    test_downloader_is_explicit_and_validates_before_installing()
    test_allegro_grasp_port_preserves_control_and_signal_contract()
    test_allegro_grasp_port_keeps_render_and_readback_explicit()
    test_allegro_grasp_port_runs_all_commands_without_implicit_readback()
