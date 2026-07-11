from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from gobot.rl.policy import (
    POLICY_MANIFEST_KEY,
    PolicyManifest,
    policy_manifest_from_checkpoint,
    read_policy_manifest_sidecar,
    scene_bundle_digest,
    write_policy_manifest_sidecar,
)
from gobot.rl.spec import ActionSpec, ObservationSpec, SpecField


def make_specs() -> tuple[ObservationSpec, ActionSpec]:
    observation = ObservationSpec(
        version="test_obs_v1",
        fields=(SpecField("command", 3), SpecField("joint", 2)),
    )
    action = ActionSpec(
        version="test_action_v1",
        fields=(SpecField("left", 1), SpecField("right", 1)),
    )
    return observation, action


def make_manifest() -> PolicyManifest:
    observation, action = make_specs()
    return PolicyManifest(
        task_name="test_velocity",
        task_version="test_velocity_v1",
        observation_spec=observation.metadata(),
        action_spec=action.metadata(),
        joint_names=("left", "right"),
        physics_dt=0.005,
        decimation=4,
        control={
            "mode": "position_offset",
            "default_joint_position": np.asarray([0.1, -0.1], dtype=np.float32),
            "action_scale": (0.25, 0.25),
        },
        scene_path="res://scene.jscn",
        scene_digest="sha256:test",
    )


def test_manifest_round_trip_and_runtime_validation() -> None:
    manifest = make_manifest()
    restored = PolicyManifest.from_json(manifest.to_json())
    assert restored == manifest
    assert restored.policy_dt == 0.02

    observation, action = make_specs()
    restored.validate_runtime(
        observation_spec=observation,
        action_spec=action,
        joint_names=("left", "right"),
        physics_dt=0.005,
        decimation=4,
        task_name="test_velocity",
        task_version="test_velocity_v1",
        scene_digest="sha256:test",
    )

    try:
        restored.validate_runtime(
            observation_spec=observation,
            action_spec=action,
            joint_names=("right", "left"),
            physics_dt=0.005,
            decimation=4,
        )
    except RuntimeError as error:
        assert "joint order mismatch" in str(error)
    else:
        raise AssertionError("joint order mismatch was not rejected")


def test_checkpoint_and_sidecar_transport() -> None:
    manifest = make_manifest()
    checkpoint = {"infos": {POLICY_MANIFEST_KEY: manifest.metadata()}}
    assert policy_manifest_from_checkpoint(checkpoint) == manifest

    with tempfile.TemporaryDirectory() as temporary_directory:
        policy_path = Path(temporary_directory) / "policy.onnx"
        policy_path.touch()
        sidecar = write_policy_manifest_sidecar(policy_path, manifest)
        assert sidecar.name == "policy.onnx.manifest.json"
        assert read_policy_manifest_sidecar(policy_path) == manifest


def test_scene_bundle_digest_tracks_transitive_resources() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        project = Path(temporary_directory)
        asset = project / "asset.bin"
        asset.write_bytes(b"first")
        child = project / "child.jscn"
        child.write_text(
            json.dumps(
                {
                    "__EXT_RESOURCES__": [
                        {"__ID__": "asset", "__PATH__": "res://asset.bin", "__TYPE__": "Resource"}
                    ],
                    "__NODES__": [
                        {
                            "type": "Link3D",
                            "name": "base",
                            "parent": -1,
                            "properties": {
                                "mass": 1.0,
                                "source_path": "res://source.xml",
                                "visible": True,
                                "visualize_debug": True,
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        root = project / "root.jscn"
        script = project / "playback.py"
        script.write_text("COMMAND = 1\n", encoding="utf-8")
        root.write_text(
            json.dumps(
                {
                    "__EXT_RESOURCES__": [
                        {"__ID__": "child", "__PATH__": "res://child.jscn", "__TYPE__": "PackedScene"},
                        {"__ID__": "script", "__PATH__": "res://playback.py", "__TYPE__": "PythonScript"},
                    ],
                    "__NODES__": [],
                }
            ),
            encoding="utf-8",
        )

        first = scene_bundle_digest(project, "res://root.jscn")
        script.write_text("COMMAND = 2\n", encoding="utf-8")
        script_only_change = scene_bundle_digest(project, "res://root.jscn")
        child_data = json.loads(child.read_text(encoding="utf-8"))
        child_data["__NODES__"][0]["properties"].update(
            source_path="res://moved.xml",
            visible=False,
            visualize_debug=False,
        )
        child.write_text(json.dumps(child_data, indent=4), encoding="utf-8")
        editor_only_change = scene_bundle_digest(project, "res://root.jscn")
        child_data["__NODES__"][0]["properties"]["mass"] = 2.0
        child.write_text(json.dumps(child_data), encoding="utf-8")
        physics_change = scene_bundle_digest(project, "res://root.jscn")
        asset.write_bytes(b"second")
        second = scene_bundle_digest(project, "res://root.jscn")
        assert first.startswith("sha256:")
        assert first == script_only_change
        assert first == editor_only_change
        assert first != physics_change
        assert physics_change != second


def main() -> None:
    test_manifest_round_trip_and_runtime_validation()
    test_checkpoint_and_sidecar_transport()
    test_scene_bundle_digest_tracks_transitive_resources()


if __name__ == "__main__":
    main()
