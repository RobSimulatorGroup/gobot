import json

import numpy as np
import pytest

from examples.g1.supermimic.prepare_reference import prepare
from gobot.rl.supermimic.motion import digest, inspect_library


def source(tmp_path):
    data = tmp_path / "data"
    (data / "npz").mkdir(parents=True)
    joints = [f"j{i}" for i in range(29)]
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"joint_names": joints}))
    # Values deliberately not representable in FP16.
    values = {
        "joint_pos": np.full((4, 29), 0.1234567, dtype=np.float32),
        "joint_vel": np.full((4, 29), 0.2345678, dtype=np.float32),
        "body_pos_w": np.full((4, 2, 3), 0.3456789, dtype=np.float32),
        "body_quat_w": np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (4, 2, 1)),
        "body_lin_vel_w": np.full((4, 2, 3), 0.4567891, dtype=np.float32),
        "body_ang_vel_w": np.full((4, 2, 3), 0.5678912, dtype=np.float32),
    }
    np.savez(
        data / "npz/clip.npz",
        fps=50.0,
        body_names=["torso_link", "pelvis"],
        joint_names=joints,
        **values,
    )
    (data / "manifest_all.jsonl").write_text(
        json.dumps(
            {
                "name": "clip",
                "keep": True,
                "frames": 4,
                "npz_sha256": digest(data / "npz/clip.npz"),
            }
        )
        + "\n"
    )
    for name in ("train_clips.json", "eval_clips.json"):
        (data / name).write_text('["clip"]')
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "ready_for_paired_training": True,
                "artifacts": {
                    n: digest(data / n)
                    for n in (
                        "manifest_all.jsonl",
                        "train_clips.json",
                        "eval_clips.json",
                    )
                },
            }
        )
    )
    return data, profile, values


def test_preserves_original_fp32_and_joint_order(tmp_path):
    data, profile, values = source(tmp_path)
    output = tmp_path / "packed"
    result = prepare(data, profile, output, workers=2)
    assert result["source"]["every_npz_hash_verified"]
    inspect_library(output, verify_hashes=True)
    for key, original in [
        ("joint_pos", values["joint_pos"]),
        ("root_pos_w", values["body_pos_w"][:, 1]),
    ]:
        actual = np.load(output / f"{key}.npy")
        np.testing.assert_array_equal(actual, original)
        assert actual.dtype == np.float32
        assert not np.array_equal(
            actual, original.astype(np.float16).astype(np.float32)
        )
    with pytest.raises(FileExistsError):
        prepare(data, profile, output)


def test_corrupt_source_is_never_published(tmp_path):
    data, profile, _ = source(tmp_path)
    with (data / "npz/clip.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        prepare(data, profile, tmp_path / "packed")
    assert not (tmp_path / "packed").exists() and not list(
        tmp_path.glob("packed.partial-*")
    )


def test_quantized_source_is_rejected(tmp_path):
    data, profile, values = source(tmp_path)
    values["joint_pos"] = values["joint_pos"].astype(np.float16)
    np.savez(
        data / "npz/clip.npz", fps=50.0, body_names=["torso_link", "pelvis"], **values
    )
    (data / "manifest_all.jsonl").write_text(
        json.dumps(
            {
                "name": "clip",
                "keep": True,
                "frames": 4,
                "npz_sha256": digest(data / "npz/clip.npz"),
            }
        )
        + "\n"
    )
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["artifacts"]["manifest_all.jsonl"] = digest(data / "manifest_all.jsonl")
    (data / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid FP32"):
        prepare(data, profile, tmp_path / "packed")
