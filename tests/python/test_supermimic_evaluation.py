"""Frozen-policy observations, full-clip clocks and strict result merging."""

from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.g1.supermimic.prepare_reference import prepare
from gobot.rl.supermimic.evaluate import (
    ANCHORS,
    PROTOCOL,
    EvaluationEnv,
    FrozenPolicy,
    configure_precision,
    evaluate,
    hash_value,
    merge_reports,
    summarize,
    validate_row,
)
from gobot.rl.supermimic.config import Config
from gobot.rl.supermimic.env import TrackingEnv
from gobot.rl.supermimic.motion import digest
from gobot.rl.supermimic.models import MultiEncoderAE
from gobot.rl.supermimic.tasks import EncoderSpec
from gobot.rl.training.workflow import config_dict


@pytest.fixture
def scene_config(tmp_path):
    project = os.environ.get("GOBOT_SUPERMIMIC_TEST_PROJECT")
    library = os.environ.get("GOBOT_SUPERMIMIC_TEST_LIBRARY")
    if not project or not library:
        pytest.skip("requires prepared G1 scene and packed motions")
    project, library = Path(project), Path(library)
    profile = project / "assets/supermimic/robot_profile.json"
    joints = json.loads(profile.read_text())["joint_names"]
    fields = {
        name: np.load(library / f"{name}.npy")[:32].astype(np.float32)
        for name in (
            "root_pos_w",
            "root_quat_w",
            "root_lin_vel_w",
            "root_ang_vel_w",
            "joint_pos",
            "joint_vel",
        )
    }
    # Short synthetic dense inputs exercise the actual converter and scene, without a full dataset.
    npz = tmp_path / "dense/npz"
    npz.mkdir(parents=True)
    values = {
        "fps": 50.0,
        "body_names": ["pelvis"],
        "joint_names": joints,
        "joint_pos": fields["joint_pos"],
        "joint_vel": fields["joint_vel"],
    }
    for suffix in ("pos_w", "quat_w", "lin_vel_w", "ang_vel_w"):
        values["body_" + suffix] = fields["root_" + suffix][:, None]
    names = ["clip_a", "clip_b"]
    for name in names:
        np.savez(npz / f"{name}.npz", **values)
    rows = [
        {"name": n, "frames": 32, "keep": True, "npz_sha256": digest(npz / f"{n}.npz")}
        for n in names
    ]
    source = npz.parent
    (source / "manifest_all.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    for name in ("train_clips.json", "eval_clips.json"):
        (source / name).write_text(json.dumps(names))
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "ready_for_paired_training": True,
                "artifacts": {
                    n: digest(source / n)
                    for n in (
                        "manifest_all.jsonl",
                        "train_clips.json",
                        "eval_clips.json",
                    )
                },
            }
        )
    )
    packed = tmp_path / "packed"
    prepare(source, profile, packed)
    return Config(
        project=project,
        library=packed,
        num_envs=2,
        backend="mujoco-cpu",
        device="cpu",
        sampling_mode="start",
        domain_randomization=False,
        reset_noise=False,
        episode_length_s=1e9,
        compile=False,
        capture_graphs=False,
        nconmax=128,
        njmax=500,
        actor_hidden_dims=(32, 16),
    )


def checkpoint(cfg):
    obs_dim = 68 if cfg.actor_obs == "rep_state" else 93
    actor = MultiEncoderAE(
        obs_dim,
        29,
        [EncoderSpec("motion", 640), EncoderSpec("velocity", 3)],
        {"motion": 1.0, "velocity": 0.0},
        32,
        cfg.actor_hidden_dims,
        0.1,
        cfg.history_length,
    )
    profile = cfg.project / "assets/supermimic/robot_profile.json"
    audit = json.loads(profile.with_name("world_physics_audit.json").read_text())
    return {
        "format": "gobot_supermimic_official_v1",
        "config": config_dict(cfg),
        "actor": actor.state_dict(),
        "identity": {"profile": digest(profile), "scene": audit["artifact_digest"]},
    }


def test_policy_and_pelvis_scoring_match_training_environment(scene_config):
    evaluation = EvaluationEnv(scene_config)
    native = TrackingEnv(scene_config)
    configure_precision()
    try:
        policy = FrozenPolicy(checkpoint(scene_config), evaluation)
        state = evaluation.reset(torch.tensor([0, 32]))
        native_state = native.reset(torch.tensor([0, 32]))
        for _ in range(8):
            obs = policy.observation(evaluation, state, evaluation.ids[:0])
            representation = policy.representation
            _, goal = representation.compute_wm_goal(
                native.motion, native.frame, native_state[:, :7]
            )
            torch.testing.assert_close(obs["goal_motion"], goal, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(
                obs["history"][:, 0],
                representation.represent(native_state),
                atol=1e-5,
                rtol=1e-5,
            )
            action = torch.zeros(2, 29)
            state, reward, term, trunc, _ = evaluation.step(action)
            native_state, native_reward, native_term, native_trunc, _ = native.step(
                action
            )
            torch.testing.assert_close(reward, native_reward, atol=1e-6, rtol=1e-5)
            assert torch.equal(term, native_term) and torch.equal(trunc, native_trunc)
        evaluation.reset(evaluation.motion.motion_ends - 2)
        _, _, _, trunc, info = evaluation.step(torch.zeros(2, 29))
        assert evaluation.motion_end.all() and trunc.all()
        assert set(info["anchor_rewards"]) == set(ANCHORS)
    finally:
        evaluation.close()
        native.close()


def test_proprio_history_resets_only_assigned_lanes(scene_config):
    cfg = replace(scene_config, actor_obs="proprio", history_length=3)
    env = EvaluationEnv(cfg)
    try:
        policy = FrozenPolicy(checkpoint(cfg), env)
        initial = env.reset(torch.tensor([0, 32]))
        sensor = env.sensor(initial)
        first = policy.observation(env, initial, env.ids)
        torch.testing.assert_close(first["history"], sensor[:, None].repeat(1, 3, 1))
        state, *_ = env.step(torch.zeros(2, 29))
        next_sensor = env.sensor(state)
        second = policy.observation(env, state, torch.tensor([1]))
        torch.testing.assert_close(
            second["history"][0, :2], sensor[0, None].repeat(2, 1)
        )
        torch.testing.assert_close(second["history"][0, 2], next_sensor[0])
        torch.testing.assert_close(
            second["history"][1], next_sensor[1, None].repeat(3, 1)
        )
        assert policy.actor(second).shape == (2, 29)
    finally:
        env.close()


@pytest.mark.parametrize("actor_obs,history", [("rep_state", 1), ("proprio", 3)])
def test_full_evaluation_and_capped_status(scene_config, tmp_path, actor_obs, history):
    cfg = replace(scene_config, actor_obs=actor_obs, history_length=history)
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint(cfg), path)
    result = evaluate(
        path, cfg.project, cfg.library, tmp_path / "full.json", device="cpu", num_envs=2
    )
    assert result["complete"] and result["summary"]["clips"] == 2
    assert all(
        r["expected_steps"] == 31 and (r["failed"] or r["executed_steps"] == 31)
        for r in result["clips"]
    )
    assert (
        result["policy_adapter"]["actor_obs"] == actor_obs
        and result["common"]["tf32"] is False
    )
    capped = evaluate(
        path,
        cfg.project,
        cfg.library,
        tmp_path / "capped.json",
        device="cpu",
        num_envs=2,
        max_steps=1,
    )
    assert not capped["complete"] and all(r["capped"] for r in capped["clips"])
    with pytest.raises(FileExistsError):
        evaluate(path, cfg.project, cfg.library, tmp_path / "full.json", device="cpu")
    broken = checkpoint(cfg)
    broken["identity"]["scene"] = "different scene"
    torch.save(broken, path)
    with pytest.raises(ValueError, match="scene differs"):
        evaluate(
            path,
            cfg.project,
            cfg.library,
            tmp_path / "wrong_scene.json",
            device="cpu",
            num_envs=2,
        )


def row(name, failed=False):
    return {
        "name": name,
        "frames": 11,
        "expected_steps": 10,
        "executed_steps": 5 if failed else 10,
        "return": 0.8,
        "returns": {"pelvis": 0.8, "torso_link": 0.7},
        "coverage": 0.5 if failed else 1.0,
        "failed": failed,
        "finished": not failed,
        "completed": not failed,
        "capped": False,
    }


def reports(tmp_path):
    names = ["a", "b", "c", "d", "e"]
    common = {"reference": "FP32", "clock": "N-1"}
    paths = []
    for index in range(2):
        rows = [row(n, n == "b") for n in names[index::2]]
        report = {
            "protocol": PROTOCOL,
            "checkpoint_sha256": "ck",
            "policy_adapter": {"kind": "gencoord"},
            "common": common,
            "common_sha256": hash_value(common),
            "selection_sha256": hash_value(names),
            "selection_count": len(names),
            "shard_index": index,
            "shard_count": 2,
            "complete": True,
            "clips": rows,
            "summary": summarize(rows),
        }
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    return paths, names


def test_merge_keeps_manifest_order_and_both_rewards(tmp_path):
    paths, names = reports(tmp_path)
    result = merge_reports(paths[::-1], names, tmp_path / "full.json")
    assert [r["name"] for r in result["clips"]] == names
    assert (
        result["summary"]["failures"] == 1 and result["summary"]["failure_rate"] == 0.2
    )
    assert result["summary"]["frame_coverage"] == 0.9
    assert result["summary"]["anchors"]["torso_link"]["mean_return"] == 0.7


@pytest.mark.parametrize(
    "damage",
    [
        "protocol",
        "common",
        "checkpoint",
        "missing",
        "duplicate",
        "membership",
        "score",
        "capped",
        "summary",
    ],
)
def test_merge_rejects_incompatible_or_incomplete_results(tmp_path, damage):
    paths, names = reports(tmp_path)
    if damage == "missing":
        paths = paths[:1]
    elif damage == "duplicate":
        paths = [paths[0], paths[0]]
    else:
        report = json.loads(paths[0].read_text())
        if damage == "protocol":
            report["protocol"] = "historical"
        if damage == "common":
            report["common"]["clock"] = "N"
        if damage == "checkpoint":
            report["checkpoint_sha256"] = "other"
        if damage == "membership":
            report["clips"][0]["name"] = "unknown"
        if damage == "score":
            report["clips"][0]["returns"]["torso_link"] = float("nan")
        if damage == "capped":
            report["clips"][0]["capped"] = True
        if damage == "summary":
            report["summary"]["failures"] += 1
        paths[0].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        merge_reports(paths, names, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("return", float("nan")),
        ("executed_steps", 12),
        ("coverage", 0.5),
        ("completed", False),
        ("failed", 1),
    ],
)
def test_invalid_clip_result_is_rejected(key, value):
    result = row("a")
    result[key] = value
    with pytest.raises(ValueError):
        validate_row(result)


def test_checkpoint_comparison_rejects_different_conditions(tmp_path):
    from copy import deepcopy
    from examples.g1.supermimic.evaluate import compare

    paths, names = reports(tmp_path)
    baseline = merge_reports(paths, names, tmp_path / "full.json")
    candidate = deepcopy(baseline)
    candidate["checkpoint_sha256"] = "second"
    candidate["clips"][1] = row("b", failed=False)
    candidate["summary"] = summarize(candidate["clips"])
    result = compare([baseline, candidate], {"a", "b"})
    assert result["checkpoints"][1]["fixed_baseline_failures"] == ["b"]
    assert result["checkpoints"][1]["filtered"]["failures"] == 0
    for change in (
        lambda r: r["common"].update(reference="fp16"),
        lambda r: r["clips"][0].update(expected_steps=9),
        lambda r: r["clips"][0].update(capped=True),
        lambda r: r.update(complete=False),
    ):
        broken = deepcopy(candidate)
        change(broken)
        with pytest.raises(ValueError):
            compare([baseline, broken])
