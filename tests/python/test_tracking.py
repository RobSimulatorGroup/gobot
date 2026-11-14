"""Tracking regression tests against independent upstream outputs and real physics."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
import gobot
from gobot.rl.tracking import TrackingConfig, TrackingEnv
from gobot.rl.tracking import mdp
from gobot.rl.tracking.math import matrix, quaternion, from_euler
from gobot.rl.tracking.motion import Motion
from gobot.rl.tracking.sampler import AdaptiveSampler
from gobot.rl.kinematics import SceneKinematics
from gobot.rl.tracking.training import default_agent

ROOT = Path(__file__).resolve().parents[2]


def oracle():
    return json.loads((ROOT / "tests/fixtures/mjlab_tracking.json").read_text())


def test_pinned_reference_and_ppo_configuration():
    ref = oracle()
    for name, digest in ref["source_sha256"].items():
        path = ROOT / "mjlab" / name
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    cfg = TrackingConfig()
    assert cfg.reward_weights == ref["weights"]
    agent = default_agent()
    for key in ("num_steps_per_env", "save_interval"):
        assert agent[key] == ref["agent"][key]
    for role in ("actor", "critic"):
        for key in ("hidden_dims", "activation", "obs_normalization"):
            assert json.loads(json.dumps(agent[role][key])) == ref["agent"][role][key]
    for key, value in ref["agent"]["algorithm"].items():
        if key != "class_name":
            assert agent["algorithm"][key] == value, key


def test_observation_reward_termination_equations_match_mjlab():
    ref = oracle()
    v = {k: torch.tensor(x, dtype=torch.float32) for k, x in ref["inputs"].items()}
    reference = tuple(v[k] for k in ("rp", "rr", "rv", "rw"))
    actual = tuple(v[k] for k in ("p", "r", "v", "w"))
    relative = mdp.reanchor(reference, actual, 1)
    torch.testing.assert_close(
        relative[0], torch.tensor(ref["relative_position"]), atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        relative[1], torch.tensor(ref["relative_orientation"]), atol=2e-6, rtol=2e-6
    )
    actor, critic = mdp.observations(
        reference,
        actual,
        v["reference_joint"],
        v["q"],
        v["qd"],
        v["action"],
        v["default"],
        v["bias"],
        anchor=1,
        imu=0,
        imu_position=v["imu_position"],
        imu_orientation=v["imu_orientation"],
    )
    for name, fields in (("actor", actor), ("critic", critic)):
        torch.testing.assert_close(
            torch.cat(tuple(fields.values()), -1),
            torch.tensor(ref[name]),
            atol=3e-6,
            rtol=3e-6,
        )
    terms = mdp.rewards(
        reference,
        actual,
        relative,
        v["q"],
        torch.tensor([[-1.0, 1.0]]).expand(3, -1),
        v["action"],
        v["previous"],
        v["self_hits"],
        anchor=1,
        weights=ref["weights"],
        dt=0.02,
    )
    for key, value in terms.items():
        torch.testing.assert_close(
            value,
            torch.tensor(ref["rates"][key]) * 0.02 * ref["weights"][key],
            rtol=3e-5,
            atol=1e-7,
        )
    ends = mdp.terminations(reference, actual, relative, anchor=1, end_effectors=[2, 3])
    for k, value in ends.items():
        assert value.tolist() == ref["terminations"][k]


def test_sampler_matches_upstream_and_accumulates_failures():
    s = AdaptiveSampler(150, 0.02, kernel_size=3)
    s.failed.copy_(torch.tensor([0.0, 0.02, 0.08, 0.0]))
    frames = torch.tensor([0, 40, 80, 149])
    failed = torch.tensor([False, True, True, False])
    s.record(frames, failed)
    result = s.sample(4, torch.Generator().manual_seed(91))
    ref = oracle()["sampling"]
    assert result.tolist() == ref["frames"]
    assert s.pending.tolist() == ref["pending"]
    p = s.probabilities()
    assert float(p.max()) == pytest.approx(ref["metrics"]["sampling_top1_prob"][0])
    s.record(torch.tensor([45]), torch.tensor([True]))
    before = s.failed.clone()
    s.update()
    torch.testing.assert_close(
        s.failed, before * 0.999 + torch.tensor([0.0, 2.0, 1.0, 0.0]) * 0.001
    )
    assert s.pending.sum() == 0
    assert s.sample(0, torch.Generator()).numel() == 0
    short = AdaptiveSampler(2, 0.02)
    assert short.probabilities().tolist() == [1.0]
    assert short.sample(5, torch.Generator()).eq(0).all()


def archive(path, names=("a", "b"), bodies=("base", "foot"), frames=4):
    q = np.arange(frames * len(names), dtype=np.float32).reshape(frames, -1)
    p = np.zeros((frames, len(bodies), 3), np.float32)
    r = np.zeros((frames, len(bodies), 4), np.float32)
    r[..., 0] = 1
    values = dict(
        joint_pos=q,
        joint_vel=q + 1,
        body_pos_w=p,
        body_quat_w=r,
        body_lin_vel_w=p,
        body_ang_vel_w=p,
        joint_names=np.array(names),
        body_names=np.array(bodies),
        fps=np.array(50.0),
    )
    np.savez(path, **values)
    return values


def test_motion_names_clock_and_invalid_data(tmp_path):
    path = tmp_path / "motion.npz"
    values = archive(path)
    m = Motion(path, ("b", "a"), ("foot", "base"))
    np.testing.assert_equal(m.joint_pos.numpy(), values["joint_pos"][:, [1, 0]])
    with pytest.raises(ValueError, match="missing motion names"):
        Motion(path, ("unknown",), ("base",))
    with pytest.raises(ValueError, match="fps"):
        Motion(path, ("a", "b"), ("base",), dt=0.01)
    del values["joint_names"]
    np.savez(path, **values)
    with pytest.raises(ValueError, match="layout"):
        Motion(path, ("a", "b"), ("base",))
    layout = tmp_path / "layout.json"
    layout.write_text(json.dumps({"joint_names": ["a", "b"]}))
    Motion(path, ("a", "b"), ("base",), layout=layout)
    values["joint_pos"][0, 0] = np.nan
    np.savez(path, **values)
    with pytest.raises(ValueError, match="finite"):
        Motion(path, ("a", "b"), ("base",), layout=layout)


def tiny_config(tmp_path, backend="mujoco-cpu"):
    source = tmp_path / "tiny.xml"
    source.write_text("""<mujoco model="tiny"><compiler angle="radian"/>
      <option gravity="0 0 0"/><worldbody><body name="base" pos="0 0 1"><freejoint/>
      <geom type="sphere" size=".05"/><body name="anchor" pos="0 0 .2">
      <joint name="hinge" axis="0 1 0" range="-2 2"/><geom type="sphere" size=".05"/>
      </body></body></worldbody><actuator><position joint="hinge" kp="20" kv="1" forcerange="-5 5"/></actuator></mujoco>""")
    gobot.set_project_path(str(tmp_path))
    gobot.import_mjcf_scene(str(source), "res://world.jscn", name="tiny")
    context = gobot.app.create_context()
    context.set_project_path(str(tmp_path))
    context.load_scene("res://world.jscn")
    try:
        kin = SceneKinematics(
            context.compile_physics_scene_snapshot(),
            robot_name="tiny",
            base_link="base",
            joint_names=("hinge",),
            tracked_links=("base", "anchor"),
        )
        from gobot.rl.tracking.prepare import _from_root

        n = 12
        p = np.zeros((n, 3), np.float32)
        p[:, 2] = 1
        r = np.zeros((n, 4), np.float32)
        r[:, 0] = 1
        q = np.zeros((n, 1), np.float32)
        arrays = _from_root(kin, p, r, np.zeros_like(p), np.zeros_like(p), q, q)
        np.savez(
            tmp_path / "motion.npz",
            **arrays,
            fps=50.0,
            joint_names=np.array(["hinge"]),
            body_names=np.array(["base", "anchor"]),
        )
    finally:
        context.clear_scene()
    return TrackingConfig(
        project_path=tmp_path,
        robot_name="tiny",
        joint_names=("hinge",),
        body_names=("base", "anchor"),
        base_link="base",
        anchor_link="anchor",
        imu_link="base",
        imu_position=(0.0, 0.0, 0.0),
        end_effectors=("anchor",),
        num_envs=2,
        backend=backend,
        domain_randomization=False,
        reset_noise=False,
        observation_noise=False,
        pushes=False,
        sampling_mode="start",
    )


def test_real_cpu_reset_clock_and_timeout(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg.episode_length_s = 0.04
    env = TrackingEnv(cfg, autoreset=False)
    try:
        assert env.frame.tolist() == [1, 1]
        saved = env.reset(frames=[3, 4]).obs["actor"].clone()
        a = torch.zeros(2, 1)
        first = env.step(a)
        assert not first.done.any()
        assert env.frame.tolist() == [4, 5]
        second = env.step(a)
        assert second.truncated.all() and not second.terminated.any()
        assert second.final_observation is not None
        assert env.sampler.failed.sum() == 0
        with pytest.raises(RuntimeError, match="reset"):
            env.step(a)
        env.reset(frames=[3, 4])
        torch.testing.assert_close(env.state.obs["actor"], saved, rtol=0, atol=0)
        replay = env.step(a)
        torch.testing.assert_close(replay.reward, first.reward, rtol=0, atol=0)
        lane = env.batch.read_state().base_pose[1].clone()
        env.reset(mask=[True, False], frames=[0])
        torch.testing.assert_close(
            env.batch.read_state().base_pose[1], lane, rtol=0, atol=0
        )
        with pytest.raises(ValueError, match="integer"):
            env.reset(frames=[0.5, 1.0])
    finally:
        env.close()


def test_motion_wrap_autoreset_and_observation_storage(tmp_path):
    env = TrackingEnv(tiny_config(tmp_path))
    try:
        env.reset(frames=[11, 10])
        obs = env.state.obs["actor"]
        saved = obs.clone()
        result = env.step(torch.zeros(2, 1))
        assert result.info["motion_wrap"].tolist() == [True, False]
        assert env.frame.tolist() == [0, 11]
        assert not result.done.any()
        torch.testing.assert_close(obs, saved, rtol=0, atol=0)
        env.max_episode_length = 2
        result = env.step(torch.zeros(2, 1))
        assert result.truncated.all()
        assert env.steps.eq(0).all()
        assert result.final_observation is not None
    finally:
        env.close()


def test_quaternion_matrix_roundtrip_and_gravity_termination():
    angles = torch.tensor(
        [[0.0, 0.0, 3.14159265], [3.14159265, 0.0, 0.0], [0.3, -0.8, 1.0]]
    )
    q = from_euler(angles)
    torch.testing.assert_close(
        matrix(quaternion(matrix(q))), matrix(q), atol=2e-6, rtol=2e-6
    )
    p = torch.zeros(3, 2, 3)
    r = q[:, None].expand(-1, 2, -1)
    ref = torch.zeros_like(r)
    ref[..., 0] = 1
    ends = mdp.terminations(
        (p, ref, p, p), (p, r, p, p), (p, ref), anchor=0, end_effectors=[1]
    )
    assert not ends["anchor_ori"][0]  # Yaw alone never triggers tilt termination.
    assert ends["anchor_ori"][1]


@pytest.mark.skipif(
    os.environ.get("GOBOT_RUN_MUJOCO_WARP_GPU_TEST") != "1",
    reason="opt-in real GPU test",
)
def test_tracking_gpu_clock_and_finite_steps(tmp_path):
    cfg = tiny_config(tmp_path, "mujoco-warp")
    env = TrackingEnv(cfg)
    try:
        env.reset(frames=[2, 3])
        state = env.batch.read_state()
        pose = state.base_pose.clone()
        pose[:, 3:] = torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5], device=env.device)
        env.batch.reset(
            [True, True],
            base_pose=pose,
            base_velocity=state.base_velocity.clone(),
            joint_position=state.joint_position.clone(),
            joint_velocity=state.joint_velocity.clone(),
            targets=state.joint_control.clone(),
        )
        before = {k: v.clone() for k, v in vars(env.batch.read_state()).items()}
        velocity = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            device=env.device,
        )
        env.batch.set_base_velocity([True, False], velocity)
        after = env.batch.read_state()
        torch.testing.assert_close(
            after.base_velocity[0], velocity[0], atol=1e-6, rtol=1e-6
        )
        for key, value in vars(after).items():
            torch.testing.assert_close(value[1], before[key][1], atol=0, rtol=0)
            if key != "base_velocity":
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        env.reset(frames=[2, 3])
        for _ in range(4):
            result = env.step(torch.zeros(2, 1, device=env.device))
            assert torch.isfinite(result.reward).all()
            assert torch.isfinite(result.obs["actor"]).all()
        assert env.frame.tolist() == [6, 7]
    finally:
        env.close()


def test_ppo_training_resume_and_evaluation(tmp_path):
    from dataclasses import asdict
    from gobot.rl.tracking.training import Config, run
    from gobot.rl.tracking.evaluate import evaluate

    cfg = tiny_config(tmp_path)
    agent = default_agent()
    agent["num_steps_per_env"] = 4
    agent["algorithm"]["num_learning_epochs"] = 1
    agent["algorithm"]["num_mini_batches"] = 2
    agent["actor"]["hidden_dims"] = [16, 16]
    agent["critic"]["hidden_dims"] = [16, 16]
    train = Config(**asdict(cfg), updates=2, output=tmp_path / "train", agent=agent)
    result = run(train)
    assert result["actor_max_weight_change"] > 0
    checkpoint = Path(result["checkpoint"])
    evaluation = evaluate(checkpoint, steps=2)
    assert evaluation["steps"] == 2
    resumed = run(
        replace(train, updates=1, output=tmp_path / "resume", resume=checkpoint)
    )
    assert resumed["last_iteration"] == 2
    with pytest.raises(ValueError, match="does not match"):
        run(
            replace(
                train,
                updates=1,
                output=tmp_path / "incompatible",
                resume=checkpoint,
                adaptive_alpha=0.01,
            )
        )


def test_seed_csv_units_and_named_joint_mapping(tmp_path):
    from scipy.spatial.transform import Rotation
    from gobot.rl.tracking.prepare import load_csv
    from gobot.rl.tracking.config import G1_JOINTS

    header = (
        ["Frame"]
        + [f"root_translate{x}" for x in "XYZ"]
        + [f"root_rotate{x}" for x in "XYZ"]
        + [n + "_dof" for n in G1_JOINTS]
    )
    data = np.zeros((3, len(header)))
    data[:, 0] = [1, 2, 3]
    data[:, 1:4] = [100.0, -20.0, 80.0]
    data[:, 4:7] = [0.0, 0.0, 90.0]
    data[:, 7:] = np.arange(29) * 2
    permutation = np.arange(len(header))[::-1]
    path = tmp_path / "seed.csv"
    np.savetxt(
        path,
        data[:, permutation],
        delimiter=",",
        header=",".join(header[i] for i in permutation),
        comments="",
    )
    result = load_csv(path, csv_format="seed")
    np.testing.assert_allclose(result[:, :3], np.tile([1.0, -0.2, 0.8], (3, 1)))
    np.testing.assert_allclose(
        Rotation.from_quat(result[:, 3:7]).apply(np.tile([1.0, 0, 0], (3, 1))),
        np.tile([0, 1.0, 0], (3, 1)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result[:, 7:], np.tile(np.deg2rad(np.arange(29) * 2), (3, 1))
    )
    data[1, 0] = 4
    np.savetxt(path, data, delimiter=",", header=",".join(header), comments="")
    with pytest.raises(ValueError, match="consecutive"):
        load_csv(path, csv_format="seed")


def test_checkpoint_state_owns_storage_and_restore_restarts_timers(tmp_path):
    cfg = tiny_config(tmp_path)
    env = TrackingEnv(cfg)
    try:
        saved = env.training_state_dict()
        env.sampler.failed.fill_(2)
        env.bias.fill_(0.1)
        env.push_time.fill_(-1.0)
        assert saved["failures"].eq(0).all() and saved["bias"].eq(0).all()
        env.load_training_state_dict(saved)
        assert ((env.push_time >= 1) & (env.push_time <= 3)).all()
    finally:
        env.close()
