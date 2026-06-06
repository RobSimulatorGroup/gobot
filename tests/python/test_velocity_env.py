from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "python", REPO_ROOT / "build/python"):
    path_string = str(path)
    while path_string in sys.path:
        sys.path.remove(path_string)
    sys.path.insert(0, path_string)

from examples.go1.scripts import go1 as go1_playback
from examples.go1.train import go1_velocity_cfg as go1_cfg
from examples.go1.train.go1_velocity_env import Go1VelocityEnv, VelocityRuntimeState
from gobot.rl.locomotion import (
    UniformVelocityCommand,
    velocity_actor_observation_schema,
    velocity_critic_observation_schema,
)


def test_go1_velocity_cfg_dimensions():
    cfg = go1_cfg.go1_velocity_cfg("go1_rough", project_path="/tmp/go1")
    assert cfg.name == "gobot_go1_velocity"
    assert cfg.robot_name == "go1"
    assert cfg.observations.height_scan_sensor == "terrain_scan"
    assert cfg.terrain_normal_upright.enabled
    assert cfg.illegal_contact.enabled
    assert cfg.domain_randomization.enabled
    assert cfg.push_enabled

    actor_schema = velocity_actor_observation_schema(len(cfg.joint_names), 15)
    critic_schema = velocity_critic_observation_schema(len(cfg.joint_names), 15, len(cfg.foot_names))
    assert actor_schema.dim == 63
    assert critic_schema.dim == 87
    assert len(cfg.joint_names) == 12

    flat = go1_cfg.go1_velocity_cfg("go1_flat", project_path="/tmp/go1")
    assert flat.name == "gobot_go1_flat_velocity"
    assert flat.observations.height_scan_sensor is None
    assert not flat.terrain_curriculum
    assert not flat.terrain_normal_upright.enabled
    assert not flat.illegal_contact.enabled


def test_go1_playback_schema_matches_training_schema():
    cfg = go1_cfg.go1_rough_velocity_cfg(project_path="/tmp/go1")
    training_schema = velocity_actor_observation_schema(len(cfg.joint_names), len(go1_playback.HEIGHT_SCAN_POINTS))
    assert go1_playback.ACTOR_OBS_SCHEMA.names == training_schema.names
    assert go1_playback.ACTOR_OBS_SCHEMA.dim == 63


def test_go1_velocity_env_reset_step_shapes():
    torch = __import__("torch")

    cfg = go1_cfg.go1_rough_velocity_cfg(project_path=REPO_ROOT / "examples/go1")
    cfg.observations.actor_noise = False
    env = Go1VelocityEnv(cfg, num_envs=1, device="cpu", seed=123, max_episode_length=4)
    try:
        observations = env.get_observations()
        assert observations["actor"].shape == (1, env.num_obs)
        assert observations["critic"].shape == (1, env.num_privileged_obs)
        assert env.num_obs == 63
        assert env.num_privileged_obs == 87
        assert env.cfg["task"] == "gobot_go1_velocity"
        assert env.cfg["obs_schema_version"] == env.actor_obs_schema.version
        assert env.cfg["obs_names"] == env.actor_obs_schema.names

        observations, reward, done, extras = env.step(torch.zeros((1, env.num_actions)))
        assert observations["actor"].shape == (1, env.num_obs)
        assert observations["critic"].shape == (1, env.num_privileged_obs)
        assert reward.shape == (1,)
        assert done.shape == (1,)
        assert np.isfinite(observations["actor"].cpu().numpy()).all()
        assert np.isfinite(observations["critic"].cpu().numpy()).all()
        assert np.isfinite(reward.cpu().numpy()).all()
        assert "/velocity/terrain_level" in extras["log"]
        assert "/velocity/velocity_error" in extras["log"]
        assert "/velocity/terrain_normal_error" in extras["log"]
        assert "/velocity/illegal_contact_count" in extras["log"]
        assert "/velocity/terrain_curriculum_limit" in extras["log"]
        assert "/velocity/push_count" in extras["log"]
    finally:
        env.close()


def test_go1_velocity_terrain_normal_plane_fit():
    env = object.__new__(Go1VelocityEnv)
    env.cfg_obj = go1_cfg.go1_rough_velocity_cfg(project_path="/tmp/go1")
    env._height_scan_dim = 4
    state = VelocityRuntimeState(
        robot={},
        base={},
        joints={},
        links={},
        sensors={
            "terrain_scan": {
                "hits": [
                    {"hit": True, "point": [0.0, 0.0, 0.0]},
                    {"hit": True, "point": [1.0, 0.0, 0.1]},
                    {"hit": True, "point": [0.0, 1.0, 0.0]},
                    {"hit": True, "point": [1.0, 1.0, 0.1]},
                ]
            }
        },
        contacts=[],
    )

    normal = env._terrain_normal_from_scan(state)
    expected = np.array([-0.1, 0.0, 1.0], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    assert np.allclose(normal, expected, atol=1.0e-5)


def test_go1_velocity_contact_summary_classifies_illegal_contacts():
    env = object.__new__(Go1VelocityEnv)
    env.cfg_obj = go1_cfg.go1_rough_velocity_cfg(project_path="/tmp/go1")
    env._foot_count = 4
    state = VelocityRuntimeState(
        robot={},
        base={},
        joints={},
        links={
            "FR_calf": {"global_transform": {"position": [10.0, 0.0, 0.0]}},
            "FL_calf": {"global_transform": {"position": [11.0, 0.0, 0.0]}},
            "RR_calf": {"global_transform": {"position": [12.0, 0.0, 0.0]}},
            "RL_calf": {"global_transform": {"position": [13.0, 0.0, 0.0]}},
        },
        sensors={},
        contacts=[
            {
                "robot_name": "go1",
                "link_name": "FR_thigh",
                "other_robot_name": "",
                "other_link_name": "",
                "position": [0.0, 0.0, 0.0],
                "normal_force": 10.0,
                "force": [0.0, 0.0, 10.0],
            },
            {
                "robot_name": "go1",
                "link_name": "trunk",
                "other_robot_name": "",
                "other_link_name": "",
                "position": [0.0, 0.0, 0.2],
                "normal_force": 3.0,
                "force": [0.0, 0.0, 3.0],
            },
            {
                "robot_name": "go1",
                "link_name": "FR_calf",
                "other_robot_name": "go1",
                "other_link_name": "FL_calf",
                "position": [0.0, 0.0, 0.4],
                "normal_force": 2.0,
                "force": [0.0, 0.0, 2.0],
            },
        ],
    )

    summary = env._contact_summary(state)
    assert summary["illegal"] == 1.0
    assert summary["trunk_head"] == 1.0
    assert summary["self_collision"] == 1.0


def test_go1_command_sampling_is_seed_reproducible():
    class DummyEnv:
        def __init__(self, seed):
            self.num_envs = 4
            self.step_dt = 0.02
            self._rng = np.random.default_rng(seed)

    cfg = go1_cfg.go1_rough_velocity_cfg(project_path="/tmp/go1").command
    first = UniformVelocityCommand(cfg, DummyEnv(123))
    second = UniformVelocityCommand(cfg, DummyEnv(123))
    env_ids = np.arange(4, dtype=np.int64)
    first.reset(env_ids)
    second.reset(env_ids)

    assert np.allclose(first.command_b, second.command_b)
    assert np.allclose(first.time_left, second.time_left)
    assert np.array_equal(first.is_heading_env, second.is_heading_env)


def test_go1_spawn_curriculum_is_seed_reproducible():
    first = _curriculum_env(seed=7)
    second = _curriculum_env(seed=7)
    assert [first._sample_spawn_index(0) for _ in range(8)] == [second._sample_spawn_index(0) for _ in range(8)]

    state = VelocityRuntimeState(
        robot={},
        base={"global_transform": {"position": [1.0, 0.0, 0.5]}},
        joints={},
        links={},
        sensors={},
        contacts=[],
    )
    first.episode_length_buf = np.array([80])
    second.episode_length_buf = np.array([80])
    first._update_terrain_curriculum_limit(0, state, reset_reason=2)
    second._update_terrain_curriculum_limit(0, state, reset_reason=2)
    assert np.allclose(first._terrain_curriculum_limits, second._terrain_curriculum_limits)


def _curriculum_env(seed: int):
    env = object.__new__(Go1VelocityEnv)
    env.cfg_obj = go1_cfg.go1_rough_velocity_cfg(project_path="/tmp/go1")
    env.cfg_obj.terrain_curriculum = True
    env._rng = np.random.default_rng(seed)
    env._curriculum_progress = 0.5
    env._spawn_origins = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    env._center_spawn_index = 0
    env._spawn_levels = np.asarray([0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    env._spawn_order = np.arange(4, dtype=np.int64)
    env._terrain_curriculum_limits = np.zeros(1, dtype=np.float32)
    env._episode_start_xy = np.zeros((1, 2), dtype=np.float32)
    env.max_episode_length = 100
    env.step_dt = 0.02
    env.command_manager = type("Command", (), {"command_b": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)})()
    return env


def main():
    test_go1_velocity_cfg_dimensions()
    test_go1_playback_schema_matches_training_schema()
    test_go1_velocity_terrain_normal_plane_fit()
    test_go1_velocity_contact_summary_classifies_illegal_contacts()
    test_go1_command_sampling_is_seed_reproducible()
    test_go1_spawn_curriculum_is_seed_reproducible()
    test_go1_velocity_env_reset_step_shapes()


if __name__ == "__main__":
    main()
