from pathlib import Path

import numpy as np

from gobot.rl.tasks import velocity
from gobot.rl.tasks.velocity.env import GobotVelocityEnv, VelocityRuntimeState


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_velocity_task_factories():
    go1_rough = velocity.velocity_task_cfg("go1_rough", project_path="/tmp/go1")
    assert go1_rough.name == "unitree_go1_rough_velocity"
    assert go1_rough.robot_name == "go1"
    assert go1_rough.observations.height_scan_sensor == "terrain_scan"
    assert go1_rough.terrain_normal_upright.enabled
    assert go1_rough.illegal_contact.enabled
    assert go1_rough.domain_randomization.enabled
    assert go1_rough.push_enabled

    go1_flat = velocity.velocity_task_cfg("go1_flat", project_path="/tmp/go1")
    assert go1_flat.name == "unitree_go1_flat_velocity"
    assert go1_flat.observations.height_scan_sensor is None
    assert not go1_flat.terrain_curriculum
    assert not go1_flat.terrain_normal_upright.enabled
    assert not go1_flat.illegal_contact.enabled

    g1 = velocity.velocity_task_cfg("g1_rough", project_path="/tmp/g1")
    assert g1.name == "unitree_g1_rough_velocity"
    assert g1.robot_family == "g1"

    try:
        velocity.GobotVelocityEnv(g1, num_envs=1, device="cpu")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("G1 velocity task should fail clearly until a Gobot G1 scene asset exists")


def test_go1_velocity_env_reset_step_shapes():
    torch = __import__("torch")

    cfg = velocity.unitree_go1_rough_velocity_cfg(project_path=REPO_ROOT / "examples/go1")
    cfg.observations.actor_noise = False
    env = velocity.GobotVelocityEnv(cfg, num_envs=1, device="cpu", seed=123, max_episode_length=4)
    try:
        observations = env.get_observations()
        assert observations["actor"].shape == (1, env.num_obs)
        assert observations["critic"].shape == (1, env.num_privileged_obs)
        assert env.num_obs == 63
        assert env.num_privileged_obs == 87

        observations, reward, done, extras = env.step(torch.zeros((1, env.num_actions)))
        assert observations["actor"].shape == (1, env.num_obs)
        assert observations["critic"].shape == (1, env.num_privileged_obs)
        assert reward.shape == (1,)
        assert done.shape == (1,)
        assert np.isfinite(reward.cpu().numpy()).all()
        assert "/velocity/terrain_level" in extras["log"]
        assert "/velocity/velocity_error" in extras["log"]
        assert "/velocity/terrain_normal_error" in extras["log"]
        assert "/velocity/illegal_contact_count" in extras["log"]
        assert "/velocity/terrain_curriculum_limit" in extras["log"]
        assert "/velocity/push_count" in extras["log"]
    finally:
        env.close()


def test_velocity_terrain_normal_plane_fit():
    env = object.__new__(GobotVelocityEnv)
    env.cfg_obj = velocity.unitree_go1_rough_velocity_cfg(project_path="/tmp/go1")
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


def test_velocity_contact_summary_classifies_illegal_contacts():
    env = object.__new__(GobotVelocityEnv)
    env.cfg_obj = velocity.unitree_go1_rough_velocity_cfg(project_path="/tmp/go1")
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


def main():
    test_velocity_task_factories()
    test_velocity_terrain_normal_plane_fit()
    test_velocity_contact_summary_classifies_illegal_contacts()
    test_go1_velocity_env_reset_step_shapes()


if __name__ == "__main__":
    main()
