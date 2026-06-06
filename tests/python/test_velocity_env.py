from pathlib import Path

import numpy as np

from gobot.rl.tasks import velocity


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_velocity_task_factories():
    go1_rough = velocity.velocity_task_cfg("go1_rough", project_path="/tmp/go1")
    assert go1_rough.name == "unitree_go1_rough_velocity"
    assert go1_rough.robot_name == "go1"
    assert go1_rough.observations.height_scan_sensor == "terrain_scan"

    go1_flat = velocity.velocity_task_cfg("go1_flat", project_path="/tmp/go1")
    assert go1_flat.name == "unitree_go1_flat_velocity"
    assert go1_flat.observations.height_scan_sensor is None
    assert not go1_flat.terrain_curriculum

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
    finally:
        env.close()


def main():
    test_velocity_task_factories()
    test_go1_velocity_env_reset_step_shapes()


if __name__ == "__main__":
    main()
