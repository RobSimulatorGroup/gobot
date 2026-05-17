import json
from pathlib import Path

import numpy as np

import gobot
from rl_test_helpers import make_cartpole_target_task


def main():
    context = gobot.app.context()
    context.set_project_path("/tmp")
    context.clear_scene()
    gobot.scene.save_cartpole_scene("res://gobot_rl_manager_cartpole.jscn")

    task = make_cartpole_target_task(
        scene="res://gobot_rl_manager_cartpole.jscn",
        backend="null",
        max_episode_steps=2,
        target_cart_position=1.0,
        force_limit=20.0,
    )
    env = gobot.rl.ManagerBasedEnv(task)
    assert env.task_config is task
    assert env.action_spec.names == ("slider_effort",)
    assert env.observation_spec.names == (
        "cart_position",
        "cart_velocity",
        "pole_angle",
        "pole_angular_velocity",
        "target_position_error",
    )
    assert env.observation_group_spec("critic").shape == (6,)

    observation, info = env.reset(seed=7)
    assert observation.shape == (1, 5)
    assert np.all(np.isfinite(observation))
    assert info["commands"]["target_cart_position"].tolist() == [1.0]
    assert info["observations"]["critic"].shape == (1, 6)

    observation, reward, terminated, truncated, info = env.step([[2.0]])
    assert env.action_manager.last_action.tolist() == [[1.0]]
    assert observation.shape == (1, 5)
    assert reward.shape == (1,)
    assert terminated.tolist() == [False]
    assert truncated.tolist() == [False]
    assert "target_distance" in info["reward_breakdown"]
    assert "cart_position_limit" in info["termination_reasons"]
    assert "time_limit" in info["termination_reasons"]

    observation, reward, terminated, truncated, info = env.step([[0.0]])
    assert truncated.tolist() == [True]
    assert "terminal_observation" in info
    assert info["terminal_observation"].shape == (1, 5)
    assert env.episode_lengths.tolist() == [0]

    config_path = Path("/tmp/gobot_rl_cartpole_task.json")
    config_path.write_text(json.dumps(task.to_dict()), encoding="utf-8")
    loaded = gobot.rl.load_task_config(config_path)
    assert loaded.name == "cartpole_target"
    assert loaded.env_config().scene == "res://gobot_rl_manager_cartpole.jscn"

    env2 = gobot.rl.ManagerBasedEnv(loaded)
    observation2, info2 = env2.reset(seed=7)
    assert observation2.shape == (1, 5)
    assert info2["commands"]["target_cart_position"].tolist() == [1.0]

    random_task = make_cartpole_target_task(
        scene="res://gobot_rl_manager_cartpole.jscn",
        backend="null",
        randomize_target_position=True,
        target_cart_position_range=(-0.25, 0.25),
        max_episode_steps=4,
    )
    random_env = gobot.rl.ManagerBasedEnv(random_task)
    _, random_info_a = random_env.reset(seed=123)
    _, random_info_b = random_env.reset(seed=123)
    assert random_info_a["commands"]["target_cart_position"].tolist() == random_info_b["commands"]["target_cart_position"].tolist()
    assert -0.25 <= random_info_a["commands"]["target_cart_position"][0] <= 0.25


if __name__ == "__main__":
    main()
