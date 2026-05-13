import numpy as np

import gobot


def main():
    context = gobot.app.context()
    context.set_project_path("/tmp")
    context.clear_scene()
    gobot.scene.save_cartpole_scene("res://gobot_rl_vector_cartpole.jscn")

    task = gobot.rl.make_cartpole_target_task(
        scene="res://gobot_rl_vector_cartpole.jscn",
        backend="null",
        max_episode_steps=2,
        force_limit=20.0,
    )
    task.num_envs = 4
    task.simulation = {"batch_size": 2, "num_workers": 2}

    env = gobot.rl.VectorEnv(task)
    assert env.num_envs == 4
    assert env.batch_size == 2
    assert env.num_workers == 2
    assert env.action_spec.names == ("slider_effort",)

    obs, info = env.reset(seed=3)
    assert obs.shape == (4, env.observation_spec.size)
    assert info["env_id"].tolist() == [0, 1, 2, 3]

    actions = np.zeros((4, env.action_spec.size), dtype=np.float64)
    obs, reward, terminated, truncated, info = env.step(actions)
    assert obs.shape == (4, env.observation_spec.size)
    assert reward.shape == (4,)
    assert terminated.tolist() == [False, False, False, False]
    assert truncated.tolist() == [False, False, False, False]
    assert info["episode_length"].tolist() == [1, 1, 1, 1]

    obs, reward, terminated, truncated, info = env.step(actions)
    assert truncated.tolist() == [True, True, True, True]
    assert "terminal_observation" in info
    assert info["terminal_observation"].shape == (4, env.observation_spec.size)
    assert env.episode_lengths.tolist() == [0, 0, 0, 0]

    subset_actions = np.zeros((2, env.action_spec.size), dtype=np.float64)
    subset_obs, subset_reward, _, _, subset_info = env.step(subset_actions, env_ids=[1, 3])
    assert subset_obs.shape == (2, env.observation_spec.size)
    assert subset_reward.shape == (2,)
    assert subset_info["env_id"].tolist() == [1, 3]

    env.async_reset()
    obs_a, reward_a, term_a, trunc_a, info_a = env.recv()
    obs_b, reward_b, term_b, trunc_b, info_b = env.recv()
    assert obs_a.shape == (2, env.observation_spec.size)
    assert obs_b.shape == (2, env.observation_spec.size)
    assert info_a["env_id"].tolist() == [0, 1]
    assert info_b["env_id"].tolist() == [2, 3]
    assert reward_a.tolist() == [0.0, 0.0]
    assert term_b.tolist() == [False, False]
    assert trunc_a.tolist() == [False, False]

    env.send(actions)
    _, reward_a, _, _, info_a = env.recv()
    _, reward_b, _, _, info_b = env.recv()
    assert reward_a.shape == (2,)
    assert reward_b.shape == (2,)
    assert info_a["env_id"].tolist() == [0, 1]
    assert info_b["env_id"].tolist() == [2, 3]


if __name__ == "__main__":
    main()
