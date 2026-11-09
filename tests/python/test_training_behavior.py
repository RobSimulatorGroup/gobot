"""Reward rates and PPO transition semantics shared by training backends.

No training run or CUDA device is needed. The physics check uses two native
CPU worlds; the adapter checks cover NumPy and device-native Torch storage.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gobot.rl import BatchEnvState, RslRlVecEnvWrapper


class _TransitionEnv:
    num_envs = 3
    num_actions = 2
    num_obs = 1
    num_privileged_obs = 1
    max_episode_length = 10

    def __init__(self, framework, *, legacy_timeout=False, final=True):
        self.accepts_device_actions = framework == "torch"
        array = torch.as_tensor if self.accepts_device_actions else np.asarray
        self.episode_length_buf = array([0, 0, 0])
        self.state = BatchEnvState(
            obs={"actor": array([[0.], [0.], [0.]]), "critic": array([[0.], [0.], [0.]])},
            reward=array([1., 1., 1.]),
            terminated=array([False, True, True]),
            truncated=array([True, False, True]),
            final_observation={"critic": array([[5.], [7.], [11.]])} if final else None,
        )
        if legacy_timeout:
            self.state.info["time_outs"] = self.state.truncated
        self.last_actions = None

    def step(self, actions):
        self.last_actions = actions
        return self.state

    def close(self):
        pass


@pytest.mark.parametrize("framework", ["numpy", "torch"])
@pytest.mark.parametrize("legacy_timeout", [False, True])
def test_custom_environment_timeout_and_final_observation(framework, legacy_timeout):
    env = _TransitionEnv(framework, legacy_timeout=legacy_timeout)
    wrapper = RslRlVecEnvWrapper(env)
    obs, reward, done, extras = wrapper.step(torch.zeros(3, 2))
    assert done.tolist() == [True, True, True]
    # Physical failure dominates a simultaneous time limit, without changing
    # the environment's masks or depending on optional info dictionary keys.
    assert extras["time_outs"].tolist() == [True, False, False]
    assert env.state.truncated.tolist() == [True, False, True]
    assert extras["_final_observation"].tolist() == [True, True, True]
    torch.testing.assert_close(extras["final_observation"]["critic"].float(), torch.tensor([[5.], [7.], [11.]]))
    assert torch.count_nonzero(obs["critic"]) == 0
    pointer = extras["time_outs"].data_ptr()
    assert wrapper.step(torch.zeros(3, 2))[3]["time_outs"].data_ptr() == pointer

    # The timeout uses V(s_final), not the reset or previous rollout state.
    torch.testing.assert_close(_stored_transition_rewards(obs, reward, done, extras),
                               torch.tensor([5.95, .5, .5]))


def _stored_transition_rewards(obs, reward, done, extras):
    """Exercise the real PPO collector with an inspectable critic and storage."""
    from gobot.rl.locomotion.transition_ppo import TransitionPPO
    from rsl_rl.storage import RolloutStorage

    class Model:
        def __call__(self, observation):
            return observation["critic"].float()

        def update_normalization(self, observation):
            pass

        def reset(self, dones):
            pass

    alg = TransitionPPO.__new__(TransitionPPO)
    alg.actor = alg.critic = Model()
    alg.device = "cpu"
    alg.gamma = .99
    alg.fall_penalty = .5
    alg.rnd = None
    alg.transition = RolloutStorage.Transition()
    alg.transition.values = torch.full((3, 1), 100.)
    recorded = []
    alg.storage = SimpleNamespace(add_transition=lambda transition: recorded.append(transition.rewards.clone()))
    alg.process_env_step(obs, reward, done, extras)
    return recorded[0]


@pytest.mark.parametrize("legacy_timeout", [False, True])
@pytest.mark.parametrize("framework", ["numpy", "torch"])
def test_finite_horizon_explicitly_disables_bootstrap(legacy_timeout, framework):
    env = _TransitionEnv(framework, legacy_timeout=legacy_timeout, final=False)
    wrapper = RslRlVecEnvWrapper(env, is_finite_horizon=True)
    obs, reward, done, extras = wrapper.step(torch.zeros(3, 2))
    assert done.all()
    assert "time_outs" not in extras
    assert "final_observation" not in extras
    assert extras["terminated"].tolist() == [False, True, True]
    # A time-limit episode ending earns no value bootstrap and no fall penalty;
    # an actual fall still receives the penalty, even at the same time limit.
    torch.testing.assert_close(_stored_transition_rewards(obs, reward, done, extras),
                               torch.tensor([1., .5, .5]))


@pytest.mark.parametrize("framework", ["numpy", "torch"])
def test_optional_action_clipping_preserves_sampled_rollout_action(framework):
    env = _TransitionEnv(framework)
    action = torch.tensor([[-2., 3.], [-.2, .3], [1., -1.]])
    if framework == "numpy":
        action = action.numpy()
    before = np.asarray(action).copy()
    wrapper = RslRlVecEnvWrapper(env, clip_actions=.5)
    wrapper.step(action)
    np.testing.assert_array_equal(action, before)
    np.testing.assert_allclose(env.last_actions, np.clip(before, -.5, .5))
    for bad in (0., -1., float("nan"), float("inf")):
        with pytest.raises(ValueError, match="clip_actions"):
            RslRlVecEnvWrapper(env, clip_actions=bad)


def test_default_wrapper_does_not_add_a_second_action_limit():
    env = _TransitionEnv("torch")
    action = torch.full((3, 2), 5.)
    RslRlVecEnvWrapper(env).step(action)
    assert env.last_actions is action


def test_cpu_reward_rate_and_timeout_survive_autoreset():
    from examples.go1.train.go1_velocity_cfg import go1_velocity_cfg
    from gobot.rl.locomotion import make_velocity_env

    project = Path(__file__).resolve().parents[2] / "examples/go1"
    cfg = go1_velocity_cfg(project_path=project)
    cfg.observations.actor_noise = cfg.domain_randomization.enabled = cfg.push_enabled = False
    cfg.randomize_reset_yaw = False
    cfg.spawn_jitter = 0.
    cfg.capture_transitions = True
    env = make_velocity_env(cfg, backend="mujoco-cpu", num_envs=2, seed=23,
                            max_episode_length=2, collect_step_extras=True)
    try:
        wrapper = RslRlVecEnvWrapper(env)
        for step in range(2):
            _, reward, done, extras = wrapper.step(torch.full((2, env.num_actions), .25))
            rates = np.sum(np.stack(tuple(env.state.info["reward_terms"].values())), axis=0)
            # Public reward terms are weighted rates; the integrated step reward
            # uses env_dt = physics_dt * decimation exactly once, including reset.
            np.testing.assert_allclose(reward.numpy(), rates * env.physics_dt * env.decimation, rtol=1e-6)
            assert extras["time_outs"].tolist() == [step == 1, step == 1]
            if step == 1:
                assert done.all()
                assert extras["_final_observation"].all()
                assert not torch.equal(extras["final_observation"]["critic"], wrapper.get_observations()["critic"])
    finally:
        env.close()


@pytest.mark.parametrize("name,value,target", [
    ("max_iterations", 2, "experiment.iterations"),
    ("save_interval", 2, "experiment.save_interval"),
    ("clip_actions", .1, "env.action_clip"),
    ("logger", "wandb", "custom TrainingSpec"),
])
def test_unused_agent_settings_rejected_before_simulation(tmp_path, monkeypatch, name, value, target):
    from gobot.rl.locomotion.experiment import Experiment
    from gobot.rl.locomotion import training

    experiment = Experiment(robot="go1", iterations=1, num_envs=2)
    agent = training.agent_config(experiment)
    agent[name] = value
    def forbidden(*args, **kwargs):
        pytest.fail("invalid runner setting reached simulation")
    monkeypatch.setattr(training, "make_velocity_env", forbidden)
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match=target):
        training.run(experiment, output, backend="mujoco-cpu", no_eval=True, agent_cfg=agent)
    assert not output.exists()


@pytest.mark.parametrize("key,value", [
    ("learning_rate", float("nan")), ("learning_rate", 0.),
    ("num_learning_epochs", True), ("num_mini_batches", 0),
    ("gamma", 1.1), ("lam", -.1), ("max_grad_norm", float("inf")),
])
def test_invalid_algorithm_values_fail_before_simulation(key, value):
    from gobot.rl.locomotion.experiment import Experiment
    from gobot.rl.locomotion.training import agent_config, validate_agent_config

    experiment = Experiment(robot="go1")
    agent = agent_config(experiment)
    agent["algorithm"][key] = value
    with pytest.raises(ValueError, match=key):
        validate_agent_config(experiment, agent)


def test_configurable_ppo_and_amp_details_remain_supported():
    from gobot.rl.locomotion.experiment import Experiment
    from gobot.rl.locomotion.training import agent_config, validate_agent_config

    for recipe in ("baseline", "amp"):
        experiment = Experiment(robot="g1", recipe=recipe)
        agent = agent_config(experiment)
        assert not set(agent) & {"max_iterations", "save_interval", "clip_actions"}
        agent["algorithm"]["learning_rate"] = 2.e-4
        agent["algorithm"]["gamma"] = .995
        agent["algorithm"]["desired_kl"] = .012
        agent["actor"]["distribution_cfg"]["init_std"] = .7
        agent["num_steps_per_env"] = 32
        validate_agent_config(experiment, agent)
