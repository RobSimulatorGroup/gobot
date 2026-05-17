"""CartPole PPO configs in the simple RSL-RL class style."""

from __future__ import annotations


class CartPoleBalancePPOCfg:
    seed = 42

    class policy:
        init_noise_std = 0.2
        actor_hidden_dims = [128, 128, 128]
        critic_hidden_dims = [128, 128, 128]
        activation = "elu"

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0
        num_learning_epochs = 4
        num_mini_batches = 4
        learning_rate = 3.0e-4
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 64
        max_iterations = 500
        save_interval = 100
        experiment_name = "cartpole_balance"
        run_name = ""
        logger = "tensorboard"
        clip_actions = 1.0
        upload_model = False


class CartPoleTargetPPOCfg:
    seed = 42

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [128, 128, 128]
        critic_hidden_dims = [128, 128, 128]
        activation = "elu"

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.005
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 300
        save_interval = 50
        experiment_name = "cartpole_target"
        run_name = ""
        logger = "tensorboard"
        clip_actions = 1.0
        upload_model = False


__all__ = ["CartPoleBalancePPOCfg", "CartPoleTargetPPOCfg"]
