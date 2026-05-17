"""CartPole environment configs in the simple Python class style."""

from __future__ import annotations


class CartPoleBalanceEnvCfg:
    name = "cartpole_balance"
    scene = "res://cartpole.jscn"
    project_path = "/home/wqq/gobot/examples/cartpole"
    backend = "mujoco"
    robot = "cartpole"

    num_envs = 64
    batch_size = 64
    num_workers = 0

    physics_dt = 1.0 / 240.0
    decimation = 1
    max_episode_steps = 500
    auto_reset = True

    slider_joint = "slider"
    hinge_joint = "hinge"
    force_limit = 10.0
    cart_position_limit = 2.4
    pole_angle_limit = 0.7

    target_cart_position = 0.0
    randomize_target_position = False
    target_cart_position_range = (-1.0, 1.0)

    initial_cart_position = 0.0
    initial_angle = 0.05
    randomize_initial_angle = True

    class rewards:
        alive = 1.0
        target_distance = -1.0
        cart_velocity = -0.02
        pole_angle = -10.0
        pole_angular_velocity = -0.05
        action_l2 = -0.002
        failure_penalty = -100.0
        target_progress = 0.0
        settle_bonus = 0.0
        fast_reach_bonus = 0.0
        overshoot = 0.0
        overspeed_penalty = 0.0
        fast_crossing_penalty = 0.0


class CartPoleTargetEnvCfg(CartPoleBalanceEnvCfg):
    name = "cartpole_target"
    force_limit = 20.0
    target_cart_position = 1.0
    randomize_target_position = True

    class rewards:
        alive = 8.0
        target_distance = -8.0
        cart_velocity = -0.5
        pole_angle = -300.0
        pole_angular_velocity = -8.0
        action_l2 = -0.02
        failure_penalty = -100.0
        target_progress = 4.0
        settle_bonus = 8.0
        fast_reach_bonus = 20.0
        overshoot = -20.0
        overspeed_penalty = -10.0
        fast_crossing_penalty = -10.0


class CartPoleBalancePlayEnvCfg(CartPoleBalanceEnvCfg):
    num_envs = 1
    batch_size = 1
    num_workers = 0


class CartPoleTargetPlayEnvCfg(CartPoleTargetEnvCfg):
    num_envs = 1
    batch_size = 1
    num_workers = 0


__all__ = [
    "CartPoleBalanceEnvCfg",
    "CartPoleBalancePlayEnvCfg",
    "CartPoleTargetEnvCfg",
    "CartPoleTargetPlayEnvCfg",
]
