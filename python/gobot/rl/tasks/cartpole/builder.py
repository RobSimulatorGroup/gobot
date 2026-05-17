"""Build runtime CartPole TaskConfig objects from simple Python cfg classes."""

from __future__ import annotations

from ... import TaskConfig


def cartpole_env_cfg(cfg: type | object) -> TaskConfig:
    slider_joint = str(cfg.slider_joint)
    hinge_joint = str(cfg.hinge_joint)
    cart_position_limit = float(cfg.cart_position_limit)
    pole_angle_limit = float(cfg.pole_angle_limit)
    initial_angle = float(cfg.initial_angle)

    return TaskConfig(
        name=str(cfg.name),
        scene=str(cfg.scene),
        backend=str(cfg.backend),
        num_envs=int(cfg.num_envs),
        physics_dt=float(cfg.physics_dt),
        decimation=int(cfg.decimation),
        episode_length_s=float(cfg.max_episode_steps) * float(cfg.physics_dt) * int(cfg.decimation),
        robot=str(cfg.robot),
        project_path=str(cfg.project_path),
        auto_reset=bool(cfg.auto_reset),
        simulation={"batch_size": int(cfg.batch_size), "num_workers": int(cfg.num_workers)},
        actions=_action_cfg(cfg, slider_joint, hinge_joint),
        observations=_observation_cfg(slider_joint, hinge_joint, cart_position_limit),
        rewards=_reward_cfg(cfg, slider_joint, hinge_joint),
        terminations=_termination_cfg(slider_joint, hinge_joint, cart_position_limit, pole_angle_limit),
        events=_event_cfg(cfg, slider_joint, hinge_joint, initial_angle),
        commands=_command_cfg(cfg),
        metadata={
            "task_kind": "cartpole",
            "objective": str(cfg.name).replace("cartpole_", ""),
            "slider_joint": slider_joint,
            "hinge_joint": hinge_joint,
            "force_limit": float(cfg.force_limit),
            "cart_position_limit": cart_position_limit,
            "pole_angle_limit": pole_angle_limit,
        },
    )


def _action_cfg(cfg: type | object, slider_joint: str, hinge_joint: str) -> dict:
    return {
        "terms": [
            {
                "name": "slider_effort",
                "type": "joint_effort",
                "joint": slider_joint,
                "scale": float(cfg.force_limit),
                "clip": [-1.0, 1.0],
                "passive_joints": [hinge_joint],
            }
        ]
    }


def _observation_cfg(slider_joint: str, hinge_joint: str, cart_position_limit: float) -> dict:
    return {
        "groups": {
            "policy": [
                {"name": "cart_position", "type": "joint_position", "joint": slider_joint, "lower": -cart_position_limit, "upper": cart_position_limit, "unit": "m"},
                {"name": "cart_velocity", "type": "joint_velocity", "joint": slider_joint, "lower": -1.0e30, "upper": 1.0e30, "unit": "m/s"},
                {"name": "pole_angle", "type": "joint_position", "joint": hinge_joint, "wrap": True, "lower": -3.141592653589793, "upper": 3.141592653589793, "unit": "rad"},
                {"name": "pole_angular_velocity", "type": "joint_velocity", "joint": hinge_joint, "lower": -1.0e30, "upper": 1.0e30, "unit": "rad/s"},
                {"name": "target_position_error", "type": "command_error", "command": "target_cart_position", "joint": slider_joint, "lower": -2.0 * cart_position_limit, "upper": 2.0 * cart_position_limit, "unit": "m"},
            ],
            "critic": [
                {"type": "group", "group": "policy"},
                {"name": "previous_action", "type": "previous_action", "action": "slider_effort", "lower": -1.0, "upper": 1.0, "unit": "normalized"},
            ],
        },
        "default_group": "policy",
    }


def _reward_cfg(cfg: type | object, slider_joint: str, hinge_joint: str) -> dict:
    reward = cfg.rewards
    return {
        "terms": [
            {"name": "alive", "type": "constant", "weight": float(reward.alive), "scale_by_dt": False},
            {"name": "target_distance", "type": "squared_command_error", "weight": float(reward.target_distance), "command": "target_cart_position", "joint": slider_joint},
            {"name": "cart_velocity", "type": "joint_velocity_l2", "weight": float(reward.cart_velocity), "joint": slider_joint},
            {"name": "pole_angle", "type": "joint_position_l2", "weight": float(reward.pole_angle), "joint": hinge_joint, "wrap": True},
            {"name": "pole_angular_velocity", "type": "joint_velocity_l2", "weight": float(reward.pole_angular_velocity), "joint": hinge_joint},
            {"name": "overshoot", "type": "overshoot_l2", "weight": float(reward.overshoot), "joint": slider_joint, "command": "target_cart_position"},
            {"name": "action_l2", "type": "action_l2", "weight": float(reward.action_l2), "action": "slider_effort"},
            {"name": "target_progress", "type": "command_error_progress", "weight": float(reward.target_progress), "joint": slider_joint, "command": "target_cart_position", "clip": [-0.05, 0.05]},
            {
                "name": "settle_bonus",
                "type": "joint_command_settle_bonus",
                "weight": float(reward.settle_bonus),
                "joint": slider_joint,
                "condition_joint": hinge_joint,
                "command": "target_cart_position",
                "target_tolerance": 0.05,
                "velocity_tolerance": 0.2,
                "condition_wrap": True,
                "condition_position_tolerance": 0.10,
            },
            {
                "name": "fast_reach_bonus",
                "type": "first_joint_command_reach_bonus",
                "weight": float(reward.fast_reach_bonus),
                "joint": slider_joint,
                "condition_joint": hinge_joint,
                "command": "target_cart_position",
                "target_tolerance": 0.1,
                "condition_wrap": True,
                "condition_position_tolerance": 0.20,
            },
            {
                "name": "overspeed_penalty",
                "type": "joint_command_overspeed",
                "weight": float(reward.overspeed_penalty),
                "joint": slider_joint,
                "command": "target_cart_position",
                "target_tolerance": 0.1,
                "velocity_limit": 0.8,
            },
            {
                "name": "fast_crossing_penalty",
                "type": "joint_command_fast_crossing",
                "weight": float(reward.fast_crossing_penalty),
                "joint": slider_joint,
                "command": "target_cart_position",
                "velocity_tolerance": 0.2,
            },
            {"name": "failure_penalty", "type": "terminated", "weight": float(reward.failure_penalty)},
        ]
    }


def _termination_cfg(slider_joint: str, hinge_joint: str, cart_position_limit: float, pole_angle_limit: float) -> dict:
    return {
        "terms": [
            {"name": "cart_position_limit", "type": "joint_position_abs_gt", "joint": slider_joint, "limit": cart_position_limit},
            {"name": "pole_angle_limit", "type": "joint_position_abs_gt", "joint": hinge_joint, "limit": pole_angle_limit, "wrap": True},
        ]
    }


def _event_cfg(cfg: type | object, slider_joint: str, hinge_joint: str, initial_angle: float) -> dict:
    return {
        "terms": [
            {
                "name": "reset_joint_state",
                "type": "reset_joint_state",
                "joints": {
                    slider_joint: {"position": float(cfg.initial_cart_position), "velocity": 0.0},
                    hinge_joint: {
                        "position": initial_angle,
                        "position_range": [-initial_angle, initial_angle] if bool(cfg.randomize_initial_angle) else None,
                        "velocity": 0.0,
                    },
                },
            }
        ]
    }


def _command_cfg(cfg: type | object) -> dict:
    return {
        "terms": [
            {
                "name": "target_cart_position",
                "type": "uniform" if bool(cfg.randomize_target_position) else "constant",
                "value": float(cfg.target_cart_position),
                "range": [float(cfg.target_cart_position_range[0]), float(cfg.target_cart_position_range[1])],
                "unit": "m",
            }
        ]
    }


__all__ = ["cartpole_env_cfg"]
