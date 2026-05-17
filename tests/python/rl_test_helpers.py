import math

import gobot


def make_cartpole_target_task(
    *,
    scene,
    backend="null",
    max_episode_steps=2,
    target_cart_position=1.0,
    target_cart_position_range=(-1.0, 1.0),
    randomize_target_position=False,
    force_limit=20.0,
    num_envs=1,
):
    cart_position_limit = 2.4
    pole_angle_limit = 0.7
    physics_dt = 1.0 / 240.0
    return gobot.rl.TaskConfig(
        name="cartpole_target",
        scene=scene,
        backend=backend,
        num_envs=num_envs,
        physics_dt=physics_dt,
        decimation=1,
        episode_length_s=float(max_episode_steps) * physics_dt,
        robot="cartpole",
        project_path="/tmp",
        actions={
            "terms": [
                {
                    "name": "slider_effort",
                    "type": "joint_effort",
                    "joint": "slider",
                    "scale": float(force_limit),
                    "clip": [-1.0, 1.0],
                    "passive_joints": ["hinge"],
                }
            ]
        },
        observations={
            "groups": {
                "policy": [
                    {
                        "name": "cart_position",
                        "type": "joint_position",
                        "joint": "slider",
                        "lower": -cart_position_limit,
                        "upper": cart_position_limit,
                        "unit": "m",
                    },
                    {
                        "name": "cart_velocity",
                        "type": "joint_velocity",
                        "joint": "slider",
                        "lower": -1.0e30,
                        "upper": 1.0e30,
                        "unit": "m/s",
                    },
                    {
                        "name": "pole_angle",
                        "type": "joint_position",
                        "joint": "hinge",
                        "wrap": True,
                        "lower": -math.pi,
                        "upper": math.pi,
                        "unit": "rad",
                    },
                    {
                        "name": "pole_angular_velocity",
                        "type": "joint_velocity",
                        "joint": "hinge",
                        "lower": -1.0e30,
                        "upper": 1.0e30,
                        "unit": "rad/s",
                    },
                    {
                        "name": "target_position_error",
                        "type": "command_error",
                        "command": "target_cart_position",
                        "joint": "slider",
                        "lower": -2.0 * cart_position_limit,
                        "upper": 2.0 * cart_position_limit,
                        "unit": "m",
                    },
                ],
                "critic": [
                    {"type": "group", "group": "policy"},
                    {
                        "name": "previous_action",
                        "type": "previous_action",
                        "action": "slider_effort",
                        "lower": -1.0,
                        "upper": 1.0,
                        "unit": "normalized",
                    },
                ],
            },
            "default_group": "policy",
        },
        rewards={
            "terms": [
                {"name": "alive", "type": "constant", "weight": 1.0, "scale_by_dt": False},
                {
                    "name": "target_distance",
                    "type": "squared_command_error",
                    "weight": -1.0,
                    "command": "target_cart_position",
                    "joint": "slider",
                },
                {"name": "action_l2", "type": "action_l2", "weight": -0.002, "action": "slider_effort"},
            ]
        },
        terminations={
            "terms": [
                {
                    "name": "cart_position_limit",
                    "type": "joint_position_abs_gt",
                    "joint": "slider",
                    "limit": cart_position_limit,
                },
                {
                    "name": "pole_angle_limit",
                    "type": "joint_position_abs_gt",
                    "joint": "hinge",
                    "limit": pole_angle_limit,
                    "wrap": True,
                },
            ]
        },
        events={
            "terms": [
                {
                    "name": "reset_joint_state",
                    "type": "reset_joint_state",
                    "joints": {
                        "slider": {"position": 0.0, "velocity": 0.0},
                        "hinge": {"position": 0.0, "velocity": 0.0},
                    },
                }
            ]
        },
        commands={
            "terms": [
                {
                    "name": "target_cart_position",
                    "type": "uniform" if randomize_target_position else "constant",
                    "value": float(target_cart_position),
                    "range": [float(target_cart_position_range[0]), float(target_cart_position_range[1])],
                    "unit": "m",
                }
            ]
        },
        metadata={"task_kind": "cartpole"},
    )
