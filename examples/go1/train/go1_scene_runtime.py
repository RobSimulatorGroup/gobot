"""Shared Go1 scene-to-runtime compilation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import gobot
from gobot.rl.locomotion.math import find_node_by_name

from ..go1_profile import (
    GO1_ARMATURE,
    GO1_EFFORT_LIMIT,
    GO1_VELOCITY_LIMIT,
)
from .go1_velocity_cfg import Go1VelocityCfg


def find_robot_node(context: gobot.AppContext, cfg: Go1VelocityCfg):
    root = context.root
    robot = root.find(cfg.robot_name) if root is not None else None
    if robot is None:
        robot = find_node_by_name(root, cfg.robot_name)
    if robot is None:
        raise RuntimeError(f"Gobot scene has no robot named {cfg.robot_name!r}")
    return robot


def find_terrain_node(context: gobot.AppContext):
    root = context.root
    terrain_world = root.find("terrain_world") if root is not None else None
    if terrain_world is None:
        raise RuntimeError("Gobot scene has no terrain_world node")
    terrain = terrain_world.find("terrain")
    if terrain is None:
        raise RuntimeError("Gobot scene terrain_world has no authored terrain node")
    if terrain.generation_error:
        raise RuntimeError(f"Gobot scene terrain generation failed: {terrain.generation_error}")
    if not terrain.generator_config:
        raise RuntimeError("Gobot scene terrain has no generator_config resource")
    return terrain


def terrain_generator_config(terrain) -> dict[str, Any]:
    config = terrain.generator_config
    properties = config.get("properties", config)
    if not isinstance(properties, Mapping):
        raise RuntimeError("Gobot scene terrain generator metadata is invalid")
    return dict(properties)


def terrain_spawn_origins(terrain) -> np.ndarray:
    origins = np.asarray(getattr(terrain, "spawn_origins", []), dtype=np.float64)
    if origins.ndim != 2 or origins.shape[1] != 3 or origins.shape[0] == 0:
        return np.zeros((1, 3), dtype=np.float64)
    return origins


def joint_gain_array(value: float | Sequence[float], joint_count: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return np.full((joint_count,), float(array), dtype=np.float32)
    array = array.reshape(-1)
    if array.shape != (joint_count,):
        raise ValueError(f"joint gain array must have shape ({joint_count},), got {array.shape}")
    return array.astype(np.float32, copy=True)


def configure_robot_drives(robot, cfg: Go1VelocityCfg) -> None:
    joint_kp = joint_gain_array(cfg.kp, len(cfg.joint_names))
    joint_kd = joint_gain_array(cfg.kd, len(cfg.joint_names))
    for joint_index, joint_name in enumerate(cfg.joint_names):
        joint = robot.find(joint_name) or find_node_by_name(robot, joint_name)
        if joint is None:
            raise RuntimeError(f"Gobot Go1 scene has no configured joint {joint_name!r}")
        joint.drive_mode = gobot.JointDriveMode.Position
        joint.drive_stiffness = float(joint_kp[joint_index])
        joint.drive_damping = float(joint_kd[joint_index])
        effort_limit = float(GO1_EFFORT_LIMIT[joint_index])
        joint.armature = float(GO1_ARMATURE[joint_index])
        joint.effort_limit = effort_limit
        joint.velocity_limit = float(GO1_VELOCITY_LIMIT[joint_index])
        joint.force_lower_limit = -effort_limit
        joint.force_upper_limit = effort_limit
        joint.control_lower_limit = 0.0
        joint.control_upper_limit = 0.0
        joint.damping = 0.0
        joint.friction_loss = 0.0


def prepare_go1_scene(
    cfg: Go1VelocityCfg,
    *,
    context: gobot.AppContext | None = None,
) -> tuple[gobot.AppContext, Any, Any]:
    runtime_context = context if context is not None else gobot.app.context()
    runtime_context.set_project_path(str(Path(cfg.project_path).resolve()))
    runtime_context.load_scene(cfg.scene_path)
    robot = find_robot_node(runtime_context, cfg)
    terrain = find_terrain_node(runtime_context)

    runtime_context.fixed_time_step = float(cfg.physics_dt)
    kp = joint_gain_array(cfg.kp, len(cfg.joint_names))
    kd = joint_gain_array(cfg.kd, len(cfg.joint_names))
    runtime_context.set_default_joint_gains(
        {
            "position_stiffness": float(kp[0]),
            "velocity_damping": float(kd[0]),
            "integral_gain": 0.0,
            "integral_limit": 0.0,
        }
    )
    if cfg.mujoco_solver_settings:
        solver_settings = runtime_context.get_mujoco_solver_settings()
        solver_settings.update(dict(cfg.mujoco_solver_settings))
        runtime_context.set_mujoco_solver_settings(solver_settings)
    configure_robot_drives(robot, cfg)
    return runtime_context, robot, terrain


__all__ = [
    "configure_robot_drives",
    "find_robot_node",
    "find_terrain_node",
    "joint_gain_array",
    "prepare_go1_scene",
    "terrain_generator_config",
    "terrain_spawn_origins",
]
