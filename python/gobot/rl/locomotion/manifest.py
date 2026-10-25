"""Shared velocity policy contract creation and validation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from gobot.rl.policy import PolicyManifest, scene_bundle_digest
from gobot.rl.rsl_rl import RslRlVecEnvWrapper

def build_policy_manifest(
    env: RslRlVecEnvWrapper,
    cfg,
    train_cfg: dict,
    project_path: Path,
) -> PolicyManifest:
    task_metadata = env.task_runtime_metadata
    scene_digest = scene_bundle_digest(project_path, cfg.scene_path)
    return PolicyManifest(
        task_name=str(cfg.name),
        task_version=str(task_metadata.version),
        observation_spec=env.observation_spec.metadata(),
        action_spec=env.action_spec.metadata(),
        joint_names=tuple(env.joint_names),
        physics_dt=float(env.physics_dt),
        decimation=int(env.decimation),
        control={
            "mode": "position_offset",
            "held_joint_targets": dict(cfg.robot.held_joint_targets),
            "held_joint_kp": cfg.robot.held_joint_kp,
            "held_joint_kd": cfg.robot.held_joint_kd,
            "default_joint_position": np.asarray(env.default_joint_pos, dtype=np.float32),
            "action_scale": np.asarray(env.action_scale, dtype=np.float32),
            "action_clip": cfg.action_clip,
            "kp": cfg.kp,
            "kd": cfg.kd,
            "reset_base_height": float(cfg.base_clearance),
        },
        model={
            "type": "mlp",
            "activation": str(train_cfg.get("actor", {}).get("activation", "elu")),
            "hidden_dims": tuple(train_cfg.get("actor", {}).get("hidden_dims", ())),
        },
        scene_path=str(cfg.scene_path),
        scene_digest=scene_digest,
        extras={
            "robot_name": str(cfg.robot_name),
            "post_step_sensing": cfg.post_step_sensing,
            "base_link": str(cfg.base_link),
            "training_backend": str(task_metadata.backend),
            "critic_observation_spec": env.critic_observation_spec.metadata(),
            "solver_settings": dict(cfg.mujoco_solver_settings),
            "height_scan_max_distance": float(cfg.observations.terrain_scan_max_distance),
            "training_profile": str(cfg.training_profile),
            "run_environment_ratio": float(cfg.command.rel_run_envs),
            "run_velocity_x": tuple(float(value) for value in cfg.command.run_velocity_x),
            "run_velocity_curriculum": tuple(
                {
                    "step": int(stage.step),
                    "run_velocity_x": tuple(float(value) for value in stage.run_velocity_x),
                }
                for stage in cfg.run_command_curriculum
                if stage.run_velocity_x is not None
            ),
            "bound_gait_reward": float(cfg.rewards.bound_gait),
            "run_progress_reward": float(cfg.rewards.run_progress),
        },
    )


def validate_policy(manifest, env, cfg):
    manifest.validate_runtime(observation_spec=env.observation_spec, action_spec=env.action_spec,
        joint_names=env.joint_names, physics_dt=env.physics_dt, decimation=env.decimation,
        task_name=cfg.name, task_version=cfg.task_version,
        scene_digest=scene_bundle_digest(Path(cfg.project_path), cfg.scene_path))
    if bool(manifest.extras.get("post_step_sensing", False)) != cfg.post_step_sensing:
        raise ValueError("policy sensor timing does not match the task")
    if dict(manifest.extras.get("solver_settings", {})) != dict(cfg.mujoco_solver_settings):
        raise ValueError("policy solver settings do not match the task")
    if cfg.robot.held_joint_targets:
        for key in ("held_joint_kp", "held_joint_kd"):
            if not np.isclose(float(manifest.control.get(key, float("nan"))), float(getattr(cfg.robot, key))):
                raise ValueError(f"policy {key} does not match the robot profile")
    if dict(manifest.control.get("held_joint_targets", {})) != dict(cfg.robot.held_joint_targets):
        raise ValueError("policy held-joint targets do not match the robot profile")
    for key, values in (("kp", cfg.kp), ("kd", cfg.kd), ("default_joint_position", cfg.default_joint_pos), ("action_scale", env.action_scale)):
        actual = np.asarray(manifest.control.get(key, ()), dtype=np.float32)
        expected = np.asarray(values, dtype=np.float32)
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-6, atol=1e-7):
            raise ValueError(f"policy control {key!r} does not match the robot profile")
