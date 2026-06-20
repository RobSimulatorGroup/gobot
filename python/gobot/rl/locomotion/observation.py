"""Observation schema and builders for velocity locomotion."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from gobot.rl.spec import ObservationSpec, SpecField


VELOCITY_OBS_SCHEMA_VERSION = "gobot_velocity_v1"
ObservationField = SpecField
ObservationSchema = ObservationSpec


def velocity_actor_observation_schema(action_dim: int, height_scan_dim: int) -> ObservationSchema:
    return ObservationSchema(
        version=VELOCITY_OBS_SCHEMA_VERSION,
        fields=(
            ObservationField("base_lin_vel_b", 3, "m/s"),
            ObservationField("base_ang_vel_b", 3, "rad/s"),
            ObservationField("projected_gravity", 3),
            ObservationField("joint_pos_rel", int(action_dim), "rad"),
            ObservationField("joint_vel", int(action_dim), "rad/s"),
            ObservationField("last_action", int(action_dim)),
            ObservationField("command", 3),
            ObservationField("height_scan", int(height_scan_dim), "normalized"),
        ),
    )


def velocity_critic_observation_schema(action_dim: int, height_scan_dim: int, foot_count: int) -> ObservationSchema:
    actor = velocity_actor_observation_schema(action_dim, height_scan_dim)
    return ObservationSchema(
        version=actor.version,
        fields=(
            *actor.fields,
            ObservationField("foot_height", int(foot_count)),
            ObservationField("foot_air_time", int(foot_count)),
            ObservationField("foot_contact", int(foot_count)),
            ObservationField("foot_contact_forces_log", int(foot_count) * 3),
        ),
    )


def build_velocity_actor_observation(
    *,
    base_lin_vel_b: Iterable[float],
    base_ang_vel_b: Iterable[float],
    projected_gravity: Iterable[float],
    joint_pos_rel: Iterable[float],
    joint_vel: Iterable[float],
    last_action: Iterable[float],
    command: Iterable[float],
    height_scan: Iterable[float],
) -> np.ndarray:
    parts = (
        base_lin_vel_b,
        base_ang_vel_b,
        projected_gravity,
        joint_pos_rel,
        joint_vel,
        last_action,
        command,
        height_scan,
    )
    return np.concatenate([np.asarray(part, dtype=np.float32).reshape(-1) for part in parts])


def log_contact_forces(foot_forces: Iterable[float]) -> np.ndarray:
    forces = np.asarray(foot_forces, dtype=np.float32).reshape(-1)
    return np.sign(forces) * np.log1p(np.abs(forces))


def build_velocity_critic_observation(
    *,
    actor_obs: Iterable[float],
    foot_height: Iterable[float],
    foot_air_time: Iterable[float],
    foot_contact: Iterable[float],
    foot_contact_forces: Iterable[float],
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(actor_obs, dtype=np.float32).reshape(-1),
            np.asarray(foot_height, dtype=np.float32).reshape(-1),
            np.asarray(foot_air_time, dtype=np.float32).reshape(-1),
            np.asarray(foot_contact, dtype=np.float32).reshape(-1),
            log_contact_forces(foot_contact_forces),
        ]
    ).astype(np.float32)


__all__ = [
    "ObservationField",
    "ObservationSchema",
    "VELOCITY_OBS_SCHEMA_VERSION",
    "build_velocity_actor_observation",
    "build_velocity_critic_observation",
    "log_contact_forces",
    "velocity_actor_observation_schema",
    "velocity_critic_observation_schema",
]
