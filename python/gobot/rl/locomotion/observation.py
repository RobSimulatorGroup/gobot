"""Observation schema and builders for velocity locomotion."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from gobot.rl.spec import ObservationSpec, SpecField


VELOCITY_OBS_SCHEMA_VERSION = "gobot_velocity_v1"


def velocity_actor_observation_schema(action_dim: int, height_scan_dim: int) -> ObservationSpec:
    return ObservationSpec(
        version=VELOCITY_OBS_SCHEMA_VERSION,
        fields=(
            SpecField("base_lin_vel_b", 3, "m/s"),
            SpecField("base_ang_vel_b", 3, "rad/s"),
            SpecField("projected_gravity", 3),
            SpecField("joint_pos_rel", int(action_dim), "rad"),
            SpecField("joint_vel", int(action_dim), "rad/s"),
            SpecField("last_action", int(action_dim)),
            SpecField("command", 3),
            SpecField("height_scan", int(height_scan_dim), "normalized"),
        ),
    )


def velocity_critic_observation_schema(
    action_dim: int,
    height_scan_dim: int,
    foot_count: int,
) -> ObservationSpec:
    actor = velocity_actor_observation_schema(action_dim, height_scan_dim)
    return ObservationSpec(
        version=actor.version,
        fields=(
            *actor.fields,
            SpecField("foot_height", int(foot_count)),
            SpecField("foot_air_time", int(foot_count)),
            SpecField("foot_contact", int(foot_count)),
            SpecField("foot_contact_forces_log", int(foot_count) * 3),
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


__all__ = [
    "VELOCITY_OBS_SCHEMA_VERSION",
    "build_velocity_actor_observation",
    "velocity_actor_observation_schema",
    "velocity_critic_observation_schema",
]
