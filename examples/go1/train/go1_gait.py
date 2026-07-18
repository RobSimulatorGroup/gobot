"""Go1 gait-shaping helpers shared by CPU training and evaluation tests."""

from __future__ import annotations

from typing import Sequence

import numpy as np


GO1_GAIT_FOOT_ORDER: tuple[str, ...] = ("FR", "FL", "RR", "RL")


def gait_foot_indices(foot_names: Sequence[str]) -> tuple[int, int, int, int]:
    index_by_name = {str(name): index for index, name in enumerate(foot_names)}
    try:
        return tuple(index_by_name[name] for name in GO1_GAIT_FOOT_ORDER)
    except KeyError as error:
        raise ValueError(
            f"Go1 gait shaping requires feet {GO1_GAIT_FOOT_ORDER}, got {tuple(foot_names)}"
        ) from error


def gait_joint_indices(
    joint_names: Sequence[str],
) -> tuple[tuple[int, int, int], ...]:
    index_by_name = {str(name): index for index, name in enumerate(joint_names)}
    try:
        return tuple(
            tuple(index_by_name[f"{foot}_{joint}_joint"] for joint in ("hip", "thigh", "calf"))
            for foot in GO1_GAIT_FOOT_ORDER
        )
    except KeyError as error:
        raise ValueError(
            f"Go1 gait shaping cannot map joints from {tuple(joint_names)}"
        ) from error


def bound_gait_score_numpy(
    foot_contact: np.ndarray,
    foot_velocity: np.ndarray,
    foot_height: np.ndarray,
    action: np.ndarray,
    base_velocity_x: np.ndarray,
    command_x: np.ndarray,
    run_mask: np.ndarray,
    *,
    foot_indices: Sequence[int],
    joint_indices: Sequence[Sequence[int]],
    motion_std: float,
    action_std: float,
    height_sync_std: float,
    height_separation_std: float,
    trot_penalty: float,
) -> np.ndarray:
    indices = np.asarray(tuple(int(index) for index in foot_indices), dtype=np.int64)
    if indices.shape != (4,):
        raise ValueError("bound gait shaping requires four foot indices")
    contact = np.asarray(foot_contact, dtype=np.float32)[:, indices] > 0.0
    velocity = np.asarray(foot_velocity, dtype=np.float32)[:, indices]
    if velocity.ndim != 3 or velocity.shape[1:] != (4, 3):
        raise ValueError(f"foot velocity must have shape (num_envs, 4, 3), got {velocity.shape}")

    sagittal = velocity[:, :, (0, 2)]
    front_delta = sagittal[:, 0] - sagittal[:, 1]
    rear_delta = sagittal[:, 2] - sagittal[:, 3]
    motion_error = 0.5 * (
        np.sum(np.square(front_delta), axis=1)
        + np.sum(np.square(rear_delta), axis=1)
    )
    motion_sync = np.exp(-motion_error / max(float(motion_std) ** 2, 1.0e-6))

    joint_index_array = np.asarray(joint_indices, dtype=np.int64)
    if joint_index_array.shape != (4, 3):
        raise ValueError("bound gait shaping requires four hip/thigh/calf joint groups")
    paired_action = np.asarray(action, dtype=np.float32)[:, joint_index_array]
    front_action_delta = paired_action[:, 0, 1:] - paired_action[:, 1, 1:]
    rear_action_delta = paired_action[:, 2, 1:] - paired_action[:, 3, 1:]
    action_error = 0.5 * (
        np.mean(np.square(front_action_delta), axis=1)
        + np.mean(np.square(rear_action_delta), axis=1)
    )
    action_sync = np.exp(-action_error / max(float(action_std) ** 2, 1.0e-6))

    height = np.asarray(foot_height, dtype=np.float32)[:, indices]
    if height.ndim != 2 or height.shape[1] != 4:
        raise ValueError(f"foot height must have shape (num_envs, 4), got {height.shape}")
    height_pair_error = 0.5 * (
        np.square(height[:, 0] - height[:, 1])
        + np.square(height[:, 2] - height[:, 3])
    )
    height_sync = np.exp(
        -height_pair_error / max(float(height_sync_std) ** 2, 1.0e-6)
    )
    front_height = 0.5 * (height[:, 0] + height[:, 1])
    rear_height = 0.5 * (height[:, 2] + height[:, 3])
    height_separation = 1.0 - np.exp(
        -np.square(front_height - rear_height)
        / max(float(height_separation_std) ** 2, 1.0e-6)
    )
    height_gait = height_sync * height_separation

    contact_float = contact.astype(np.float32)
    contact_sync = 1.0 - 0.5 * (
        np.logical_xor(contact[:, 0], contact[:, 1]).astype(np.float32)
        + np.logical_xor(contact[:, 2], contact[:, 3]).astype(np.float32)
    )
    front_contact = 0.5 * (contact_float[:, 0] + contact_float[:, 1])
    rear_contact = 0.5 * (contact_float[:, 2] + contact_float[:, 3])
    support_separation = np.abs(front_contact - rear_contact) * contact_sync
    trot_support = (
        (contact[:, 0] & ~contact[:, 1] & ~contact[:, 2] & contact[:, 3])
        | (~contact[:, 0] & contact[:, 1] & contact[:, 2] & ~contact[:, 3])
    ).astype(np.float32)

    command = np.maximum(np.asarray(command_x, dtype=np.float32), 0.1)
    speed_progress = np.clip(
        np.asarray(base_velocity_x, dtype=np.float32) / command,
        0.0,
        1.0,
    )
    score = speed_progress * (
        0.25 * action_sync
        + 0.15 * motion_sync
        + 0.15 * contact_sync
        + 0.15 * support_separation
        + 0.30 * height_gait
        - float(trot_penalty) * trot_support
    )
    return (score * np.asarray(run_mask, dtype=bool)).astype(np.float32, copy=False)


__all__ = [
    "GO1_GAIT_FOOT_ORDER",
    "bound_gait_score_numpy",
    "gait_foot_indices",
    "gait_joint_indices",
]
