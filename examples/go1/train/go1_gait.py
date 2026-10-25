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


from gobot.rl.locomotion.gait import bound_gait_score_numpy
