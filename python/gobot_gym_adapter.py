"""Compatibility shim for old imports.

Prefer importing from ``gobot.gym_adapter`` in new code.
"""

from gobot.gym_adapter import GobotBox, GobotGymEnv, space_from_spec

__all__ = ["GobotBox", "GobotGymEnv", "space_from_spec"]
