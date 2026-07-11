from typing import Any, Mapping

GO1_TASK_NAME: str
GO1_TASK_VERSION: str
GO1_PHYSICS_DT: float
GO1_DECIMATION: int
GO1_JOINT_NAMES: tuple[str, ...]
GO1_DEFAULT_JOINT_POS: tuple[float, ...]
GO1_DEFAULT_BASE_POSITION: tuple[float, float, float]
GO1_FOOT_NAMES: tuple[str, ...]
GO1_FOOT_LINK_NAMES: tuple[str, ...]
GO1_KP: tuple[float, ...]
GO1_KD: tuple[float, ...]
GO1_ARMATURE: tuple[float, ...]
GO1_EFFORT_LIMIT: tuple[float, ...]
GO1_VELOCITY_LIMIT: tuple[float, ...]
GO1_MUJOCO_SOLVER_SETTINGS: Mapping[str, Any]

__all__: list[str]
