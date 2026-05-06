from __future__ import annotations

from enum import Enum
from types import ModuleType
from typing import Any, ClassVar, Sequence

Vector3 = tuple[float, float, float]


class PhysicsBackendType(Enum):
    Null: ClassVar[PhysicsBackendType]
    MuJoCoCpu: ClassVar[PhysicsBackendType]
    PhysXCpu: ClassVar[PhysicsBackendType]
    PhysXGpu: ClassVar[PhysicsBackendType]
    NewtonGpu: ClassVar[PhysicsBackendType]
    RigidIpcCpu: ClassVar[PhysicsBackendType]


class JointType(Enum):
    Fixed: ClassVar[JointType]
    Revolute: ClassVar[JointType]
    Continuous: ClassVar[JointType]
    Prismatic: ClassVar[JointType]
    Floating: ClassVar[JointType]
    Planar: ClassVar[JointType]


class RobotMode(Enum):
    Assembly: ClassVar[RobotMode]
    Motion: ClassVar[RobotMode]


class LinkRole(Enum):
    Physical: ClassVar[LinkRole]
    VirtualRoot: ClassVar[LinkRole]


class JointControllerGains:
    position_stiffness: float
    velocity_damping: float
    integral_gain: float
    integral_limit: float

    def __init__(self) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    @staticmethod
    def from_dict(value: dict[str, Any]) -> JointControllerGains: ...


class RLEnvironmentRewardSettings:
    alive_reward: float
    fallen_reward: float
    target_forward_velocity: float
    forward_velocity_reward_scale: float
    action_rate_penalty_scale: float
    effort_penalty_scale: float
    minimum_base_height: float
    maximum_base_tilt_radians: float
    terminate_on_fall: bool

    def __init__(self) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    @staticmethod
    def from_dict(value: dict[str, Any]) -> RLEnvironmentRewardSettings: ...


class RLVectorSpec:
    version: str
    names: list[str]
    lower_bounds: list[float]
    upper_bounds: list[float]
    units: list[str]

    def __init__(self) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    @staticmethod
    def from_dict(value: dict[str, Any]) -> RLVectorSpec: ...


class PhysicsBackendInfo:
    type: PhysicsBackendType
    name: str
    available: bool
    cpu: bool
    gpu: bool
    robotics_focused: bool
    status: str

    def __init__(self) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    @staticmethod
    def from_dict(value: dict[str, Any]) -> PhysicsBackendInfo: ...


class AppContext:
    project_path: str
    scene_path: str
    root: Node | None
    backend_type: PhysicsBackendType
    has_scene: bool
    has_world: bool
    simulation_time: float
    frame_count: int
    gravity: Vector3

    def set_project_path(self, project_path: str) -> None: ...
    def load_scene(self, scene_path: str) -> Node: ...
    def clear_scene(self) -> None: ...
    def notify_scene_changed(self) -> None: ...
    def build_world(self, backend_type: PhysicsBackendType = PhysicsBackendType.Null) -> None: ...
    def rebuild_world(self, preserve_state: bool = True) -> None: ...
    def clear_world(self) -> None: ...
    def reset_simulation(self) -> None: ...
    def step_once(self) -> None: ...
    def step(self, ticks: int = 1) -> None: ...
    def get_last_error(self) -> str: ...


class Scene:
    root: Node


class Node:
    name: str
    type: str
    child_count: int
    children: list[Node]
    parent: Node | None

    def child(self, index: int) -> Node: ...
    def find(self, path: str) -> Node | None: ...
    def add_child(self, child: Node, force_readable_name: bool = False) -> None: ...
    def remove_child(self, child: Node, delete: bool = False) -> None: ...
    def reparent(self, parent: Node) -> None: ...
    def get(self, property: str) -> Any: ...
    def set(self, property: str, value: Any) -> None: ...
    def property_names(self) -> list[str]: ...


class Node3D(Node):
    position: Vector3
    rotation_degrees: Vector3
    scale: Vector3
    visible: bool


class Robot3D(Node3D):
    source_path: str
    mode: RobotMode


class Link3D(Node3D):
    has_inertial: bool
    mass: float
    center_of_mass: Vector3
    inertia_diagonal: Vector3
    role: LinkRole


class Joint3D(Node3D):
    joint_type: JointType
    parent_link: str
    child_link: str
    axis: Vector3
    lower_limit: float
    upper_limit: float
    effort_limit: float
    velocity_limit: float
    joint_position: float


class CollisionShape3D(Node3D):
    disabled: bool


class MeshInstance3D(Node3D):
    surface_color: tuple[float, float, float, float]


class RLControllerConfig:
    controlled_joints: list[str]
    default_action: list[float]
    joint_gains: dict[str, Any]

    def __init__(self) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    @staticmethod
    def from_dict(config: dict[str, Any]) -> RLControllerConfig: ...


class RLEnvironment:
    def __init__(self, scene_path: str = "", robot: str = "robot", backend: str = "null") -> None: ...
    def reset(self, seed: int = 0) -> tuple[list[float], dict[str, Any]]: ...
    def step(self, action: Sequence[float]) -> tuple[list[float], float, bool, bool, dict[str, Any]]: ...
    def reset_result(self, seed: int = 0) -> dict[str, Any]: ...
    def step_result(self, action: Sequence[float]) -> dict[str, Any]: ...
    def get_observation(self) -> list[float]: ...
    def get_action_size(self) -> int: ...
    def get_observation_size(self) -> int: ...
    def get_controlled_joint_names(self) -> list[str]: ...
    def get_contact_link_names(self) -> list[str]: ...
    def get_action_spec(self) -> dict[str, Any]: ...
    def get_observation_spec(self) -> dict[str, Any]: ...
    def set_reward_settings(self, settings: dict[str, Any]) -> None: ...
    def get_reward_settings(self) -> dict[str, Any]: ...
    def set_default_joint_gains(self, gains: dict[str, Any]) -> None: ...
    def get_default_joint_gains(self) -> dict[str, Any]: ...
    def get_controller_config(self) -> RLControllerConfig: ...
    def apply_controller_config(self, config: dict[str, Any]) -> None: ...
    def get_last_error(self) -> str: ...


def set_project_path(project_path: str) -> None: ...
def load_scene(scene_path: str) -> Scene: ...
def create_node(type_name: str, name: str = "") -> Node: ...
def create_box_collision(name: str, size: Vector3, position: Vector3 = (0.0, 0.0, 0.0)) -> CollisionShape3D: ...
def create_box_visual(name: str, size: Vector3, position: Vector3 = (0.0, 0.0, 0.0)) -> Node3D: ...
def save_scene(root: Node, path: str) -> None: ...
def load_resource(path: str, type_hint: str = "") -> dict[str, Any]: ...
def create_test_scene() -> Scene: ...
def backend_infos() -> list[dict[str, Any]]: ...

app: ModuleType
physics: ModuleType
rl: ModuleType
scene: ModuleType
sim: ModuleType
