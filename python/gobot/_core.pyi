from __future__ import annotations

from enum import Enum
from types import ModuleType
from typing import Any, Callable, ClassVar, Sequence

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
    scene_epoch: int
    scene_dirty: bool
    can_undo: bool
    can_redo: bool
    undo_name: str
    redo_name: str
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
    def mark_scene_clean(self) -> None: ...
    def undo(self) -> bool: ...
    def redo(self) -> bool: ...
    def begin_transaction(self, name: str = "Scene Transaction") -> None: ...
    def commit_transaction(self) -> None: ...
    def cancel_transaction(self) -> None: ...
    def transaction(self, name: str = "Scene Transaction") -> SceneTransaction: ...
    def build_world(self, backend_type: PhysicsBackendType = PhysicsBackendType.Null) -> None: ...
    def rebuild_world(self, preserve_state: bool = True) -> None: ...
    def clear_world(self) -> None: ...
    def reset_simulation(self) -> None: ...
    def step_once(self) -> None: ...
    def step(self, ticks: int = 1) -> None: ...
    def set_robot_action(self, robot: str, action: Sequence[float]) -> None: ...
    def set_robot_named_action(self, robot: str, joint_names: Sequence[str], action: Sequence[float]) -> None: ...
    def set_default_joint_gains(self, gains: dict[str, Any]) -> None: ...
    def get_default_joint_gains(self) -> dict[str, Any]: ...
    def set_joint_position_target(self, robot: str, joint: str, target_position: float) -> None: ...
    def set_joint_velocity_target(self, robot: str, joint: str, target_velocity: float) -> None: ...
    def set_joint_effort_target(self, robot: str, joint: str, target_effort: float) -> None: ...
    def set_joint_passive(self, robot: str, joint: str) -> None: ...
    def get_last_error(self) -> str: ...


class Scene:
    root: Node
    scene_epoch: int


class SceneTransaction:
    def __enter__(self) -> SceneTransaction: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool: ...


class Node:
    id: int
    name: str
    type: str
    type_name: str
    path: str
    valid: bool
    child_count: int
    children: list[Node]
    parent: Node | None

    def child(self, index: int) -> Node: ...
    def find(self, path: str) -> Node | None: ...
    def add_child(self, child: Node, force_readable_name: bool = False) -> Node: ...
    def remove_child(self, child: Node, delete: bool = False) -> Node | None: ...
    def remove(self, delete: bool = True) -> Node | None: ...
    def reparent(self, parent: Node) -> None: ...
    def get(self, property: str) -> Any: ...
    def get_property(self, property: str) -> Any: ...
    def set(self, property: str, value: Any) -> None: ...
    def set_property(self, property: str, value: Any) -> None: ...
    def property_names(self) -> list[str]: ...
    def to_dict(self) -> dict[str, Any]: ...


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
    damping: float
    joint_position: float


class CollisionShape3D(Node3D):
    disabled: bool


class MeshInstance3D(Node3D):
    surface_color: tuple[float, float, float, float]


def set_project_path(project_path: str) -> None: ...
def load_scene(scene_path: str) -> Scene: ...
def create_node(type_name: str, name: str = "") -> Node: ...
def transaction(name: str = "Scene Transaction") -> SceneTransaction: ...
def undo() -> bool: ...
def redo() -> bool: ...
def create_box_collision(name: str, size: Vector3, position: Vector3 = (0.0, 0.0, 0.0)) -> CollisionShape3D: ...
def create_box_visual(name: str, size: Vector3, position: Vector3 = (0.0, 0.0, 0.0)) -> Node3D: ...
def save_scene(root: Node, path: str) -> None: ...
def load_resource(path: str, type_hint: str = "") -> dict[str, Any]: ...
def create_test_scene() -> Scene: ...
def backend_infos() -> list[dict[str, Any]]: ...
def set_editor_tick_callback(callback: Callable[[float], None] | None) -> None: ...
def clear_editor_tick_callback() -> None: ...
def set_editor_physics_callback(callback: Callable[[float], None] | None) -> None: ...
def clear_editor_physics_callback() -> None: ...

app: ModuleType
physics: ModuleType
rl: ModuleType
scene: ModuleType
sim: ModuleType
