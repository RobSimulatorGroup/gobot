from __future__ import annotations

from enum import Enum
from types import ModuleType
from typing import Any, ClassVar, Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
VectorLike = Sequence[float] | FloatArray
Vector2 = FloatArray
Vector3 = FloatArray
Vector4 = FloatArray
Quaternion = FloatArray


class PhysicsBackendType(Enum):
    Null: ClassVar[PhysicsBackendType]
    MuJoCoCpu: ClassVar[PhysicsBackendType]


class PhysicsSolverType(Enum):
    ProjectedGaussSeidel: ClassVar[PhysicsSolverType]
    ConjugateGradient: ClassVar[PhysicsSolverType]
    Newton: ClassVar[PhysicsSolverType]


class PhysicsIntegratorType(Enum):
    Euler: ClassVar[PhysicsIntegratorType]
    RungeKutta4: ClassVar[PhysicsIntegratorType]
    Implicit: ClassVar[PhysicsIntegratorType]
    ImplicitFast: ClassVar[PhysicsIntegratorType]


class PhysicsFrictionConeType(Enum):
    Pyramidal: ClassVar[PhysicsFrictionConeType]
    Elliptic: ClassVar[PhysicsFrictionConeType]


class PhysicsJacobianType(Enum):
    Dense: ClassVar[PhysicsJacobianType]
    Sparse: ClassVar[PhysicsJacobianType]
    Auto: ClassVar[PhysicsJacobianType]


class JointType(Enum):
    Fixed: ClassVar[JointType]
    Revolute: ClassVar[JointType]
    Continuous: ClassVar[JointType]
    Prismatic: ClassVar[JointType]
    Floating: ClassVar[JointType]
    Planar: ClassVar[JointType]


class JointDriveMode(Enum):
    Passive: ClassVar[JointDriveMode]
    Motor: ClassVar[JointDriveMode]
    Position: ClassVar[JointDriveMode]
    Velocity: ClassVar[JointDriveMode]


class RobotMode(Enum):
    Assembly: ClassVar[RobotMode]
    Motion: ClassVar[RobotMode]


class LinkRole(Enum):
    Physical: ClassVar[LinkRole]
    VirtualRoot: ClassVar[LinkRole]


class TerrainColorMode(Enum):
    SurfaceColor: ClassVar[TerrainColorMode]
    HeightRamp: ClassVar[TerrainColorMode]
    Palette: ClassVar[TerrainColorMode]


class TerrainSubTerrainType(Enum):
    Flat: ClassVar[TerrainSubTerrainType]
    PyramidStairs: ClassVar[TerrainSubTerrainType]
    PyramidSlope: ClassVar[TerrainSubTerrainType]
    RandomRough: ClassVar[TerrainSubTerrainType]
    Wave: ClassVar[TerrainSubTerrainType]


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
    fixed_time_step: float
    max_sub_steps: int
    batch_env_count: int
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
    def compile_scene_artifact(self, backend_type: PhysicsBackendType = PhysicsBackendType.MuJoCoCpu) -> dict[str, Any]: ...
    def compiled_scene_artifact(self) -> dict[str, Any]: ...
    def rebuild_world(self, preserve_state: bool = True) -> None: ...
    def clear_world(self) -> None: ...
    def reset_simulation(self) -> None: ...
    def step_once(self) -> None: ...
    def step(self, ticks: int = 1) -> None: ...
    def configure_batch_world(self, num_envs: int) -> None: ...
    def create_locomotion_batch_view(
        self,
        robot: str,
        base_link: str,
        joint_names: Sequence[str],
        foot_link_names: Sequence[str],
        foot_height_sensor_names: Sequence[str],
        foot_contact_sensor_names: Sequence[str],
        height_scan_sensor: str = "",
        thigh_shape_patterns: Sequence[str] = (),
        shank_shape_patterns: Sequence[str] = (),
        trunk_head_shape_patterns: Sequence[str] = (),
        terminate_on_thigh_contact: bool = True,
        ground_force_threshold: float = 50.0,
        self_collision_force_threshold: float = 20.0,
        link_names: Sequence[str] = (),
    ) -> Any: ...
    def reset_batch_env(self, env_id: int) -> None: ...
    def step_batch_env(self, env_id: int, ticks: int = 1) -> None: ...
    def step_batch(self, ticks: int = 1, workers: int = 0) -> None: ...
    def resolved_batch_workers(self, workers: int = 0) -> int: ...
    def set_batch_joint_position_target(
        self, env_id: int, robot: str, joint: str, target_position: float
    ) -> None: ...
    def set_batch_joint_position_targets(
        self, robot: str, joint_names: Sequence[str], target_positions: Any
    ) -> None: ...
    def reset_batch_joint_state(
        self, env_id: int, robot: str, joint: str, position: float, velocity: float = 0.0
    ) -> None: ...
    def reset_batch_link_state(
        self,
        env_id: int,
        robot: str,
        link: str,
        position: VectorLike,
        orientation: VectorLike = (1.0, 0.0, 0.0, 0.0),
        linear_velocity: VectorLike = (0.0, 0.0, 0.0),
        angular_velocity: VectorLike = (0.0, 0.0, 0.0),
    ) -> None: ...
    def set_default_joint_gains(self, gains: dict[str, Any]) -> None: ...
    def get_default_joint_gains(self) -> dict[str, Any]: ...
    def set_mujoco_solver_settings(self, settings: dict[str, Any]) -> None: ...
    def get_mujoco_solver_settings(self) -> dict[str, Any]: ...
    def get_batch_runtime_state(self, env_id: int) -> dict[str, Any]: ...
    def get_batch_robot_state(
        self,
        robot: str,
        base_link: str,
        joint_names: Sequence[str],
        link_names: Sequence[str] = (),
        sensor_names: Sequence[str] = (),
    ) -> dict[str, Any]: ...
    def get_last_error(self) -> str: ...


class Scene:
    root: Node
    scene_epoch: int


class SceneTransaction:
    def __enter__(self) -> SceneTransaction: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool: ...


class NodeScript:
    node: Node | None
    root: Node | None
    context: AppContext | None

    def __init__(self) -> None: ...
    def _attach(self, node: Node, root: Node | None, context: AppContext) -> None: ...
    def get_node(self, path: str) -> Node | None: ...
    def get_root(self) -> Node | None: ...


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


class VelocityCommandDebug3D(Node3D):
    enabled: bool
    show_command_velocity: bool
    show_measured_velocity: bool
    show_yaw_rate: bool
    arrow_scale: float
    z_offset: float


class Robot3D(Node3D):
    source_path: str
    mode: RobotMode


class Link3D(Node3D):
    has_inertial: bool
    mass: float
    center_of_mass: Vector3
    inertia_diagonal: Vector3
    role: LinkRole

    def reset_runtime_state(
        self,
        position: VectorLike,
        orientation: VectorLike = (1.0, 0.0, 0.0, 0.0),
        linear_velocity: VectorLike = (0.0, 0.0, 0.0),
        angular_velocity: VectorLike = (0.0, 0.0, 0.0),
    ) -> None: ...
    def get_runtime_state(self) -> dict[str, Any]: ...


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
    initial_position: float
    drive_mode: JointDriveMode
    drive_stiffness: float
    drive_damping: float
    control_lower_limit: float
    control_upper_limit: float
    force_lower_limit: float
    force_upper_limit: float
    gear: list[float]

    def set_position_target(self, target: float) -> None: ...
    def set_velocity_target(self, target: float) -> None: ...
    def set_effort_target(self, target: float) -> None: ...
    def set_passive(self) -> None: ...
    def reset_runtime_state(self, position: float, velocity: float = 0.0) -> None: ...
    def get_runtime_state(self) -> dict[str, Any]: ...


class CollisionShape3D(Node3D):
    disabled: bool


class MeshInstance3D(Node3D):
    surface_color: tuple[float, float, float, float]


class Terrain3D(Node3D):
    box_count: int
    heightfield_count: int
    mesh_patch_count: int
    generator_config: dict[str, Any]
    generation_error: str
    spawn_origins: list[Vector3]
    surface_color: tuple[float, float, float, float]
    color_mode: TerrainColorMode
    height_low_color: tuple[float, float, float, float]
    height_high_color: tuple[float, float, float, float]
    height_range_min: float
    height_range_max: float
    friction: Vector3
    solref: Vector2
    solimp: list[float]

    def clear_terrain(self) -> None: ...
    def regenerate_terrain(self) -> None: ...
    def add_box(
        self,
        center: Vector3,
        size: Vector3,
        rotation_degrees: Vector3 = (0.0, 0.0, 0.0),
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None: ...
    def add_heightfield(
        self,
        center: Vector3,
        size: VectorLike,
        rows: int,
        cols: int,
        heights: Sequence[float],
        base_thickness: float = 0.1,
        normalized_elevation: Sequence[float] = (),
        z_offset: float = 0.0,
    ) -> None: ...
    def add_mesh_patch(
        self,
        center: Vector3,
        vertices: Sequence[Vector3],
        indices: Sequence[int],
        rotation_degrees: Vector3 = (0.0, 0.0, 0.0),
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None: ...
    def get_heightfield_heights(self, index: int) -> list[float]: ...


class Sensor3D(Node3D):
    enabled: bool
    sensor_period: float
    noise_stddev: float
    visualize_debug: bool
    debug_marker_radius: float

    def get_runtime_state(self) -> dict[str, Any]: ...


class IMUSensor3D(Sensor3D):
    pass


class AngularMomentumSensor3D(Sensor3D):
    pass


class ContactSensor3D(Sensor3D):
    radius: float
    min_threshold: float
    max_threshold: float


class RayReductionMode(Enum):
    None_: ClassVar[RayReductionMode]
    Min: ClassVar[RayReductionMode]
    Max: ClassVar[RayReductionMode]
    Mean: ClassVar[RayReductionMode]


class RayPatternMode(Enum):
    Custom: ClassVar[RayPatternMode]
    Grid: ClassVar[RayPatternMode]


class RayAlignmentMode(Enum):
    World: ClassVar[RayAlignmentMode]
    Base: ClassVar[RayAlignmentMode]
    Yaw: ClassVar[RayAlignmentMode]


class RayCastSensor3D(Sensor3D):
    sample_offsets: list[Vector3]
    ray_direction: Vector3
    ray_direction_world_space: bool
    max_distance: float
    pattern_mode: RayPatternMode
    grid_size: Vector2
    grid_resolution: float
    ray_alignment: RayAlignmentMode


class TerrainHeightSensor3D(RayCastSensor3D):
    reduction_mode: RayReductionMode


class HeightScanner3D(TerrainHeightSensor3D):
    pass


def set_project_path(project_path: str) -> None: ...
def load_scene(scene_path: str) -> Scene: ...
def create_node(type_name: str, name: str = "") -> Node: ...
def transaction(name: str = "Scene Transaction") -> SceneTransaction: ...
def undo() -> bool: ...
def redo() -> bool: ...
def create_box_collision(name: str, size: Vector3, position: Vector3 = (0.0, 0.0, 0.0)) -> CollisionShape3D: ...
def create_box_visual(name: str, size: Vector3, position: Vector3 = (0.0, 0.0, 0.0)) -> Node3D: ...
def save_scene(root: Node, path: str) -> None: ...
def import_mjcf_scene(xml_path: str, scene_path: str, name: str | None = None, script: str | None = None) -> None: ...
def load_resource(path: str, type_hint: str = "") -> dict[str, Any]: ...
def _node_from_id(id: int, context: AppContext | None = None) -> Node | None: ...
def create_test_scene() -> Scene: ...
def backend_infos() -> list[dict[str, Any]]: ...

app: ModuleType
physics: ModuleType
rl: ModuleType
scene: ModuleType
sim: ModuleType
