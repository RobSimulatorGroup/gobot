from ._core import (
    AngularMomentumSensor3D,
    AppContext,
    CollisionShape3D,
    ContactSensor3D,
    HeightScanner3D,
    IMUSensor3D,
    Joint3D,
    JointControllerGains,
    JointDriveMode,
    JointType,
    Link3D,
    LinkRole,
    MeshInstance3D,
    Node,
    Node3D,
    NodeScript,
    PhysicsBackendInfo,
    PhysicsBackendType,
    Robot3D,
    RobotMode,
    Scene,
    Sensor3D,
    TerrainColorMode,
    Terrain3D,
    backend_infos,
    create_box_collision,
    create_box_visual,
    create_node,
    create_test_scene,
    import_mjcf_scene,
    load_resource,
    load_scene,
    save_scene,
    set_project_path,
    version,
    version_info,
)
from .scene_helpers import create_cartpole_scene, save_cartpole_scene

from . import app, physics, rl, scene, sim, terrain

__version__: str

def version() -> str: ...
def version_info() -> dict[str, int | str]: ...

__all__: list[str]
