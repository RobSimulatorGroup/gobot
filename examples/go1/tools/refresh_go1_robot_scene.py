"""Regenerate the Go1 robot scene from the Go1 MJCF asset."""

from __future__ import annotations

from pathlib import Path
import re

import gobot
from gobot.rl.tasks.go1 import (
    GO1_JOINT_NAMES,
    GO1_ARMATURE,
    GO1_EFFORT_LIMIT,
    GO1_KD,
    GO1_KP,
    GO1_VELOCITY_LIMIT,
)


FOOT_LINKS: dict[str, str] = {
    "FR": "FR_calf",
    "FL": "FL_calf",
    "RR": "RR_calf",
    "RL": "RL_calf",
}

FOOT_SAMPLE_OFFSETS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (0.04, 0.0, 0.0),
    (0.0, 0.04, 0.0),
    (-0.04, 0.0, 0.0),
    (0.0, -0.04, 0.0),
)
FOOT_COLLISION_PATTERN = re.compile(r"^[FR][LR]_foot_collision$")
COLLISION_PATTERN = re.compile(r"_collision\d*$")


def _required_child(root: gobot.Node, path: str) -> gobot.Node:
    node = root.find(path)
    if node is None:
        raise RuntimeError(f"expected Go1 scene node '{path}' while refreshing robot sensors")
    return node


def _find_by_name(root: gobot.Node, name: str) -> gobot.Node | None:
    if root.name == name:
        return root
    for child in root.children:
        found = _find_by_name(child, name)
        if found is not None:
            return found
    return None


def _required_name(root: gobot.Node, name: str) -> gobot.Node:
    node = _find_by_name(root, name)
    if node is None:
        raise RuntimeError(f"expected Go1 scene node named '{name}' while refreshing robot sensors")
    return node


def _remove_if_present(root: gobot.Node, path: str) -> None:
    node = root.find(path)
    if node is not None:
        node.remove(delete=True)


def _configure_sensor(sensor: gobot.Sensor3D) -> None:
    sensor.enabled = True
    sensor.sensor_period = 0.0
    sensor.noise_stddev = 0.0
    sensor.visualize_debug = False


def _add_trunk_sensors(root: gobot.Node) -> None:
    trunk = _required_name(root, "trunk")
    for name in ("gyro", "terrain_scan"):
        _remove_if_present(trunk, name)

    root_angmom = trunk.find("root_angmom")
    if root_angmom is None:
        root_angmom = gobot.create_node("AngularMomentumSensor3D", "root_angmom")
        trunk.add_child(root_angmom)
    _configure_sensor(root_angmom)

    imu = trunk.find("imu")
    if imu is None:
        imu = gobot.create_node("IMUSensor3D", "imu")
        trunk.add_child(imu)
    _configure_sensor(imu)

    terrain_scan = gobot.create_node("HeightScanner3D", "terrain_scan")
    _configure_sensor(terrain_scan)
    terrain_scan.pattern_mode = gobot.RayPatternMode.Grid
    terrain_scan.grid_size = (1.6, 1.0)
    terrain_scan.grid_resolution = 0.1
    terrain_scan.ray_alignment = gobot.RayAlignmentMode.Yaw
    terrain_scan.ray_direction = (0.0, 0.0, -1.0)
    terrain_scan.ray_direction_world_space = True
    terrain_scan.max_distance = 5.0
    terrain_scan.reduction_mode = gobot.RayReductionMode.None_
    trunk.add_child(terrain_scan)


def _add_foot_sensors(root: gobot.Node) -> None:
    for foot_name, link_name in FOOT_LINKS.items():
        link = _required_name(root, link_name)
        for name in (f"{foot_name}_foot_height_scan", f"{foot_name}_foot_contact"):
            _remove_if_present(link, name)

        height_scan = gobot.create_node("TerrainHeightSensor3D", f"{foot_name}_foot_height_scan")
        _configure_sensor(height_scan)
        height_scan.position = (0.0, 0.0, -0.213)
        height_scan.sample_offsets = list(FOOT_SAMPLE_OFFSETS)
        height_scan.pattern_mode = gobot.RayPatternMode.Custom
        height_scan.grid_size = (1.0, 1.0)
        height_scan.grid_resolution = 0.1
        height_scan.ray_alignment = gobot.RayAlignmentMode.Yaw
        height_scan.ray_direction = (0.0, 0.0, -1.0)
        height_scan.ray_direction_world_space = True
        height_scan.max_distance = 1.0
        height_scan.reduction_mode = gobot.RayReductionMode.Min
        link.add_child(height_scan)

        contact = gobot.create_node("ContactSensor3D", f"{foot_name}_foot_contact")
        _configure_sensor(contact)
        contact.position = (0.0, 0.0, -0.213)
        contact.radius = 0.03
        contact.min_threshold = 0.0
        contact.max_threshold = 0.0
        link.add_child(contact)


def _apply_go1_joint_dynamics(root: gobot.Node) -> None:
    for joint_index, joint_name in enumerate(GO1_JOINT_NAMES):
        joint = _required_name(root, joint_name)
        effort_limit = float(GO1_EFFORT_LIMIT[joint_index])
        joint.drive_mode = gobot.JointDriveMode.Position
        joint.drive_stiffness = float(GO1_KP[joint_index])
        joint.drive_damping = float(GO1_KD[joint_index])
        joint.armature = float(GO1_ARMATURE[joint_index])
        joint.effort_limit = effort_limit
        joint.velocity_limit = float(GO1_VELOCITY_LIMIT[joint_index])
        joint.force_lower_limit = -effort_limit
        joint.force_upper_limit = effort_limit
        joint.control_lower_limit = 0.0
        joint.control_upper_limit = 0.0
        joint.damping = 0.0
        joint.friction_loss = 0.0


def _apply_go1_collision_config(node: gobot.Node) -> None:
    if isinstance(node, gobot.CollisionShape3D) and COLLISION_PATTERN.search(node.name) is not None:
        is_foot = FOOT_COLLISION_PATTERN.fullmatch(node.name) is not None
        node.set_property("contype", 1)
        node.set_property("conaffinity", 1)
        node.set_property("condim", 6 if is_foot else 1)
        node.set_property("priority", 1 if is_foot else 0)
        node.set_property("solref", (0.01, 1.0))
        if is_foot:
            node.set_property("friction", (1.0, 5.0e-3, 5.0e-4))
    for child in node.children:
        _apply_go1_collision_config(child)


def main() -> None:
    project_path = Path(__file__).resolve().parents[1]
    gobot.set_project_path(str(project_path))
    gobot.import_mjcf_scene("res://assets/xml/go1.xml", "res://go1.jscn", name="go1")
    scene = gobot.load_scene("res://go1.jscn")
    root = scene.root
    _apply_go1_joint_dynamics(root)
    _apply_go1_collision_config(root)
    _add_trunk_sensors(root)
    _add_foot_sensors(root)
    gobot.save_scene(root, "res://go1.jscn")


if __name__ == "__main__":
    main()
